"""Isolated inference cost and accuracy-vs-sample-count study for the four filtering estimators.

For every dataset / method / training seed in a build_results_paper source map, the selected checkpoint is
loaded and evaluated on the test windows with n in --counts:

  NormFlowSDE  n independent posterior paths from the amortized filter (cost scales with n)
  NF-DPF       bootstrap filtering with n particles (systematic resampling), n per-time draws
  NSF / FlowSDE learned-dynamics bootstrap PF with n particles (selected jitter), n per-time draws

Latency (seed 0 only): one method at a time on one device; inputs are placed on the device first, the first
call (compilation) is timed separately, then --repeats batch-size-one full-window calls over different test
windows are timed with block_until_ready (device time, no host transfer). Accuracy: all test windows, keys
fold_in(PRNGKey(99001), window), scored with metrics_from_samples exactly as the benchmark.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .cache import load_cache
from .evaluation_metrics import metrics_from_samples

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("normflowsde", "nf_dpf", "nsf", "flowsde")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", required=True, type=Path, help="build_results_paper source map.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--counts", nargs="+", type=int, default=[16, 32, 64, 128, 256, 512, 1024])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--chunk", type=int, default=8, help="Windows per batched accuracy call.")
    p.add_argument("--windows", type=int, help="Limit test windows (smoke only).")
    return p.parse_args(argv)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_sampler(method, run, data):
    """Return sample(x0, y, u, key, n) -> (n, T, d) normalized draws, and a description."""
    meta = json.loads((run / "run_metadata.json").read_text())
    if method == "normflowsde":
        from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
        from .train_normflowsde_filter import filtering_inputs
        model = eqx.tree_deserialise_leaves(run / "model.eqx", RealNVPFlowSDE(**meta["model_config"], key=jr.PRNGKey(0)))
        solver = meta.get("solver") or (meta.get("recipe") or {}).get("solver", "euler")

        def sample(x0, y, u, key, n):
            return model.sample_posterior(filtering_inputs(y, u), x0, key, n_particles=n, solver_name=solver)
        return sample, dict(solver=solver)
    if method == "nf_dpf":
        from bayesinf.applications.state_estimation.normalizing_flow_dpf import NFDPFConfig, NormalizingFlowDPF
        model = eqx.tree_deserialise_leaves(run / "model.eqx",
                                            NormalizingFlowDPF(NFDPFConfig(**meta["model_config"]), key=jr.PRNGKey(0)))

        def sample(x0, y, u, key, n):
            filter_key, draw_key = jr.split(key)
            result = model.filter(y, x0, u, key=filter_key, num_particles=n, initial_std=0., resampling="systematic")
            keys = jr.split(draw_key, y.shape[0])
            return jax.vmap(lambda k, p, w: p[jr.categorical(k, jnp.log(w), shape=(n,))])(
                keys, result.particles, result.weights).swapaxes(0, 1)
        return sample, dict(resampling="systematic")
    from .train_state_filter import build
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").is_file() else {}
    jitter = float(summary.get("jitter", meta.get("jitter", 0.)) or 0.)
    arguments = meta.get("arguments") or {}
    if "hidden_size" not in arguments:  # jitter-tuning runs record the training run they re-evaluated
        source = resolve(meta.get("run") or arguments.get("run"))
        arguments = json.loads((source / "run_metadata.json").read_text())["arguments"]
    model = eqx.tree_deserialise_leaves(run / "model.eqx", build(Namespace(**arguments), data))

    def sample(x0, y, u, key, n):
        return model.sample(y, x0, u, key=key, num_particles=n, num_samples=n, jitter=jitter)
    return sample, dict(jitter=jitter)


def latency(sample, split, n, repeats):
    fn = jax.jit(lambda x0, y, u, k: sample(x0, y, u, k, n))
    windows = len(split["states"])
    inputs = [tuple(jax.device_put(jnp.asarray(split[name][i % windows])) for name in ("states", "observations", "controls"))
              for i in range(repeats + 1)]
    x, y, u = inputs[0]
    started = time.perf_counter()
    jax.block_until_ready(fn(x[0], y, u, jr.PRNGKey(0)))
    compile_seconds = time.perf_counter() - started
    times = []
    for r, (x, y, u) in enumerate(inputs[1:]):
        started = time.perf_counter()
        jax.block_until_ready(fn(x[0], y, u, jr.PRNGKey(r + 1)))
        times.append(1e3 * (time.perf_counter() - started))
    horizon = split["observations"].shape[1]
    return dict(compile_seconds=compile_seconds, ms_per_window_median=float(np.median(times)),
                ms_per_window_p90=float(np.percentile(times, 90)), ms_per_step_median=float(np.median(times)) / horizon)


def accuracy(sample, split, data, n, chunk, windows):
    fn = jax.jit(jax.vmap(lambda x, y, u, k: sample(x[0], y, u, k, n)))
    count = min(windows or data.protocol["eval_windows"], len(split["states"]))
    out = []
    for start in range(0, count, chunk):
        index = np.arange(start, min(start + chunk, count))
        keys = jnp.stack([jr.fold_in(jr.PRNGKey(99001), int(i)) for i in index])
        out.append(np.asarray(fn(*(jnp.asarray(split[k][index]) for k in ("states", "observations", "controls")), keys)))
    samples, truth = np.concatenate(out), np.asarray(split["states"][:count, 1:])
    if not np.isfinite(samples).all():
        return dict(nonfinite=True)
    meta, protocol = data.metadata, data.protocol
    metrics = metrics_from_samples(samples, truth, observed_indices=protocol["observed_indices"],
                                   metric_space=protocol["metric_space"], state_mean=meta["state_mean"],
                                   state_std=meta["state_std"])["metrics"]
    return {k: metrics[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width", "energy_score")}


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rows.jsonl"
    done = set()
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            row = json.loads(line)
            done.add((row["dataset"], row["method"], row["seed"], row["n"]))
    sources = json.loads(args.sources.read_text())
    datasets = args.datasets or [d for d in sources if not d.startswith("_")]
    device = str(jax.devices()[0])
    for dataset in datasets:
        spec = sources[dataset]
        data = load_cache(resolve(spec["data_cache"]))
        for method in args.methods:
            source = spec["methods"][method]
            for seed in args.seeds:
                run = resolve(source["seeds"][str(seed)])
                sample, info = load_sampler(method, run, data)
                for n in args.counts:
                    if (dataset, method, seed, n) in done:
                        continue
                    row = dict(dataset=dataset, method=method, seed=seed, n=n, run=str(run), device=device, **info)
                    if seed == args.seeds[0]:
                        row["latency"] = latency(sample, data.test, n, args.repeats)
                    row["accuracy"] = accuracy(sample, data.test, data, n, args.chunk, args.windows)
                    with rows_path.open("a") as handle:
                        handle.write(json.dumps(row) + "\n")
                    print(json.dumps(dict(dataset=dataset, method=method, seed=seed, n=n,
                                          ms=row.get("latency", {}).get("ms_per_window_median"),
                                          rmse=row["accuracy"].get("rmse"))), flush=True)
    (args.output / "run.json").write_text(json.dumps(dict(finished_at=datetime.now(timezone.utc).isoformat(),
        sources=str(args.sources), device=device, arguments={k: str(v) if isinstance(v, Path) else v
                                                              for k, v in vars(args).items()}), indent=2) + "\n")


if __name__ == "__main__":
    main()
