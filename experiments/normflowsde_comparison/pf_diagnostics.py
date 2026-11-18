"""Evaluation-only diagnostics for the particle-filter baselines (NF-DPF, NSF-PF, FlowSDE-PF).

No training. For each dataset / method / seed checkpoint of a cell map (JSON: {dataset: {method: {seed: [runs]}}},
the first run being the selected one and further runs alternative FlowSDE diffusion candidates):

  particles    accuracy and effective sample size at several particle counts (selected model, selected jitter)
  measurement  pendulum / Lorenz-96 only: learned measurement density vs. the known data-generating density
               (per-coordinate NLL at true states on the test split, and the filter rerun with the true density)
  jitter       NSF / FlowSDE only: process-noise jitter grid x diffusion candidates at the evaluation particle
               count, so that the setting can be re-selected on validation CRPS

Every configuration is scored on validation (selection) and test (report) with metrics_from_samples, with keys
fold_in(PRNGKey(seed_base), window) as in the benchmark. Rows are appended to <output>/rows.jsonl (resumable).
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import logsumexp

from .cache import load_cache
from .evaluation_metrics import metrics_from_samples

ROOT = Path(__file__).resolve().parents[2]
JITTERS = (0., .005, .01, .02, .03, .05, .075, .1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--particle-counts", nargs="+", type=int, default=[128, 200, 500, 1000, 2000])
    p.add_argument("--validation-windows", type=int, default=100)
    p.add_argument("--test-windows", type=int, help="Limit test windows (smoke only; default: protocol eval_windows).")
    p.add_argument("--chunk", type=int, default=4)
    p.add_argument("--parts", nargs="+", default=["particles", "measurement", "jitter"])
    p.add_argument("--datasets", nargs="+")
    p.add_argument("--data-root", type=Path, default=Path("results/results_arxiv/filter_campaign_20261019_a30/data_cache"),
                   help="Folder holding one data cache per dataset (default: the Gaussian 2026-10-19 caches).")
    return p.parse_args(argv)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


class TrueMeasurement(eqx.Module):
    """Known measurement density in the caches' normalized coordinates (Gaussian or Lorenz GMM noise)."""
    obs_index: jax.Array
    state_mean: jax.Array
    state_std: jax.Array
    obs_mean: jax.Array
    obs_std: jax.Array
    log_w: jax.Array
    offsets: jax.Array
    stds: jax.Array

    def __init__(self, meta):
        f = lambda k: jnp.asarray(np.asarray(meta[k], float), jnp.float32)
        self.obs_index = jnp.asarray(meta["observation_indices"])
        self.state_mean, self.state_std, self.obs_mean, self.obs_std = (
            f("state_mean"), f("state_std"), f("observation_mean"), f("observation_std"))
        scale = float(meta["physical_observation_std"])
        if meta.get("observation_noise") == "gmm":
            self.log_w = jnp.log(f("gmm_weights"))
            self.offsets, self.stds = f("gmm_offsets") * scale, f("gmm_stds") * scale
        else:
            self.log_w, self.offsets, self.stds = jnp.zeros(1), jnp.zeros(1), jnp.full(1, scale)

    def log_prob(self, value, condition):
        """log p(y_norm | x_norm); value (..., O), condition (..., D), broadcast over leading axes."""
        y = value * self.obs_std + self.obs_mean
        hx = (condition * self.state_std + self.state_mean)[..., self.obs_index]
        r = (y - hx)[..., None] - self.offsets
        comp = self.log_w - jnp.log(self.stds) - .5 * jnp.log(2 * jnp.pi) - .5 * (r / self.stds) ** 2
        return jnp.sum(logsumexp(comp, axis=-1) + jnp.log(self.obs_std), axis=-1)


def load(method, run, data):
    meta = json.loads((run / "run_metadata.json").read_text())
    if method == "nf_dpf":
        from bayesinf.applications.state_estimation.normalizing_flow_dpf import NFDPFConfig, NormalizingFlowDPF
        model = eqx.tree_deserialise_leaves(run / "model.eqx",
                                            NormalizingFlowDPF(NFDPFConfig(**meta["model_config"]), key=jr.PRNGKey(0)))
        return model, 0.
    from .train_state_filter import build
    summary = json.loads((run / "summary.json").read_text()) if (run / "summary.json").is_file() else {}
    jitter = float(summary.get("jitter", meta.get("jitter", 0.)) or 0.)
    arguments = meta.get("arguments") or {}
    if "hidden_size" not in arguments:
        arguments = json.loads((resolve(meta.get("run") or arguments.get("run")) / "run_metadata.json").read_text())["arguments"]
    return eqx.tree_deserialise_leaves(run / "model.eqx", build(Namespace(**arguments), data)), jitter


def make_filter(method, model, n, jitter):
    """(x0, y, u, key) -> draws (n, T, D), ESS (T,)."""
    def run(x, y, u, key):
        x0 = x[0]  # only the known initial state; later states are scoring targets
        filter_key, draw_key = jr.split(key)
        if method == "nf_dpf":
            res = model.filter(y, x0, u, key=filter_key, num_particles=n, initial_std=0., resampling="systematic")
            particles, weights, ess = res.particles, res.weights, res.ess
        else:
            particles, weights, ess = model.filter(y, x0, u, key=filter_key, num_particles=n, jitter=jitter)
        keys = jr.split(draw_key, y.shape[0])
        draws = jax.vmap(lambda k, p, w: p[jr.categorical(k, jnp.log(w), shape=(n,))])(keys, particles, weights)
        return draws.swapaxes(0, 1), ess
    return jax.jit(jax.vmap(run))


def score(fn, split, data, windows, chunk, seed_base, n):
    count = min(windows, len(split["states"]))
    draws, ess = [], []
    for start in range(0, count, chunk):
        index = np.arange(start, min(start + chunk, count))
        keys = jnp.stack([jr.fold_in(jr.PRNGKey(seed_base), int(i)) for i in index])
        d, e = fn(*(jnp.asarray(split[k][index]) for k in ("states", "observations", "controls")), keys)
        draws.append(np.asarray(d)); ess.append(np.asarray(e))
    samples, ess = np.concatenate(draws), np.concatenate(ess) / n
    truth = np.asarray(split["states"][:count, 1:])
    if not np.isfinite(samples).all():
        return dict(nonfinite=True)
    meta, protocol = data.metadata, data.protocol
    m = metrics_from_samples(samples, truth, observed_indices=protocol["observed_indices"],
                             metric_space=protocol["metric_space"], state_mean=meta["state_mean"],
                             state_std=meta["state_std"])["metrics"]
    out = {k: m[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width", "interval_score_90")}
    out.update(ess_fraction_mean=float(ess.mean()), ess_fraction_p10=float(np.percentile(ess, 10)),
               steps_ess_below_10pct=float((ess < .1).mean()), windows=count)
    return out


def measurement_nll(model, truth_measurement, split):
    """Per-coordinate NLL of the observations at the true states: learned vs. known density."""
    y = jnp.asarray(split["observations"]).reshape(-1, split["observations"].shape[-1])
    x = jnp.asarray(split["states"][:, 1:]).reshape(-1, split["states"].shape[-1])
    learned = -jnp.mean(jax.vmap(model.measurement.log_prob)(y, x)) / y.shape[-1]
    true = -jnp.mean(jax.vmap(truth_measurement.log_prob)(y, x)) / y.shape[-1]
    return dict(learned_nll=float(learned), true_nll=float(true), kl_per_coordinate=float(learned - true))


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    rows_path = args.output / "rows.jsonl"
    done = {json.dumps(r["key"]) for r in map(json.loads, rows_path.read_text().splitlines())} if rows_path.exists() else set()

    def emit(key, **row):
        if json.dumps(key) in done:
            return
        row = dict(key=key, **row)
        with rows_path.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    cells = json.loads(args.cells.read_text())
    for dataset, methods in cells.items():
        if args.datasets and dataset not in args.datasets:
            continue
        data = load_cache(resolve(args.data_root / dataset))
        n_eval = data.protocol["posterior_samples"] if dataset != "lorenz96" else 200
        for method, seeds in methods.items():
            for seed, runs in seeds.items():
                selected = resolve(runs[0])
                model, jitter = load(method, selected, data)
                n_sel = int(json.loads((selected / "run_metadata.json").read_text()).get("particles") or n_eval)
                splits = dict(validation=(data.validation, args.validation_windows, 77001),
                              test=(data.test, args.test_windows or data.protocol["eval_windows"], 99001))

                def run_all(tag, fn, n, extra):
                    for split_name, (split, windows, base) in splits.items():
                        key = [dataset, method, seed, tag, split_name, extra]
                        if json.dumps(key) in done:
                            continue
                        emit(key, dataset=dataset, method=method, seed=int(seed), part=tag, split=split_name,
                             n=n, **extra, run=str(runs[0]), **score(fn, split, data, windows, args.chunk, base, n))

                if "particles" in args.parts:
                    for n in sorted(set(args.particle_counts) | {n_sel}):
                        run_all("particles", make_filter(method, model, n, jitter), n, dict(jitter=jitter, n_particles=n))
                if "measurement" in args.parts and dataset in ("pendulum", "lorenz96"):
                    truth = TrueMeasurement(data.metadata)
                    nll_key = [dataset, method, seed, "measurement_nll", "test", {}]
                    if json.dumps(nll_key) not in done:
                        emit(nll_key, dataset=dataset, method=method, seed=int(seed), part="measurement_nll",
                             split="test", run=str(runs[0]), **measurement_nll(model, truth, data.test))
                    swapped = eqx.tree_at(lambda m: m.measurement, model, truth)
                    run_all("true_measurement", make_filter(method, swapped, n_sel, jitter), n_sel,
                            dict(jitter=jitter, n_particles=n_sel))
                if "jitter" in args.parts and method in ("nsf", "flowsde"):
                    for run in runs:
                        candidate, _ = load(method, resolve(run), data)
                        diffusion = json.loads((resolve(run) / "run_metadata.json").read_text()).get(
                            "arguments", {}).get("diffusion")
                        for j in JITTERS:
                            run_all("jitter", make_filter(method, candidate, n_sel, j), n_sel,
                                    dict(jitter=j, diffusion=diffusion, candidate=str(run), n_particles=n_sel))


if __name__ == "__main__":
    main()
