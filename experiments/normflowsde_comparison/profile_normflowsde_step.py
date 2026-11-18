"""Where does the inference time of the amortized NormFlowSDE filter go?

For the selected seed-0 checkpoint of each dataset, one full-window filtering call with the protocol number of
draws is timed as a whole and in parts, with the latency protocol of inference_cost.py (jitted batch-size-one
calls, inputs on the device, the first compiling call excluded, median over --repeats calls on different test
windows, block_until_ready).

  full            model.sample_posterior, as timed for the paper's latency column
  encoder         one pass of the window encoder over the whole window (context c[0:T])
  encoder_twice   the two encoder passes that sample_posterior requests (drift context and noise context)
  rollout_*       the Euler rollout of n draws from a precomputed context, with only the named networks active:
                    none          scan, random numbers and the state update, no network
                    drift         drift network only
                    scale         learned per-coordinate scale network only
                    flow_layer1   first coupling layer of the noise flow only
                    flow          all coupling layers of the noise flow
                    all           drift + flow + scale (the full update)

The cost attributed to a component is its rollout time minus rollout_none. Parts of a jitted program are not
exactly additive, so full is reported next to encoder_twice + rollout_all as a consistency check. Ablated
rollouts use the trained weights but may leave the data range; only their run time is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .cache import load_cache
from .train_normflowsde_filter import filtering_inputs

ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", required=True, type=Path, help="build_results_paper source map.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--specs", nargs="+", required=True, help="dataset:draws, e.g. pendulum:128")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repeats", type=int, default=30)
    return p.parse_args(argv)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def count(tree):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))))


def timed(fn, inputs):
    """inputs: list of argument tuples already on the device; the first one compiles and is excluded."""
    started = time.perf_counter()
    jax.block_until_ready(fn(*inputs[0], jr.PRNGKey(0)))
    compile_seconds = time.perf_counter() - started
    times = []
    for r, item in enumerate(inputs[1:]):
        started = time.perf_counter()
        jax.block_until_ready(fn(*item, jr.PRNGKey(r + 1)))
        times.append(1e3 * (time.perf_counter() - started))
    return dict(ms_median=float(np.median(times)), ms_p90=float(np.percentile(times, 90)),
                ms_min=float(np.min(times)), compile_seconds=compile_seconds)


def profile(dataset, n, run, data, repeats):
    from bayesinf.applications.state_estimation.flow_sde import LearnedScaleDiffusionNet, RealNVPFlowSDE
    meta = json.loads((run / "run_metadata.json").read_text())
    model = eqx.tree_deserialise_leaves(run / "model.eqx", RealNVPFlowSDE(**meta["model_config"], key=jr.PRNGKey(0)))
    solver = meta.get("solver") or (meta.get("recipe") or {}).get("solver", "euler")
    if solver != "euler":
        raise ValueError(f"{dataset}: the part timings assume the Euler update, the run uses {solver}")
    if model.physics_fn is not None:
        raise ValueError("physics_fn is not covered by the part timings")
    has_scale = isinstance(model.diffusion, LearnedScaleDiffusionNet)
    flow = model.diffusion.flow if has_scale else model.diffusion
    dt, stride = model.time_step, model.observation_stride
    root_dt = math.sqrt(dt)

    split, windows = data.test, len(data.test["states"])
    raw = [tuple(jnp.asarray(split[name][i % windows]) for name in ("states", "observations", "controls"))
           for i in range(repeats + 1)]
    # (x0, y_in) per call, on the device; y_in are the measurements concatenated with the controls.
    calls = [(jax.device_put(x[0]), jax.device_put(filtering_inputs(y, u))) for x, y, u in raw]
    horizon = int(calls[0][1].shape[0])

    def encode_both(y_in):
        return model.encode_observation_stride(y_in, stride), model.encode_diffusion_stride(y_in, stride)

    contexts = jax.jit(encode_both)
    ctx_calls = []
    for x0, y_in in calls:
        c, cd = jax.block_until_ready(contexts(y_in))
        ctx_calls.append((x0, c, cd))

    def rollout(parts, flow_layers=None):
        layers = flow.layers if flow_layers is None else flow.layers[:flow_layers]

        def fn(x0, c, cd, key):
            ts = jnp.arange(horizon, dtype=x0.dtype) * dt

            def one(k):
                eps = jr.normal(k, (horizon, model.d_x), x0.dtype)

                def step(z, item):
                    c_t, cd_t, t, e = item
                    update = z
                    if "drift" in parts:
                        update = update + dt * model.drift(z, c_t, t)
                    noise = e
                    if "flow" in parts:
                        for layer in layers:
                            noise = layer.apply(noise, z, cd_t, t)
                    if "scale" in parts:
                        noise = jnp.exp(model.diffusion.log_scale(z, cd_t, t)) * noise
                    z1 = update + root_dt * noise
                    return z1, z1

                return jax.lax.scan(step, x0, (c, cd, ts, eps))[1]

            return jax.vmap(one)(jr.split(key, n))
        return jax.jit(fn)

    variants = {
        "full": (jax.jit(lambda x0, y_in, k: model.sample_posterior(y_in, x0, k, n_particles=n, solver_name=solver)), calls),
        "encoder": (jax.jit(lambda x0, y_in, k: model.encode_observation_stride(y_in, stride)), calls),
        "encoder_twice": (jax.jit(lambda x0, y_in, k: encode_both(y_in)), calls),
        "rollout_none": (rollout(()), ctx_calls),
        "rollout_drift": (rollout(("drift",)), ctx_calls),
        "rollout_flow_layer1": (rollout(("flow",), 1), ctx_calls),
        "rollout_flow": (rollout(("flow",)), ctx_calls),
        "rollout_all": (rollout(("drift", "flow", "scale") if has_scale else ("drift", "flow")), ctx_calls),
    }
    if has_scale:
        variants["rollout_scale"] = (rollout(("scale",)), ctx_calls)

    timings = {}
    for name, (fn, inputs) in variants.items():
        timings[name] = timed(fn, inputs)
        print(f"  {dataset:10s} {name:20s} {timings[name]['ms_median']:8.3f} ms", flush=True)
    base = timings["rollout_none"]["ms_median"]
    attributed = {name[len("rollout_"):]: timings[name]["ms_median"] - base
                  for name in timings if name.startswith("rollout_") and name != "rollout_none"}
    return dict(
        dataset=dataset, n=n, run=str(run), solver=solver, horizon=horizon, state_dim=model.d_x,
        observation_window=model.observation_window, coupling_layers=len(flow.layers), learned_scale=has_scale,
        parameters=dict(total=count(model), encoder=count(model.encoder), drift=count(model.drift), flow=count(flow),
                        scale=count(model.diffusion.scale) if has_scale else 0),
        timings=timings, rollout_overhead_ms=base, attributed_ms=attributed,
        check=dict(full_ms=timings["full"]["ms_median"],
                   encoder_twice_plus_rollout_all_ms=timings["encoder_twice"]["ms_median"] + timings["rollout_all"]["ms_median"],
                   encoder_once_plus_rollout_all_ms=timings["encoder"]["ms_median"] + timings["rollout_all"]["ms_median"]),
    )


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    device = str(jax.devices()[0])
    rows = []
    for spec in args.specs:
        dataset, n = spec.split(":")
        entry = sources[dataset]
        data = load_cache(resolve(entry["data_cache"]))
        run = resolve(entry["methods"]["normflowsde"]["seeds"][str(args.seed)])
        print(f"{dataset}: n={n}, run={run}, device={device}", flush=True)
        row = profile(dataset, int(n), run, data, args.repeats) | dict(device=device, repeats=args.repeats)
        rows.append(row)
        with (args.output / "rows.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    (args.output / "run.json").write_text(json.dumps(dict(
        finished_at=datetime.now(timezone.utc).isoformat(), device=device, sources=str(args.sources),
        specs=args.specs, seed=args.seed, repeats=args.repeats), indent=2) + "\n")
    print("\ndataset     full  enc  enc x2  none  drift  scale  flow  flow/layer1  all   (ms per window, median)")
    for row in rows:
        t = {k: v["ms_median"] for k, v in row["timings"].items()}
        print(f"{row['dataset']:10s} {t['full']:5.2f} {t['encoder']:5.2f} {t['encoder_twice']:6.2f} {t['rollout_none']:5.2f} "
              f"{t['rollout_drift']:6.2f} {t.get('rollout_scale', float('nan')):6.2f} {t['rollout_flow']:5.2f} "
              f"{t['rollout_flow_layer1']:11.2f} {t['rollout_all']:5.2f}")


if __name__ == "__main__":
    main()
