"""Can the shared-trunk NormFlowSDE filter be sampled faster without retraining? (2026-11-16)

Times model.sample_posterior against a rewritten Euler rollout of the same model and weights that moves everything
that does not depend on the sampled state out of the step loop:
  * context c[k] and the time embedding of every step, and their product with the first trunk layer (one matrix
    product for the whole window instead of one per step and draw);
  * the per-coordinate noise scale exp(log sigma) (constant);
  * the context lookup, the time-step check and the per-draw loops (one scan over time on a (draws, d) batch).
Optionally the scan is unrolled (--unroll k: k steps per loop iteration). Both samplers get the same Gaussian
increments, so their draws are compared directly (max abs difference, normalized units).
Latency protocol of inference_cost.py: jitted batch-size-one full-window calls, inputs on the device, the compiling
call excluded, median of --repeats calls on different test windows.
"""
import argparse, json, math, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np
from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.train_normflowsde_filter import filtering_inputs

def fast_sampler(model, n, unroll):
    trunk, noise, d, dt = model.drift, model.diffusion, model.d_x, model.time_step
    first, rest = trunk.net.layers[0], trunk.net.layers[1:]
    def sample(y_in, x0, key):
        T = y_in.shape[0]
        c = model.encode_observation_stride(y_in, model.observation_stride)                       # (T, d_c)
        ts = (jnp.arange(T + 1) * (T * dt / T))[:-1]
        temb = jax.vmap(trunk.time_emb)(ts if trunk.use_time else jnp.zeros_like(ts))
        pre = jnp.concatenate([c, temb], -1) @ first.weight[:, d:].T + first.bias                # (T, H)
        w_state = first.weight[:, :d].T
        sigma = math.sqrt(dt) * jnp.exp(noise.log_scale())
        eps = jax.vmap(lambda k: jr.normal(k, (T, d)))(jr.split(key, n)).swapaxes(0, 1)          # (T, n, d), as sample_posterior
        def step(z, item):
            pre_t, e = item
            h = jax.nn.gelu(z @ w_state + pre_t)
            for layer in rest:
                h = jax.nn.gelu(h @ layer.weight.T + layer.bias)
            v = e
            for layer in noise.layers:
                v = jax.vmap(layer.apply)(v, h)
            z1 = z + dt * (h @ trunk.head.weight.T + trunk.head.bias) + sigma * v
            return z1, z1
        return jax.lax.scan(step, jnp.broadcast_to(x0, (n, d)), (pre, eps), unroll=unroll)[1].swapaxes(0, 1)
    return jax.jit(sample)

def timed(fn, calls):
    started = time.perf_counter(); jax.block_until_ready(fn(*calls[0], jr.PRNGKey(0))); compile_s = time.perf_counter() - started
    times = []
    for r, item in enumerate(calls[1:]):
        started = time.perf_counter(); jax.block_until_ready(fn(*item, jr.PRNGKey(r + 1))); times.append(1e3 * (time.perf_counter() - started))
    return float(np.median(times)), float(np.percentile(times, 90)), compile_s

p = argparse.ArgumentParser(); p.add_argument("--sources", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
p.add_argument("--specs", nargs="+", required=True); p.add_argument("--repeats", type=int, default=30)
p.add_argument("--unroll", nargs="+", type=int, default=[1, 2, 4, 8])
args = p.parse_args(); sources = json.loads(args.sources.read_text()); rows = []
for spec in args.specs:
    dataset, n = spec.split(":"); n = int(n)
    run = ROOT / sources[dataset]["methods"]["normflowsde"]["seeds"]["0"]
    meta = json.loads((run / "run_metadata.json").read_text())
    model = eqx.tree_deserialise_leaves(run / "model.eqx", RealNVPFlowSDE(**meta["model_config"], key=jr.PRNGKey(0)))
    test = load_cache(ROOT / sources[dataset]["data_cache"]).test; windows = len(test["states"])
    calls = [(jax.device_put(filtering_inputs(jnp.asarray(test["observations"][i % windows]), jnp.asarray(test["controls"][i % windows]))),
              jax.device_put(jnp.asarray(test["states"][i % windows][0]))) for i in range(args.repeats + 1)]
    base = jax.jit(lambda y_in, x0, k: model.sample_posterior(y_in, x0, k, n_particles=n, solver_name="euler"))
    row = dict(dataset=dataset, n=n, horizon=int(calls[0][0].shape[0]), run=str(run.relative_to(ROOT)), device=str(jax.devices()[0]), variants={})
    ms, p90, comp = timed(base, calls); row["variants"]["sample_posterior"] = dict(ms=ms, p90=p90, compile_s=comp)
    reference = np.asarray(base(*calls[1], jr.PRNGKey(11)))
    for k in args.unroll:
        fn = fast_sampler(model, n, k); ms, p90, comp = timed(fn, calls)
        diff = float(np.abs(np.asarray(fn(*calls[1], jr.PRNGKey(11))) - reference).max())
        row["variants"][f"hoisted_unroll{k}"] = dict(ms=ms, p90=p90, compile_s=comp, max_abs_difference=diff)
    rows.append(row)
    print(dataset, "n", n, "T", row["horizon"], {name: (round(v["ms"], 2), "diff %.1e" % v["max_abs_difference"] if "max_abs_difference" in v else "", "compile %.0fs" % v["compile_s"]) for name, v in row["variants"].items()}, flush=True)
args.output.write_text(json.dumps(dict(rows=rows, repeats=args.repeats), indent=1) + "\n")
