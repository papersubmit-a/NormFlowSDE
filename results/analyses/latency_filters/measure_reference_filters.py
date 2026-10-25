"""Latency of the model-based reference filters (EKF, UKF, bootstrap PF with the exact mixture likelihood) on the test
windows, measured exactly like inference_cost.py for the learned methods: one jitted batch-size-one full-window call,
compilation timed separately, then --repeats calls over different test windows with block_until_ready (device time).
Each filter runs at its validation-selected noise level (filters/<dataset>/summary.json, seed 0, no misspecification)
with the protocol particle count and posterior draw count. Output: latency.json next to this script."""
import json, time, argparse
from pathlib import Path
import jax, jax.numpy as jnp, jax.random as jr, numpy as np
from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.model_based_baselines import (pendulum_model, lorenz_model, measurement, ekf, ukf,
                                                                       particle_filter, physical_split)
P = Path(__file__).resolve().parents[2]  # <archive>/results; OUT = Path(__file__).resolve().parent
p = argparse.ArgumentParser(); p.add_argument("--repeats", type=int, default=30); args = p.parse_args()
rows = []
for ds in ("pendulum", "lorenz96"):
    data = load_cache(P / "datasets" / ds)
    meta = dict(data.metadata, dt=data.dt, state_dim=data.train["states"].shape[-1], observation_indices=data.protocol["observed_indices"])
    particles, draws = data.protocol["num_particles"], data.protocol["posterior_samples"]
    chosen = {r["filter"]: r["noise"] for r in json.loads((P / "filters" / ds / "summary.json").read_text())["rows"]
              if r["misspecification"] == 0 and r["seed"] == 0}
    x0s, ys, us, _ = physical_split(data, "test", data.protocol["eval_windows"])
    obs = jnp.asarray(meta["observation_indices"]); model = pendulum_model if ds == "pendulum" else lorenz_model
    for name in ("ekf", "ukf", "pf"):
        mean_step, sample_step, Q = model(meta, 0.0, chosen[name]); loglik, R = measurement(meta, ds, gaussian_only=False)
        if name in ("ekf", "ukf"):
            filt = ekf if name == "ekf" else ukf
            def sample(x0, y, u, key, _filt=filt, _ms=mean_step, _Q=Q, _R=R):
                means, covs = _filt(_ms, _Q, obs, _R, x0, y, u)
                chol = jnp.linalg.cholesky(covs + 1e-10 * jnp.eye(covs.shape[-1]))
                eps = jr.normal(key, (draws,) + means.shape)
                return means[None] + jnp.einsum("tij,mtj->mti", chol, eps)
        else:
            def sample(x0, y, u, key, _ss=sample_step, _ll=loglik):
                return particle_filter(_ss, _ll, obs, particles, draws, x0, y, u, key)
        fn = jax.jit(sample); windows = len(x0s)
        inputs = [tuple(jax.device_put(jnp.asarray(a[i % windows])) for a in (x0s[:, 0], ys, us)) for i in range(args.repeats + 1)]
        x, y, u = inputs[0]; started = time.perf_counter(); jax.block_until_ready(fn(x, y, u, jr.PRNGKey(0))); compile_seconds = time.perf_counter() - started
        times = []
        for r, (x, y, u) in enumerate(inputs[1:]):
            started = time.perf_counter(); jax.block_until_ready(fn(x, y, u, jr.PRNGKey(r + 1))); times.append(1e3 * (time.perf_counter() - started))
        row = dict(dataset=ds, filter=name, noise=chosen[name], particles=particles if name == "pf" else None, draws=draws,
                   device=str(jax.devices()[0]), repeats=args.repeats, compile_seconds=compile_seconds,
                   ms_per_window_median=float(np.median(times)), ms_per_window_p90=float(np.percentile(times, 90)),
                   ms_per_step_median=float(np.median(times)) / ys.shape[1])
        rows.append(row); print(json.dumps(row), flush=True)
(OUT / "latency.json").write_text(json.dumps(dict(protocol="as inference_cost.py: jitted batch-size-one full-window calls, compilation excluded, median over repeats", rows=rows), indent=1) + "\n")
print("saved", OUT / "latency.json")
