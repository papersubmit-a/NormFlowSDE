"""Model-based filtering baselines (EKF, UKF, bootstrap PF) with exact and misspecified physics.

Pendulum and Lorenz-96 are simulated, so the data-generating model is known. Each filter is run with the
oracle model and with a physical-parameter error of relative size s, on the comparison's test windows,
with the same filtering access (x0 known, measurement y[k] used for x[k+1]), the same particle / draw
counts as the learned estimators, and the same scoring (metrics_from_samples on posterior.npz):

  pendulum  length x (1 + s), damping x (1 - s); measurement noise per cache metadata: Gaussian std 0.1 (2026-10-19
            cache) or the GMM mixture (2026-10-20 campaign; exact in "pf", moment-matched Gaussian in EKF / UKF / "pf_gauss")
  lorenz96  forcing F x (1 - s) (deterministic RK4 with x0 known: the oracle is essentially exact); GMM measurement noise: exact in the PF ("pf"), moment-matched Gaussian in the
            EKF / UKF and in "pf_gauss" (a likelihood-misspecified PF)

The process-noise scale is selected per filter, condition and seed on the validation windows (lowest RMSE), as
a practitioner would, from a fixed grid: inflation of the true pendulum diffusion, and an additive state-noise
std for the deterministic Lorenz-96 model. EKF / UKF posteriors are sampled from their Gaussian filtering
marginals; the PF draws from its weighted particles at each time (per-time marginals, as for the learned PFs).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .cache import load_cache
from .evaluation_metrics import metrics_from_samples

jax.config.update("jax_enable_x64", True)

FILTERS = dict(pendulum=("ekf", "ukf", "pf", "pf_gauss"), lorenz96=("ekf", "ukf", "pf", "pf_gauss"))
NOISE_GRID = dict(pendulum=(1.0, 2.0, 4.0, 8.0), lorenz96=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3))
MISSPECIFICATION = dict(pendulum=(0.0, 0.1, 0.25, 0.5), lorenz96=(0.0, 0.02, 0.05, 0.1, 0.25))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=tuple(FILTERS))
    p.add_argument("--data-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--filters", nargs="+")
    p.add_argument("--misspecification", nargs="+", type=float, help="Default: MISSPECIFICATION[dataset].")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--smoke", action="store_true", help="Few windows, few particles; plumbing check only.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------------------------
# Physical models
# ---------------------------------------------------------------------------------------------

def pendulum_model(meta, s, inflation):
    """8-substep Euler(-Maruyama) pendulum of ControlledStochasticPendulum with length x(1+s), damping x(1-s)."""
    gravity, mass, length, damping = 9.81, 1.0, 1.0 * (1 + s), 0.2 * (1 - s)
    substeps, dt = meta["integration_substeps"], meta["dt"]
    diffusion = np.sqrt(inflation) * np.array([0.01, 0.2])
    sub = dt / substeps

    def drift(x, u):
        return jnp.stack((x[1], u[0] / (mass * length ** 2) - damping * x[1] - gravity / length * jnp.sin(x[0])))

    def mean_step(x, u):
        return jax.lax.fori_loop(0, substeps, lambda _, z: z + drift(z, u) * sub, x)

    def sample_step(key, x, u):
        noise = jr.normal(key, (substeps, 2))
        return jax.lax.scan(lambda z, e: (z + drift(z, u) * sub + diffusion * jnp.sqrt(sub) * e, None), x, noise)[0]

    return mean_step, sample_step, jnp.diag(diffusion ** 2 * dt)


def lorenz_model(meta, s, noise_std):
    forcing, dt = meta["forcing"] * (1 - s), meta["dt"]

    def rhs(x):
        return (jnp.roll(x, -1) - jnp.roll(x, 2)) * jnp.roll(x, 1) - x + forcing

    def mean_step(x, u):
        k1 = rhs(x); k2 = rhs(x + dt / 2 * k1); k3 = rhs(x + dt / 2 * k2); k4 = rhs(x + dt * k3)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def sample_step(key, x, u):
        return mean_step(x, u) + noise_std * jr.normal(key, x.shape)

    return mean_step, sample_step, noise_std ** 2 * jnp.eye(meta["state_dim"])


def measurement(meta, dataset, gaussian_only):
    """Physical measurement log-likelihood of y given the observed coordinates, and its Gaussian variance.

    The noise family follows the cache metadata (2026-10-22 fix: the pendulum branch used to hard-code the
    Gaussian likelihood, which misspecifies the GMM pendulum data of the 2026-10-20 campaign)."""
    scale = meta["physical_observation_std"]
    if meta.get("observation_noise", "gaussian") != "gmm":
        return (lambda y, hx: jnp.sum(-0.5 * ((y - hx) / scale) ** 2)), scale ** 2
    w, mu, sd = (np.asarray(meta[k], float) for k in ("gmm_weights", "gmm_offsets", "gmm_stds"))
    mu, sd = mu * scale, sd * scale
    variance = float(np.sum(w * (sd ** 2 + mu ** 2)) - np.sum(w * mu) ** 2)
    if gaussian_only:
        return (lambda y, hx: jnp.sum(-0.5 * (y - hx) ** 2 / variance)), variance

    def gmm(y, hx):
        r = (y - hx)[:, None] - mu[None]
        comp = np.log(w) - np.log(sd) - 0.5 * (r / sd) ** 2
        return jnp.sum(jax.scipy.special.logsumexp(comp, axis=1))
    return gmm, variance


# ---------------------------------------------------------------------------------------------
# Filters (one window; vmapped over windows). Inputs physical: x0 (d,), ys (T, o), us (T, c).
# ---------------------------------------------------------------------------------------------

def kalman_update(m, P, y, obs, R):
    H = jnp.eye(m.shape[0])[obs]
    S = H @ P @ H.T + R * jnp.eye(len(obs))
    K = jnp.linalg.solve(S, H @ P).T
    A = jnp.eye(m.shape[0]) - K @ H
    return m + K @ (y - H @ m), A @ P @ A.T + R * K @ K.T


def ekf(mean_step, Q, obs, R, x0, ys, us):
    def step(carry, inp):
        m, P = carry
        y, u = inp
        F = jax.jacfwd(mean_step)(m, u)
        m, P = mean_step(m, u), F @ P @ F.T + Q
        m, P = kalman_update(m, P, y, obs, R)
        return (m, P), (m, P)
    return jax.lax.scan(step, (x0, 1e-8 * jnp.eye(x0.shape[0])), (ys, us))[1]


def ukf(mean_step, Q, obs, R, x0, ys, us):
    n = x0.shape[0]

    def step(carry, inp):  # alpha = 1, beta = 0, kappa = 0: 2n equally weighted sigma points
        m, P = carry
        y, u = inp
        L = jnp.linalg.cholesky(P + 1e-10 * jnp.eye(n)) * jnp.sqrt(n)
        sigma = jnp.concatenate((m + L.T, m - L.T))
        moved = jax.vmap(lambda z: mean_step(z, u))(sigma)
        m = moved.mean(0)
        P = (moved - m).T @ (moved - m) / (2 * n) + Q
        m, P = kalman_update(m, P, y, obs, R)
        return (m, P), (m, P)
    return jax.lax.scan(step, (x0, 1e-8 * jnp.eye(n)), (ys, us))[1]


def particle_filter(sample_step, loglik, obs, particles, draws, x0, ys, us, key):
    def step(carry, inp):
        x, logw, k = carry
        y, u, t = inp
        k, k_move, k_draw, k_res = jr.split(jr.fold_in(k, t), 4)
        x = jax.vmap(sample_step)(jr.split(k_move, particles), x, jnp.broadcast_to(u, (particles,) + u.shape))
        logw = logw + jax.vmap(lambda z: loglik(y, z[obs]))(x)
        logw = logw - jax.scipy.special.logsumexp(logw)
        out = x[jr.categorical(k_draw, logw, shape=(draws,))]
        w = jnp.exp(logw)
        ess = 1.0 / jnp.sum(w ** 2)
        positions = (jr.uniform(k_res) + jnp.arange(particles)) / particles
        resampled = x[jnp.clip(jnp.searchsorted(jnp.cumsum(w), positions), 0, particles - 1)]
        resample = ess < particles / 2
        x = jnp.where(resample, resampled, x)
        logw = jnp.where(resample, jnp.full(particles, -jnp.log(particles)), logw)
        return (x, logw, k), out
    start = (jnp.broadcast_to(x0, (particles,) + x0.shape), jnp.full(particles, -jnp.log(particles)), key)
    return jax.lax.scan(step, start, (ys, us, jnp.arange(ys.shape[0])))[1]


# ---------------------------------------------------------------------------------------------

def physical_split(data, split, count):
    meta = data.metadata
    sm, ss = np.asarray(meta["state_mean"]), np.asarray(meta["state_std"])
    om, os = np.asarray(meta["observation_mean"]), np.asarray(meta["observation_std"])
    cm, cs = np.asarray(meta["control_mean"]), np.asarray(meta["control_std"])
    s = getattr(data, split)
    count = min(count, len(s["states"]))
    return (s["states"][:count] * ss + sm, s["observations"][:count] * os + om, s["controls"][:count] * cs + cm,
            s["states"][:count, 1:])


def run_filter(name, dataset, meta, s, noise, x0, ys, us, key, particles, draws):
    model = pendulum_model if dataset == "pendulum" else lorenz_model
    mean_step, sample_step, Q = model(meta, s, noise)
    obs = jnp.asarray(meta["observation_indices"])
    loglik, R = measurement(meta, dataset, gaussian_only=(name == "pf_gauss"))
    keys = jr.split(key, len(x0))
    if name in ("ekf", "ukf"):
        fn = ekf if name == "ekf" else ukf
        means, covs = jax.jit(jax.vmap(lambda a, b, c: fn(mean_step, Q, obs, R, a, b, c)))(x0, ys, us)
        chol = jnp.linalg.cholesky(covs + 1e-10 * jnp.eye(covs.shape[-1]))
        eps = jax.vmap(lambda k: jr.normal(k, (draws,) + means.shape[1:]))(keys)
        return np.asarray(means[:, None] + jnp.einsum("wtij,wmtj->wmti", chol, eps))
    run = jax.jit(jax.vmap(lambda a, b, c, k: particle_filter(sample_step, loglik, obs, particles, draws, a, b, c, k)))
    out = []
    for start in range(0, len(x0), 8):
        out.append(np.asarray(run(x0[start:start + 8], ys[start:start + 8], us[start:start + 8], keys[start:start + 8])))
    return np.concatenate(out).transpose(0, 2, 1, 3)  # (windows, draws, T, d)


def score(samples_physical, truth_normalized, data):
    meta, protocol = data.metadata, data.protocol
    normalized = (samples_physical - np.asarray(meta["state_mean"])) / np.asarray(meta["state_std"])
    result = metrics_from_samples(normalized, truth_normalized, observed_indices=protocol["observed_indices"],
                                  metric_space=protocol["metric_space"], state_mean=meta["state_mean"],
                                  state_std=meta["state_std"])
    return normalized, result


def main(argv=None):
    args = parse_args(argv)
    data = load_cache(args.data_cache)
    meta = dict(data.metadata, dt=data.dt, state_dim=data.train["states"].shape[-1],
                observation_indices=data.protocol["observed_indices"])
    if data.protocol.get("dataset") != args.dataset:
        raise ValueError("cache/dataset mismatch")
    particles, draws = data.protocol["num_particles"], data.protocol["posterior_samples"]
    val_windows, test_windows = len(data.validation["states"]), data.protocol["eval_windows"]
    if args.smoke:
        particles, draws, val_windows, test_windows = 64, 16, 4, 4
    validation = physical_split(data, "validation", val_windows)
    test = physical_split(data, "test", test_windows)
    summary = []
    for name in args.filters or FILTERS[args.dataset]:
        for s in args.misspecification or MISSPECIFICATION[args.dataset]:
            for seed in args.seeds:
                target = args.output / name / f"s{s:g}" / f"seed{seed}"
                if (target / "summary.json").exists():
                    summary.append(json.loads((target / "summary.json").read_text()))
                    continue
                started = time.perf_counter()
                selection = []
                for noise in NOISE_GRID[args.dataset]:
                    x0, ys, us, truth = validation
                    samples = run_filter(name, args.dataset, meta, s, noise, x0[:, 0], ys, us, jr.PRNGKey(77001 + seed),
                                         particles, draws)
                    if not np.isfinite(samples).all():
                        selection.append(dict(noise=noise, rmse=None))
                        continue
                    selection.append(dict(noise=noise, rmse=score(samples, truth, data)[1]["metrics"]["rmse"]))
                valid = [row for row in selection if row["rmse"] is not None]
                chosen = min(valid, key=lambda row: row["rmse"])["noise"]
                x0, ys, us, truth = test
                samples = run_filter(name, args.dataset, meta, s, chosen, x0[:, 0], ys, us, jr.PRNGKey(99001 + seed),
                                     particles, draws)
                normalized, result = score(samples, truth, data)
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / "posterior.npz", samples_normalized=normalized.astype(np.float32),
                                    truth_normalized=truth.astype(np.float32))
                (target / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
                row = dict(dataset=args.dataset, filter=name, misspecification=s, seed=seed, noise=chosen,
                           selection=selection, particles=particles if name.startswith("pf") else None, draws=draws,
                           seconds=time.perf_counter() - started,
                           test={k: result["metrics"][k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width")})
                (target / "summary.json").write_text(json.dumps(row, indent=2) + "\n")
                summary.append(row)
                print(json.dumps({k: row[k] for k in ("filter", "misspecification", "seed", "noise", "test")}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(dict(created_at=datetime.now(timezone.utc).isoformat(),
        dataset=args.dataset, data_cache=str(args.data_cache), smoke=args.smoke, rows=summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
