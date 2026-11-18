"""Does the centered refinement of a *trained* NormFlowSDE transition recover a standard SDE?

Ground truth is a 2-D Ornstein-Uhlenbeck process  dZ = -K Z dt + S dW  (rotating drift K, correlated noise S).
Training transitions at step h0 use non-Gaussian increments with the same first two moments,

    z' = z - h0 K z + sqrt(h0) S eta,   eta_i i.i.d. skewed two-component mixture, E[eta] = 0, Cov[eta] = I,

so the one-step law is clearly non-Gaussian, while the only diffusion limit with these first two moments is
the OU process itself. A NormFlowSDE transition (SDEDriftNet drift + conditional RealNVP noise flow, time-
invariant) is fitted to these pairs by its exact Euler-flow likelihood. Then, with the networks fixed, the
paper's centered refinement (b_eff = f + m / sqrt(h0), eta_c = g - m, step h = h0 / 2^k) is compared with
the analytic OU law at time T. Also shown: the uncentered refinement z + h f + sqrt(h) g (control), and the
limit SDE of the trained model itself (Euler-Maruyama at h0 / 128), whose distance to the OU is the
learning error that remains when h -> 0. The learned drift / diffusion b_eff(z), A(z) are reported next to
-K z and S S^T.
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
import optax
import scipy.linalg

from bayesinf.applications.state_estimation.flow_sde import RealNVPDiffusionNet, SDEDriftNet
from .weak_convergence_check import gauss_hermite_2d, moments, simulate

jax.config.update("jax_default_matmul_precision", "highest")

K = np.array([[1.0, 0.8], [-0.8, 1.0]])
S = np.array([[0.6, 0.0], [0.3, 0.4]])
Z0 = np.array([2.0, -1.0])
# Skewed mixture: w N(a, s^2) + (1 - w) N(-3a, s^2) with w = 0.75, 3 a^2 + s^2 = 1 (zero mean, unit variance).
MIX_A, MIX_W = 0.4, 0.75
MIX_S = math.sqrt(1 - 3 * MIX_A ** 2)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h0", type=float, default=0.1)
    p.add_argument("--horizon", type=float, default=2.0)
    p.add_argument("--train-steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--refinements", type=int, default=5)
    p.add_argument("--reference-factor", type=int, default=128)
    p.add_argument("--paths", type=int, default=65536)
    p.add_argument("--reference-paths", type=int, default=131072)
    p.add_argument("--truth-paths", type=int, default=1048576)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--quadrature", type=int, default=24)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.train_steps, args.refinements, args.reference_factor = 50, 1, 4
        args.paths = args.reference_paths = args.truth_paths = 512
        args.chunk, args.quadrature = 256, 8
    return args


class TrainedUpdate(eqx.Module):
    """Time-invariant NormFlowSDE transition z' = z + h0 f(z) + sqrt(h0) g(eps; z); constant (zero) context."""
    drift: SDEDriftNet
    flow: RealNVPDiffusionNet
    context: jnp.ndarray

    def __init__(self, key, d=2, d_c=4, hidden=64):
        k_drift, k_flow = jr.split(key)
        self.drift = SDEDriftNet(d, d_c, hidden, 2, 16, k_drift, use_time=False)
        self.flow = RealNVPDiffusionNet(d, d_c, hidden, 4, 16, k_flow, alpha=1.5, use_time=False)
        self.context = jnp.zeros(d_c)

    def f(self, z, t):
        return self.drift(z, self.context, t)

    def g(self, eps, z, t):
        return self.flow.apply(eps, z, self.context, t)

    def nll(self, z, z_next, h0):
        residual = (z_next - z - h0 * self.f(z, 0.)) / jnp.sqrt(h0)
        eps, log_det = self.flow.inverse_and_log_det(residual, z, self.context, 0.)
        return 0.5 * jnp.sum(eps ** 2) + log_det + 0.5 * eps.shape[0] * jnp.log(2 * jnp.pi * h0)


def mixture_noise(key, shape):
    k_label, k_normal = jr.split(key)
    first = jr.bernoulli(k_label, MIX_W, shape)
    mean = jnp.where(first, MIX_A, -3 * MIX_A)
    return mean + MIX_S * jr.normal(k_normal, shape)


def training_pairs(key, batch, h0):
    k_state, k_noise = jr.split(key)
    z = 2.5 * jr.normal(k_state, (batch, 2))
    eta = mixture_noise(k_noise, (batch, 2))
    return z, z - h0 * z @ jnp.asarray(K).T + jnp.sqrt(h0) * eta @ jnp.asarray(S).T


def train(model, args):
    schedule = optax.warmup_cosine_decay_schedule(0., args.learning_rate, min(500, args.train_steps // 10 + 1), args.train_steps, args.learning_rate * .05)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adam(schedule))
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(m, s, key):
        z, z_next = training_pairs(key, args.batch, args.h0)
        loss, grads = eqx.filter_value_and_grad(
            lambda mm: jnp.mean(jax.vmap(lambda a, b: mm.nll(a, b, args.h0))(z, z_next)))(m)
        updates, s = optimizer.update(grads, s, eqx.filter(m, eqx.is_inexact_array))
        return eqx.apply_updates(m, updates), s, loss

    history = []
    for step in range(args.train_steps):
        model, state, loss = update(model, state, jr.fold_in(jr.PRNGKey(args.seed + 10), step))
        if (step + 1) % max(1, args.train_steps // 20) == 0:
            history.append(dict(step=step + 1, nll=float(loss)))
            print(json.dumps(history[-1]), flush=True)
    return model, history


def ou_law(t):
    """Analytic mean and covariance of the OU at time t from Z0 (Van Loan)."""
    A, Q = -K, S @ S.T
    block = scipy.linalg.expm(np.block([[A, Q], [np.zeros((2, 2)), -A.T]]) * t)
    F, G = block[:2, :2], block[:2, 2:]
    return F @ Z0, G @ F.T


def gaussian_expectations(mu, C):
    a = C[0, 0] + 2 * C[0, 1] + C[1, 1]
    M = np.eye(2) + 2 * C
    return {"z1": mu[0], "z2": mu[1], "z1^2": C[0, 0] + mu[0] ** 2, "z1*z2": C[0, 1] + mu[0] * mu[1],
            "cos(z1+z2)": math.cos(mu[0] + mu[1]) * math.exp(-a / 2),
            "exp(-|z|^2)": float(np.linalg.det(M) ** -0.5 * math.exp(-mu @ np.linalg.solve(M, mu))),
            "skewness z1": 0., "excess kurtosis z1": 0., "skewness z2": 0., "excess kurtosis z2": 0.}


def exact_grid_max(args):
    """E[max_j |Z_{j h0}|_inf] for the OU by exact transition sampling on the common grid."""
    F = scipy.linalg.expm(-K * args.h0)
    _, C = ou_law(args.h0)
    # Transition covariance from any start equals the from-Z0 covariance at h0 (it does not depend on the start).
    L = np.linalg.cholesky(C)
    rng = np.random.default_rng(7)
    z = np.tile(Z0, (args.truth_paths, 1))
    running = np.zeros(args.truth_paths)
    for _ in range(round(args.horizon / args.h0)):
        z = z @ F.T + rng.standard_normal(z.shape) @ L.T
        running = np.maximum(running, np.abs(z).max(1))
    return float(running.mean()), float(running.std(ddof=1) / math.sqrt(len(running)))


def summarize(z, maxima):
    out = {}
    for name, fn in {"z1": lambda x: x[:, 0], "z2": lambda x: x[:, 1], "z1^2": lambda x: x[:, 0] ** 2,
                     "z1*z2": lambda x: x[:, 0] * x[:, 1], "cos(z1+z2)": lambda x: np.cos(x[:, 0] + x[:, 1]),
                     "exp(-|z|^2)": lambda x: np.exp(-np.sum(x ** 2, 1))}.items():
        v = fn(z).astype(np.float64)
        out[name] = dict(mean=float(v.mean()), se=float(v.std(ddof=1) / math.sqrt(len(v))))
    out["E[max_t |Z_t|_inf]"] = dict(mean=float(maxima.mean()), se=float(maxima.std(ddof=1) / math.sqrt(len(maxima))))
    c = z - z.mean(0)
    for i in range(2):
        sd = c[:, i].std()
        out[f"skewness z{i + 1}"] = dict(mean=float(np.mean(c[:, i] ** 3) / sd ** 3), se=None)
        out[f"excess kurtosis z{i + 1}"] = dict(mean=float(np.mean(c[:, i] ** 4) / sd ** 4 - 3), se=None)
    return out


def errors(stats, truth):
    return {name: dict(value=v["mean"], truth=truth[name]["mean"], error=v["mean"] - truth[name]["mean"],
                       se=None if v["se"] is None else math.sqrt(v["se"] ** 2 + (truth[name]["se"] or 0.) ** 2))
            for name, v in stats.items()}


def main(argv=None):
    args = parse_args(argv)
    if (args.output / "results.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, history = train(TrainedUpdate(jr.PRNGKey(args.seed)), args)
    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
    nodes, weights = gauss_hermite_2d(args.quadrature)

    # Learned limit coefficients vs the OU's.
    grid = jnp.asarray([[x, y] for x in (-2., 0., 2.) for y in (-2., 0., 2.)])
    m, A = jax.vmap(lambda z: moments(model, nodes, weights, z, 0.))(grid)
    b_eff = jax.vmap(lambda z: model.f(z, 0.))(grid) + m / math.sqrt(args.h0)
    coefficients = dict(points=np.asarray(grid).tolist(), learned_drift=np.asarray(b_eff).tolist(),
                        true_drift=(-np.asarray(grid) @ K.T).tolist(), learned_diffusion=np.asarray(A).tolist(),
                        true_diffusion=(S @ S.T).tolist(),
                        max_drift_error=float(np.abs(np.asarray(b_eff) + np.asarray(grid) @ K.T).max()),
                        max_diffusion_error=float(np.abs(np.asarray(A) - S @ S.T).max()),
                        flow_noise_mean_range=[float(jnp.min(jnp.abs(m))), float(jnp.max(jnp.abs(m)))])
    print(json.dumps({k: v for k, v in coefficients.items() if k.startswith(("max", "flow"))}), flush=True)

    mu, C = ou_law(args.horizon)
    truth = {k: dict(mean=float(v), se=None) for k, v in gaussian_expectations(mu, C).items()}
    grid_max = exact_grid_max(args)
    truth["E[max_t |Z_t|_inf]"] = dict(mean=grid_max[0], se=grid_max[1])

    steps_h0 = round(args.horizon / args.h0)
    h_ref = args.h0 / args.reference_factor
    limit = summarize(*simulate(model, nodes, weights, "reference", h_ref, args.h0, steps_h0 * args.reference_factor,
                                args.reference_factor, jr.PRNGKey(1000), args.reference_paths, args.chunk, z0=Z0))
    rows = [dict(scheme="trained limit SDE", h=h_ref, errors=errors(limit, truth))]
    for scheme in ("centered", "uncentered"):
        for k in range(args.refinements + 1):
            factor = 2 ** k
            stats = summarize(*simulate(model, nodes, weights, scheme, args.h0 / factor, args.h0, steps_h0 * factor,
                                        factor, jr.PRNGKey(2000 + 100 * k + (scheme == "uncentered")), args.paths,
                                        args.chunk, z0=Z0))
            rows.append(dict(scheme=scheme, h=args.h0 / factor, errors=errors(stats, truth)))
            print(json.dumps(dict(scheme=scheme, h=args.h0 / factor,
                                  **{n: round(e["error"], 4) for n, e in rows[-1]["errors"].items()})), flush=True)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  devices=[str(d) for d in jax.devices()], ou=dict(K=K.tolist(), S=S.tolist(), z0=Z0.tolist()),
                  increment_noise=dict(kind="per-coordinate skewed mixture", w=MIX_W, a=MIX_A, s=MIX_S,
                                       skewness=float(MIX_W * (MIX_A ** 3 + 3 * MIX_A * MIX_S ** 2)
                                                      + (1 - MIX_W) * ((-3 * MIX_A) ** 3 - 9 * MIX_A * MIX_S ** 2))),
                  training=history, coefficients=coefficients, truth=truth, rows=rows,
                  elapsed_seconds=time.perf_counter() - started)
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.output)


def plot(result, output, stem="weak_convergence_ou"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    names = [("z1", "mean $E[z_1]$"), ("z1^2", "second moment $E[z_1^2]$"),
             ("E[max_t |Z_t|_inf]", "path maximum $E[\\max\\,\\|Z_t\\|_\\infty]$")]
    fig, axes = plt.subplots(1, 3, figsize=(7, 2.6), constrained_layout=True)
    limit = next(r for r in result["rows"] if r["scheme"] == "trained limit SDE")
    for ax, (name, title) in zip(axes, names):
        for scheme, style, label in (("centered", "o-", "centered refinement (Theorem)"),
                                     ("uncentered", "s--", "uncentered refinement")):
            rows = [r for r in result["rows"] if r["scheme"] == scheme]
            h = np.array([r["h"] for r in rows])
            err = np.array([abs(r["errors"][name]["error"]) for r in rows])
            se = np.array([r["errors"][name]["se"] or 0. for r in rows])
            ax.errorbar(h, err, yerr=1.96 * se, fmt=style, capsize=2, ms=3, label=label)
        ax.axhline(abs(limit["errors"][name]["error"]), color="gray", ls=":", label="limit SDE of the trained model")
        ax.set_xscale("log"), ax.set_yscale("log"), ax.invert_xaxis()  # refinement reads left to right
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("step $h$ (refined from $h_0=0.1$ $\\rightarrow$)")
    axes[0].set_ylabel("|model $-$ OU| at $T=2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=3, fontsize=8, frameon=False)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{ext}", dpi=200)

if __name__ == "__main__":
    main()
