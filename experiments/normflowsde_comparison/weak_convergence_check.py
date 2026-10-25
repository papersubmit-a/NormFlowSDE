"""Numerical check of the weak-convergence theorem (paper appendix, thm:weak) for the NormFlowSDE update.

The theorem fixes a trained update Z' = z + h0 f(z,t) + sqrt(h0) g(eps; z,t), eps ~ N(0, I), and refines the step
h -> 0 with the networks held fixed, using the centered, drift-corrected scheme

    Z_{k+1} = Z_k + h b_eff(Z_k, t_k) + sqrt(h) eta_c(Z_k, t_k, eps_k),
    m = E_eps[g],  eta_c = g - m,  b_eff = f + m / sqrt(h0),  A = E[eta_c eta_c^T],

whose piecewise-constant paths converge in law to dZ = b_eff dt + sigma dW with sigma sigma^T = A. No rate is
claimed. This script uses the actual SDEDriftNet / RealNVPDiffusionNet classes with fixed (untrained) weights
in d = 2, adds a contracting linear drift so the assumptions (Lipschitz, linear growth, bounded third noise
moment) hold, computes m and A per state by 2-D Gauss-Hermite quadrature (the centering corollary amplifies a
fixed mean error by 1/sqrt(h), so the quadrature error is checked), and compares, at h = h0 / 2^k:

  centered     the theorem's scheme (equals the trained update at h = h0)
  uncentered   z + h f + sqrt(h) g, the implemented kernel's naive refinement (a different limit: the mean
               term vanishes and the noise second moment replaces the covariance)

against an Euler-Maruyama reference of the limit SDE at a much finer step. Reported per scheme and step:
E[phi(Z_T)] for several test functions, skewness / excess kurtosis of Z_T, and E[max_t |Z_t|_inf] on the common
grid t = j h0, each with Monte Carlo standard errors.
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

from bayesinf.applications.state_estimation.flow_sde import RealNVPDiffusionNet, SDEDriftNet

# Full float32 matmuls: GPU TF32 defaults leave ~1e-3 quadrature errors in the flow mean, which the centering
# corollary amplifies by 1/sqrt(h) at the finest steps.
jax.config.update("jax_default_matmul_precision", "highest")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0, help="Network initialization seed.")
    p.add_argument("--h0", type=float, default=0.1, help="Training step of the fixed update.")
    p.add_argument("--horizon", type=float, default=2.0)
    p.add_argument("--refinements", type=int, default=5, help="Steps h = h0 / 2^k, k = 0..refinements.")
    p.add_argument("--reference-factor", type=int, default=128, help="Reference EM step = h0 / factor.")
    p.add_argument("--paths", type=int, default=65536)
    p.add_argument("--reference-paths", type=int, default=131072)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--quadrature", type=int, default=24, help="Gauss-Hermite nodes per dimension.")
    p.add_argument("--alpha", type=float, default=1.5, help="RealNVP coupling log-scale bound.")
    p.add_argument("--noise-scale", type=float, default=0.6)
    p.add_argument("--drift-scale", type=float, default=0.3)
    p.add_argument("--contraction", type=float, default=1.0)
    p.add_argument("--coupling-gain", type=float, default=8.0,
                   help="Multiplies each coupling net's output layer so the fixed flow noise is clearly non-Gaussian.")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.refinements, args.reference_factor, args.paths, args.reference_paths, args.chunk = 2, 8, 512, 512, 256
    return args


class FixedUpdate(eqx.Module):
    """f(z,t) = -K z + s_f MLP(z,c,t);  g(eps; z,t) = s_g RealNVP(eps; z,c,t), fixed weights, constant context."""
    drift: SDEDriftNet
    flow: RealNVPDiffusionNet
    context: jnp.ndarray
    contraction: float = eqx.field(static=True)
    drift_scale: float = eqx.field(static=True)
    noise_scale: float = eqx.field(static=True)

    def __init__(self, key, *, alpha, contraction, drift_scale, noise_scale, coupling_gain=1.0, d=2, d_c=4, hidden=32):
        k_drift, k_flow, k_context = jr.split(key, 3)
        self.drift = SDEDriftNet(d, d_c, hidden, 2, 16, k_drift)
        flow = RealNVPDiffusionNet(d, d_c, hidden, 4, 16, k_flow, alpha=alpha)
        for i in range(len(flow.layers)):
            flow = eqx.tree_at(lambda f: (f.layers[i].net.layers[-1].weight, f.layers[i].net.layers[-1].bias), flow,
                               (coupling_gain * flow.layers[i].net.layers[-1].weight,
                                coupling_gain * flow.layers[i].net.layers[-1].bias))
        self.flow = flow
        self.context = jr.normal(k_context, (d_c,))
        self.contraction, self.drift_scale, self.noise_scale = contraction, drift_scale, noise_scale

    def f(self, z, t):
        return -self.contraction * z + self.drift_scale * self.drift(z, self.context, t)

    def g(self, eps, z, t):
        return self.noise_scale * self.flow.apply(eps, z, self.context, t)


def gauss_hermite_2d(n):
    x, w = np.polynomial.hermite_e.hermegauss(n)  # weight exp(-x^2/2)
    w = w / math.sqrt(2 * math.pi)
    nodes = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
    weights = np.outer(w, w).reshape(-1)
    return jnp.asarray(nodes, jnp.float32), jnp.asarray(weights, jnp.float32)


def moments(model, nodes, weights, z, t):
    """m = E[g], A = Cov[g] at one state by quadrature."""
    values = jax.vmap(lambda e: model.g(e, z, t))(nodes)
    m = weights @ values
    centered = values - m
    return m, (centered * weights[:, None]).T @ centered


def simulate(model, nodes, weights, scheme, h, h0, steps, record_every, key, paths, chunk, z0=None):
    """Final states and the running max of |Z|_inf on the common grid t = j h0."""
    d = model.context.shape[0] // 2

    def one_path(path_key):
        def step(carry, k):
            z, running_max = carry
            t = k * h
            noise = jr.normal(jr.fold_in(path_key, k), (d,))
            if scheme == "uncentered":
                z_next = z + h * model.f(z, t) + jnp.sqrt(h) * model.g(noise, z, t)
            else:
                m, cov = moments(model, nodes, weights, z, t)
                drift = model.f(z, t) + m / jnp.sqrt(h0)
                if scheme == "centered":
                    z_next = z + h * drift + jnp.sqrt(h) * (model.g(noise, z, t) - m)
                else:  # Euler-Maruyama reference of the limit SDE
                    sigma = jnp.linalg.cholesky(cov + 1e-9 * jnp.eye(d))
                    z_next = z + h * drift + jnp.sqrt(h) * sigma @ noise
            on_grid = (k + 1) % record_every == 0
            running_max = jnp.where(on_grid, jnp.maximum(running_max, jnp.max(jnp.abs(z_next))), running_max)
            return (z_next, running_max), None

        start = jnp.zeros(d) if z0 is None else jnp.asarray(z0, jnp.float32)
        (z, running_max), _ = jax.lax.scan(step, (start, jnp.asarray(0.)), jnp.arange(steps))
        return z, running_max

    run = eqx.filter_jit(jax.vmap(one_path))
    keys = jr.split(key, paths)
    finals, maxima = [], []
    for start in range(0, paths, chunk):
        z, mx = run(keys[start:start + chunk])
        finals.append(np.asarray(z)), maxima.append(np.asarray(mx))
    return np.concatenate(finals), np.concatenate(maxima)


TEST_FUNCTIONS = {
    "z1": lambda z: z[:, 0], "z2": lambda z: z[:, 1], "z1^2": lambda z: z[:, 0] ** 2,
    "z1*z2": lambda z: z[:, 0] * z[:, 1], "cos(z1+z2)": lambda z: np.cos(z[:, 0] + z[:, 1]),
    "exp(-|z|^2)": lambda z: np.exp(-np.sum(z ** 2, 1)),
}


def summarize(z, maxima):
    out = {}
    for name, fn in TEST_FUNCTIONS.items():
        values = fn(z).astype(np.float64)
        out[name] = dict(mean=float(values.mean()), se=float(values.std(ddof=1) / math.sqrt(len(values))))
    out["E[max_t |Z_t|_inf]"] = dict(mean=float(maxima.mean()), se=float(maxima.std(ddof=1) / math.sqrt(len(maxima))))
    centered = z - z.mean(0)
    for i in range(z.shape[1]):
        std = centered[:, i].std()
        out[f"skewness z{i + 1}"] = dict(mean=float(np.mean(centered[:, i] ** 3) / std ** 3), se=None)
        out[f"excess kurtosis z{i + 1}"] = dict(mean=float(np.mean(centered[:, i] ** 4) / std ** 4 - 3), se=None)
    return out


def main(argv=None):
    args = parse_args(argv)
    if (args.output / "results.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = FixedUpdate(jr.PRNGKey(args.seed), alpha=args.alpha, contraction=args.contraction,
                        drift_scale=args.drift_scale, noise_scale=args.noise_scale, coupling_gain=args.coupling_gain)
    nodes, weights = gauss_hermite_2d(args.quadrature)
    fine_nodes, fine_weights = gauss_hermite_2d(40)

    # Quadrature accuracy and non-Gaussianity of the one-step noise at a few states.
    probe = jr.normal(jr.PRNGKey(1), (64, 2)) * 1.5
    m_q, a_q = jax.vmap(lambda z: moments(model, nodes, weights, z, 0.3))(probe)
    m_f, a_f = jax.vmap(lambda z: moments(model, fine_nodes, fine_weights, z, 0.3))(probe)
    noise = jax.vmap(lambda e: model.g(e, jnp.zeros(2), 0.))(jr.normal(jr.PRNGKey(2), (200000, 2)))
    noise = np.asarray(noise) - np.asarray(noise).mean(0)
    one_step = dict(
        quadrature_nodes=args.quadrature * args.quadrature,
        max_mean_error_vs_40x40=float(jnp.max(jnp.abs(m_q - m_f))),
        max_cov_error_vs_40x40=float(jnp.max(jnp.abs(a_q - a_f))),
        mean_norm_range=[float(jnp.min(jnp.linalg.norm(m_f, axis=1))), float(jnp.max(jnp.linalg.norm(m_f, axis=1)))],
        noise_skewness_at_z0=[float(np.mean(noise[:, i] ** 3) / noise[:, i].std() ** 3) for i in range(2)],
        noise_excess_kurtosis_at_z0=[float(np.mean(noise[:, i] ** 4) / noise[:, i].std() ** 4 - 3) for i in range(2)])
    print(json.dumps(dict(one_step=one_step)), flush=True)

    steps_h0 = round(args.horizon / args.h0)
    started = time.perf_counter()
    h_ref = args.h0 / args.reference_factor
    reference = summarize(*simulate(model, nodes, weights, "reference", h_ref, args.h0, steps_h0 * args.reference_factor,
                                    args.reference_factor, jr.PRNGKey(1000), args.reference_paths, args.chunk))
    print(json.dumps(dict(reference_step=h_ref, elapsed=time.perf_counter() - started)), flush=True)
    rows = []
    for scheme in ("centered", "uncentered"):
        for k in range(args.refinements + 1):
            factor = 2 ** k
            h = args.h0 / factor
            stats = summarize(*simulate(model, nodes, weights, scheme, h, args.h0, steps_h0 * factor, factor,
                                        jr.PRNGKey(2000 + 100 * k + (scheme == "uncentered")), args.paths, args.chunk))
            errors = {}
            for name, value in stats.items():
                ref = reference[name]
                se = None if value["se"] is None else math.sqrt(value["se"] ** 2 + ref["se"] ** 2)
                errors[name] = dict(value=value["mean"], reference=ref["mean"], error=value["mean"] - ref["mean"], se=se)
            rows.append(dict(scheme=scheme, h=h, refinement=factor, errors=errors))
            print(json.dumps(dict(scheme=scheme, h=h, **{n: round(e["error"], 5) for n, e in errors.items()})), flush=True)
    result = dict(created_at=datetime.now(timezone.utc).isoformat(), arguments={k: str(v) if isinstance(v, Path) else v
                  for k, v in vars(args).items()}, devices=[str(d) for d in jax.devices()], one_step=one_step,
                  reference=reference, reference_step=h_ref, rows=rows, elapsed_seconds=time.perf_counter() - started,
                  theorem="paper/normflowsde/sections/weak_convergence.tex thm:weak (no rate claimed)")
    (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    plot(result, args.output)


def plot(result, output, stem="weak_convergence"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["z1", "z1^2", "cos(z1+z2)", "E[max_t |Z_t|_inf]"]
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 3.4), constrained_layout=True)
    for ax, name in zip(axes, names):
        for scheme, style in (("centered", "o-"), ("uncentered", "s--")):
            rows = [r for r in result["rows"] if r["scheme"] == scheme]
            h = np.array([r["h"] for r in rows])
            err = np.array([abs(r["errors"][name]["error"]) for r in rows])
            se = np.array([r["errors"][name]["se"] for r in rows])
            ax.errorbar(h, err, yerr=1.96 * se, fmt=style, capsize=3, label=scheme)
        ax.set_xscale("log"), ax.set_yscale("log"), ax.invert_xaxis()  # refinement reads left to right
        ax.set_xlabel("inference step $h$, refined from $h_0$=0.1  $\\rightarrow$"), ax.set_title(f"|error| of E[{name}]")
    axes[0].set_ylabel("absolute error vs limit SDE (95% MC CI)")
    axes[0].legend()
    for ext in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{ext}", dpi=200)


if __name__ == "__main__":
    main()
