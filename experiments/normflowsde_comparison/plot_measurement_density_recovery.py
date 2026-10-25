"""Learned measurement density vs the true GMM observation noise (option 1 of the noise-recovery figures).

For the three particle-filter baselines (NF-DPF, NSF, FlowSDE) the learned conditional flow q(y | x) is evaluated on
one-coordinate slices y = x + delta e_j around test states x (observation = identity on ALOHA), normalized over the
slice, and averaged over states and coordinates. NormFlowSDE has no measurement model; its panel shows the implied
residual distribution y - E[x | y] from its saved posterior, which is also drawn (dashed) in the other panels.
Reference: the analytic GMM of the cache protocol and the empirical residuals y - x of the test data.

Usage: python -m experiments.normflowsde_comparison.plot_measurement_density_recovery --dataset aloha \
         --sources results/results_arxiv/equal_n_gmm_final_20261021/sources.json --out results/results_normflowsde_paper/analyses/noise_recovery/aloha
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cache import load_cache
from .inference_cost import resolve

p = argparse.ArgumentParser()
p.add_argument("--dataset", default="aloha")
p.add_argument("--sources", type=Path, required=True)
p.add_argument("--out", required=True, help="output stem")
p.add_argument("--seed", default="0")
p.add_argument("--states", type=int, default=100, help="random test (window, step) pairs per method")
p.add_argument("--grid", type=float, default=None, help="half-width of the residual grid in noise units (default 8 obs_std)")
p.add_argument("--points", type=int, default=161)
p.add_argument("--font-scale", type=float, default=1.4)
p.add_argument("--no-titles", action="store_true")
args = p.parse_args()

src = json.loads(args.sources.read_text())[args.dataset]
data = load_cache(resolve(src["data_cache"]))
test = data.test
obs_idx = np.array(data.metadata.get("observation_indices") or list(range(test["observations"].shape[-1])))
proto = data.protocol
sigma = float(proto["obs_std"])
w_mix = np.array(data.metadata.get("gmm_weights") or [0.7, 0.15, 0.15])
mu_mix = np.array(data.metadata.get("gmm_offsets") or [0.0, 2.5, -2.5]) * sigma
sd_mix = np.array(data.metadata.get("gmm_stds") or [0.5, 1.5, 1.5]) * sigma
assert proto.get("obs_noise_type") == "gmm", proto.get("obs_noise_type")
# Pendulum / Lorenz-96 caches store normalized arrays but define the noise in physical units (metadata
# physical_observation_std); ALOHA / REASSEMBLE add the noise after normalization. Residuals are compared in the units of
# the noise definition, so normalized residuals of the former are rescaled per coordinate by the observation std.
physical = "physical_observation_std" in data.metadata
scale = np.array(data.metadata["observation_std"], dtype=float) if physical else np.ones(len(obs_idx))
if physical: assert abs(float(data.metadata["physical_observation_std"]) - sigma) < 1e-9
units = "physical units" if physical else "normalized units"
if args.grid is None: args.grid = 8.0 * sigma

def gmm_pdf(d):
    return sum(w / (sd * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((d - m) / sd) ** 2) for w, m, sd in zip(w_mix, mu_mix, sd_mix))

grid = np.linspace(-args.grid, args.grid, args.points); dd = grid[1] - grid[0]
x_next = test["states"][:, 1:]                   # observation k measures state k+1 (all three PF models train that pairing)
resid_data = ((test["observations"] - x_next[..., obs_idx]) * scale).ravel()
print(f"data residuals: std {resid_data.std():.4f} (analytic {np.sqrt((w_mix*(sd_mix**2+mu_mix**2)).sum()-(w_mix*mu_mix).sum()**2):.4f})")

def load_measurement(method, run):
    meta = json.loads((run / "run_metadata.json").read_text())
    if method == "nf_dpf":
        from bayesinf.applications.state_estimation.normalizing_flow_dpf import NFDPFConfig, NormalizingFlowDPF
        model = eqx.tree_deserialise_leaves(run / "model.eqx", NormalizingFlowDPF(NFDPFConfig(**meta["model_config"]), key=jr.PRNGKey(0)))
        return model.measurement
    from .train_state_filter import build
    arguments = meta.get("arguments") or {}
    if "hidden_size" not in arguments:
        source = resolve(meta.get("run") or arguments.get("run"))
        arguments = json.loads((source / "run_metadata.json").read_text())["arguments"]
    model = eqx.tree_deserialise_leaves(run / "model.eqx", build(Namespace(**arguments), data))
    return model.measurement

rng = np.random.default_rng(0)
W, T = x_next.shape[:2]
pairs = np.stack([rng.integers(0, W, args.states), rng.integers(0, T, args.states)], 1)
d_y = len(obs_idx)

def density_slices(flow):
    @jax.jit
    def slices(x):
        y0 = x[obs_idx]
        ys = y0[None, None, :] + (grid[None, :, None] / scale[:, None, None]) * jnp.eye(d_y)[:, None, :]  # (d_y, G, d_y), grid in noise units
        lp = flow.log_prob(ys.reshape(-1, d_y), jnp.broadcast_to(x, (d_y * args.points, x.shape[0]))).reshape(d_y, args.points)
        lp = lp - jax.scipy.special.logsumexp(lp, axis=1, keepdims=True)
        return jnp.exp(lp) / dd                                                           # normalized over the slice
    out = np.stack([np.asarray(slices(jnp.asarray(x_next[w, k]))) for w, k in pairs])    # (S, d_y, G)
    return out.reshape(-1, args.points)

methods = [("normflowsde", "NormFlowSDE"), ("nf_dpf", "NF-DPF"), ("nsf", "NSF"), ("flowsde", "FlowSDE")]
results = {}
for key, label in methods:
    run = resolve(src["methods"][key]["seeds"][args.seed])
    post = np.load(run / "posterior.npz")
    mean = post["samples_normalized"].mean(axis=1)                                        # (W, T, d_x) estimates of x[k+1]
    implied = ((test["observations"][:mean.shape[0], :mean.shape[1]] - mean[..., obs_idx]) * scale).ravel()
    entry = dict(run=str(run), implied_resid_std=float(implied.std()), implied=implied)
    if key != "normflowsde":
        sl = density_slices(load_measurement(key, run))
        entry.update(slice_mean=sl.mean(0), slice_lo=np.percentile(sl, 10, 0), slice_hi=np.percentile(sl, 90, 0),
                     tv_to_gmm=float(0.5 * np.abs(sl.mean(0) - gmm_pdf(grid)).sum() * dd))
        print(f"{label:12s} learned-density slice: TV distance to GMM {entry['tv_to_gmm']:.3f}; implied residual std {implied.std():.4f}")
    else:
        print(f"{label:12s} implied residual std {implied.std():.4f} (no measurement model)")
    results[key] = entry

FS = args.font_scale
INK, INK2, RULE = "#3a3936", "#52514e", "#c3c2b7"          # text / secondary text / axis rules
C_LEARN, C_IMPL, C_DATA = "#2a78d6", "#eb6834", "#e4e3df"  # learned flow, implied residual, data histogram
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK})
from scipy.stats import gaussian_kde
def kde(sample, factor=0.075, n=60000):
    sub = sample if sample.size <= n else np.random.default_rng(1).choice(sample, n, replace=False)
    return gaussian_kde(sub, bw_method=factor)(grid)

truth = gmm_pdf(grid)
ytop = 1.12 * max([truth.max()] + [e["slice_hi"].max() for e in results.values() if "slice_hi" in e] + [kde(e["implied"]).max() for e in results.values()])
ytop = float(np.ceil(ytop * 10 ** -np.floor(np.log10(ytop)) * 2) / 2 * 10 ** np.floor(np.log10(ytop)))  # round up to a half-decade step
bins = np.linspace(-args.grid, args.grid, 65)
fig, axes = plt.subplots(2, 4, figsize=(17, 6.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 0.85], hspace=0.08, wspace=0.12))
for col, (key, label) in enumerate(methods):
    e = results[key]; implied = kde(e["implied"])
    for row in range(2):
        ax = axes[row, col]
        ax.hist(resid_data, bins=bins, density=True, color=C_DATA, edgecolor="none", label="data residual $y-x$ (GMM sample)", zorder=1)
        ax.plot(grid, truth, color=INK, lw=2.0, label="true GMM density", zorder=4)
        if "slice_mean" in e:
            ax.fill_between(grid, e["slice_lo"], e["slice_hi"], color=C_LEARN, alpha=0.18, lw=0, zorder=2)
            ax.plot(grid, e["slice_mean"], color=C_LEARN, lw=2.2, label="learned measurement flow $q(y\\mid x)$", zorder=5)
        ax.plot(grid, implied, color=C_IMPL, lw=2.0, ls=(0, (4, 2)), label="implied residual $y-\\mathbb{E}[x\\mid y]$", zorder=6)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=10 * FS, length=3, color=RULE)
        ax.set_xlim(-args.grid, args.grid)
        if row == 0:
            ax.set_ylim(0, ytop)
            if not args.no_titles: ax.set_title(label, fontsize=13 * FS, color=INK, pad=8)
            note = f"TV to GMM {e['tv_to_gmm']:.2f}" if "tv_to_gmm" in e else "no noise model"
            lines = [note, f"implied std {e['implied_resid_std']:.3f}"] + ([f"data std {resid_data.std():.3f}"] if col == 0 else [])
            ax.text(0.03, 0.96, "\n".join(lines), transform=ax.transAxes, ha="left", va="top", fontsize=9.5 * FS, color=INK2, linespacing=1.35)
        else:
            ax.set_yscale("log"); ax.set_ylim(ytop * 3e-4, ytop * 1.8)
        if col == 0:
            ax.set_ylabel("density" if row == 0 else "density (log)", fontsize=11 * FS)
        else:
            ax.tick_params(labelleft=False)
fig.supxlabel(f"observation residual ({units})", fontsize=11 * FS, color=INK, y=0.075)
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10.5 * FS, frameon=False, bbox_to_anchor=(0.5, -0.01),
           handlelength=2.4, columnspacing=2.0)
fig.subplots_adjust(left=0.05, right=0.995, top=0.92 if not args.no_titles else 0.98, bottom=0.17)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}_measurement_density.{ext}", dpi=200, bbox_inches="tight")
json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) and vv.ndim == 1 and vv.size < 1000 else vv) for kk, vv in e.items() if kk != "implied"}
           for k, e in results.items()} | dict(grid=grid.tolist(), gmm=truth.tolist(), data_resid_std=float(resid_data.std()), units=units, obs_std=sigma),
          open(f"{args.out}_measurement_density.json", "w"), indent=1)
print("saved", f"{args.out}_measurement_density.{{png,pdf,json}}")
