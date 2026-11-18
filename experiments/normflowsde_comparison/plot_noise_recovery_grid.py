"""Observation-noise recovery, all eight benchmark methods in one 2x4 figure (one dataset).

Top row: the four particle filters (NF-DPF, NSF-PF, FlowSDE-PF, NormFlowSDE-PF). Each panel shows its learned
measurement flow q(y | x) (slice average and 10-90 % band over states, read from the saved per-dataset JSON of
plot_measurement_density_recovery.py; no model is reloaded) and the implied residuals y - E[x | y].
Bottom row: the four direct estimators (NormFlowSDE, latent NSF, latent SDE, signature-kernel SDE), which are
scored only through their implied residuals.

Implied residuals are recomputed from the seed-0 test posteriors, streamed window by window so that the ~1 GB
posterior files are never held in memory. The computation is the one of plot_measurement_density_recovery.py
(observation k measures state k+1; physical rescaling for caches that define the noise in physical units), and
the recomputed standard deviations are checked against the values stored in the JSON for the five methods that
have one.

Dependencies: numpy, scipy, matplotlib only (runs in the local plotting venv).

Usage (from the NormFlowSDE root):
  python -I experiments/normflowsde_comparison/plot_noise_recovery_grid.py --root . --dataset reassemble \
      --out results/analyses/noise_recovery_grid/reassemble --copy-to paper/figures/measurement_density_reassemble_grid.pdf
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, default=Path("."))
p.add_argument("--dataset", default="reassemble")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True, help="output stem")
p.add_argument("--copy-to", type=Path, default=None, help="also copy the PDF here (paper figure)")
p.add_argument("--font-scale", type=float, default=1.0)
p.add_argument("--sources", type=Path, default=None,
               help="(2026-11-16) sources.json of inference_cost.py; if given, the run directory of every method is "
                    "read from it instead of the default results/<method>/... layout")
p.add_argument("--saved", type=Path, default=None,
               help="(2026-11-16) <dataset>_measurement_density.json of plot_measurement_density_recovery.py for the "
                    "runs of --sources (default: results/analyses/normflowsde_pf/noise_recovery/, the 2026-11-01 runs)")
args = p.parse_args()
R = args.root / "results"

PF = [("nf_dpf", "NF-DPF", "nf_dpf/{d}/seed{s}/s2"), ("nsf", "NSF-PF", "nsf/{d}/seed{s}/jitter"),
      ("flowsde", "FlowSDE-PF", None), ("normflowsde_pf", "NormFlowSDE-PF", "normflowsde_pf/{d}/seed{s}/learned_scale")]
DIRECT = [("normflowsde", "NormFlowSDE", "normflowsde/{d}/seed{s}/s2"), ("latent_nsf", "Latent NSF", "latent_nsf/{d}/seed{s}/run"),
          ("latent_sde", "Latent SDE", "latent_sde/{d}/seed{s}/run"), ("sigker_sde", "SigKer SDE", "sigker_sde/{d}/seed{s}/run")]

# Learned measurement-flow slices of the particle filters (incl. NormFlowSDE-PF) from the 2026-11-01 JSON.
saved = json.loads((args.saved or R / f"analyses/normflowsde_pf/noise_recovery/{args.dataset}_measurement_density.json").read_text())
sources = json.loads(args.sources.read_text())[args.dataset]["methods"] if args.sources else None
grid, truth = np.array(saved["grid"]), np.array(saved["gmm"])
dd = grid[1] - grid[0]


def local_run(stored):
    """Map a server path (/home/<user>/NormFlowSDE/results/...) to the local results tree."""
    return R / stored.split("/results/", 1)[1]


def run_dir(key, pattern):
    if sources is not None:
        return args.root / sources[key]["seeds"][str(args.seed)]
    if pattern is None:  # FlowSDE: the selected diffusion level differs per dataset, take the path the JSON used
        return local_run(saved[key]["run"])
    return R / pattern.format(d=args.dataset, s=args.seed)


# Test data and the residual scaling of plot_measurement_density_recovery.py.
cache = json.loads((R / f"datasets/{args.dataset}/cache.json").read_text())
meta = cache["metadata"]
half = float(grid[-1])
bins = np.linspace(-half, half, 65)
test_file = R / f"datasets/{args.dataset}/test.npz"
have_test = test_file.exists()  # the test arrays are not part of the light local copy
if have_test:
    test = np.load(test_file)
    observations, states = test["observations"], test["states"]
    obs_idx = np.array(meta.get("observation_indices") or list(range(observations.shape[-1])))
    physical = "physical_observation_std" in meta
    scale = np.array(meta["observation_std"], dtype=float) if physical else np.ones(len(obs_idx))
    resid_data = ((observations - states[:, 1:][..., obs_idx]) * scale).ravel()
    data_residual = dict(hist=np.histogram(resid_data, bins=bins, density=True)[0].tolist(), bins=bins.tolist(),
                         std=float(resid_data.std()), source="test.npz")


def posterior_mean(path):
    """Mean over draws of samples_normalized (W, M, T, D), read one window at a time from the npz member."""
    with zipfile.ZipFile(path) as z, z.open("samples_normalized.npy") as f:
        version = np.lib.format.read_magic(f)
        read_header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read_header(f)
        assert not fortran and len(shape) == 4, shape
        per_window = int(np.prod(shape[1:]))
        out = np.empty((shape[0], shape[2], shape[3]))
        for w in range(shape[0]):
            block = np.frombuffer(f.read(per_window * dtype.itemsize), dtype=dtype).reshape(shape[1:])
            out[w] = block.mean(axis=0)
    return out


def kde(sample, factor=0.075, n=60000):
    sub = sample if sample.size <= n else np.random.default_rng(1).choice(sample, n, replace=False)
    return gaussian_kde(sub, bw_method=factor)(grid)


cache_file = Path(f"{args.out}_noise_recovery_grid.cache.json")
results = {}
if cache_file.exists():  # skip the ~8 GB posterior read when only the figure changes
    results = {k: {kk: (np.array(vv) if isinstance(vv, list) else vv) for kk, vv in e.items()}
               for k, e in json.loads(cache_file.read_text()).items()}
cached_residual = results.pop("_data_residual", None)
if not have_test:  # fall back to the cached histogram of the data residuals
    assert cached_residual is not None, f"{test_file} is missing and the cache holds no data-residual histogram"
    data_residual = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in cached_residual.items()}
    assert np.allclose(data_residual["bins"], bins)


def write_cache():
    payload = {k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in e.items()} for k, e in results.items()}
    cache_file.write_text(json.dumps(payload | {"_data_residual": data_residual}))


for key, label, pattern in PF + DIRECT:
    if key in results:
        continue
    assert have_test, f"{key} is not cached and {test_file} is missing"
    run = run_dir(key, pattern)
    mean = posterior_mean(run / "posterior.npz")
    implied = ((observations[:mean.shape[0], :mean.shape[1]] - mean[..., obs_idx]) * scale).ravel()
    density = kde(implied)
    entry = dict(label=label, run=str(run.relative_to(args.root)), implied_resid_std=float(implied.std()),
                 implied_tv_to_gmm=float(0.5 * np.abs(density - truth).sum() * dd), implied_density=density)
    if key in saved and "implied_resid_std" in saved[key]:
        ref = saved[key]["implied_resid_std"]
        assert abs(ref - entry["implied_resid_std"]) < 1e-3 * ref, (key, ref, entry["implied_resid_std"])
        entry["implied_resid_std_check"] = ref
    if key in saved and "slice_mean" in saved[key]:
        for k in ("slice_mean", "slice_lo", "slice_hi"):
            entry[k] = np.array(saved[key][k])
        entry["tv_to_gmm"] = saved[key]["tv_to_gmm"]
    results[key] = entry
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_cache()
    print(f"{label:15s} implied std {entry['implied_resid_std']:.4f}  TV(implied, GMM) {entry['implied_tv_to_gmm']:.3f}"
          + (f"  TV(flow, GMM) {entry['tv_to_gmm']:.3f}" if "tv_to_gmm" in entry else ""), flush=True)

# Similarity to the true mixture: overlap coefficient int min(p, q) = 1 - TV(p, q), 1 for identical densities.
for e in results.values():
    e["implied_overlap_with_gmm"] = 1.0 - e["implied_tv_to_gmm"]
    if "tv_to_gmm" in e:
        e["overlap_with_gmm"] = 1.0 - e["tv_to_gmm"]
    print(f"{e['label']:15s} overlap implied {e['implied_overlap_with_gmm']:.3f}"
          + (f"  overlap flow {e['overlap_with_gmm']:.3f}" if "overlap_with_gmm" in e else ""), flush=True)
write_cache()

# ------------------------------------------------------------------------------------------------- figure
FS = args.font_scale
INK, INK2, RULE = "#3a3936", "#52514e", "#c3c2b7"
C_LEARN, C_IMPL, C_DATA = "#2a78d6", "#eb6834", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK})
ytop = 1.12 * max([truth.max()] + [e["implied_density"].max() for e in results.values()]
                  + [e["slice_hi"].max() for e in results.values() if "slice_hi" in e])
ytop = float(np.ceil(ytop * 10 ** -np.floor(np.log10(ytop)) * 2) / 2 * 10 ** np.floor(np.log10(ytop)))

fig, axes = plt.subplots(2, 4, figsize=(10, 6.6), sharex=True, sharey=True, gridspec_kw=dict(hspace=0.30, wspace=0.08))
for row, group in enumerate((PF, DIRECT)):
    for col, (key, label, _) in enumerate(group):
        ax, e = axes[row, col], results[key]
        ax.stairs(data_residual["hist"], bins, fill=True, color=C_DATA, edgecolor="none", label="data residual $y-x$", zorder=1)
        ax.plot(grid, truth, color=INK, lw=2.0, label="true mixture density", zorder=4)
        if "slice_mean" in e:  # learned measurement flow q(y | x): average and 10-90 % band over states
            ax.fill_between(grid, e["slice_lo"], e["slice_hi"], color=C_LEARN, alpha=0.22, lw=0, zorder=3)
            ax.plot(grid, e["slice_mean"], color=C_LEARN, lw=2.0, label="observation model $q(y-x\\mid x)$", zorder=5)
        ax.plot(grid, e["implied_density"], color=C_IMPL, lw=2.0, ls=(0, (4, 2)),
                label="implied residual $y-\\mathbb{E}[x\\mid y]$", zorder=6)
        # overlap of the implied residuals with the true mixture
        ax.text(0.04, 0.96, "overlap", transform=ax.transAxes, ha="left", va="top", fontsize=12.5 * FS, color=INK2)
        ax.text(0.04, 0.84, f"{e['implied_overlap_with_gmm']:.2f}", transform=ax.transAxes, ha="left", va="top",
                fontsize=13.5 * FS, color=C_IMPL, fontweight="bold")
        if "overlap_with_gmm" in e:  # overlap of the learned measurement flow with the true mixture
            ax.text(0.04, 0.72, f"{e['overlap_with_gmm']:.2f}", transform=ax.transAxes, ha="left", va="top",
                    fontsize=13.5 * FS, color=C_LEARN, fontweight="bold")
        ax.set_title(f"({'abcdefgh'[4 * row + col]}) {label}", fontsize=14.5 * FS, color=INK, pad=6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=13 * FS, length=3, color=RULE)
        ax.set_xticks([-0.5, 0.0, 0.5])
        ax.set_xlim(-half, half)
        ax.set_ylim(0, ytop)
    axes[row, 0].set_ylabel(("particle filters" if row == 0 else "direct estimators") + "\ndensity", fontsize=14 * FS)
fig.supxlabel(f"observation residual ({saved['units']})", fontsize=14 * FS, color=INK, y=0.085)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13.5 * FS, frameon=False, bbox_to_anchor=(0.5, -0.07),
           handlelength=2.4, columnspacing=2.0)
fig.subplots_adjust(left=0.10, right=0.995, top=0.95, bottom=0.175)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(f"{args.out}_noise_recovery_grid.{ext}", dpi=200, bbox_inches="tight")
json.dump({k: {kk: (vv.tolist() if isinstance(vv, np.ndarray) else vv) for kk, vv in e.items()} for k, e in results.items()}
          | dict(grid=grid.tolist(), gmm=truth.tolist(), data_resid_std=data_residual["std"], data_residual=data_residual,
                 similarity="overlap coefficient int min(p, q) = 1 - total variation, on the stored grid", units=saved["units"],
                 dataset=args.dataset, seed=args.seed),
          open(f"{args.out}_noise_recovery_grid.json", "w"), indent=1)
if args.copy_to:
    shutil.copyfile(f"{args.out}_noise_recovery_grid.pdf", args.copy_to)
print("saved", f"{args.out}_noise_recovery_grid.{{png,pdf,json}}")
