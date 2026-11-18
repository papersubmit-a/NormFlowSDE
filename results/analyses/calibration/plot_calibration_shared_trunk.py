"""Reliability curves for the shared-trunk campaign (2026-11-16): the paper's figures/calibration.pdf.

Same figure and the same coverage() as plot_calibration_all.py, but every learned curve is computed here from the
saved test posteriors of the runs in the benchmark tables:
  runs       results/shared_trunk_20261114/BENCHMARK_SUMMARY.sources.json (the run directory of every table cell)
  ablation   results/shared_trunk_20261114/normflowsde_pf/pendulum/seed<k>/fixed_scale (drawn if all three exist)
  references the model-based filters, unchanged, from calibration.json
Output: calibration_shared_trunk.{json,pdf,png}; --copy-to also copies the PDF (paper/figures/calibration.pdf).

  python results/analyses/calibration/plot_calibration_shared_trunk.py --copy-to paper/figures/calibration.pdf
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent
P = OUT.parents[1]
ROOT = P.parent
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
DS = [("pendulum", "Pendulum"), ("lorenz96", "Lorenz–96"), ("aloha", "ALOHA"), ("reassemble", "REASSEMBLE")]
DASH = (0, (5, 2))
METHODS = [  # key, label, color, linestyle, marker
    ("normflowsde", "NormFlowSDE", "#2a78d6", "-", "o"),
    ("latent_nsf", "Latent NSF", "#17a6b8", "-", "o"),
    ("latent_sde", "Latent SDE", "#8c5a3c", "-", "o"),
    ("sigker_sde", "SigKer SDE", "#d6609f", "-", "o"),
    ("nf_dpf", "NF-DPF", "#eb6834", DASH, "s"),
    ("nsf", "NSF-PF", "#1baf7a", DASH, "s"),
    ("flowsde", "FlowSDE-PF", "#eda100", DASH, "s"),
    ("normflowsde_pf", "NormFlowSDE-PF", "#7b3fb8", DASH, "s"),
]
ABL = ("normflowsde_pf_fixed_scale", "NormFlowSDE-PF, fixed scale", "#b58cd9", (0, (1.5, 1.5)), "^")
REFS = [("pf", "Bootstrap PF, true model", (0, (4, 2))), ("ekf", "EKF, true model", (0, (1, 1.5)))]


def coverage(path):
    """Identical to plot_calibration.coverage: central-interval coverage at every level (interpolated quantiles)."""
    z = np.load(path); x = z["samples_normalized"]; t = z["truth_normalized"][:, :x.shape[2]]
    M = x.shape[1]; hits = np.zeros(len(LEVELS)); count = 0
    pos = np.array([[(1 - q) / 2 * (M - 1), (1 + q) / 2 * (M - 1)] for q in LEVELS])
    i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, M - 1); frac = pos - i0
    for w0 in range(0, x.shape[0], 8):
        xs = np.sort(x[w0:w0 + 8].astype(np.float32), axis=1); ts = t[w0:w0 + 8]
        for li in range(len(LEVELS)):
            lo = xs[:, i0[li, 0]] * (1 - frac[li, 0]) + xs[:, i1[li, 0]] * frac[li, 0]
            hi = xs[:, i0[li, 1]] * (1 - frac[li, 1]) + xs[:, i1[li, 1]] * frac[li, 1]
            hits[li] += np.sum((ts >= lo) & (ts <= hi))
        count += ts.size
    return (hits / count).tolist()


p = argparse.ArgumentParser()
p.add_argument("--copy-to", type=Path, default=None)
args = p.parse_args()

cells = json.loads((P / "shared_trunk_20261114/BENCHMARK_SUMMARY.sources.json").read_text())["cells"]
runs = {f"{key.split('/')[1]}/{key.split('/')[0]}": cell["runs"] for key, cell in cells.items()}   # <dataset>/<method>
store = OUT / "calibration_shared_trunk.json"
saved = json.loads(store.read_text()) if store.exists() else {}
curves, sources = saved.get("curves", {}), saved.get("runs", {})
ablation = [f"results/shared_trunk_20261114/normflowsde_pf/pendulum/seed{s}/fixed_scale" for s in range(3)]
# drawn from the saved posteriors, or from the cached curves where the posteriors are absent (local replot)
if all((ROOT / run / "posterior.npz").exists() for run in ablation) or sources.get("pendulum/" + ABL[0]) == ablation:
    runs["pendulum/" + ABL[0]] = ablation
else:
    print("fixed-scale ablation runs not complete: curve omitted", flush=True)

for key, paths in runs.items():
    if sources.get(key) != paths:      # recompute a cell whose runs changed
        curves[key] = {str(seed): coverage(ROOT / run / "posterior.npz") for seed, run in enumerate(paths)}
        sources[key] = paths
        store.write_text(json.dumps(dict(levels=LEVELS.tolist(), curves=curves, runs=sources), indent=1))
    print(key, np.round(np.mean(list(curves[key].values()), 0)[[4, 8]], 3).tolist(), flush=True)
reference = json.loads((OUT / "calibration.json").read_text())["curves"]
for ds, _ in DS:
    for ref, _, _ in REFS:
        if f"{ds}/{ref}" in reference:
            curves[f"{ds}/{ref}"] = reference[f"{ds}/{ref}"]
curves = {key: value for key, value in curves.items() if key in runs or key.split("/")[1] in ("pf", "ekf")}
store.write_text(json.dumps(dict(levels=LEVELS.tolist(), curves=curves, runs={k: v for k, v in sources.items() if k in runs}), indent=1))

INK, INK2, RULE = "#3a3936", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.labelcolor": INK, "text.color": INK})
ROWS = [("Direct estimators", METHODS[:4]), ("Particle filters", METHODS[4:] + [ABL])]
# Drawn at the paper's text width (6.5 in), so the font sizes below are the sizes on the page.
FS = 8
fig, axes = plt.subplots(2, 4, figsize=(6.5, 5.0), sharex=True, sharey=True)
fig.subplots_adjust(left=0.105, right=0.99, top=0.955, bottom=0.185, wspace=0.10, hspace=0.42)
for row, (rowname, methods) in enumerate(ROWS):
    for col, (ds, name) in enumerate(DS):
        ax = axes[row, col]
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.8, zorder=1)
        for m, label, color, ls, mk in methods:
            if f"{ds}/{m}" not in curves: continue
            c = np.array(list(curves[f"{ds}/{m}"].values())); mean = c.mean(0)
            ax.fill_between(LEVELS, c.min(0), c.max(0), color=color, alpha=0.15, lw=0, zorder=2)
            ax.plot(LEVELS, mean, color=color, lw=1.7 if m == "normflowsde" else 1.1, ls=ls, marker=mk, ms=2.2,
                    label=label, zorder=6 if m == "normflowsde" else 4)
        for r, label, ls in REFS:
            if f"{ds}/{r}" in curves:
                ax.plot(LEVELS, np.mean(list(curves[f"{ds}/{r}"].values()), 0), color=INK2, lw=0.9, ls=ls, label=label, zorder=3)
        ax.set(xlim=(0.05, 1.0), ylim=(0.0, 1.0), xticks=[0.2, 0.4, 0.6, 0.8, 0.95], yticks=[0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0.2", "0.4", "0.6", "0.8", ".95"])
        if row == 0: ax.set_title(name, fontsize=FS + 1, pad=5)
        if row == 1: ax.set_xlabel("nominal level", fontsize=FS)
        if col == 0: ax.set_ylabel(f"{rowname}\nempirical coverage", fontsize=FS)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=FS, length=2.5, color=RULE)
    # one legend per row, under its panels (row 1: under the tick labels and the axis label)
    handles = {}
    for ax in axes[row]:
        handles.update({label: handle for handle, label in zip(*ax.get_legend_handles_labels())})
    y = axes[row, 0].get_position().y0 - (0.02 if row == 0 else 0.085)
    fig.legend(list(handles.values()), list(handles), loc="upper center", bbox_to_anchor=(0.545, y), ncol=4, fontsize=FS,
               frameon=False, handlelength=2.4, columnspacing=1.1, labelspacing=0.3, borderaxespad=0)
for ext in ("png", "pdf"): fig.savefig(OUT / f"calibration_shared_trunk.{ext}", dpi=300)
if args.copy_to:
    shutil.copyfile(OUT / "calibration_shared_trunk.pdf", args.copy_to)
print("saved", OUT / "calibration_shared_trunk.{png,pdf,json}")
