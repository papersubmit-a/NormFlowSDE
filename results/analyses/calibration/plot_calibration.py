"""Reliability curves: empirical coverage of central marginal intervals vs nominal level, pooled over test windows,
steps and coordinates, from the saved posterior samples of the final runs (3 seeds) and the reference filters."""
import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
P = Path(__file__).resolve().parents[2]  # <archive>/results; OUT = Path(__file__).resolve().parent
LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
DS = [("pendulum", "Pendulum"), ("lorenz96", "Lorenz–96"), ("aloha", "ALOHA"), ("reassemble", "REASSEMBLE")]
METHODS = [("normflowsde", "NormFlowSDE", "#2a78d6"), ("nf_dpf", "NF-DPF", "#eb6834"), ("nsf", "NSF", "#1baf7a"), ("flowsde", "FlowSDE", "#eda100")]
REFS = [("pf", "Bootstrap PF, true model", (0, (4, 2))), ("ekf", "EKF, true model", (0, (1, 1.5)))]
STAGE = {"normflowsde": "s2", "nf_dpf": "s2", "nsf": "jitter", "flowsde": "D*"}   # the selected run, not its stage-1 parent
def runs(ds, m):
    if m in STAGE:
        return {s: sorted(glob.glob(f"{P}/{m}/{ds}/seed{s}/{STAGE[m]}/posterior.npz"))[0] for s in range(3)}
    return {s: f"{P}/filters/{ds}/{m}/s0/seed{s}/posterior.npz" for s in range(3)}
def coverage(path):
    """Central-interval coverage at every level from one sort per window block (linear-interpolated quantiles)."""
    z = np.load(path); x = z["samples_normalized"]; t = z["truth_normalized"][:, :x.shape[2]]
    M = x.shape[1]; hits = np.zeros(len(LEVELS)); count = 0
    pos = np.array([[(1 - q) / 2 * (M - 1), (1 + q) / 2 * (M - 1)] for q in LEVELS])   # fractional order statistics
    i0 = np.floor(pos).astype(int); i1 = np.minimum(i0 + 1, M - 1); frac = pos - i0
    for w0 in range(0, x.shape[0], 8):
        xs = np.sort(x[w0:w0 + 8].astype(np.float32), axis=1); ts = t[w0:w0 + 8]
        for li in range(len(LEVELS)):
            lo = xs[:, i0[li, 0]] * (1 - frac[li, 0]) + xs[:, i1[li, 0]] * frac[li, 0]
            hi = xs[:, i0[li, 1]] * (1 - frac[li, 1]) + xs[:, i1[li, 1]] * frac[li, 1]
            hits[li] += np.sum((ts >= lo) & (ts <= hi))
        count += ts.size
    return (hits / count).tolist()
curves = {}
for ds, _ in DS:
    for m, *_ in METHODS + [(r,) for r, *_ in REFS]:
        if m in ("pf", "ekf") and ds not in ("pendulum", "lorenz96"): continue
        curves[f"{ds}/{m}"] = {str(s): coverage(p) for s, p in runs(ds, m).items()}
        print(ds, m, np.round(np.mean([v for v in curves[f"{ds}/{m}"].values()], 0), 3))
json.dump(dict(levels=LEVELS.tolist(), curves=curves), open(OUT / "calibration.json", "w"), indent=1)
INK, INK2, RULE = "#3a3936", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK})
fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), sharey=True)
for ax, (ds, name) in zip(axes, DS):
    ax.plot([0, 1], [0, 1], color=RULE, lw=1.2, zorder=1)
    for m, label, color in METHODS:
        c = np.array(list(curves[f"{ds}/{m}"].values())); mean = c.mean(0)
        ax.fill_between(LEVELS, c.min(0), c.max(0), color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(LEVELS, mean, color=color, lw=2.0, marker="o", ms=4, label=label, zorder=4)
    for r, label, ls in REFS:
        if f"{ds}/{r}" in curves:
            ax.plot(LEVELS, np.mean(list(curves[f"{ds}/{r}"].values()), 0), color=INK2, lw=1.6, ls=ls, label=label, zorder=3)
    ax.set(xlim=(0.05, 1.0), ylim=(0.05, 1.0), title=name, xticks=[0.2, 0.4, 0.6, 0.8, 0.95]); ax.set_title(name, fontsize=13, pad=8)
    ax.set_xlabel("nominal level", fontsize=11)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=10, length=3, color=RULE)
axes[0].set_ylabel("empirical coverage", fontsize=11)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.04), handlelength=2.6, columnspacing=1.8)
fig.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.26, wspace=0.12)
for ext in ("png", "pdf"): fig.savefig(OUT / f"calibration.{ext}", dpi=200, bbox_inches="tight")
print("saved", OUT / "calibration.{png,pdf,json}")
