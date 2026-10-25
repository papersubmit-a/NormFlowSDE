"""Test RMSE, 90% coverage and latency against the number of samples / particles M per dataset and method (rows.jsonl)."""
import json, numpy as np, matplotlib, matplotlib.ticker
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pathlib import Path
OUT = Path(__file__).resolve().parent
rows = [json.loads(l) for l in open(OUT / "rows.jsonl")]
DS = [("pendulum", "Pendulum", 128), ("lorenz96", "Lorenz–96", 200), ("aloha", "ALOHA", 200), ("reassemble", "REASSEMBLE", 500)]
METHODS = [("normflowsde", "NormFlowSDE", "#2a78d6"), ("nf_dpf", "NF-DPF", "#eb6834"), ("nsf", "NSF", "#1baf7a"), ("flowsde", "FlowSDE", "#eda100")]
INK, INK2, RULE = "#3a3936", "#52514e", "#c3c2b7"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": RULE, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK})
fig, axes = plt.subplots(3, 4, figsize=(15, 8.2), sharex="col")
PANELS = [("rmse", "test RMSE"), ("coverage_90", "90% coverage"), ("latency", "latency (ms per window)")]
for col, (ds, name, M) in enumerate(DS):
    ns = sorted({r["n"] for r in rows if r["dataset"] == ds})
    for row_i, (key, ylabel) in enumerate(PANELS):
        ax = axes[row_i, col]
        for m, label, color in METHODS:
            if key == "latency":
                y = [np.median([r["latency"]["ms_per_window_median"] for r in rows if r["dataset"] == ds and r["method"] == m and r["n"] == n and r.get("latency")]) for n in ns]
                ax.plot(ns, y, color=color, lw=2.0, marker="o", ms=4, label=label)
            else:
                vals = np.array([[r["accuracy"][key] for r in rows if r["dataset"] == ds and r["method"] == m and r["n"] == n] for n in ns])
                ax.fill_between(ns, vals.min(1), vals.max(1), color=color, alpha=0.18, lw=0)
                ax.plot(ns, vals.mean(1), color=color, lw=2.0, marker="o", ms=4, label=label)
        ax.axvline(M, color=RULE, lw=1.2, ls=(0, (3, 2)))
        if key == "coverage_90": ax.axhline(0.9, color=RULE, lw=1.0)
        ax.set_xscale("log", base=2); ax.set_xticks(ns); ax.set_xticklabels([str(n) for n in ns], fontsize=9)
        if key == "latency":
            ax.set_yscale("log"); ax.set_xlabel("samples / particles $M$", fontsize=11)
            ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.set_yticks([t for t in (5, 10, 20, 40, 80) if ax.get_ylim()[0] <= t <= ax.get_ylim()[1]])
        if row_i == 0: ax.set_title(name, fontsize=13, pad=8)
        if col == 0: ax.set_ylabel(ylabel, fontsize=11)
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=10, length=3, color=RULE)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02), handlelength=2.6, columnspacing=2.0)
fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.11, hspace=0.18, wspace=0.22)
for ext in ("png", "pdf"): fig.savefig(OUT / f"sample_efficiency.{ext}", dpi=200, bbox_inches="tight")
print("saved", OUT / "sample_efficiency.{png,pdf}")
