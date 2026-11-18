#!/usr/bin/env python3
"""Draw a schematic of the weak-convergence proof, not simulated sample paths.

Regenerate the publication vector PDF and an optional raster preview with:
    python scripts/plot_convergence_proof.py --preview /tmp/convergence_proof.png
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/normflowsde-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
INK = "#253A4A"
MUTED = "#617580"
TEAL = "#007F86"
INDIGO = "#645893"
ORANGE = "#C76A32"
LINE = "#CBD6DB"


def draw_convergence_proof(preview=None):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    width, height = 7.4, 4.0
    fig = plt.figure(figsize=(width, height), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    canvas.axis("off")

    def label(x, y, text, size=9.5, color=INK, weight="normal", ha="left"):
        return canvas.text(
            x, y, text, fontsize=size, color=color, weight=weight,
            ha=ha, va="center", linespacing=1.3,
        )

    def arrow(start, end, color=MUTED, **kwargs):
        canvas.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=9,
                linewidth=1.1, color=color, shrinkA=1, shrinkB=1, **kwargs,
            )
        )

    def box(x, y, w, h, text, color=INK, face="#F5F8FA"):
        canvas.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0,rounding_size=.065",
                facecolor=face, edgecolor=LINE, linewidth=.9,
            )
        )
        label(x + w / 2, y + h / 2, text, color=color, ha="center")

    label(.16, 3.84, "Three path behaviors controlled by the proof", size=10.2,
          weight="bold")
    positions = [.30, 2.78, 5.26]
    titles = ["2. Escape to large states", "3. A persistent jump", "4. Rapid oscillations"]
    colors = [ORANGE, INDIGO, TEAL]
    axes = []
    for left, title, color in zip(positions, titles, colors):
        label(left - .14, 3.52, title, size=9.6, color=color, weight="bold")
        ax = fig.add_axes([left / width, 2.22 / height, 1.98 / width, 1.10 / height])
        ax.set(xlim=(0, 1), ylim=(-1.25, 1.80))
        ax.set_xticks([0, 1], labels=["0", "$T$"])
        ax.set_yticks([])
        ax.tick_params(axis="x", length=0, pad=3, labelsize=8.5, colors=MUTED)
        for spine in ("left", "right", "top"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(LINE)
        ax.spines["bottom"].set_linewidth(.8)
        axes.append(ax)

    # A path that leaves every fixed region is excluded by Step 2.
    time = np.linspace(0, 1, 800)
    ax = axes[0]
    ax.axhspan(-.8, .8, facecolor="#F1F4F6", edgecolor="none")
    for level, name in [(.8, "$R$"), (-.8, "$-R$")]:
        ax.axhline(level, color=MUTED, linestyle=(0, (3, 3)), linewidth=.8)
        ax.text(-.035, level, name, ha="right", va="center", color=MUTED, fontsize=8.5)
    escaping = .10 * np.sin(12 * time) + 2.1 * time**3 - .30
    ax.plot(time, escaping, color=ORANGE, linewidth=1.6)
    ax.plot([0], [escaping[0]], "o", markersize=3, color=ORANGE)
    label(.16, 1.88, "Control the largest excursion.", size=9.0)

    # A jump may persist even if all paths remain in a bounded region.
    ax = axes[1]
    x = np.array([0, .13, .27, .42, .50, .60, .72, .85, 1])
    y = np.array([-.35, -.25, -.31, -.20, 1.02, .96, 1.08, .97, 1.04])
    ax.step(x, y, where="post", color=INDIGO, linewidth=1.6)
    ax.annotate("", xy=(.55, 1.02), xytext=(.55, -.20),
                arrowprops=dict(arrowstyle="<->", color=INDIGO, lw=.9, shrinkA=0, shrinkB=0))
    ax.text(.59, .40, "$a$", color=INDIGO, fontsize=10, va="center")
    label(2.64, 1.88, "Make the largest jump vanish.", size=9.0)

    # Bounded continuous paths can oscillate arbitrarily fast without jumps.
    ax = axes[2]
    rapid = .65 * np.sin(2 * np.pi * (1.2 * time + 7 * time**3)) + .10
    ax.axvspan(.81, .89, color=TEAL, alpha=.10, linewidth=0)
    ax.plot(time, rapid, color=TEAL, linewidth=1.45)
    ax.annotate("", xy=(.89, -.94), xytext=(.81, -.94),
                arrowprops=dict(arrowstyle="|-|", color=TEAL, lw=.8, shrinkA=0, shrinkB=0))
    ax.text(.85, -1.12, "$\\delta$", color=TEAL, fontsize=9, ha="center", va="center")
    label(5.12, 1.88, "Control changes in short intervals.", size=9.0)

    # The two independent parts of the proof join at identification of a limit.
    canvas.plot([.16, 7.24], [1.66, 1.66], color=LINE, linewidth=.8)
    box(.16, .97, 2.10, .53, "1. Local expected change", color=TEAL, face="#EDF6F5")
    box(.16, .15, 2.10, .53, "2–4. Control whole paths", color=INDIGO, face="#F3F0F8")
    box(2.65, .15, 2.30, .53, "Continuous\nsubsequential limits")
    box(2.65, .97, 2.30, .53, "5. Identify the limiting SDE\nUnique law from $z_0$")
    box(5.35, .97, 1.89, .53, "Full sequence\nconverges in law", color=TEAL, face="#EDF6F5")
    arrow((2.26, 1.235), (2.65, 1.235), color=TEAL)
    arrow((2.26, .415), (2.65, .415), color=INDIGO)
    arrow((3.80, .68), (3.80, .97), color=INDIGO)
    arrow((4.95, 1.235), (5.35, 1.235), color=TEAL)

    output = PAPER / "figures/convergence_proof.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white", metadata={
        "Title": "Schematic of the conditional weak-convergence proof",
        "Subject": "Illustrative paths and proof structure; not simulation results",
        "Creator": "NormFlowSDE proof schematic",
    })
    if preview:
        fig.savefig(preview, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview", type=Path, help="Optional PNG preview output")
    args = parser.parse_args()
    draw_convergence_proof(args.preview)
