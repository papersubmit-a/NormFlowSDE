#!/usr/bin/env python3
"""Draw two adjacent, independently initialized multiple-shooting segments.

All paths are illustrative, not simulation results. L_S counts state points,
so adjacent full segments overlap at their common global endpoint time and
have local indices k=0,...,L_S-1. A shared global-index axis shows the
corresponding sequence positions. Regenerate the PDF, SVG, and PNG with
``python scripts/plot_multiple_shooting.py`` from the paper folder.
"""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/normflowsde-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.transforms import Bbox
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
INK = "#253A4A"
MUTED = "#617580"
TEAL = "#007F86"
INDIGO = "#645893"
LINE = "#CBD6DB"
REFERENCE = "#9AA9AF"


def draw_multiple_shooting():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.fontset": "dejavusans",
    })
    width, height = 7.4, 3.0
    fig = plt.figure(figsize=(width, height), facecolor="white")
    canvas = fig.add_axes([0, 0, 1, 1], xlim=(0, width), ylim=(0, height))
    canvas.axis("off")

    left, right = .36, 7.15
    bottom, plot_height = 1.02, 1.60
    ax = fig.add_axes([left / width, bottom / height,
                       (right - left) / width, plot_height / height])
    ax.set(xlim=(-.3, 8.35), ylim=(-.02, 1.72))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # A shared reference path shows that the predicted endpoint and the next
    # supplied start can differ even though they have the same global time.
    reference = np.array([.27, .47, .42, .70, .62, .45, .65, .87, .75])
    paths = [
        (0, np.array([.27, .55, .83, 1.00, 1.28]), TEAL),
        (4, np.array([.62, .44, .22, .46, .83]), INDIGO),
    ]
    for start, states, color in paths:
        ax.axvspan(start, start + len(states) - 1, color=color, alpha=.045, lw=0)
    ax.axvline(4, color=LINE, linestyle=(0, (2, 3)), lw=.8, zorder=0)
    ax.plot(np.arange(len(reference)), reference, color=REFERENCE,
            linestyle=(0, (3, 2)), marker="o", markersize=3.1,
            lw=1.25, zorder=2)
    for start, states, color in paths:
        time = np.arange(start, start + len(states))
        ax.plot(time, states, color=color, linewidth=1.9, marker="o",
                markersize=3.6, zorder=4)
        ax.plot(time[0], states[0], marker="o", markersize=7.4, mfc="white",
                mec=color, mew=1.7, linestyle="none", zorder=6)
        ax.plot(time[-1], states[-1], marker="s", markersize=5.6, mfc=color,
                mec="white", mew=.7, linestyle="none", zorder=6)

    # The boundary gap is vertical because it compares values at the same time.
    ax.annotate("", xy=(4, 1.245), xytext=(4, .655),
                arrowprops={"arrowstyle": "<->", "color": MUTED, "lw": 1.0,
                            "linestyle": (0, (2, 2)), "shrinkA": 0,
                            "shrinkB": 0}, zorder=5)
    ax.text(4.18, .97, "boundary gap", fontsize=8.7, color=MUTED,
            va="center", ha="left")
    ax.text(.12, .16, r"$z_0^{(s)}$", color=TEAL, fontsize=10, va="top")
    ax.text(3.91, 1.41, r"$z_{L_S-1}^{(s)}$", color=TEAL, fontsize=10,
            ha="right")
    ax.text(4.19, .52, r"$z_0^{(s+1)}$", color=INDIGO, fontsize=10, va="top")
    ax.text(8.04, 1.00, r"$z_{L_S-1}^{(s+1)}$", color=INDIGO, fontsize=10,
            ha="right")
    for center, segment, color in [(2, "s", TEAL), (6, "s+1", INDIGO)]:
        ax.text(center, 1.60, rf"Segment ${segment}$", color=color,
                fontsize=10, weight="bold", ha="center", va="center")

    handles = [
        Line2D([0], [0], color=REFERENCE, linestyle=(0, (3, 2)), marker="o",
               markersize=3, lw=1.2, label=r"Reference $x$"),
        Line2D([0], [0], color=TEAL, lw=1.8, label="Segment path"),
        Line2D([0], [0], color=MUTED, marker="o", mfc="white", mew=1.5,
               markersize=6, linestyle="none", label="Supplied start"),
        Line2D([0], [0], color=MUTED, marker="s", markersize=5,
               linestyle="none", label="Predicted end"),
    ]
    fig.legend(handles=handles, ncol=4, frameon=False, loc="center",
               bbox_to_anchor=(.5, 2.80 / height), fontsize=8.7,
               handlelength=1.6, handletextpad=.5, columnspacing=1.4,
               labelcolor=INK)

    # Local indices reset at the boundary; the global index remains shared.
    def page_x(time):
        return left + (time + .3) / 8.65 * (right - left)

    for start, states, color in paths:
        a, b = page_x(start), page_x(start + len(states) - 1)
        canvas.add_patch(FancyArrowPatch((a + .025, .86), (b - .025, .86),
                         arrowstyle="|-|", mutation_scale=4, linewidth=1.4,
                         color=color, shrinkA=0, shrinkB=0))
        # Put the label in the bracket line to avoid a separate text row.
        canvas.text((a + b) / 2, .86, r"Local $k=0,\ldots,L_S-1$",
                    fontsize=10, color=color, ha="center", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none",
                          "boxstyle": "square,pad=0.15"})

    global_axis_y = .54
    canvas.add_patch(FancyArrowPatch(
        (page_x(0), global_axis_y), (page_x(8), global_axis_y),
        arrowstyle="-", linewidth=1.0, color=MUTED, shrinkA=0, shrinkB=0))
    for position, label, alignment in [
        (0, r"Global $(s-1)(L_S-1)$", "left"),
        (4, r"$s(L_S-1)$", "center"),
        (8, r"$(s+1)(L_S-1)$", "right"),
    ]:
        x = page_x(position)
        canvas.add_patch(FancyArrowPatch(
            (x, global_axis_y - .045), (x, global_axis_y + .045),
            arrowstyle="-", linewidth=1.0, color=MUTED, shrinkA=0, shrinkB=0))
        canvas.text(x, .32, label, fontsize=10, color=INK,
                    ha=alignment, va="center")
    output = PAPER / "figures/multiple_shooting"
    output.parent.mkdir(parents=True, exist_ok=True)
    # Crop to the drawn content, excluding the invisible full-page axes.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    artists = (list(ax.lines) + list(ax.patches) + list(ax.texts)
               + list(canvas.patches) + list(canvas.texts) + fig.legends)
    bounds = [artist.get_tightbbox(renderer) for artist in artists]
    crop = Bbox.union([bound for bound in bounds if bound is not None])
    crop = crop.transformed(fig.dpi_scale_trans.inverted())
    # Artist bounds omit marker-edge strokes; allow one point to keep them whole.
    crop = crop.padded(1 / 72)
    export = {"facecolor": "white", "bbox_inches": crop, "pad_inches": 0}
    fig.savefig(output.with_suffix(".pdf"), **export, metadata={
        "Title": "Two adjacent multiple-shooting segments",
        "Subject": "Schematic paths with local and global time indices and a same-time boundary gap",
        "Creator": "NormFlowSDE multiple-shooting schematic",
    })
    fig.savefig(output.with_suffix(".svg"), **export, metadata={
        "Title": "Two adjacent multiple-shooting segments",
        "Description": "Illustrative paths, not simulation results",
    })
    fig.savefig(output.with_suffix(".png"), dpi=220, **export)
    plt.close(fig)


if __name__ == "__main__":
    draw_multiple_shooting()
