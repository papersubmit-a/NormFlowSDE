#!/usr/bin/env python3
"""Draw the vector estimator schematic; the displayed curves are illustrative.

This figure contains no experimental measurements or performance estimates.
Run independently or through build_artifacts.py to regenerate PDF and SVG.
"""
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/normflowsde-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
INK = '#253A4A'
MUTED = '#617580'
TEAL = '#007F86'
INDIGO = '#645893'
ORANGE = '#C76A32'
LINE = '#CBD6DB'


def draw_model_overview():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9.5,
                         'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'svg.fonttype': 'none', 'mathtext.fontset': 'dejavusans'})
    fig = plt.figure(figsize=(7.5, 4.6), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 7.5), ylim=(0, 4.6))
    ax.axis('off')

    def label(x, y, text, size=9.5, color=INK, weight='normal', ha='left', **kwargs):
        return ax.text(x, y, text, fontsize=size, color=color, weight=weight,
                       ha=ha, va='center', **kwargs)

    def panel(x, y, w, h, face='white', edge=LINE, radius=.08, lw=.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle=f'round,pad=0,rounding_size={radius}',
                     facecolor=face, edgecolor=edge, linewidth=lw))

    def arrow(start, end, color=MUTED, lw=1.2, **kwargs):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                     mutation_scale=9, color=color, linewidth=lw,
                     shrinkA=0, shrinkB=0, **kwargs))

    # Training is a compact, one-time operation, separate from inference.
    panel(.10, 3.98, 7.30, .54, face='#F8F5F0', edge='#E3DDD3')
    label(.24, 4.32, '01  LEARN ONCE', size=9, color=ORANGE, weight='bold')
    label(.24, 4.13, 'Paired reference states + measurements', size=9)
    arrow((3.00, 4.25), (3.42, 4.25), color=ORANGE)
    label(3.58, 4.32, 'Fit the encoder, drift and flow', size=9.5, weight='bold')
    label(3.58, 4.13, 'Conditional likelihood training', size=9, color=MUTED)
    arrow((4.13, 3.98), (4.13, 3.77), color=ORANGE, linestyle=(0, (3, 2)))
    label(4.29, 3.87, 'shared weights', size=8.5, color=ORANGE)

    # The estimator is the dominant, single inference route.
    panel(1.87, .63, 3.76, 3.12, face='#F3F7F9', edge='#AEC2CD', lw=1.0)
    label(2.05, 3.53, 'NormFlowSDE state estimator', size=11, weight='bold')
    label(2.05, 3.30, '02  REUSE FOR EACH NEW SEQUENCE', size=8.6,
          color=MUTED, weight='bold')
    panel(2.08, 2.74, 3.34, .37, face='white', edge='#A9BFC9')
    label(3.75, 2.925, 'Shared observation encoder', size=9.5, weight='bold', ha='center')

    # Observation context conditions both nominal evolution and uncertainty.
    arrow((2.91, 2.74), (2.91, 2.39), color=TEAL)
    arrow((4.56, 2.74), (4.56, 2.39), color=INDIGO)
    label(3.75, 2.57, 'measurement context', size=8.4, color=MUTED, ha='center',
          bbox=dict(facecolor='#F3F7F9', edgecolor='none', pad=1))
    panel(2.08, 1.57, 1.55, .79, face='#E8F3F2', edge='#B0D2CF')
    label(2.855, 2.17, 'Learned drift', size=9.5, color=TEAL, weight='bold', ha='center')
    label(2.855, 1.98, 'Nominal evolution', size=8.5, color=TEAL, ha='center')
    tt = np.linspace(0, 1, 70)
    ax.plot(2.37 + .94*tt, 1.69 + .12*(tt + .32*np.sin(3.8*tt)),
            color=TEAL, lw=1.8, solid_capstyle='round')
    panel(3.87, 1.57, 1.55, .79, face='#F0EDF7', edge='#C9C0DF')
    label(4.645, 2.17, 'Conditional flow', size=9.5, color=INDIGO, weight='bold', ha='center')
    label(4.645, 1.98, 'Flow-transformed noise', size=8.5, color=INDIGO, ha='center')
    # Base-noise transformation glyph conveys the role of the invertible map.
    gx = np.linspace(-2.6, 2.6, 80)
    ax.plot(4.03 + .062*(gx+2.6), 1.64 + .16*np.exp(-gx*gx/2), color=MUTED, lw=1.1)
    arrow((4.43, 1.73), (4.70, 1.73), color=INDIGO, lw=.9)
    density = .95*np.exp(-(gx+.9)**2/.45)+.57*np.exp(-(gx-1.1)**2/.68)
    ax.plot(4.79 + .090*(gx+2.6), 1.64 + .16*density, color=INDIGO, lw=1.4)

    arrow((2.85, 1.57), (2.85, 1.30), color=TEAL)
    arrow((4.64, 1.57), (4.64, 1.30), color=INDIGO)
    panel(2.08, .94, 3.34, .36, face='white', edge='#A9BFC9')
    label(3.75, 1.12, 'Current state  +  drift  +  noise', size=9.0, ha='center')
    # A simple recurrence arrow marks path generation, without a second method.
    arrow((5.30, .94), (2.19, .94), color=MUTED, lw=.85,
          connectionstyle='bar,fraction=-0.027')
    label(3.75, .72, 'Repeat in time; draw multiple state paths', size=8.4,
          color=MUTED, ha='center')

    label(.15, 3.52, 'Measurements', size=10.5, weight='bold')
    label(.15, 3.30, 'Noisy and partial', size=9, color=MUTED)
    label(5.98, 3.52, 'State estimates', size=10.5, weight='bold')
    label(5.98, 3.30, 'With uncertainty', size=9, color=MUTED)

    def trace_axes(x, y, w, h):
        sub = fig.add_axes([x/7.5, y/4.6, w/7.5, h/4.6])
        sub.set(xlim=(0, 1), ylim=(-1.45, 1.45))
        sub.set_xticks([])
        sub.set_yticks([])
        for spine in ('left', 'top', 'right'):
            sub.spines[spine].set_visible(False)
        sub.spines['bottom'].set_color(LINE)
        sub.spines['bottom'].set_linewidth(.7)
        sub.patch.set_alpha(0)
        return sub

    time = np.linspace(0, 1, 180)
    mean1 = .80*np.sin(5.7*time-.3) + .16*np.cos(11*time)
    mean2 = .68*np.cos(5.6*time+.4) - .12*np.sin(13*time)
    sensor2 = .55*np.sin(7*time+.6) + .25*np.cos(3*time-.4)
    rng = np.random.default_rng(20261007)
    for idx, (mean, y) in enumerate([(mean1, 2.35), (sensor2, 1.47)]):
        inp = trace_axes(.16, y, 1.32, .54)
        points = np.arange(5, len(time)-3, 11)
        if idx == 1:
            points = points[(points < 65) | (points > 120)]
            inp.axvspan(.36, .67, color='#EEF1F3', linewidth=0)
        inp.scatter(time[points], mean[points]+rng.normal(0, .21, len(points)),
                    s=9, color=ORANGE, edgecolors='white', linewidths=.35, zorder=3)
        label(.16, y+.69, f'Sensor channel {idx+1}', size=8.6, color=MUTED)
    label(1.48, 1.31, 'time', size=8, color=MUTED, ha='right')
    panel(.15, .65, 1.33, .43, face='#FAFBFC', edge='#DCE3E7', radius=.04)
    label(.815, .865, 'Controls +\ninitial-state info.', size=8.8, ha='center')
    arrow((1.50, 2.925), (2.07, 2.925), color=ORANGE, lw=1.3)
    arrow((1.49, .865), (1.87, .865), color=MUTED, lw=1.0)

    # Illustrative posterior paths are not empirical evidence or calibrated intervals.
    for idx, (mean, y, color) in enumerate([(mean1, 2.35, TEAL), (mean2, 1.47, INDIGO)]):
        out = trace_axes(5.99, y, 1.34, .54)
        spread = (.18 if idx == 0 else .36) + (.05 if idx == 0 else .12)*np.sin(4*time)**2
        out.fill_between(time, mean-spread, mean+spread, color=color, alpha=.17, lw=0)
        for phase in (.6, 2.4, 4.3):
            trajectory = mean + spread*.65*np.sin(5*time+phase)
            out.plot(time, trajectory, color=color, alpha=.35, lw=.6)
        out.plot(time, mean, color=color, lw=1.7)
        label(5.99, y+.69, 'Observed state' if idx == 0 else 'Unobserved state',
              size=8.6, color=color)
    label(7.33, 1.31, 'time', size=8, color=MUTED, ha='right')
    ax.plot([6.02, 6.21], [1.05, 1.05], color=TEAL, lw=1.7)
    label(6.28, 1.05, 'Mean estimate', size=8.4)
    ax.fill_between([6.02, 6.21], [.81, .81], [.92, .92], color=TEAL, alpha=.17, lw=0)
    label(6.28, .865, 'Uncertainty band', size=8.4)
    arrow((5.43, 1.12), (5.82, 1.12), color=TEAL, lw=1.4)
    arrow((5.82, 1.12), (5.82, 2.67), color=TEAL, lw=1.1,
          connectionstyle='angle,angleA=0,angleB=90,rad=0')
    arrow((5.82, 2.67), (5.97, 2.67), color=TEAL, lw=1.1)
    arrow((5.82, 1.75), (5.97, 1.75), color=INDIGO, lw=1.1)

    # User-facing contribution, not an alternative filtering pipeline.
    ax.plot([.15, 7.35], [.42, .42], color=LINE, lw=.8)
    label(3.75, .21, 'Learn from data   •   No physical equations required   •   No per-sequence refitting',
          size=9.1, color=INK, ha='center')
    for suffix in ('pdf', 'svg'):
        fig.savefig(PAPER / f'figures/model_overview.{suffix}', facecolor='white',
                    metadata={'Creator': 'NormFlowSDE vector schematic'})
    plt.close(fig)


if __name__ == '__main__':
    draw_model_overview()
