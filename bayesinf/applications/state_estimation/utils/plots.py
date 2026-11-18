"""Shared trajectory and diagnostic plots for state-estimation experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_state_estimation_plots(
    output_dir,
    method,
    x_true,
    y_obs,
    samples,
    *,
    dt=0.02,
    ci=90.0,
    n_windows=3,
):
    """Save trajectory panels, diagnostics, CSV, and NPZ artifacts.

    Arrays use shapes ``x_true=(M,T,D)``, ``y_obs=(M,T,D_y)``, and
    ``samples=(M,S,T,D)``.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x_true = np.asarray(x_true)
    y_obs = np.asarray(y_obs)
    samples = np.asarray(samples)
    horizon = min(x_true.shape[1], samples.shape[2])
    x_true = x_true[:, :horizon]
    y_obs = y_obs[:, :horizon]
    samples = samples[:, :, :horizon]

    mean = samples.mean(axis=1)
    alpha = (100.0 - ci) / 2.0
    lo = np.percentile(samples, alpha, axis=1)
    hi = np.percentile(samples, 100.0 - alpha, axis=1)
    error = mean - x_true
    rmse_per_dim = np.sqrt(np.mean(error**2, axis=(0, 1)))
    mae_per_dim = np.mean(np.abs(error), axis=(0, 1))
    coverage_per_dim = np.mean((x_true >= lo) & (x_true <= hi), axis=(0, 1))
    spread = samples.std(axis=1)
    rmse = float(np.mean(np.sqrt(np.mean(error**2, axis=(1, 2)))))
    mae = float(np.mean(np.abs(error)))
    coverage = float(np.mean((x_true >= lo) & (x_true <= hi)))

    stem = output_dir / f"{method}_eval"
    np.savez(
        f"{stem}_metrics.npz",
        x_true=x_true,
        y_obs=y_obs,
        samples=samples,
        posterior_mean=mean,
        ci_low=lo,
        ci_high=hi,
        rmse=rmse,
        mae=mae,
        coverage=coverage,
        rmse_per_dim=rmse_per_dim,
        mae_per_dim=mae_per_dim,
        coverage_per_dim=coverage_per_dim,
    )
    with open(f"{stem}_metrics.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("method", "rmse", "mae", f"coverage_{int(ci)}"))
        writer.writerow((method, rmse, mae, coverage))

    time = np.arange(horizon) * dt
    dims = np.unique(np.linspace(0, x_true.shape[-1] - 1, 4, dtype=int))
    windows = min(n_windows, x_true.shape[0])
    fig, axes = plt.subplots(
        windows, len(dims), figsize=(4.0 * len(dims), 2.8 * windows), squeeze=False
    )
    for row in range(windows):
        for col, dim in enumerate(dims):
            ax = axes[row, col]
            ax.fill_between(time, lo[row, :, dim], hi[row, :, dim],
                            color="#4C78A8", alpha=0.22, label=f"{ci:.0f}% CI")
            ax.plot(time, x_true[row, :, dim], color="black", lw=1.6, label="Truth")
            ax.plot(time, mean[row, :, dim], color="#4C78A8", lw=1.5,
                    ls="--", label="Posterior mean")
            if dim < y_obs.shape[-1]:
                ax.scatter(time, y_obs[row, :, dim], color="#E15759", s=7,
                           alpha=0.45, label="Noisy observation")
            ax.set_title(f"Window {row + 1} · joint {dim}")
            ax.set_xlabel("Time [s]")
            ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        f"{method.replace('_', ' ')} trajectories · RMSE={rmse:.4f} · "
        f"{ci:.0f}% coverage={coverage * 100:.1f}%"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(f"{stem}_trajectories.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    idx = np.arange(x_true.shape[-1])
    axes[0].bar(idx, rmse_per_dim, color="#4C78A8")
    axes[0].set(title="RMSE per joint", xlabel="Joint index", ylabel="RMSE")
    axes[1].bar(idx, coverage_per_dim * 100, color="#59A14F")
    axes[1].axhline(ci, color="black", ls="--", lw=1, label=f"{ci:.0f}% target")
    axes[1].set(title="CI coverage per joint", xlabel="Joint index", ylabel="Coverage [%]",
                ylim=(0, 100))
    axes[1].legend(frameon=False)
    abs_error = np.abs(error).reshape(-1)
    flat_spread = spread.reshape(-1)
    take = np.linspace(0, len(abs_error) - 1, min(5000, len(abs_error)), dtype=int)
    axes[2].scatter(flat_spread[take], abs_error[take], s=5, alpha=0.18,
                    color="#F28E2B")
    limit = max(float(flat_spread.max()), float(abs_error.max()), 1e-6)
    axes[2].plot([0, limit], [0, limit], color="black", ls="--", lw=1)
    axes[2].set(title="Posterior spread vs absolute error",
                xlabel="Posterior standard deviation", ylabel="Absolute error")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        f"{method.replace('_', ' ')} diagnostics · MAE={mae:.4f} · "
        f"coverage={coverage * 100:.1f}%"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(f"{stem}_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
