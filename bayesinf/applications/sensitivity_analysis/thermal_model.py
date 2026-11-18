"""Sobol sensitivity analysis for the thermal cabin calibration FMU data."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PARAMETER_NAMES = (
    "CF_heatUp", "CF_coolDown", "CF_zetaFBT", "CF_zetaRBT", "CF_zetaTFR/BFR",
    "CF_zetaFLR/RLR", "CF_airExchangeFBT", "CF_airExchangeRBT",
    "CF_airExchangeTFR/BFR", "CF_airExchangeFLR/RLR", "CF_alpha_external",
    "CF_alpha_internal", "CF_hull_capacity", "CF_hull_insulation", "ratioBottomTop",
    "ratioFrontRear", "A_internal_mass", "CF_alpha_mass", "CF_v_air_mass",
)

SENSITIVITY_STAGES = (
    tuple(range(19)),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 17, 18),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 18),
    (0, 1, 2, 3, 4, 5, 9),
    (5, 9),
)


@dataclass(frozen=True)
class SobolResult:
    first_order: np.ndarray  # sensor, time, parameter
    total_order: np.ndarray
    mean_first_order: np.ndarray
    mean_total_order: np.ndarray
    ranking: tuple[int, ...]
    base_samples: int


def saltelli_first_total(outputs: np.ndarray, parameter_count: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Calculate SALib-compatible Saltelli 2010 first/total-order estimators.

    The legacy files use the ``calc_second_order=False`` row layout
    ``A, AB_1, ..., AB_D, B`` for each base Sobol point.
    Return raw finite-sample estimates: negative values and estimates above
    one must remain visible rather than being clipped before averaging/ranking.
    """
    stride = parameter_count + 2
    if outputs.shape[0] % stride:
        raise ValueError(
            f"{outputs.shape[0]} rows are incompatible with D={parameter_count} Saltelli samples"
        )
    # SALib normalizes each scalar model output before separating Saltelli
    # matrices; reproduce that behavior independently for every time/sensor.
    output_std = outputs.std(axis=0)
    outputs = (outputs - outputs.mean(axis=0)) / np.where(output_std > 1e-14, output_std, 1.0)
    a = outputs[0::stride]
    b = outputs[stride - 1::stride]
    variance = np.var(np.concatenate((a, b), axis=0), axis=0, ddof=0)
    safe_variance = np.where(variance > 1e-14, variance, np.nan)
    first, total = [], []
    for index in range(parameter_count):
        ab = outputs[index + 1::stride]
        first.append(np.mean(b * (ab - a), axis=0) / safe_variance)
        total.append(0.5 * np.mean((a - ab) ** 2, axis=0) / safe_variance)
    first = np.nan_to_num(np.stack(first, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)
    total = np.nan_to_num(np.stack(total, axis=-1), nan=0.0, posinf=0.0, neginf=0.0)
    return first, total, len(a)


def analyze_saved_fmu(sensor_data: np.ndarray, selected_indices,
                      sensor_indices=tuple(range(0, 16, 2))) -> SobolResult:
    selected_indices = tuple(selected_indices)
    outputs = np.asarray(sensor_data)[:, :, list(sensor_indices)]
    first, total, base_samples = saltelli_first_total(outputs, len(selected_indices))
    # estimator output is time, sensor, parameter
    first, total = first.transpose(1, 0, 2), total.transpose(1, 0, 2)
    mean_first = first.mean(axis=(0, 1))
    mean_total = total.mean(axis=(0, 1))
    ranking = tuple(np.asarray(selected_indices)[np.argsort(mean_first)[::-1]].tolist())
    return SobolResult(first, total, mean_first, mean_total, ranking, base_samples)


def _plot_mean(result: SobolResult, selected, output: Path, stage: int):
    names = [PARAMETER_NAMES[index] for index in selected]
    order = np.argsort(result.mean_first_order)[::-1]
    positions = np.arange(len(selected))
    fig, axis = plt.subplots(figsize=(max(12, len(selected) * .75), 7))
    axis.bar(positions - .2, result.mean_first_order[order], .4, label="First order $S_1$")
    axis.bar(positions + .2, result.mean_total_order[order], .4, label="Total order $S_T$", alpha=.65)
    axis.set_xticks(
        positions, np.asarray(names)[order], rotation=55, ha="right", fontsize=14
    )
    axis.tick_params(axis="y", labelsize=14)
    axis.set_ylabel("Mean Sobol index", fontsize=16)
    axis.set_title(f"Thermal calibration sensitivity — stage {stage}", fontsize=18)
    axis.grid(axis="y", alpha=.25)
    axis.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    fig.savefig(output / f"stage_{stage:02d}_mean_sobol.png", dpi=180)
    fig.savefig(output / f"stage_{stage:02d}_mean_sobol.pdf")
    plt.close(fig)


def _plot_time(result: SobolResult, selected, output: Path, stage: int, sample_seconds=18):
    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 3.2 * rows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).reshape(-1)
    time_minutes = np.arange(result.first_order.shape[1]) * sample_seconds / 60
    sensor_mean = result.first_order.mean(axis=0)
    for position, parameter in enumerate(selected):
        axes[position].plot(time_minutes, sensor_mean[:, position], color="#2563eb")
        axes[position].set_title(PARAMETER_NAMES[parameter])
        axes[position].grid(alpha=.2)
        axes[position].axhline(0, color="grey", linewidth=.6)
    for axis in axes[len(selected):]:
        axis.set_visible(False)
    fig.supxlabel("Time [min]")
    fig.supylabel("Sensor-mean first-order Sobol index")
    fig.suptitle(f"Time-resolved thermal sensitivity — stage {stage}")
    fig.tight_layout()
    fig.savefig(output / f"stage_{stage:02d}_first_order_over_time.png", dpi=180)
    plt.close(fig)


def run_saved_fmu_analysis(source_dir: Path | str, output_dir: Path | str):
    source, output = Path(source_dir), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with (output / "sobol_rankings.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("stage", "rank", "parameter_index", "parameter_name",
                         "mean_first_order", "mean_total_order"))
        for stage, selected in enumerate(SENSITIVITY_STAGES, 1):
            count = len(selected)
            parameter_values = np.load(source / f"sensitivity_analysis_param_values_{count}.npy")
            sensor_data = np.load(source / f"sensitivity_analysis_sensor_data_{count}.npy")
            if parameter_values.shape[1] != count or len(parameter_values) != len(sensor_data):
                raise ValueError(f"Misaligned stage {stage} sensitivity inputs")
            result = analyze_saved_fmu(sensor_data, selected)
            np.savez_compressed(
                output / f"stage_{stage:02d}_sobol_indices.npz",
                selected_indices=np.asarray(selected), first_order=result.first_order,
                total_order=result.total_order, mean_first_order=result.mean_first_order,
                mean_total_order=result.mean_total_order,
            )
            for rank, parameter in enumerate(result.ranking, 1):
                local = selected.index(parameter)
                writer.writerow((stage, rank, parameter, PARAMETER_NAMES[parameter],
                                 result.mean_first_order[local], result.mean_total_order[local]))
            summary = {"stage": stage, "selected_indices": list(selected),
                       "base_samples": result.base_samples, "ranking": list(result.ranking)}
            summaries.append(summary)
            _plot_mean(result, selected, output, stage)
            _plot_time(result, selected, output, stage)
            print(f"stage {stage}: {summary}", flush=True)
    (output / "sensitivity_summary.json").write_text(json.dumps({
        "source": str(source), "method": "Saltelli 2010, calc_second_order=False",
        "index_reporting": "raw finite-sample estimates; no clipping",
        "sensors": list(range(0, 16, 2)), "sample_interval_seconds": 18,
        "stages": summaries,
    }, indent=2))
    return summaries
