"""Dataset adapters preserving the existing robot state-estimation protocols.

The preceding state is explicitly included in ``states[:, 0]``. Therefore
``controls[:, t]`` advances ``states[:, t]`` to ``states[:, t + 1]`` and
``observations[:, t]`` measures ``states[:, t + 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BenchmarkData:
    train: dict[str, np.ndarray]
    validation: dict[str, np.ndarray]
    test: dict[str, np.ndarray]
    metadata: dict[str, Any]
    protocol: dict[str, Any]
    dt: float


def _json_value(value):
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _prepare_split(split, filter_controls, limit=None):
    if limit is not None and limit < 1:
        raise ValueError("Window limits must be positive")
    selected = slice(None, limit)
    states = np.asarray(split["x"], dtype=np.float32)[selected]
    initial = np.asarray(split["x0"], dtype=np.float32)[selected]
    result = {
        "states": np.concatenate((initial[:, None], states), axis=1),
        "observations": np.asarray(split["y"], dtype=np.float32)[selected],
        "controls": np.asarray(filter_controls, dtype=np.float32)[selected],
    }
    for key in ("episode_index", "start_frame"):
        result[key] = np.asarray(split[key])[selected]
    return result


def _protocol(defaults, overrides):
    unknown = set(overrides).difference(defaults)
    if unknown:
        raise TypeError(f"Unknown benchmark overrides: {sorted(unknown)}")
    protocol = dict(defaults)
    protocol.update({key: value for key, value in overrides.items() if value is not None})
    protocol["overrides"] = {
        key: value for key, value in overrides.items()
        if value is not None and value != defaults[key]
    }
    protocol["matched_defaults"] = not protocol["overrides"]
    protocol["inference_access"] = "causal"
    protocol["initial_std"] = 0.01
    protocol["metric_coordinates"] = "training-episode normalized states"
    protocol["metric_space"] = "normalized"
    protocol["posterior_samples"] = protocol["num_particles"]
    protocol["rmse_aggregation"] = "mean of per-trajectory root mean squared errors"
    return protocol


def load_aloha(variant="dataset_noisy", **overrides) -> BenchmarkData:
    """Load an ALOHA variant with the original NSF/FM particle-filter settings.

    Explicit limits are useful for smoke tests and are recorded in ``protocol``.
    They are applied after the original splits, normalization, and observation
    corruption, so reducing a budget does not change the selected data values.
    """
    from experiments.aloha_state_estimation.run_nsf_fm_flowsde import (
        VARIANTS,
        preceding_controls,
    )
    from experiments.aloha_state_estimation.utils.dataset import (
        load_aloha_insertion_dataset,
    )

    if variant not in VARIANTS:
        raise ValueError(f"Unknown ALOHA variant {variant!r}; choose {tuple(VARIANTS)}")
    selected_variant = VARIANTS[variant]
    defaults = {
        "data_path": selected_variant["data_path"],
        "seed": 0, "horizon": 250, "stride": 25, "dt": 0.02,
        "obs_std": 0.1, "train_frac": 0.8, "val_frac": 0.1,
        "train_steps": 3000, "batch_size": 256, "learning_rate": 3e-4,
        "hidden_size": 128, "hidden_depth": 3, "flow_layers": 6,
        "num_particles": 200, "eval_windows": 50,
        "max_train_windows": None, "max_eval_windows": None,
        "max_validation_windows": None,
    }
    protocol = _protocol(defaults, overrides)
    protocol.update({
        "dataset": "aloha", "variant": variant,
        "obs_noise_type": selected_variant["obs_noise_type"],
        "pseudo_process_uncertainty": selected_variant["pseudo_process_uncertainty"],
        "gmm_weights": [0.70, 0.15, 0.15],
        "gmm_offsets": [0.0, 2.5, -2.5],
        "gmm_stds": [0.50, 1.50, 1.50],
        "reference": "experiments/aloha_state_estimation/run_nsf_fm_flowsde.py",
    })
    splits, info = load_aloha_insertion_dataset(
        protocol["data_path"], horizon=protocol["horizon"],
        stride=protocol["stride"], obs_std=protocol["obs_std"],
        obs_noise_type=protocol["obs_noise_type"], seed=protocol["seed"],
        train_frac=protocol["train_frac"], val_frac=protocol["val_frac"],
        normalize=True, clean_initial_state=True,
        pseudo_process_uncertainty=protocol["pseudo_process_uncertainty"],
        dt=protocol["dt"],
    )
    test_limit = protocol["eval_windows"]
    if protocol["max_eval_windows"] is not None:
        test_limit = min(test_limit, protocol["max_eval_windows"])
    limits = {
        "train": protocol["max_train_windows"],
        "val": protocol["max_validation_windows"], "test": test_limit,
    }
    prepared = {
        name: _prepare_split(
            split, preceding_controls(
                split, protocol["data_path"], info, protocol["horizon"]),
            limits[name],
        )
        for name, split in splits.items()
    }
    metadata = _json_value(vars(info))
    metadata["control_mean"] = metadata["action_mean"]
    metadata["control_std"] = metadata["action_std"]
    metadata["observation_mean"] = metadata["state_mean"]
    metadata["observation_std"] = metadata["state_std"]
    metadata["observation_indices"] = metadata["observed_indices"]
    metadata["normalization"] = "training episodes only"
    metadata["available_windows"] = {name: len(split["x"]) for name, split in splits.items()}
    metadata["selected_windows"] = {name: len(split["states"]) for name, split in prepared.items()}
    protocol["observed_indices"] = metadata["observed_indices"]
    return BenchmarkData(
        prepared["train"], prepared["val"], prepared["test"], metadata,
        _json_value(protocol), protocol["dt"],
    )


def load_reassemble(**overrides) -> BenchmarkData:
    """Load the official REASSEMBLE episode split and matched robot PF budgets."""
    from experiments.reassemble_state_estimation.dataset import (
        load_reassemble_dataset,
    )

    defaults = {
        "data_path": "data/reassemble",
        "seed": 0, "horizon": 250, "stride": 250, "dt": 1 / 30,
        "obs_std": 0.1, "val_fraction": 0.15,
        "train_steps": 5000, "batch_size": 256, "learning_rate": 3e-4,
        "hidden_size": 128, "hidden_depth": 3, "flow_layers": 6,
        "num_particles": 500, "eval_windows": 300,
        "max_train_windows": 2000, "max_eval_windows": 300,
        "max_validation_windows": None, "obs_noise_type": "gaussian",
    }
    protocol = _protocol(defaults, overrides)
    if protocol["obs_noise_type"] == "gmm":  # opt-in (2026-10-20); multiples of obs_std
        protocol.update(gmm_weights=[0.70, 0.15, 0.15], gmm_offsets=[0.0, 2.5, -2.5], gmm_stds=[0.50, 1.50, 1.50])
    protocol.update({
        "dataset": "reassemble",
        "observed_indices": list(range(7)),
        "reference": "experiments/reassemble_state_estimation/run_nsf_fm.py",
        "historical_comparison_note": (
            "The published three-model CSV contains calibrated smoothers; "
            "these adapters retain the causal particle-filter protocol."
        ),
    })
    splits, info = load_reassemble_dataset(
        protocol["data_path"], horizon=protocol["horizon"],
        stride=protocol["stride"], obs_std=protocol["obs_std"],
        seed=protocol["seed"], val_fraction=protocol["val_fraction"],
        max_train_windows=protocol["max_train_windows"],
        max_eval_windows=protocol["max_eval_windows"],
        obs_noise_type=protocol["obs_noise_type"],
    )
    limits = {"train": None, "val": protocol["max_validation_windows"],
              "test": protocol["eval_windows"]}
    prepared = {
        name: _prepare_split(split, split["filter_u"], limits[name])
        for name, split in splits.items()
    }
    metadata = _json_value(info)
    metadata["observed_indices"] = list(range(7))
    metadata["observation_indices"] = list(range(7))
    metadata["control_mean"] = metadata["action_mean"]
    metadata["control_std"] = metadata["action_std"]
    metadata["observation_mean"] = metadata["state_mean"]
    metadata["observation_std"] = metadata["state_std"]
    metadata["selected_windows"] = {name: len(split["states"]) for name, split in prepared.items()}
    metadata["normalization"] = "training episodes only"
    return BenchmarkData(
        prepared["train"], prepared["val"], prepared["test"], metadata,
        _json_value(protocol), protocol["dt"],
    )
