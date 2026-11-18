"""Actuator force-estimation windows built from the prepared ``inputs_*`` / ``outputs_*`` arrays.

The prepared arrays hold consecutive, non-overlapping 10-step windows of one chronological recording
stream: ``inputs_<split>.npy`` (N, 10, 2) are the two force channels (left, right) and
``outputs_<split>.npy`` (N, 10, 4) are four measured actuator signals. The splits are chronological
(train, then val, then test). As in the earlier conditional-flow study of this data, the forces are the
quantity to infer and the four measured signals are what the estimator sees:

    states       = forces (2)            hidden, to be estimated
    observations = measured signals (4)  filter input
    controls     = none (width 0)

The 10-step windows are joined back into continuous series, cut at recording boundaries (a jump in a
measured signal far above its usual step), and re-windowed to ``horizon`` transitions. Following the
benchmark convention, ``states[:, 0]`` is the known initial state and ``observations[:, k]`` is measured
at the time of ``states[:, k + 1]``. Everything is z-scored with training-split statistics.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from experiments.nf_dpf_state_estimation.datasets import BenchmarkData, _json_value

SPLIT_FILES = {"train": "train", "validation": "val", "test": "test"}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def continuous_segments(outputs, jump_factor):
    """Split a (T, 4) series where any channel steps by more than mean + jump_factor * std of its steps."""
    step = np.abs(np.diff(outputs, axis=0))
    threshold = step.mean(axis=0) + jump_factor * step.std(axis=0)
    breaks = np.flatnonzero((step > threshold).any(axis=1)) + 1
    edges = np.concatenate(([0], breaks, [len(outputs)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:])], threshold


def _windows(force, signal, segments, horizon, stride, history):
    """``history`` rows are the measurements just before observations[0] (NaN before the recording starts)."""
    states, observations, starts, earlier = [], [], [], []
    for begin, end in segments:
        for start in range(begin, end - horizon, stride):
            states.append(force[start:start + horizon + 1])
            observations.append(signal[start + 1:start + horizon + 1])
            starts.append(start)
            past = np.full((history, signal.shape[1]), np.nan)
            available = min(history, start + 1 - begin)
            if available:
                past[history - available:] = signal[start + 1 - available:start + 1]
            earlier.append(past)
    if not states:
        raise ValueError(f"No continuous stretch holds {horizon + 1} samples")
    return np.asarray(states), np.asarray(observations), np.asarray(starts), np.asarray(earlier)


def load_actuator(data_path, *, horizon=100, train_stride=50, eval_stride=None, dt=0.01,
                  jump_factor=10.0, num_particles=200, seed=0, history=0) -> BenchmarkData:
    data_path = Path(data_path)
    eval_stride = horizon if eval_stride is None else eval_stride
    raw, sources = {}, []
    for name, suffix in SPLIT_FILES.items():
        files = [data_path / f"inputs_{suffix}.npy", data_path / f"outputs_{suffix}.npy"]
        force, signal = (np.load(path) for path in files)
        if force.ndim != 3 or signal.ndim != 3 or force.shape[:2] != signal.shape[:2]:
            raise ValueError(f"Unexpected prepared array shapes in {name}: {force.shape}, {signal.shape}")
        raw[name] = (force.reshape(-1, force.shape[-1]), signal.reshape(-1, signal.shape[-1]))
        sources += [dict(path=str(path), size_bytes=path.stat().st_size, sha256=_sha256(path)) for path in files]
    state_mean, state_std = raw["train"][0].mean(axis=0), raw["train"][0].std(axis=0)
    observation_mean, observation_std = raw["train"][1].mean(axis=0), raw["train"][1].std(axis=0)
    splits, segment_report = {}, {}
    for name, (force, signal) in raw.items():
        segments, threshold = continuous_segments(signal, jump_factor)
        stride = train_stride if name == "train" else eval_stride
        states, observations, starts, earlier = _windows(
            (force - state_mean) / state_std, (signal - observation_mean) / observation_std,
            segments, horizon, stride, history)
        splits[name] = dict(states=states.astype(np.float32), observations=observations.astype(np.float32),
                            controls=np.zeros(observations.shape[:2] + (0,), np.float32), start_sample=starts)
        if history:  # same windows as history=0; only this extra array is added
            splits[name]["history"] = earlier.astype(np.float32)
        lengths = [b - a for a, b in segments]
        segment_report[name] = dict(samples=len(force), segments=len(segments),
                                    usable_segments=int(sum(length > horizon for length in lengths)),
                                    longest=int(max(lengths)), jump_threshold=threshold.tolist(),
                                    windows=len(states), stride=stride)
    protocol = dict(dataset="actuator", seed=seed, horizon=horizon, train_stride=train_stride,
                    eval_stride=eval_stride, dt=dt, jump_factor=jump_factor, history=history, num_particles=num_particles,
                    posterior_samples=num_particles, eval_windows=len(splits["test"]["states"]),
                    observed_indices=[], metric_space="normalized", inference_access="causal",
                    state="force left/right (prepared inputs_*.npy)",
                    observation="four measured actuator signals (prepared outputs_*.npy)",
                    alignment="observations[k] is measured at the time of states[k+1]; states[0] is given",
                    rmse_aggregation="mean of per-trajectory root mean squared errors")
    metadata = dict(state_mean=state_mean, state_std=state_std, observation_mean=observation_mean,
                    observation_std=observation_std, observed_indices=[], normalization="training split only",
                    segments=segment_report, prepared_files=sources,
                    selected_windows={name: len(split["states"]) for name, split in splits.items()})
    return BenchmarkData(splits["train"], splits["validation"], splits["test"],
                         _json_value(metadata), _json_value(protocol), dt)
