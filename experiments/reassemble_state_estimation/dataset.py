"""Windowed joint-state dataset for the REASSEMBLE LeRobot v3 snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq


def load_reassemble_dataset(
    root: str | Path = "data/reassemble", *, horizon: int = 250,
    stride: int = 250, obs_std: float = 0.1, seed: int = 0,
    val_fraction: float = 0.15, max_train_windows: int = 2000,
    max_eval_windows: int = 300, obs_noise_type: str = "gaussian",
    gmm_weights=(0.70, 0.15, 0.15), gmm_offsets=(0.0, 2.5, -2.5),
    gmm_stds=(0.50, 1.50, 1.50),
):
    # obs_noise_type="gmm" (opt-in, 2026-10-20): three-component mixture, offsets and
    # standard deviations in multiples of obs_std (the Lorenz-96 / ALOHA convention).
    if obs_noise_type not in ("gaussian", "gmm"):
        raise ValueError("obs_noise_type must be gaussian or gmm")
    root = Path(root)
    paths = sorted((root / "data").glob("**/*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No REASSEMBLE Parquet shards below {root}")
    columns = ["observation.state.joint_position", "action",
               "episode_index", "frame_index"]
    tables = [pq.read_table(path, columns=columns) for path in paths]
    state = np.concatenate([
        np.asarray(table[columns[0]].to_pylist(), dtype=np.float32)
        for table in tables])
    action = np.concatenate([
        np.asarray(table[columns[1]].to_pylist(), dtype=np.float32)
        for table in tables])
    episode = np.concatenate([np.asarray(table[columns[2]]) for table in tables])
    frame = np.concatenate([np.asarray(table[columns[3]]) for table in tables])
    split_map = json.loads((root / "meta/splits.json").read_text())
    train_ids = np.asarray([int(key) for key, value in split_map.items()
                            if value["split"] == "train"], dtype=np.int64)
    test_ids = np.asarray([int(key) for key, value in split_map.items()
                           if value["split"] == "test"], dtype=np.int64)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(train_ids)
    val_count = max(1, int(round(val_fraction * len(shuffled))))
    val_ids, train_ids = shuffled[:val_count], shuffled[val_count:]
    train_rows = np.isin(episode, train_ids)
    state_mean, state_std = state[train_rows].mean(0), state[train_rows].std(0)
    action_mean, action_std = action[train_rows].mean(0), action[train_rows].std(0)
    state_std, action_std = np.maximum(state_std, 1e-6), np.maximum(action_std, 1e-6)
    state = (state - state_mean) / state_std
    action = (action - action_mean) / action_std

    def windows(ids, limit, noise_seed):
        xs, ys, us, filter_us, x0s, episode_ids, starts = [], [], [], [], [], [], []
        noise_rng = np.random.default_rng(noise_seed)
        for episode_id in ids:
            rows = np.flatnonzero(episode == episode_id)
            order = np.argsort(frame[rows]); rows = rows[order]
            for start in range(1, len(rows) - horizon + 1, stride):
                selected = rows[start:start + horizon]
                xs.append(state[selected]); us.append(action[selected])
                filter_us.append(action[rows[start - 1:start + horizon - 1]])
                if obs_noise_type == "gaussian":
                    noise = noise_rng.normal(0.0, obs_std, (horizon, 7))
                else:
                    component = noise_rng.choice(len(gmm_weights), size=(horizon, 7), p=gmm_weights)
                    noise = obs_std * (np.asarray(gmm_offsets)[component] + np.asarray(gmm_stds)[component]
                                       * noise_rng.standard_normal((horizon, 7)))
                ys.append(state[selected] + noise.astype(np.float32))
                x0s.append(state[rows[start - 1]])
                episode_ids.append(episode_id); starts.append(frame[selected[0]])
        if not xs:
            raise ValueError(f"No windows for horizon={horizon}, stride={stride}")
        order = noise_rng.permutation(len(xs))[:limit]
        def take(values): return jnp.asarray(np.asarray(values)[order])
        return {"x": take(xs), "y": take(ys), "u": take(us),
                "filter_u": take(filter_us), "x0": take(x0s),
                "episode_index": take(episode_ids), "start_frame": take(starts)}

    splits = {
        "train": windows(train_ids, max_train_windows, seed + 101),
        "val": windows(val_ids, max_eval_windows, seed + 102),
        "test": windows(test_ids, max_eval_windows, seed + 103),
    }
    metadata = {"state_mean": state_mean, "state_std": state_std,
                "action_mean": action_mean, "action_std": action_std,
                "train_episodes": train_ids, "val_episodes": val_ids,
                "test_episodes": test_ids}
    return splits, metadata
