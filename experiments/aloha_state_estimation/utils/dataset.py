"""ALOHA simulated insertion data for controlled state-estimation experiments.

The LeRobot dataset stores clean robot proprioception and commanded actions.
This module regards the recorded joint positions as the reference state and
constructs noisy (and optionally missing) observations.  Episodes are split
before windows are extracted, preventing frames from one demonstration from
appearing in more than one split.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import pyarrow.parquet as pq


ArrayDict = Dict[str, jnp.ndarray]


@dataclass(frozen=True)
class AlohaInsertionMetadata:
    """Information needed to interpret and de-normalize a prepared dataset."""

    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    observed_indices: Tuple[int, ...]
    train_episodes: Tuple[int, ...]
    val_episodes: Tuple[int, ...]
    test_episodes: Tuple[int, ...]
    fps: int = 50
    disturbance_matrix: Optional[np.ndarray] = None
    disturbance_config: Optional[Mapping[str, float]] = None


def _parquet_files(path: str | Path) -> list[Path]:
    path = Path(path).expanduser()
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    files = sorted(path.glob("data/**/*.parquet"))
    if not files:
        files = sorted(path.glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No parquet files found below {path}. Expected a LeRobot "
            "aloha_sim_insertion_human snapshot or a parquet file."
        )
    return files


def _stack_vector_column(values: Iterable[object], name: str) -> np.ndarray:
    rows = [np.asarray(value, dtype=np.float32) for value in values]
    if not rows:
        raise ValueError(f"Column {name!r} is empty")
    try:
        result = np.stack(rows)
    except ValueError as exc:
        raise ValueError(f"Column {name!r} contains inconsistent vector sizes") from exc
    if result.ndim != 2:
        raise ValueError(f"Column {name!r} must contain vectors, got {result.shape}")
    return result


def read_aloha_insertion_parquet(
    path: str | Path,
) -> Dict[str, np.ndarray]:
    """Read the proprioceptive columns from a LeRobot dataset snapshot."""

    columns = ["observation.state", "action", "episode_index", "frame_index"]
    parts: Dict[str, list[np.ndarray]] = {name: [] for name in columns}
    for file in _parquet_files(path):
        table = pq.read_table(file, columns=columns)
        data = table.to_pydict()
        parts["observation.state"].append(
            _stack_vector_column(data["observation.state"], "observation.state")
        )
        parts["action"].append(_stack_vector_column(data["action"], "action"))
        parts["episode_index"].append(
            np.asarray(data["episode_index"], dtype=np.int64)
        )
        parts["frame_index"].append(np.asarray(data["frame_index"], dtype=np.int64))

    result = {name: np.concatenate(chunks, axis=0) for name, chunks in parts.items()}
    order = np.lexsort((result["frame_index"], result["episode_index"]))
    return {name: value[order] for name, value in result.items()}


def split_episode_ids(
    episode_ids: Sequence[int],
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split unique episode identifiers."""

    if not (0 < train_frac < 1 and 0 <= val_frac < 1):
        raise ValueError("train_frac must be in (0, 1) and val_frac in [0, 1)")
    if train_frac + val_frac >= 1:
        raise ValueError("train_frac + val_frac must be less than 1")
    ids = np.unique(np.asarray(episode_ids, dtype=np.int64))
    if len(ids) < 3:
        raise ValueError("At least three episodes are required for train/val/test")
    ids = np.random.default_rng(seed).permutation(ids)
    n_train = max(1, int(len(ids) * train_frac))
    n_val = max(1, int(len(ids) * val_frac))
    if n_train + n_val >= len(ids):
        n_val = 1
        n_train = len(ids) - 2
    return ids[:n_train], ids[n_train : n_train + n_val], ids[n_train + n_val :]


def _safe_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    return mean, np.maximum(std, np.float32(1e-6))


def generate_pseudo_process_trajectory(
    state: np.ndarray,
    action: np.ndarray,
    episode: np.ndarray,
    *,
    seed: int,
    dt: float = 0.02,
    disturbance_dim: int = 4,
    rho_d: float = 0.98,
    sigma_d: float = 0.05,
    regime_gain: float = 2.0,
    mixing_scale: float = 0.35,
    regime_reference: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Integrate an episode-consistent latent trajectory with AR(1) disturbance.

    The interaction regime is the RMS action-tracking discrepancy,
    ``||action_t-state_t|| / sqrt(D)``, normalized by ``regime_reference``:

    ``r_t = clip(tracking_error/reference, 0, 3)``
    ``d_{t+1} = rho_d*d_t + sigma_d*(1 + regime_gain*r_t)*xi_t``
    ``v^latent_{t+1} = v^data_{t+1} + B*d_t``
    ``q^latent_{t+1} = q^latent_t + dt*v^latent_{t+1}``
    """

    if not (0 <= rho_d < 1):
        raise ValueError("rho_d must be in [0, 1)")
    if dt <= 0 or disturbance_dim < 1 or sigma_d < 0 or mixing_scale < 0:
        raise ValueError("dt and disturbance_dim must be positive; scales nonnegative")
    n_state = state.shape[1]
    if disturbance_dim > n_state:
        raise ValueError("disturbance_dim cannot exceed the state dimension")

    tracking = np.linalg.norm(action - state, axis=1) / np.sqrt(n_state)
    if regime_reference is None:
        positive = tracking[tracking > 1e-8]
        regime_reference = float(np.median(positive)) if len(positive) else 1.0
    regime_reference = max(float(regime_reference), 1e-6)

    rng_b = np.random.default_rng(seed + 7_919)
    basis, _ = np.linalg.qr(rng_b.normal(size=(n_state, disturbance_dim)))
    B = (mixing_scale * basis[:, :disturbance_dim]).astype(np.float32)
    latent = np.empty_like(state, dtype=np.float32)
    disturbances = np.zeros((len(state), disturbance_dim), dtype=np.float32)
    regimes = np.zeros(len(state), dtype=np.float32)

    for episode_id in np.unique(episode):
        idx = np.flatnonzero(episode == episode_id)
        q = state[idx]
        latent[idx[0]] = q[0]
        rng = np.random.default_rng(seed + 104_729 * (int(episode_id) + 1))
        d = np.zeros(disturbance_dim, dtype=np.float32)
        disturbances[idx[0]] = d
        for local_t in range(len(idx) - 1):
            global_t = idx[local_t]
            next_t = idx[local_t + 1]
            r_t = np.float32(np.clip(tracking[global_t] / regime_reference, 0.0, 3.0))
            regimes[global_t] = r_t
            v_data = (q[local_t + 1] - q[local_t]) / dt
            v_latent = v_data + B @ d
            latent[next_t] = latent[global_t] + dt * v_latent
            innovation_scale = sigma_d * (1.0 + regime_gain * r_t)
            d = rho_d * d + innovation_scale * rng.normal(
                size=disturbance_dim
            ).astype(np.float32)
            disturbances[next_t] = d
        regimes[idx[-1]] = regimes[idx[-2]] if len(idx) > 1 else 0.0
    return latent, disturbances, regimes, regime_reference


def _make_windows(
    state: np.ndarray,
    action: np.ndarray,
    episode: np.ndarray,
    frame: np.ndarray,
    episode_ids: np.ndarray,
    horizon: int,
    stride: int,
    observed_indices: np.ndarray,
    obs_std: float,
    obs_noise_type: str,
    obs_df: float,
    gmm_weights: Tuple[float, ...],
    gmm_offsets: Tuple[float, ...],
    gmm_stds: Tuple[float, ...],
    dropout_prob: float,
    include_mask: bool,
    clean_initial_state: bool,
    rng: np.random.Generator,
) -> ArrayDict:
    xs, ys, us, x0s, eps, starts = [], [], [], [], [], []

    def sample_observation_noise(shape):
        if obs_noise_type == "gaussian":
            return rng.normal(0.0, obs_std, shape)
        if obs_noise_type == "student_t":
            return obs_std * rng.standard_t(obs_df, shape)
        if obs_noise_type == "gmm":
            weights = np.asarray(gmm_weights, dtype=np.float64)
            components = rng.choice(len(weights), shape, p=weights)
            offsets = obs_std * np.asarray(gmm_offsets, dtype=np.float64)
            stds = obs_std * np.asarray(gmm_stds, dtype=np.float64)
            return offsets[components] + stds[components] * rng.normal(size=shape)
        raise ValueError(f"Unknown observation noise: {obs_noise_type}")

    for episode_id in episode_ids:
        idx = np.flatnonzero(episode == episode_id)
        if len(idx) <= horizon:
            continue
        # Starting at one supplies a preceding noisy observation for x0.
        for start in range(1, len(idx) - horizon + 1, stride):
            window_idx = idx[start : start + horizon]
            clean = state[window_idx]
            observed = clean[:, observed_indices]
            noise = sample_observation_noise(observed.shape)
            noisy = observed + noise.astype(np.float32)
            valid = rng.random(observed.shape) >= dropout_prob
            noisy = np.where(valid, noisy, 0.0).astype(np.float32)
            y = np.concatenate((noisy, valid.astype(np.float32)), axis=-1) if include_mask else noisy

            prev = state[idx[start - 1]].copy()
            if not clean_initial_state:
                prev += sample_observation_noise(prev.shape).astype(np.float32)
                if dropout_prob:
                    prev_valid = rng.random(prev.shape) >= dropout_prob
                    prev = np.where(prev_valid, prev, 0.0)

            xs.append(clean)
            ys.append(y)
            us.append(action[window_idx])
            x0s.append(prev)
            eps.append(episode_id)
            starts.append(frame[window_idx[0]])

    if not xs:
        raise ValueError(
            f"No windows produced: horizon={horizon} may be too long for this split"
        )
    return {
        "x": jnp.asarray(np.stack(xs)),
        "y": jnp.asarray(np.stack(ys)),
        "u": jnp.asarray(np.stack(us)),
        "x0": jnp.asarray(np.stack(x0s)),
        "episode_index": jnp.asarray(eps),
        "start_frame": jnp.asarray(starts),
    }


def load_aloha_insertion_dataset(
    path: str | Path,
    horizon: int = 50,
    stride: int = 25,
    obs_std: float = 0.1,
    obs_noise_type: str = "gaussian",
    obs_df: float = 4.0,
    gmm_weights: Sequence[float] = (0.70, 0.15, 0.15),
    gmm_offsets: Sequence[float] = (0.0, 2.5, -2.5),
    gmm_stds: Sequence[float] = (0.50, 1.50, 1.50),
    dropout_prob: float = 0.0,
    observed_indices: Optional[Sequence[int]] = None,
    include_mask: Optional[bool] = None,
    seed: int = 0,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    normalize: bool = True,
    clean_initial_state: bool = False,
    pseudo_process_uncertainty: bool = False,
    disturbance_dim: int = 4,
    rho_d: float = 0.98,
    sigma_d: float = 0.05,
    regime_gain: float = 2.0,
    mixing_scale: float = 0.35,
    dt: float = 0.02,
) -> Tuple[Mapping[str, ArrayDict], AlohaInsertionMetadata]:
    """Load, split, normalize, corrupt, and window ALOHA insertion trajectories.

    ``obs_std`` is measured in normalized state units by default.  Consequently
    a value such as 0.1 has the same interpretation for joints with different
    physical ranges.  Set ``normalize=False`` to use joint-position units.
    """

    if horizon < 1 or stride < 1:
        raise ValueError("horizon and stride must be positive")
    if obs_std < 0 or not (0 <= dropout_prob < 1):
        raise ValueError("obs_std must be nonnegative and dropout_prob in [0, 1)")
    if obs_noise_type not in {"gaussian", "student_t", "gmm"}:
        raise ValueError("obs_noise_type must be gaussian, student_t, or gmm")
    if obs_df <= 0:
        raise ValueError("obs_df must be positive")
    gmm_weights = tuple(float(value) for value in gmm_weights)
    gmm_offsets = tuple(float(value) for value in gmm_offsets)
    gmm_stds = tuple(float(value) for value in gmm_stds)
    if not (
        len(gmm_weights) == len(gmm_offsets) == len(gmm_stds) > 0
        and all(value > 0 for value in gmm_weights)
        and all(value > 0 for value in gmm_stds)
    ):
        raise ValueError("GMM vectors must have equal lengths and positive weights/stds")
    weight_sum = sum(gmm_weights)
    gmm_weights = tuple(value / weight_sum for value in gmm_weights)

    raw = read_aloha_insertion_parquet(path)
    state = raw["observation.state"]
    action = raw["action"]
    if action.shape[1] < 1:
        raise ValueError("The action column must contain at least one control dimension")
    train_ids, val_ids, test_ids = split_episode_ids(
        raw["episode_index"], seed, train_frac, val_frac
    )
    train_rows = np.isin(raw["episode_index"], train_ids)
    disturbance_matrix = None
    disturbance_config = None
    if pseudo_process_uncertainty:
        tracking = np.linalg.norm(state - action, axis=1) / np.sqrt(state.shape[1])
        positive_train = tracking[train_rows & (tracking > 1e-8)]
        regime_reference = (
            float(np.median(positive_train)) if len(positive_train) else 1.0
        )
        state, disturbances, regimes, regime_reference = generate_pseudo_process_trajectory(
            state,
            action,
            raw["episode_index"],
            seed=seed,
            dt=dt,
            disturbance_dim=disturbance_dim,
            rho_d=rho_d,
            sigma_d=sigma_d,
            regime_gain=regime_gain,
            mixing_scale=mixing_scale,
            regime_reference=regime_reference,
        )
        # Reconstruct B deterministically for metadata without storing window-level
        # disturbance arrays in every training batch.
        rng_b = np.random.default_rng(seed + 7_919)
        basis, _ = np.linalg.qr(rng_b.normal(size=(state.shape[1], disturbance_dim)))
        disturbance_matrix = (mixing_scale * basis[:, :disturbance_dim]).astype(np.float32)
        disturbance_config = {
            "rho_d": float(rho_d),
            "sigma_d": float(sigma_d),
            "regime_gain": float(regime_gain),
            "mixing_scale": float(mixing_scale),
            "disturbance_dim": float(disturbance_dim),
            "dt": float(dt),
            "regime_reference": float(regime_reference),
            "disturbance_rms": float(np.sqrt(np.mean(disturbances**2))),
            "regime_mean": float(np.mean(regimes)),
        }
    state_mean, state_std = _safe_stats(state[train_rows])
    action_mean, action_std = _safe_stats(action[train_rows])
    if normalize:
        state = (state - state_mean) / state_std
        action = (action - action_mean) / action_std
    else:
        state_mean = np.zeros_like(state_mean)
        state_std = np.ones_like(state_std)
        action_mean = np.zeros_like(action_mean)
        action_std = np.ones_like(action_std)

    if observed_indices is None:
        observed = np.arange(state.shape[1], dtype=np.int64)
    else:
        observed = np.asarray(observed_indices, dtype=np.int64)
        if observed.ndim != 1 or len(observed) == 0:
            raise ValueError("observed_indices must be a non-empty 1-D sequence")
        if np.any(observed < 0) or np.any(observed >= state.shape[1]):
            raise ValueError("observed_indices contains an out-of-range joint")
    if include_mask is None:
        include_mask = dropout_prob > 0

    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    split_data = {}
    for offset, (name, ids) in enumerate(split_ids.items()):
        split_data[name] = _make_windows(
            state,
            action,
            raw["episode_index"],
            raw["frame_index"],
            ids,
            horizon,
            stride,
            observed,
            obs_std,
            obs_noise_type,
            obs_df,
            gmm_weights,
            gmm_offsets,
            gmm_stds,
            dropout_prob,
            include_mask,
            clean_initial_state,
            np.random.default_rng(seed + 10_000 + offset),
        )

    metadata = AlohaInsertionMetadata(
        state_mean=state_mean,
        state_std=state_std,
        action_mean=action_mean,
        action_std=action_std,
        observed_indices=tuple(int(i) for i in observed),
        train_episodes=tuple(int(i) for i in train_ids),
        val_episodes=tuple(int(i) for i in val_ids),
        test_episodes=tuple(int(i) for i in test_ids),
        disturbance_matrix=disturbance_matrix,
        disturbance_config=disturbance_config,
    )
    return split_data, metadata


def denormalize_aloha_state(
    state: np.ndarray | jnp.ndarray, metadata: AlohaInsertionMetadata
) -> np.ndarray:
    """Convert normalized joint positions back to their recorded units."""

    return np.asarray(state) * metadata.state_std + metadata.state_mean


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Inspect noisy proprioceptive ALOHA insertion windows"
    )
    parser.add_argument("path", help="Dataset snapshot directory or parquet file")
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--obs-std", type=float, default=0.1)
    parser.add_argument("--dropout-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    prepared, info = load_aloha_insertion_dataset(
        args.path,
        horizon=args.horizon,
        stride=args.stride,
        obs_std=args.obs_std,
        dropout_prob=args.dropout_prob,
        seed=args.seed,
    )
    print(f"state dimension: {len(info.state_mean)}")
    print(f"observed joints: {info.observed_indices}")
    for split_name, split in prepared.items():
        print(
            f"{split_name}: x={tuple(split['x'].shape)}, "
            f"y={tuple(split['y'].shape)}, u={tuple(split['u'].shape)}, "
            f"episodes={len(set(np.asarray(split['episode_index']).tolist()))}"
        )
