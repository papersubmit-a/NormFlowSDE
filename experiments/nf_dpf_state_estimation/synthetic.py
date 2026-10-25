"""Synthetic NF-DPF datasets with the established filtering test splits.

Pendulum uses the data key and trajectory permutation of the pendulum filter
benchmark. Lorenz-96 uses the direct seed key of the dedicated FlowSDE evaluator
and classical-filter benchmark. All evaluation sequences start only at x0;
the state labels after x0 are targets, never intermediate filter resets.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.controlled_systems import (
    ControlledStochasticPendulum,
    sample_controlled_pendulum_data,
)
from experiments.lorenz96_state_estimation.utils.dataset import (
    generate_lorenz96_dataset,
    train_val_test_split,
)
from .datasets import BenchmarkData


def _finish(splits, metadata, protocol, overrides, dt):
    """Apply explicit reduced budgets after constructing the matched split."""
    defaults = dict(protocol)
    defaults.update(max_train_windows=None, max_eval_windows=None,
                    max_validation_windows=None)
    unknown = set(overrides) - set(defaults)
    if unknown:
        raise TypeError(f"Unknown benchmark overrides: {sorted(unknown)}")
    protocol = {**defaults, **{k: v for k, v in overrides.items() if v is not None}}
    for name in ("train_steps", "batch_size", "num_particles", "posterior_samples", "eval_windows"):
        if protocol[name] < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_train_windows", "max_eval_windows", "max_validation_windows"):
        if protocol[name] is not None and protocol[name] < 1:
            raise ValueError(f"{name} must be positive")
    protocol["overrides"].update({
        key: value for key, value in overrides.items()
        if value is not None and value != defaults[key]
    })
    protocol["matched_defaults"] = not protocol["overrides"]
    protocol.update(
        metric_space="physical", metric_coordinates="physical state coordinates",
        inference_access="causal", observed_indices=metadata["observation_indices"],
        rmse_aggregation="mean of per-trajectory root mean squared errors",
    )
    test_limit = protocol["eval_windows"]
    if protocol["max_eval_windows"] is not None:
        test_limit = min(test_limit, protocol["max_eval_windows"])
    limits = (protocol["max_train_windows"], protocol["max_validation_windows"], test_limit)
    limited = tuple({name: value[:limit] for name, value in split.items()}
                    for split, limit in zip(splits, limits))
    metadata["available_windows"] = metadata["split_counts"]
    metadata["selected_windows"] = dict(zip(("train", "validation", "test"),
                                           (len(s["states"]) for s in limited)))
    return BenchmarkData(train=limited[0], validation=limited[1], test=limited[2],
                         metadata=metadata, protocol=protocol, dt=float(dt))


def _check_settings(n_samples, horizon, dt, obs_std, train_frac, val_frac):
    if n_samples < 3 or horizon < 1 or dt <= 0 or obs_std <= 0:
        raise ValueError("Need at least three trajectories and positive horizon, dt, obs_std")
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 - train_frac:
        raise ValueError("Training and validation fractions must leave a test split")
    if int(n_samples * train_frac) < 1 or int(n_samples * val_frac) < 1:
        raise ValueError("Training and validation splits must each contain a trajectory")


def _normalize(splits, observation_indices):
    """Fit transforms solely on training trajectories, including their x0."""
    train = splits[0]
    state_mean = np.asarray(train["states"]).mean(axis=(0, 1))
    state_std = np.maximum(np.asarray(train["states"]).std(axis=(0, 1)), 1e-6)
    control_mean = np.asarray(train["controls"]).mean(axis=(0, 1))
    control_std = np.maximum(np.asarray(train["controls"]).std(axis=(0, 1)), 1e-6)
    # Coordinate observations use the same affine transform as their states.
    # This preserves the physical observation operator in normalized units.
    observation_mean = state_mean[observation_indices]
    observation_std = state_std[observation_indices]
    normalized = tuple({
        "states": jnp.asarray((np.asarray(split["states"]) - state_mean) / state_std),
        "observations": jnp.asarray(
            (np.asarray(split["observations"]) - observation_mean) / observation_std),
        "controls": jnp.asarray(
            (np.asarray(split["controls"]) - control_mean) / control_std),
    } for split in splits)
    metadata = {
        "state_mean": state_mean.tolist(), "state_std": state_std.tolist(),
        "observation_mean": observation_mean.tolist(),
        "observation_std": observation_std.tolist(),
        "control_mean": control_mean.tolist(), "control_std": control_std.tolist(),
        "normalization_source": "training trajectories only, including initial states",
        "arrays_normalized": True, "metric_units": "physical",
        "observation_indices": list(observation_indices),
        "known_initial_state": True, "true_intermediate_state_resets": False,
        "split_counts": dict(zip(("train", "validation", "test"),
                                 (int(s["states"].shape[0]) for s in splits))),
    }
    return normalized, metadata


def load_pendulum(
    noise_model="gaussian", *, seed=2026, n_samples=320, horizon=60,
    dt=0.05, obs_std=0.1, train_frac=0.7, val_frac=0.15,
    integration_substeps=8, gmm_separation=0.8, observation_noise="gaussian",
    obs_gmm_weights=(0.70, 0.15, 0.15), obs_gmm_offsets=(0.0, 2.5, -2.5),
    obs_gmm_stds=(0.50, 1.50, 1.50), **overrides,
):
    """Load the 224/48/48 pendulum filtering split (defaults).

    ``noise_model`` describes process noise. Observations are full-state and
    independent; ``observation_noise="gmm"`` (opt-in, 2026-10-20) replaces the
    Gaussian measurement noise by the Lorenz-96 three-component mixture with
    offsets and standard deviations in multiples of ``obs_std``.
    """
    if observation_noise not in ("gaussian", "gmm"):
        raise ValueError("observation_noise must be gaussian or gmm")
    _check_settings(n_samples, horizon, dt, obs_std, train_frac, val_frac)
    process = ControlledStochasticPendulum(
        noise_model=noise_model, integration_substeps=integration_substeps,
        gmm_separation=gmm_separation,
    )
    data_key = jr.split(jr.key(seed), 3)[0]
    data = sample_controlled_pendulum_data(
        data_key, process, n_samples, horizon, delta_t=dt,
    )
    if observation_noise == "gaussian":
        observations = data["states"][:, 1:] + obs_std * jr.normal(
            jr.fold_in(data_key, 71_021), data["states"][:, 1:].shape,
        )
    else:
        shape = data["states"][:, 1:].shape
        component = jr.choice(jr.fold_in(data_key, 71_022), len(obs_gmm_weights), shape,
                              p=jnp.asarray(obs_gmm_weights))
        offsets, stds = jnp.asarray(obs_gmm_offsets)[component], jnp.asarray(obs_gmm_stds)[component]
        observations = data["states"][:, 1:] + obs_std * (
            offsets + stds * jr.normal(jr.fold_in(data_key, 71_021), shape))
    permutation = np.asarray(jr.permutation(jr.key(seed + 1), n_samples))
    train_end = int(train_frac * n_samples)
    validation_end = train_end + int(val_frac * n_samples)
    indices = (permutation[:train_end], permutation[train_end:validation_end],
               permutation[validation_end:])
    raw_splits = tuple({
        "states": data["states"][ix], "observations": observations[ix],
        "controls": data["controls"][ix],
    } for ix in indices)
    splits, metadata = _normalize(raw_splits, [0, 1])
    metadata.update({
        "dataset": "pendulum", "seed": seed, "horizon": horizon, "dt": dt,
        "process_noise": noise_model, "observation_noise": observation_noise,
        "physical_observation_std": obs_std,
        **({"gmm_weights": list(obs_gmm_weights), "gmm_offsets": list(obs_gmm_offsets),
            "gmm_stds": list(obs_gmm_stds), "gmm_parameter_units": "multiples of obs_std"}
           if observation_noise == "gmm" else {}),
        "integration_substeps": integration_substeps,
        "gmm_separation": gmm_separation if noise_model == "gmm" else None,
        "state_units": ["radian (unwrapped)", "radian/second"],
        "rng_policy": "data=split(key(seed),3)[0]; observations=fold_in(data,71021); split=key(seed+1)",
        "trajectory_indices": dict(zip(("train", "validation", "test"),
                                       (ix.tolist() for ix in indices))),
        "reference": "experiments/nsf_state_estimation/run_pendulum_filter_baseline.py",
    })
    protocol = {
        "dataset": "pendulum", "noise_model": noise_model,
        "horizon": horizon, "n_samples": n_samples, "dt": dt,
        "obs_std": obs_std, "obs_noise_type": observation_noise,
        "train_steps": 4400, "batch_size": 256, "learning_rate": 2e-4,
        "hidden_size": 64, "hidden_depth": 2, "flow_layers": 4,
        "num_particles": 500, "posterior_samples": 128,
        "eval_windows": len(indices[2]), "seed": seed, "initial_std": 0.0,
        "ci": 90.0,
        "budget_reference": "2400 pretraining + 2000 joint updates in pendulum reconstruction",
        "budget_note": "Update count matches the transition baseline; the NF-DPF objective and sample usage differ.",
        "comparison_caveats": [
            "Compare filtering metrics to the pendulum filtering baselines; recursive transition reconstruction receives no observations.",
            "The learned amortized and latent filtering baselines used up to 500 epochs rather than a fixed update count.",
        ],
    }
    dataset_settings = dict(seed=seed, n_samples=n_samples, horizon=horizon, dt=dt,
                            obs_std=obs_std, train_frac=train_frac, val_frac=val_frac,
                            integration_substeps=integration_substeps,
                            gmm_separation=gmm_separation, observation_noise=observation_noise)
    dataset_defaults = dict(seed=2026, n_samples=320, horizon=60, dt=0.05,
                            obs_std=0.1, train_frac=0.7, val_frac=0.15,
                            integration_substeps=8, gmm_separation=0.8, observation_noise="gaussian")
    protocol["overrides"] = {k: v for k, v in dataset_settings.items()
                             if v != dataset_defaults[k]}
    return _finish(splits, metadata, protocol, overrides, dt)


def load_lorenz96(
    *, seed=0, n_samples=2000, horizon=100, dt=0.02, obs_std=0.5,
    train_frac=0.8, val_frac=0.1, N=12, obs_interval=2, F=8.0,
    spinup_steps=500, noise_model="gmm", obs_df=5.0,
    gmm_weights=(0.70, 0.15, 0.15), gmm_offsets=(0.0, 2.5, -2.5),
    gmm_stds=(0.50, 1.50, 1.50), **overrides,
):
    """Load Lorenz-96 N12 under the documented common filtering protocol.

    The direct seed key intentionally matches ``benchmark_filters.py`` and
    ``evaluate_flow_sde.py``. Legacy learned-model training scripts instead use
    a split seed key; that difference is recorded rather than silently hidden.
    """
    _check_settings(n_samples, horizon, dt, obs_std, train_frac, val_frac)
    if N < 4 or obs_interval < 1 or spinup_steps < 0:
        raise ValueError("Lorenz-96 needs N >= 4, a positive observation interval and nonnegative spinup")
    if noise_model not in ("gaussian", "student_t", "gmm"):
        raise ValueError("noise_model must be gaussian, student_t, or gmm")
    data_key = jr.PRNGKey(seed)
    data = generate_lorenz96_dataset(
        data_key, n_samples=n_samples, horizon=horizon, N=N,
        obs_interval=obs_interval, obs_std=obs_std,
        obs_df=obs_df if noise_model == "student_t" else None,
        obs_noise_type=noise_model, gmm_weights=tuple(gmm_weights),
        gmm_offsets=tuple(gmm_offsets), gmm_stds=tuple(gmm_stds),
        dt=dt, F=F, spinup_steps=spinup_steps, d_u=1,
    )
    raw = train_val_test_split(data, jr.fold_in(data_key, 1), train_frac, val_frac)
    raw_splits = tuple({
        "states": jnp.concatenate((split["x0"][:, None], split["x"]), axis=1),
        "observations": split["y"], "controls": split["u"],
    } for split in raw)
    observation_indices = list(range(0, N, obs_interval))
    splits, metadata = _normalize(raw_splits, observation_indices)
    metadata.update({
        "dataset": "lorenz96", "seed": seed, "horizon": horizon, "dt": dt,
        "state_dim": N, "forcing": F, "spinup_steps": spinup_steps,
        "integrator": "RK4", "process_noise": "deterministic",
        "observation_noise": noise_model, "physical_observation_std": obs_std,
        "obs_df": obs_df if noise_model == "student_t" else None,
        "gmm_weights": list(gmm_weights), "gmm_offsets": list(gmm_offsets),
        "gmm_stds": list(gmm_stds), "gmm_parameter_units": "multiples of obs_std",
        "state_units": ["Lorenz-96 state"] * N,
        "rng_policy": "data=PRNGKey(seed); split=fold_in(data,1)",
        "reference": "experiments/lorenz96_state_estimation/benchmark_filters.py",
    })
    protocol = {
        "dataset": "lorenz96", "noise_model": noise_model,
        "horizon": horizon, "n_samples": n_samples, "dt": dt,
        "obs_std": obs_std, "obs_noise_type": noise_model,
        "train_steps": 12500, "batch_size": 64, "learning_rate": 3e-4,
        "hidden_size": 128, "hidden_depth": 3, "flow_layers": 6,
        "num_particles": 1000, "posterior_samples": 200,
        "eval_windows": min(100, len(raw[2]["x"])), "seed": seed,
        "initial_std": 0.0, "ci": 90.0,
        "budget_reference": "500 epochs * 25 batches for 1600 training trajectories and batch size 64",
        "budget_note": "Update count matches the FlowSDE maximum; NF-DPF uses transition batches, so example usage and objectives differ.",
        "comparison_caveats": [
            "Existing aggregate results mix observation distributions, horizon conventions, particle counts and access to future observations.",
            "The common protocol uses horizon 100, 100 test windows and 200 posterior draws; NF-DPF retains 1000 internal particles to match PF baselines.",
            "Legacy learned-model training uses split(PRNGKey(seed),3)[0]; this adapter matches the direct-key dedicated evaluator and classical-filter split.",
            "Evaluation is strictly causal with only known x0, without true intermediate segment starts.",
        ],
    }
    dataset_settings = dict(seed=seed, n_samples=n_samples, horizon=horizon, dt=dt,
                            obs_std=obs_std, train_frac=train_frac, val_frac=val_frac,
                            N=N, obs_interval=obs_interval, F=F,
                            spinup_steps=spinup_steps, noise_model=noise_model,
                            obs_df=obs_df, gmm_weights=tuple(gmm_weights),
                            gmm_offsets=tuple(gmm_offsets), gmm_stds=tuple(gmm_stds))
    dataset_defaults = dict(seed=0, n_samples=2000, horizon=100, dt=0.02,
                            obs_std=0.5, train_frac=0.8, val_frac=0.1,
                            N=12, obs_interval=2, F=8.0, spinup_steps=500,
                            noise_model="gmm", obs_df=5.0,
                            gmm_weights=(0.70, 0.15, 0.15),
                            gmm_offsets=(0.0, 2.5, -2.5),
                            gmm_stds=(0.50, 1.50, 1.50))
    protocol["overrides"] = {k: v for k, v in dataset_settings.items()
                             if v != dataset_defaults[k]}
    return _finish(splits, metadata, protocol, overrides, dt)
