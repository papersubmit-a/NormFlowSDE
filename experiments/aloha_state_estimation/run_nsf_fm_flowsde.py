"""Train and evaluate NSF or FM-FlowSDE on ALOHA insertion variants.

Both models learn the controlled latent-state transition and are evaluated as
bootstrap-particle-filter transition priors.  Thus observations enter through
the likelihood, rather than being accidentally used as an uncorrected proposal.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import logsumexp

from bayesinf.applications.state_estimation import (
    FlowMatchingSDE,
    FlowSDEConfig,
    NSFTrainingConfig,
    NeuralStochasticFlow,
    flow_sde_loss,
    initialise_fm_sde_optimizer,
    make_fm_sde_train_step,
    train_nsf_nll,
)
from bayesinf.applications.state_estimation.filtering_baselines import (
    _observation_log_likelihood,
)
from bayesinf.applications.state_estimation.utils.plots import (
    save_state_estimation_plots,
)
from experiments.aloha_state_estimation.utils.dataset import (
    load_aloha_insertion_dataset,
    read_aloha_insertion_parquet,
)


VARIANTS = {
    "dataset_noisy": {
        "data_path": "data/aloha_sim_insertion_human",
        "obs_noise_type": "gaussian",
        "pseudo_process_uncertainty": False,
    },
    "mujoco_random": {
        "data_path": "data/aloha_mujoco/mujoco_random_lerobot_matched.parquet",
        "obs_noise_type": "gmm",
        "pseudo_process_uncertainty": False,
    },
    "pseudo_process_uncertainty": {
        "data_path": "data/aloha_sim_insertion_human",
        "obs_noise_type": "gaussian",
        "pseudo_process_uncertainty": True,
    },
    # Opt-in (2026-10-20): the same disturbed trajectories with three-component mixture observation noise.
    "pseudo_process_uncertainty_gmm": {
        "data_path": "data/aloha_sim_insertion_human",
        "obs_noise_type": "gmm",
        "pseudo_process_uncertainty": True,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("nsf", "fm_flowsde"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/results_state_estimation/Aloha_sim_insertion"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=250)
    parser.add_argument("--window-stride", type=int, default=25)
    parser.add_argument("--obs-std", type=float, default=0.1)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-depth", type=int, default=3)
    parser.add_argument("--flow-layers", type=int, default=6)
    parser.add_argument("--diffusion", type=float, default=0.0025)
    parser.add_argument("--bridge-noise-scale", type=float, default=0.01)
    parser.add_argument("--num-particles", type=int, default=200)
    parser.add_argument("--eval-windows", type=int, default=50)
    parser.add_argument(
        "--transition-temperature", type=float, default=1.0,
        help="Scale transition noise at filtering time without changing its mean.",
    )
    return parser.parse_args()


def transitions(split, dt):
    states, controls = jnp.asarray(split["x"]), jnp.asarray(split["u"])
    batch, horizon = states.shape[:2]
    times = jnp.broadcast_to(dt * jnp.arange(horizon), (batch, horizon))
    return {
        "x_current": states[:, :-1].reshape((-1, states.shape[-1])),
        "x_next": states[:, 1:].reshape((-1, states.shape[-1])),
        "control": controls[:, :-1].reshape((-1, controls.shape[-1])),
        "delta_t": jnp.full((batch * (horizon - 1),), dt),
        "t_current": times[:, :-1].reshape(-1),
    }


def train_nsf(train, validation, args, state_dim, control_dim):
    model = NeuralStochasticFlow(
        state_dim, control_dim, hidden_size=args.hidden_size,
        hidden_depth=args.hidden_depth, num_coupling_layers=args.flow_layers,
        include_absolute_time=False, key=jr.key(args.seed + 1),
    )
    config = NSFTrainingConfig(
        steps=args.train_steps, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed + 2,
    )
    def progress(step, loss):
        if step == 0 or (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "train_nll": float(loss)}), flush=True)
    model, losses = train_nsf_nll(model, train, config, progress=progress)
    val_nll = float(-jnp.mean(model.log_prob(
        validation["x_next"], validation["x_current"], validation["delta_t"],
        validation["control"])))
    return model, np.asarray(losses), {"validation_nll": val_nll}


def train_fm(train, validation, args, state_dim, control_dim):
    model = FlowMatchingSDE(
        state_dim, control_dim=control_dim, width=args.hidden_size,
        depth=args.hidden_depth, diffusion=args.diffusion,
        use_potential=False, key=jr.key(args.seed + 1),
    )
    config = FlowSDEConfig(
        diffusion=args.diffusion, bridge_noise_scale=args.bridge_noise_scale,
        solver="euler", dt0=args.dt,
    )
    optimizer, state = initialise_fm_sde_optimizer(
        model, learning_rate=args.learning_rate)
    update = make_fm_sde_train_step(optimizer)
    losses = []
    started = time.perf_counter()
    count = train["x_current"].shape[0]
    for step in range(args.train_steps):
        batch_key, loss_key = jr.split(jr.fold_in(jr.key(args.seed + 2), step))
        ids = jr.randint(batch_key, (args.batch_size,), 0, count)
        model, state, parts, grad_norm, finite = update(
            model, state, train["x_current"][ids], train["x_next"][ids],
            train["t_current"][ids],
            train["t_current"][ids] + train["delta_t"][ids], loss_key,
            train["control"][ids], config)
        if not bool(finite):
            raise FloatingPointError(f"non-finite FM-FlowSDE gradient at step {step + 1}")
        losses.append(float(parts.total))
        if step == 0 or (step + 1) % 100 == 0:
            print(json.dumps({"step": step + 1, "train_loss": losses[-1],
                              "grad_norm": float(grad_norm)}), flush=True)
    val = flow_sde_loss(
        model, validation["x_current"], validation["x_next"],
        validation["t_current"], validation["t_current"] + validation["delta_t"],
        key=jr.key(args.seed + 77), controls=validation["control"], config=config)
    return model, np.asarray(losses), {
        "validation_flow_matching": float(val.flow_matching),
        "training_seconds": time.perf_counter() - started,
    }


def preceding_controls(test, data_path, metadata, horizon):
    """Align action[t-1] with the transition from x[t-1] to x[t]."""
    raw = read_aloha_insertion_parquet(data_path)
    normalized = (raw["action"] - metadata.action_mean) / metadata.action_std
    result = []
    for episode_id, start_frame, window_controls in zip(
        np.asarray(test["episode_index"]), np.asarray(test["start_frame"]),
        np.asarray(test["u"]),
    ):
        episode_rows = np.flatnonzero(raw["episode_index"] == int(episode_id))
        episode_frames = raw["frame_index"][episode_rows]
        local = np.flatnonzero(episode_frames == int(start_frame) - 1)
        if len(local) != 1:
            raise ValueError("Could not locate the action preceding a test window")
        first = normalized[episode_rows[int(local[0])]]
        result.append(np.concatenate((first[None], window_controls[:-1]), axis=0))
    controls = np.stack(result)
    if controls.shape[1] != horizon:
        raise AssertionError("filter-control alignment produced the wrong horizon")
    return jnp.asarray(controls)


def filter_samples(model, method, test, filter_controls, args, obs_noise_type):
    count = min(args.eval_windows, int(test["x"].shape[0]))
    truth = jnp.asarray(test["x"][:count])
    observations = jnp.asarray(test["y"][:count])
    controls = jnp.asarray(filter_controls[:count])
    initial = jnp.asarray(test["x0"][:count])
    state_dim = truth.shape[-1]
    weights, offsets, stds = (0.70, 0.15, 0.15), (0.0, 2.5, -2.5), (0.50, 1.50, 1.50)
    filter_type = getattr(args, "filter_type", "bootstrap")
    if filter_type not in ("bootstrap", "gaussian"):
        raise ValueError(f"Unknown filter type: {filter_type}")
    if filter_type == "gaussian" and obs_noise_type != "gaussian":
        raise ValueError("The Gaussian update requires Gaussian observation noise")

    def sequence(y, commands, x0, key):
        key, init_key = jr.split(key)
        particles = x0 + 1e-2 * jr.normal(init_key, (args.num_particles, state_dim))
        def step(carry, inputs):
            particles, key, index = carry
            observation, control = inputs
            key, transition_key, resample_key = jr.split(key, 3)
            if method == "nsf":
                normalized = (particles - model.state_mean) / model.state_std
                normalized_control = (control - model.control_mean) / model.control_std
                particle_keys = jr.split(transition_key, args.num_particles)
                def tempered_draw(draw_key, current):
                    condition = model._condition(
                        current, args.dt, normalized_control, jnp.asarray(0.0))
                    mean, std = model._base_parameters(current, condition, args.dt)
                    base = mean + args.transition_temperature * std * jr.normal(
                        draw_key, (model.state_dim,), dtype=current.dtype)
                    following, _ = model._forward_single(base, condition, args.dt)
                    return following
                propagated = jax.vmap(tempered_draw)(particle_keys, normalized)
                propagated = propagated * model.state_std + model.state_mean
            else:
                drift = jax.vmap(lambda x: model(x, index * args.dt, 0.0, control))(
                    particles)
                propagated = particles + args.dt * drift
                propagated += (args.transition_temperature
                    * jnp.sqrt(2.0 * args.diffusion * args.dt)) * jr.normal(
                    transition_key, particles.shape)
            if filter_type == "bootstrap":
                residual = observation[None] - propagated
                log_weights = _observation_log_likelihood(
                    residual, obs_noise_type, args.obs_std, 4.0,
                    weights, offsets, stds)
                log_weights -= logsumexp(log_weights)
                selected = jr.categorical(
                    resample_key, log_weights, shape=(args.num_particles,))
                following = propagated[selected]
            else:
                # Moment-match the learned transition ensemble, then perform
                # the exact linear-Gaussian update for the fully observed state.
                prior_mean = jnp.mean(propagated, axis=0)
                centered = propagated - prior_mean
                prior_cov = centered.T @ centered / (args.num_particles - 1)
                observation_cov = (args.obs_std ** 2) * jnp.eye(state_dim)
                innovation_cov = prior_cov + observation_cov
                gain = jnp.linalg.solve(innovation_cov, prior_cov).T
                posterior_mean = prior_mean + gain @ (observation - prior_mean)
                posterior_cov = prior_cov - gain @ prior_cov
                posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)
                eigenvalues, eigenvectors = jnp.linalg.eigh(posterior_cov)
                root = eigenvectors * jnp.sqrt(jnp.maximum(eigenvalues, 1e-8))
                noise = jr.normal(resample_key, propagated.shape)
                following = posterior_mean + noise @ root.T
            return (following, key, index + 1), following
        _, paths = jax.lax.scan(
            step, (particles, key, jnp.asarray(0)), (y, commands))
        return jnp.swapaxes(paths, 0, 1)

    keys = jr.split(jr.key(args.seed + 99), count)
    samples = jax.jit(jax.vmap(sequence))(observations, controls, initial, keys)
    return np.asarray(truth), np.asarray(observations), np.asarray(samples)


def main():
    args = parse_args()
    variant = VARIANTS[args.variant]
    output = args.output_root / args.variant
    output.mkdir(parents=True, exist_ok=True)
    splits, metadata = load_aloha_insertion_dataset(
        variant["data_path"], horizon=args.horizon, stride=args.window_stride,
        obs_std=args.obs_std, obs_noise_type=variant["obs_noise_type"],
        seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac,
        normalize=True, clean_initial_state=True,
        pseudo_process_uncertainty=variant["pseudo_process_uncertainty"], dt=args.dt)
    train, validation = transitions(splits["train"], args.dt), transitions(
        splits["val"], args.dt)
    state_dim, control_dim = train["x_current"].shape[-1], train["control"].shape[-1]
    started = time.perf_counter()
    if args.method == "nsf":
        model, losses, diagnostics = train_nsf(
            train, validation, args, state_dim, control_dim)
    else:
        model, losses, diagnostics = train_fm(
            train, validation, args, state_dim, control_dim)
    diagnostics["total_seconds_before_evaluation"] = time.perf_counter() - started
    checkpoint = output / f"{args.method}_aloha_insertion.eqx"
    eqx.tree_serialise_leaves(checkpoint, model)
    np.save(output / f"{args.method}_training_losses.npy", losses)
    filter_controls = preceding_controls(
        splits["test"], variant["data_path"], metadata, args.horizon)
    truth, observations, samples = filter_samples(
        model, args.method, splits["test"], filter_controls, args,
        variant["obs_noise_type"])
    method_name = "nsf_particle_filter" if args.method == "nsf" else "fm_flowsde_particle_filter"
    save_state_estimation_plots(
        output, method_name, truth, observations, samples, dt=args.dt)
    metrics_path = output / f"{method_name}_eval_metrics.csv"
    with metrics_path.open(newline="") as handle:
        metrics = next(csv.DictReader(handle))
    diagnostics.update({key: float(value) if key != "method" else value
                        for key, value in metrics.items()})
    diagnostics.update(vars(args))
    diagnostics["output_root"] = str(diagnostics["output_root"])
    with (output / f"{args.method}_run_metadata.json").open("w") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
