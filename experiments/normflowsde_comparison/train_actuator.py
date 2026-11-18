"""NormFlowSDE on the actuator force-estimation data (first test, 2026-11-16).

Stage 1 trains ``RealNVPFlowSDE`` from scratch with the manuscript objective (the filtering runner's
``filtering_loss_parts``). Stage 2 warm-starts the selected stage-1 checkpoint, adds the learned noise
scale and fine-tunes the full-horizon energy score (``improve_pendulum_normflowsde.loss_fn``), as the
paper's s1 / s2 runs do. Checkpoints are selected on the validation split; the test split is scored
once per stage with the selected checkpoint. Two deterministic references are scored on the same test
windows: holding the known initial force, and a ridge regression from the causal observation window.

The filter sees x0 and the measured signals up to the current step; later forces are scoring targets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.applications.state_estimation.flow_sde import LearnedScaleDiffusionNet, RealNVPFlowSDE
from experiments.actuator_state_estimation.dataset import load_actuator
from .cache import load_cache, save_cache
from .evaluation_metrics import metrics_from_samples
from .improve_pendulum_normflowsde import loss_fn as full_horizon_loss
from .train_normflowsde_filter import ACCESS, filtering_inputs, filtering_loss_parts

class WindowInputFlowSDE(RealNVPFlowSDE):
    """RealNVPFlowSDE whose input row k already holds its causal window y[k-w+1..k], flattened.

    The window comes from the continuous recording, so it can reach back before the first transition
    (warm-up); entries that are not available are NaN and are skipped by the encoder exactly as the base
    class skips the missing prefix. With all pre-window entries NaN this equals the base class.
    """

    def encode(self, y):
        window = y.reshape(y.shape[0], self.observation_window, -1)
        valid = ~jnp.isnan(window[..., 0])
        return jax.vmap(self.encoder.window_context)(jnp.nan_to_num(window), valid)


def window_inputs(split, width, warmup):
    """(N, T, width * d) rows of causal windows; pre-window measurements only with ``warmup``."""
    observations = np.asarray(split["observations"])
    count, steps, size = observations.shape
    past = np.full((count, width - 1, size), np.nan, np.float32)
    if warmup and width > 1:
        if "history" not in split or split["history"].shape[1] < width - 1:
            raise ValueError("The data cache holds too little pre-window history for this warm-up")
        past = np.asarray(split["history"])[:, split["history"].shape[1] - (width - 1):]
    series = np.concatenate((past, observations), axis=1)
    index = np.arange(steps)[:, None] + np.arange(width)[None]
    return series[:, index].reshape(count, steps, width * size)


SUMMARY_KEYS = ("rmse", "pooled_rmse", "crps", "energy_score", "coverage_90", "mean_interval_width",
                "calibration_mae")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True, help="Run directory; s1/ and s2/ are created inside.")
    p.add_argument("--data-cache", type=Path, default=Path("results/datasets/actuator"))
    p.add_argument("--raw", type=Path, default=Path.home() / "data/actuator_analysis",
                   help="Directory with inputs_*.npy / outputs_*.npy; used only to build a missing cache.")
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--train-stride", type=int, default=50)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--particles", type=int, default=200)
    # Architecture (REASSEMBLE layout with a smaller context; 4 coupling layers for the 2-D state).
    p.add_argument("--context-dim", type=int, default=32)
    p.add_argument("--encoder-hidden", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--hidden-depth", type=int, default=3)
    p.add_argument("--flow-layers", type=int, default=4)
    p.add_argument("--diffusion-hidden-size", type=int, default=64)
    p.add_argument("--observation-window", type=int, default=10)
    p.add_argument("--no-time", action="store_true", help="Time-invariant drift and noise flow (use_time=False).")
    p.add_argument("--warmup", action="store_true",
                   help="Fill the encoder window with the measurements recorded just before each window "
                        "instead of starting with an empty history.")
    p.add_argument("--history", type=int, default=0,
                   help="Pre-window measurements stored when a missing cache is built (>= observation-window - 1 "
                        "for --warmup).")
    # Stage 1 (manuscript objective).
    p.add_argument("--stage1-steps", type=int, default=6000)
    p.add_argument("--stage1-batch", type=int, default=16)
    p.add_argument("--stage1-learning-rate", type=float, default=6e-4)
    p.add_argument("--segment-length", type=int, default=11)
    p.add_argument("--training-draws", type=int, default=4)
    p.add_argument("--rollout-every", type=int, default=5)
    p.add_argument("--rollout-weight", type=float, default=1.0)
    p.add_argument("--continuity-weight", type=float, default=1.0)
    p.add_argument("--state-sigma", type=float, default=1.0)
    # Stage 2 (full-horizon energy score, learned noise scale).
    p.add_argument("--stage2-steps", type=int, default=4000)
    p.add_argument("--stage2-batch", type=int, default=16)
    p.add_argument("--stage2-learning-rate", type=float, default=3e-4)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--select-metric", choices=("rmse", "crps"), default="rmse")
    p.add_argument("--validation-every", type=int, default=250)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-norm", type=float, default=1.0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny budgets; a code check, not an accuracy run.")
    args = p.parse_args(argv)
    if args.smoke:
        args.stage1_steps, args.stage2_steps, args.validation_every, args.particles = 20, 10, 10, 8
    return args


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def parameter_count(model):
    return int(sum(leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))))


def make_sampler(particles, chunk=64):
    @eqx.filter_jit
    def draw(model, initial, observations, controls, key):
        return jax.vmap(lambda x0, y, u, k: model.sample_posterior(
            filtering_inputs(y, u), x0, k, n_particles=particles, solver_name="euler"))(
                initial, observations, controls, jr.split(key, initial.shape[0]))

    def sample(model, split, seed, limit=None):
        """(windows, particles, horizon, d_x) draws; only x0 and the measurements enter the model."""
        count = len(split["states"]) if limit is None else min(limit, len(split["states"]))
        outputs = []
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            outputs.append(np.asarray(draw(model, jnp.asarray(split["states"][start:stop, 0]),
                jnp.asarray(split["observations"][start:stop]), jnp.asarray(split["controls"][start:stop]),
                jr.fold_in(jr.PRNGKey(seed), start))))
        samples = np.concatenate(outputs)
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite posterior draws")
        return samples, np.asarray(split["states"][:count, 1:])
    return sample


def quick_scores(samples, truth):
    """Validation selection scores in normalized units (per-window RMSE mean, marginal CRPS, 90% coverage)."""
    error = samples.mean(axis=1) - truth
    ordered = np.sort(samples, axis=1)
    draws = samples.shape[1]
    ranks = (2 * np.arange(1, draws + 1) - draws - 1).reshape((1, -1, 1, 1))
    crps = np.abs(samples - truth[:, None]).mean(axis=1) - np.sum(ranks * ordered, axis=1) / draws ** 2
    low, high = np.percentile(samples, [5, 95], axis=1)
    return dict(rmse=float(np.sqrt(np.mean(error ** 2, axis=(1, 2))).mean()), crps=float(crps.mean()),
                coverage_90=float(np.mean((truth >= low) & (truth <= high))))


def full_metrics(samples, truth, data):
    """Normalized and physical (force unit) metrics of the same draws."""
    result = {}
    for space in ("normalized", "physical"):
        metrics = metrics_from_samples(samples, truth, observed_indices=[], metric_space=space,
                                       state_mean=data.metadata["state_mean"], state_std=data.metadata["state_std"])
        result[space] = dict(metrics=metrics["metrics"], per_dimension=metrics["per_dimension"],
                             calibration=metrics["diagnostics"]["calibration"])
    return result


def point_metrics(prediction, truth, std):
    error = prediction - truth
    per_window = lambda e: float(np.sqrt(np.mean(e ** 2, axis=(1, 2))).mean())
    return dict(rmse_normalized=per_window(error), rmse_physical=per_window(error * std),
                per_dimension_rmse_physical=np.sqrt(np.mean((error * std) ** 2, axis=(0, 1))).tolist())


def baselines(data, window):
    """Deterministic references on the test windows (same targets as the filter)."""
    std = np.asarray(data.metadata["state_std"])

    def features(observations):  # causal window ending at the current step, zero-padded at the start
        padded = np.concatenate((np.zeros((observations.shape[0], window - 1, observations.shape[2]), np.float32),
                                 observations), axis=1)
        stacked = np.stack([padded[:, lag:lag + observations.shape[1]] for lag in range(window)], axis=2)
        flat = stacked.reshape(-1, window * observations.shape[2])
        return np.concatenate((flat, np.ones((len(flat), 1), np.float32)), axis=1).astype(np.float64)

    design, target = features(data.train["observations"]), data.train["states"][:, 1:].reshape(-1, std.size)
    weights = np.linalg.solve(design.T @ design + 1e-3 * np.eye(design.shape[1]), design.T @ target)
    truth = data.test["states"][:, 1:]
    ridge = (features(data.test["observations"]) @ weights).reshape(truth.shape)
    hold = np.repeat(data.test["states"][:, :1], truth.shape[1], axis=1)
    return dict(hold_initial_force=point_metrics(hold, truth, std),
                ridge_observation_window=dict(point_metrics(ridge, truth, std), window=window, ridge=1e-3),
                zero_mean_predictor=point_metrics(np.zeros_like(truth), truth, std))


def run_stage(name, model, data, args, output, *, steps, batch, schedule, loss, sample, report):
    """Shared loop: update, validate every ``validation_every``, keep the best checkpoint, score test once."""
    output.mkdir(parents=True)
    optimizer = optax.chain(optax.clip_by_global_norm(args.grad_norm),
                            optax.adamw(schedule, weight_decay=args.weight_decay))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    training = {key: jnp.asarray(data.train[key]) for key in ("states", "observations", "controls")}

    @eqx.filter_jit
    def update(candidate, state, x, y, u, key, active):
        (value, parts), grads = eqx.filter_value_and_grad(loss, has_aux=True)(candidate, x, y, u, key, active)
        updates, state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), state, value, optax.global_norm(grads), parts

    def validate(candidate):
        return quick_scores(*sample(candidate, data.validation, 77001))

    report = dict(report, stage=name, status="running", parameter_count=parameter_count(model),
                  devices=[str(device) for device in jax.devices()])
    write_json(output / "run_metadata.json", report)
    started = time.perf_counter()
    initial = validate(model)
    best, best_step, best_scores = initial[args.select_metric], 0, initial
    eqx.tree_serialise_leaves(output / "model.eqx", model)
    print(json.dumps(dict(stage=name, step=0, **initial)), flush=True)
    with (output / "history.jsonl").open("w") as history:
        for step in range(steps):
            batch_key, loss_key = jr.split(jr.fold_in(jr.PRNGKey(args.seed + (100 if name == "s1" else 700)), step))
            ids = jr.randint(batch_key, (batch,), 0, len(training["states"]))
            active = bool(args.rollout_weight and (step + 1) % args.rollout_every == 0)
            model, opt_state, value, gradient_norm, parts = update(
                model, opt_state, *(training[key][ids] for key in ("states", "observations", "controls")),
                loss_key, active)
            row = dict(step=step + 1, loss=float(value), gradient_norm=float(gradient_norm),
                       **{key: float(item) for key, item in parts.items()})
            if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
                raise FloatingPointError(f"Nonfinite loss/gradient at {name} step {step + 1}")
            if (step + 1) % args.validation_every == 0 or step + 1 == steps:
                scores = validate(model)
                row.update({f"validation_{key}": item for key, item in scores.items()},
                           elapsed_seconds=time.perf_counter() - started)
                if scores[args.select_metric] < best:
                    best, best_step, best_scores = scores[args.select_metric], step + 1, scores
                    eqx.tree_serialise_leaves(output / "model.eqx", model)
                print(json.dumps(dict(stage=name, **row)), flush=True)
            history.write(json.dumps(row) + "\n")
    model = eqx.tree_deserialise_leaves(output / "model.eqx", model)
    samples, truth = sample(model, data.test, 99001)
    metrics = full_metrics(samples, truth, data)
    np.savez_compressed(output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    write_json(output / "metrics.json", metrics)
    report.update(status="complete", training_seconds=time.perf_counter() - started, best_step=best_step,
                  initial_validation=initial, best_validation=best_scores, select_metric=args.select_metric,
                  test={space: {key: metrics[space]["metrics"][key] for key in SUMMARY_KEYS} for space in metrics})
    write_json(output / "run_metadata.json", report)
    return model, report


def main(argv=None):
    args = parse_args(argv)
    if not args.allow_cpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU required; pass --allow-cpu to run on CPU")
    if args.output.exists():
        raise FileExistsError(f"Preserve the existing run; choose a new output: {args.output}")
    if not args.data_cache.exists():
        save_cache(load_actuator(args.raw, horizon=args.horizon, train_stride=args.train_stride, dt=args.dt,
                                 num_particles=args.particles, history=args.history), args.data_cache)
    data = load_cache(args.data_cache)
    if data.protocol.get("dataset") != "actuator" or data.protocol["horizon"] != args.horizon:
        raise ValueError("Data cache is not the requested actuator cache")
    d_x, d_obs = data.train["states"].shape[-1], data.train["observations"].shape[-1]
    model_config = dict(d_x=d_x, d_y=d_obs, d_u=0, d_c=args.context_dim, d_hidden=args.hidden_size,
                        depth=args.hidden_depth, n_realnvp_layers=args.flow_layers,
                        observation_window=args.observation_window, d_hidden_enc=args.encoder_hidden,
                        d_hidden_diffusion=args.diffusion_hidden_size, direct_control=False,
                        observation_stride=1, time_step=float(data.dt), flow_type="realnvp")
    if args.no_time:
        model_config["use_time"] = False
    model = WindowInputFlowSDE(**model_config, key=jr.PRNGKey(args.seed))
    reference = baselines(data, args.observation_window)  # on the plain measurements, before windowing
    for split in (data.train, data.validation, data.test):
        split["observations"] = window_inputs(split, args.observation_window, args.warmup)
    sample = make_sampler(args.particles)
    shapes = {name: {key: list(value.shape) for key, value in getattr(data, name).items()}
              for name in ("train", "validation", "test")}
    report = dict(dataset="actuator", method="normflowsde", training_seed=args.seed, inference_access=ACCESS,
                  known_initial_state=True, intermediate_truth_resets_at_evaluation=False, post_calibration=False,
                  encoder_history=("measurements before the window (warm-up)" if args.warmup else "empty at window start"),
                  smoke=args.smoke, solver="euler",
                  arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  model_config=model_config, data=dict(protocol=data.protocol, shapes=shapes,
                                                       segments=data.metadata["segments"]))
    args.output.mkdir(parents=True)
    write_json(args.output / "baselines.json", reference)

    recipe = dict(solver="euler", segment_length=args.segment_length, training_draws=args.training_draws,
                  rollout_draws=args.training_draws, state_sigma=args.state_sigma,
                  continuity_weight=args.continuity_weight, rollout_weight=args.rollout_weight)
    model, first = run_stage(
        "s1", model, data, args, args.output / "s1", steps=args.stage1_steps, batch=args.stage1_batch,
        schedule=optax.constant_schedule(args.stage1_learning_rate),
        loss=lambda m, x, y, u, key, active: filtering_loss_parts(m, x, y, u, key, recipe, active),
        sample=sample, report=dict(report, recipe=recipe, objective="manuscript segment objective"))

    # Stage 2: zero-initialized learned noise scale in front of the flow (step-0 model unchanged).
    wrapped = LearnedScaleDiffusionNet(model.diffusion, model_config["d_c"], model_config["d_hidden_diffusion"],
                                       model.diffusion.layers[0].time_emb.dim, jr.fold_in(jr.PRNGKey(args.seed), 3),
                                       use_time=not args.no_time)
    model = eqx.tree_at(lambda m: m.diffusion, model, wrapped)
    stage2 = argparse.Namespace(draws=args.draws, objective="energy", energy_beta=1., energy_spread_weight=.5,
                                mean_mse_weight=0.)
    ones = jnp.ones(d_x)  # score in normalized units, the cache's metric space
    schedule = optax.warmup_cosine_decay_schedule(0., args.stage2_learning_rate, min(200, args.stage2_steps // 10 + 1),
                                                  args.stage2_steps, args.stage2_learning_rate * .1)
    model, second = run_stage(
        "s2", model, data, args, args.output / "s2", steps=args.stage2_steps, batch=args.stage2_batch,
        schedule=schedule, loss=lambda m, x, y, u, key, active: full_horizon_loss(m, x, y, u, key, ones, stage2),
        sample=sample, report=dict(report, model_config=dict(model_config, learned_scale=True), warm_start="s1",
                                   objective="full-horizon per-time energy score, 16 draws"))
    summary = dict(seed=args.seed, smoke=args.smoke, windows=data.metadata["selected_windows"],
                   baselines=json.loads((args.output / "baselines.json").read_text()),
                   **{stage["stage"]: dict(best_step=stage["best_step"], validation=stage["best_validation"],
                                           test=stage["test"], parameters=stage["parameter_count"],
                                           training_seconds=stage["training_seconds"]) for stage in (first, second)})
    write_json(args.output / "summary.json", summary)
    print(json.dumps(dict(status="complete", output=str(args.output))), flush=True)


if __name__ == "__main__":
    main()
