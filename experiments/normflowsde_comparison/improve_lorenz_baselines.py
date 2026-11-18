"""Accuracy fine-tuning for the NF-DPF, NSF and FlowSDE baselines (Lorenz96, pendulum).

Implements the variants proposed in
results/results_arxiv/lorenz96_baseline_accuracy_audit_20261018 (proposed_training_plan.json):

* warm start from the same-seed campaign checkpoint (which stays a candidate:
  step 0 is scored and can be selected);
* real multi-trajectory batches for NSF/FlowSDE (the campaign runner trained on
  max(1, 64 // 100) = 1 trajectory per update);
* horizon curricula ending in full 100-step windows anchored at the known x0;
* NSF/FlowSDE: proper-score (fair CRPS) + ensemble-mean MSE on generated
  recursive rollouts, with the original objective kept as a weighted anchor;
* NF-DPF: longer differentiable filtering windows, more training particles and
  a weighted-particle CRPS term next to the filtering-mean MSE; optionally
  (2026-10-19, defaults off) an ESS weight-collapse penalty, a reweighted or
  disabled mean-MSE term, rematerialized filter steps, and skipping nonfinite
  updates. ``remat`` is the one opt-in keyword added to the library filter;
  with defaults the library code path is unchanged;
* checkpoint selection on all 200 validation windows.

Inference access is unchanged: NF-DPF stays causal filtering, NSF/FlowSDE keep
their whole-window observation encoders, every evaluation starts from the known
x0 without truth resets. The test split is scored once, after training, with
the campaign protocol (first 100 windows, seed 99001, 200 draws, 1000 PF
particles). Frozen campaign sources are imported, never modified.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.evaluation_metrics import empirical_crps, metrics_from_samples
from experiments.normflowsde_comparison.run import access_label, score_samples
from experiments.nf_dpf_state_estimation.run import sample_training_windows

METHODS = ("nf_dpf", "nsf", "flowsde")
DATASETS = ("lorenz96", "pendulum", "aloha", "reassemble")
DEFAULT_HORIZONS = {"lorenz96": {"nf_dpf": "8,25,50,100", "nsf": "10,25,50,100", "flowsde": "10,25,50,100"},
                    "pendulum": {"nf_dpf": "8,25,60", "nsf": "10,25,60", "flowsde": "10,25,60"},
                    "aloha": {"nf_dpf": "8,25,50,250", "nsf": "10,50,125,250", "flowsde": "10,50,125,250"},
                    "reassemble": {"nf_dpf": "8,25,50,250", "nsf": "10,50,125,250", "flowsde": "10,50,125,250"}}
DEFAULT_BATCHES = {"nf_dpf": "32,16,8,8", "nsf": "16", "flowsde": "16"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dataset", default="lorenz96", choices=DATASETS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-cache", required=True, type=Path)
    parser.add_argument("--init-run", required=True, type=Path,
                        help="Campaign run directory with run_metadata.json and model.eqx.")
    parser.add_argument("--from-scratch", action="store_true",
                        help="Use the campaign architecture but not its trained weights.")
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--horizons", help="Comma-separated curriculum horizons (equal step shares).")
    parser.add_argument("--batches", help="Trajectories per update; one value or one per stage.")
    parser.add_argument("--draws", type=int, default=8, help="NSF/FlowSDE rollout ensemble size.")
    parser.add_argument("--rollout-weight", type=float, default=1.)
    parser.add_argument("--mean-mse-weight", type=float, default=1.,
                        help="Ensemble/filter-mean MSE weight (NSF/FlowSDE rollouts and NF-DPF).")
    parser.add_argument("--anchor-weight", type=float, default=.1,
                        help="Weight of the method's original objective (NSF NLL+consistency, "
                             "FlowSDE flow matching+regularizers, NF-DPF component NLL).")
    parser.add_argument("--epr-weight", type=float,
                        help="FlowSDE stationary-EPR weight override (audit ablation; 0 disables).")
    parser.add_argument("--training-particles", type=int, default=64)
    parser.add_argument("--crps-weight", type=float, default=1., help="NF-DPF weighted-particle CRPS.")
    parser.add_argument("--ess-weight", type=float, default=0.,
                        help="NF-DPF weight-collapse penalty: mean over steps of log N - log ESS.")
    parser.add_argument("--measurement-nll-weight", type=float, default=0.,
                        help="NF-DPF (opt-in, 2026-10-20): weight of the supervised measurement NLL "
                             "-log q(y[k] | x[k+1]) per observation coordinate (labels used in training only).")
    parser.add_argument("--measurement-supervised-only", action="store_true",
                        help="NF-DPF (opt-in, 2026-10-20): stop the filtering-loss gradient into the measurement "
                             "density, so it is trained by its supervised NLL only (a proper likelihood, as for "
                             "the NSF-PF/FlowSDE-PF measurement flows); transition and proposal stay end-to-end.")
    parser.add_argument("--measurement-batch-windows", type=int, default=0,
                        help="NF-DPF (opt-in, 2026-10-20): evaluate the measurement NLL on this many random full "
                             "training windows per update (as the NSF-PF/FlowSDE-PF measurement flows: 16) instead "
                             "of the filtering batch; 0 = use the filtering batch.")
    parser.add_argument("--remat", action="store_true",
                        help="NF-DPF: recompute filter steps in the backward pass (memory for long windows).")
    parser.add_argument("--skip-nonfinite", action="store_true",
                        help="Discard updates with a nonfinite loss/gradient instead of stopping the run.")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--validation-windows", type=int, default=200)
    parser.add_argument("--validation-draws", type=int, default=32)
    parser.add_argument("--eval-chunk", type=int, default=25)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny CPU check; not an accuracy run.")
    args = parser.parse_args(argv)
    args.horizons = [int(v) for v in (args.horizons or DEFAULT_HORIZONS[args.dataset][args.method]).split(",")]
    batches = [int(v) for v in (args.batches or DEFAULT_BATCHES[args.method]).split(",")]
    if len(batches) == 1:
        batches *= len(args.horizons)
    if len(batches) != len(args.horizons) or min(batches + args.horizons) < 1:
        parser.error("batches must give one positive value or one per horizon")
    args.batches = batches
    if args.draws < 2 or args.steps < 1:
        parser.error("draws >= 2 and steps >= 1 required")
    return args


# ----------------------------------------------------------------------------
# Model construction


def load_model(args):
    metadata = json.loads((args.init_run / "run_metadata.json").read_text())
    if metadata["method"] != args.method or metadata["dataset"] != args.dataset:
        raise ValueError("init-run does not match the requested dataset and method")
    config = dict(metadata["model_config"])
    key = jr.PRNGKey(args.seed)
    if args.method == "nf_dpf":
        from bayesinf.applications.state_estimation.normalizing_flow_dpf import NFDPFConfig, NormalizingFlowDPF
        model = NormalizingFlowDPF(NFDPFConfig(**config), key=key)
    else:
        from bayesinf.applications.state_estimation.observation_conditioned_baselines import (
            ObservationConditionedConfig, ObservationConditionedFlowSDE, ObservationConditionedNSF)
        if args.method == "flowsde" and args.epr_weight is not None:
            config["epr_weight"] = args.epr_weight
        cls = ObservationConditionedNSF if args.method == "nsf" else ObservationConditionedFlowSDE
        model = cls(ObservationConditionedConfig(**config), key=key)
    if not args.from_scratch:
        # Static configuration fields are not serialized, so overriding a loss
        # weight keeps the checkpoint's leaves loadable.
        model = eqx.tree_deserialise_leaves(args.init_run / "model.eqx", model)
    return model, metadata


# ----------------------------------------------------------------------------
# Generated rollouts (normalized coordinates, differentiable)


def rollout(model, method, states, observations, controls, start, horizon, key, draws):
    """`draws` recursive paths from states[start] for `horizon` steps -> (M,H,D).

    Contexts are encoded from the complete observation window exactly as at
    inference; physical times keep their absolute window positions.
    """
    config = model.config
    if method == "nsf":
        conditions = jnp.concatenate((controls, model.encode(observations, controls)), axis=-1)
    else:
        conditions = model._conditions(observations, controls)
    conditions = jax.lax.dynamic_slice_in_dim(conditions, start, horizon)
    indices = start + jnp.arange(horizon)
    initial = jnp.broadcast_to(states[start], (draws, config.state_dim))

    if method == "nsf":
        def step(current, inputs):
            index, condition = inputs
            following = model.transition.sample(
                jr.fold_in(key, index), current, config.dt, condition,
                num_samples=1, t_current=config.dt * index.astype(current.dtype))[0]
            return following, following
    else:
        sub_dt = config.dt / config.flow_substeps
        noise_scale = jnp.sqrt(2 * config.diffusion * sub_dt)

        def step(current, inputs):
            index, condition = inputs

            def substep(sub_index, state):
                time = (index * config.flow_substeps + sub_index) * sub_dt
                drift = jax.vmap(lambda x: model.dynamics(x, time, control=condition))(state)
                noise = jr.normal(jr.fold_in(key, index * config.flow_substeps + sub_index),
                                  state.shape, dtype=state.dtype)
                return state + sub_dt * drift + noise_scale * noise

            following = jax.lax.fori_loop(0, config.flow_substeps, substep, current)
            return following, following

    _, path = jax.lax.scan(step, initial, (indices, conditions))
    return jnp.swapaxes(path, 0, 1)


def fair_crps(samples, truth):
    """Unbiased (distinct-pair) ensemble CRPS, averaged over all coordinates."""
    count = samples.shape[0]
    accuracy = jnp.mean(jnp.abs(samples - truth[None]), axis=0)
    spread = jnp.sum(jnp.abs(samples[:, None] - samples[None, :]), axis=(0, 1)) / (count * (count - 1))
    return jnp.mean(accuracy - .5 * spread)


def weighted_crps(particles, weights, truth):
    """CRPS of a weighted particle marginal, averaged over coordinates."""
    accuracy = jnp.sum(weights[:, None] * jnp.abs(particles - truth[None]), axis=0)
    pair = weights[:, None, None] * weights[None, :, None] * jnp.abs(particles[:, None] - particles[None, :])
    return jnp.mean(accuracy - .5 * jnp.sum(pair, axis=(0, 1)))


# ----------------------------------------------------------------------------
# Objectives


def generated_rollout_loss(model, batch, key, horizon, args):
    states, observations, controls = batch
    start_key, draw_key, anchor_key = jr.split(key, 3)
    length = observations.shape[1]
    starts = jr.randint(start_key, (states.shape[0],), 0, length - horizon + 1)
    keys = jr.split(draw_key, states.shape[0])
    paths = jax.vmap(lambda x, y, u, s, k: rollout(model, args.method, x, y, u, s, horizon, k, args.draws))(
        states, observations, controls, starts, keys)
    targets = jax.vmap(lambda x, s: jax.lax.dynamic_slice_in_dim(x, s + 1, horizon))(states, starts)
    crps = jnp.mean(jax.vmap(fair_crps)(paths, targets))
    mse = jnp.mean((paths.mean(axis=1) - targets) ** 2)
    anchor = (model.loss_parts(states, observations, controls, key=anchor_key).total
              if args.anchor_weight else jnp.zeros_like(mse))
    total = args.rollout_weight * (crps + args.mean_mse_weight * mse) + args.anchor_weight * anchor
    return total, dict(rollout_crps=crps, rollout_mean_mse=mse, anchor=anchor)


def nf_dpf_window_loss(model, batch, key, horizon, args):
    from bayesinf.applications.state_estimation.normalizing_flow_dpf import nf_dpf_component_loss
    states, observations, controls = batch
    keys = jr.split(key, states.shape[0])
    filter_model = model
    if args.measurement_supervised_only:
        frozen = jax.tree_util.tree_map(lambda v: jax.lax.stop_gradient(v) if eqx.is_array(v) else v,
                                        model.measurement)
        filter_model = eqx.tree_at(lambda m: m.measurement, model, frozen)

    def one(x, y, u, k):
        result = filter_model.filter(y, x[0], u, key=k, num_particles=args.training_particles,
                              resampling="ot", initial_std=0., remat=args.remat)
        crps = jax.vmap(weighted_crps)(result.particles, result.weights, x[1:])
        # log N - log ESS: zero for uniform weights, log N when one particle holds all mass.
        deficit = jnp.log(args.training_particles) - jnp.log(result.ess)
        return jnp.mean((result.mean - x[1:]) ** 2), jnp.mean(crps), jnp.mean(deficit), jnp.mean(result.ess)

    mse, crps, deficit, ess = (jnp.mean(v) for v in jax.vmap(one)(states, observations, controls, keys))
    nll = (nf_dpf_component_loss(model, states, observations, controls)
           if args.anchor_weight else jnp.zeros_like(mse))
    total = (args.mean_mse_weight * mse + args.crps_weight * crps + args.anchor_weight * nll
             + args.ess_weight * deficit)
    parts = dict(filter_mse=mse, filter_crps=crps, component_nll=nll, ess_deficit=deficit, mean_ess=ess)
    if args.measurement_nll_weight or args.measurement_supervised_only:
        if args.measurement_batch_windows:
            ids = jr.randint(jr.fold_in(key, 7), (args.measurement_batch_windows,), 0,
                             args.measurement_pairs["states"].shape[0])
            pair_states = args.measurement_pairs["states"][ids][:, 1:]
            pair_observations = args.measurement_pairs["observations"][ids]
        else:
            pair_states, pair_observations = states[:, 1:], observations
        measurement_nll = (-jnp.mean(model.measurement.log_prob(pair_observations, pair_states))
                           / observations.shape[-1])
        total = total + args.measurement_nll_weight * measurement_nll
        parts["measurement_nll"] = measurement_nll
    return total, parts


def training_batch(training, key, horizon, batch, method):
    if method == "nf_dpf":
        return sample_training_windows(training, key, batch * horizon, horizon)
    ids = jr.randint(key, (batch,), 0, len(training["states"]))
    return [training[name][ids] for name in ("states", "observations", "controls")]


# ----------------------------------------------------------------------------
# Evaluation, mirroring run.draw_samples key usage


def draw_samples(model, method, states, observations, controls, key, draws, particles):
    if method == "nf_dpf":
        filter_key, draw_key = jr.split(key)
        result = model.filter(observations, states[0], controls, key=filter_key,
                              num_particles=particles, initial_std=0., resampling="systematic")
        keys = jr.split(draw_key, observations.shape[0])
        picked = jax.vmap(lambda k, p, w: p[jr.categorical(k, jnp.log(w), shape=(draws,))])(
            keys, result.particles, result.weights)
        return picked.swapaxes(0, 1)
    return model.sample(observations, states[0], controls, key=key, num_samples=draws)


def evaluate(model, method, split, count, seed, draws, particles, chunk):
    @eqx.filter_jit
    def infer(candidate, states, observations, controls, keys):
        return jax.vmap(lambda x, y, u, k: draw_samples(candidate, method, x, y, u, k, draws, particles))(
            states, observations, controls, keys)

    count = min(count, len(split["states"]))
    outputs = []
    for begin in range(0, count, chunk):
        index = np.arange(begin, min(begin + chunk, count))
        keys = jnp.stack([jr.fold_in(jr.PRNGKey(seed), int(i)) for i in index])
        outputs.append(np.asarray(infer(model, *(jnp.asarray(split[name][index])
                                                  for name in ("states", "observations", "controls")), keys)))
    samples = np.concatenate(outputs)
    if not np.isfinite(samples).all():
        raise FloatingPointError("Nonfinite generated states")
    return samples, np.asarray(split["states"][:count, 1:])


def physical_crps(samples, truth, data):
    std = np.asarray(data.metadata["state_std"])
    return float(np.mean(empirical_crps(samples * std, truth * std, axis=1)))


# ----------------------------------------------------------------------------


def main(argv=None):
    args = parse_args(argv)
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_cache(args.data_cache)
    if data.protocol.get("dataset") != args.dataset:
        raise ValueError("Data cache does not match --dataset")
    model, init_metadata = load_model(args)
    if args.smoke:
        args.validation_windows, args.validation_draws, args.eval_chunk = 4, 4, 2
    training = {name: jnp.asarray(data.train[name]) for name in ("states", "observations", "controls")}
    args.measurement_pairs = training if args.measurement_batch_windows else None
    val_particles = data.protocol["num_particles"] if not args.smoke else 16

    schedule = optax.warmup_cosine_decay_schedule(
        0., args.learning_rate, min(200, args.steps // 10 + 1), args.steps, args.learning_rate * .1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-5))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    loss_fn = nf_dpf_window_loss if args.method == "nf_dpf" else generated_rollout_loss

    @eqx.filter_jit
    def update(candidate, state, batch, key, horizon):
        (loss, parts), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            candidate, batch, key, horizon, args)
        gradient_norm = optax.global_norm(grads)
        updates, new_state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
        new_candidate = eqx.apply_updates(candidate, updates)
        if args.skip_nonfinite:  # keep the previous parameters and optimizer state for a bad batch
            finite = jnp.isfinite(loss) & jnp.isfinite(gradient_norm)
            keep = lambda new, old: jnp.where(finite, new, old) if eqx.is_array(new) else new
            new_candidate = jax.tree_util.tree_map(keep, new_candidate, candidate)
            new_state = jax.tree_util.tree_map(keep, new_state, state)
        return new_candidate, new_state, loss, gradient_norm, parts

    def validate(candidate):
        samples, truth = evaluate(candidate, args.method, data.validation, args.validation_windows,
                                  77001, args.validation_draws, val_particles, args.eval_chunk)
        scores = score_samples(samples, truth, data)
        return scores["rmse"], physical_crps(samples, truth, data), scores["coverage_90"]

    stage_bounds = np.linspace(0, args.steps, len(args.horizons) + 1).round().astype(int)
    report = dict(status="running", method=args.method, seed=args.seed, init_run=str(args.init_run),
                  warm_start=not args.from_scratch, inference_access=access_label(args.method, getattr(model.config, "encoder_direction", "bidirectional")),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                             if k != "measurement_pairs"},
                  model_config=dataclasses.asdict(model.config),
                  parameter_count=int(sum(x.size for x in jax.tree_util.tree_leaves(
                      eqx.filter(model, eqx.is_inexact_array)))),
                  curriculum=[dict(horizon=h, trajectories_per_update=b, steps=int(stage_bounds[i + 1] - stage_bounds[i]))
                              for i, (h, b) in enumerate(zip(args.horizons, args.batches))],
                  devices=[str(d) for d in jax.devices()])
    if report["parameter_count"] != init_metadata.get("parameter_count", report["parameter_count"]):
        raise ValueError("Architecture differs from the campaign checkpoint")
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")

    started = time.perf_counter()
    rmse, crps, coverage = validate(model)
    best = rmse
    report.update(initial_validation=dict(rmse=rmse, crps=crps, coverage_90=coverage),
                  best_step=0, best_validation_rmse=rmse, best_validation_crps=crps)
    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
    print(json.dumps(dict(step=0, validation_rmse=rmse, validation_crps=crps, coverage_90=coverage)), flush=True)
    trajectories = transitions = skipped = 0
    with (args.output / "history.jsonl").open("w") as history:
        history.write(json.dumps(dict(step=0, validation_rmse=rmse, validation_crps=crps,
                                      validation_coverage_90=coverage)) + "\n")
        for step in range(args.steps):
            stage = int(np.searchsorted(stage_bounds, step, side="right") - 1)
            horizon, batch_size = args.horizons[stage], args.batches[stage]
            batch_key, loss_key = jr.split(jr.fold_in(jr.PRNGKey(args.seed + 500), step))
            batch = training_batch(training, batch_key, horizon, batch_size, args.method)
            model, optimizer_state, loss, gradient_norm, parts = update(
                model, optimizer_state, batch, loss_key, horizon)
            trajectories += batch_size
            transitions += batch_size * horizon
            row = dict(step=step + 1, horizon=horizon, loss=float(loss), gradient_norm=float(gradient_norm),
                       **{name: float(value) for name, value in parts.items()})
            if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
                if not args.skip_nonfinite:
                    raise FloatingPointError(f"Nonfinite loss/gradient at step {step + 1}")
                skipped += 1
                row["skipped_nonfinite"] = True
                if skipped > max(10, args.steps // 100):
                    raise FloatingPointError(f"{skipped} nonfinite updates by step {step + 1}")
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                rmse, crps, coverage = validate(model)
                row.update(validation_rmse=rmse, validation_crps=crps, validation_coverage_90=coverage)
                if rmse < best:
                    best = rmse
                    report.update(best_step=step + 1, best_validation_rmse=rmse, best_validation_crps=crps)
                    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
                row["best_validation_rmse"] = best
                row["elapsed_seconds"] = time.perf_counter() - started
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + "\n")
            history.flush()
    report.update(training_seconds=time.perf_counter() - started, sampled_trajectories=trajectories,
                  sampled_transition_targets=transitions, skipped_nonfinite_updates=skipped)

    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    if not args.skip_test:
        count = data.protocol["eval_windows"] if not args.smoke else 2
        samples, truth = evaluate(model, args.method, data.test, count, 99001,
                                  data.protocol["posterior_samples"],
                                  data.protocol["num_particles"] if not args.smoke else 16, args.eval_chunk)
        metrics = score_samples(samples, truth, data)
        proper = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                      metric_space=data.protocol["metric_space"],
                                      state_mean=data.metadata["state_mean"],
                                      state_std=data.metadata["state_std"], energy_method="exact")
        metrics.update(proper["metrics"])
        metrics.update(eval_windows=len(truth), posterior_samples=samples.shape[1],
                       inference_access=access_label(args.method, getattr(model.config, "encoder_direction", "bidirectional")))
        np.savez_compressed(args.output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        report["test"] = {k: metrics[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width")
                          if k in metrics}
    report["status"] = "complete"
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(dict(
        method=args.method, seed=args.seed, best_step=report["best_step"],
        initial_validation=report["initial_validation"], best_validation_rmse=report["best_validation_rmse"],
        best_validation_crps=report["best_validation_crps"], test=report.get("test")), indent=2) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.output), test=report.get("test"))), flush=True)


if __name__ == "__main__":
    main()
