"""Observation-access-only filtering adaptations of the saved prediction recipes.

The shared cache stores observations[k] as a noisy measurement of states[k+1].
This runner feeds that measurement directly to transition k in both optimization
and inference. The unchanged bounded encoder can use only observations[:k+1].
The resulting kernel is a learned observation-conditioned filtering update, not
an autonomous SDE transition density. Training never performs test inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
from . import run as common
from . import train_lorenz_heun as heun
from .evaluation_metrics import metrics_from_samples
from .lorenz_heun_objective import loss_components_single
from .tracking import JournalTracker, validate_uri


VARIANT = "normflowsde-filtering-v1"
SELECTION_SCHEMA = "normflowsde-filtering-test-v1"
ACCESS = "filtering: x[k+1] uses actual noisy observations through x[k+1], bounded causal window; no future observations"
EXTRA_SOURCES = (
    "experiments/normflowsde_comparison/train_normflowsde_filter.py",
    "experiments/normflowsde_comparison/evaluation_metrics.py",
    "experiments/normflowsde_comparison/lorenz_heun_objective.py",
    "experiments/normflowsde_comparison/train_lorenz_heun.py",
    "experiments/normflowsde_comparison/lorenz_heun_array.sh",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=("pendulum", "lorenz96", "aloha", "reassemble"))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data-cache", required=True, type=Path)
    p.add_argument("--reference-metadata", required=True, type=Path)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--expected-manifest", type=Path)
    p.add_argument("--selection", type=Path)
    p.add_argument("--selection-sha256")
    p.add_argument("--test-output", type=Path)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke-updates", type=int,
                   help="Explicit CPU smoke only: short budget, batch 2, validation 2 x 4 draws")
    p.add_argument("--tracking-uri")
    p.add_argument("--tracking-experiment", default="normflowsde-comparison")
    p.add_argument("--tracking-flush-every", type=int, default=25)
    # Explicit, recorded deviations from the reference recipe (filtering benchmark campaign).
    p.add_argument("--train-steps", type=int, help="Override the optimizer-update budget (Lorenz: epochs follow).")
    p.add_argument("--validation-windows", type=int, help="Override checkpoint-selection validation windows.")
    p.add_argument("--segment-length", type=int, help="Override the objective's segment length (state points).")
    p.add_argument("--continuity-weight", type=float, help="Override the continuity-term weight.")
    p.add_argument("--rollout-weight", type=float, help="Override the full-path energy (rollout) weight.")
    p.add_argument("--state-sigma", type=float, help="Override the state-likelihood sigma.")
    p.add_argument("--learning-rate", type=float,
                   help="Opt-in (2026-10-21): override the peak learning rate (update budget unchanged).")
    p.add_argument("--batch-trajectories", type=int,
                   help="Opt-in (2026-10-21): override trajectories per update (update budget unchanged).")
    p.add_argument("--grad-norm", type=float,
                   help="Opt-in (2026-10-21): override the global gradient-norm clipping threshold.")
    p.add_argument("--weight-decay", type=float,
                   help="Opt-in (2026-10-21): override the AdamW weight decay.")
    p.add_argument("--patience", type=int,
                   help="Opt-in (2026-10-21): override the Lorenz recipe's early-stopping patience in epochs "
                        "(0 disables early stopping so the whole update budget is used).")
    p.add_argument("--realnvp-alpha", type=float, help="Override the RealNVP coupling log-scale bound alpha.")
    p.add_argument("--observation-window", type=int,
                   help="Opt-in (2026-10-21): override the encoder's causal observation window (parameter count unchanged).")
    p.add_argument("--no-time", action="store_true", help="Time-invariant drift and noise flow (use_time=False).")
    p.add_argument("--learned-scale", action="store_true", help="Learned per-coordinate noise scale.")
    p.add_argument("--allow-data-change", action="store_true",
                   help="Opt-in (2026-10-20): train on a data cache that differs from the reference run's data "
                        "(e.g. GMM observation noise); the reference supplies only the recipe. Differences are recorded.")
    p.add_argument("--allow-source-drift", action="store_true",
                   help="Record, instead of refusing, reference source/package differences.")
    args = p.parse_args(argv)
    if (args.train_steps is not None and args.train_steps < 1) or (
            args.validation_windows is not None and args.validation_windows < 1):
        p.error("train-steps and validation-windows must be positive")
    if args.seed < 0 or args.tracking_flush_every < 1:
        p.error("seed must be nonnegative and tracking-flush-every positive")
    if args.smoke_updates is not None and (args.smoke_updates < 1 or not args.allow_cpu):
        p.error("smoke-updates must be positive and explicitly requires --allow-cpu")
    if args.evaluate_test and (not args.selection or not args.selection_sha256 or not args.test_output):
        p.error("--evaluate-test requires --selection, --selection-sha256 and --test-output")
    if not args.evaluate_test and (args.selection or args.selection_sha256 or args.test_output):
        p.error("selection and test-output flags are accepted only with --evaluate-test")
    validate_uri(args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"))
    return args


def scientific_arguments(args):
    return {name: value for name, value in vars(args).items() if name not in {
        "output", "data_cache", "reference_metadata", "train", "evaluate_test", "expected_manifest",
        "selection", "selection_sha256", "test_output", "tracking_uri", "tracking_experiment",
        "tracking_flush_every"}}


def preparation_manifest(report):
    result = {name: report[name] for name in (
        "source_sha256", "model_config", "scientific_arguments", "data", "reference_recipe", "recipe")}
    result["packages"] = report["environment"]["packages"]
    return result


def filtering_inputs(observations, controls):
    """Destination-aligned noisy measurements plus controls; no reference states."""
    if observations.ndim not in (2, 3) or observations.ndim != controls.ndim:
        raise ValueError("Expected observations/controls with matching time or batch-time axes")
    if observations.shape[:-1] != controls.shape[:-1]:
        raise ValueError("Observation and control sequence shapes differ")
    return jnp.concatenate((observations, controls), axis=-1)


def resolve_recipe(reference, dataset):
    """Extract and enforce the intended saved recipe without model warm starts."""
    a = reference["arguments"]
    if dataset in ("pendulum", "aloha", "reassemble"):
        if dataset == "pendulum" and (a.get("segment_length") != 11
                or reference["model_config"].get("realnvp_alpha", 1.) != 1.
                or reference.get("parameter_count") != 91014):
            raise ValueError("Pendulum reference must be the unchanged segment-11 alpha-1 recipe")
        if dataset != "pendulum" and (a.get("capacity_profile") != "legacy-aligned-v1"
                or a.get("settings_profile") != "legacy-compatible-v1"):
            raise ValueError("Robot reference must be a legacy-aligned comparison-runner NormFlowSDE run")
        protocol = reference["data"]["protocol"]
        # Comparison-runner trajectories per update (run.py full_trajectory_training_batch).
        batch = 4 if dataset == "pendulum" else reference["full_trajectory_training_batch"]
        return dict(solver="euler", train_steps=protocol["train_steps"], batch_trajectories=batch,
            learning_rate=protocol["learning_rate"], weight_decay=1e-5, grad_norm=1.,
            segment_length=a["segment_length"], training_draws=a["training_draws"],
            rollout_draws=a.get("rollout_draws", a["training_draws"]),
            rollout_every=a.get("rollout_every", 5), state_sigma=a["state_sigma"],
            continuity_weight=a["continuity_weight"], rollout_weight=a["rollout_weight"],
            validation_every=a["validation_every"], validation_windows=a["validation_windows"],
            validation_particles=protocol["posterior_samples"], posterior_samples=protocol["posterior_samples"],
            eval_windows=protocol["eval_windows"], final_validation_windows=reference["data"]["shapes"]["validation"]["states"][0],
            max_epochs=None, patience=0, warmup_steps=0, sampling="with replacement",
            initialization="from scratch, PRNGKey(training_seed); no checkpoint warm start")
    if (reference.get("solver") != "heun" or a.get("realnvp_alpha") != 1.5
            or a.get("transition_weight") != .1 or reference.get("parameter_count") != 349348):
        raise ValueError("Lorenz reference must be promoted alpha-1.5 transition-0.1 Heun recipe")
    names = ("learning_rate", "weight_decay", "grad_norm", "segment_length", "rollout_draws", "rollout_every",
             "state_sigma", "continuity_weight", "rollout_weight", "validation_windows", "validation_particles",
             "posterior_samples", "eval_windows", "max_epochs", "patience", "warmup_steps",
             "validation_every_epochs", "shooting_noise_std", "transition_weight")
    return dict({name: a[name] for name in names}, solver="heun", batch_trajectories=a["batch_size"],
        train_steps=reference["data"]["protocol"]["train_steps"],
        final_validation_windows=reference["data"]["shapes"]["validation"]["states"][0],
        sampling="epoch shuffle without replacement", initialization="from scratch, PRNGKey(training_seed); no checkpoint warm start")


def prepare(args):
    reference = json.loads(args.reference_metadata.read_text())
    if (reference.get("dataset") != args.dataset or reference.get("method") != "normflowsde"
            or reference.get("training_seed") != args.seed or reference.get("status") != "complete"):
        raise ValueError("Reference must be a completed same-seed NormFlowSDE prediction run")
    if not reference.get("inference_access", "").startswith("prediction:"):
        raise ValueError("The observation-access control must reference prediction")
    recipe = resolve_recipe(reference, args.dataset)
    overrides = {}
    if args.train_steps is not None:
        overrides["train_steps"] = recipe["train_steps"] = args.train_steps
    if args.validation_windows is not None:
        overrides["validation_windows"] = recipe["validation_windows"] = args.validation_windows
    for name in ("segment_length", "continuity_weight", "rollout_weight", "state_sigma", "learning_rate",
                 "batch_trajectories", "patience", "grad_norm", "weight_decay"):
        if getattr(args, name) is not None:
            overrides[name] = recipe[name] = getattr(args, name)
    if args.train_steps is not None and args.dataset == "lorenz96":
        # After the batch override, so the epoch cap still allows the full update budget.
        per_epoch = math.ceil(reference["data"]["shapes"]["train"]["states"][0] / recipe["batch_trajectories"])
        overrides["max_epochs"] = recipe["max_epochs"] = math.ceil(args.train_steps / per_epoch)
    if recipe["segment_length"] < 2 or recipe["state_sigma"] <= 0 or min(recipe["continuity_weight"], recipe["rollout_weight"]) < 0:
        raise ValueError("segment_length >= 2, state_sigma > 0 and nonnegative weights required")
    if recipe["learning_rate"] <= 0 or recipe["batch_trajectories"] < 1:
        raise ValueError("learning-rate must be positive and batch-trajectories at least 1")
    if recipe["patience"] < 0:
        raise ValueError("patience must be nonnegative")
    if recipe["grad_norm"] <= 0 or recipe["weight_decay"] < 0:
        raise ValueError("grad-norm must be positive and weight-decay nonnegative")
    model_overrides = dict(**({"use_time": False} if args.no_time else {}),
                           **({"learned_scale": True} if args.learned_scale else {}),
                           **({"realnvp_alpha": args.realnvp_alpha} if args.realnvp_alpha is not None else {}),
                           **({"observation_window": args.observation_window} if args.observation_window else {}))
    if args.realnvp_alpha is not None and args.realnvp_alpha <= 0:
        raise ValueError("realnvp-alpha must be positive")
    model_config = dict(reference["model_config"], **model_overrides)
    if args.dataset == "lorenz96":
        original = heun.parse_args(["--output", str(args.output), "--data-cache", str(args.data_cache)])
        for name, value in reference["arguments"].items():
            if hasattr(original, name) and name not in {"output", "data_cache", "expected_manifest", "train", "allow_cpu"}:
                setattr(original, name, value)
        model, data, report = heun.prepare(original)
        if model_overrides:
            model = RealNVPFlowSDE(**model_config, physics_fn=None, key=jax.random.PRNGKey(args.seed))
    else:
        original = common.parse_args(["--dataset", args.dataset, "--method", "normflowsde", "--output", str(args.output),
                                      "--data-cache", str(args.data_cache)])
        for name, value in reference["arguments"].items():
            if hasattr(original, name) and name not in {"output", "data_cache", "prepared_matrix", "train",
                                                      "tracking_uri", "tracking_experiment", "tracking_flush_every"}:
                setattr(original, name, value)
        if args.allow_data_change:
            cached = json.loads((Path(args.data_cache) / "cache.json").read_text())["protocol"]
            if args.dataset == "aloha":
                original.variant = cached.get("variant", original.variant)
            original.observation_noise = cached.get("obs_noise_type")
        data = common.load_data(original)
        report = common.prepare_report(data, original)
        model = RealNVPFlowSDE(**model_config, key=jax.random.PRNGKey(args.seed))
    data_changes = [name for name in ("split_sha256", "metadata", "shapes", "protocol", "dt")
                    if report["data"][name] != reference["data"][name]]
    if data_changes and not args.allow_data_change:
        raise ValueError(f"Reference dataset differs: {data_changes[0]}")
    root = Path(__file__).resolve().parents[2]
    mismatches = [name for name, digest in reference["source_sha256"].items()
                  if not (root / name).is_file() or sha256(root / name) != digest]
    if mismatches and not args.allow_source_drift:
        raise ValueError(f"Reference scientific source epoch differs: {mismatches}")
    count = sum(leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    if report["model_config"] != reference["model_config"] or (
            count != reference["parameter_count"] and not args.learned_scale):
        raise ValueError("Reference architecture or trainable parameter count differs")
    package_drift = report["environment"]["packages"] != reference["environment"]["packages"]
    if package_drift and not args.allow_source_drift:
        raise ValueError("Reference package versions differ")
    if args.smoke_updates is not None:
        recipe.update(train_steps=args.smoke_updates, batch_trajectories=2, validation_every=1,
                      validation_every_epochs=1, validation_windows=2, validation_particles=4,
                      final_validation_windows=2, posterior_samples=4, eval_windows=2,
                      warmup_steps=0, max_epochs=1 if args.dataset == "lorenz96" else None)
    source_hashes = dict(common.source_hashes())
    source_hashes.update({name: sha256(root / name) for name in EXTRA_SOURCES})
    report.update(variant=VARIANT, inference_access=ACCESS, model_config=model_config,
        parameter_count=count, source_sha256=source_hashes, recipe=recipe,
        scientific_arguments=scientific_arguments(args),
        arguments={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        reference_recipe=dict(metadata_sha256=sha256(args.reference_metadata),
            scientific_source_sha256=reference["source_sha256"],
            original_inference_access=reference["inference_access"],
            all_original_scientific_sources_match=not mismatches, source_drift=mismatches,
            data_changes=data_changes,
            package_drift=package_drift, same_model_initialization=not model_overrides,
            access_only_intervention=not (overrides or model_overrides),
            recipe_overrides=overrides, model_overrides=model_overrides, checkpoint_warm_start=False,
            preserved=("architecture, data, solver, optimizer, losses, budget, minibatch RNG and validation selection"
                       if not (overrides or model_overrides) else
                       "data, solver, optimizer, losses and minibatch RNG; overrides listed in recipe_overrides/model_overrides")),
        full_trajectory_training_batch=recipe["batch_trajectories"], solver=recipe["solver"],
        observation_alignment=dict(cached_observation_index="observations[k] measures states[k+1]",
            encoder_input="unshifted actual noisy destination measurements; unchanged bounded causal window",
            cutoff="x[k+1] uses noisy observations through x[k+1] only",
            future_observations=False, reference_states_in_encoder=False, first_input="actual noisy y1; no synthetic clean y0"),
        objective_interpretation="learned observation-conditioned filtering transition; conditional reference NLL when present is not an autonomous SDE likelihood",
        known_initial_state=True, intermediate_truth_resets_at_evaluation=False, post_calibration=False,
        comparison_caveats=["Filtering access differs from the saved prediction control.",
            "NSF and FlowSDE comparison runs retain future observation access; they are smoothing baselines.",
            "All reference loss terms are preserved, including terms that may encourage narrow distributions."],
        evaluation_split="validation", test_evaluated=False)
    if args.expected_manifest and preparation_manifest(report) != json.loads(args.expected_manifest.read_text()):
        raise ValueError("Frozen source, reference, data, model, recipe, arguments or packages changed")
    return model, data, report


def filtering_loss_parts(model, states, observations, controls, key, recipe, include_rollout):
    inputs = filtering_inputs(observations, controls)
    if recipe["solver"] == "heun":
        parts = jax.vmap(lambda x, y, k: loss_components_single(model, x, y, k,
            segment_length=recipe["segment_length"], state_sigma=recipe["state_sigma"],
            shooting_noise_std=recipe["shooting_noise_std"], rollout_active=include_rollout,
            rollout_draws=recipe["rollout_draws"]))(states, inputs, jax.random.split(key, states.shape[0]))
        parts = {name: jnp.mean(value) for name, value in parts.items()}
        total = (parts["state_nll"] + recipe["transition_weight"] * parts["transition"]
                 + recipe["continuity_weight"] * parts["continuity"] + recipe["rollout_weight"] * parts["rollout"])
    else:
        parts = jax.vmap(lambda x, y, k: model.paper_loss_components_single(x, y, k,
            seg_length=recipe["segment_length"], K=recipe["training_draws"],
            rollout_K=recipe["rollout_draws"], state_sigma=recipe["state_sigma"],
            include_rollout=include_rollout))(states, inputs, jax.random.split(key, states.shape[0]))
        parts = {name: jnp.mean(value) for name, value in parts.items()}
        total = (parts["state"] + parts["transition_entropy"] + recipe["continuity_weight"] * parts["continuity"]
                 + recipe["rollout_weight"] * parts["rollout"])
    return total, parts


def make_evaluator(data, solver):
    @eqx.filter_jit
    def infer(model, initial, observations, controls, key, particles):
        return model.sample_posterior(filtering_inputs(observations, controls), initial, key,
                                      n_particles=particles, solver_name=solver)

    def evaluate(model, split, count, particles, draw_seed):
        outputs = []
        started = time.perf_counter()
        for index in range(min(count, len(split["states"]))):
            # Only x0 is passed to the model. Later true states are scoring targets.
            output = np.asarray(infer(model, jnp.asarray(split["states"][index, 0]),
                jnp.asarray(split["observations"][index]), jnp.asarray(split["controls"][index]),
                jax.random.fold_in(jax.random.PRNGKey(draw_seed), index), particles))
            expected = (particles, split["observations"].shape[1], split["states"].shape[-1])
            if output.shape != expected or not np.isfinite(output).all():
                raise FloatingPointError(f"Invalid filtering posterior: shape={output.shape}, expected={expected}")
            outputs.append(output)
        elapsed = time.perf_counter() - started
        samples, truth = np.stack(outputs), np.asarray(split["states"][:len(outputs), 1:])
        return common.score_samples(samples, truth, data), samples, truth, elapsed
    return evaluate


def train(model, data, args, report, tracker):
    if not args.allow_cpu and not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("GPU required; --allow-cpu is only for explicit smoke validation")
    r = report["recipe"]
    total_steps = r["train_steps"]
    if args.dataset == "lorenz96":
        warmup = min(r["warmup_steps"], max(0, total_steps - 1))
        if warmup:
            schedule = optax.join_schedules([optax.linear_schedule(0., r["learning_rate"], warmup),
                optax.cosine_decay_schedule(r["learning_rate"], total_steps - warmup)], [warmup])
        else:
            schedule = optax.cosine_decay_schedule(r["learning_rate"], total_steps)
    else:
        schedule = optax.constant_schedule(r["learning_rate"])
    optimizer = optax.chain(optax.clip_by_global_norm(r["grad_norm"]),
                           optax.adamw(schedule, weight_decay=r["weight_decay"]))
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    training = {name: jnp.asarray(data.train[name]) for name in ("states", "observations", "controls")}

    @eqx.filter_jit
    def update(candidate, opt_state, x, y, u, key, active):
        (loss, parts), gradients = eqx.filter_value_and_grad(filtering_loss_parts, has_aux=True)(
            candidate, x, y, u, key, r, active)
        updates, opt_state = optimizer.update(gradients, opt_state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), opt_state, loss, optax.global_norm(gradients), parts

    evaluate = make_evaluator(data, r["solver"])
    started, best, step, best_epoch = time.perf_counter(), float("inf"), 0, 0
    report.update(status="running", trained=True, started_at=datetime.now(timezone.utc).isoformat())
    atomic_json(args.output / "run_metadata.json", report)

    def validate(epoch):
        nonlocal best, best_epoch
        scores, _, _, _ = evaluate(model, data.validation, r["validation_windows"], r["validation_particles"], 77001)
        if scores["rmse"] < best:
            best, best_epoch = scores["rmse"], epoch
            report.update(best_step=step, best_validation_rmse=best)
            if epoch is not None:
                report["best_epoch"] = epoch
            eqx.tree_serialise_leaves(args.output / "model.eqx", model)
        row = dict(step=step, validation_rmse=scores["rmse"], best_validation_rmse=best)
        if epoch is not None:
            row["epoch"] = epoch
        with (args.output / "validation.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        tracker.metrics({"validation/rmse": scores["rmse"], "validation/best_rmse": best}, step)
        report.update(completed_steps=step, elapsed_training_seconds=time.perf_counter() - started)
        atomic_json(args.output / "run_metadata.json", report)
        print(json.dumps(row), flush=True)

    with (args.output / "history.jsonl").open("x") as history:
        epochs = range(1, r["max_epochs"] + 1) if args.dataset == "lorenz96" else (None,)
        for epoch in epochs:
            if args.dataset == "lorenz96":
                order = jax.random.permutation(jax.random.fold_in(jax.random.PRNGKey(args.seed + 100), epoch),
                                               len(training["states"]))
                batches = [order[start:start + r["batch_trajectories"]] for start in range(0, len(order), r["batch_trajectories"])]
            else:
                batches = range(total_steps)
            for batch in batches:
                if step >= total_steps:
                    break
                if args.dataset == "lorenz96":
                    indices = batch
                    loss_key = jax.random.fold_in(jax.random.PRNGKey(args.seed + 200), step)
                else:
                    batch_key, loss_key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed + 100), step))
                    indices = jax.random.randint(batch_key, (r["batch_trajectories"],), 0, len(training["states"]))
                active = bool(r["rollout_weight"] and (step + 1) % r["rollout_every"] == 0)
                model, state, loss, gradient_norm, parts = update(model, state,
                    *[training[name][indices] for name in ("states", "observations", "controls")], loss_key, active)
                step += 1
                row = dict(step=step, loss=float(loss), gradient_norm=float(gradient_norm),
                           learning_rate=float(schedule(step - 1)), rollout_active=active,
                           loss_components={name: float(value) for name, value in parts.items()})
                if epoch is not None:
                    row["epoch"] = epoch
                if not np.isfinite([row["loss"], row["gradient_norm"], *row["loss_components"].values()]).all():
                    raise FloatingPointError(f"Nonfinite filtering objective/gradient at step {step}")
                history.write(json.dumps(row) + "\n")
                history.flush()
                metrics = {f"train/{name}": row[name] for name in ("loss", "gradient_norm", "learning_rate")}
                metrics.update({f"train/components/{name}": value for name, value in row["loss_components"].items()})
                metrics["train/rollout_active"] = int(active)
                if epoch is not None:
                    metrics["train/epoch"] = epoch
                tracker.metrics(metrics, step)
                if args.dataset != "lorenz96" and (step % r["validation_every"] == 0 or step == total_steps):
                    validate(None)
            if epoch is not None:
                report["completed_epochs"] = epoch
                if epoch % r["validation_every_epochs"] == 0 or epoch == r["max_epochs"] or step == total_steps:
                    validate(epoch)
                    if r["patience"] and epoch - best_epoch >= r["patience"]:
                        report["early_stopped"] = True
                        break
            if step >= total_steps:
                break
    report.update(training_seconds_including_compilation=time.perf_counter() - started, completed_steps=step)
    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    evaluate(model, data.validation, 1, r["posterior_samples"], 77000)
    _, samples, truth, elapsed = evaluate(model, data.validation, r["final_validation_windows"], r["posterior_samples"], 77001)
    scores = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
        metric_space=data.protocol["metric_space"], state_mean=data.metadata["state_mean"],
        state_std=data.metadata["state_std"])["metrics"]
    scores.update(eval_windows=len(truth), horizon=truth.shape[1], posterior_samples=samples.shape[1],
        inference_access=ACCESS, solver=r["solver"], milliseconds_per_trajectory=1000 * elapsed / len(truth))
    np.savez_compressed(args.output / "validation_posterior.npz", samples_normalized=samples, truth_normalized=truth)
    atomic_json(args.output / "validation_metrics.json", scores)
    report.update(status="complete", validation=scores, test_evaluated=False, evaluation_split="validation",
                  completed_at=datetime.now(timezone.utc).isoformat())
    tracker.metrics({f"validation_final/{name}": value for name, value in scores.items()}, step)


def verify_test_selection(args):
    if sha256(args.selection) != args.selection_sha256:
        raise ValueError("Test receipt SHA256 does not match")
    selection = json.loads(args.selection.read_text())
    if (selection.get("schema") != SELECTION_SCHEMA or selection.get("test_allowed") is not True
            or selection.get("selection_basis") != "validation_only"):
        raise ValueError("Test requires an explicit frozen validation-only completion receipt")
    expected = {f"{dataset}/seed{seed}" for dataset in ("pendulum", "lorenz96") for seed in range(3)}
    if set(selection.get("runs", {})) != expected:
        raise ValueError("Test receipt must cover all six dataset/seed identities")
    proof = selection["runs"][f"{args.dataset}/seed{args.seed}"]
    if Path(proof["train_output"]).resolve() != args.output.resolve():
        raise ValueError("Training path differs from test receipt")
    if proof["checkpoint_sha256"] != sha256(args.output / "model.eqx"):
        raise ValueError("Selected checkpoint SHA256 does not match")
    if proof["metadata_sha256"] != sha256(args.output / "run_metadata.json"):
        raise ValueError("Selected training metadata SHA256 does not match")
    return selection


def evaluate_test(args):
    selection = verify_test_selection(args)  # Guard before loading data or any test inference.
    saved = json.loads((args.output / "run_metadata.json").read_text())
    if (saved.get("status") != "complete" or saved.get("test_evaluated") is not False
            or saved.get("variant") != VARIANT or saved.get("dataset") != args.dataset
            or saved.get("training_seed") != args.seed):
        raise ValueError("Test requires the completed matching validation-only filtering run")
    model, data, current = prepare(args)
    if preparation_manifest(current) != preparation_manifest(saved):
        raise ValueError("Test source/data/reference/recipe/packages differ from training")
    if args.test_output.exists():
        raise FileExistsError(f"Preserve existing test output: {args.test_output}")
    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    r = saved["recipe"]
    evaluate = make_evaluator(data, r["solver"])
    evaluate(model, data.validation, 1, r["posterior_samples"], 99000)
    _, samples, truth, elapsed = evaluate(model, data.test, r["eval_windows"], r["posterior_samples"], 99001)
    scores = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
        metric_space="physical", state_mean=data.metadata["state_mean"], state_std=data.metadata["state_std"])["metrics"]
    scores.update(eval_windows=len(truth), horizon=truth.shape[1], posterior_samples=samples.shape[1],
        inference_access=ACCESS, solver=r["solver"], post_calibration=False,
        milliseconds_per_trajectory=1000 * elapsed / len(truth))
    args.test_output.mkdir(parents=True)
    np.savez_compressed(args.test_output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    atomic_json(args.test_output / "metrics.json", scores)
    atomic_json(args.test_output / "run_metadata.json", dict(saved, test=scores, test_evaluated=True, evaluation_split="test",
        checkpoint_sha256=sha256(args.output / "model.eqx"),
        selection_sha256=args.selection_sha256, selection=selection, evaluated_at=datetime.now(timezone.utc).isoformat()))


def main(argv=None):
    args = parse_args(argv)
    if args.evaluate_test:
        evaluate_test(args)
        return
    if args.output.exists():
        raise FileExistsError(f"Preserve existing run: {args.output}")
    model, data, report = prepare(args)
    args.output.mkdir(parents=True)
    atomic_json(args.output / "run_metadata.json", report)
    if args.train:
        tracker = JournalTracker(args.output, report, uri=args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"),
            experiment=args.tracking_experiment, flush_every=args.tracking_flush_every)
        handlers = {}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                handlers[signum] = signal.signal(signum, interrupted)
            train(model, data, args, report, tracker)
        except BaseException as error:
            report.update(status="killed" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for signum, handler in handlers.items():
                signal.signal(signum, handler)
            atomic_json(args.output / "run_metadata.json", report)
            tracker.finish({"complete": "FINISHED", "killed": "KILLED"}.get(report["status"], "FAILED"), report)
    print(json.dumps(dict(status=report["status"], output=str(args.output), variant=VARIANT)), flush=True)


if __name__ == "__main__":
    main()
