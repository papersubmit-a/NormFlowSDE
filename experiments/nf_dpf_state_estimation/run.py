"""Train and evaluate NF-DPF using the existing state-estimation datasets.

The full test trajectory receives only its initial state and causal observations.
Short training windows are drawn exclusively from the training partition.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bayesinf.applications.state_estimation.normalizing_flow_dpf import (
    NFDPFConfig, NormalizingFlowDPF, nf_dpf_component_loss, nf_dpf_loss,
)
from bayesinf.applications.state_estimation.utils.plots import save_state_estimation_plots
from .datasets import load_aloha, load_reassemble
from .synthetic import load_lorenz96, load_pendulum


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        choices=("pendulum", "lorenz96", "aloha", "reassemble"))
    parser.add_argument("--variant", default="dataset_noisy", choices=(
        "dataset_noisy", "mujoco_random", "pseudo_process_uncertainty",
        "pseudo_process_uncertainty_gmm"))
    parser.add_argument("--noise-model", choices=("gaussian", "gmm"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-particles", type=int)
    parser.add_argument("--posterior-samples", type=int)
    parser.add_argument("--eval-windows", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--hidden-depth", type=int)
    parser.add_argument("--flow-layers", type=int)
    parser.add_argument("--training-window", type=int, default=8)
    parser.add_argument("--training-particles", type=int, default=32)
    parser.add_argument("--warmup-fraction", type=float, default=.2)
    parser.add_argument("--nll-weight", type=float, default=.01)
    parser.add_argument("--resampling", choices=("ot", "soft", "systematic"), default="ot")
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--smoke", action="store_true",
                        help="Two optimization steps and two test windows; not benchmark evidence.")
    parser.add_argument("--evaluate-only", action="store_true",
                        help="Evaluate the best checkpoint in --output on the same dataset.")
    args = parser.parse_args(argv)
    if not 0 <= args.warmup_fraction < 1:
        parser.error("warmup-fraction must lie in [0, 1)")
    if args.nll_weight < 0:
        parser.error("nll-weight must be nonnegative")
    for name in ("training_window", "training_particles", "validation_every", "validation_windows"):
        if getattr(args, name) < 1:
            parser.error(f"{name.replace('_', '-')} must be positive")
    return args


def load_data(args):
    overrides = {name: getattr(args, name) for name in (
        "seed", "train_steps", "batch_size", "learning_rate", "num_particles",
        "eval_windows", "hidden_size", "hidden_depth", "flow_layers",
    ) if getattr(args, name) is not None}
    if args.dataset == "pendulum":
        data = load_pendulum(noise_model=args.noise_model or "gaussian", **overrides)
    elif args.dataset == "lorenz96":
        data = load_lorenz96(noise_model=args.noise_model or "gmm", **overrides)
    elif args.dataset == "aloha":
        if args.noise_model is not None:
            raise ValueError("Aloha noise is fixed by --variant")
        data = load_aloha(args.variant, **overrides)
    else:
        if args.noise_model not in (None, "gaussian"):
            raise ValueError("Reassemble uses the existing Gaussian observation protocol")
        data = load_reassemble(**overrides)
    if args.posterior_samples is not None:
        data.protocol["posterior_samples"] = args.posterior_samples
        data.protocol.setdefault("overrides", {})["posterior_samples"] = args.posterior_samples
        data.protocol["matched_defaults"] = False
    if args.smoke:
        data.protocol.update(train_steps=2, batch_size=8, hidden_size=8,
                             hidden_depth=1, flow_layers=2, num_particles=16,
                             posterior_samples=16, eval_windows=2)
        args.training_window = 4
        args.training_particles = 8
        args.validation_windows = 1
        args.validation_every = 1
        data.protocol["overrides"].update({name: data.protocol[name] for name in (
            "train_steps", "batch_size", "hidden_size", "hidden_depth", "flow_layers",
            "num_particles", "posterior_samples", "eval_windows")})
        data.protocol["matched_defaults"] = False
    for name in ("train_steps", "batch_size", "hidden_size", "hidden_depth", "flow_layers",
                 "num_particles", "posterior_samples", "eval_windows"):
        if data.protocol[name] < 1:
            raise ValueError(f"{name} must be positive")
    if data.protocol["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive")
    return data


def fingerprint(split):
    """Hash actual input arrays, so comparisons can verify identical data."""
    digest = hashlib.sha256()
    for name in ("states", "observations", "controls"):
        array = np.ascontiguousarray(split[name], dtype=np.float32)
        digest.update(name.encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def sample_training_windows(split, key, batch_size, window):
    """A batch contains batch_size transition targets (rounded down to windows)."""
    count, horizon = split["observations"].shape[:2]
    window = min(window, horizon)
    sequences = max(1, batch_size // window)
    sequence_key, time_key = jax.random.split(key)
    sequence_ids = jax.random.randint(sequence_key, (sequences,), 0, count)
    start = jax.random.randint(time_key, (sequences,), 0, horizon - window + 1)
    times = start[:, None] + jnp.arange(window)[None, :]
    state_times = start[:, None] + jnp.arange(window + 1)[None, :]
    return (split["states"][sequence_ids[:, None], state_times],
            split["observations"][sequence_ids[:, None], times],
            split["controls"][sequence_ids[:, None], times])


def train(model, data, args):
    protocol = data.protocol
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        protocol["learning_rate"], weight_decay=1e-5))
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    split = {name: jnp.asarray(data.train[name]) for name in (
        "states", "observations", "controls")}
    warmup = max(1, int(protocol["train_steps"] * args.warmup_fraction)) if args.warmup_fraction else 0
    # Always include at least one end-to-end step, including smoke runs.
    warmup = min(warmup, protocol["train_steps"] - 1)

    @eqx.filter_jit
    def update(candidate, optimizer_state, states, observations, controls, key, pretrain):
        def objective(current):
            if pretrain:
                return nf_dpf_component_loss(current, states, observations, controls)
            return nf_dpf_loss(current, states, observations, controls, key=key,
                               num_particles=args.training_particles,
                               resampling=args.resampling, nll_weight=args.nll_weight,
                               initial_std=protocol.get("initial_std", 0.))
        loss, gradients = eqx.filter_value_and_grad(objective)(candidate)
        updates, optimizer_state = optimizer.update(
            gradients, optimizer_state, eqx.filter(candidate, eqx.is_inexact_array))
        grad_norm = optax.global_norm(gradients)
        return eqx.apply_updates(candidate, updates), optimizer_state, loss, grad_norm

    state_std = jnp.asarray(data.metadata["state_std"])
    if protocol.get("metric_space") == "normalized":
        state_std = jnp.ones_like(state_std)

    @eqx.filter_jit
    def validate(candidate, observations, initial, controls, truth, key):
        result = candidate.filter(observations, initial, controls, key=key,
                                  num_particles=args.training_particles,
                                  initial_std=protocol.get("initial_std", 0.),
                                  resampling="systematic")
        return jnp.sqrt(jnp.mean(((result.mean - truth) * state_std) ** 2))

    @eqx.filter_jit
    def draw(key):
        return sample_training_windows(split, key, protocol["batch_size"], args.training_window)

    started = time.perf_counter()
    history, best_score, best_step = [], float("inf"), 0
    key = jax.random.PRNGKey(protocol["seed"] + 2)
    for step in range(protocol["train_steps"]):
        batch_key, filter_key = jax.random.split(jax.random.fold_in(key, step))
        batch = draw(batch_key)
        model, state, loss, grad_norm = update(model, state, *batch, filter_key, step < warmup)
        loss_value, gradient_value = float(loss), float(grad_norm)
        if not np.isfinite((loss_value, gradient_value)).all():
            raise FloatingPointError(f"nonfinite loss or gradient at step {step + 1}")
        row = {"step": step + 1, "stage": "density_warmup" if step < warmup else "filter",
               "loss": loss_value, "gradient_norm": gradient_value}
        if step == 0 or (step + 1) % args.validation_every == 0 or step + 1 == protocol["train_steps"]:
            scores = []
            validation = data.validation
            for index in range(min(args.validation_windows, len(validation["states"]))):
                scores.append(float(validate(
                    model, jnp.asarray(validation["observations"][index]),
                    jnp.asarray(validation["states"][index, 0]),
                    jnp.asarray(validation["controls"][index]),
                    jnp.asarray(validation["states"][index, 1:]),
                    jax.random.fold_in(jax.random.PRNGKey(protocol["seed"] + 77), index))))
            row["validation_rmse"] = float(np.mean(scores))
            if not np.isfinite(row["validation_rmse"]):
                raise FloatingPointError(f"nonfinite validation metric at step {step + 1}")
            # Only end-to-end checkpoints are eligible for the final estimator.
            if step >= warmup and row["validation_rmse"] < best_score:
                best_score, best_step = row["validation_rmse"], step + 1
                eqx.tree_serialise_leaves(args.output / "model.eqx", model)
            row["elapsed_seconds"] = time.perf_counter() - started
            print(json.dumps(row), flush=True)
        history.append(row)
        with (args.output / "training_history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    return model, {"training_seconds": time.perf_counter() - started,
                   "best_validation_rmse": best_score, "best_step": best_step,
                   "warmup_steps": warmup, "end_to_end_steps": protocol["train_steps"] - warmup,
                   "transition_targets_per_step": max(1, protocol["batch_size"] // min(
                       args.training_window, split["observations"].shape[1])) * min(
                           args.training_window, split["observations"].shape[1])}


def evaluate(model, data, args):
    protocol, test = data.protocol, data.test
    count = min(protocol["eval_windows"], len(test["states"]))
    samples_count = protocol["posterior_samples"]

    @eqx.filter_jit
    def infer(candidate, observations, initial, controls, key):
        filter_key, draw_key = jax.random.split(key)
        result = candidate.filter(observations, initial, controls, key=filter_key,
                                  num_particles=protocol["num_particles"],
                                  initial_std=protocol.get("initial_std", 0.), resampling="systematic")
        # Draw from each filtering marginal; these are not smoothing paths.
        keys = jax.random.split(draw_key, observations.shape[0])
        samples = jax.vmap(lambda k, p, w: p[jax.random.categorical(
            k, jnp.log(w), shape=(samples_count,))])(keys, result.particles, result.weights)
        return samples.swapaxes(0, 1), result.mean, result.ess, result.log_evidence

    def one(index):
        return infer(model, jnp.asarray(test["observations"][index]),
                     jnp.asarray(test["states"][index, 0]), jnp.asarray(test["controls"][index]),
                     jax.random.fold_in(jax.random.PRNGKey(protocol["seed"] + 99), index))

    jax.block_until_ready(one(0))
    started = time.perf_counter()
    evaluated = [jax.tree_util.tree_map(np.asarray, one(i)) for i in range(count)]
    seconds = time.perf_counter() - started
    samples = np.stack([row[0] for row in evaluated])
    means = np.stack([row[1] for row in evaluated])
    ess = np.stack([row[2] for row in evaluated])
    log_evidence = np.asarray([row[3] for row in evaluated])
    truth = np.asarray(test["states"][:count, 1:])
    observations = np.asarray(test["observations"][:count])
    if protocol.get("metric_space") == "physical":
        mean, std = np.asarray(data.metadata["state_mean"]), np.asarray(data.metadata["state_std"])
        samples, means, truth = samples * std + mean, means * std + mean, truth * std + mean
        observations = (observations * np.asarray(data.metadata["observation_std"])
                        + np.asarray(data.metadata["observation_mean"]))
    if not all(np.isfinite(value).all() for value in (samples, means, truth, ess, log_evidence)):
        raise FloatingPointError("nonfinite test outputs")
    observed = np.asarray(protocol["observed_indices"], dtype=int)
    # The shared plotting helper expects state-aligned observation columns.
    plot_observations = np.full_like(truth, np.nan)
    plot_observations[..., observed] = observations
    save_state_estimation_plots(args.output, "nf_dpf", truth, plot_observations,
                                samples, dt=data.dt)
    sample_mean = samples.mean(axis=1)
    low, high = np.percentile(samples, [5, 95], axis=1)
    error = sample_mean - truth
    unobserved = np.setdiff1d(np.arange(truth.shape[-1]), observed)
    rmse = lambda error: float(np.sqrt(np.mean(error ** 2, axis=(1, 2))).mean())
    metrics = {"rmse": rmse(error), "mae": float(np.abs(error).mean()),
               "coverage_90": float(np.mean((truth >= low) & (truth <= high))),
               "mean_interval_width": float(np.mean(high - low)),
               "rmse_observed": rmse(error[..., observed]),
               "rmse_unobserved": rmse(error[..., unobserved]) if len(unobserved) else None,
               "weighted_mean_rmse": rmse(means - truth),
               "mean_ess": float(ess.mean()), "min_ess": float(ess.min()),
               "log_evidence_normalized_mean": float(log_evidence.mean()),
               "evaluation_seconds_excluding_compilation": seconds,
               "milliseconds_per_trajectory": 1000 * seconds / count,
               "eval_windows": count, "horizon": truth.shape[1],
               "num_particles": protocol["num_particles"], "posterior_samples": samples_count,
               "metric_space": protocol.get("metric_space", "normalized")}
    np.savez_compressed(args.output / "filter_diagnostics.npz", weighted_mean=means,
                        ess=ess, log_evidence_normalized=log_evidence)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps({"test": metrics}), flush=True)
    return metrics


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / "run_metadata.json"
    if metadata_path.exists() and not args.evaluate_only:
        raise FileExistsError(f"Refusing to overwrite an existing run: {args.output}")
    data = load_data(args)
    protocol = data.protocol
    config = NFDPFConfig(state_dim=data.train["states"].shape[-1],
                        observation_dim=data.train["observations"].shape[-1],
                        control_dim=data.train["controls"].shape[-1],
                        hidden_size=protocol["hidden_size"], hidden_depth=protocol["hidden_depth"],
                        num_flow_layers=protocol["flow_layers"])
    hashes = {name: fingerprint(getattr(data, name)) for name in ("train", "validation", "test")}
    report = {"method": "nf_dpf", "paper": "https://arxiv.org/abs/2403.01499",
              "implementation": "independent JAX/Equinox implementation",
              "status": "running", "dataset": args.dataset, "smoke": args.smoke,
              "protocol": protocol, "data_metadata": data.metadata,
              "split_sha256": hashes, "model_config": asdict(config),
              "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "inference_access": "causal, known x0 only, no intermediate truth resets",
              "training": "component density warmup, then supervised filter MSE plus component NLL",
              "resampling": {"training": args.resampling, "evaluation": "systematic"},
              "devices": [str(device) for device in jax.devices()],
              "jax_version": jax.__version__, "equinox_version": eqx.__version__}
    if args.evaluate_only:
        previous = json.loads(metadata_path.read_text())
        if hashes != previous["split_sha256"]:
            raise ValueError("Checkpoint dataset fingerprints do not match the requested data")
        config = NFDPFConfig(**previous["model_config"])
        report = previous
        report["evaluation_protocol"] = protocol
        report["evaluation_arguments"] = {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    model = NormalizingFlowDPF(config, key=jax.random.PRNGKey(protocol["seed"] + 1))
    report["parameter_count"] = sum(x.size for x in jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_inexact_array)))
    report["status"] = "running"
    report.pop("error", None)
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        if args.evaluate_only:
            model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
        else:
            model, diagnostics = train(model, data, args)
            report.update(diagnostics)
        report["test"] = evaluate(model, data, args)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
