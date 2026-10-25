"""Prepare, or explicitly train, a manuscript-aligned state-estimation run.

Preparation is the default. Training requires --train. Existing historical
checkpoints and paper evidence are never used as new comparison results.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import os
import signal
import subprocess
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.nf_dpf_state_estimation.datasets import load_aloha, load_reassemble
from experiments.nf_dpf_state_estimation.synthetic import load_lorenz96, load_pendulum
from experiments.nf_dpf_state_estimation.run import fingerprint, sample_training_windows
from experiments.normflowsde_comparison.tracking import JournalTracker, validate_uri


METHODS = ("normflowsde", "nf_dpf", "nsf", "flowsde")
DATASETS = ("lorenz96", "pendulum", "aloha", "reassemble")
ACCESS = {
    "normflowsde": "prediction: x[k+1] uses observations through x[k], bounded preceding window",
    "nf_dpf": "filtering: x[k+1] incorporates its current measurement y[k+1]",
    "nsf": "smoothing: whole-sequence observation encoder, direct trajectory sampling",
    "flowsde": "smoothing: whole-sequence observation encoder, direct SDE integration",
}
CAUSAL_ACCESS = {
    "nsf": "filtering: x[k+1] uses observations through x[k+1] via a causal forward-only encoder; "
           "no future observations; direct trajectory sampling",
    "flowsde": "filtering: x[k+1] uses observations through x[k+1] via a causal forward-only encoder; "
               "no future observations; direct SDE integration",
}


def access_label(method, encoder_direction="bidirectional"):
    """NSF/FlowSDE access follows their encoder; other methods have fixed access."""
    if method in CAUSAL_ACCESS and encoder_direction == "causal":
        return CAUSAL_ACCESS[method]
    return ACCESS[method]
SOURCE_PATHS = (
    "requirements.txt",
    "paper/sections/method.tex",
    "paper/sections/related_work.tex",
    "bayesinf/applications/state_estimation/__init__.py",
    "bayesinf/applications/state_estimation/flow_sde.py",
    "bayesinf/applications/state_estimation/neural_stochastic_flow.py",
    "bayesinf/applications/state_estimation/fm_flow_sde.py",
    "bayesinf/applications/state_estimation/normalizing_flow_dpf.py",
    "bayesinf/applications/state_estimation/observation_conditioned_baselines.py",
    "bayesinf/applications/state_estimation/stochastic_interpolant.py",
    "bayesinf/models/normalizing_flow.py",
    "bayesinf/models/flow_matching.py",
    "bayesinf/models/score_based_diffusion_model.py",
    "bayesinf/models/bayesian_neural_network.py",
    "bayesinf/models/variational_autoencoder.py",
    "bayesinf/models/denoising_diffusion_model.py",
    "bayesinf/models/networks.py",
    "bayesinf/utils/generate_data.py",
    "experiments/nf_dpf_state_estimation/datasets.py",
    "experiments/nf_dpf_state_estimation/synthetic.py",
    "experiments/aloha_state_estimation/utils/dataset.py",
    "experiments/aloha_state_estimation/run_nsf_fm_flowsde.py",
    "experiments/reassemble_state_estimation/dataset.py",
    "experiments/lorenz96_state_estimation/utils/dataset.py",
    "bayesinf/applications/state_estimation/controlled_systems.py",
    "bayesinf/applications/state_estimation/stochastic_solvers.py",
    "experiments/nf_dpf_state_estimation/run.py",
    "experiments/normflowsde_comparison/run.py",
    "experiments/normflowsde_comparison/tracking.py",
    "experiments/normflowsde_comparison/cache.py",
    "experiments/normflowsde_comparison/prepare.py",
    "experiments/normflowsde_comparison/capacity.py",
    "experiments/normflowsde_comparison/alignment.py",
    "experiments/normflowsde_comparison/prepare_matched.py",
    "experiments/normflowsde_comparison/summarize.py",
    "experiments/normflowsde_comparison/evaluation_metrics.py",
    "experiments/normflowsde_comparison/pendulum_normflowsde_objective.py",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--data-cache", type=Path,
                        help="Read frozen, hash-verified benchmark arrays instead of regenerating data.")
    parser.add_argument("--tracking-uri", help="MLflow HTTP(S) server; defaults to MLFLOW_TRACKING_URI. "
                        "If unset, journal locally for later replay.")
    parser.add_argument("--tracking-experiment", help="MLflow experiment; defaults to "
                        "MLFLOW_EXPERIMENT_NAME or normflowsde-comparison.")
    parser.add_argument("--tracking-flush-every", type=int, default=25,
                        help="Upload interval in optimizer steps; every step is journaled locally.")
    parser.add_argument("--train", action="store_true",
                        help="Explicitly opt in to optimization and final test evaluation.")
    parser.add_argument("--validation-only", action="store_true",
                        help="After training, save the best checkpoint's posterior on the complete "
                             "validation split; defer test scoring until hyperparameters are selected.")
    parser.add_argument("--prepared-matrix", type=Path,
                        help="Verify resolved data, source hashes and settings against a prepared run.")
    parser.add_argument("--capacity-profile", choices=("matched-v1", "legacy-aligned-v1", "legacy-aligned-causal-v1"),
                        help="Verify the named parameter budget using explicit matching architecture flags.")
    parser.add_argument("--settings-profile", choices=("audited", "legacy-compatible-v1"),
                        help="Record and verify an explicit historical-settings transfer profile.")
    parser.add_argument("--seed", type=int, default=0, help="Model/optimizer seed; never changes data.")
    parser.add_argument("--data-seed", type=int,
                        help="Defaults to the established dataset seed (pendulum 2026, others 0).")
    parser.add_argument("--variant", default="dataset_noisy", choices=(
        "dataset_noisy", "mujoco_random", "pseudo_process_uncertainty",
        "pseudo_process_uncertainty_gmm"))
    parser.add_argument("--noise-model", choices=("gaussian", "gmm"))
    parser.add_argument("--observation-noise", choices=("gaussian", "gmm"),
                        help="Opt-in observation noise for pendulum/REASSEMBLE (pendulum --noise-model is process noise).")
    parser.add_argument("--train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--posterior-samples", type=int)
    parser.add_argument("--eval-windows", type=int)
    parser.add_argument("--num-particles", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--hidden-depth", type=int)
    parser.add_argument("--flow-layers", type=int)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--encoder-direction", choices=("bidirectional", "causal"), default="bidirectional",
                        help="NSF/FlowSDE observation encoder: bidirectional (smoothing) or causal (filtering).")
    parser.add_argument("--diffusion-hidden-size", type=int,
                        help="NormFlowSDE noise-flow width; defaults to its drift width.")
    parser.add_argument("--observation-window", type=int, default=10)
    parser.add_argument("--segment-length", type=int, default=11,
                        help="NormFlowSDE state points per segment, including initial state.")
    parser.add_argument("--training-draws", type=int, default=4)
    parser.add_argument("--rollout-draws", type=int,
                        help="NormFlowSDE full-path energy ensemble; defaults to training-draws.")
    parser.add_argument("--rollout-every", type=int, default=5,
                        help="NormFlowSDE full-path energy cadence in optimizer updates.")
    parser.add_argument("--state-sigma", type=float, default=1.)
    parser.add_argument("--continuity-weight", type=float, default=1.)
    parser.add_argument("--rollout-weight", type=float, default=1.)
    parser.add_argument("--consistency-weight", type=float, default=.1)
    parser.add_argument("--diffusion", type=float, default=.01)
    parser.add_argument("--flow-substeps", type=int, default=4)
    parser.add_argument("--epr-weight", type=float, default=1e-3)
    parser.add_argument("--hjb-weight", type=float, default=1e-3)
    parser.add_argument("--orthogonality-weight", type=float, default=1e-3)
    parser.add_argument("--training-particles", type=int, default=32)
    parser.add_argument("--warmup-fraction", type=float, default=.2)
    parser.add_argument("--nll-weight", type=float, default=.01)
    parser.add_argument("--validation-every", type=int, default=250)
    parser.add_argument("--validation-windows", type=int, default=8)
    parser.add_argument("--pendulum-objective", choices=("proper_rollout", "reference_nll"),
                        help="Opt-in pendulum NormFlowSDE proper-score training variant.")
    parser.add_argument("--validation-metric", choices=("rmse", "crps"), default="rmse",
                        help="Checkpoint selection score; CRPS uses normalized state coordinates.")
    args = parser.parse_args(argv)
    try:
        validate_uri(args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"))
    except ValueError as error:
        parser.error(str(error))
    for name in ("context_dim", "observation_window", "flow_substeps",
                 "training_particles", "validation_every", "validation_windows", "tracking_flush_every",
                 "rollout_every"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.segment_length < 2 or args.training_draws < 2:
        parser.error("segment-length and training-draws must be at least two")
    if args.rollout_draws is not None and args.rollout_draws < 2:
        parser.error("rollout-draws must be at least two")
    if args.diffusion_hidden_size is not None and args.diffusion_hidden_size < 1:
        parser.error("diffusion-hidden-size must be positive")
    if args.state_sigma <= 0 or args.diffusion <= 0:
        parser.error("state-sigma and diffusion must be positive")
    for name in ("continuity_weight", "rollout_weight", "consistency_weight", "nll_weight",
                 "epr_weight", "hjb_weight", "orthogonality_weight"):
        if getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("warmup-fraction must lie in [0, 1)")
    if args.pendulum_objective is not None and (args.dataset, args.method) != ("pendulum", "normflowsde"):
        parser.error("pendulum-objective requires dataset pendulum and method normflowsde")
    return args


def load_data(args):
    overrides = {name: getattr(args, name) for name in (
        "train_steps", "batch_size", "learning_rate", "num_particles", "eval_windows",
        "hidden_size", "hidden_depth", "flow_layers") if getattr(args, name) is not None}
    if args.data_seed is not None:
        overrides["seed"] = args.data_seed
    if args.data_cache is not None:
        from experiments.normflowsde_comparison.cache import load_cache
        data = load_cache(args.data_cache)
        if data.protocol.get("dataset") != args.dataset:
            raise ValueError("Cached dataset does not match --dataset")
        expected_seed = args.data_seed if args.data_seed is not None else (2026 if args.dataset == "pendulum" else 0)
        if data.protocol["seed"] != expected_seed:
            raise ValueError("Cached dataset seed does not match --data-seed")
        if args.dataset == "aloha":
            if data.protocol.get("variant") != args.variant or args.noise_model is not None:
                raise ValueError("Cached ALOHA variant/noise does not match requested settings")
        elif data.protocol.get("noise_model", "gaussian") != (args.noise_model or ("gmm" if args.dataset == "lorenz96" else "gaussian")):
            raise ValueError("Cached noise model does not match --noise-model")
        if (getattr(args, "observation_noise", None) is not None
                and data.protocol.get("obs_noise_type", "gaussian") != args.observation_noise):
            raise ValueError("Cached observation noise does not match --observation-noise")
        # Cache identity fixes data generation and splits. Only training and
        # evaluation budget/architecture settings can be changed after loading.
        changed = {key: value for key, value in overrides.items()
                   if key != "seed" and value != data.protocol.get(key)}
        recorded = dict(data.protocol.get("overrides", {}))
        recorded.update(changed)
        data.protocol.update({key: value for key, value in overrides.items() if key != "seed"})
        if changed:
            data.protocol["overrides"] = recorded
            data.protocol["matched_defaults"] = False
    elif args.dataset == "pendulum":
        data = load_pendulum(noise_model=args.noise_model or "gaussian",
                             observation_noise=getattr(args, "observation_noise", None) or "gaussian", **overrides)
    elif args.dataset == "lorenz96":
        data = load_lorenz96(noise_model=args.noise_model or "gmm", **overrides)
    elif args.dataset == "aloha":
        if args.noise_model is not None:
            raise ValueError("ALOHA noise is determined by --variant")
        data = load_aloha(args.variant, **overrides)
    else:
        if args.noise_model not in (None, "gaussian"):
            raise ValueError("REASSEMBLE uses Gaussian observation corruption")
        data = load_reassemble(obs_noise_type=getattr(args, "observation_noise", None) or "gaussian", **overrides)
    if args.posterior_samples is not None:
        data.protocol["posterior_samples"] = args.posterior_samples
    data.protocol.setdefault("adapter_initial_std", data.protocol.get("initial_std", 0.))
    data.protocol["initial_std"] = 0.
    data.protocol["inference_access"] = "method-specific; see run inference_access"
    for name in ("train_steps", "batch_size", "num_particles", "posterior_samples",
                 "eval_windows", "hidden_size", "hidden_depth", "flow_layers"):
        if data.protocol[name] < 1:
            raise ValueError(f"{name} must be positive")
    if data.protocol["learning_rate"] <= 0:
        raise ValueError("learning-rate must be positive")
    return data


def preceding_observations(states, observations, metadata):
    """Destination-aligned measurements -> preceding measurements, without truth leakage.

    Only x0 is read from reference states. The first context is its known clean
    observed coordinates, because these adapters do not provide a measured y0.
    Later contexts are actual noisy measurements. This shift is essential:
    observations[:, k] in the shared data measures states[:, k+1].
    """
    indices = jnp.asarray(metadata["observation_indices"], dtype=jnp.int32)
    physical_initial = states[:, 0] * jnp.asarray(metadata["state_std"]) + jnp.asarray(metadata["state_mean"])
    initial = (physical_initial[:, indices] - jnp.asarray(metadata["observation_mean"])) / jnp.asarray(metadata["observation_std"])
    return jnp.concatenate((initial[:, None], observations[:, :-1]), axis=1)


def model_config(data, args):
    protocol = data.protocol
    common = dict(state_dim=data.train["states"].shape[-1],
                  observation_dim=data.train["observations"].shape[-1],
                  control_dim=data.train["controls"].shape[-1])
    if args.method == "nf_dpf":
        return dict(**common, hidden_size=protocol["hidden_size"],
                    hidden_depth=protocol["hidden_depth"], num_flow_layers=protocol["flow_layers"])
    if args.method == "normflowsde":
        return dict(d_x=common["state_dim"], d_y=common["observation_dim"] + common["control_dim"],
                    d_u=common["control_dim"], d_c=args.context_dim,
                    d_hidden=protocol["hidden_size"], depth=protocol["hidden_depth"],
                    n_realnvp_layers=protocol["flow_layers"], observation_window=args.observation_window,
                    d_hidden_enc=args.context_dim,
                    d_hidden_diffusion=(args.diffusion_hidden_size if args.diffusion_hidden_size is not None
                                        else protocol["hidden_size"]),
                    direct_control=True, observation_stride=1, time_step=data.dt,
                    flow_type="realnvp")
    return dict(**common, dt=data.dt, encoder_hidden=args.context_dim,
                context_dim=args.context_dim, hidden_size=protocol["hidden_size"],
                hidden_depth=protocol["hidden_depth"], flow_layers=protocol["flow_layers"],
                consistency_weight=args.consistency_weight, diffusion=args.diffusion,
                flow_substeps=args.flow_substeps, use_potential=True,
                epr_weight=args.epr_weight, hjb_weight=args.hjb_weight,
                orthogonality_weight=args.orthogonality_weight,
                **({"encoder_direction": "causal"} if getattr(args, "encoder_direction", "bidirectional") == "causal" else {}))


def build_model(data, args):
    config, key = model_config(data, args), jax.random.PRNGKey(args.seed)
    if args.method == "normflowsde":
        from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
        return RealNVPFlowSDE(**config, key=key)
    if args.method == "nf_dpf":
        from bayesinf.applications.state_estimation.normalizing_flow_dpf import NFDPFConfig, NormalizingFlowDPF
        return NormalizingFlowDPF(NFDPFConfig(**config), key=key)
    from bayesinf.applications.state_estimation.observation_conditioned_baselines import (
        ObservationConditionedConfig, ObservationConditionedNSF, ObservationConditionedFlowSDE)
    cls = ObservationConditionedNSF if args.method == "nsf" else ObservationConditionedFlowSDE
    return cls(ObservationConditionedConfig(**config), key=key)


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


def dataset_manifest(data):
    return dict(protocol=data.protocol, metadata=data.metadata, dt=data.dt,
                split_sha256={name: fingerprint(getattr(data, name))
                              for name in ("train", "validation", "test")},
                shapes={name: {key: list(np.shape(value)) for key, value in getattr(data, name).items()}
                        for name in ("train", "validation", "test")})


def prepare_report(data, args):
    settings_alignment = None
    if args.settings_profile is not None:
        from .alignment import alignment_metadata
        settings_alignment = alignment_metadata(args.dataset, args.method, args.settings_profile)
        if settings_alignment["applied"]:
            actual = {name: getattr(args, name) for name in settings_alignment["settings"]}
            if actual != settings_alignment["settings"] or args.segment_length != 5:
                raise ValueError("Historical settings profile differs from the explicit run arguments")
    versions = {}
    for package in ("jax", "jaxlib", "equinox", "optax", "diffrax", "numba", "numpy",
                    "scipy", "h5py", "pyarrow", "matplotlib", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                              capture_output=True, check=False).stdout.strip()
    report = dict(status="prepared", trained=False, dataset=args.dataset, method=args.method,
                training_seed=args.seed, data_seed=data.protocol["seed"],
                arguments={key: str(value) if isinstance(value, Path) else value
                           for key, value in vars(args).items()
                           if not ((key in ("capacity_profile", "settings_profile", "diffusion_hidden_size", "rollout_draws", "pendulum_objective")
                                    and value is None) or (key == "rollout_every" and value == 5)
                                   or (key == "validation_only" and not value)
                                   or (key == "validation_metric" and value == "rmse")
                                   or (key == "encoder_direction" and value == "bidirectional"))},
                model_config=model_config(data, args), data=dataset_manifest(data),
                inference_access=access_label(args.method, getattr(args, "encoder_direction", "bidirectional")),
                known_initial_state=True, intermediate_truth_resets_at_evaluation=False,
                full_trajectory_training_batch=max(1, data.protocol["batch_size"] // data.protocol["horizon"]),
                training_window=8 if args.method == "nf_dpf" else data.protocol["horizon"],
                normflowsde_rollout_every=args.rollout_every,
                comparison_caveats=[
                    "Prediction, filtering and smoothing use different observation information; report access next to every score.",
                    "Update budgets follow the existing datasets, not an equal-FLOP or equal-example budget.",
                    "Hyperparameters for new encoder-conditioned adaptations require validation; defaults are not published best settings.",
                    "NormFlowSDE initial context is known clean x0; subsequent contexts use shifted actual measurements.",
                    "NF-DPF saved draws are filtering marginals, not coherent trajectory posterior paths."],
                source_sha256=source_hashes(), git_revision=revision,
                environment=dict(python=platform.python_version(), executable=sys.executable,
                                 platform=platform.platform(),
                                 packages=versions, devices=[str(d) for d in jax.devices()]))
    if settings_alignment is not None:
        report["settings_alignment"] = settings_alignment
    if getattr(args, "pendulum_objective", None) is not None:
        from .pendulum_normflowsde_objective import objective_definition
        report["objective_definition"] = objective_definition(
            args.pendulum_objective, rollout_every=args.rollout_every,
        )
    return report


def verify_prepared_report(report, args):
    """Fail before writing outputs if a frozen command no longer resolves identically."""
    if args.prepared_matrix is None:
        return
    matrix = json.loads(args.prepared_matrix.read_text())
    matches = [row for row in matrix["runs"] if (row["dataset"], row["method"], row["training_seed"])
               == (args.dataset, args.method, args.seed)]
    if len(matches) != 1:
        raise ValueError("Prepared matrix must identify exactly one requested run")
    expected = matches[0]
    manifest = json.loads((args.prepared_matrix.parent / expected["data_manifest"]).read_text())
    for name in ("source_sha256", "model_config"):
        if report[name] != expected[name]:
            raise ValueError(f"Prepared {name} changed; audit and prepare a new matrix")
    if report["environment"]["packages"] != expected["environment"]["packages"]:
        raise ValueError("Prepared package versions changed; use the recorded environment or prepare a new matrix")
    if report["data"] != manifest:
        raise ValueError("Prepared dataset arrays/protocol changed; prepare a new matrix")
    actual_arguments, expected_arguments = dict(report["arguments"]), dict(expected["arguments"])
    # --train controls execution, not the experiment definition.
    actual_arguments.pop("train")
    expected_arguments.pop("train")
    # Tracking endpoints and upload cadence do not change model/data semantics.
    for name in ("tracking_uri", "tracking_experiment", "tracking_flush_every"):
        actual_arguments.pop(name, None)
        expected_arguments.pop(name, None)
    if actual_arguments != expected_arguments:
        raise ValueError("Run arguments differ from the prepared experiment")


def normflowsde_rollout_active(args, step):
    """Whether a zero-based optimizer step includes the full-path energy term."""
    return args.rollout_weight != 0. and (step + 1) % args.rollout_every == 0


def objective_parts(model, states, observations, controls, key, data, args, step,
                    include_rollout=None):
    if args.method == "normflowsde":
        preceding = preceding_observations(states, observations, data.metadata)
        inputs = jnp.concatenate((preceding, controls), axis=-1)
        rollout_active = (normflowsde_rollout_active(args, step) if include_rollout is None
                          else include_rollout and args.rollout_weight != 0.)
        if getattr(args, "pendulum_objective", None) is not None:
            from .pendulum_normflowsde_objective import loss_parts
            return loss_parts(
                model, states, inputs, key, segment_length=args.segment_length,
                training_draws=args.training_draws, rollout_draws=args.rollout_draws,
                state_sigma=args.state_sigma, profile=args.pendulum_objective,
                include_rollout=rollout_active,
            )
        parts = jax.vmap(lambda x, y, draw_key: model.paper_loss_components_single(
            x, y, draw_key, seg_length=args.segment_length, K=args.training_draws,
            rollout_K=args.rollout_draws,
            state_sigma=args.state_sigma,
            include_rollout=rollout_active))(
                states, inputs, jax.random.split(key, states.shape[0]))
        total = jnp.mean(parts["state"] + parts["transition_entropy"]
                         + args.continuity_weight * parts["continuity"]
                         + args.rollout_weight * parts["rollout"])
        return total, {name: jnp.mean(value) for name, value in parts.items()}
    if args.method == "nf_dpf":
        from bayesinf.applications.state_estimation.normalizing_flow_dpf import nf_dpf_component_loss, nf_dpf_loss
        warmup = min(int(data.protocol["train_steps"] * args.warmup_fraction),
                     data.protocol["train_steps"] - 1)
        if step < warmup:
            nll = nf_dpf_component_loss(model, states, observations, controls)
            return nll, dict(component_nll=nll)
        mse = nf_dpf_loss(model, states, observations, controls, key=key,
                           num_particles=args.training_particles, resampling="ot",
                           initial_std=0., nll_weight=0.)
        nll = (nf_dpf_component_loss(model, states, observations, controls)
               if args.nll_weight else jnp.zeros_like(mse))
        return mse + args.nll_weight * nll, dict(filter_mse=mse, component_nll=nll)
    parts = model.loss_parts(states, observations, controls, key=key)
    return parts.total, {name: value for name, value in parts._asdict().items() if name != "total"}


def objective(model, states, observations, controls, key, data, args, step):
    return objective_parts(model, states, observations, controls, key, data, args, step)[0]


def draw_samples(model, states, observations, controls, key, data, args):
    count = data.protocol["posterior_samples"]
    if args.method == "normflowsde":
        preceding = preceding_observations(states[None], observations[None], data.metadata)[0]
        inputs = jnp.concatenate((preceding, controls), axis=-1)
        return model.sample_posterior(inputs, states[0], key, n_particles=count, solver_name="euler")
    if args.method == "nf_dpf":
        filter_key, draw_key = jax.random.split(key)
        result = model.filter(observations, states[0], controls, key=filter_key,
                              num_particles=data.protocol["num_particles"],
                              initial_std=0., resampling="systematic")
        keys = jax.random.split(draw_key, observations.shape[0])
        draws = jax.vmap(lambda k, p, w: p[jax.random.categorical(k, jnp.log(w), shape=(count,))])(
            keys, result.particles, result.weights)
        return draws.swapaxes(0, 1)
    return model.sample(observations, states[0], controls, key=key, num_samples=count)


def score_samples(samples, truth, data):
    if data.protocol["metric_space"] == "physical":
        mean, std = np.asarray(data.metadata["state_mean"]), np.asarray(data.metadata["state_std"])
        samples, truth = samples * std + mean, truth * std + mean
    error = samples.mean(axis=1) - truth
    low, high = np.percentile(samples, [5, 95], axis=1)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2, axis=(1, 2))).mean())
    observed = np.asarray(data.protocol["observed_indices"])
    unobserved = np.setdiff1d(np.arange(truth.shape[-1]), observed)
    return dict(rmse=rmse(error), mae=float(np.abs(error).mean()),
                coverage_90=float(np.mean((truth >= low) & (truth <= high))),
                mean_interval_width=float(np.mean(high - low)),
                rmse_observed=rmse(error[..., observed]),
                rmse_unobserved=rmse(error[..., unobserved]) if len(unobserved) else None,
                metric_space=data.protocol["metric_space"])


def train_and_evaluate(model, data, args, report, tracker=None):
    """Only reached after the explicit --train option; no training in preparation."""
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(
        data.protocol["learning_rate"], weight_decay=1e-5))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    training = {key: jnp.asarray(data.train[key]) for key in ("states", "observations", "controls")}
    batch_size = report["full_trajectory_training_batch"]

    @eqx.filter_jit
    def update(candidate, state, states, observations, controls, key, include_rollout, warmup):
        # Static flags avoid recompiling for every integer optimizer step.
        objective_step = 0
        if args.method == "nf_dpf":
            objective_step = 0 if warmup else data.protocol["train_steps"]
        (loss, components), grads = eqx.filter_value_and_grad(objective_parts, has_aux=True)(
            candidate, states, observations, controls, key, data, args, objective_step,
            include_rollout=include_rollout)
        updates, state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), state, loss, optax.global_norm(grads), components

    @eqx.filter_jit
    def infer(candidate, states, observations, controls, key):
        return draw_samples(candidate, states, observations, controls, key, data, args)

    def evaluate_split(candidate, split, count, evaluation_seed):
        outputs = []
        inference_started = time.perf_counter()
        for index in range(min(count, len(split["states"]))):
            inputs = [jnp.asarray(split[name][index]) for name in ("states", "observations", "controls")]
            output = np.asarray(infer(candidate, *inputs,
                                      jax.random.fold_in(jax.random.PRNGKey(evaluation_seed), index)))
            if output.shape != (data.protocol["posterior_samples"], inputs[1].shape[0], inputs[0].shape[-1]):
                raise ValueError(f"Incorrect posterior shape: {output.shape}")
            outputs.append(output)
        inference_seconds = time.perf_counter() - inference_started
        samples = np.stack(outputs)
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite generated states")
        truth = np.asarray(split["states"][:len(outputs), 1:])
        return score_samples(samples, truth, data), samples, truth, inference_seconds

    started, best = time.perf_counter(), float("inf")
    warmup = min(int(data.protocol["train_steps"] * args.warmup_fraction), data.protocol["train_steps"] - 1)
    report["status"] = "running"
    report["trained"] = True
    metadata_path = args.output / "run_metadata.json"
    metadata_path.write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "history.jsonl").open("x") as history:
        for step in range(data.protocol["train_steps"]):
            batch_key, loss_key = jax.random.split(jax.random.fold_in(jax.random.PRNGKey(args.seed + 100), step))
            if args.method == "nf_dpf":
                batch = sample_training_windows(training, batch_key, data.protocol["batch_size"], 8)
            else:
                ids = jax.random.randint(batch_key, (batch_size,), 0, len(training["states"]))
                batch = [training[name][ids] for name in ("states", "observations", "controls")]
            model, optimizer_state, loss, gradient_norm, components = update(
                model, optimizer_state, *batch, loss_key,
                normflowsde_rollout_active(args, step), step < warmup)
            row = dict(step=step + 1, loss=float(loss), gradient_norm=float(gradient_norm))
            row["loss_components"] = {name: float(value) for name, value in components.items()}
            if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
                raise FloatingPointError(f"Nonfinite loss/gradient at step {step + 1}")
            if (step + 1) % args.validation_every == 0 or step + 1 == data.protocol["train_steps"]:
                validation, validation_samples, validation_truth, _ = evaluate_split(
                    model, data.validation, args.validation_windows, 77001)
                row["validation_rmse"] = validation["rmse"]
                validation_metric = getattr(args, "validation_metric", "rmse")
                score = validation["rmse"]
                if validation_metric == "crps":
                    from .evaluation_metrics import empirical_crps
                    score = float(np.mean(empirical_crps(validation_samples, validation_truth, axis=1)))
                    row["validation_crps"] = score
                if (args.method != "nf_dpf" or step >= warmup) and score < best:
                    best = score
                    report.update(best_validation_rmse=validation["rmse"], best_step=step + 1)
                    if validation_metric != "rmse":
                        report.update(best_validation_score=best, selection_metric=validation_metric,
                                      selection_metric_space="training_normalized")
                    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
                row["best_validation_rmse"] = report.get("best_validation_rmse", best)
                if validation_metric != "rmse":
                    row["best_validation_score"] = best
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + "\n")
            history.flush()
            if tracker is not None:
                values = {"train/loss": row["loss"], "train/gradient_norm": row["gradient_norm"],
                          "train/warmup": int(args.method == "nf_dpf" and step < warmup),
                          "train/rollout_active": int(args.method == "normflowsde"
                                                      and normflowsde_rollout_active(args, step))}
                values.update({f"train/components/{name}": value for name, value in row["loss_components"].items()})
                if "validation_rmse" in row:
                    values.update({"validation/rmse": row["validation_rmse"],
                                   "validation/best_rmse": row["best_validation_rmse"]})
                    if "validation_crps" in row:
                        values.update({"validation/crps_normalized": row["validation_crps"],
                                       "validation/best_score": best})
                tracker.metrics(values, step + 1)
    report["training_seconds_including_compilation"] = time.perf_counter() - started
    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    validation_only = getattr(args, "validation_only", False)
    final_split = data.validation if validation_only else data.test
    evaluation_seed = 77001 if validation_only else 99001
    evaluation_count = len(final_split["states"]) if validation_only else data.protocol["eval_windows"]
    # Compile and transfer one trajectory before timing the final evaluation.
    first = [jnp.asarray(final_split[name][0]) for name in ("states", "observations", "controls")]
    np.asarray(infer(model, *first, jax.random.PRNGKey(evaluation_seed)))
    metrics, samples, truth, inference_seconds = evaluate_split(
        model, final_split, evaluation_count, evaluation_seed)
    metrics.update(eval_windows=len(truth), horizon=truth.shape[1],
                   posterior_samples=samples.shape[1],
                   inference_access=access_label(args.method, getattr(args, "encoder_direction", "bidirectional")),
                   milliseconds_per_trajectory=1000 * inference_seconds / len(truth))
    sample_name = "validation_posterior.npz" if validation_only else "posterior.npz"
    metrics_name = "validation_metrics.json" if validation_only else "metrics.json"
    np.savez_compressed(args.output / sample_name, samples_normalized=samples,
                        truth_normalized=truth)
    (args.output / metrics_name).write_text(json.dumps(metrics, indent=2) + "\n")
    if validation_only:
        from .evaluation_metrics import metrics_from_samples
        proper = metrics_from_samples(
            samples, truth, observed_indices=data.protocol["observed_indices"],
            metric_space=data.protocol["metric_space"],
            state_mean=data.metadata["state_mean"], state_std=data.metadata["state_std"],
            energy_method="exact")
        metrics.update(proper["metrics"])
        (args.output / metrics_name).write_text(json.dumps(metrics, indent=2) + "\n")
        report.update(status="complete", validation=metrics, evaluation_split="validation", test_evaluated=False)
    else:
        report.update(status="complete", test=metrics)
    if tracker is not None:
        prefix = "validation_final" if validation_only else "test"
        tracker.metrics({**{f"{prefix}/{name}": value for name, value in metrics.items()},
                         "runtime/training_seconds_including_compilation": report["training_seconds_including_compilation"]},
                        data.protocol["train_steps"])


def main(argv=None):
    args = parse_args(argv)
    output = args.output / "run_metadata.json"
    if output.exists():
        raise FileExistsError(f"Choose a new run directory; existing metadata at {output}")
    data = load_data(args)
    report = prepare_report(data, args)
    verify_prepared_report(report, args)
    model = build_model(data, args)
    report["parameter_count"] = sum(leaf.size for leaf in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    if args.capacity_profile is not None:
        from .capacity import validate_capacity
        report["capacity"] = validate_capacity(model, data, args)
    args.output.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    if args.train:
        tracker = JournalTracker(args.output, report,
                    uri=args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"),
                    experiment=args.tracking_experiment or os.environ.get("MLFLOW_EXPERIMENT_NAME", "normflowsde-comparison"),
                    flush_every=args.tracking_flush_every)
        previous_handlers = {}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")
        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, interrupted)
            train_and_evaluate(model, data, args, report, tracker)
        except BaseException as error:
            report.update(status="killed" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            output.write_text(json.dumps(report, indent=2) + "\n")
            tracker.finish({"complete": "FINISHED", "killed": "KILLED"}.get(report["status"], "FAILED"), report)
    print(json.dumps(dict(status=report["status"], output=str(args.output),
                         method=args.method, dataset=args.dataset)), flush=True)


if __name__ == "__main__":
    main()
