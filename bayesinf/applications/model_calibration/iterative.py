"""JAX implementation of iterative forward/inverse thermal calibration."""

from __future__ import annotations

import csv
import json
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Protocol, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import optax

from bayesinf.models.normalizing_flow import BNNormFlow
from bayesinf.models.flow_matching import StatefulConditionalFlowMatching
from bayesinf.applications.sensitivity_analysis.thermal_model import PARAMETER_NAMES


DEFAULT_IDENTIFICATION_GROUPS = (
    (11, 14, 15, 16),
    (12, 13, 17),
    (6, 7, 8, 10, 18),
    (0, 1, 2, 3, 4),
    (5, 9),
)

# Physical sampling bounds used to generate the FMU calibration data.  Keeping
# these fixed across stages makes the inverse-model coordinates independent of
# the random train/validation split.
THERMAL_PARAMETER_BOUNDS = np.asarray((
    (18.0, 27.0), (35.0, 55.0), (0.05, 1.0), (0.5, 1.5), (1.0, 2.5),
    (0.5, 1.5), (0.05, 1.0), (0.05, 1.0), (0.05, 1.0), (0.5, 1.5),
    (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.05, 1.0), (0.05, 1.0),
    (0.05, 1.0), (0.5, 1.5), (20.0, 35.0), (2.0, 3.5),
), dtype=np.float32)


def _plot_intermediate_inference(samples: np.ndarray, truth: np.ndarray,
                                 group: Sequence[int], epoch: int,
                                 stage_dir: Path) -> Path:
    """Save posterior histograms for one calibration target during training."""
    plot_dir = stage_dir / "intermediate_inference"
    plot_dir.mkdir(parents=True, exist_ok=True)
    columns = min(4, len(group))
    rows = int(np.ceil(len(group) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.4 * rows),
                             squeeze=False, layout="constrained")
    for position, (index, target) in enumerate(zip(group, truth)):
        axis = axes.flat[position]
        axis.hist(samples[:, position], bins=25, density=True, alpha=.7,
                  color="#60a5fa")
        axis.axvline(target, color="black", ls="--", lw=2, label="Truth")
        axis.axvline(samples[:, position].mean(), color="#dc2626", lw=2,
                     label="Posterior mean")
        axis.set(title=f"Parameter {index}", xlabel="Physical value", ylabel="Density")
        axis.grid(alpha=.2)
    for axis in axes.flat[len(group):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"Stage inference at epoch {epoch}")
    path = plot_dir / f"epoch_{epoch:05d}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _plot_intermediate_regression(samples: np.ndarray, truth: np.ndarray,
                                  group: Sequence[int], epoch: int,
                                  stage_dir: Path, stage_number: int) -> tuple[Path, Path]:
    """Plot held-out posterior regression in the final evaluator's format."""
    plot_dir = stage_dir / "intermediate_regression"
    plot_dir.mkdir(parents=True, exist_ok=True)
    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, (0.05, 0.95), axis=0)
    columns = min(2, len(group))
    rows = int(np.ceil(len(group) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4.5 * rows),
                             squeeze=False)
    for position, (axis, index) in enumerate(zip(axes.flat, group)):
        # The posterior mean can lie outside the central 90% interval for a
        # strongly skewed distribution. Draw the interval independently so
        # Matplotlib never receives an invalid negative error-bar length.
        axis.vlines(truth[:, position], lower[:, position], upper[:, position],
                    color="#2563eb", alpha=.45, lw=1)
        axis.scatter(truth[:, position], mean[:, position], s=10,
                     alpha=.65, color="#2563eb")
        limits = (min(truth[:, position].min(), lower[:, position].min()),
                  max(truth[:, position].max(), upper[:, position].max()))
        axis.plot(limits, limits, color="black", ls="--", lw=1)
        axis.set_title(PARAMETER_NAMES[index])
        axis.set_xlabel("True parameter")
        axis.set_ylabel("Posterior mean and 90% CI")
        axis.grid(alpha=.25)
    for axis in tuple(axes.flat)[len(group):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Stage {stage_number} BayesFlow posterior regression — held-out test set — epoch {epoch}"
    )
    fig.tight_layout()
    png_path = plot_dir / f"epoch_{epoch:05d}_parameter_regression.png"
    pdf_path = plot_dir / f"epoch_{epoch:05d}_parameter_regression.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path, pdf_path


def _save_stage_regression_evaluation(samples: np.ndarray, truth: np.ndarray,
                                      group: Sequence[int], stage: int,
                                      output: Path, *, save_pdf: bool = True) -> dict:
    """Persist held-out posterior regression before advancing a stage."""
    output.mkdir(parents=True, exist_ok=True)
    mean = samples.mean(axis=0)
    lower, upper = np.quantile(samples, (0.05, 0.95), axis=0)
    error = mean - truth
    residual_sum_squares = np.sum(error ** 2, axis=0)
    total_sum_squares = np.sum(
        (truth - truth.mean(axis=0, keepdims=True)) ** 2, axis=0)
    per_parameter_r2 = 1.0 - residual_sum_squares / np.maximum(
        total_sum_squares, np.finfo(float).eps)
    metrics = {
        "stage": stage + 1,
        "group": list(group),
        "validation_samples": int(len(truth)),
        "posterior_samples": int(samples.shape[0]),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "coverage_90": float(np.mean((truth >= lower) & (truth <= upper))),
        "mean_interval_width_90": float(np.mean(upper - lower)),
        "per_parameter_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "per_parameter_r2": per_parameter_r2.tolist(),
        "per_parameter_mae": np.mean(np.abs(error), axis=0).tolist(),
        "per_parameter_coverage_90": np.mean(
            (truth >= lower) & (truth <= upper), axis=0).tolist(),
    }
    np.save(output / "posterior_samples.npy", samples.astype(np.float32))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))

    columns = len(group)
    rows = 1
    fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 4.2),
                             squeeze=False, layout="constrained")
    for position, (axis, parameter_index) in enumerate(zip(axes.flat, group)):
        axis.vlines(truth[:, position], lower[:, position], upper[:, position],
                    color="#2563eb", alpha=.35, lw=.8)
        axis.scatter(truth[:, position], mean[:, position], s=10,
                     alpha=.6, color="#2563eb")
        limits = (min(truth[:, position].min(), lower[:, position].min()),
                  max(truth[:, position].max(), upper[:, position].max()))
        axis.plot(limits, limits, color="black", ls="--", lw=1)
        axis.set(title=PARAMETER_NAMES[parameter_index], xlabel="True parameter",
                 ylabel="Posterior mean and 90% CI")
        parameter_rmse = np.sqrt(np.mean(error[:, position] ** 2))
        axis.text(
            0.04, 0.96,
            f"RMSE = {parameter_rmse:.4f}\n$R^2$ = {per_parameter_r2[position]:.4f}",
            transform=axis.transAxes, ha="left", va="top",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white",
                  "edgecolor": "0.75", "alpha": 0.9},
        )
        axis.grid(alpha=.25)
    for axis in tuple(axes.flat)[len(group):]:
        axis.set_visible(False)
    fig.suptitle(
        f"Stage {stage + 1} flow-matching regression — validation set (n={len(truth)})")
    fig.savefig(output / "parameter_regression.png", dpi=180)
    if save_pdf:
        fig.savefig(output / "parameter_regression.pdf")
    plt.close(fig)
    return metrics


class ForwardModel(Protocol):
    def __call__(self, parameters: np.ndarray, key: jax.Array) -> np.ndarray: ...


@dataclass(frozen=True)
class StageSignals:
    """Signals used to train and validate one calibration stage."""

    train: np.ndarray
    validation: np.ndarray
    source: str
    test: np.ndarray | None = None
    train_mask: np.ndarray | None = None
    validation_mask: np.ndarray | None = None
    inference_target: str = "prior_marginal"


def prior_component_mask(bundle) -> np.ndarray:
    """Recover original-prior simulations from a labeled empirical mixture.

    Resampling a finite posterior file creates atoms, so it has no ordinary
    continuous proposal density. Keeping the independently labeled prior
    component is an exact rejection correction; pretending the empirical
    posterior samples have a known smooth density would not be valid.
    """
    if "proposal_epsilon" not in bundle or float(bundle["proposal_epsilon"]) <= 0:
        raise ValueError("Original-prior inference needs a nonzero labeled prior component; regenerate prior data")
    if "proposal_component" not in bundle or "parameters" not in bundle:
        raise ValueError("Proposal correction requires retained parameters and component labels")
    if set(map(int, bundle["fixed_indices"])) - set(map(int, bundle["proposal_indices"])):
        raise ValueError("Fixed nuisance parameters cannot recover the original-prior marginal")
    mask = np.asarray(bundle["proposal_component"]) == 1
    if mask.shape != (len(bundle["parameters"]),) or not mask.any():
        raise ValueError("No prior-component draws available; regenerate with a larger prior fraction")
    return mask


class StageSignalProvider:
    """Construct stage data in one place instead of branching inside training."""

    def __init__(self, forward_model: ForwardModel,
                 saved: dict[int, StageSignals] | None = None):
        self.forward_model = forward_model
        self.saved = dict(saved or {})

    def __call__(self, stage: int, data, fixed: dict[int, float],
                 key: jax.Array) -> StageSignals:
        if stage in self.saved:
            result = self.saved[stage]
        elif stage == 0 or not fixed:
            result = StageSignals(
                data.train.signals, data.validation.signals, "fmu", data.test.signals
            )
        else:
            train_parameters = data.train.parameters.copy()
            validation_parameters = data.validation.parameters.copy()
            for index, value in fixed.items():
                train_parameters[:, index] = value
                validation_parameters[:, index] = value
            train_key, validation_key = jr.split(key)
            result = StageSignals(
                self.forward_model(train_parameters, train_key),
                self.forward_model(validation_parameters, validation_key),
                "pels_vae",
            )
        test = result.test
        if test is None:
            same_as_validation = (
                len(data.test.parameters) == len(data.validation.parameters)
                and np.array_equal(data.test.parameters, data.validation.parameters)
            )
            if same_as_validation:
                test = result.validation
            elif stage == 0:
                test = data.test.signals
            else:
                test_parameters = data.test.parameters.copy()
                for index, value in fixed.items():
                    test_parameters[:, index] = value
                test = self.forward_model(test_parameters, jr.fold_in(key, 2))
            result = replace(result, test=test)
        if len(result.train) != len(data.train.signals):
            raise ValueError("Stage training signals have the wrong sample count")
        if len(result.validation) != len(data.validation.signals):
            raise ValueError("Stage validation signals have the wrong sample count")
        if len(result.test) != len(data.test.signals):
            raise ValueError("Stage test signals have the wrong sample count")
        return result


@eqx.filter_jit
def _stateful_flow_step(model, state, signals, parameters, optimizer_state, optimizer):
    def objective(candidate, current_state):
        log_probability, updated_state = candidate.log_prob_with_state(
            parameters, signals, current_state, inference=False
        )
        return -jnp.mean(log_probability), updated_state
    (loss, state), gradients = eqx.filter_value_and_grad(
        objective, has_aux=True
    )(model, state)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    return eqx.apply_updates(model, updates), state, optimizer_state, loss


@eqx.filter_jit
def _stateful_flow_validation(model, state, signals, parameters):
    """Evaluate exactly as an individual posterior will be evaluated."""
    log_probability, _ = model.log_prob_with_state(
        parameters, signals, state, inference=True
    )
    return -jnp.mean(log_probability)


@eqx.filter_jit
def _stateful_fm_step(model, state, signals, parameters, key,
                      optimizer_state, optimizer):
    def objective(candidate, current_state):
        return candidate.loss_with_state(
            parameters, signals, key, current_state, inference=False)
    (loss, state), gradients = eqx.filter_value_and_grad(
        objective, has_aux=True)(model, state)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    return eqx.apply_updates(model, updates), state, optimizer_state, loss


@eqx.filter_jit
def _stateful_fm_validation(model, state, signals, parameters, key):
    return model.loss_with_state(
        parameters, signals, key, state, inference=True)[0]


@dataclass(frozen=True)
class CalibrationConfig:
    groups: tuple[tuple[int, ...], ...] = DEFAULT_IDENTIFICATION_GROUPS
    epochs: int = 10_000
    batch_size: int = 100
    learning_rate: float = 5e-4
    stage_learning_rates: tuple[float, ...] | None = None
    weight_decay: float = 1e-4
    patience: int = 1_000
    min_delta: float = 1e-4
    posterior_samples: int = 100
    n_coupling: int = 4
    flow_type: str = "realnvp"
    fm_integration_steps: int = 100
    width: int = 64
    summary_dim: int = 64
    rnn_hidden: int = 512
    recurrent_layers: int = 1
    conditioner_linear_hidden: int = 512
    conditioner_linear_layers: int = 4
    conditioner_type: str = "pytorch_lstm"
    use_actnorm: bool = False
    clamp: float = 1.0
    lr_hold_epochs: int = 500
    min_learning_rate: float = 0.0
    learning_rate_schedule: str = "exponential"
    exponential_decay_factor: float = 0.8
    exponential_decay_steps: int = 2_000
    intermediate_eval_every: int = 100
    intermediate_posterior_samples: int = 100
    save_intermediate_plots: bool = True
    save_pdf_figures: bool = True
    seed: int = 0
    bounded_parameters: bool = True
    normalize_parameters: bool = True
    normalize_signals: bool = True
    standalone_stage_output: bool = False
    resume_training: bool = False
    early_stopping_metric: str = "validation"
    inference_target: str = "prior_marginal"

    def __post_init__(self):
        if self.bounded_parameters and not self.normalize_parameters:
            raise ValueError("Bounded parameters require normalization by physical prior bounds")
        if self.inference_target not in {"prior_marginal", "conditional_plugin", "proposal_conditional"}:
            raise ValueError("Unknown calibration inference_target")


@dataclass(frozen=True)
class CalibrationResult:
    estimate: np.ndarray
    truth: np.ndarray
    posterior_samples: tuple[np.ndarray, ...]
    groups: tuple[tuple[int, ...], ...]

    @property
    def relative_error_percent(self) -> np.ndarray:
        scale = np.maximum(np.abs(self.truth), 1e-8)
        return np.abs(self.estimate - self.truth) / scale * 100.0


class PELSVAEForwardModel:
    """Batch adapter for the existing Equinox :class:`PELS_VAE` surrogate."""

    def __init__(self, model, state, normalization, *, passes: int = 1):
        self.model = eqx.nn.inference_mode(model)
        self.state = state
        self.normalization = normalization
        self.passes = passes

    def __call__(self, parameters: np.ndarray, key: jax.Array) -> np.ndarray:
        theta = jnp.asarray(self.normalization.normalize_parameters(parameters))
        prediction = np.asarray(self.model.predict(
            theta, self.state, key=key, passes=self.passes), dtype=np.float32)
        return self.normalization.denormalize_signals(prediction)


def _global_minmax(train: np.ndarray, *others: np.ndarray):
    """Map signals to [-1, 1] using one training-only global range."""
    minimum = np.asarray(train.min(), dtype=np.float32)
    maximum = np.asarray(train.max(), dtype=np.float32)
    scale = np.maximum(maximum - minimum, np.float32(1e-6))
    transform = lambda value: 2.0 * (value - minimum) / scale - 1.0
    return (transform(train), *(transform(value) for value in others), minimum, maximum)


def _parameter_minmax(parameters: np.ndarray, indices: Sequence[int], *, bounds=None):
    """Map selected physical parameters to [-1, 1] dimension by dimension."""
    del indices
    if bounds is None:
        minimum = parameters.min(axis=0, keepdims=True)
        maximum = parameters.max(axis=0, keepdims=True)
    else:
        bounds = np.asarray(bounds, dtype=np.float32)
        if bounds.shape != (parameters.shape[-1], 2) or not np.all(bounds[:, 1] > bounds[:, 0]):
            raise ValueError("Physical parameter bounds must have shape [dimension, 2] and positive widths")
        minimum, maximum = bounds[:, 0][None], bounds[:, 1][None]
    if np.any(maximum <= minimum):
        raise ValueError("Parameter normalization requires nonzero widths")
    normalized = 2.0 * (parameters - minimum) / (maximum - minimum) - 1.0
    return normalized, minimum, maximum


def _parameter_transform(parameters: np.ndarray, minimum: np.ndarray,
                         maximum: np.ndarray):
    """Apply training-derived parameter coordinates to another split."""
    width = np.maximum(maximum - minimum, np.float32(1e-6))
    return (2.0 * (parameters - minimum) / width - 1.0).astype(np.float32)


def _parameter_inverse(normalized: np.ndarray, minimum: np.ndarray,
                       maximum: np.ndarray):
    return 0.5 * (normalized + 1.0) * (maximum - minimum) + minimum


def _bounded_to_flow(normalized: np.ndarray):
    """Map the open prior support to R; endpoints have no finite inverse."""
    normalized = np.asarray(normalized)
    if not np.isfinite(normalized).all() or np.any(np.abs(normalized) >= 1):
        raise ValueError("Bounded training targets must lie strictly inside the physical prior bounds")
    return np.arctanh(normalized).astype(np.float32)


def _flow_to_bounded(value: np.ndarray):
    """Map unconstrained flow output back inside the physical prior bounds."""
    return np.tanh(value)


class IterativeModelCalibrator:
    """Identify sensitivity-ordered groups and regenerate training signals."""

    def __init__(self, config: CalibrationConfig, output_dir: Path | str, tracker=None,
                 tracker_loss_only: bool = False):
        self.config = config
        self.output_dir = Path(output_dir)
        self.tracker = tracker
        self.tracker_loss_only = tracker_loss_only
        self._tracking_step = 0

    def _fit_group(self, train_x, train_theta, val_x, val_theta, test_x, test_theta,
                   group, key, stage_dir, normalized_target, theta_min, theta_max,
                   truth_group, stage):
        model_key, loop_key = jr.split(key)
        common = dict(
            encoded_dim=self.config.summary_dim,
            encoder_hidden_size=self.config.rnn_hidden,
            encoder_layers=self.config.recurrent_layers,
            encoder_linear_hidden_size=self.config.conditioner_linear_hidden,
            encoder_linear_layers=self.config.conditioner_linear_layers,
            conditioner_type=self.config.conditioner_type,
            sequence_length=train_x.shape[1],
        )
        if self.config.flow_type == "fm":
            model = StatefulConditionalFlowMatching(
                len(group), train_x.shape[-1], model_key,
                hidden_dim=self.config.width, depth=4,
                integration_steps=self.config.fm_integration_steps, **common)
        elif self.config.flow_type == "realnvp":
            model = BNNormFlow(
                len(group), train_x.shape[-1], model_key,
                n_coupling=self.config.n_coupling,
                hidden_dims=(self.config.width,) * 4,
                use_actnorm=self.config.use_actnorm,
                activation=jnp.tanh, clamp=self.config.clamp,
                clamp_activation="I", **common)
        else:
            raise ValueError("flow_type must be 'realnvp' or 'fm'")
        flow_state = model.initial_batchnorm_state()
        initial_learning_rate = (
            self.config.stage_learning_rates[stage]
            if self.config.stage_learning_rates is not None
            else self.config.learning_rate
        )
        flow_train_theta = (_bounded_to_flow(train_theta)
                            if self.config.bounded_parameters else train_theta)
        checkpoint_path = stage_dir / "bayesflow.eqx"
        state_path = stage_dir / "bayesflow_state.eqx"
        history_path = stage_dir / "history.csv"
        prior_history = []
        epoch_offset = 0
        if self.config.resume_training:
            if not checkpoint_path.exists() or not state_path.exists():
                raise FileNotFoundError(
                    "--resume-training requires existing bayesflow.eqx and "
                    "bayesflow_state.eqx checkpoints"
                )
            normalizer_path = stage_dir / "normalization.npz"
            if not normalizer_path.exists():
                raise ValueError("Resuming calibration requires the checkpoint's normalization.npz")
            with np.load(normalizer_path) as previous:
                if (not np.array_equal(previous["theta_min"], theta_min)
                        or not np.array_equal(previous["theta_max"], theta_max)):
                    raise ValueError("Checkpoint parameter coordinates differ; retrain with the corrected prior bounds")
            metadata_path = stage_dir / "metadata.json"
            if metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                previous_target = metadata.get(
                    "inference_target", "conditional_plugin" if metadata.get("fixed_before_stage") else "prior_marginal")
                if previous_target != self.config.inference_target:
                    raise ValueError("Checkpoint inference target differs; start a fresh corrected run")
            model = eqx.tree_deserialise_leaves(checkpoint_path, model)
            flow_state = eqx.tree_deserialise_leaves(state_path, flow_state)
            if history_path.exists():
                with history_path.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        prior_history.append((int(row["epoch"]), float(row["train_loss"]),
                                              float(row["validation_loss"])))
                if prior_history:
                    epoch_offset = prior_history[-1][0]
        else:
            model = model.initialize_actnorm(
                jnp.asarray(flow_train_theta), jnp.asarray(train_x), flow_state
            )
        steps_per_epoch = max(1, (len(train_x) + self.config.batch_size - 1) // self.config.batch_size)
        if self.config.learning_rate_schedule == "exponential":
            exponential_kwargs = dict(
                init_value=initial_learning_rate,
                transition_steps=self.config.exponential_decay_steps,
                decay_rate=self.config.exponential_decay_factor,
                staircase=False,
            )
            if self.config.min_learning_rate > 0:
                exponential_kwargs["end_value"] = self.config.min_learning_rate
            base_learning_rate = optax.exponential_decay(**exponential_kwargs)
            learning_rate = lambda step: base_learning_rate(
                step + epoch_offset * steps_per_epoch
            )
        elif self.config.learning_rate_schedule == "cosine":
            def learning_rate(step):
                epoch = jnp.asarray(step, dtype=jnp.float32) / steps_per_epoch
                decay_epochs = max(1, self.config.epochs - self.config.lr_hold_epochs)
                progress = jnp.clip(
                    (epoch - self.config.lr_hold_epochs) / decay_epochs, 0.0, 1.0
                )
                cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
                return self.config.min_learning_rate + (
                    initial_learning_rate - self.config.min_learning_rate
                ) * cosine
        else:
            raise ValueError("learning_rate_schedule must be 'exponential' or 'cosine'")
        optimizer = optax.chain(
            optax.add_decayed_weights(self.config.weight_decay),
            optax.scale_by_adam(),
            optax.scale_by_learning_rate(learning_rate, flip_sign=True),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        x, vx = jnp.asarray(train_x), jnp.asarray(val_x)
        y = jnp.asarray(flow_train_theta)
        vy = jnp.asarray(_bounded_to_flow(val_theta) if self.config.bounded_parameters
                         else val_theta)
        validation_key = jr.fold_in(loop_key, 9173)
        validate = (lambda candidate, candidate_state:
                    _stateful_fm_validation(candidate, candidate_state, vx, vy,
                                            validation_key)) if self.config.flow_type == "fm" else (
                    lambda candidate, candidate_state:
                    _stateful_flow_validation(candidate, candidate_state, vx, vy))
        initial_validation_loss = (
            float(validate(model, flow_state))
            if self.config.resume_training else np.inf
        )
        initial_selection_loss = (
            initial_validation_loss
            if self.config.early_stopping_metric == "validation" else np.inf
        )
        best, best_state = model, flow_state
        best_loss, best_validation_loss, stale = (
            initial_selection_loss, initial_validation_loss, 0
        )
        history = list(prior_history)
        intermediate_path = stage_dir / "intermediate_metrics.csv"
        if not self.config.resume_training or not intermediate_path.exists():
            with intermediate_path.open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("epoch", "validation_loss", "posterior_rmse", "posterior_mae",
                                 *(f"mean_parameter_{index}" for index in group)))
        training_start = time.perf_counter()
        for epoch in range(self.config.epochs):
            global_epoch = epoch_offset + epoch + 1
            loop_key, permutation_key, step_key = jr.split(loop_key, 3)
            permutation = np.asarray(jr.permutation(permutation_key, len(x)))
            losses = []
            for begin in range(0, len(x), self.config.batch_size):
                ids = permutation[begin:begin + self.config.batch_size]
                step_key, batch_key = jr.split(step_key)
                if self.config.flow_type == "fm":
                    model, flow_state, opt_state, loss = _stateful_fm_step(
                        model, flow_state, x[ids], y[ids], batch_key,
                        opt_state, optimizer)
                else:
                    model, flow_state, opt_state, loss = _stateful_flow_step(
                        model, flow_state, x[ids], y[ids], opt_state, optimizer)
                losses.append(float(loss))
            train_loss = float(np.mean(losses))
            val_loss = float(validate(model, flow_state))
            history.append((global_epoch, train_loss, val_loss))
            selection_loss = (train_loss if self.config.early_stopping_metric == "train"
                              else val_loss)
            if (np.isfinite(selection_loss)
                    and selection_loss < best_loss - self.config.min_delta):
                best, best_state = model, flow_state
                best_loss, best_validation_loss, stale = selection_loss, val_loss, 0
                # A long-running cluster job can outlive transient cleanup of a
                # newly-created output directory. Re-establish it at every
                # checkpoint boundary before opening checkpoint files.
                stage_dir.mkdir(parents=True, exist_ok=True)
                eqx.tree_serialise_leaves(stage_dir / "bayesflow.eqx", best)
                eqx.tree_serialise_leaves(stage_dir / "bayesflow_state.eqx", best_state)
            else:
                stale += 1
            self._tracking_step += 1
            if self.config.learning_rate_schedule == "exponential":
                current_lr = max(
                    self.config.min_learning_rate,
                    initial_learning_rate * self.config.exponential_decay_factor ** (
                        (global_epoch * steps_per_epoch) /
                        self.config.exponential_decay_steps
                    ),
                )
            elif epoch + 1 <= self.config.lr_hold_epochs:
                current_lr = initial_learning_rate
            else:
                progress = min(1.0, (epoch + 1 - self.config.lr_hold_epochs) /
                               max(1, self.config.epochs - self.config.lr_hold_epochs))
                current_lr = self.config.min_learning_rate + (
                    initial_learning_rate - self.config.min_learning_rate
                ) * 0.5 * (1.0 + np.cos(np.pi * progress))
            if self.tracker is not None:
                payload = {"train/loss": history[-1][1]}
                if not self.tracker_loss_only:
                    payload.update({
                        "workflow_step": self._tracking_step,
                        "stage": stage + 1,
                        "stage_epoch": global_epoch,
                        "validation/loss": val_loss,
                        "optimization/learning_rate": current_lr,
                        "optimization/stale_epochs": stale,
                    })
                self.tracker.log(payload, step=self._tracking_step)
            check_now = epoch == 0 or global_epoch % self.config.intermediate_eval_every == 0
            if check_now:
                # Long-running cluster jobs may overlap external result-folder
                # maintenance. Re-establish the stage directory before every
                # intermediate artifact, as is already done for checkpoints.
                stage_dir.mkdir(parents=True, exist_ok=True)
                step_key, check_key = jr.split(step_key)
                normalized_samples = np.asarray(model.sample(
                    jnp.asarray(normalized_target), check_key,
                    self.config.intermediate_posterior_samples, flow_state,
                ))[:, 0]
                if self.config.bounded_parameters:
                    normalized_samples = _flow_to_bounded(normalized_samples)
                physical_samples = (
                    _parameter_inverse(normalized_samples, theta_min, theta_max)
                    if self.config.normalize_parameters else normalized_samples
                )
                posterior_mean = physical_samples.mean(axis=0)
                posterior_rmse = float(np.sqrt(np.mean((posterior_mean - truth_group) ** 2)))
                posterior_mae = float(np.mean(np.abs(posterior_mean - truth_group)))
                np.save(stage_dir / "latest_intermediate_posterior.npy",
                        physical_samples.astype(np.float32))
                intermediate = {
                    "epoch": global_epoch, "validation_loss": val_loss,
                    "posterior_rmse": posterior_rmse, "posterior_mae": posterior_mae,
                    "posterior_mean": posterior_mean.tolist(), "group": list(group),
                }
                (stage_dir / "latest_intermediate_metrics.json").write_text(
                    json.dumps(intermediate, indent=2)
                )
                with intermediate_path.open("a", newline="") as handle:
                    csv.writer(handle).writerow((global_epoch, val_loss, posterior_rmse,
                                                 posterior_mae, *posterior_mean))
                inference_plot = None
                regression_plot = None
                regression_due = (
                    global_epoch % self.config.intermediate_eval_every == 0
                    and (self.config.save_intermediate_plots
                         or (self.tracker is not None and not self.tracker_loss_only))
                )
                if regression_due:
                    if self.config.save_intermediate_plots:
                        inference_plot = _plot_intermediate_inference(
                            physical_samples, truth_group, group, global_epoch, stage_dir
                        )
                    test_predictions = []
                    for begin in range(0, len(test_x), self.config.batch_size):
                        step_key, regression_key = jr.split(step_key)
                        batch_samples = np.asarray(model.sample(
                            jnp.asarray(test_x[begin:begin + self.config.batch_size]),
                            regression_key, self.config.intermediate_posterior_samples,
                            flow_state,
                        ))
                        if self.config.bounded_parameters:
                            batch_samples = _flow_to_bounded(batch_samples)
                        if self.config.normalize_parameters:
                            batch_samples = _parameter_inverse(
                                batch_samples, theta_min, theta_max
                            )
                        test_predictions.append(batch_samples)
                    test_samples = np.concatenate(test_predictions, axis=1)
                    if self.config.save_intermediate_plots:
                        regression_plot, _ = _plot_intermediate_regression(
                            test_samples, test_theta, group, global_epoch, stage_dir,
                            stage + 1,
                        )
                    else:
                        import wandb
                        with tempfile.TemporaryDirectory(
                                prefix="bayesflow-regression-") as temporary:
                            temporary_plot, _ = _plot_intermediate_regression(
                                test_samples, test_theta, group, global_epoch,
                                Path(temporary), stage + 1,
                            )
                            self.tracker.log(
                                {"intermediate/parameter_regression": wandb.Image(
                                    str(temporary_plot)
                                )},
                                step=self._tracking_step,
                            )
                if self.tracker is not None and not self.tracker_loss_only:
                    tracked = {
                        "workflow_step": self._tracking_step,
                        "intermediate/posterior_rmse": posterior_rmse,
                        "intermediate/posterior_mae": posterior_mae,
                    }
                    tracked.update({f"intermediate/parameter_{index}_mean": float(value)
                                    for index, value in zip(group, posterior_mean)})
                    if inference_plot is not None:
                        import wandb
                        tracked["intermediate/inference_plot"] = wandb.Image(
                            str(inference_plot)
                        )
                    if regression_plot is not None:
                        import wandb
                        tracked["intermediate/parameter_regression"] = wandb.Image(
                            str(regression_plot)
                        )
                    self.tracker.log(tracked, step=self._tracking_step)
            if epoch == 0 or global_epoch % 25 == 0:
                print(f"stage={stage + 1} epoch={global_epoch} train={history[-1][1]:.6f} "
                      f"validation={val_loss:.6f} stale={stale}", flush=True)
            if stale >= self.config.patience:
                break
        training_seconds = time.perf_counter() - training_start
        # Covers the unlikely case where all validation losses are non-finite.
        if not (stage_dir / "bayesflow.eqx").exists():
            eqx.tree_serialise_leaves(stage_dir / "bayesflow.eqx", best)
            eqx.tree_serialise_leaves(stage_dir / "bayesflow_state.eqx", best_state)
        with (stage_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("epoch", "train_loss", "validation_loss"))
            writer.writerows(history)
        return best, best_state, best_validation_loss, len(history), training_seconds

    def run(self, data, forward_model: ForwardModel, *, real_test_index: int = 0,
            max_stages: int | None = None, start_stage: int = 0,
            initial_fixed: dict[int, float] | None = None,
            stage_signal_overrides: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
            signal_provider: StageSignalProvider | None = None,
            ) -> CalibrationResult:
        """Run all groups for one noisy proxy-real observation trajectory."""
        if not 0 <= real_test_index < len(data.real_test.signals):
            raise IndexError("real_test_index is outside the real-test partition")
        parameter_dim = data.train.parameters.shape[1]
        flattened = [index for group in self.config.groups for index in group]
        if sorted(flattened) != list(range(parameter_dim)):
            raise ValueError("groups must contain every parameter exactly once")
        stage_count = len(self.config.groups) if max_stages is None else max_stages
        if not 1 <= stage_count <= len(self.config.groups):
            raise ValueError("max_stages must be between one and the number of groups")
        if not 0 <= start_stage < stage_count:
            raise ValueError("start_stage must identify a stage before max_stages")
        required_fixed = {
            index for group in self.config.groups[:start_stage] for index in group
        }
        supplied_fixed = set((initial_fixed or {}).keys())
        if supplied_fixed != required_fixed:
            raise ValueError(
                "initial_fixed must contain exactly the parameters completed "
                f"before start_stage; expected {sorted(required_fixed)}, "
                f"received {sorted(supplied_fixed)}"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        key = jr.PRNGKey(self.config.seed)
        fixed: dict[int, float] = dict(initial_fixed or {})
        posteriors = []
        timings = []
        workflow_start = time.perf_counter()
        target = data.real_test.signals[real_test_index:real_test_index + 1]
        if signal_provider is None:
            saved = {
                stage: StageSignals(train, validation, "saved",
                                    inference_target="conditional_plugin")
                for stage, (train, validation) in (stage_signal_overrides or {}).items()
            }
            signal_provider = StageSignalProvider(forward_model, saved)

        for stage, group in enumerate(self.config.groups[:stage_count]):
            if stage < start_stage:
                continue
            stage_start = time.perf_counter()
            stage_root = (self.output_dir if self.config.standalone_stage_output
                          else self.output_dir / f"stage_{stage + 1:02d}")
            stage_dir = stage_root / "model" if self.config.standalone_stage_output else stage_root
            stage_dir.mkdir(parents=True, exist_ok=True)
            if self.config.standalone_stage_output:
                (stage_root / "evaluation").mkdir(exist_ok=True)
                (stage_root / "feature_map").mkdir(exist_ok=True)
            key, generation_key, training_key, sampling_key = jr.split(key, 4)
            generation_start = time.perf_counter()
            conditioning_values = (fixed if self.config.inference_target == "conditional_plugin" else {})
            stage_signals = signal_provider(stage, data, conditioning_values, generation_key)
            if (stage_signals.source.startswith("saved")
                    and stage_signals.inference_target != self.config.inference_target):
                raise ValueError("Saved stage data and calibration inference_target disagree")
            train_x, val_x = stage_signals.train, stage_signals.validation
            train_mask = (np.ones(len(train_x), dtype=bool) if stage_signals.train_mask is None
                          else np.asarray(stage_signals.train_mask, dtype=bool))
            val_mask = (np.ones(len(val_x), dtype=bool) if stage_signals.validation_mask is None
                        else np.asarray(stage_signals.validation_mask, dtype=bool))
            if (train_mask.shape != (len(train_x),) or val_mask.shape != (len(val_x),)
                    or not train_mask.any() or not val_mask.any()):
                raise ValueError("Stage proposal correction must retain training and validation prior draws")
            train_x, val_x = train_x[train_mask], val_x[val_mask]
            generation_seconds = time.perf_counter() - generation_start

            preprocessing_start = time.perf_counter()
            if self.config.normalize_signals:
                (normalized_train_x, normalized_val_x, normalized_target,
                 normalized_test_x, x_min, x_max) = _global_minmax(
                    train_x, val_x, target, stage_signals.test
                )
                signal_transform = "global_minmax_-1_1"
            else:
                normalized_train_x = np.asarray(train_x, dtype=np.float32)
                normalized_val_x = np.asarray(val_x, dtype=np.float32)
                normalized_target = np.asarray(target, dtype=np.float32)
                normalized_test_x = np.asarray(stage_signals.test, dtype=np.float32)
                x_min = np.asarray(train_x.min(), dtype=np.float32)
                x_max = np.asarray(train_x.max(), dtype=np.float32)
                signal_transform = "physical_celsius"
            group_train = data.train.parameters[train_mask][:, list(group)]
            group_val = data.validation.parameters[val_mask][:, list(group)]
            theta_min = group_train.min(axis=0, keepdims=True)
            theta_max = group_train.max(axis=0, keepdims=True)
            if self.config.normalize_parameters:
                normalized_train_theta, theta_min, theta_max = _parameter_minmax(
                    group_train, group,
                    bounds=(THERMAL_PARAMETER_BOUNDS[list(group)]
                            if self.config.bounded_parameters else None),
                )
                normalized_val_theta = _parameter_transform(
                    group_val, theta_min, theta_max
                )
            else:
                normalized_train_theta = group_train.astype(np.float32)
                normalized_val_theta = group_val.astype(np.float32)
            preprocessing_seconds = time.perf_counter() - preprocessing_start
            model, flow_state, val_loss, epochs_trained, training_seconds = self._fit_group(
                normalized_train_x, normalized_train_theta,
                normalized_val_x, normalized_val_theta,
                normalized_test_x, data.test.parameters[:, list(group)],
                group, training_key, stage_dir, normalized_target,
                theta_min, theta_max,
                data.real_test.parameters[real_test_index, list(group)], stage,
            )
            evaluation_start = time.perf_counter()
            key, regression_key = jr.split(key)
            validation_predictions = []
            for begin in range(0, len(normalized_test_x), self.config.batch_size):
                regression_key, batch_key = jr.split(regression_key)
                batch_samples = np.asarray(model.sample(
                    jnp.asarray(normalized_test_x[begin:begin + self.config.batch_size]),
                    batch_key, self.config.posterior_samples, flow_state,
                ))
                if self.config.bounded_parameters:
                    batch_samples = _flow_to_bounded(batch_samples)
                if self.config.normalize_parameters:
                    batch_samples = _parameter_inverse(
                        batch_samples, theta_min, theta_max)
                validation_predictions.append(batch_samples)
            validation_samples = np.concatenate(validation_predictions, axis=1)
            evaluation_dir = self.output_dir / f"evaluation_stage_{stage + 1:02d}"
            regression_metrics = _save_stage_regression_evaluation(
                validation_samples, data.test.parameters[:, list(group)], group,
                stage, evaluation_dir, save_pdf=self.config.save_pdf_figures,
            )
            samples = np.asarray(model.sample(
                jnp.asarray(normalized_target), sampling_key,
                self.config.posterior_samples, flow_state,
            ))[:, 0]
            if self.config.bounded_parameters:
                samples = _flow_to_bounded(samples)
            if self.config.normalize_parameters:
                samples = _parameter_inverse(samples, theta_min, theta_max)
            posteriors.append(samples.astype(np.float32))
            means = samples.mean(axis=0)
            fixed.update(zip(group, map(float, means)))
            evaluation_seconds = time.perf_counter() - evaluation_start
            np.save(stage_dir / "posterior_samples.npy", samples.astype(np.float32))
            np.savez(stage_dir / "normalization.npz", x_min=x_min, x_max=x_max,
                     theta_min=theta_min, theta_max=theta_max,
                     signal_transform=signal_transform,
                     parameter_transform=(
                         "physical_unnormalized" if not self.config.normalize_parameters
                         else ("prior_minmax_atanh" if self.config.bounded_parameters
                               else "prior_minmax_-1_1")
                     ))
            (stage_dir / "metadata.json").write_text(json.dumps({
                "group": list(group), "fixed_before_stage": sorted(set(fixed) - set(group)),
                "signal_source": stage_signals.source,
                "inference_target": self.config.inference_target,
                "posterior_semantics": (
                    "Original-prior group marginals under the recorded simulator/surrogate law; "
                    "samples from different groups are not joint draws."
                    if self.config.inference_target == "prior_marginal" else
                    "Conditional approximation under fixed estimates or the recorded proposal."
                ),
                "training_samples_after_proposal_correction": int(train_mask.sum()),
                "validation_samples_after_proposal_correction": int(val_mask.sum()),
                "validation_loss": val_loss,
                "regression_evaluation": regression_metrics,
            }, indent=2))
            timing = {
                "stage": stage + 1,
                "group": list(group),
                "epochs_trained": epochs_trained,
                "generation_seconds": generation_seconds,
                "preprocessing_seconds": preprocessing_seconds,
                "training_seconds": training_seconds,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": time.perf_counter() - stage_start,
            }
            timings.append(timing)
            (stage_dir / "timing.json").write_text(json.dumps(timing, indent=2))
            print(f"stage {stage + 1} group={group} timing={timing}", flush=True)
            if self.tracker is not None and not self.tracker_loss_only:
                tracked = {
                    "workflow_step": self._tracking_step,
                    "stage_result/stage": stage + 1,
                    "stage_result/best_validation_loss": val_loss,
                    "stage_result/epochs_trained": epochs_trained,
                    "stage_result/training_seconds": training_seconds,
                    "stage_result/evaluation_seconds": evaluation_seconds,
                    "stage_result/total_seconds": timing["total_seconds"],
                }
                tracked.update({f"stage_result/parameter_{index}_estimate": float(value)
                                for index, value in zip(group, means)})
                self.tracker.log(tracked, step=self._tracking_step)

        estimate = np.full(parameter_dim, np.nan, dtype=np.float32)
        for index, value in fixed.items():
            estimate[index] = value
        result = CalibrationResult(
            estimate, data.real_test.parameters[real_test_index].copy(),
            tuple(posteriors), self.config.groups[start_stage:stage_count]
        )
        np.savez(self.output_dir / "calibration_result.npz", estimate=result.estimate,
                 truth=result.truth, relative_error_percent=result.relative_error_percent)
        (self.output_dir / "config.json").write_text(json.dumps(asdict(self.config), indent=2))
        timing_summary = {
            "workflow_seconds": time.perf_counter() - workflow_start,
            "total_generation_seconds": sum(row["generation_seconds"] for row in timings),
            "total_training_seconds": sum(row["training_seconds"] for row in timings),
            "total_evaluation_seconds": sum(row["evaluation_seconds"] for row in timings),
            "stages": timings,
        }
        (self.output_dir / "timing_summary.json").write_text(json.dumps(timing_summary, indent=2))
        if self.tracker is not None and not self.tracker_loss_only:
            self.tracker.summary["workflow_seconds"] = timing_summary["workflow_seconds"]
            self.tracker.summary["total_training_seconds"] = timing_summary["total_training_seconds"]
            self.tracker.summary["final_mean_relative_error_percent"] = float(
                np.nanmean(result.relative_error_percent)
            )
        with (self.output_dir / "timings.csv").open("w", newline="") as handle:
            fields = tuple(key for key in timings[0] if key != "group") + ("group",)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in timings:
                writer.writerow({**row, "group": " ".join(map(str, row["group"]))})
        return result
