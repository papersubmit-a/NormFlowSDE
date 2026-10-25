"""Common physical-unit transition interface and ensemble benchmark metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NeuralStochasticFlow,
)
from bayesinf.applications.optimal_control.controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
)


class TransitionKernel(Protocol):
    """Physical-unit interface shared by finite-time and integrated models."""

    state_dim: int
    control_dim: int
    supports_direct_long_gap: bool

    def sample(self, key, x_current, delta_t, control=None, num_samples=1,
               t_current=None): ...
    def log_prob(self, x_next, x_current, delta_t, control=None,
                 t_current=None): ...
    def sample_rollout(self, key, x_initial, delta_t, controls=None,
                       num_samples=1, t_start=0.0): ...


@dataclass(frozen=True)
class BenchmarkTrainingConfig:
    """Matched exact-NLL optimization settings for transition adapters."""

    steps: int = 1_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning rate and gradient clip must be positive")


class NSFTransitionAdapter(eqx.Module):
    """Thin common-interface adapter around a physical-unit NSF."""

    model: NeuralStochasticFlow
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    supports_direct_long_gap: bool = eqx.field(static=True)

    def __init__(self, model: NeuralStochasticFlow):
        self.model = model
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim
        self.supports_direct_long_gap = model.control_dim == 0

    def sample(self, key, x_current, delta_t, control=None, num_samples=1,
               t_current=None):
        return self.model.sample(
            key, x_current, delta_t, control, num_samples, t_current
        )

    def log_prob(self, x_next, x_current, delta_t, control=None,
                 t_current=None):
        return self.model.log_prob(
            x_next, x_current, delta_t, control, t_current
        )

    def sample_rollout(self, key, x_initial, delta_t, controls=None,
                       num_samples=1, t_start=0.0):
        return self.model.sample_rollout(
            key,
            x_initial,
            delta_t,
            controls,
            num_samples=num_samples,
            t_start=t_start,
        )


class NormFlowSDETransitionAdapter(eqx.Module):
    """Physical-unit adapter for the causal controlled RealNVP FlowSDE."""

    model: ControlledRealNVPFlowSDE
    _state_mean: tuple[float, ...] = eqx.field(static=True)
    _state_std: tuple[float, ...] = eqx.field(static=True)
    _control_mean: tuple[float, ...] = eqx.field(static=True)
    _control_std: tuple[float, ...] = eqx.field(static=True)
    solver_name: str = eqx.field(static=True)
    rollout_mode: str = eqx.field(static=True)
    ito_moment_samples: int = eqx.field(static=True)
    ito_jitter: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    supports_direct_long_gap: bool = eqx.field(static=True, default=False)

    def __init__(
        self,
        model: ControlledRealNVPFlowSDE,
        *,
        state_mean,
        state_std,
        control_mean,
        control_std,
        solver_name="euler",
        rollout_mode="normflowsde",
        ito_moment_samples=64,
        ito_jitter=1e-6,
    ):
        self.model = model
        self.state_dim = model.state_dim
        self.control_dim = model.control_dim
        if solver_name not in {"euler", "heun", "reversible_heun", "midpoint"}:
            raise ValueError("unsupported common-benchmark solver")
        if rollout_mode not in {"normflowsde", "itosde"}:
            raise ValueError("rollout_mode must be 'normflowsde' or 'itosde'")
        if ito_moment_samples < 2:
            raise ValueError("ito_moment_samples must be at least two")
        self.solver_name = solver_name
        self.rollout_mode = rollout_mode
        self.ito_moment_samples = int(ito_moment_samples)
        self.ito_jitter = float(ito_jitter)
        self._state_mean, self._state_std = _normalizer_tuples(
            state_mean, state_std, self.state_dim, "state"
        )
        self._control_mean, self._control_std = _normalizer_tuples(
            control_mean, control_std, self.control_dim, "control"
        )

    @property
    def state_mean(self):
        return jnp.asarray(self._state_mean)

    @property
    def state_std(self):
        return jnp.asarray(self._state_std)

    @property
    def control_mean(self):
        return jnp.asarray(self._control_mean)

    @property
    def control_std(self):
        return jnp.asarray(self._control_std)

    def _validate_delta_t(self, delta_t):
        gaps = jnp.asarray(delta_t)
        valid = jnp.isfinite(gaps) & (jnp.abs(gaps - self.model.dt) <= 1e-6 * self.model.dt + 1e-8)
        return eqx.error_if(gaps, jnp.any(~valid),
            "NormFlowSDE adapter delta_t must equal its trained fixed model.dt")

    def sample(self, key, x_current, delta_t, control=None, num_samples=1,
               t_current=None):
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive Python integer")
        delta_t = self._validate_delta_t(delta_t)
        current = jnp.asarray(x_current) + 0.0 * jnp.sum(delta_t)
        if current.ndim < 1 or current.shape[-1] != self.state_dim:
            raise ValueError(f"last state axis must be {self.state_dim}")
        batch_shape = current.shape[:-1]
        current = current.reshape((-1, self.state_dim))
        batch_size = current.shape[0]
        if control is None:
            raise ValueError("control is required by controlled NormFlowSDE")
        command = jnp.broadcast_to(
            jnp.asarray(control), batch_shape + (self.control_dim,)
        ).reshape((batch_size, self.control_dim))
        if t_current is None:
            time = jnp.zeros((batch_size,), dtype=current.dtype)
        else:
            time = jnp.broadcast_to(
                jnp.asarray(t_current, dtype=current.dtype), batch_shape or ()
            ).reshape((batch_size,))
        normalized_state = (current - self.state_mean) / self.state_std
        normalized_control = (command - self.control_mean) / self.control_std
        keys = jr.split(key, batch_size)

        def draw(state, action, start_time, sample_key):
            start_step = start_time / self.model.dt
            return self.model.sample_rollout(
                state,
                action[None],
                sample_key,
                n_particles=num_samples,
                start_step=start_step,
                solver_name=self.solver_name,
                rollout_mode=self.rollout_mode,
                ito_moment_samples=self.ito_moment_samples,
                ito_jitter=self.ito_jitter,
            )[:, 0]

        # vmap produces (B,S,D); public kernels use a leading sample axis.
        samples = jax.vmap(draw)(normalized_state, normalized_control, time, keys)
        samples = jnp.swapaxes(samples, 0, 1)
        samples = samples * self.state_std + self.state_mean
        return samples.reshape((num_samples,) + batch_shape + (self.state_dim,))

    def log_prob(self, x_next, x_current, delta_t, control=None,
                 t_current=None):
        """Density of the selected sampler, when the exact Euler law is used."""
        if self.solver_name != "euler" or self.rollout_mode != "normflowsde":
            raise ValueError("Exact log_prob is available only for the Euler normflowsde sampler; use euler_reference_log_prob to score an explicitly different reference kernel")
        return self.euler_reference_log_prob(x_next, x_current, delta_t, control, t_current)

    def euler_reference_log_prob(self, x_next, x_current, delta_t, control=None,
                                 t_current=None):
        """Exact Euler flow-noise density, independent of selected rollout mode."""
        delta_t = self._validate_delta_t(delta_t)
        current, following = jnp.asarray(x_current) + 0.0 * jnp.sum(delta_t), jnp.asarray(x_next)
        if current.shape != following.shape or current.shape[-1] != self.state_dim:
            raise ValueError("current and next states must have matching shapes")
        batch_shape = current.shape[:-1]
        if control is None:
            raise ValueError("control is required by controlled NormFlowSDE")
        command = jnp.broadcast_to(
            jnp.asarray(control), batch_shape + (self.control_dim,)
        )
        time = jnp.zeros(batch_shape or (), dtype=current.dtype) if t_current is None else jnp.broadcast_to(
            jnp.asarray(t_current, dtype=current.dtype), batch_shape or ()
        )
        normalized_current = (current - self.state_mean) / self.state_std
        normalized_following = (following - self.state_mean) / self.state_std
        normalized_control = (command - self.control_mean) / self.control_std
        normalized_log_prob = self.model.transition_log_prob(
            normalized_following,
            normalized_current,
            normalized_control,
            time,
        )
        return normalized_log_prob - jnp.sum(jnp.log(self.state_std))

    def sample_rollout(self, key, x_initial, delta_t, controls=None,
                       num_samples=1, t_start=0.0):
        if controls is None:
            raise ValueError("controls are required by controlled NormFlowSDE")
        initial = jnp.asarray(x_initial)
        commands = jnp.asarray(controls)
        if initial.shape != (self.state_dim,):
            raise ValueError(f"x_initial must have shape {(self.state_dim,)}")
        if commands.ndim != 2 or commands.shape[-1] != self.control_dim:
            raise ValueError(
                f"controls must have shape (H, {self.control_dim})"
            )
        horizon = commands.shape[0]
        gaps = jnp.asarray(delta_t)
        if gaps.ndim == 0:
            gaps = jnp.broadcast_to(gaps, (horizon,))
        if gaps.shape != (horizon,):
            raise ValueError(f"delta_t must be scalar or have shape {(horizon,)}")
        gaps = self._validate_delta_t(gaps)
        normalized_initial = (initial + 0.0 * jnp.sum(gaps) - self.state_mean) / self.state_std
        normalized_controls = (commands - self.control_mean) / self.control_std
        start_step = jnp.asarray(t_start) / self.model.dt
        paths = self.model.sample_rollout(
            normalized_initial,
            normalized_controls,
            key,
            n_particles=num_samples,
            start_step=start_step,
            solver_name=self.solver_name,
            rollout_mode=self.rollout_mode,
            ito_moment_samples=self.ito_moment_samples,
            ito_jitter=self.ito_jitter,
        )
        return paths * self.state_std + self.state_mean


def _normalizer_tuples(mean, std, dimension, name):
    mean, std = np.asarray(mean, dtype=float), np.asarray(std, dtype=float)
    if mean.shape != (dimension,) or std.shape != (dimension,):
        raise ValueError(f"{name} normalizers must have shape {(dimension,)}")
    if np.any(std <= 0) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError(f"{name} normalizers must be finite with positive std")
    return tuple(float(value) for value in mean), tuple(float(value) for value in std)


def ensemble_transition_metrics(
    samples: jax.Array,
    target: jax.Array,
    *,
    reference_samples: Optional[jax.Array] = None,
    coverage_levels: Sequence[float] = (0.5, 0.8, 0.95),
):
    """Proper scores and moment diagnostics for a leading-axis ensemble."""
    samples, target = jnp.asarray(samples), jnp.asarray(target)
    if samples.ndim < 2 or samples.shape[1:] != target.shape:
        raise ValueError("samples must have shape (S, ...) matching target")
    mean = jnp.mean(samples, axis=0)
    error = mean - target
    residual_norm = jnp.linalg.norm(samples - target[None], axis=-1)
    pairwise = samples[:, None] - samples[None, :]
    pairwise_norm = jnp.linalg.norm(pairwise, axis=-1)
    energy = jnp.mean(residual_norm) - 0.5 * jnp.mean(pairwise_norm)
    marginal_crps = jnp.mean(jnp.abs(samples - target[None])) - 0.5 * jnp.mean(
        jnp.abs(pairwise)
    )
    metrics = {
        "rmse": jnp.sqrt(jnp.mean(error**2)),
        "mae": jnp.mean(jnp.abs(error)),
        "energy_score": energy,
        "crps": marginal_crps,
    }
    for level in coverage_levels:
        lower_q, upper_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0
        lower = jnp.quantile(samples, lower_q, axis=0)
        upper = jnp.quantile(samples, upper_q, axis=0)
        label = int(round(100 * level))
        metrics[f"coverage_{label}"] = jnp.mean(
            (target >= lower) & (target <= upper)
        )
        metrics[f"width_{label}"] = jnp.mean(upper - lower)

    if reference_samples is None:
        metrics["marginal_wasserstein"] = jnp.mean(
            jnp.abs(samples - target[None])
        )
        metrics["covariance_error"] = jnp.asarray(jnp.nan, dtype=samples.dtype)
    else:
        reference = jnp.asarray(reference_samples, dtype=samples.dtype)
        if reference.ndim != samples.ndim or reference.shape[1:] != target.shape:
            raise ValueError("reference_samples must match sample event dimensions")
        quantile_count = min(samples.shape[0], reference.shape[0])
        quantiles = (jnp.arange(quantile_count, dtype=samples.dtype) + 0.5) / quantile_count
        model_quantiles = jnp.quantile(samples, quantiles, axis=0)
        reference_quantiles = jnp.quantile(reference, quantiles, axis=0)
        metrics["marginal_wasserstein"] = jnp.mean(
            jnp.abs(model_quantiles - reference_quantiles)
        )

        def average_covariance(ensemble):
            flat = ensemble.reshape((ensemble.shape[0], -1, ensemble.shape[-1]))
            centered = flat - jnp.mean(flat, axis=0, keepdims=True)
            covariances = jnp.einsum("snd,sne->nde", centered, centered)
            covariances = covariances / max(ensemble.shape[0] - 1, 1)
            return jnp.mean(covariances, axis=0)

        model_covariance = average_covariance(samples)
        reference_covariance = average_covariance(reference)
        metrics["covariance_error"] = jnp.linalg.norm(
            model_covariance - reference_covariance
        )
    return metrics


def count_trainable_parameters(model) -> int:
    """Count inexact array parameters for budget reporting."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    return int(sum(leaf.size for leaf in leaves))


def train_transition_kernel_nll(
    kernel,
    transitions: Mapping[str, jax.Array],
    config: BenchmarkTrainingConfig = BenchmarkTrainingConfig(),
):
    """Train any common-interface kernel on identical seeded minibatches."""
    required = {"x_current", "x_next", "delta_t"}
    missing = required.difference(transitions)
    if missing:
        raise ValueError(f"missing transition arrays: {sorted(missing)}")
    arrays = {name: jnp.asarray(value) for name, value in transitions.items()}
    count = arrays["x_current"].shape[0]
    if count < 1:
        raise ValueError("at least one transition is required")
    for name, value in arrays.items():
        if value.shape[0] != count:
            raise ValueError(f"{name} has inconsistent leading dimension")
    if kernel.control_dim and "control" not in arrays:
        raise ValueError("controlled kernels require a control array")

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(eqx.filter(kernel, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(candidate, state, indices):
        def objective(current):
            values = current.log_prob(
                arrays["x_next"][indices],
                arrays["x_current"][indices],
                arrays["delta_t"][indices],
                arrays["control"][indices] if "control" in arrays else None,
                arrays["t_current"][indices] if "t_current" in arrays else None,
            )
            return -jnp.mean(values)

        loss, gradients = eqx.filter_value_and_grad(objective)(candidate)
        updates, state = optimizer.update(gradients, state, candidate)
        return eqx.apply_updates(candidate, updates), state, loss

    key = jr.key(config.seed)
    losses = []
    for _ in range(config.steps):
        key, batch_key = jr.split(key)
        indices = jr.randint(batch_key, (config.batch_size,), 0, count)
        kernel, opt_state, loss = step(kernel, opt_state, indices)
        losses.append(loss)
    return kernel, jnp.stack(losses)


__all__ = (
    "BenchmarkTrainingConfig",
    "NSFTransitionAdapter",
    "NormFlowSDETransitionAdapter",
    "TransitionKernel",
    "count_trainable_parameters",
    "ensemble_transition_metrics",
    "train_transition_kernel_nll",
)
