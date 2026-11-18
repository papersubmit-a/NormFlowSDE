"""Solver-free Neural Stochastic Flow transition model.

This is an independent implementation of Kiyohara, Johns, and Li (2025),
arXiv:2510.25769. Unlike :class:`RealNVPFlowSDE`, this module learns a complete
finite-time transition distribution in one operation; it does not learn local
drift/diffusion fields and does not invoke an SDE solver.

The public ``log_prob`` and ``sample`` methods use physical state/control units.
Internally, states and controls are standardized outside the invertible flow.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Callable, NamedTuple, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax


LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class NSFConfig:
    """Architecture choices not fixed by the NSF paper."""

    state_dim: int
    control_dim: int = 0
    hidden_size: int = 64
    hidden_depth: int = 2
    num_coupling_layers: int = 4
    include_absolute_time: bool = False
    min_scale: float = 1e-4
    max_log_scale: float = 3.0
    zero_time_policy: str = "error"
    dtype: str = "float32"

    def __post_init__(self):
        if self.state_dim < 1 or self.control_dim < 0:
            raise ValueError("state_dim must be positive and control_dim non-negative")
        if self.hidden_size < 1 or self.hidden_depth < 1:
            raise ValueError("hidden_size and hidden_depth must be positive")
        if self.num_coupling_layers < 1:
            raise ValueError("num_coupling_layers must be positive")
        if self.min_scale <= 0 or self.max_log_scale <= 0:
            raise ValueError("scale bounds must be positive")
        if self.zero_time_policy not in {"error", "deterministic"}:
            raise ValueError("zero_time_policy must be 'error' or 'deterministic'")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")


@dataclass(frozen=True)
class NSFTrainingConfig:
    """Reproducible NLL-only optimization settings for Milestone 3."""

    steps: int = 1_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    gradient_clip_norm: float = 1.0
    seed: int = 0

    def __post_init__(self):
        if self.steps < 1 or self.batch_size < 1:
            raise ValueError("steps and batch_size must be positive")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("learning_rate and gradient_clip_norm must be positive")


class _CouplingLayer(eqx.Module):
    """One time-scaled affine coupling from paper Equation (7)."""

    network: eqx.nn.MLP
    mask: jax.Array
    state_dim: int = eqx.field(static=True)
    max_log_scale: float = eqx.field(static=True)

    def __init__(self, state_dim, condition_dim, hidden_size, hidden_depth,
                 max_log_scale, mask, dtype, key):
        self.state_dim = state_dim
        self.max_log_scale = max_log_scale
        self.mask = jnp.asarray(mask, dtype=bool)
        self.network = eqx.nn.MLP(
            in_size=state_dim + condition_dim,
            out_size=2 * state_dim,
            width_size=hidden_size,
            depth=hidden_depth,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            dtype=dtype,
            key=key,
        )

    def _parameters(self, value, condition):
        masked = jnp.where(self.mask, value, 0.0)
        raw_scale, shift = jnp.split(
            self.network(jnp.concatenate((masked, condition))), 2
        )
        log_scale = self.max_log_scale * jnp.tanh(raw_scale)
        transformed = ~self.mask
        return jnp.where(transformed, log_scale, 0.0), jnp.where(transformed, shift, 0.0)

    def forward_and_log_det(self, value, condition, delta_t):
        log_scale, shift = self._parameters(value, condition)
        scaled_log_scale = delta_t * log_scale
        output = jnp.where(
            self.mask,
            value,
            value * jnp.exp(scaled_log_scale) + delta_t * shift,
        )
        return output, jnp.sum(scaled_log_scale)

    def inverse_and_log_det(self, value, condition, delta_t):
        # The conditioning partition is unchanged, so parameters can be
        # evaluated from the output in an affine coupling inverse.
        log_scale, shift = self._parameters(value, condition)
        scaled_log_scale = delta_t * log_scale
        output = jnp.where(
            self.mask,
            value,
            (value - delta_t * shift) * jnp.exp(-scaled_log_scale),
        )
        return output, -jnp.sum(scaled_log_scale)


class NeuralStochasticFlow(eqx.Module):
    """Conditional finite-time transition kernel implemented as an NSF.

    ``x_current`` and ``x_next`` may be ``(D,)`` or ``(..., D)``. A scalar
    ``delta_t`` broadcasts across batches; otherwise it must match the leading
    batch shape. ``sample`` returns ``(S, D)`` for an unbatched condition and
    ``(S, ..., D)`` for batched conditions.
    """

    base_network: eqx.nn.MLP
    coupling_layers: tuple[_CouplingLayer, ...]
    _state_mean: tuple[float, ...] = eqx.field(static=True)
    _state_std: tuple[float, ...] = eqx.field(static=True)
    _control_mean: tuple[float, ...] = eqx.field(static=True)
    _control_std: tuple[float, ...] = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    include_absolute_time: bool = eqx.field(static=True)
    min_scale: float = eqx.field(static=True)
    zero_time_policy: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        control_dim: int = 0,
        *,
        hidden_size: int = 64,
        hidden_depth: int = 2,
        num_coupling_layers: int = 4,
        include_absolute_time: bool = False,
        min_scale: float = 1e-4,
        max_log_scale: float = 3.0,
        zero_time_policy: str = "error",
        dtype: str = "float32",
        state_mean: Optional[jax.Array] = None,
        state_std: Optional[jax.Array] = None,
        control_mean: Optional[jax.Array] = None,
        control_std: Optional[jax.Array] = None,
        key: jax.Array = jr.PRNGKey(0),
    ):
        config = NSFConfig(
            state_dim=state_dim,
            control_dim=control_dim,
            hidden_size=hidden_size,
            hidden_depth=hidden_depth,
            num_coupling_layers=num_coupling_layers,
            include_absolute_time=include_absolute_time,
            min_scale=min_scale,
            max_log_scale=max_log_scale,
            zero_time_policy=zero_time_policy,
            dtype=dtype,
        )
        self.state_dim = config.state_dim
        self.control_dim = config.control_dim
        self.include_absolute_time = config.include_absolute_time
        self.min_scale = config.min_scale
        self.zero_time_policy = config.zero_time_policy
        self.dtype = config.dtype
        parameter_dtype = jnp.dtype(config.dtype)

        state_mean_array, state_std_array = _normalizer_parameters(
            state_mean, state_std, state_dim, "state", parameter_dtype
        )
        control_mean_array, control_std_array = _normalizer_parameters(
            control_mean, control_std, control_dim, "control", parameter_dtype
        )
        self._state_mean = tuple(float(value) for value in np.asarray(state_mean_array))
        self._state_std = tuple(float(value) for value in np.asarray(state_std_array))
        self._control_mean = tuple(float(value) for value in np.asarray(control_mean_array))
        self._control_std = tuple(float(value) for value in np.asarray(control_std_array))
        condition_dim = state_dim + control_dim + 1 + int(include_absolute_time)
        keys = jr.split(key, num_coupling_layers + 1)
        self.base_network = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=2 * state_dim,
            width_size=hidden_size,
            depth=hidden_depth,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            dtype=parameter_dtype,
            key=keys[0],
        )
        masks = [((jnp.arange(state_dim) + layer) % 2) == 0
                 for layer in range(num_coupling_layers)]
        self.coupling_layers = tuple(
            _CouplingLayer(
                state_dim, condition_dim, hidden_size, hidden_depth,
                max_log_scale, mask, parameter_dtype, layer_key,
            )
            for mask, layer_key in zip(masks, keys[1:])
        )

    @classmethod
    def from_config(cls, config: NSFConfig, *, key: jax.Array, **normalization):
        """Construct from a serializable configuration."""
        return cls(**vars(config), key=key, **normalization)

    @property
    def state_mean(self):
        return jnp.asarray(self._state_mean, dtype=jnp.dtype(self.dtype))

    @property
    def state_std(self):
        return jnp.asarray(self._state_std, dtype=jnp.dtype(self.dtype))

    @property
    def control_mean(self):
        return jnp.asarray(self._control_mean, dtype=jnp.dtype(self.dtype))

    @property
    def control_std(self):
        return jnp.asarray(self._control_std, dtype=jnp.dtype(self.dtype))

    def _condition(self, x_current, delta_t, control, t_current):
        pieces = [x_current]
        if self.control_dim:
            pieces.append(control)
        pieces.append(jnp.atleast_1d(delta_t))
        if self.include_absolute_time:
            pieces.append(jnp.atleast_1d(t_current))
        return jnp.concatenate(pieces)

    def _base_parameters(self, x_current, condition, delta_t):
        drift, raw_scale = jnp.split(self.base_network(condition), 2)
        diffusion_scale = jax.nn.softplus(raw_scale) + self.min_scale
        mean = x_current + delta_t * drift
        std = jnp.sqrt(delta_t) * diffusion_scale
        return mean, std

    def _forward_single(self, base_value, condition, delta_t):
        value, log_det = base_value, jnp.asarray(0.0, dtype=base_value.dtype)
        for layer in self.coupling_layers:
            value, increment = layer.forward_and_log_det(value, condition, delta_t)
            log_det = log_det + increment
        return value, log_det

    def _inverse_single(self, next_value, condition, delta_t):
        value, log_det = next_value, jnp.asarray(0.0, dtype=next_value.dtype)
        for layer in reversed(self.coupling_layers):
            value, increment = layer.inverse_and_log_det(value, condition, delta_t)
            log_det = log_det + increment
        return value, log_det

    def _log_prob_single(self, x_next, x_current, delta_t, control, t_current):
        condition = self._condition(x_current, delta_t, control, t_current)
        base_value, inverse_log_det = self._inverse_single(x_next, condition, delta_t)
        mean, std = self._base_parameters(x_current, condition, delta_t)
        z = (base_value - mean) / std
        base_log_prob = jnp.sum(-0.5 * (z**2 + LOG_2PI) - jnp.log(std))
        # Convert normalized-coordinate density back to physical units.
        physical_log_prob = base_log_prob + inverse_log_det - jnp.sum(jnp.log(self.state_std))
        return jnp.where(delta_t > 0, physical_log_prob, -jnp.inf)

    def log_prob(self, x_next, x_current, delta_t, control=None, t_current=None):
        """Evaluate the exact physical-unit transition log density."""
        x_next, x_current, dt, control, time, batch_shape = self._prepare_inputs(
            x_next, x_current, delta_t, control, t_current
        )
        self._eager_validate_time(dt, density=True)
        values = jax.vmap(self._log_prob_single)(x_next, x_current, dt, control, time)
        return values.reshape(batch_shape) if batch_shape else values[0]

    def _sample_single(self, key, x_current, delta_t, control, t_current):
        condition = self._condition(x_current, delta_t, control, t_current)
        mean, std = self._base_parameters(x_current, condition, delta_t)
        base_value = mean + std * jr.normal(key, (self.state_dim,), dtype=x_current.dtype)
        value, _ = self._forward_single(base_value, condition, delta_t)
        return jnp.where(delta_t == 0, x_current, value)

    def sample(self, key, x_current, delta_t, control=None, num_samples=1,
               t_current=None):
        """Draw transitions with fresh subkeys, returning a leading sample axis."""
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive Python integer")
        _, current, dt, control, time, batch_shape = self._prepare_inputs(
            None, x_current, delta_t, control, t_current
        )
        self._eager_validate_time(dt, density=False)
        batch_size = current.shape[0]
        keys = jr.split(key, num_samples * batch_size).reshape(
            num_samples, batch_size, *key.shape
        )
        draw_batch = jax.vmap(self._sample_single)
        samples = jax.vmap(lambda ks: draw_batch(ks, current, dt, control, time))(keys)
        samples = samples * self.state_std + self.state_mean
        return samples.reshape((num_samples,) + batch_shape + (self.state_dim,))

    def transition_sample(self, x_current, key, *, delta_t=1.0, control=None,
                          t_current=None):
        """Single-draw adapter for model-agnostic particle-filter call sites."""
        return self.sample(
            key, x_current, delta_t, control=control, num_samples=1,
            t_current=t_current,
        )[0]

    def sample_rollout(
        self,
        key,
        x_initial,
        delta_t,
        controls=None,
        *,
        num_samples=1,
        t_start=0.0,
    ):
        """Recursively sample a piecewise-constant controlled trajectory.

        ``controls`` has shape ``(H, control_dim)`` and each command is held on
        its corresponding interval. ``delta_t`` is scalar (broadcast over the
        control horizon) or has shape ``(H,)``. Autonomous rollouts require an
        explicit ``(H,)`` time-gap vector. The return shape is ``(S,H,D)`` and
        excludes the supplied initial state.

        A control sequence is never compressed into one direct long-gap NSF
        condition: doing so would change the documented one-interval control
        model and requires a separate sequence encoder.
        """
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive Python integer")
        initial = jnp.asarray(x_initial, dtype=jnp.dtype(self.dtype))
        if initial.shape != (self.state_dim,):
            raise ValueError(f"x_initial must have shape {(self.state_dim,)}")
        gaps = jnp.asarray(delta_t, dtype=jnp.dtype(self.dtype))
        if self.control_dim:
            if controls is None:
                raise ValueError("controls are required when control_dim > 0")
            commands = jnp.asarray(controls, dtype=jnp.dtype(self.dtype))
            if commands.ndim != 2 or commands.shape[-1] != self.control_dim:
                raise ValueError(
                    f"controls must have shape (H, {self.control_dim})"
                )
            horizon = commands.shape[0]
        else:
            if controls is not None:
                commands = jnp.asarray(controls, dtype=jnp.dtype(self.dtype))
                if commands.ndim != 2 or commands.shape[-1] != 0:
                    raise ValueError("autonomous controls must be omitted or shape (H,0)")
                horizon = commands.shape[0]
            else:
                if gaps.ndim != 1:
                    raise ValueError(
                        "autonomous rollout requires a one-dimensional delta_t vector"
                    )
                horizon = gaps.shape[0]
                commands = jnp.empty((horizon, 0), dtype=initial.dtype)
        if horizon < 1:
            raise ValueError("rollout horizon must be positive")
        if gaps.ndim == 0:
            gaps = jnp.broadcast_to(gaps, (horizon,))
        if gaps.shape != (horizon,):
            raise ValueError(f"delta_t must be scalar or have shape {(horizon,)}")
        if not isinstance(gaps, jax.core.Tracer) and np.any(np.asarray(gaps) <= 0):
            raise ValueError("rollout time gaps must be strictly positive")
        start = jnp.asarray(t_start, dtype=initial.dtype)
        start_times = start + jnp.concatenate(
            (jnp.zeros((1,), dtype=initial.dtype), jnp.cumsum(gaps[:-1]))
        )
        step_keys = jr.split(key, horizon)
        particles = jnp.broadcast_to(
            initial, (num_samples, self.state_dim)
        )

        def step(current, inputs):
            step_key, gap, command, current_time = inputs
            control = command if self.control_dim else None
            following = self.sample(
                step_key,
                current,
                gap,
                control=control,
                num_samples=1,
                t_current=current_time,
            )[0]
            return following, following

        _, path = jax.lax.scan(
            step, particles, (step_keys, gaps, commands, start_times)
        )
        return jnp.swapaxes(path, 0, 1)

    def nll_loss(self, x_next, x_current, delta_t, control=None, t_current=None):
        """Mean negative transition log likelihood."""
        return -jnp.mean(self.log_prob(x_next, x_current, delta_t, control, t_current))

    def _prepare_inputs(self, x_next, x_current, delta_t, control, t_current):
        dtype = jnp.dtype(self.dtype)
        current_phys = jnp.asarray(x_current, dtype=dtype)
        if current_phys.ndim < 1 or current_phys.shape[-1] != self.state_dim:
            raise ValueError(f"last state axis must be {self.state_dim}")
        batch_shape = current_phys.shape[:-1]
        current = ((current_phys - self.state_mean) / self.state_std).reshape(-1, self.state_dim)
        batch_size = current.shape[0]

        if x_next is None:
            following = jnp.zeros_like(current)
        else:
            next_array = jnp.asarray(x_next, dtype=dtype)
            if next_array.ndim < 1 or next_array.shape[-1] != self.state_dim:
                raise ValueError(f"last next-state axis must be {self.state_dim}")
            following_phys = jnp.broadcast_to(
                next_array, batch_shape + (self.state_dim,)
            )
            following = ((following_phys - self.state_mean) / self.state_std).reshape(-1, self.state_dim)
        dt = jnp.broadcast_to(
            jnp.asarray(delta_t, dtype=dtype), batch_shape or ()
        ).reshape(batch_size)

        if self.control_dim:
            if control is None:
                raise ValueError("control is required when control_dim > 0")
            control_array = jnp.asarray(control, dtype=dtype)
            if control_array.ndim < 1 or control_array.shape[-1] != self.control_dim:
                raise ValueError(f"last control axis must be {self.control_dim}")
            control_phys = jnp.broadcast_to(
                control_array, batch_shape + (self.control_dim,)
            )
            control_value = ((control_phys - self.control_mean) / self.control_std).reshape(
                batch_size, self.control_dim
            )
        else:
            if control is not None and jnp.asarray(control).shape[-1:] != (0,):
                raise ValueError("control must be omitted when control_dim == 0")
            control_value = jnp.empty((batch_size, 0), dtype=current.dtype)

        if self.include_absolute_time:
            if t_current is None:
                raise ValueError("t_current is required for non-autonomous NSF")
            time = jnp.broadcast_to(
                jnp.asarray(t_current, dtype=dtype), batch_shape or ()
            ).reshape(batch_size)
        else:
            time = jnp.zeros((batch_size,), dtype=current.dtype)
        return following, current, dt, control_value, time, batch_shape

    def _eager_validate_time(self, delta_t, *, density):
        if isinstance(delta_t, jax.core.Tracer):
            return
        concrete = np.asarray(delta_t)
        if np.any(concrete < 0):
            raise ValueError("delta_t must be non-negative")
        if np.any(concrete == 0) and (density or self.zero_time_policy == "error"):
            operation = "log_prob" if density else "sample"
            raise ValueError(f"delta_t must be strictly positive for {operation}")


class NSFBridge(eqx.Module):
    """Endpoint-consistent auxiliary bridge from paper Equations (19)--(21).

    The public API uses the two positive gaps around the intermediate state:
    ``delta_t_before=t_j-t_i`` and ``delta_t_after=t_k-t_j``. States are in
    physical units; the bridge internally shares the transition normalizer.
    """

    base_network: eqx.nn.MLP
    coupling_layers: tuple[_CouplingLayer, ...]
    _state_mean: tuple[float, ...] = eqx.field(static=True)
    _state_std: tuple[float, ...] = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    include_absolute_time: bool = eqx.field(static=True)
    min_scale: float = eqx.field(static=True)
    dtype: str = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_size: int = 64,
        hidden_depth: int = 2,
        num_coupling_layers: int = 4,
        include_absolute_time: bool = False,
        min_scale: float = 1e-4,
        max_log_scale: float = 3.0,
        dtype: str = "float32",
        state_mean: Optional[jax.Array] = None,
        state_std: Optional[jax.Array] = None,
        key: jax.Array = jr.PRNGKey(0),
    ):
        config = NSFConfig(
            state_dim=state_dim,
            hidden_size=hidden_size,
            hidden_depth=hidden_depth,
            num_coupling_layers=num_coupling_layers,
            include_absolute_time=include_absolute_time,
            min_scale=min_scale,
            max_log_scale=max_log_scale,
            dtype=dtype,
        )
        self.state_dim = state_dim
        self.include_absolute_time = include_absolute_time
        self.min_scale = min_scale
        self.dtype = dtype
        parameter_dtype = jnp.dtype(dtype)
        mean_array, std_array = _normalizer_parameters(
            state_mean, state_std, state_dim, "state", parameter_dtype
        )
        self._state_mean = tuple(float(value) for value in np.asarray(mean_array))
        self._state_std = tuple(float(value) for value in np.asarray(std_array))

        # c_br = (x_i, x_k, total_delta_t, tau, optional t_i)
        condition_dim = 2 * state_dim + 2 + int(include_absolute_time)
        keys = jr.split(key, num_coupling_layers + 1)
        self.base_network = eqx.nn.MLP(
            in_size=condition_dim,
            out_size=2 * state_dim,
            width_size=hidden_size,
            depth=hidden_depth,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            dtype=parameter_dtype,
            key=keys[0],
        )
        masks = [
            ((jnp.arange(state_dim) + layer) % 2) == 0
            for layer in range(num_coupling_layers)
        ]
        self.coupling_layers = tuple(
            _CouplingLayer(
                state_dim,
                condition_dim,
                hidden_size,
                hidden_depth,
                max_log_scale,
                mask,
                parameter_dtype,
                layer_key,
            )
            for mask, layer_key in zip(masks, keys[1:])
        )

    @classmethod
    def from_transition(
        cls,
        transition: NeuralStochasticFlow,
        *,
        key: jax.Array,
        hidden_size: int = 64,
        hidden_depth: int = 2,
        num_coupling_layers: int = 4,
        min_scale: float = 1e-4,
        max_log_scale: float = 3.0,
    ):
        """Construct a bridge with the transition's units and time convention."""
        if transition.control_dim:
            raise ValueError(
                "controlled consistency requires a control-path bridge and is not implemented"
            )
        return cls(
            transition.state_dim,
            hidden_size=hidden_size,
            hidden_depth=hidden_depth,
            num_coupling_layers=num_coupling_layers,
            include_absolute_time=transition.include_absolute_time,
            min_scale=min_scale,
            max_log_scale=max_log_scale,
            dtype=transition.dtype,
            state_mean=transition.state_mean,
            state_std=transition.state_std,
            key=key,
        )

    @property
    def state_mean(self):
        return jnp.asarray(self._state_mean, dtype=jnp.dtype(self.dtype))

    @property
    def state_std(self):
        return jnp.asarray(self._state_std, dtype=jnp.dtype(self.dtype))

    def _condition(self, x_start, x_end, total_delta_t, tau, t_start):
        pieces = [
            x_start,
            x_end,
            jnp.atleast_1d(total_delta_t),
            jnp.atleast_1d(tau),
        ]
        if self.include_absolute_time:
            pieces.append(jnp.atleast_1d(t_start))
        return jnp.concatenate(pieces)

    def _base_parameters(self, x_start, x_end, condition, total_delta_t, tau):
        correction, raw_scale = jnp.split(self.base_network(condition), 2)
        alpha = tau * (1.0 - tau)
        mean = x_start + tau * (x_end - x_start) + alpha * correction
        scale = jax.nn.softplus(raw_scale) + self.min_scale
        std = jnp.sqrt(alpha * total_delta_t) * scale
        return mean, std, alpha

    def _forward_single(self, base_value, condition, alpha):
        value, log_det = base_value, jnp.asarray(0.0, dtype=base_value.dtype)
        for layer in self.coupling_layers:
            value, increment = layer.forward_and_log_det(value, condition, alpha)
            log_det = log_det + increment
        return value, log_det

    def _inverse_single(self, middle_value, condition, alpha):
        value, log_det = middle_value, jnp.asarray(0.0, dtype=middle_value.dtype)
        for layer in reversed(self.coupling_layers):
            value, increment = layer.inverse_and_log_det(value, condition, alpha)
            log_det = log_det + increment
        return value, log_det

    def _log_prob_single(
        self, x_middle, x_start, x_end, delta_before, delta_after, t_start
    ):
        total = delta_before + delta_after
        tau = delta_before / total
        condition = self._condition(x_start, x_end, total, tau, t_start)
        mean, std, alpha = self._base_parameters(
            x_start, x_end, condition, total, tau
        )
        base_value, inverse_log_det = self._inverse_single(
            x_middle, condition, alpha
        )
        standardized = (base_value - mean) / std
        base_log_prob = jnp.sum(
            -0.5 * (standardized**2 + LOG_2PI) - jnp.log(std)
        )
        physical = (
            base_log_prob
            + inverse_log_det
            - jnp.sum(jnp.log(self.state_std))
        )
        valid = (delta_before > 0) & (delta_after > 0)
        return jnp.where(valid, physical, -jnp.inf)

    def log_prob(
        self,
        x_middle,
        x_start,
        x_end,
        delta_t_before,
        delta_t_after,
        t_start=None,
    ):
        """Evaluate ``log b(x_j | x_i, x_k, t_i, t_j, t_k)``."""
        prepared = self._prepare_inputs(
            x_middle, x_start, x_end, delta_t_before, delta_t_after, t_start
        )
        middle, start, end, before, after, time, batch_shape = prepared
        self._eager_validate_gaps(before, after, density=True)
        values = jax.vmap(self._log_prob_single)(
            middle, start, end, before, after, time
        )
        return values.reshape(batch_shape) if batch_shape else values[0]

    def _sample_single(self, key, x_start, x_end, before, after, t_start):
        total = before + after
        tau = jnp.where(total > 0, before / total, 0.0)
        condition = self._condition(x_start, x_end, total, tau, t_start)
        mean, std, alpha = self._base_parameters(
            x_start, x_end, condition, total, tau
        )
        base = mean + std * jr.normal(
            key, (self.state_dim,), dtype=x_start.dtype
        )
        middle, _ = self._forward_single(base, condition, alpha)
        middle = jnp.where(before == 0, x_start, middle)
        return jnp.where(after == 0, x_end, middle)

    def sample(
        self,
        key,
        x_start,
        x_end,
        delta_t_before,
        delta_t_after,
        *,
        num_samples=1,
        t_start=None,
    ):
        """Draw bridge states; exact endpoints are deterministic when requested."""
        if not isinstance(num_samples, int) or num_samples < 1:
            raise ValueError("num_samples must be a positive Python integer")
        prepared = self._prepare_inputs(
            None, x_start, x_end, delta_t_before, delta_t_after, t_start
        )
        _, start, end, before, after, time, batch_shape = prepared
        self._eager_validate_gaps(before, after, density=False)
        batch_size = start.shape[0]
        keys = jr.split(key, num_samples * batch_size).reshape(
            num_samples, batch_size, *key.shape
        )
        draw_batch = jax.vmap(self._sample_single)
        samples = jax.vmap(
            lambda sample_keys: draw_batch(
                sample_keys, start, end, before, after, time
            )
        )(keys)
        samples = samples * self.state_std + self.state_mean
        return samples.reshape(
            (num_samples,) + batch_shape + (self.state_dim,)
        )

    def _prepare_inputs(
        self, x_middle, x_start, x_end, delta_before, delta_after, t_start
    ):
        dtype = jnp.dtype(self.dtype)
        start_physical = jnp.asarray(x_start, dtype=dtype)
        if start_physical.ndim < 1 or start_physical.shape[-1] != self.state_dim:
            raise ValueError(f"last start-state axis must be {self.state_dim}")
        batch_shape = start_physical.shape[:-1]
        start = ((start_physical - self.state_mean) / self.state_std).reshape(
            -1, self.state_dim
        )
        batch_size = start.shape[0]

        end_physical = jnp.asarray(x_end, dtype=dtype)
        if end_physical.ndim < 1 or end_physical.shape[-1] != self.state_dim:
            raise ValueError(f"last end-state axis must be {self.state_dim}")
        end_physical = jnp.broadcast_to(
            end_physical, batch_shape + (self.state_dim,)
        )
        end = ((end_physical - self.state_mean) / self.state_std).reshape(
            batch_size, self.state_dim
        )
        if x_middle is None:
            middle = jnp.zeros_like(start)
        else:
            middle_physical = jnp.asarray(x_middle, dtype=dtype)
            if (
                middle_physical.ndim < 1
                or middle_physical.shape[-1] != self.state_dim
            ):
                raise ValueError(f"last middle-state axis must be {self.state_dim}")
            middle_physical = jnp.broadcast_to(
                middle_physical, batch_shape + (self.state_dim,)
            )
            middle = (
                (middle_physical - self.state_mean) / self.state_std
            ).reshape(batch_size, self.state_dim)
        before = jnp.broadcast_to(
            jnp.asarray(delta_before, dtype=dtype), batch_shape or ()
        ).reshape(batch_size)
        after = jnp.broadcast_to(
            jnp.asarray(delta_after, dtype=dtype), batch_shape or ()
        ).reshape(batch_size)
        if self.include_absolute_time:
            if t_start is None:
                raise ValueError("t_start is required for a non-autonomous bridge")
            time = jnp.broadcast_to(
                jnp.asarray(t_start, dtype=dtype), batch_shape or ()
            ).reshape(batch_size)
        else:
            time = jnp.zeros((batch_size,), dtype=dtype)
        return middle, start, end, before, after, time, batch_shape

    @staticmethod
    def _eager_validate_gaps(before, after, *, density):
        if isinstance(before, jax.core.Tracer) or isinstance(after, jax.core.Tracer):
            return
        before_value, after_value = np.asarray(before), np.asarray(after)
        if np.any(before_value < 0) or np.any(after_value < 0):
            raise ValueError("bridge time gaps must be non-negative")
        if np.any(before_value + after_value <= 0):
            raise ValueError("bridge total time must be positive")
        if density and (np.any(before_value == 0) or np.any(after_value == 0)):
            raise ValueError("bridge log_prob requires a strict interior time")


# Concise alias matching the naming style of RealNVPFlowSDE.
NSF = NeuralStochasticFlow


@dataclass(frozen=True)
class NSFConsistencyConfig:
    """Configuration for the variational Chapman--Kolmogorov bounds."""

    mode: str = "bidirectional"
    consistency_weight: float = 1.0
    one_to_two_weight: float = 1.0
    two_to_one_weight: float = 1.0
    bridge_inner_steps: int = 3
    bridge_learning_rate: float = 1e-3
    num_mc_samples: int = 1

    def __post_init__(self):
        if self.mode not in {"none", "one_to_two", "two_to_one", "bidirectional"}:
            raise ValueError("invalid consistency mode")
        if min(
            self.consistency_weight,
            self.one_to_two_weight,
            self.two_to_one_weight,
        ) < 0:
            raise ValueError("consistency weights must be non-negative")
        if self.bridge_inner_steps < 0 or self.bridge_learning_rate <= 0:
            raise ValueError("bridge update settings are invalid")
        if self.num_mc_samples < 1:
            raise ValueError("num_mc_samples must be positive")


class ConsistencyLosses(NamedTuple):
    one_to_two: jax.Array
    two_to_one: jax.Array
    total: jax.Array


class NSFConsistencyTrainingResult(NamedTuple):
    """Result with both optimizer states exposed for checkpoint/resume support."""

    model: NeuralStochasticFlow
    bridge: NSFBridge
    history: Mapping[str, jax.Array]
    model_opt_state: optax.OptState
    bridge_opt_state: optax.OptState


def flow_consistency_loss(
    model: NeuralStochasticFlow,
    bridge: NSFBridge,
    key: jax.Array,
    x_initial: jax.Array,
    delta_t_before: jax.Array,
    delta_t_after: jax.Array,
    *,
    t_start: Optional[jax.Array] = None,
    mode: str = "bidirectional",
    one_to_two_weight: float = 1.0,
    two_to_one_weight: float = 1.0,
    num_mc_samples: int = 1,
) -> ConsistencyLosses:
    """Monte Carlo estimates of paper Equations (8)--(10).

    This implements autonomous or explicitly time-conditioned dynamics only.
    Controlled multi-interval consistency requires a defined control path and is
    deliberately rejected rather than silently reusing one interval's control.
    """
    if model.control_dim:
        raise ValueError(
            "flow consistency for controlled NSF requires a control-path specification"
        )
    if mode not in {"none", "one_to_two", "two_to_one", "bidirectional"}:
        raise ValueError("invalid consistency mode")
    if num_mc_samples < 1:
        raise ValueError("num_mc_samples must be positive")
    if one_to_two_weight < 0 or two_to_one_weight < 0:
        raise ValueError("consistency direction weights must be non-negative")
    if model.state_dim != bridge.state_dim:
        raise ValueError("transition and bridge state dimensions must agree")
    if model.include_absolute_time != bridge.include_absolute_time:
        raise ValueError("transition and bridge time conventions must agree")
    if model._state_mean != bridge._state_mean or model._state_std != bridge._state_std:
        raise ValueError("transition and bridge normalization must agree")

    before = jnp.asarray(delta_t_before, dtype=jnp.dtype(model.dtype))
    after = jnp.asarray(delta_t_after, dtype=jnp.dtype(model.dtype))
    if not isinstance(before, jax.core.Tracer) and not isinstance(
        after, jax.core.Tracer
    ):
        if np.any(np.asarray(before) <= 0) or np.any(np.asarray(after) <= 0):
            raise ValueError("consistency time gaps must be strictly positive")
    total = before + after
    direct_key, middle_key, second_key = jr.split(key, 3)
    zero = jnp.asarray(0.0, dtype=jnp.dtype(model.dtype))

    one_to_two = zero
    if mode in {"one_to_two", "bidirectional"}:
        direct_ends = model.sample(
            direct_key,
            x_initial,
            total,
            num_samples=num_mc_samples,
            t_current=t_start,
        )
        bridge_keys = jr.split(middle_key, num_mc_samples)
        middles = jax.vmap(
            lambda sample_key, x_end: bridge.sample(
                sample_key,
                x_initial,
                x_end,
                before,
                after,
                num_samples=1,
                t_start=t_start,
            )[0]
        )(bridge_keys, direct_ends)

        def one_to_two_terms(x_middle, x_end):
            direct_log_prob = model.log_prob(
                x_end, x_initial, total, t_current=t_start
            )
            bridge_log_prob = bridge.log_prob(
                x_middle,
                x_initial,
                x_end,
                before,
                after,
                t_start=t_start,
            )
            first_log_prob = model.log_prob(
                x_middle, x_initial, before, t_current=t_start
            )
            second_start = None if t_start is None else t_start + before
            second_log_prob = model.log_prob(
                x_end, x_middle, after, t_current=second_start
            )
            return direct_log_prob + bridge_log_prob - first_log_prob - second_log_prob

        one_to_two = jnp.mean(jax.vmap(one_to_two_terms)(middles, direct_ends))

    two_to_one = zero
    if mode in {"two_to_one", "bidirectional"}:
        middles = model.sample(
            middle_key,
            x_initial,
            before,
            num_samples=num_mc_samples,
            t_current=t_start,
        )
        second_keys = jr.split(second_key, num_mc_samples)
        second_start = None if t_start is None else t_start + before
        recursive_ends = jax.vmap(
            lambda sample_key, x_middle: model.sample(
                sample_key,
                x_middle,
                after,
                num_samples=1,
                t_current=second_start,
            )[0]
        )(second_keys, middles)

        def two_to_one_terms(x_middle, x_end):
            first_log_prob = model.log_prob(
                x_middle, x_initial, before, t_current=t_start
            )
            second_log_prob = model.log_prob(
                x_end, x_middle, after, t_current=second_start
            )
            bridge_log_prob = bridge.log_prob(
                x_middle,
                x_initial,
                x_end,
                before,
                after,
                t_start=t_start,
            )
            direct_log_prob = model.log_prob(
                x_end, x_initial, total, t_current=t_start
            )
            return first_log_prob + second_log_prob - bridge_log_prob - direct_log_prob

        two_to_one = jnp.mean(
            jax.vmap(two_to_one_terms)(middles, recursive_ends)
        )

    total_loss = one_to_two_weight * one_to_two + two_to_one_weight * two_to_one
    return ConsistencyLosses(one_to_two, two_to_one, total_loss)


@eqx.filter_jit
def make_step_nsf_bridge(
    bridge,
    model,
    key,
    x_initial,
    delta_t_before,
    delta_t_after,
    opt_state,
    optimizer,
    config: NSFConsistencyConfig,
    t_start=None,
):
    """One auxiliary bridge update with the transition parameters fixed."""
    def loss_fn(candidate):
        return flow_consistency_loss(
            model,
            candidate,
            key,
            x_initial,
            delta_t_before,
            delta_t_after,
            t_start=t_start,
            mode=config.mode,
            one_to_two_weight=config.one_to_two_weight,
            two_to_one_weight=config.two_to_one_weight,
            num_mc_samples=config.num_mc_samples,
        ).total

    loss, gradients = eqx.filter_value_and_grad(loss_fn)(bridge)
    updates, opt_state = optimizer.update(gradients, opt_state, bridge)
    bridge = eqx.apply_updates(bridge, updates)
    return bridge, opt_state, loss


@eqx.filter_jit
def make_step_nsf_consistency(
    model,
    bridge,
    key,
    x_next,
    x_current,
    delta_t,
    x_initial,
    delta_t_before,
    delta_t_after,
    opt_state,
    optimizer,
    config: NSFConsistencyConfig,
    control=None,
    t_current=None,
    triplet_t_start=None,
):
    """One NSF update for NLL plus a selected consistency bound."""
    def loss_fn(candidate):
        nll = candidate.nll_loss(
            x_next, x_current, delta_t, control, t_current
        )
        consistency = flow_consistency_loss(
            candidate,
            bridge,
            key,
            x_initial,
            delta_t_before,
            delta_t_after,
            t_start=triplet_t_start,
            mode=config.mode,
            one_to_two_weight=config.one_to_two_weight,
            two_to_one_weight=config.two_to_one_weight,
            num_mc_samples=config.num_mc_samples,
        )
        total = nll + config.consistency_weight * consistency.total
        return total, (nll, consistency)

    (total, auxiliary), gradients = eqx.filter_value_and_grad(
        loss_fn, has_aux=True
    )(model)
    updates, opt_state = optimizer.update(gradients, opt_state, model)
    model = eqx.apply_updates(model, updates)
    nll, consistency = auxiliary
    return model, opt_state, total, nll, consistency


def _normalizer_parameters(mean, std, dimension, name, dtype):
    if dimension == 0:
        return jnp.empty((0,), dtype=dtype), jnp.empty((0,), dtype=dtype)
    mean_array = (
        jnp.zeros((dimension,), dtype=dtype)
        if mean is None else jnp.asarray(mean, dtype=dtype)
    )
    std_array = (
        jnp.ones((dimension,), dtype=dtype)
        if std is None else jnp.asarray(std, dtype=dtype)
    )
    if mean_array.shape != (dimension,) or std_array.shape != (dimension,):
        raise ValueError(f"{name} normalizers must have shape {(dimension,)}")
    if np.any(np.asarray(std_array) <= 0):
        raise ValueError(f"{name}_std must be strictly positive")
    return mean_array, std_array


def fit_normalization(states, controls=None, *, min_std=1e-6):
    """Fit per-feature normalizers from training data only."""
    states = jnp.asarray(states)
    axes = tuple(range(states.ndim - 1))
    state_mean = jnp.mean(states, axis=axes)
    state_std = jnp.maximum(jnp.std(states, axis=axes), min_std)
    if controls is None:
        return state_mean, state_std, None, None
    controls = jnp.asarray(controls)
    control_axes = tuple(range(controls.ndim - 1))
    control_mean = jnp.mean(controls, axis=control_axes)
    control_std = jnp.maximum(jnp.std(controls, axis=control_axes), min_std)
    return state_mean, state_std, control_mean, control_std


def make_transition_pairs(states, times, controls=None):
    """Flatten adjacent transitions without crossing trajectory boundaries."""
    states, times = jnp.asarray(states), jnp.asarray(times)
    if states.ndim != 3 or times.shape != states.shape[:2]:
        raise ValueError("expected states=(B,T,D) and times=(B,T)")
    delta_t = jnp.diff(times, axis=1)
    if np.any(np.asarray(delta_t) <= 0):
        raise ValueError("times must increase strictly within each trajectory")
    batch = {
        "x_current": states[:, :-1].reshape(-1, states.shape[-1]),
        "x_next": states[:, 1:].reshape(-1, states.shape[-1]),
        "delta_t": delta_t.reshape(-1),
    }
    if controls is not None:
        controls = jnp.asarray(controls)
        if controls.shape[:2] != (states.shape[0], states.shape[1] - 1):
            raise ValueError("controls must have shape (B,T-1,U)")
        batch["control"] = controls.reshape(-1, controls.shape[-1])
    return batch


@eqx.filter_jit
def make_step_nsf(model, x_next, x_current, delta_t, opt_state, optimizer,
                  control=None, t_current=None):
    """One Optax maximum-likelihood update, compatible with project trainers."""
    loss, grads = eqx.filter_value_and_grad(
        lambda candidate: candidate.nll_loss(
            x_next, x_current, delta_t, control, t_current
        )
    )(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def train_nsf_nll(
    model: NeuralStochasticFlow,
    transitions: Mapping[str, jax.Array],
    config: NSFTrainingConfig = NSFTrainingConfig(),
    *,
    progress: Optional[Callable[[int, jax.Array], None]] = None,
):
    """Fit an NSF by transition maximum likelihood.

    Required entries are ``x_current``, ``x_next``, and ``delta_t``. Optional
    ``control`` and ``t_current`` arrays are routed to the corresponding model
    arguments. Sampling is with replacement and driven only by ``config.seed``.
    Returns the trained model and a device array containing every step loss.
    """
    required = {"x_current", "x_next", "delta_t"}
    missing = required.difference(transitions)
    if missing:
        raise ValueError(f"missing transition arrays: {sorted(missing)}")
    arrays = {name: jnp.asarray(value) for name, value in transitions.items()}
    num_transitions = arrays["x_current"].shape[0]
    if num_transitions < 1:
        raise ValueError("at least one transition is required")
    for name, value in arrays.items():
        if value.shape[0] != num_transitions:
            raise ValueError(f"{name} has inconsistent leading dimension")

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip_norm),
        optax.adam(config.learning_rate),
    )
    trainable = eqx.filter(model, eqx.is_inexact_array)
    opt_state = optimizer.init(trainable)
    key = jr.key(config.seed)
    losses = []
    for step in range(config.steps):
        key, batch_key = jr.split(key)
        indices = jr.randint(
            batch_key, (config.batch_size,), 0, num_transitions
        )
        model, opt_state, loss = make_step_nsf(
            model,
            arrays["x_next"][indices],
            arrays["x_current"][indices],
            arrays["delta_t"][indices],
            opt_state,
            optimizer,
            arrays.get("control", None)[indices]
            if "control" in arrays else None,
            arrays.get("t_current", None)[indices]
            if "t_current" in arrays else None,
        )
        losses.append(loss)
        if progress is not None:
            progress(step, loss)
    return model, jnp.stack(losses)


def sample_uniform_consistency_triplets(
    key: jax.Array,
    x_initial: jax.Array,
    *,
    total_time_min: float = 0.1,
    total_time_max: float = 1.0,
    interior_margin: float = 0.05,
    t_start: Optional[jax.Array] = None,
):
    """Attach strictly interior uniformly sampled time triplets to states.

    This is the uniform component of the Appendix B time sampler. Combining it
    with observed trajectory end times is dataset-specific and is intentionally
    left to experiment code rather than hidden in the model API.
    """
    states = jnp.asarray(x_initial)
    if states.ndim < 2:
        raise ValueError("x_initial must have shape (N, ..., state_dim)")
    if states.shape[0] < 1:
        raise ValueError("at least one initial state is required")
    if not 0 < total_time_min <= total_time_max:
        raise ValueError("total-time bounds must be positive and ordered")
    if not 0 < interior_margin < 0.5:
        raise ValueError("interior_margin must lie strictly between 0 and 0.5")
    total_key, fraction_key = jr.split(key)
    shape = (states.shape[0],)
    dtype = states.dtype
    total = jr.uniform(
        total_key,
        shape,
        dtype=dtype,
        minval=total_time_min,
        maxval=total_time_max,
    )
    fraction = jr.uniform(
        fraction_key,
        shape,
        dtype=dtype,
        minval=interior_margin,
        maxval=1.0 - interior_margin,
    )
    triplets = {
        "x_initial": states,
        "delta_t_before": fraction * total,
        "delta_t_after": (1.0 - fraction) * total,
    }
    if t_start is not None:
        start = jnp.asarray(t_start, dtype=dtype)
        triplets["t_start"] = jnp.broadcast_to(start, shape)
    return triplets


def train_nsf_with_consistency(
    model: NeuralStochasticFlow,
    bridge: NSFBridge,
    transitions: Mapping[str, jax.Array],
    triplets: Mapping[str, jax.Array],
    training_config: NSFTrainingConfig = NSFTrainingConfig(),
    consistency_config: NSFConsistencyConfig = NSFConsistencyConfig(),
    *,
    progress: Optional[Callable[[int, Mapping[str, jax.Array]], None]] = None,
) -> NSFConsistencyTrainingResult:
    """Train with NLL and selected variational flow-consistency bounds.

    Each outer iteration first performs ``bridge_inner_steps`` updates of the
    auxiliary bridge and then one transition-model update. The optimizers and
    their states are separate. Every objective component is retained in the
    returned history, including the last bridge loss at each outer iteration.
    """
    transition_required = {"x_current", "x_next", "delta_t"}
    triplet_required = {"x_initial", "delta_t_before", "delta_t_after"}
    missing_transitions = transition_required.difference(transitions)
    missing_triplets = triplet_required.difference(triplets)
    if missing_transitions:
        raise ValueError(
            f"missing transition arrays: {sorted(missing_transitions)}"
        )
    if missing_triplets:
        raise ValueError(f"missing triplet arrays: {sorted(missing_triplets)}")
    if model.control_dim:
        raise ValueError(
            "consistency training for controlled NSF requires a control path"
        )
    if model.state_dim != bridge.state_dim:
        raise ValueError("transition and bridge state dimensions must agree")
    if model.include_absolute_time != bridge.include_absolute_time:
        raise ValueError("transition and bridge time conventions must agree")
    if model._state_mean != bridge._state_mean or model._state_std != bridge._state_std:
        raise ValueError("transition and bridge normalization must agree")

    transition_arrays = {
        name: jnp.asarray(value) for name, value in transitions.items()
    }
    triplet_arrays = {name: jnp.asarray(value) for name, value in triplets.items()}
    num_transitions = transition_arrays["x_current"].shape[0]
    num_triplets = triplet_arrays["x_initial"].shape[0]
    if num_transitions < 1 or num_triplets < 1:
        raise ValueError("at least one transition and one triplet are required")
    for name, value in transition_arrays.items():
        if value.shape[0] != num_transitions:
            raise ValueError(f"{name} has inconsistent leading dimension")
    for name, value in triplet_arrays.items():
        if value.shape[0] != num_triplets:
            raise ValueError(f"{name} has inconsistent leading dimension")
    before = np.asarray(triplet_arrays["delta_t_before"])
    after = np.asarray(triplet_arrays["delta_t_after"])
    if np.any(before <= 0) or np.any(after <= 0):
        raise ValueError("consistency triplet gaps must be strictly positive")

    model_optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.gradient_clip_norm),
        optax.adam(training_config.learning_rate),
    )
    bridge_optimizer = optax.chain(
        optax.clip_by_global_norm(training_config.gradient_clip_norm),
        optax.adam(consistency_config.bridge_learning_rate),
    )
    model_opt_state = model_optimizer.init(
        eqx.filter(model, eqx.is_inexact_array)
    )
    bridge_opt_state = bridge_optimizer.init(
        eqx.filter(bridge, eqx.is_inexact_array)
    )
    history = {
        "total": [],
        "nll": [],
        "flow": [],
        "one_to_two": [],
        "two_to_one": [],
        "bridge": [],
    }
    key = jr.key(training_config.seed)
    for step in range(training_config.steps):
        key, transition_key, triplet_key, model_key = jr.split(key, 4)
        transition_indices = jr.randint(
            transition_key,
            (training_config.batch_size,),
            0,
            num_transitions,
        )
        triplet_indices = jr.randint(
            triplet_key,
            (training_config.batch_size,),
            0,
            num_triplets,
        )
        x_initial = triplet_arrays["x_initial"][triplet_indices]
        delta_before = triplet_arrays["delta_t_before"][triplet_indices]
        delta_after = triplet_arrays["delta_t_after"][triplet_indices]
        triplet_time = (
            triplet_arrays["t_start"][triplet_indices]
            if "t_start" in triplet_arrays else None
        )

        bridge_loss = jnp.asarray(0.0, dtype=jnp.dtype(model.dtype))
        should_update_bridge = (
            consistency_config.mode != "none"
            and consistency_config.consistency_weight > 0
        )
        if should_update_bridge:
            for _ in range(consistency_config.bridge_inner_steps):
                key, bridge_key = jr.split(key)
                bridge, bridge_opt_state, bridge_loss = make_step_nsf_bridge(
                    bridge,
                    model,
                    bridge_key,
                    x_initial,
                    delta_before,
                    delta_after,
                    bridge_opt_state,
                    bridge_optimizer,
                    consistency_config,
                    triplet_time,
                )

        model, model_opt_state, total, nll, consistency = (
            make_step_nsf_consistency(
                model,
                bridge,
                model_key,
                transition_arrays["x_next"][transition_indices],
                transition_arrays["x_current"][transition_indices],
                transition_arrays["delta_t"][transition_indices],
                x_initial,
                delta_before,
                delta_after,
                model_opt_state,
                model_optimizer,
                consistency_config,
                transition_arrays["control"][transition_indices]
                if "control" in transition_arrays else None,
                transition_arrays["t_current"][transition_indices]
                if "t_current" in transition_arrays else None,
                triplet_time,
            )
        )
        metrics = {
            "total": total,
            "nll": nll,
            "flow": consistency.total,
            "one_to_two": consistency.one_to_two,
            "two_to_one": consistency.two_to_one,
            "bridge": bridge_loss,
        }
        for name, value in metrics.items():
            history[name].append(value)
        if progress is not None:
            progress(step, metrics)

    stacked_history = {
        name: jnp.stack(values) for name, values in history.items()
    }
    return NSFConsistencyTrainingResult(
        model,
        bridge,
        stacked_history,
        model_opt_state,
        bridge_opt_state,
    )


def sample_ou_transition_data(
    key: jax.Array,
    process: "OrnsteinUhlenbeck",
    num_transitions: int,
    state_dim: int,
    *,
    delta_t_min: float = 0.05,
    delta_t_max: float = 1.0,
    initial_std: float = 1.5,
    dtype=jnp.float32,
):
    """Generate independent analytic OU transition pairs for validation."""
    if num_transitions < 1 or state_dim < 1:
        raise ValueError("num_transitions and state_dim must be positive")
    if not 0 < delta_t_min <= delta_t_max or initial_std <= 0:
        raise ValueError("time bounds and initial_std must be positive and ordered")
    current_key, time_key, transition_key = jr.split(key, 3)
    process_mean = jnp.broadcast_to(
        jnp.asarray(process.mean, dtype=dtype), (state_dim,)
    )
    x_current = process_mean + initial_std * jr.normal(
        current_key, (num_transitions, state_dim), dtype=dtype
    )
    delta_t = jr.uniform(
        time_key,
        (num_transitions,),
        dtype=dtype,
        minval=delta_t_min,
        maxval=delta_t_max,
    )
    x_next = process.sample(transition_key, x_current, delta_t)
    return {
        "x_current": x_current,
        "x_next": x_next,
        "delta_t": delta_t,
    }


def evaluate_ou_moments(
    model: NeuralStochasticFlow,
    process: "OrnsteinUhlenbeck",
    key: jax.Array,
    x_current: jax.Array,
    delta_t: float,
    *,
    num_samples: int = 10_000,
):
    """Compare NSF Monte Carlo moments with the analytic OU transition."""
    samples = model.sample(key, x_current, delta_t, num_samples=num_samples)
    predicted_mean = jnp.mean(samples, axis=0)
    predicted_variance = jnp.var(samples, axis=0)
    analytic_mean = process.transition_mean(x_current, jnp.asarray(delta_t))
    analytic_variance = jnp.broadcast_to(
        process.transition_variance(jnp.asarray(delta_t)), predicted_variance.shape
    )
    return {
        "predicted_mean": predicted_mean,
        "predicted_variance": predicted_variance,
        "analytic_mean": analytic_mean,
        "analytic_variance": analytic_variance,
        "mean_absolute_error": jnp.mean(jnp.abs(predicted_mean - analytic_mean)),
        "variance_absolute_error": jnp.mean(
            jnp.abs(predicted_variance - analytic_variance)
        ),
    }


@dataclass(frozen=True)
class OrnsteinUhlenbeck:
    """Analytic OU reference: ``dX=kappa*(mean-X)dt+diffusion*dW``."""

    mean: object = 0.0
    kappa: object = 1.0
    diffusion: object = 1.0

    def __post_init__(self):
        if np.any(np.asarray(self.kappa) <= 0) or np.any(np.asarray(self.diffusion) < 0):
            raise ValueError("kappa must be positive and diffusion non-negative")

    def transition_mean(self, x_current, delta_t):
        mean, kappa = jnp.asarray(self.mean), jnp.asarray(self.kappa)
        dt = jnp.asarray(delta_t)
        decay = jnp.exp(-kappa * (dt[..., None] if dt.ndim else dt))
        return mean + (jnp.asarray(x_current) - mean) * decay

    def transition_variance(self, delta_t):
        kappa, diffusion = jnp.asarray(self.kappa), jnp.asarray(self.diffusion)
        dt = jnp.asarray(delta_t)
        dt = dt[..., None] if dt.ndim else dt
        return -(diffusion**2) * jnp.expm1(-2.0 * kappa * dt) / (2.0 * kappa)

    def log_prob(self, x_next, x_current, delta_t):
        dt = jnp.asarray(delta_t)
        if not isinstance(dt, jax.core.Tracer) and np.any(np.asarray(dt) <= 0):
            raise ValueError("delta_t must be strictly positive for log_prob")
        variance = self.transition_variance(dt)
        residual = jnp.asarray(x_next) - self.transition_mean(x_current, dt)
        return jnp.sum(-0.5 * (jnp.log(2 * jnp.pi * variance) + residual**2 / variance), -1)

    def sample(self, key, x_current, delta_t):
        location = self.transition_mean(x_current, delta_t)
        return location + jnp.sqrt(self.transition_variance(delta_t)) * jr.normal(
            key, location.shape, dtype=location.dtype
        )


__all__: Sequence[str] = (
    "ConsistencyLosses",
    "NSF",
    "NSFBridge",
    "NSFConfig",
    "NSFConsistencyConfig",
    "NSFConsistencyTrainingResult",
    "NSFTrainingConfig",
    "NeuralStochasticFlow",
    "OrnsteinUhlenbeck",
    "evaluate_ou_moments",
    "fit_normalization",
    "flow_consistency_loss",
    "make_step_nsf",
    "make_step_nsf_bridge",
    "make_step_nsf_consistency",
    "make_transition_pairs",
    "sample_uniform_consistency_triplets",
    "sample_ou_transition_data",
    "train_nsf_nll",
    "train_nsf_with_consistency",
)
