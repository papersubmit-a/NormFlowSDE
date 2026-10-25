"""Bounded, receding-horizon MPPI tracking of sampled reference trajectories.

The planner supplying a reference can be a conditional stochastic interpolant,
a diffusion model, or a conventional path planner. References describe desired
coordinates *after* each simulated action; they do not include the current state.
"""

from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


@dataclass(frozen=True)
class TrackingMPPIConfig:
    """Sampling budget and quadratic tracking costs.

    ``noise_sigma`` is an action-unit standard deviation, either a scalar or one
    value per actuator. The default pair suits linear/angular velocity commands.
    ``normalize_costs=False`` preserves the physical scale of ``temperature``.
    """

    horizon: int = 20
    num_candidates: int = 256
    iterations: int = 3
    temperature: float = 1.0
    noise_sigma: float | tuple[float, ...] = (0.25, 0.6)
    tracking_weight: float = 10.0
    terminal_weight: float = 20.0
    control_weight: float = 0.01
    smoothness_weight: float = 0.1
    normalize_costs: bool = False

    def __post_init__(self):
        for name in ("horizon", "num_candidates", "iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        for name in ("tracking_weight", "terminal_weight", "control_weight", "smoothness_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        sigma = np.asarray(self.noise_sigma)
        if sigma.ndim > 1 or sigma.size == 0 or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("noise_sigma must contain finite nonnegative standard deviations")


class TrajectoryMPPIController:
    """Practical MPPI with deterministic JAX dynamics and reference tracking.

    ``dynamics(state, action)`` returns the next state. ``state_to_reference``
    projects a state into path coordinates and defaults to ``state[:2]``.
    Optional ``running_cost(next_state, action, step, context)`` adds collision
    or task costs; ``context`` may be any dynamic JAX pytree of environment data.
    All callbacks must support JAX tracing and batching and return arrays.

    Gaussian action perturbations receive exponential cost weights, as in the
    repository's practical MPPI baseline. This is not the importance-corrected
    control-affine path-integral derivation. The unperturbed nominal is included
    in every population, and the weighted average is re-scored against the best
    sample so averaging distinct maneuvers cannot silently increase plan cost.

    This stateful wrapper assumes each returned action is applied once. Reset
    between episodes. If every candidate is invalid, it returns the bounded
    zero-action sequence, with zero effective sample size and infinite cost.
    This is a numerical fallback, not a collision-safety guarantee.
    """

    def __init__(
        self,
        dynamics: Callable,
        action_low,
        action_high,
        config: TrackingMPPIConfig = TrackingMPPIConfig(),
        state_to_reference: Callable | None = None,
        running_cost: Callable | None = None,
    ):
        self.dynamics, self.config = dynamics, config
        self.state_to_reference = (
            (lambda state: state[:2]) if state_to_reference is None else state_to_reference
        )
        self.running_cost = running_cost
        if not callable(dynamics) or not callable(self.state_to_reference):
            raise ValueError("dynamics and state_to_reference must be callable")
        if running_cost is not None and not callable(running_cost):
            raise ValueError("running_cost must be callable or None")
        low, high = np.asarray(action_low), np.asarray(action_high)
        if low.ndim != 1 or low.size == 0 or high.shape != low.shape:
            raise ValueError("action bounds must be nonempty vectors with identical shapes")
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(low > high):
            raise ValueError("action bounds must be finite with action_low <= action_high")
        self.low, self.high = jnp.asarray(low, dtype=float), jnp.asarray(high, dtype=float)
        sigma = jnp.asarray(config.noise_sigma, dtype=self.low.dtype)
        if sigma.ndim == 1 and sigma.shape != self.low.shape:
            raise ValueError("noise_sigma must be scalar or have one value per action dimension")
        self.noise_sigma = jnp.broadcast_to(sigma, self.low.shape)
        self._stop = jnp.clip(jnp.zeros_like(self.low), self.low, self.high)
        self.reset()

        def score(state, actions, reference, previous_action, context):
            def step(carry, inputs):
                current_state, last_action = carry
                action, target, index = inputs
                next_state = self.dynamics(current_state, action)
                coordinates = self.state_to_reference(next_state)
                tracking_error = jnp.sum((coordinates - target) ** 2)
                cost = (
                    config.tracking_weight * tracking_error
                    + config.control_weight * jnp.sum(action ** 2)
                    + config.smoothness_weight * jnp.sum((action - last_action) ** 2)
                )
                if self.running_cost is not None:
                    extra_cost = jnp.asarray(self.running_cost(next_state, action, index, context))
                    if extra_cost.ndim != 0:
                        raise ValueError("running_cost must return a scalar")
                    cost = cost + extra_cost
                valid = jnp.all(jnp.isfinite(next_state)) & jnp.all(jnp.isfinite(coordinates))
                return (next_state, action), (cost, tracking_error, valid)

            _, (stage_costs, tracking_errors, valid) = jax.lax.scan(
                step, (state, previous_action),
                (actions, reference, jnp.arange(config.horizon)),
            )
            total = jnp.sum(stage_costs) + config.terminal_weight * tracking_errors[-1]
            return jnp.where(jnp.all(valid) & jnp.isfinite(total), total, jnp.inf)

        def optimize(state, reference, nominal, previous_action, key, context):
            def iteration(carry, iteration_key):
                current_plan, _, _ = carry
                noise = jr.normal(
                    iteration_key, (config.num_candidates - 1,) + current_plan.shape,
                    dtype=current_plan.dtype,
                ) * self.noise_sigma
                samples = jnp.clip(current_plan[None] + noise, self.low, self.high)
                candidates = jnp.concatenate((current_plan[None], samples), axis=0)
                costs = jax.vmap(score, in_axes=(None, 0, None, None, None))(
                    state, candidates, reference, previous_action, context,
                )
                finite = jnp.isfinite(costs)
                any_valid = jnp.any(finite)
                safe_costs = jnp.where(finite, costs, 0.0)
                count = jnp.maximum(jnp.sum(finite), 1)
                # Normalize before computing variance to avoid overflow for
                # very large, but finite, obstacle penalties.
                magnitude = jnp.maximum(jnp.max(jnp.abs(safe_costs)), 1.0)
                scaled_costs = safe_costs / magnitude
                scaled_mean = jnp.sum(scaled_costs) / count
                variance = jnp.sum(jnp.where(finite, (scaled_costs - scaled_mean) ** 2, 0.0)) / count
                scale = jnp.where(
                    config.normalize_costs,
                    jnp.maximum(jnp.sqrt(variance) * magnitude, 1e-6), 1.0,
                )
                minimum = jnp.min(costs)
                logits = jnp.where(finite, -(costs - minimum) / scale / config.temperature, -jnp.inf)
                # All-invalid populations must not produce softmax NaNs.
                weights = jax.nn.softmax(jnp.where(any_valid, logits, jnp.zeros_like(logits)))
                weights = jnp.where(any_valid, weights, 0.0)
                mean = jnp.clip(jnp.sum(weights[:, None, None] * candidates, axis=0), self.low, self.high)
                mean_cost = score(state, mean, reference, previous_action, context)
                best_index = jnp.argmin(costs)
                use_mean = mean_cost <= costs[best_index]
                next_plan = jnp.where(use_mean, mean, candidates[best_index])
                next_cost = jnp.minimum(mean_cost, costs[best_index])
                next_plan = jnp.where(any_valid, next_plan, self._stop[None])
                next_cost = jnp.where(any_valid, next_cost, jnp.inf)
                ess = jnp.where(any_valid, 1.0 / jnp.maximum(jnp.sum(weights ** 2), 1e-30), 0.0)
                return (next_plan, next_cost, ess), None

            (plan, cost, ess), _ = jax.lax.scan(
                iteration,
                (nominal, jnp.asarray(jnp.inf, nominal.dtype), jnp.asarray(0.0, nominal.dtype)),
                jr.split(key, config.iterations),
            )
            return plan, cost, ess

        self._optimize = jax.jit(optimize)

    def reset(self):
        """Discard the warm start, previous applied action, and diagnostics."""
        self.plan = jnp.broadcast_to(self._stop, (self.config.horizon, self.low.size))
        self._previous_action = self._stop
        self._has_plan = False
        self.last_effective_sample_size = 0.0
        self.last_plan_cost = float("inf")

    def command(self, state, reference, key, *, context=None, nominal=None):
        """Return ``(first_action, plan)`` for an ``(H, reference_dim)`` path.

        Explicit ``nominal`` seeds the current horizon without shifting. In its
        absence, the previous plan shifts once and repeats its last action.
        State, reference, and nominal must be finite. Invalid model rollouts or
        nonfinite callback costs are excluded from the sampling distribution.
        """
        state = jnp.asarray(state, dtype=self.low.dtype)
        reference = jnp.asarray(reference, dtype=self.low.dtype)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(np.asarray(state))):
            raise ValueError("state must be a finite nonempty vector")
        coordinates = jnp.asarray(self.state_to_reference(state))
        if coordinates.ndim != 1 or coordinates.size == 0 or not np.all(np.isfinite(np.asarray(coordinates))):
            raise ValueError("state_to_reference must return a finite nonempty coordinate vector")
        if reference.shape != (self.config.horizon, coordinates.size):
            raise ValueError(f"reference must have shape ({self.config.horizon}, {coordinates.size})")
        if not np.all(np.isfinite(np.asarray(reference))):
            raise ValueError("reference must contain only finite values")
        if nominal is not None:
            seed = jnp.asarray(nominal, dtype=self.low.dtype)
            if seed.shape != self.plan.shape or not np.all(np.isfinite(np.asarray(seed))):
                raise ValueError(f"nominal must be finite with shape {self.plan.shape}")
            seed = jnp.clip(seed, self.low, self.high)
        elif self._has_plan:
            seed = jnp.concatenate((self.plan[1:], self.plan[-1:]), axis=0)
        else:
            seed = self.plan
        plan, cost, ess = self._optimize(
            state, reference, seed, self._previous_action, key, context,
        )
        self.plan = plan
        self._previous_action = plan[0]
        self._has_plan = True
        self.last_plan_cost = float(cost)
        self.last_effective_sample_size = float(ess)
        return plan[0], plan
