"""Projected gradient MPC for learned models and differentiable Brax envs."""

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from .sampling_mpc import observe_latent, remember_transition


Array = jnp.ndarray
CostFn = Callable[[Array, Array, int], Array]
BraxCostFn = Callable[[Any, Array, int], Array]


@dataclass(frozen=True)
class GradientMPCConfig:
    horizon: int = 20
    iterations: int = 80
    learning_rate: float = 0.03
    momentum: float = 0.9

    def __post_init__(self):
        if self.horizon < 1 or self.iterations < 1:
            raise ValueError("horizon and iterations must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")


def _projected_gradient(initial_controls, value_and_grad, objective_args,
                        low, high, config):
    controls = initial_controls
    velocity = jnp.zeros_like(controls)
    for _ in range(config.iterations):
        _, gradient = value_and_grad(controls, *objective_args)
        velocity = config.momentum * velocity + (1.0 - config.momentum) * gradient
        controls = jnp.clip(
            controls - config.learning_rate * velocity, low, high
        )
    return controls


class GradientMPC:
    """Projected gradient MPC through a model's nominal dynamics."""

    def __init__(self, model, action_low: Array, action_high: Array,
                 stage_cost: CostFn,
                 config: GradientMPCConfig = GradientMPCConfig()):
        self.model, self.stage_cost, self.config = model, stage_cost, config
        self.low, self.high = jnp.asarray(action_low), jnp.asarray(action_high)
        if self.low.shape != self.high.shape or self.low.ndim != 1:
            raise ValueError("action bounds must be equal-shaped vectors")
        if bool(jnp.any(self.low >= self.high)):
            raise ValueError("every lower action bound must be below its upper bound")
        self.plan = jnp.zeros((config.horizon, self.low.size))
        self._value_and_grad = jax.jit(jax.value_and_grad(self._objective))

    def reset(self):
        self.plan = jnp.zeros_like(self.plan)
        self._previous_transition = None

    def _objective(self, controls: Array, state_context: Array,
                   previous_controls: Array, initial_latent=None) -> Array:
        if self.model.history_len == 1:
            states = self.model.rollout_deterministic(state_context, controls,
                **({"initial_latent": initial_latent} if hasattr(self.model, "infer_latent") else {}))
        else:
            states = self.model.rollout_deterministic_history(
                state_context, previous_controls, controls,
                **({"initial_latent": initial_latent} if hasattr(self.model, "infer_latent") else {})
            )
        return jnp.sum(jax.vmap(self.stage_cost)(
            states, controls, jnp.arange(controls.shape[0])
        ))

    def command(self, state_context: Array,
                previous_controls: Array = None) -> Tuple[Array, Array]:
        """Optimize from x_t (H=1) or H states and H-1 past controls."""
        if previous_controls is None:
            previous_controls = jnp.zeros(
                (max(0, self.model.history_len - 1), self.low.size)
            )
        latent = observe_latent(self, state_context, previous_controls)
        warm_start = jnp.concatenate([self.plan[1:], self.plan[-1:]], axis=0)
        self.plan = _projected_gradient(
            warm_start, self._value_and_grad,
            (state_context, previous_controls, latent), self.low, self.high, self.config,
        )
        remember_transition(self, state_context, previous_controls, self.plan[0])
        return self.plan[0], self.plan.copy()


class BraxGradientMPC:
    """Gradient MPC that differentiates directly through ``brax.envs.Env.step``.

    ``stage_cost`` receives the *next* Brax ``State``, action, and horizon index.
    The controller accepts/returns Brax State pytrees without flattening their
    physics state, so contacts and pipeline internals remain differentiable.
    """

    def __init__(self, env, stage_cost: BraxCostFn,
                 config: GradientMPCConfig = GradientMPCConfig(),
                 action_low=-1.0, action_high=1.0):
        try:
            import brax  # noqa: F401 - validate the optional dependency here
        except ImportError as error:
            raise ImportError("BraxGradientMPC requires `pip install brax`") from error
        self.env, self.stage_cost, self.config = env, stage_cost, config
        action_size = int(env.action_size)
        self.low = jnp.broadcast_to(jnp.asarray(action_low), (action_size,))
        self.high = jnp.broadcast_to(jnp.asarray(action_high), (action_size,))
        if bool(jnp.any(self.low >= self.high)):
            raise ValueError("every lower action bound must be below its upper bound")
        self.plan = jnp.zeros((config.horizon, action_size))
        self._value_and_grad = jax.jit(jax.value_and_grad(self._objective))

    def reset(self):
        self.plan = jnp.zeros_like(self.plan)
        self._previous_transition = None

    def _objective(self, controls, initial_state):
        def step(state, inputs):
            action, time = inputs
            next_state = self.env.step(state, action)
            return next_state, self.stage_cost(next_state, action, time)
        _, costs = jax.lax.scan(
            step, initial_state, (controls, jnp.arange(controls.shape[0]))
        )
        return jnp.sum(costs)

    def command(self, brax_state) -> Tuple[Array, Array]:
        warm_start = jnp.concatenate([self.plan[1:], self.plan[-1:]], axis=0)
        self.plan = _projected_gradient(
            warm_start, self._value_and_grad, (brax_state,),
            self.low, self.high, self.config,
        )
        return self.plan[0], self.plan.copy()
