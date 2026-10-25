"""Particle rollouts, CEM stochastic MPC, and an MPPI baseline."""

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr


Array = jnp.ndarray
CostFn = Callable[[Array, Array, int], Array]


@dataclass(frozen=True)
class SamplingMPCConfig:
    horizon: int = 20
    num_candidates: int = 256
    num_particles: int = 8
    iterations: int = 5
    elite_fraction: float = 0.1
    initial_std: float = 0.25
    min_std: float = 0.01
    temperature: float = 1.0
    risk_weight: float = 0.0
    smoothness_weight: float = 0.0
    smoothness_scale: float = 1.0
    normalize_costs: bool = False


def _candidate_cost(model, x_history, u_history, candidates, keys,
                    num_particles: int, stage_cost: CostFn,
                    risk_weight: float, smoothness_weight: float = 0.0,
                    smoothness_scale: Array | float = 1.0, initial_latent=None) -> Array:
    def evaluate(actions, key):
        rollouts = model.sample_rollout(
            x_history, u_history, actions, key, num_particles,
            **({"initial_latent": initial_latent} if hasattr(model, "infer_latent") else {})
        )
        time = jnp.arange(actions.shape[0])
        costs = jax.vmap(
            lambda trajectory: jnp.sum(jax.vmap(stage_cost)(
                trajectory, actions, time
            ))
        )(rollouts)
        # Mean + lambda*std is a simple risk-sensitive objective. lambda=0 is
        # risk-neutral Monte Carlo MPC and still uses the full flow uncertainty.
        action_rate = jnp.diff(actions, axis=0) / jnp.asarray(smoothness_scale)
        smoothness_cost = smoothness_weight * jnp.sum(jnp.mean(action_rate ** 2, axis=-1))
        return jnp.mean(costs) + risk_weight * jnp.std(costs) + smoothness_cost
    return jax.vmap(evaluate)(candidates, keys)


def mppi_weighted_update(nominal, candidates, costs, config):
    """Shared bounded exponential-cost update for both rollout adapters."""
    scale = jnp.where(config.normalize_costs, jnp.maximum(jnp.std(costs), 1e-6), 1.0)
    weights = jax.nn.softmax(-(costs - jnp.min(costs)) / (config.temperature * scale))
    return jnp.sum(weights[:, None, None] * candidates, axis=0), weights


def observe_latent(controller, states, controls):
    """Condition a persistent flow SSM on the transition just observed."""
    model = controller.model
    if not hasattr(model, "infer_latent"):
        return None
    previous = getattr(controller, "_previous_transition", None)
    if previous is None:
        return model._initial_latent(jnp.atleast_2d(states), controls)
    history, past_controls, action = previous
    return model.infer_latent(history, action, jnp.atleast_2d(states)[-1], past_controls)


def remember_transition(controller, states, controls, action):
    if hasattr(controller.model, "infer_latent"):
        controller._previous_transition = (jnp.atleast_2d(states), controls, action)


class CEMStochasticMPC:
    """Cross-entropy MPC over action sequences and learned flow particles."""

    def __init__(self, model, action_low: Array, action_high: Array,
                 stage_cost: CostFn, config: SamplingMPCConfig = SamplingMPCConfig()):
        self.model, self.stage_cost, self.config = model, stage_cost, config
        self.low, self.high = jnp.asarray(action_low), jnp.asarray(action_high)
        self.mean = jnp.zeros((config.horizon, self.low.size))
        n_elite = max(1, int(config.num_candidates * config.elite_fraction))

        # Compile the complete optimizer.  Keeping sampling, scoring, elite
        # selection, and distribution updates in one scan avoids a device
        # round-trip for every CEM iteration.
        def optimize(x_history, u_history, previous_mean, key, initial_latent):
            mean = jnp.concatenate(
                [previous_mean[1:], previous_mean[-1:]], axis=0
            )
            std = jnp.full_like(mean, config.initial_std)

            def iteration(carry, iteration_key):
                current_mean, current_std = carry
                sample_key, rollout_key = jr.split(iteration_key)
                candidates = jnp.clip(
                    current_mean + current_std * jr.normal(
                        sample_key,
                        (config.num_candidates,) + current_mean.shape,
                    ),
                    self.low, self.high,
                )
                costs = _candidate_cost(
                    self.model, x_history, u_history, candidates,
                    jr.split(rollout_key, config.num_candidates),
                    config.num_particles, self.stage_cost,
                    config.risk_weight, config.smoothness_weight,
                    config.smoothness_scale, initial_latent,
                )
                # top_k is a partial selection, avoiding a full O(N log N)
                # argsort when only the elite candidates are needed.
                _, elite_indices = jax.lax.top_k(-costs, n_elite)
                elite = candidates[elite_indices]
                next_mean = jnp.mean(elite, axis=0)
                next_std = jnp.maximum(jnp.std(elite, axis=0), config.min_std)
                return (next_mean, next_std), None

            (mean, _), _ = jax.lax.scan(
                iteration, (mean, std), jr.split(key, config.iterations)
            )
            return jnp.clip(mean, self.low, self.high)

        self._optimize = jax.jit(optimize)
        # Retain the scoring kernel for callers that use it for diagnostics.
        self._cost = jax.jit(lambda x_history, u_history, candidates, keys, initial_latent=None:
            _candidate_cost(
                self.model, x_history, u_history, candidates, keys,
                self.config.num_particles, self.stage_cost,
                self.config.risk_weight, self.config.smoothness_weight,
                self.config.smoothness_scale, initial_latent,
            ))

    def reset(self):
        self.mean = jnp.zeros_like(self.mean)
        self._previous_transition = None

    def command(self, x_history: Array, u_history: Array,
                key: jr.PRNGKey) -> Tuple[Array, Array]:
        latent = observe_latent(self, x_history, u_history)
        self.mean = self._optimize(x_history, u_history, self.mean, key, latent)
        remember_transition(self, x_history, u_history, self.mean[0])
        return self.mean[0], self.mean.copy()


class MPPIController:
    """Standard path-integral-style sampling baseline using the same model.

    This implements exponential cost weighting of Gaussian perturbations.  The
    optional control-noise likelihood/covariance correction from the original
    control-affine MPPI derivation is intentionally omitted, making this the
    common practical, model-agnostic MPPI variant.
    """

    def __init__(self, model, action_low: Array, action_high: Array,
                 stage_cost: CostFn, config: SamplingMPCConfig = SamplingMPCConfig()):
        self.model, self.stage_cost, self.config = model, stage_cost, config
        self.low, self.high = jnp.asarray(action_low), jnp.asarray(action_high)
        self.plan = jnp.zeros((config.horizon, self.low.size))
        self.last_effective_sample_size = float(config.num_candidates)
        self._cost = jax.jit(lambda x_history, u_history, candidates, keys, initial_latent=None:
            _candidate_cost(
                self.model, x_history, u_history, candidates, keys,
                self.config.num_particles, self.stage_cost,
                self.config.risk_weight, self.config.smoothness_weight,
                self.config.smoothness_scale, initial_latent,
            ))

    def reset(self):
        self.plan = jnp.zeros_like(self.plan)
        self._previous_transition = None

    def command(self, x_history: Array, u_history: Array,
                key: jr.PRNGKey) -> Tuple[Array, Array]:
        cfg = self.config
        latent = observe_latent(self, x_history, u_history)
        nominal = jnp.concatenate([self.plan[1:], self.plan[-1:]], axis=0)
        weights = jnp.ones((cfg.num_candidates,)) / cfg.num_candidates
        for iteration in range(cfg.iterations):
            iteration_key = jr.fold_in(key, iteration)
            noise_key, rollout_key = jr.split(iteration_key)
            noise = cfg.initial_std * jr.normal(
                noise_key, (cfg.num_candidates,) + nominal.shape
            )
            candidates = jnp.clip(nominal + noise, self.low, self.high)
            costs = self._cost(
                x_history, u_history, candidates,
                jr.split(rollout_key, cfg.num_candidates), latent,
            )
            nominal, weights = mppi_weighted_update(nominal, candidates, costs, cfg)
            nominal = jnp.clip(nominal, self.low, self.high)
        self.plan = nominal
        remember_transition(self, x_history, u_history, self.plan[0])
        self.last_effective_sample_size = float(
            1.0 / jnp.sum(weights ** 2)
        )
        return self.plan[0], self.plan.copy()
