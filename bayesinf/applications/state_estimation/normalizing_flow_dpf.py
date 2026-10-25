"""Normalizing flow-based differentiable particle filtering (NF-DPF).

Independent implementation of Chen and Li, arXiv:2403.01499v2,
https://arxiv.org/abs/2403.01499. Transition, observation-conditioned proposal,
and measurement distributions all have normalized, tractable densities. The
importance update includes both transition and proposal densities (Eq. 5).

The multivariate flows use conditional RealNVP layers. One-dimensional flows
use invertible sinh-arcsinh layers because affine coupling alone is Gaussian
in one dimension. Residual transitions, conditional transition couplings, and
supervised component-NLL warmup are implementation choices, not a reproduction
of any one experiment in the paper. Inputs must be standardized by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp


@dataclass(frozen=True)
class NFDPFConfig:
    state_dim: int
    observation_dim: int
    control_dim: int = 0
    hidden_size: int = 64
    hidden_depth: int = 2
    num_flow_layers: int = 4
    max_log_scale: float = 2.0
    base_scale: float = 0.2

    def __post_init__(self):
        if self.state_dim < 1 or self.observation_dim < 1 or self.control_dim < 0:
            raise ValueError("state/observation dimensions must be positive; control_dim >= 0")
        if self.hidden_size < 1 or self.hidden_depth < 1 or self.num_flow_layers < 1:
            raise ValueError("hidden_size, hidden_depth, and num_flow_layers must be positive")
        if self.max_log_scale <= 0 or self.base_scale <= 0:
            raise ValueError("max_log_scale and base_scale must be positive")


def _near_identity_mlp(in_size, out_size, width, depth, key):
    network = eqx.nn.MLP(in_size, out_size, width, depth, activation=jax.nn.silu, key=key)
    # Small, nonzero final weights retain gradients in every hidden layer.
    return eqx.tree_at(
        lambda m: (m.layers[-1].weight, m.layers[-1].bias), network,
        (0.1 * network.layers[-1].weight, jnp.zeros_like(network.layers[-1].bias)),
    )


def _log_cosh(value):
    return jnp.logaddexp(value, -value) - math.log(2.0)


class _FlowLayer(eqx.Module):
    network: eqx.nn.MLP
    dimension: int = eqx.field(static=True)
    parity: int = eqx.field(static=True)
    max_log_scale: float = eqx.field(static=True)

    def __init__(self, dimension, condition_dim, width, depth, parity, max_log_scale, key):
        self.dimension = dimension
        self.parity = parity
        self.max_log_scale = max_log_scale
        self.network = _near_identity_mlp(
            dimension + condition_dim, 4 if dimension == 1 else 2 * dimension,
            width, depth, key,
        )

    def _parameters(self, value, condition):
        # For scalar flows no input coordinate may condition its own parameters.
        mask = ((jnp.arange(self.dimension) + self.parity) % 2 == 0)
        if self.dimension == 1:
            mask = jnp.zeros((1,), dtype=bool)
        raw = self.network(jnp.concatenate((jnp.where(mask, value, 0), condition)))
        if self.dimension == 1:
            shift, scale, skew, tail = raw
            return shift, self.max_log_scale * jnp.tanh(scale), 0.25 * jnp.tanh(skew), 0.25 * jnp.tanh(tail)
        raw_scale, shift = jnp.split(raw, 2)
        return (
            jnp.where(mask, 0, self.max_log_scale * jnp.tanh(raw_scale)),
            jnp.where(mask, 0, shift),
        )

    def forward(self, value, condition):
        if self.dimension == 1:
            shift, log_scale, skew, log_tail = self._parameters(value, condition)
            inside = (jnp.arcsinh(value) + skew) * jnp.exp(log_tail)
            output = jnp.sinh(inside) * jnp.exp(log_scale) + shift
            log_det = log_scale + log_tail + _log_cosh(inside) - 0.5 * jnp.log1p(value**2)
            return output, jnp.sum(log_det)
        log_scale, shift = self._parameters(value, condition)
        return value * jnp.exp(log_scale) + shift, jnp.sum(log_scale)

    def inverse(self, value, condition):
        if self.dimension == 1:
            shift, log_scale, skew, log_tail = self._parameters(value, condition)
            scaled = (value - shift) * jnp.exp(-log_scale)
            inside = jnp.arcsinh(scaled) * jnp.exp(-log_tail) - skew
            output = jnp.sinh(inside)
            log_det = -log_scale - log_tail + _log_cosh(inside) - 0.5 * jnp.log1p(scaled**2)
            return output, jnp.sum(log_det)
        log_scale, shift = self._parameters(value, condition)
        return (value - shift) * jnp.exp(-log_scale), -jnp.sum(log_scale)


class ConditionalNormalizingFlow(eqx.Module):
    """Conditional density with exact inverses and reparameterized sampling.

    ``condition`` has shape ``(..., condition_dim)``. Samples have the same
    leading dimensions; ``log_prob`` broadcasts value/condition batch shapes.
    ``forward_and_log_det`` and ``inverse_and_log_det`` act on single vectors
    and include the learned Gaussian affine transform.
    """

    base_network: eqx.nn.MLP
    layers: tuple[_FlowLayer, ...]
    value_dim: int = eqx.field(static=True)
    condition_dim: int = eqx.field(static=True)
    base_scale: float = eqx.field(static=True)
    max_log_scale: float = eqx.field(static=True)

    def __init__(self, value_dim, condition_dim, *, hidden_size=64, hidden_depth=2,
                 num_flow_layers=4, base_scale=1.0, max_log_scale=2.0, key):
        if value_dim < 1 or condition_dim < 1:
            raise ValueError("value_dim and condition_dim must be positive")
        if min(hidden_size, hidden_depth, num_flow_layers) < 1:
            raise ValueError("flow architecture sizes must be positive")
        if base_scale <= 0 or max_log_scale <= 0:
            raise ValueError("flow scales must be positive")
        self.value_dim = value_dim
        self.condition_dim = condition_dim
        self.base_scale = base_scale
        self.max_log_scale = max_log_scale
        keys = jr.split(key, num_flow_layers + 1)
        self.base_network = _near_identity_mlp(condition_dim, 2 * value_dim, hidden_size, hidden_depth, keys[0])
        self.layers = tuple(
            _FlowLayer(value_dim, condition_dim, hidden_size, hidden_depth, i % 2, max_log_scale, k)
            for i, k in enumerate(keys[1:])
        )

    def _base_parameters(self, condition):
        location, raw_scale = jnp.split(self.base_network(condition), 2)
        return location, math.log(self.base_scale) + self.max_log_scale * jnp.tanh(raw_scale)

    def forward_and_log_det(self, noise, condition):
        location, log_scale = self._base_parameters(condition)
        value, log_det = location + jnp.exp(log_scale) * noise, jnp.sum(log_scale)
        for layer in self.layers:
            value, increment = layer.forward(value, condition)
            log_det = log_det + increment
        return value, log_det

    def inverse_and_log_det(self, value, condition):
        log_det = jnp.zeros((), dtype=value.dtype)
        for layer in reversed(self.layers):
            value, increment = layer.inverse(value, condition)
            log_det = log_det + increment
        location, log_scale = self._base_parameters(condition)
        return (value - location) * jnp.exp(-log_scale), log_det - jnp.sum(log_scale)

    def sample_and_log_prob(self, key, condition):
        condition = jnp.asarray(condition)
        if condition.ndim < 1 or condition.shape[-1] != self.condition_dim:
            raise ValueError("condition has the wrong trailing dimension")
        leading = condition.shape[:-1]
        noise = jr.normal(key, (*leading, self.value_dim), dtype=condition.dtype)
        values, log_det = jax.vmap(self.forward_and_log_det)(
            noise.reshape(-1, self.value_dim), condition.reshape(-1, self.condition_dim),
        )
        base_lp = -0.5 * jnp.sum(noise**2 + math.log(2 * math.pi), axis=-1)
        return values.reshape(*leading, self.value_dim), base_lp - log_det.reshape(leading)

    def log_prob(self, value, condition):
        value, condition = jnp.asarray(value), jnp.asarray(condition)
        if value.ndim < 1 or value.shape[-1] != self.value_dim:
            raise ValueError("value has the wrong trailing dimension")
        if condition.ndim < 1 or condition.shape[-1] != self.condition_dim:
            raise ValueError("condition has the wrong trailing dimension")
        leading = jnp.broadcast_shapes(value.shape[:-1], condition.shape[:-1])
        value = jnp.broadcast_to(value, (*leading, self.value_dim))
        condition = jnp.broadcast_to(condition, (*leading, self.condition_dim))
        noise, log_det = jax.vmap(self.inverse_and_log_det)(
            value.reshape(-1, self.value_dim), condition.reshape(-1, self.condition_dim),
        )
        return (-0.5 * jnp.sum(noise**2 + math.log(2 * math.pi), axis=-1) + log_det).reshape(leading)


class NFDPFResult(NamedTuple):
    particles: jax.Array
    weights: jax.Array
    mean: jax.Array
    log_evidence: jax.Array
    ess: jax.Array


def _resample(particles, log_weights, key, method, soft_alpha, ot_epsilon, ot_iterations):
    """Return particles and *unnormalized* inherited log masses.

    Keeping the soft-resampling mass w(a)/(N r(a)) is essential: normalizing it
    prematurely changes the evidence estimator. Categorical ancestor indices
    are nondifferentiable; the correction and selected particles retain paths.
    """
    count = particles.shape[0]
    uniform = jnp.full_like(log_weights, -math.log(count))
    if method == "ot":
        differences = particles[:, None, :] - particles[None, :, :]
        costs = jnp.mean(differences**2, axis=-1)
        scale = jax.lax.stop_gradient(jnp.maximum(jnp.mean(costs), 1e-6))
        log_kernel = -costs / (ot_epsilon * scale)
        # Source marginal is the weighted cloud; destination is uniform.
        def sinkhorn(_, potentials):
            _, v = potentials
            u = log_weights - logsumexp(log_kernel + v[None, :], axis=1)
            v = uniform - logsumexp(log_kernel + u[:, None], axis=0)
            return u, v
        u, v = jax.lax.fori_loop(0, ot_iterations, sinkhorn, (uniform, uniform))
        log_plan = log_kernel + u[:, None] + v[None, :]
        # Explicit column normalization keeps barycentres convex even before
        # the finite Sinkhorn iteration has exactly matched both marginals.
        transform = jax.nn.softmax(log_plan, axis=0).T
        return transform @ particles, uniform
    if method == "soft":
        log_sampling = jnp.logaddexp(
            math.log(soft_alpha) + log_weights,
            jnp.asarray(-jnp.inf if soft_alpha == 1 else math.log1p(-soft_alpha)) - math.log(count),
        )
        indices = jr.categorical(key, log_sampling, shape=(count,))
        return particles[indices], log_weights[indices] - log_sampling[indices] - math.log(count)
    if method == "systematic":
        positions = (jr.uniform(key, ()) + jnp.arange(count)) / count
        cumulative = jnp.cumsum(jnp.exp(log_weights)).at[-1].set(1.0)
        indices = jnp.minimum(jnp.searchsorted(cumulative, positions), count - 1)
    else:  # multinomial / categorical evaluation
        indices = jr.categorical(key, log_weights, shape=(count,))
    return particles[indices], uniform


class NormalizingFlowDPF(eqx.Module):
    """Learned transition, proposal, and measurement models in a causal PF.

    Controls at index t drive x[t] -> x[t+1], and observations[t] measure
    x[t+1]. Returned estimates therefore exclude the supplied initial state.
    """

    transition: ConditionalNormalizingFlow
    proposal: ConditionalNormalizingFlow
    measurement: ConditionalNormalizingFlow
    config: NFDPFConfig = eqx.field(static=True)

    def __init__(self, config: NFDPFConfig, *, key):
        self.config = config
        transition_key, proposal_key, measurement_key = jr.split(key, 3)
        shared = dict(hidden_size=config.hidden_size, hidden_depth=config.hidden_depth,
                      num_flow_layers=config.num_flow_layers, max_log_scale=config.max_log_scale)
        self.transition = ConditionalNormalizingFlow(
            config.state_dim, config.state_dim + config.control_dim,
            base_scale=config.base_scale, key=transition_key, **shared,
        )
        self.proposal = ConditionalNormalizingFlow(
            config.state_dim, config.state_dim + config.control_dim + config.observation_dim,
            base_scale=config.base_scale, key=proposal_key, **shared,
        )
        self.measurement = ConditionalNormalizingFlow(
            config.observation_dim, config.state_dim, base_scale=1.0,
            key=measurement_key, **shared,
        )

    def filter(self, observations, x0, controls=None, *, key, num_particles=32,
               resampling="ot", initial_std=0.0, resampling_threshold=0.5,
               soft_alpha=0.5, ot_epsilon=0.1, ot_iterations=32, remat=False):
        """Filter standardized observations of shape (T, O).

        ``ot`` is the paper's differentiable entropy-regularized transport
        resampler; its finite-particle evidence is approximate. ``soft`` uses
        mixture sampling with an importance correction. ``systematic`` and
        ``multinomial``/``categorical`` provide ordinary evaluation resampling.
        A threshold of zero disables resampling. Ordinary categorical methods
        have no gradients through ancestor selection. OT uses fixed iterations
        and a dimension-normalized squared Euclidean transport cost.
        ``remat=True`` recomputes each step in the backward pass instead of
        storing its intermediates (same values, far less memory on long windows).
        """
        observations, x0 = jnp.asarray(observations), jnp.asarray(x0)
        config = self.config
        if observations.ndim != 2 or observations.shape[-1] != config.observation_dim:
            raise ValueError("observations must have shape (T, observation_dim)")
        if observations.shape[0] < 1 or x0.shape != (config.state_dim,):
            raise ValueError("need at least one observation and x0 of shape (state_dim,)")
        if num_particles < 1 or initial_std < 0 or not 0 <= resampling_threshold <= 1:
            raise ValueError("invalid particle count, initial_std, or resampling_threshold")
        if resampling not in {"ot", "soft", "systematic", "multinomial", "categorical"}:
            raise ValueError("unknown resampling method")
        if not 0 < soft_alpha <= 1 or ot_epsilon <= 0 or ot_iterations < 1:
            raise ValueError("invalid soft_alpha or Sinkhorn settings")
        length = observations.shape[0]
        if controls is None:
            if config.control_dim:
                raise ValueError("controls are required when control_dim > 0")
            controls = jnp.zeros((length, 0), dtype=x0.dtype)
        controls = jnp.asarray(controls)
        if controls.shape != (length, config.control_dim):
            raise ValueError("controls must have shape (T, control_dim)")
        initial_key, filter_key = jr.split(key)
        initial_particles = x0 + initial_std * jr.normal(initial_key, (num_particles, config.state_dim), dtype=x0.dtype)
        initial_weights = jnp.full((num_particles,), -math.log(num_particles), dtype=x0.dtype)

        def step(carry, inputs):
            previous, log_weights, log_evidence = carry
            observation, control, index = inputs
            step_key = jr.fold_in(filter_key, index)
            resample_key, proposal_key = jr.split(step_key)
            ess = jnp.exp(-logsumexp(2 * log_weights))
            if resampling_threshold == 0 or length == 1:
                inherited = log_weights
            else:
                previous, inherited = jax.lax.cond(
                    (index > 0) & (ess < resampling_threshold * num_particles),
                    lambda args: _resample(*args, resample_key, resampling, soft_alpha, ot_epsilon, ot_iterations),
                    lambda args: args,
                    (previous, log_weights),
                )
            transition_condition = jnp.concatenate((previous, jnp.broadcast_to(control, (num_particles, config.control_dim))), axis=-1)
            proposal_condition = jnp.concatenate((transition_condition, jnp.broadcast_to(observation, (num_particles, config.observation_dim))), axis=-1)
            residual, log_q = self.proposal.sample_and_log_prob(proposal_key, proposal_condition)
            particles = previous + residual
            log_p = self.transition.log_prob(residual, transition_condition)
            log_likelihood = self.measurement.log_prob(observation, particles)
            unnormalized = inherited + log_p + log_likelihood - log_q
            normalizer = logsumexp(unnormalized)
            log_weights = unnormalized - normalizer
            weights = jnp.exp(log_weights)
            mean = jnp.sum(weights[:, None] * particles, axis=0)
            ess = jnp.exp(-logsumexp(2 * log_weights))
            return (particles, log_weights, log_evidence + normalizer), (particles, weights, mean, ess)

        final, output = jax.lax.scan(
            jax.checkpoint(step) if remat else step,
            (initial_particles, initial_weights, jnp.zeros((), x0.dtype)),
            (observations, controls, jnp.arange(length)),
        )
        particles, weights, means, ess = output
        return NFDPFResult(particles, weights, means, final[-1], ess)


def _training_arrays(model, states, observations, controls):
    states, observations = jnp.asarray(states), jnp.asarray(observations)
    config = model.config
    if states.ndim != 3 or states.shape[-1] != config.state_dim or states.shape[1] < 2:
        raise ValueError("states must have shape (B, T+1, state_dim), T >= 1")
    batch, length = states.shape[0], states.shape[1] - 1
    if observations.shape != (batch, length, config.observation_dim):
        raise ValueError("observations must have shape (B, T, observation_dim)")
    if controls is None:
        if config.control_dim:
            raise ValueError("controls are required when control_dim > 0")
        controls = jnp.zeros((batch, length, 0), states.dtype)
    controls = jnp.asarray(controls)
    if controls.shape != (batch, length, config.control_dim):
        raise ValueError("controls must have shape (B, T, control_dim)")
    return states, observations, controls


def nf_dpf_component_loss(model, states, observations, controls=None, *, normalize_dimensions=True):
    """Mean supervised transition/proposal/measurement NLL for warmup.

    Each component is divided by its event dimension by default so that a
    high-dimensional measurement does not dominate the two state densities.
    This optional auxiliary loss uses training labels only, never at inference.
    """
    states, observations, controls = _training_arrays(model, states, observations, controls)
    previous, following = states[:, :-1], states[:, 1:]
    transition_condition = jnp.concatenate((previous, controls), axis=-1)
    proposal_condition = jnp.concatenate((transition_condition, observations), axis=-1)
    residual = following - previous
    state_scale = model.config.state_dim if normalize_dimensions else 1
    observation_scale = model.config.observation_dim if normalize_dimensions else 1
    transition = -jnp.mean(model.transition.log_prob(residual, transition_condition)) / state_scale
    proposal = -jnp.mean(model.proposal.log_prob(residual, proposal_condition)) / state_scale
    measurement = -jnp.mean(model.measurement.log_prob(observations, following)) / observation_scale
    return (transition + proposal + measurement) / 3


def nf_dpf_loss(model, states, observations, controls=None, *, key, num_particles=32,
                resampling="ot", nll_weight=0.0, loss_type="mse",
                initial_std=0.0, resampling_threshold=0.5, soft_alpha=0.5,
                ot_epsilon=0.1, ot_iterations=32):
    """End-to-end supervised filtering loss, optionally plus component NLL.

    Batches contain states (B, T+1, D), observations (B, T, O), controls
    (B, T, U). Only states[:, 0] initialize the filter; future states are used
    exclusively as loss targets. Sampling and OT resampling are differentiable.
    """
    states, observations, controls = _training_arrays(model, states, observations, controls)
    if loss_type not in {"mse", "rmse"} or nll_weight < 0:
        raise ValueError("loss_type must be mse/rmse and nll_weight non-negative")
    keys = jr.split(key, states.shape[0])
    def estimate(x, y, u, k):
        return model.filter(
            y, x, u, key=k, num_particles=num_particles, resampling=resampling,
            initial_std=initial_std, resampling_threshold=resampling_threshold,
            soft_alpha=soft_alpha, ot_epsilon=ot_epsilon, ot_iterations=ot_iterations,
        ).mean
    predictions = jax.vmap(estimate)(states[:, 0], observations, controls, keys)
    loss = jnp.mean((predictions - states[:, 1:]) ** 2)
    if loss_type == "rmse":
        loss = jnp.sqrt(loss + 1e-12)
    if nll_weight:
        loss = loss + nll_weight * nf_dpf_component_loss(model, states, observations, controls)
    return loss
