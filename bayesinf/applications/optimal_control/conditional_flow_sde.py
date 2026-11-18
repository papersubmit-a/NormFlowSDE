"""Reusable conditional-RealNVP FlowSDE and Gaussian dynamics ablation."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


def make_mlp(input_dim, output_dim, width, depth, key):
    return eqx.nn.MLP(input_dim, output_dim, width, depth,
                      activation=jax.nn.silu, final_activation=lambda x: x,
                      key=key)


class ConditionalCoupling(eqx.Module):
    network: eqx.nn.MLP
    mask: jax.Array
    scale_limit: float = eqx.field(static=True)

    def __init__(self, dim, condition_dim, mask, key, width=256, scale_limit=2.0):
        self.network = make_mlp(dim + condition_dim, 2 * dim, width, 2, key)
        # A coupling partition is structural, not an optimizable parameter.
        self.mask = jnp.asarray(mask, dtype=jnp.bool_)
        self.scale_limit = scale_limit

    def parameters(self, fixed, condition):
        shift, raw_scale = jnp.split(self.network(jnp.concatenate((fixed, condition))), 2)
        active = 1.0 - self.mask
        log_scale = self.scale_limit * jnp.tanh(raw_scale / self.scale_limit)
        return active * shift, active * log_scale

    def forward(self, value, condition):
        fixed = self.mask * value
        shift, log_scale = self.parameters(fixed, condition)
        return fixed + (1 - self.mask) * (value * jnp.exp(log_scale) + shift), log_scale.sum()

    def inverse(self, value, condition):
        fixed = self.mask * value
        shift, log_scale = self.parameters(fixed, condition)
        return fixed + (1 - self.mask) * (value - shift) * jnp.exp(-log_scale), -log_scale.sum()


class ConditionalRealNVP(eqx.Module):
    layers: tuple
    dim: int = eqx.field(static=True)

    def __init__(self, dim, condition_dim, n_layers, key, width=256, scale_limit=2.0):
        keys = jr.split(key, n_layers)
        indices = jnp.arange(dim)
        self.layers = tuple(ConditionalCoupling(
            dim, condition_dim, (indices + layer) % 2 == 0, keys[layer], width, scale_limit
        ) for layer in range(n_layers))
        self.dim = dim

    def forward(self, base, condition):
        value, log_det = base, jnp.asarray(0.0)
        for layer in self.layers:
            value, change = layer.forward(value, condition); log_det += change
        return value, log_det

    def inverse(self, value, condition):
        base, log_det = value, jnp.asarray(0.0)
        for layer in reversed(self.layers):
            base, change = layer.inverse(base, condition); log_det += change
        return base, log_det


class FlowSDE(eqx.Module):
    drift: eqx.nn.MLP
    conditioner: eqx.nn.MLP
    flow: ConditionalRealNVP
    input_dim: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, input_dim, action_dim, state_dim, key, dt=0.02,
                 width=256, depth=3, condition_dim=128, flow_layers=4,
                 flow_scale_limit=2.0):
        kd, kc, kf = jr.split(key, 3)
        joint = input_dim + action_dim
        self.drift = make_mlp(joint, state_dim, width, depth, kd)
        self.conditioner = make_mlp(joint, condition_dim, width, 2, kc)
        self.flow = ConditionalRealNVP(state_dim, condition_dim, flow_layers, kf, width,
                                       flow_scale_limit)
        self.input_dim, self.action_dim, self.state_dim = input_dim, action_dim, state_dim
        self.dt = float(dt)

    def nominal_delta(self, history, action):
        return self.dt * self.drift(jnp.concatenate((history, action)))

    def mean_delta(self, history, action, *, key=jr.PRNGKey(0), samples=64):
        """Monte Carlo conditional mean, including the flow's nonzero mean.

        Antithetic base draws reduce error and integrate affine flow means
        exactly. This is a moment estimate, not an exact general flow integral.
        """
        if samples < 2 or samples % 2:
            raise ValueError("mean estimation requires an even samples >= 2")
        joint = jnp.concatenate((history, action))
        eps = jr.normal(key, (samples // 2, self.state_dim))
        eps = jnp.concatenate((eps, -eps))
        residuals = jax.vmap(lambda z: self.flow.forward(
            z, self.conditioner(joint))[0])(eps)
        return self.nominal_delta(history, action) + math.sqrt(self.dt) * residuals.mean(0)

    def step(self, history, action, key, stochastic=True):
        if not stochastic:
            return self.mean_delta(history, action)
        joint = jnp.concatenate((history, action))
        delta = self.dt * self.drift(joint)
        if stochastic:
            residual, _ = self.flow.forward(jr.normal(key, (self.state_dim,)),
                                             self.conditioner(joint))
            delta += math.sqrt(self.dt) * residual
        return delta

    def log_prob(self, delta, history, action):
        joint = jnp.concatenate((history, action))
        residual = (delta - self.dt * self.drift(joint)) / math.sqrt(self.dt)
        base, inverse_log_det = self.flow.inverse(residual, self.conditioner(joint))
        base_log_prob = -0.5 * jnp.sum(base ** 2 + math.log(2 * math.pi))
        return base_log_prob + inverse_log_det - 0.5 * self.state_dim * math.log(self.dt)


class GaussianSDE(eqx.Module):
    drift: eqx.nn.MLP
    log_std: jax.Array
    state_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, drift, state_dim, dt=0.02, log_std=None):
        self.drift, self.state_dim, self.dt = drift, state_dim, float(dt)
        self.log_std = (jnp.full((state_dim,), -2.0) if log_std is None
                        else jnp.asarray(log_std))

    def mean_delta(self, history, action):
        return self.dt * self.drift(jnp.concatenate((history, action)))

    def step(self, history, action, key, stochastic=True):
        delta = self.mean_delta(history, action)
        return delta + (math.sqrt(self.dt) * jnp.exp(self.log_std)
                        * jr.normal(key, (self.state_dim,)) if stochastic else 0.0)

    def log_prob(self, delta, history, action):
        mean, log_std = self.mean_delta(history, action), self.log_std + 0.5 * math.log(self.dt)
        return -0.5 * jnp.sum(((delta - mean) / jnp.exp(log_std)) ** 2
                              + 2 * log_std + math.log(2 * math.pi))


def load_flow_sde_checkpoint(path, template):
    """Load a fixed-mask model, migrating only mathematically valid old masks.

    Historical floating masks may have been optimized away from their binary
    partition. Such checkpoints cannot be repaired by rounding: their learned
    networks optimized an invalid density and need retraining.
    """
    masks = lambda model: tuple(layer.mask for layer in model.flow.layers)
    try:
        model = eqx.tree_deserialise_leaves(path, template)
    except RuntimeError:
        # Equinox checks leaf dtypes. Retry old float storage, then validate
        # every value before allowing a structural dtype migration.
        legacy = eqx.tree_at(masks, template,
                            tuple(mask.astype(jnp.float32) for mask in masks(template)))
        model = eqx.tree_deserialise_leaves(path, legacy)
    expected, actual = masks(template), masks(model)
    if any(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(actual, expected)):
        raise ValueError("Checkpoint coupling masks were optimized away from their binary partition; retrain the flow")
    return eqx.tree_at(masks, model, expected)
