"""Direct observation-conditioned NSF and FlowSDE state generators.

These are the supervised adaptations specified in related_work.tex, rather
than the transition-prior particle filters in the older experiment runners.
By default the bidirectional encoder sees the complete observation window
(smoothing). ``encoder_direction="causal"`` uses a forward-only encoder, so the
context of transition k depends on observations[:k+1] only: x[k+1] uses
measurements through its own, with no future access (filtering). Inference
receives its initial state, observations and controls only; each subsequent
state is generated recursively, without particle weights.

States are supplied in the caller's training-normalized coordinates. Index k
of observations measures states[k+1]; controls[k] drives states[k:k+2]. The
fixed physical interval is config.dt. NSF consistency splits *one* such
interval, holding its encoded context and command constant on both pieces.
It never identifies changing control paths with a single constant command.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .fm_flow_sde import FlowMatchingSDE, FlowSDEConfig, flow_sde_loss
from .neural_stochastic_flow import NSFBridge, NeuralStochasticFlow


@dataclass(frozen=True)
class ObservationConditionedConfig:
    state_dim: int
    observation_dim: int
    control_dim: int = 0
    dt: float = 0.02
    encoder_hidden: int = 32
    context_dim: int = 32
    hidden_size: int = 64
    hidden_depth: int = 2
    flow_layers: int = 4
    consistency_weight: float = 0.1
    consistency_samples: int = 1
    diffusion: float = 0.01
    bridge_noise_scale: float | None = None
    flow_substeps: int = 4
    use_potential: bool = True
    epr_weight: float = 1e-3
    hjb_weight: float = 1e-3
    orthogonality_weight: float = 1e-3
    regularizer_measure: str = "boltzmann_importance"
    encoder_direction: str = "bidirectional"

    def __post_init__(self):
        if min(self.state_dim, self.observation_dim) < 1 or self.control_dim < 0:
            raise ValueError("positive state/observation dimensions and control_dim >= 0 required")
        sizes = (self.encoder_hidden, self.context_dim, self.hidden_size,
                 self.hidden_depth, self.flow_layers, self.consistency_samples,
                 self.flow_substeps)
        if min(sizes) < 1 or self.dt <= 0 or self.diffusion <= 0:
            raise ValueError("architecture sizes, dt and diffusion must be positive")
        if min(self.consistency_weight, self.epr_weight, self.hjb_weight,
               self.orthogonality_weight) < 0:
            raise ValueError("loss weights must be nonnegative")
        if self.bridge_noise_scale is not None and self.bridge_noise_scale <= 0:
            raise ValueError("bridge_noise_scale must be positive")
        if self.regularizer_measure not in {"empirical", "boltzmann_importance"}:
            raise ValueError("invalid regularizer_measure")
        if self.encoder_direction not in {"bidirectional", "causal"}:
            raise ValueError("encoder_direction must be 'bidirectional' or 'causal'")


class BaselineLosses(NamedTuple):
    total: jax.Array
    nll: jax.Array
    consistency: jax.Array
    flow_matching: jax.Array
    epr: jax.Array
    hjb: jax.Array
    orthogonality: jax.Array


class ObservationEncoder(eqx.Module):
    """Bidirectional (or causal forward-only) GRU and a per-time context projection."""

    forward: eqx.nn.GRUCell
    backward: eqx.nn.GRUCell | None
    projection: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    causal: bool = eqx.field(static=True)

    def __init__(self, input_size, hidden_size, context_dim, *, key, causal=False):
        forward_key, backward_key, projection_key = jr.split(key, 3)
        self.hidden_size = hidden_size
        self.causal = causal
        self.forward = eqx.nn.GRUCell(input_size, hidden_size, key=forward_key)
        self.backward = None if causal else eqx.nn.GRUCell(input_size, hidden_size, key=backward_key)
        self.projection = eqx.nn.Linear((1 if causal else 2) * hidden_size, context_dim, key=projection_key)

    def __call__(self, observations, controls):
        # c_k=C(y_{0:K})_k in related_work.tex; commands enter the state
        # generator separately, rather than adding future control features.
        del controls
        inputs = observations
        initial = jnp.zeros(self.hidden_size, dtype=inputs.dtype)

        def forward_step(hidden, value):
            hidden = self.forward(value, hidden)
            return hidden, hidden

        def backward_step(hidden, value):
            hidden = self.backward(value, hidden)
            return hidden, hidden

        _, forward = jax.lax.scan(forward_step, initial, inputs)
        if self.causal:
            return jax.vmap(self.projection)(forward)
        _, backward = jax.lax.scan(backward_step, initial, inputs, reverse=True)
        return jax.vmap(self.projection)(jnp.concatenate((forward, backward), axis=-1))


def _controls(observations, controls, config):
    observations = jnp.asarray(observations)
    if observations.ndim not in (2, 3) or observations.shape[-1] != config.observation_dim:
        raise ValueError("observations must have shape (T,O) or (B,T,O)")
    if observations.shape[-2] < 1:
        raise ValueError("at least one observation is required")
    if controls is None:
        if config.control_dim:
            raise ValueError("controls are required when control_dim > 0")
        controls = jnp.zeros((*observations.shape[:-1], 0), dtype=observations.dtype)
    controls = jnp.asarray(controls)
    if controls.shape != (*observations.shape[:-1], config.control_dim):
        raise ValueError("controls must align with observations and have control_dim columns")
    return observations, controls


def _training_arrays(states, observations, controls, config):
    observations, controls = _controls(observations, controls, config)
    states = jnp.asarray(states)
    if observations.ndim != 3 or states.shape != (
        observations.shape[0], observations.shape[1] + 1, config.state_dim
    ):
        raise ValueError("states must have shape (B,T+1,D), including x0; observations (B,T,O)")
    return states, observations, controls


def _inference_arrays(observations, x0, controls, config, num_samples):
    observations, controls = _controls(observations, controls, config)
    x0 = jnp.asarray(x0)
    if observations.ndim != 2 or x0.shape != (config.state_dim,):
        raise ValueError("sample expects observations (T,O) and x0 (D,)")
    if not isinstance(num_samples, int) or num_samples < 1:
        raise ValueError("num_samples must be a positive Python integer")
    return observations, x0, controls


class ObservationConditionedNSF(eqx.Module):
    """Exact conditional trajectory likelihood plus variational consistency.

    The auxiliary bridge is trainable and endpoint-conditioned. A bridge need
    not depend on measurement context for the KL bounds to be valid; omitting
    that additional context is an explicit capacity choice. Both KL bounds
    and the bridge are optimized jointly. Losses are per transition and state
    coordinate; log_prob returns the unscaled joint trajectory log density.
    """

    encoder: ObservationEncoder
    transition: NeuralStochasticFlow
    bridge: NSFBridge
    config: ObservationConditionedConfig = eqx.field(static=True)

    def __init__(self, config: ObservationConditionedConfig, *, key):
        encoder_key, transition_key, bridge_key = jr.split(key, 3)
        self.config = config
        self.encoder = ObservationEncoder(
            config.observation_dim, config.encoder_hidden,
            config.context_dim, key=encoder_key,
            causal=config.encoder_direction == "causal",
        )
        self.transition = NeuralStochasticFlow(
            config.state_dim, config.control_dim + config.context_dim,
            hidden_size=config.hidden_size, hidden_depth=config.hidden_depth,
            num_coupling_layers=config.flow_layers, include_absolute_time=True,
            key=transition_key,
        )
        self.bridge = NSFBridge(
            config.state_dim, hidden_size=config.hidden_size,
            hidden_depth=config.hidden_depth, num_coupling_layers=config.flow_layers,
            include_absolute_time=True, key=bridge_key,
        )

    def encode(self, observations, controls=None):
        observations, controls = _controls(observations, controls, self.config)
        if observations.ndim == 3:
            return jax.vmap(self.encoder)(observations, controls)
        return self.encoder(observations, controls)

    def log_prob(self, states, observations, controls=None):
        states, observations, controls = _training_arrays(states, observations, controls, self.config)
        contexts = self.encode(observations, controls)
        conditions = jnp.concatenate((controls, contexts), axis=-1)
        times = self.config.dt * jnp.arange(observations.shape[1], dtype=states.dtype)
        log_prob = self.transition.log_prob(
            states[:, 1:], states[:, :-1], self.config.dt, conditions,
            t_current=jnp.broadcast_to(times, observations.shape[:2]),
        )
        return jnp.sum(log_prob, axis=-1)

    def _consistency(self, initial, conditions, times, key):
        """Two variational joint KL bounds with one fixed context per interval."""
        config = self.config
        gap_key, direct_key, bridge_key, middle_key, end_key = jr.split(key, 5)
        before = config.dt * jr.uniform(gap_key, initial.shape[:-1], minval=0.1, maxval=0.9)
        after = config.dt - before
        direct = self.transition.sample(
            direct_key, initial, config.dt, conditions,
            num_samples=config.consistency_samples, t_current=times,
        )
        bridge_keys = jr.split(bridge_key, config.consistency_samples)
        direct_middle = jax.vmap(lambda sample_key, end: self.bridge.sample(
            sample_key, initial, end, before, after, num_samples=1, t_start=times
        )[0])(bridge_keys, direct)

        def one_to_two(middle, end):
            return (
                self.transition.log_prob(end, initial, config.dt, conditions, t_current=times)
                + self.bridge.log_prob(middle, initial, end, before, after, t_start=times)
                - self.transition.log_prob(middle, initial, before, conditions, t_current=times)
                - self.transition.log_prob(end, middle, after, conditions, t_current=times + before)
            )

        middle = self.transition.sample(
            middle_key, initial, before, conditions,
            num_samples=config.consistency_samples, t_current=times,
        )
        end_keys = jr.split(end_key, config.consistency_samples)
        recursive = jax.vmap(lambda sample_key, mid: self.transition.sample(
            sample_key, mid, after, conditions, num_samples=1, t_current=times + before
        )[0])(end_keys, middle)

        def two_to_one(mid, end):
            return (
                self.transition.log_prob(mid, initial, before, conditions, t_current=times)
                + self.transition.log_prob(end, mid, after, conditions, t_current=times + before)
                - self.bridge.log_prob(mid, initial, end, before, after, t_start=times)
                - self.transition.log_prob(end, initial, config.dt, conditions, t_current=times)
            )

        return (jnp.mean(jax.vmap(one_to_two)(direct_middle, direct))
                + jnp.mean(jax.vmap(two_to_one)(middle, recursive))) / config.state_dim

    def loss_parts(self, states, observations, controls=None, *, key):
        states, observations, controls = _training_arrays(states, observations, controls, self.config)
        config = self.config
        contexts = self.encode(observations, controls)
        conditions = jnp.concatenate((controls, contexts), axis=-1)
        times = jnp.broadcast_to(
            config.dt * jnp.arange(observations.shape[1], dtype=states.dtype), observations.shape[:2]
        )
        nll = -jnp.mean(self.transition.log_prob(
            states[:, 1:], states[:, :-1], config.dt, conditions, t_current=times
        )) / config.state_dim
        zero = jnp.zeros((), dtype=states.dtype)
        consistency = (self._consistency(states[:, :-1], conditions, times, key)
                       if config.consistency_weight else zero)
        return BaselineLosses(nll + config.consistency_weight * consistency,
                              nll, consistency, zero, zero, zero, zero)

    def loss(self, states, observations, controls=None, *, key):
        return self.loss_parts(states, observations, controls, key=key).total

    def sample(self, observations, x0, controls=None, *, key, num_samples=100):
        observations, x0, controls = _inference_arrays(
            observations, x0, controls, self.config, num_samples
        )
        conditions = jnp.concatenate((controls, self.encode(observations, controls)), axis=-1)
        times = self.config.dt * jnp.arange(observations.shape[0], dtype=x0.dtype)

        def step(current, inputs):
            index, condition, time = inputs
            following = self.transition.sample(
                jr.fold_in(key, index), current, self.config.dt, condition,
                num_samples=1, t_current=time,
            )[0]
            return following, following

        _, path = jax.lax.scan(step, jnp.broadcast_to(x0, (num_samples, self.config.state_dim)),
                              (jnp.arange(observations.shape[0]), conditions, times))
        return jnp.swapaxes(path, 0, 1)


class ObservationConditionedFlowSDE(eqx.Module):
    """Gaussian-bridge drift regression and conditional SDE trajectory draws."""

    encoder: ObservationEncoder
    dynamics: FlowMatchingSDE
    config: ObservationConditionedConfig = eqx.field(static=True)

    def __init__(self, config: ObservationConditionedConfig, *, key):
        encoder_key, dynamics_key = jr.split(key)
        self.config = config
        self.encoder = ObservationEncoder(
            config.observation_dim, config.encoder_hidden,
            config.context_dim, key=encoder_key,
            causal=config.encoder_direction == "causal",
        )
        # Besides the measured context and control, expose the physical gap.
        self.dynamics = FlowMatchingSDE(
            config.state_dim, control_dim=config.control_dim + config.context_dim + 1,
            width=config.hidden_size, depth=config.hidden_depth,
            diffusion=config.diffusion, use_potential=config.use_potential,
            key=dynamics_key,
        )

    def encode(self, observations, controls=None):
        observations, controls = _controls(observations, controls, self.config)
        if observations.ndim == 3:
            return jax.vmap(self.encoder)(observations, controls)
        return self.encoder(observations, controls)

    def _conditions(self, observations, controls):
        return jnp.concatenate((
            controls, self.encode(observations, controls),
            jnp.full((*observations.shape[:-1], 1), self.config.dt, dtype=observations.dtype),
        ), axis=-1)

    def loss_parts(self, states, observations, controls=None, *, key):
        states, observations, controls = _training_arrays(states, observations, controls, self.config)
        config = self.config
        conditions = self._conditions(observations, controls)
        times = jnp.broadcast_to(
            config.dt * jnp.arange(observations.shape[1], dtype=states.dtype), observations.shape[:2]
        ).reshape(-1)
        parts = flow_sde_loss(
            self.dynamics, states[:, :-1].reshape(-1, config.state_dim),
            states[:, 1:].reshape(-1, config.state_dim), times, times + config.dt,
            key=key, controls=conditions.reshape(-1, self.dynamics.control_dim),
            config=FlowSDEConfig(
                diffusion=config.diffusion, bridge_noise_scale=config.bridge_noise_scale,
                epr_weight=config.epr_weight, hjb_weight=config.hjb_weight,
                orthogonality_weight=config.orthogonality_weight,
                regularizer_measure=config.regularizer_measure,
            ),
        )
        zero = jnp.zeros((), dtype=states.dtype)
        return BaselineLosses(parts.total, zero, zero, parts.flow_matching,
                              parts.epr, parts.hjb, parts.orthogonality)

    def loss(self, states, observations, controls=None, *, key):
        return self.loss_parts(states, observations, controls, key=key).total

    def sample(self, observations, x0, controls=None, *, key, num_samples=100):
        observations, x0, controls = _inference_arrays(
            observations, x0, controls, self.config, num_samples
        )
        config = self.config
        conditions = self._conditions(observations, controls)
        sub_dt = config.dt / config.flow_substeps
        noise_scale = jnp.sqrt(2 * config.diffusion * sub_dt)

        def interval(current, inputs):
            index, condition = inputs

            def substep(sub_index, state):
                time = (index * config.flow_substeps + sub_index) * sub_dt
                drift = jax.vmap(lambda x: self.dynamics(x, time, control=condition))(state)
                step_key = jr.fold_in(key, index * config.flow_substeps + sub_index)
                return state + sub_dt * drift + noise_scale * jr.normal(step_key, state.shape, dtype=state.dtype)

            following = jax.lax.fori_loop(0, config.flow_substeps, substep, current)
            return following, following

        _, path = jax.lax.scan(interval, jnp.broadcast_to(x0, (num_samples, config.state_dim)),
                              (jnp.arange(observations.shape[0]), conditions))
        return jnp.swapaxes(path, 0, 1)


def observation_conditioned_loss(model, states, observations, controls=None, *, key):
    """Common scalar objective for the two direct conditional baselines."""
    return model.loss(states, observations, controls, key=key)


__all__ = ["ObservationConditionedConfig", "ObservationConditionedNSF",
           "ObservationConditionedFlowSDE", "BaselineLosses",
           "observation_conditioned_loss"]
