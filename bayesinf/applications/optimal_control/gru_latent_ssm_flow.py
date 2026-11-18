"""Causal, control-conditioned LatentSSMFlow dynamics.

The model represents ``p(x[t+1] | x[t-H+1:t], u[t])``. A GRU embeds the
state history, a conditional normalizing flow maps the next state (or state
increment) to a latent variable, and a stable linear Gaussian SSM supplies
the latent sequence prior.
"""

from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.pointwise_state_flow import (
    build_flow_transforms,
)
from bayesinf.models.normalizing_flow import ActNorm, MAFLayer, RandomPermutation


def _forward(transforms, value, condition):
    z, logdet = value[None], jnp.zeros((1,))
    for transform in transforms:
        if isinstance(transform, (ActNorm, RandomPermutation)):
            z, ldj = transform(z, reverse=False)
        elif isinstance(transform, MAFLayer):
            z, ldj = transform(z, cond=condition[None], reverse=False)
        else:
            z, ldj = transform(z, condition[None], reverse=False)
        logdet = logdet + ldj
    return z[0], logdet[0]


def _inverse(transforms, latent, condition):
    value = latent[None]
    for transform in reversed(transforms):
        if isinstance(transform, (ActNorm, RandomPermutation)):
            value, _ = transform(value, reverse=True)
        elif isinstance(transform, MAFLayer):
            value, _ = transform(value, cond=condition[None], reverse=True)
        else:
            value, _ = transform(value, condition[None], reverse=True)
    return value[0]


class GRULatentSSMFlow(eqx.Module):
    """History-conditioned flow transition with a linear latent SSM."""

    gru: eqx.nn.GRUCell
    transforms: list
    F_raw: jnp.ndarray
    log_process_scale: jnp.ndarray
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    history_len: int = eqx.field(static=True)
    d_out: int = eqx.field(static=True)
    predict_delta: bool = eqx.field(static=True)
    max_rho: float = eqx.field(static=True)
    history_control_conditioned: bool = eqx.field(static=True)

    def __init__(
        self, state_dim: int, control_dim: int, history_len: int = 1,
        d_out: int = 64, key=jr.PRNGKey(0), *, predict_delta: bool = True,
        n_coupling: int = 4, hidden_size: int = 64, n_hidden: int = 2,
        flow_type: str = "realnvp", num_bins: int = 8, max_rho: float = 0.9,
        history_control_conditioned: bool = True,
        **_,
    ):
        if min(state_dim, control_dim, history_len, d_out) < 1:
            raise ValueError("dimensions and history_len must be positive")
        self.state_dim, self.control_dim = state_dim, control_dim
        self.history_len, self.d_out = history_len, d_out
        self.predict_delta, self.max_rho = predict_delta, max_rho
        self.history_control_conditioned = history_control_conditioned
        k_gru, k_flow, k_ssm = jr.split(key, 3)
        gru_input_dim = state_dim + (control_dim if history_control_conditioned else 0)
        self.gru = eqx.nn.GRUCell(gru_input_dim, d_out, key=k_gru)
        self.transforms = build_flow_transforms(
            input_dim=state_dim, cond_dim=d_out + control_dim,
            n_blocks=n_coupling, hidden_size=hidden_size, n_hidden=n_hidden,
            flow_type=flow_type, key=k_flow, num_bins=num_bins,
        )
        # One-step training supplies no evidence about temporal correlation.
        # Start independent; multi-step likelihood can learn a nonzero F.
        self.F_raw = jnp.zeros((state_dim, state_dim))
        self.log_process_scale = jnp.zeros((state_dim,))

    @property
    def F(self):
        norm = jnp.linalg.norm(self.F_raw, ord=2)
        return self.F_raw * jnp.minimum(1.0, self.max_rho / (norm + 1e-6))

    @property
    def process_scale(self):
        return jax.nn.softplus(self.log_process_scale) + 1e-4

    def encode_history(self, states, controls=None):
        """Encode H states and the H-1 controls applied between them."""
        if controls is None:
            controls = jnp.zeros((max(0, len(states) - 1), self.control_dim))
        padded_controls = jnp.concatenate([
            controls[-max(0, len(states) - 1):],
            jnp.zeros((1, self.control_dim)),
        ], axis=0)
        hidden = jnp.zeros((self.d_out,))
        for state, action in zip(states, padded_controls):
            gru_input = (
                jnp.concatenate([state, action])
                if self.history_control_conditioned else state
            )
            hidden = self.gru(gru_input, hidden)
        return hidden

    def condition(self, state_history, action, control_history=None):
        return jnp.concatenate([
            self.encode_history(state_history, control_history), action
        ])

    def _target(self, next_state, current_state):
        return next_state - current_state if self.predict_delta else next_state

    def _decode(self, latent, state_history, action, control_history=None):
        value = _inverse(
            self.transforms, latent,
            self.condition(state_history, action, control_history),
        )
        return state_history[-1] + value if self.predict_delta else value

    def infer_latent(self, state_history, action, next_state, control_history=None):
        """Recover the observed transition's latent for a continuing SSM."""
        history = jnp.asarray(state_history)[-self.history_len:]
        return _forward(self.transforms, self._target(next_state, history[-1]),
                        self.condition(history, action, control_history))[0]

    def _initial_latent(self, states, controls, initial_latent=None):
        if initial_latent is not None:
            return jnp.asarray(initial_latent)
        if len(states) > self.history_len:
            return self.infer_latent(states[-self.history_len-1:-1], controls[-1],
                                     states[-1], controls[-self.history_len:-1])
        # Exactly H observations are a fresh-window conditional forecast.
        # Continuing forecasts need one extra state or an explicit latent.
        return jnp.zeros((self.state_dim,))

    def predict_next(self, state_history, action, control_history=None, initial_latent=None):
        history = jnp.atleast_2d(state_history)
        controls = (jnp.zeros((max(0, len(history)-1), self.control_dim))
                    if control_history is None else control_history)
        latent = self._initial_latent(history, controls, initial_latent)
        return self._decode(self.F @ latent, history[-self.history_len:], action,
                            controls[-(self.history_len-1):] if self.history_len > 1 else controls[:0])

    def history_one_step_nll(self, states, controls):
        """NLL for batches shaped ``(B,H+1,D)`` and ``(B,H,U)``."""
        def example(x, u):
            history = x[:self.history_len]
            target = self._target(x[self.history_len], history[-1])
            z, ldj = _forward(
                self.transforms, target,
                self.condition(
                    history, u[self.history_len - 1],
                    u[:self.history_len - 1],
                ),
            )
            scale = self.process_scale
            logp = -0.5 * jnp.sum((z / scale) ** 2)
            logp -= jnp.sum(jnp.log(scale)) + 0.5 * self.state_dim * jnp.log(2 * jnp.pi)
            return -(logp + ldj)
        return jnp.mean(jax.vmap(example)(states, controls))

    def nll_loss(self, states, controls):
        """Exact flow likelihood with transition-aligned controls."""
        horizon = states.shape[1] - self.history_len

        def example(x, u):
            z_prev = jnp.zeros((self.state_dim,))
            total = 0.0
            for offset in range(horizon):
                t = self.history_len + offset
                history = x[t - self.history_len:t]
                target = self._target(x[t], x[t - 1])
                z, ldj = _forward(
                    self.transforms, target,
                    self.condition(
                        history, u[t - 1],
                        u[t - self.history_len:t - 1],
                    ),
                )
                mean, scale = self.F @ z_prev, self.process_scale
                logp = -0.5 * jnp.sum(((z - mean) / scale) ** 2)
                logp -= jnp.sum(jnp.log(scale)) + 0.5 * self.state_dim * jnp.log(2 * jnp.pi)
                total = total - logp - ldj
                z_prev = z
            return total / horizon
        return jnp.mean(jax.vmap(example)(states, controls))

    def noisy_context_nll_loss(self, states, controls, key, noise_std):
        noise = noise_std * jr.normal(key, states.shape)
        # Preserve every supervised next-state target and perturb contexts only.
        mask = jnp.ones(states.shape[1])
        mask = mask.at[self.history_len::].set(0.0)
        return self.nll_loss(states + noise * mask[None, :, None], controls)

    def rollout_deterministic_history(self, state_history, control_history, actions,
                                      initial_latent=None):
        latent = self._initial_latent(state_history, control_history, initial_latent)
        history = state_history[-self.history_len:]
        past_controls = (control_history[-(self.history_len - 1):]
                         if self.history_len > 1 else control_history[:0])
        def step(carry, action):
            history, past_controls, latent = carry
            latent = self.F @ latent
            next_state = self._decode(latent, history, action, past_controls)
            history = jnp.concatenate([history[1:], next_state[None]], axis=0)
            if self.history_len > 1:
                past_controls = jnp.concatenate([past_controls[1:], action[None]], axis=0)
            return (history, past_controls, latent), next_state
        _, outputs = jax.lax.scan(step, (history, past_controls, latent), actions)
        return outputs

    def rollout_deterministic(self, state_context, actions, initial_latent=None):
        history = jnp.asarray(state_context)
        if history.ndim == 1:
            history = history[None]
        return self.rollout_deterministic_history(
            history, jnp.zeros((max(0, history.shape[0] - 1), self.control_dim)), actions, initial_latent
        )

    def sample_rollout(self, state_history, control_history, actions, key,
                       n_particles=1, initial_latent=None):
        """Continue an SSM when a latent or H+1 observations are provided.

        Exactly H observations without a latent explicitly define a fresh
        window with latent zero, matching the training-window likelihood.
        """
        initial_latent = self._initial_latent(state_history, control_history, initial_latent)
        def particle(particle_key):
            history = state_history[-self.history_len:]
            past_controls = (control_history[-(self.history_len - 1):]
                             if self.history_len > 1 else control_history[:0])
            latent = initial_latent
            noises = jr.split(particle_key, actions.shape[0])

            def step(carry, inputs):
                history, past_controls, latent = carry
                action, noise = inputs
                latent = self.F @ latent + self.process_scale * jr.normal(
                    noise, (self.state_dim,)
                )
                next_state = self._decode(
                    latent, history, action, past_controls
                )
                history = jnp.concatenate([history[1:], next_state[None]], axis=0)
                if self.history_len > 1:
                    past_controls = jnp.concatenate([
                        past_controls[1:], action[None]
                    ], axis=0)
                return (history, past_controls, latent), next_state

            _, outputs = jax.lax.scan(
                step, (history, past_controls, latent), (actions, noises)
            )
            return outputs
        return jax.vmap(particle)(jr.split(key, n_particles))
