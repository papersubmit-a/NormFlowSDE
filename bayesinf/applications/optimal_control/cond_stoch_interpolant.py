"""Conditional stochastic-interpolant policy for action chunks.

Every policy starts from an independent standard Gaussian action chunk.  No
inference/base network is present.  Three standalone variants are supported:

* flow matching (``score_mode="flow"``): velocity network only, ODE sampling;
* analytic score (``score_mode="analytic"``): velocity network plus the exact
  Gaussian-source velocity/score identity, with no learned score network;
* learned score (``score_mode="learned"``): the state-estimation CSI objective
  with separate shared-architecture velocity and scaled-score networks.

Observation histories may be encoded by either a compact MLP (the default for
control) or a transformer.  Both produce one context vector per future action
step.

The offline interpolant is

    I_t = (1-t) x_0 + t x_1 + sigma_0 t(1-t) z,

where ``x_1`` is an expert action chunk and ``x_0 ~ N(0,I)``. At inference,
the state-estimation refinement SDE transports that source to an action chunk.
Its explicit Gaussian densities support DPPO's inner-MDP PPO loss.
"""
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.stochastic_interpolant import (
    ObsTransformerEncoder,
    SITransformerNet,
    linear_interpolant_coefficients,
    refinement_sde_coefficients,
    unscale_interpolant_score,
)
from bayesinf.applications.optimal_control.dppo_diffusion_jax import (
    cosine_beta_schedule,
)


class MLPContextEncoder(eqx.Module):
    """Encode one observation history into horizon-specific contexts."""

    layers: list
    output: eqx.nn.Linear
    horizon: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)

    def __init__(self, input_dim, horizon, context_dim, hidden_size, depth, key):
        keys = jr.split(key, depth + 1)
        sizes = [input_dim] + [hidden_size] * depth
        self.layers = [
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(depth)
        ]
        self.output = eqx.nn.Linear(
            sizes[-1], horizon * context_dim, key=keys[-1]
        )
        self.horizon, self.context_dim = horizon, context_dim

    def __call__(self, state_history):
        value = state_history.reshape(-1)
        for layer in self.layers:
            value = jax.nn.mish(layer(value))
        return self.output(value).reshape(self.horizon, self.context_dim)


ConditionalVelocity = SITransformerNet


class InterpolantSample(NamedTuple):
    """Generated action chunks and their complete inner-MDP chain."""

    actions: jnp.ndarray
    chain: jnp.ndarray


class CondStochInterpolantPolicy(eqx.Module):
    """State-conditioned stochastic-interpolant action policy.

    States have shape ``(batch, cond_steps, obs_dim)`` and action chunks have
    shape ``(batch, horizon, action_dim)``.  The MLP encoder maps a flattened
    history directly to horizon-specific contexts.  The optional transformer
    instead repeats that history over the future horizon and distinguishes
    future steps with positional embeddings.
    """

    encoder: ObsTransformerEncoder | MLPContextEncoder
    vel_net: SITransformerNet
    score_net: SITransformerNet | None

    obs_dim: int = eqx.field(static=True)
    cond_steps: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    integration_steps: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    sigma_0: float = eqx.field(static=True)
    sampling_std: float = eqx.field(static=True)
    # Legacy checkpoint/config field. Exact score conversion now uses only a
    # numerical 1e-8 guard; an arbitrary positive floor changes the SI law.
    score_sigma_floor: float = eqx.field(static=True)
    sde_eta: float = eqx.field(static=True)
    sde_etas: jnp.ndarray
    sde_eta_schedule: str = eqx.field(static=True)
    path_loss_weight: float = eqx.field(static=True)
    path_loss_steps: int = eqx.field(static=True)
    encoder_type: str = eqx.field(static=True)
    score_loss_weight: float = eqx.field(static=True)
    score_mode: str = eqx.field(static=True)
    sampler_matched_sde: bool = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        key: jr.PRNGKey,
        *,
        cond_steps: int = 1,
        integration_steps: int = 5,
        sigma_0: float = 0.1,
        sampling_std: float = 0.1,
        score_sigma_floor: float = 0.02,
        sde_eta: float = 0.05,
        sde_eta_schedule: str = "fixed",
        path_loss_weight: float = 0.0,
        path_loss_steps: int = 10,
        hidden_size: int = 256,
        depth: int = 3,
        time_dim: int = 16,
        context_dim: int = 32,
        n_heads: int = 4,
        score_head_width: int = 0,
        score_loss_weight: float = 0.0,
        score_mode: str = "analytic",
        sampler_matched_sde: bool = False,
        encoder_type: str = "transformer",
    ):
        if min(
            obs_dim,
            action_dim,
            horizon,
            cond_steps,
            integration_steps,
            hidden_size,
            depth,
            time_dim,
            context_dim,
            n_heads,
            path_loss_steps,
        ) < 1:
            raise ValueError("dimensions and integration_steps must be positive")
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if min(
            sigma_0, sampling_std, score_sigma_floor, sde_eta,
            path_loss_weight,
        ) < 0:
            raise ValueError("noise, score, and SDE parameters must be nonnegative")
        if score_sigma_floor == 0:
            raise ValueError("score_sigma_floor must be positive")
        if score_mode not in ("flow", "analytic", "learned"):
            raise ValueError("score_mode must be flow, analytic, or learned")
        if score_mode == "learned" and sigma_0 == 0:
            raise ValueError("learned scaled scores require sigma_0 > 0; use analytic or flow mode for a zero-noise interpolant")
        if score_mode == "learned" and score_head_width < 1:
            raise ValueError(f"score_head_width must be positive for {score_mode} score")
        if score_mode == "learned" and score_head_width % n_heads:
            raise ValueError("score_head_width must be divisible by n_heads")
        if encoder_type not in ("mlp", "transformer"):
            raise ValueError("encoder_type must be mlp or transformer")
        if sde_eta_schedule not in ("fixed", "ddpm"):
            raise ValueError("sde_eta_schedule must be fixed or ddpm")

        k_encoder, k_velocity, k_score = jr.split(key, 3)
        state_size = obs_dim * cond_steps
        if encoder_type == "mlp":
            self.encoder = MLPContextEncoder(
                state_size, horizon, context_dim, hidden_size, depth, k_encoder
            )
        else:
            self.encoder = ObsTransformerEncoder(
                d_y=state_size, d_c=context_dim, d_model=hidden_size,
                n_heads=n_heads, depth=depth, time_emb_dim=time_dim,
                key=k_encoder,
            )
        self.vel_net = SITransformerNet(
            d_x=action_dim,
            d_c=context_dim,
            d_model=hidden_size,
            n_heads=n_heads,
            depth=depth,
            time_emb_dim=time_dim,
            condition_on_mu=False,
            key=k_velocity,
        )
        if score_mode in ("flow", "analytic"):
            self.score_net = None
        else:
            self.score_net = SITransformerNet(
                d_x=action_dim, d_c=context_dim, d_model=score_head_width,
                n_heads=n_heads, depth=depth, time_emb_dim=time_dim,
                condition_on_mu=False, key=k_score,
            )
        self.obs_dim, self.cond_steps = obs_dim, cond_steps
        self.action_dim, self.horizon = action_dim, horizon
        self.integration_steps, self.context_dim = integration_steps, context_dim
        self.sigma_0, self.sampling_std = float(sigma_0), float(sampling_std)
        self.score_sigma_floor = float(score_sigma_floor)
        self.sde_eta = float(sde_eta)
        self.sde_eta_schedule = sde_eta_schedule
        if sde_eta_schedule == "ddpm":
            betas = cosine_beta_schedule(integration_steps)
            alphas = 1.0 - betas
            alpha_bar = jnp.cumprod(alphas)
            alpha_bar_prev = jnp.concatenate([jnp.ones(1), alpha_bar[:-1]])
            posterior_variance = betas * (1.0 - alpha_bar_prev) / (
                1.0 - alpha_bar
            )
            # CSI integrates source -> data, opposite to DDPM's index order.
            # 2 * eta_k * dt equals DDPM's reverse-transition variance.
            self.sde_etas = posterior_variance[::-1] * integration_steps / 2.0
        else:
            self.sde_etas = jnp.full((integration_steps,), self.sde_eta)
        self.path_loss_weight = float(path_loss_weight)
        self.path_loss_steps = int(path_loss_steps)
        self.encoder_type = encoder_type
        self.score_loss_weight = float(score_loss_weight)
        self.score_mode = score_mode
        self.sampler_matched_sde = bool(sampler_matched_sde)

    @property
    def velocity(self) -> SITransformerNet:
        """Backward-compatible alias for the shared velocity network."""
        return self.vel_net

    @property
    def nfe(self) -> int:
        return self.integration_steps

    def _check_states(self, states: jnp.ndarray) -> None:
        if states.ndim != 3 or states.shape[1:] != (
            self.cond_steps,
            self.obs_dim,
        ):
            raise ValueError(
                f"states must have shape (batch, {self.cond_steps}, {self.obs_dim})"
            )

    def _features(self, state: jnp.ndarray) -> jnp.ndarray:
        """Return one observation-context vector per future action step."""
        if state.shape != (self.cond_steps, self.obs_dim):
            raise ValueError(
                f"state must have shape ({self.cond_steps}, {self.obs_dim})"
            )
        if self.encoder_type == "mlp":
            return self.encoder(state)
        state_vector = state.reshape(-1)
        state_sequence = jnp.broadcast_to(
            state_vector, (self.horizon, state_vector.shape[0])
        )
        return self.encoder(state_sequence)

    def base_params(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return the fixed standard-Gaussian source parameters."""
        self._features(state)  # validates the history shape
        zeros = jnp.zeros((self.horizon, self.action_dim))
        return zeros, zeros

    def loss_components_with_score(
        self, actions: jnp.ndarray, states: jnp.ndarray, key: jr.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return total, velocity, score, and sampler-matched endpoint losses."""
        self._check_states(states)
        if actions.ndim != 3 or actions.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"actions must have shape (batch, {self.horizon}, {self.action_dim})"
            )
        if len(actions) != len(states):
            raise ValueError("actions and states must have the same batch size")
        keys = jr.split(key, len(actions))

        def example_loss(action, state, sample_key):
            kt, kb, kz = jr.split(sample_key, 3)
            t = jr.uniform(kt, (), minval=0.02, maxval=1.0 - 0.02)
            context = self._features(state)
            x0 = jr.normal(kb, action.shape)
            noise = jr.normal(kz, action.shape)
            alpha, beta, sigma, alpha_dot, beta_dot, sigma_dot = (
                linear_interpolant_coefficients(t, self.sigma_0)
            )
            interpolated = alpha * x0 + beta * action + sigma * noise
            target = alpha_dot * x0 + beta_dot * action + sigma_dot * noise
            prediction = self.vel_net(interpolated, context, t)
            velocity_loss = jnp.mean((prediction - target) ** 2)
            if self.score_net is None or self.score_loss_weight == 0:
                score_loss = jnp.asarray(0.0, velocity_loss.dtype)
            else:
                scaled_score = self.score_net(interpolated, context, t)
                score_loss = jnp.mean((scaled_score + noise) ** 2)
            return velocity_loss, score_loss

        velocity_losses, score_losses = jax.vmap(example_loss)(actions, states, keys)
        velocity_loss, score_loss = jnp.mean(velocity_losses), jnp.mean(score_losses)
        if self.path_loss_weight == 0:
            path_loss = jnp.asarray(0.0, dtype=velocity_loss.dtype)
        else:
            path_keys = jr.split(jr.fold_in(key, 1), len(actions))

            def example_path_loss(action, state, sample_key):
                context = self._features(state)
                k1, k2 = jr.split(sample_key)
                source1 = jr.normal(k1, action.shape)
                source2 = jr.normal(k2, action.shape)
                if self.sampler_matched_sde:
                    k1n, k2n = jr.split(jr.fold_in(sample_key, 17))
                    sample1 = self._sde_euler_single(source1, context, k1n, self.path_loss_steps)
                    sample2 = self._sde_euler_single(source2, context, k2n, self.path_loss_steps)
                else:
                    sample1 = self._ode_heun_single(source1, context, self.path_loss_steps)
                    sample2 = self._ode_heun_single(source2, context, self.path_loss_steps)
                flatten = lambda value: value.reshape(-1)
                error1 = jnp.linalg.norm(flatten(sample1 - action))
                error2 = jnp.linalg.norm(flatten(sample2 - action))
                spread = jnp.linalg.norm(flatten(sample1 - sample2))
                scale = jnp.sqrt(action.size)
                return (0.5 * (error1 + error2) - 0.5 * spread) / scale

            path_loss = jnp.mean(
                jax.vmap(example_path_loss)(actions, states, path_keys)
            )
        total = (velocity_loss + self.path_loss_weight * path_loss
                 + self.score_loss_weight * score_loss)
        return total, velocity_loss, score_loss, path_loss

    def loss_components(
        self, actions: jnp.ndarray, states: jnp.ndarray, key: jr.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return total, velocity, and endpoint losses (legacy public API)."""
        total, velocity, _, path = self.loss_components_with_score(
            actions, states, key
        )
        return total, velocity, path

    def loss(
        self, actions: jnp.ndarray, states: jnp.ndarray, key: jr.PRNGKey
    ) -> jnp.ndarray:
        """Stochastic-interpolant velocity-matching objective."""
        return self.loss_components(actions, states, key)[0]

    def _corrected_score(self, x, velocity, context, t):
        if self.score_mode == "learned":
            scaled_score = self.score_net(x, context, t)
            sigma = linear_interpolant_coefficients(t, self.sigma_0)[2]
            return unscale_interpolant_score(
                scaled_score, sigma, 1e-8
            )
        if self.score_mode == "analytic":
            return self._score_from_velocity(x, velocity, t)
        raise ValueError("flow-only policy has no score and must use an ODE sampler")

    def _eta_at(self, index):
        """Coefficient on the original trained integration grid."""
        return self.sde_etas[index]

    def _sampling_coefficients(self, t, dt):
        """One consistent score/noise pair, resampled on the requested grid.

        DDPM schedule values encode half the target variance per unit time;
        fixed schedules encode eta in a=eta*sigma(t). The tiny PPO variance
        floor is included in BOTH coefficients, preserving a=b**2/2.
        """
        grid = (jnp.arange(self.integration_steps) + .5) / self.integration_steps
        eta = jnp.interp(t, grid, self.sde_etas)
        sigma = linear_interpolant_coefficients(t, self.sigma_0)[2]
        sigma_for_schedule = 1.0 if self.sde_eta_schedule == "ddpm" else sigma
        score_weight, _ = refinement_sde_coefficients(sigma_for_schedule, eta)
        score_weight = jnp.maximum(score_weight, 0.5e-6 / dt)
        return score_weight, jnp.sqrt(2.0 * score_weight * dt)

    def _sde_euler_single(self, initial, context, key, steps):
        """Differentiable sampler identical to deployed Euler--Maruyama SDE."""
        dt = 1.0 / steps
        keys = jr.split(key, steps)
        def step(current, inputs):
            index, noise_key = inputs
            t = (index.astype(current.dtype) + .5) * dt
            velocity = self.vel_net(current, context, t)
            score = self._corrected_score(current, velocity, context, t)
            score_weight, transition_std = self._sampling_coefficients(t, dt)
            following = (current + dt * (velocity + score_weight * score)
                         + transition_std
                         * jr.normal(noise_key, current.shape))
            return following, following
        final, _ = jax.lax.scan(step, initial, (jnp.arange(steps), keys))
        return final

    def _ode_heun_single(self, initial, context, steps):
        """Differentiable fixed-step Heun integration for one action chunk."""
        dt = 1.0 / steps

        def step(current, index):
            t = index.astype(current.dtype) * dt
            first = self.vel_net(current, context, t)
            predictor = current + dt * first
            second = self.vel_net(predictor, context, jnp.minimum(t + dt, 1.0))
            following = current + 0.5 * dt * (first + second)
            return following, following

        final, generated = jax.lax.scan(step, initial, jnp.arange(steps))
        return final

    def _actual_score(self, x, context, t):
        """Analytic score recovered from velocity for a fixed N(0,I) source.

        For ``x_t = beta*x_1 + rho*epsilon``, where
        ``rho=sqrt(alpha**2+gamma**2)``, the velocity/score identity is

        ``score = -(beta_dot*x - beta*velocity) /
                   (rho*(beta_dot*rho-rho_dot*beta))``.

        It is valid only for the fixed Gaussian source used by the Door CSI.
        """
        velocity = self.vel_net(x, context, t)
        return self._score_from_velocity(x, velocity, t)

    def _score_from_velocity(self, x, velocity, t):
        """Evaluate the analytic identity from an already-computed velocity."""
        alpha = 1.0 - t
        alpha_dot = -1.0
        beta = t
        beta_dot = 1.0
        gamma = self.sigma_0 * t * (1.0 - t)
        gamma_dot = self.sigma_0 * (1.0 - 2.0 * t)
        rho = jnp.sqrt(alpha ** 2 + gamma ** 2)
        rho_dot = (alpha * alpha_dot + gamma * gamma_dot) / jnp.maximum(
            rho, 1e-8
        )
        denominator = rho * (beta_dot * rho - rho_dot * beta)
        denominator = jnp.maximum(denominator, 1e-8)
        return -(beta_dot * x - beta * velocity) / denominator

    def sample(
        self,
        states: jnp.ndarray,
        key: jr.PRNGKey,
        *,
        deterministic: bool = False,
        clip: float | None = 1.0,
        inference_steps: int | None = None,
    ) -> InterpolantSample:
        """Generate chunks with the sampler appropriate for this policy."""
        steps = self.integration_steps if inference_steps is None else int(inference_steps)
        if not 1 <= steps <= self.integration_steps:
            raise ValueError(
                f"inference_steps must be in [1, {self.integration_steps}]"
            )
        if deterministic:
            return self.sample_ode(states, key, clip=clip, steps=steps)
        if self.score_mode == "flow":
            if steps != self.integration_steps:
                raise ValueError(
                    "flow PPO exploration requires the trained integration grid"
                )
            return self.sample_flow_exploration(states, key, clip=clip)
        return self.sample_sde(states, key, clip=clip, steps=steps)

    def sample_flow_exploration(self, states, key, *, clip=1.0):
        """Euler flow matching with Gaussian exploration for PPO.

        The exploration kernel is not a score model: its fixed variance only
        makes the inner transitions stochastic and gives PPO a tractable
        likelihood. Evaluation still uses the deterministic ODE above.
        """
        self._check_states(states)
        batch = len(states)
        contexts = jax.vmap(self._features)(states)
        initial_key, noise_key = jr.split(key)
        initial = jr.normal(initial_key, (batch, self.horizon, self.action_dim))
        dt = 1.0 / self.integration_steps
        keys = jr.split(noise_key, self.integration_steps)

        def step(current, inputs):
            index, step_key = inputs
            t = (index.astype(current.dtype) + 0.5) * dt
            drift = jax.vmap(self.vel_net)(
                current, contexts, jnp.full((batch,), t)
            )
            following = (current + dt * drift
                         + self.sampling_std * jnp.sqrt(dt)
                         * jr.normal(step_key, current.shape))
            return following, following

        final, generated = jax.lax.scan(
            step, initial, (jnp.arange(self.integration_steps), keys)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_ode(self, states, key, *, clip=1.0, steps=None):
        """Deterministic probability-flow transport using the velocity head."""
        self._check_states(states)
        batch = len(states)
        contexts = jax.vmap(self._features)(states)
        initial = jr.normal(key, (batch, self.horizon, self.action_dim))
        steps = self.integration_steps if steps is None else int(steps)
        dt = 1.0 / steps

        def step(current, index):
            t = (index.astype(current.dtype) + 0.5) * dt
            times = jnp.full((batch,), t)
            drift = jax.vmap(self.vel_net)(current, contexts, times)
            following = current + dt * drift
            return following, following

        final, generated = jax.lax.scan(
            step, initial, jnp.arange(steps)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_ode_heun(self, states, key, *, clip=1.0):
        """Second-order Heun probability-flow sampling."""
        self._check_states(states)
        batch = len(states)
        contexts = jax.vmap(self._features)(states)
        initial = jr.normal(key, (batch, self.horizon, self.action_dim))
        dt = 1.0 / self.integration_steps

        def step(current, index):
            t = index.astype(current.dtype) * dt
            times = jnp.full((batch,), t)
            first = jax.vmap(self.vel_net)(current, contexts, times)
            predictor = current + dt * first
            following_times = jnp.minimum(times + dt, 1.0)
            second = jax.vmap(self.vel_net)(predictor, contexts, following_times)
            following = current + 0.5 * dt * (first + second)
            return following, following

        final, generated = jax.lax.scan(
            step, initial, jnp.arange(self.integration_steps)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_sde(self, states, key, *, clip=1.0, steps=None):
        """State-estimation-style refinement SDE for action chunks."""
        if self.score_mode == "flow":
            raise ValueError("flow-only policy does not define an SDE score")
        self._check_states(states)
        batch = len(states)
        contexts = jax.vmap(self._features)(states)
        initial_key, noise_key = jr.split(key)
        initial = jr.normal(initial_key, (batch, self.horizon, self.action_dim))
        steps = self.integration_steps if steps is None else int(steps)
        if not 1 <= steps <= self.integration_steps:
            raise ValueError(f"steps must be in [1, {self.integration_steps}]")
        dt = 1.0 / steps
        step_keys = jr.split(noise_key, steps)

        def step(current, inputs):
            index, noise_key = inputs
            t = (index.astype(current.dtype) + 0.5) * dt
            times = jnp.full((batch,), t)
            velocity = jax.vmap(self.vel_net)(current, contexts, times)
            score = jax.vmap(self._corrected_score)(
                current, velocity, contexts, times
            )
            score_weight, transition_std = self._sampling_coefficients(t, dt)
            drift = velocity + score_weight * score
            following = (
                current + dt * drift
                + transition_std
                * jr.normal(noise_key, current.shape)
            )
            return following, following

        final, generated = jax.lax.scan(
            step, initial, (jnp.arange(steps), step_keys)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_sde_heun(self, states, key, *, clip=1.0):
        """Stochastic Heun sampling with shared additive noise per step."""
        self._check_states(states)
        batch = len(states)
        if self.score_mode == "flow":
            raise ValueError("flow-only policy does not define an SDE score")
        contexts = jax.vmap(self._features)(states)
        initial_key, noise_key = jr.split(key)
        initial = jr.normal(initial_key, (batch, self.horizon, self.action_dim))
        dt = 1.0 / self.integration_steps
        step_keys = jr.split(noise_key, self.integration_steps)

        def drift(current, times, score_weight):
            velocity = jax.vmap(self.vel_net)(current, contexts, times)
            score = jax.vmap(self._corrected_score)(
                current, velocity, contexts, times
            )
            return velocity + score_weight * score

        def step(current, inputs):
            index, step_key = inputs
            t = (index.astype(current.dtype) + .5) * dt
            times = jnp.full((batch,), t)
            score_weight, transition_std = self._sampling_coefficients(t, dt)
            noise = transition_std * jr.normal(step_key, current.shape)
            first = drift(current, times, score_weight)
            predictor = current + dt * first + noise
            # Midpoint time coefficients avoid evaluating the singular
            # Gaussian-source score identity at the data endpoint t=1.
            second = drift(predictor, times, score_weight)
            following = current + 0.5 * dt * (first + second) + noise
            return following, following

        final, generated = jax.lax.scan(
            step, initial, (jnp.arange(self.integration_steps), step_keys)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def transition_log_prob(
        self,
        states: jnp.ndarray,
        chain: jnp.ndarray,
        indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Elementwise log density of selected inner transitions."""
        self._check_states(states)
        expected = (
            len(states),
            self.integration_steps + 1,
            self.horizon,
            self.action_dim,
        )
        if chain.shape != expected:
            raise ValueError(f"chain must have shape {expected}")
        if self.score_mode == "flow" and self.sampling_std <= 0:
            raise ValueError("flow PPO exploration requires sampling_std > 0")
        if self.score_mode != "flow" and self.sde_eta_schedule == "fixed" and self.sde_eta <= 0:
            raise ValueError("SDE likelihood requires sde_eta > 0")
        if indices is None:
            indices = jnp.arange(self.integration_steps)
        indices = jnp.asarray(indices)
        previous, following = chain[:, indices], chain[:, indices + 1]
        contexts = jax.vmap(self._features)(states)
        dt = 1.0 / self.integration_steps
        times = (indices.astype(chain.dtype) + 0.5) * dt

        etas = self.sde_etas[indices]

        def per_transition(prev, nxt, t, eta, context):
            velocity = self.vel_net(prev, context, t)
            if self.score_mode == "flow":
                drift = velocity
                std = jnp.maximum(self.sampling_std * jnp.sqrt(dt), 1e-10)
            else:
                score = self._corrected_score(prev, velocity, context, t)
                score_weight, std = self._sampling_coefficients(t, dt)
                drift = velocity + score_weight * score
            transition_mean = prev + dt * drift
            return -0.5 * (
                ((nxt - transition_mean) / std) ** 2
                + 2.0 * jnp.log(std)
                + jnp.log(2.0 * jnp.pi)
            )

        def per_batch(prevs, nexts, context):
            return jax.vmap(
                per_transition, in_axes=(0, 0, 0, 0, None)
            )(prevs, nexts, times, etas, context)

        return jax.vmap(per_batch)(previous, following, contexts)


class FlowMatchingPolicy(CondStochInterpolantPolicy):
    """Standalone Gaussian-source flow-matching policy (velocity/ODE only)."""

    def __init__(self, *args, **kwargs):
        kwargs["score_mode"] = "flow"
        kwargs["score_head_width"] = 0
        kwargs["score_loss_weight"] = 0.0
        super().__init__(*args, **kwargs)


class AnalyticScorePolicy(CondStochInterpolantPolicy):
    """Standalone SDE policy using the exact Gaussian-source analytic score."""

    def __init__(self, *args, **kwargs):
        kwargs["score_mode"] = "analytic"
        kwargs["score_head_width"] = 0
        kwargs["score_loss_weight"] = 0.0
        super().__init__(*args, **kwargs)


class LearnedScorePolicy(CondStochInterpolantPolicy):
    """Standalone SDE policy with an independently learned conditional score."""

    def __init__(self, *args, **kwargs):
        kwargs["score_mode"] = "learned"
        super().__init__(*args, **kwargs)
