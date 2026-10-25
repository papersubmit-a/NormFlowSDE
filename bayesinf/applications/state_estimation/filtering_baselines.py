"""Model-agnostic extended/unscented Kalman and particle filters."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import gammaln, logsumexp

@dataclass
@jax.tree_util.register_pytree_node_class
class FilterResult:
    """Filtering posterior summaries, with optional weighted diagnostics."""

    mean: jnp.ndarray                 # (batch, time, state)
    covariance: jnp.ndarray           # (batch, time, state, state)
    particles: Optional[jnp.ndarray] = None  # (batch, time, particle, state)
    weights: Optional[jnp.ndarray] = None  # (batch, time, particle)
    effective_sample_size: Optional[jnp.ndarray] = None  # (batch, time)
    resampled: Optional[jnp.ndarray] = None  # (batch, time)
    credible_intervals: Optional[jnp.ndarray] = None  # (batch,time,level,2,state)
    ancestors: Optional[jnp.ndarray] = None  # (batch,time,particle)

    def tree_flatten(self):
        return (
            (
                self.mean,
                self.covariance,
                self.particles,
                self.weights,
                self.effective_sample_size,
                self.resampled,
                self.credible_intervals,
                self.ancestors,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


@dataclass(frozen=True)
class ParticleFilterConfig:
    """Bootstrap-filter choices for learned stochastic transition models."""

    num_particles: int = 1_000
    initial_std: float = 0.1
    resampling_threshold: float = 0.5
    resampling_method: str = "systematic"
    credible_levels: tuple[float, ...] = (0.5, 0.8, 0.95)
    track_ancestors: bool = False

    def __post_init__(self):
        if self.num_particles < 2:
            raise ValueError("num_particles must be at least two")
        if self.initial_std < 0:
            raise ValueError("initial_std must be non-negative")
        if not 0 <= self.resampling_threshold <= 1:
            raise ValueError("resampling_threshold must lie in [0, 1]")
        if self.resampling_method not in {"systematic", "stratified"}:
            raise ValueError("resampling_method must be systematic or stratified")
        if not self.credible_levels or any(
            not 0 < level < 1 for level in self.credible_levels
        ):
            raise ValueError("credible_levels must lie strictly between zero and one")


def _as_batch(y: jnp.ndarray, x0: jnp.ndarray):
    single = y.ndim == 2
    if single:
        y, x0 = y[None], x0[None]
    if y.ndim != 3 or x0.ndim != 2 or y.shape[0] != x0.shape[0]:
        raise ValueError("Expected y=(B,T,Dy), x0=(B,Dx), or their unbatched forms")
    return y, x0, single


def extended_kalman_filter(
    y: jnp.ndarray,
    x0: jnp.ndarray,
    H: jnp.ndarray,
    transition: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    obs_std: float = 0.5,
    process_std: float = 0.05,
    initial_std: float = 0.0,
) -> FilterResult:
    """Run an EKF using a supplied transition and its automatic Jacobian."""
    if obs_std <= 0 or process_std < 0 or initial_std < 0:
        raise ValueError("obs_std must be positive and state-noise scales non-negative")
    y, x0, single = _as_batch(jnp.asarray(y), jnp.asarray(x0))
    H = jnp.asarray(H)
    n, dy = x0.shape[-1], y.shape[-1]
    if H.shape != (dy, n):
        raise ValueError(f"H must have shape {(dy, n)}, got {H.shape}")

    jacobian = jax.jacfwd(transition)
    eye = jnp.eye(n, dtype=x0.dtype)
    Q = process_std**2 * eye
    R = obs_std**2 * jnp.eye(dy, dtype=x0.dtype)

    def one_sequence(obs, initial):
        def step(carry, measurement):
            mean, cov = carry
            A = jacobian(mean)
            pred = transition(mean)
            pred_cov = A @ cov @ A.T + Q
            innovation_cov = H @ pred_cov @ H.T + R
            # solve is more stable than explicitly forming inv(S)
            gain = jnp.linalg.solve(innovation_cov, H @ pred_cov).T
            mean = pred + gain @ (measurement - H @ pred)
            ikh = eye - gain @ H
            cov = ikh @ pred_cov @ ikh.T + gain @ R @ gain.T
            cov = 0.5 * (cov + cov.T)
            return (mean, cov), (mean, cov)

        init_cov = initial_std**2 * eye
        _, (means, covs) = jax.lax.scan(step, (initial, init_cov), obs)
        return means, covs

    means, covs = jax.vmap(one_sequence)(y, x0)
    if single:
        means, covs = means[0], covs[0]
    return FilterResult(means, covs)


def unscented_kalman_filter(
    y: jnp.ndarray,
    x0: jnp.ndarray,
    H: jnp.ndarray,
    transition: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    obs_std: float = 0.5,
    process_std: float = 0.05,
    initial_std: float = 0.0,
    alpha: float = 0.5,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> FilterResult:
    """Run a scaled unscented Kalman filter with the exact RK4 transition."""
    if obs_std <= 0 or process_std < 0 or initial_std < 0 or alpha <= 0:
        raise ValueError("Noise scales must be valid and alpha positive")
    y, x0, single = _as_batch(jnp.asarray(y), jnp.asarray(x0))
    H = jnp.asarray(H)
    n, dy = x0.shape[-1], y.shape[-1]
    if H.shape != (dy, n):
        raise ValueError(f"H must have shape {(dy, n)}, got {H.shape}")

    lam = alpha**2 * (n + kappa) - n
    scale = n + lam
    wm = jnp.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=x0.dtype)
    wc = wm.at[0].set(lam / scale + (1.0 - alpha**2 + beta))
    wm = wm.at[0].set(lam / scale)
    eye = jnp.eye(n, dtype=x0.dtype)
    Q = process_std**2 * eye
    R = obs_std**2 * jnp.eye(dy, dtype=x0.dtype)

    def sigma_points(mean, covariance):
        root = jnp.linalg.cholesky(scale * (covariance + 1e-7 * eye))
        offsets = root.T
        return jnp.concatenate((mean[None], mean + offsets, mean - offsets), axis=0)

    def one_sequence(obs, initial):
        def step(carry, measurement):
            mean, covariance = carry
            sigma = sigma_points(mean, covariance)
            propagated = jax.vmap(transition)(sigma)
            pred_mean = jnp.sum(wm[:, None] * propagated, axis=0)
            dx = propagated - pred_mean
            pred_cov = jnp.einsum("s,si,sj->ij", wc, dx, dx) + Q

            # H is linear: include process uncertainty in the measurement
            # moments, not just in the predicted state covariance.
            obs_mean = H @ pred_mean
            innovation_cov = H @ pred_cov @ H.T + R
            cross_cov = pred_cov @ H.T
            gain = jnp.linalg.solve(innovation_cov, cross_cov.T).T
            mean = pred_mean + gain @ (measurement - obs_mean)
            covariance = pred_cov - gain @ innovation_cov @ gain.T
            covariance = 0.5 * (covariance + covariance.T)
            return (mean, covariance), (mean, covariance)

        init_cov = initial_std**2 * eye
        _, (means, covs) = jax.lax.scan(step, (initial, init_cov), obs)
        return means, covs

    means, covs = jax.vmap(one_sequence)(y, x0)
    if single:
        means, covs = means[0], covs[0]
    return FilterResult(means, covs)


def _observation_log_likelihood(
    residual: jnp.ndarray,
    noise_type: str,
    obs_std: float,
    obs_df: float,
    gmm_weights: Sequence[float],
    gmm_offsets: Sequence[float],
    gmm_stds: Sequence[float],
) -> jnp.ndarray:
    """Log likelihood per particle; observation dimensions are independent."""
    if noise_type == "gaussian":
        z = residual / obs_std
        return jnp.sum(-0.5 * z**2 - jnp.log(obs_std) - 0.5 * jnp.log(2 * jnp.pi), -1)
    if noise_type == "student_t":
        nu = jnp.asarray(obs_df, residual.dtype)
        z = residual / obs_std
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(obs_std)
        return jnp.sum(c - 0.5 * (nu + 1) * jnp.log1p(z**2 / nu), -1)
    if noise_type == "gmm":
        w = jnp.asarray(gmm_weights, residual.dtype)
        mu = obs_std * jnp.asarray(gmm_offsets, residual.dtype)
        sd = obs_std * jnp.asarray(gmm_stds, residual.dtype)
        if not (w.ndim == mu.ndim == sd.ndim == 1 and len(w) == len(mu) == len(sd)):
            raise ValueError("GMM weights, offsets, and stds must be equal-length vectors")
        z = (residual[..., None] - mu) / sd
        components = jnp.log(w / jnp.sum(w)) - jnp.log(sd) - 0.5 * (z**2 + jnp.log(2 * jnp.pi))
        return jnp.sum(logsumexp(components, axis=-1), axis=-1)
    raise ValueError(f"Unknown observation noise type: {noise_type}")


def bootstrap_particle_filter(
    y: jnp.ndarray,
    x0: jnp.ndarray,
    H: jnp.ndarray,
    key: jax.Array,
    transition: Callable[[jnp.ndarray], jnp.ndarray],
    *,
    obs_std: float = 0.5,
    process_std: float = 0.05,
    initial_std: float = 0.0,
    num_particles: int = 1000,
    noise_type: str = "student_t",
    obs_df: float = 5.0,
    gmm_weights: Sequence[float] = (0.70, 0.15, 0.15),
    gmm_offsets: Sequence[float] = (0.0, 2.5, -2.5),
    gmm_stds: Sequence[float] = (0.50, 1.50, 1.50),
) -> FilterResult:
    """Bootstrap/SIR particle filter with the experiment's exact noise model."""
    if num_particles < 2 or obs_std <= 0 or process_std < 0 or initial_std < 0:
        raise ValueError("Need >=2 particles, positive obs_std, and non-negative state noise")
    if noise_type == "gmm" and (
        any(value <= 0 for value in gmm_weights)
        or any(value <= 0 for value in gmm_stds)
    ):
        raise ValueError("GMM weights and standard deviations must be positive")
    y, x0, single = _as_batch(jnp.asarray(y), jnp.asarray(x0))
    H = jnp.asarray(H)
    batch, _, n = y.shape[0], y.shape[1], x0.shape[-1]
    if H.shape != (y.shape[-1], n):
        raise ValueError(f"H must have shape {(y.shape[-1], n)}, got {H.shape}")
    sequence_keys = jr.split(key, batch)

    def one_sequence(obs, initial, seq_key):
        seq_key, init_key = jr.split(seq_key)
        particles = initial + initial_std * jr.normal(init_key, (num_particles, n))

        def step(carry, measurement):
            particles, step_key = carry
            step_key, noise_key, sample_key = jr.split(step_key, 3)
            particles = jax.vmap(transition)(particles)
            particles = particles + process_std * jr.normal(noise_key, particles.shape)
            residual = measurement[None, :] - particles @ H.T
            logw = _observation_log_likelihood(
                residual, noise_type, obs_std, obs_df,
                gmm_weights, gmm_offsets, gmm_stds,
            )
            logw = logw - logsumexp(logw)
            indices = jr.categorical(sample_key, logw, shape=(num_particles,))
            particles = particles[indices]
            mean = jnp.mean(particles, axis=0)
            centered = particles - mean
            cov = centered.T @ centered / (num_particles - 1)
            return (particles, step_key), (mean, cov, particles)

        _, (means, covs, history) = jax.lax.scan(step, (particles, seq_key), obs)
        return means, covs, history

    means, covs, particles = jax.vmap(one_sequence)(y, x0, sequence_keys)
    if single:
        means, covs, particles = means[0], covs[0], particles[0]
    return FilterResult(means, covs, particles)


def _normalize_log_weights(log_weights):
    normalizer = logsumexp(log_weights)
    normalized = log_weights - normalizer
    uniform = jnp.full_like(log_weights, -jnp.log(log_weights.shape[0]))
    return jnp.where(jnp.isfinite(normalizer), normalized, uniform)


def _resampling_indices(key, weights, method):
    """Systematic or stratified resampling from normalized weights."""
    count = weights.shape[0]
    if method == "systematic":
        positions = (
            jr.uniform(key, (), dtype=weights.dtype) + jnp.arange(count)
        ) / count
    elif method == "stratified":
        positions = (
            jnp.arange(count)
            + jr.uniform(key, (count,), dtype=weights.dtype)
        ) / count
    else:
        raise ValueError("resampling method must be systematic or stratified")
    cumulative = jnp.cumsum(weights).at[-1].set(1.0)
    return jnp.searchsorted(cumulative, positions, side="right").astype(jnp.int32)


def _weighted_particle_statistics(particles, log_weights, credible_levels):
    weights = jnp.exp(log_weights)
    mean = jnp.sum(weights[:, None] * particles, axis=0)
    centered = particles - mean
    covariance = jnp.einsum("n,ni,nj->ij", weights, centered, centered)
    covariance = 0.5 * (covariance + covariance.T)
    ess = 1.0 / jnp.sum(weights**2)
    quantiles = jnp.asarray(
        tuple(
            quantile
            for level in credible_levels
            for quantile in ((1.0 - level) / 2.0, (1.0 + level) / 2.0)
        ),
        dtype=particles.dtype,
    )

    def coordinate_quantiles(values):
        order = jnp.argsort(values)
        sorted_values = values[order]
        cumulative = jnp.cumsum(weights[order]).at[-1].set(1.0)
        indices = jnp.searchsorted(cumulative, quantiles, side="left")
        return sorted_values[jnp.clip(indices, 0, values.shape[0] - 1)]

    intervals = jax.vmap(coordinate_quantiles, in_axes=1, out_axes=1)(particles)
    intervals = intervals.reshape((len(credible_levels), 2, particles.shape[-1]))
    return weights, mean, covariance, ess, intervals


def bootstrap_particle_filter_nsf(
    observations: jnp.ndarray,
    x0: jnp.ndarray,
    times: jnp.ndarray,
    key: jax.Array,
    transition_model,
    observation_model,
    *,
    controls: Optional[jnp.ndarray] = None,
    mask: Optional[jnp.ndarray] = None,
    config: ParticleFilterConfig = ParticleFilterConfig(),
) -> FilterResult:
    """Bootstrap filter using an NSF transition as the proposal distribution.

    ``observations`` has shape ``(T,O)`` or ``(B,T,O)`` and includes the
    measurement at the initial time. ``x0`` is the corresponding prior mean.
    The first measurement updates initialized particles without a transition;
    later steps call ``transition_model.sample`` once on the full particle
    batch. Posterior weights are retained for summaries, while conditional
    resampling affects only the particles carried to the next time.
    """
    observations = jnp.asarray(observations)
    x0 = jnp.asarray(x0, dtype=observations.dtype)
    single = observations.ndim == 2
    if single:
        observations, x0 = observations[None], x0[None]
    if observations.ndim != 3 or x0.ndim != 2:
        raise ValueError("expected observations=(B,T,O), x0=(B,D), or unbatched forms")
    batch_size, num_times, observation_dim = observations.shape
    if num_times < 1 or x0.shape[0] != batch_size:
        raise ValueError("observations and x0 must have matching non-empty batches")
    if observation_dim != observation_model.observation_dim:
        raise ValueError("observation dimension does not match observation model")
    if x0.shape[-1] != transition_model.state_dim:
        raise ValueError("x0 state dimension does not match transition model")

    times = jnp.asarray(times, dtype=observations.dtype)
    if times.ndim == 1:
        times = jnp.broadcast_to(times, (batch_size, num_times))
    elif single and times.ndim == 2 and times.shape == (1, num_times):
        pass
    if times.shape != (batch_size, num_times):
        raise ValueError(f"times must have shape {(num_times,)} or {(batch_size, num_times)}")
    delta_t = jnp.diff(times, axis=1)
    if not isinstance(delta_t, jax.core.Tracer) and np.any(np.asarray(delta_t) <= 0):
        raise ValueError("times must be strictly increasing")

    if mask is None:
        masks = jnp.ones_like(observations, dtype=bool)
    else:
        masks = jnp.asarray(mask, dtype=bool)
        if single and masks.ndim == 2:
            masks = masks[None]
        masks = jnp.broadcast_to(masks, observations.shape)

    control_dim = int(getattr(transition_model, "control_dim", 0))
    if controls is None:
        if control_dim:
            raise ValueError("controls are required by the transition model")
        controls = jnp.empty(
            (batch_size, max(num_times - 1, 0), 0), dtype=observations.dtype
        )
    else:
        controls = jnp.asarray(controls, dtype=observations.dtype)
        if single and controls.ndim == 2:
            controls = controls[None]
        expected = (batch_size, max(num_times - 1, 0), control_dim)
        if controls.shape != expected:
            raise ValueError(f"controls must have shape {expected}")

    sequence_keys = jr.split(key, batch_size)
    count = config.num_particles
    uniform_log_weights = jnp.full(
        (count,), -jnp.log(count), dtype=observations.dtype
    )
    identity_indices = jnp.arange(count, dtype=jnp.int32)

    def one_sequence(obs, initial, sequence_times, gaps, sequence_controls,
                     sequence_mask, sequence_key):
        init_key, first_resample_key, scan_key = jr.split(sequence_key, 3)
        particles = initial + config.initial_std * jr.normal(
            init_key, (count, transition_model.state_dim), dtype=initial.dtype
        )

        def measurement_update(current_particles, prior_log_weights,
                               measurement, measurement_mask):
            likelihood = observation_model.log_prob(
                measurement, current_particles, measurement_mask
            )
            posterior_log_weights = _normalize_log_weights(
                prior_log_weights + likelihood
            )
            statistics = _weighted_particle_statistics(
                current_particles,
                posterior_log_weights,
                config.credible_levels,
            )
            return posterior_log_weights, statistics

        def carry_after_update(current_particles, posterior_log_weights,
                               ess, resample_key):
            should_resample = ess < config.resampling_threshold * count
            proposed_indices = _resampling_indices(
                resample_key,
                jnp.exp(posterior_log_weights),
                config.resampling_method,
            )
            indices = jnp.where(
                should_resample, proposed_indices, identity_indices
            )
            carried_particles = current_particles[indices]
            carried_log_weights = jnp.where(
                should_resample, uniform_log_weights, posterior_log_weights
            )
            return carried_particles, carried_log_weights, should_resample, indices

        first_log_weights, first_statistics = measurement_update(
            particles, uniform_log_weights, obs[0], sequence_mask[0]
        )
        first_weights, first_mean, first_cov, first_ess, first_intervals = (
            first_statistics
        )
        first_posterior_particles = particles
        particles, carried_log_weights, first_resampled, first_ancestors = (
            carry_after_update(
                particles, first_log_weights, first_ess, first_resample_key
            )
        )
        first_output = (
            first_mean,
            first_cov,
            first_posterior_particles,
            first_weights,
            first_ess,
            first_resampled,
            first_intervals,
            first_ancestors,
        )

        step_keys = jr.split(scan_key, max(num_times - 1, 0))

        def step(carry, inputs):
            current_particles, prior_log_weights = carry
            measurement, gap, control, measurement_mask, start_time, step_key = inputs
            transition_key, resample_key = jr.split(step_key)
            control_value = control if control_dim else None
            predicted = transition_model.sample(
                transition_key,
                current_particles,
                gap,
                control=control_value,
                num_samples=1,
                t_current=start_time,
            )[0]
            posterior_log_weights, statistics = measurement_update(
                predicted, prior_log_weights, measurement, measurement_mask
            )
            weights, mean, covariance, ess, intervals = statistics
            next_particles, next_log_weights, resampled, ancestors = (
                carry_after_update(
                    predicted, posterior_log_weights, ess, resample_key
                )
            )
            output = (
                mean,
                covariance,
                predicted,
                weights,
                ess,
                resampled,
                intervals,
                ancestors,
            )
            return (next_particles, next_log_weights), output

        _, remaining = jax.lax.scan(
            step,
            (particles, carried_log_weights),
            (
                obs[1:],
                gaps,
                sequence_controls,
                sequence_mask[1:],
                sequence_times[:-1],
                step_keys,
            ),
        )
        return tuple(
            jnp.concatenate((first[None], rest), axis=0)
            for first, rest in zip(first_output, remaining)
        )

    outputs = jax.vmap(one_sequence)(
        observations,
        x0,
        times,
        delta_t,
        controls,
        masks,
        sequence_keys,
    )
    means, covariances, particles, weights, ess, resampled, intervals, ancestors = outputs
    if not config.track_ancestors:
        ancestors = None
    if single:
        means = means[0]
        covariances = covariances[0]
        particles = particles[0]
        weights = weights[0]
        ess = ess[0]
        resampled = resampled[0]
        intervals = intervals[0]
        if ancestors is not None:
            ancestors = ancestors[0]
    return FilterResult(
        means,
        covariances,
        particles,
        weights,
        ess,
        resampled,
        intervals,
        ancestors,
    )


# Alternate word order retained for discoverability in experiment code.
bootstrap_nsf_particle_filter = bootstrap_particle_filter_nsf


def reconstruct_particle_trajectories(
    result: FilterResult,
    final_indices: Optional[jnp.ndarray] = None,
):
    """Trace posterior particle lineages backward through stored ancestors.

    The return shape is ``(N,T,D)`` for an unbatched result and ``(B,N,T,D)``
    for a batched result. ``ancestors[t, j]`` identifies the time-``t``
    posterior particle selected into carry slot ``j`` for the next transition;
    the final time's resampling record is intentionally unused.
    """
    if result.particles is None or result.ancestors is None:
        raise ValueError("particle histories and ancestor tracking are required")
    particles = jnp.asarray(result.particles)
    ancestors = jnp.asarray(result.ancestors)
    single = particles.ndim == 3
    if single:
        particles, ancestors = particles[None], ancestors[None]
    if particles.ndim != 4 or ancestors.shape != particles.shape[:-1]:
        raise ValueError("incompatible particle and ancestor history shapes")
    batch_size, num_times, num_particles, _ = particles.shape
    if final_indices is None:
        indices = jnp.broadcast_to(
            jnp.arange(num_particles, dtype=jnp.int32),
            (batch_size, num_particles),
        )
    else:
        indices = jnp.asarray(final_indices, dtype=jnp.int32)
        if single and indices.ndim == 1:
            indices = indices[None]
        if indices.ndim != 2 or indices.shape[0] != batch_size:
            raise ValueError("final_indices must have shape (N,) or (B,N)")

    batch_indices = jnp.arange(batch_size)[:, None]
    states = [particles[batch_indices, num_times - 1, indices]]
    for time_index in range(num_times - 2, -1, -1):
        indices = ancestors[batch_indices, time_index, indices]
        states.append(particles[batch_indices, time_index, indices])
    trajectories = jnp.stack(states[::-1], axis=2)
    return trajectories[0] if single else trajectories
