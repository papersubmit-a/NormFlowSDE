"""Proper Monte Carlo scores for learned predictive distributions."""

import jax.numpy as jnp


def energy_score(samples, target, *, weights=None, spread_weight=0.5):
    """Unbiased energy-score estimator, over independent particle draws.

    Particle axis is first; remaining non-feature axes are scored separately.
    ``spread_weight=.5`` is the proper energy score. Other values explicitly
    choose a different, generally uncalibrated training objective.
    """
    count = samples.shape[0]
    if count < 2:
        raise ValueError("energy-score training requires at least two particles")
    weights = 1.0 if weights is None else jnp.asarray(weights)
    distance = lambda delta: jnp.sqrt(jnp.sum(weights * delta ** 2, axis=-1) + 1e-6)
    accuracy = distance(samples - target[None]).mean(axis=0)
    pairs = distance(samples[:, None] - samples[None, :])
    # The smoothed diagonal is nonzero, so explicitly remove it as well.
    mask = (1.0 - jnp.eye(count)).reshape((count, count) + (1,) * (pairs.ndim - 2))
    spread = jnp.sum(pairs * mask, axis=(0, 1)) / (count * (count - 1))
    return accuracy - spread_weight * spread


def squared_mean_error(samples, target, *, weights=None):
    """Unbiased squared error of the population mean, without variance/N."""
    count = samples.shape[0]
    if count < 2:
        raise ValueError("mean-score training requires at least two particles")
    weights = 1.0 if weights is None else jnp.asarray(weights)
    residual = samples - target
    cross = (jnp.sum(residual, axis=0) ** 2 - jnp.sum(residual ** 2, axis=0))
    return jnp.mean(weights * cross / (count * (count - 1)))
