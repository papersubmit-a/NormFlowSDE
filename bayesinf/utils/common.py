"""Common utility functions for Bayesian inference."""

import jax
import jax.numpy as jnp
from typing import Tuple, Callable


def gaussian_kl_divergence(mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Compute KL divergence between N(mean, var) and N(0, 1).

    Args:
        mean: Mean of the distribution (batch_size, latent_dim)
        logvar: Log variance of the distribution (batch_size, latent_dim)

    Returns:
        KL divergence summed over latent dimensions, shape (batch_size,)
    """
    return -0.5 * jnp.sum(1 + logvar - mean**2 - jnp.exp(logvar), axis=-1)


def reparameterize(rng: jax.random.PRNGKey, mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Reparameterization trick: sample from N(mean, var) using N(0, 1).

    Args:
        rng: JAX random key
        mean: Mean of the distribution
        logvar: Log variance of the distribution

    Returns:
        Sampled latent vector
    """
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, mean.shape)
    return mean + eps * std


def gaussian_log_prob(x: jnp.ndarray, mean: jnp.ndarray, logvar: jnp.ndarray) -> jnp.ndarray:
    """
    Compute log probability of x under N(mean, var).

    Args:
        x: Sample
        mean: Mean of the distribution
        logvar: Log variance of the distribution

    Returns:
        Log probability
    """
    return -0.5 * (jnp.log(2 * jnp.pi) + logvar + ((x - mean) ** 2) / jnp.exp(logvar))


def split_key(rng: jax.random.PRNGKey, num: int = 2) -> Tuple[jax.random.PRNGKey, ...]:
    """
    Split a JAX random key into multiple keys.

    Args:
        rng: JAX random key
        num: Number of keys to generate

    Returns:
        Tuple of random keys
    """
    return tuple(jax.random.split(rng, num))
