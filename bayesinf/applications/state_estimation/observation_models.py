"""JAX-compatible observation models for Bayesian state estimation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Optional, Protocol, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


LOG_2PI = float(np.log(2.0 * np.pi))


class ObservationModel(Protocol):
    """Minimal observation-model interface consumed by particle filters."""

    observation_dim: int

    def mean(self, state: jax.Array) -> jax.Array:
        """Return the conditional observation mean for one or more states."""

    def log_prob(
        self,
        observation: jax.Array,
        state: jax.Array,
        mask: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Return log ``p(observation | state)`` over observed coordinates."""


def _noise_parameters(noise_std, observation_dim):
    values = np.asarray(noise_std, dtype=float)
    if values.ndim == 0:
        values = np.full((observation_dim,), float(values))
    if values.shape != (observation_dim,):
        raise ValueError(
            f"noise_std must be scalar or have shape {(observation_dim,)}"
        )
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise ValueError("noise_std must be finite and strictly positive")
    return tuple(float(value) for value in values)


def _diagonal_gaussian_log_prob(observation, predicted, noise_std, mask):
    observation = jnp.asarray(observation, dtype=predicted.dtype)
    observation = jnp.broadcast_to(observation, predicted.shape)
    std = jnp.asarray(noise_std, dtype=predicted.dtype)
    residual = observation - predicted
    terms = -0.5 * ((residual / std) ** 2 + LOG_2PI) - jnp.log(std)
    if mask is not None:
        valid = jnp.broadcast_to(jnp.asarray(mask, dtype=bool), predicted.shape)
        # Replacing missing entries after evaluating their residual also makes
        # NaN placeholders harmless to the returned likelihood.
        terms = jnp.where(valid, terms, 0.0)
    return jnp.sum(terms, axis=-1)


class _DiagonalGaussianObservation:
    """Shared likelihood and sampling operations for fixed Gaussian noise."""

    @property
    def noise_std(self):
        return jnp.asarray(self._noise_std)

    def log_prob(self, observation, state, mask=None):
        predicted = self.mean(state)
        return _diagonal_gaussian_log_prob(
            observation, predicted, self._noise_std, mask
        )

    def sample(self, key, state):
        predicted = self.mean(state)
        std = jnp.asarray(self._noise_std, dtype=predicted.dtype)
        return predicted + std * jr.normal(key, predicted.shape, dtype=predicted.dtype)


class IdentityGaussianObservation(_DiagonalGaussianObservation, eqx.Module):
    """Identity observation ``y=x+v`` with diagonal Gaussian noise."""

    _noise_std: tuple[float, ...] = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)

    def __init__(self, state_dim: int, noise_std: float | Sequence[float]):
        if state_dim < 1:
            raise ValueError("state_dim must be positive")
        self.observation_dim = state_dim
        self._noise_std = _noise_parameters(noise_std, state_dim)

    def mean(self, state):
        state = jnp.asarray(state)
        if state.ndim < 1 or state.shape[-1] != self.observation_dim:
            raise ValueError(
                f"last state axis must be {self.observation_dim}"
            )
        return state


class LinearGaussianObservation(_DiagonalGaussianObservation, eqx.Module):
    """Linear observation ``y=H x+v``, including partial observations."""

    matrix: jax.Array
    _noise_std: tuple[float, ...] = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(self, matrix: jax.Array, noise_std: float | Sequence[float]):
        matrix = jnp.asarray(matrix)
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ValueError("observation matrix must have shape (O, D)")
        self.matrix = matrix
        self.observation_dim, self.state_dim = matrix.shape
        self._noise_std = _noise_parameters(noise_std, self.observation_dim)

    def mean(self, state):
        state = jnp.asarray(state, dtype=self.matrix.dtype)
        if state.ndim < 1 or state.shape[-1] != self.state_dim:
            raise ValueError(f"last state axis must be {self.state_dim}")
        return state @ self.matrix.T


class NonlinearGaussianObservation(_DiagonalGaussianObservation, eqx.Module):
    """Callable nonlinear observation mean with diagonal Gaussian noise."""

    function: Callable = eqx.field(static=True)
    _noise_std: tuple[float, ...] = eqx.field(static=True)
    observation_dim: int = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[jax.Array], jax.Array],
        observation_dim: int,
        noise_std: float | Sequence[float],
    ):
        if not callable(function) or observation_dim < 1:
            raise ValueError("function must be callable and observation_dim positive")
        self.function = function
        self.observation_dim = observation_dim
        self._noise_std = _noise_parameters(noise_std, observation_dim)

    def mean(self, state):
        state = jnp.asarray(state)
        if state.ndim < 1:
            raise ValueError("state must have a final feature axis")
        if state.ndim == 1:
            value = jnp.asarray(self.function(state))
        else:
            flat = state.reshape((-1, state.shape[-1]))
            value = jax.vmap(self.function)(flat).reshape(
                state.shape[:-1] + (self.observation_dim,)
            )
        if value.shape[-1:] != (self.observation_dim,):
            raise ValueError(
                f"observation function must return {self.observation_dim} values"
            )
        return value


# Descriptive aliases for downstream code that includes the ``Model`` suffix.
IdentityGaussianObservationModel = IdentityGaussianObservation
LinearGaussianObservationModel = LinearGaussianObservation
NonlinearGaussianObservationModel = NonlinearGaussianObservation


__all__ = (
    "IdentityGaussianObservation",
    "IdentityGaussianObservationModel",
    "LinearGaussianObservation",
    "LinearGaussianObservationModel",
    "NonlinearGaussianObservation",
    "NonlinearGaussianObservationModel",
    "ObservationModel",
)
