"""Synthetic controlled stochastic systems for transition benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


@dataclass(frozen=True)
class ControlledDuffingOscillator:
    """Stochastic nonlinear controlled Duffing oscillator.

    The physical state is position/velocity and the scalar command is held
    piecewise constant over each observation interval:

    ``dq = v dt + sigma_q dW_q``
    ``dv = (gain*u - damping*v - linear*q - cubic*q^3)dt + sigma_v dW_v``.

    Euler--Maruyama substeps define the synthetic reference process; this class
    is a validation system, not a claim of an analytic finite-time density.
    """

    damping: float = 0.35
    linear_stiffness: float = 0.8
    cubic_stiffness: float = 0.25
    control_gain: float = 1.0
    position_diffusion: float = 0.04
    velocity_diffusion: float = 0.18
    integration_substeps: int = 8
    state_dim: int = 2
    control_dim: int = 1

    def __post_init__(self):
        if self.damping < 0 or self.linear_stiffness < 0 or self.cubic_stiffness < 0:
            raise ValueError("damping and stiffness coefficients must be non-negative")
        if self.control_gain <= 0:
            raise ValueError("control_gain must be positive")
        if self.position_diffusion < 0 or self.velocity_diffusion < 0:
            raise ValueError("diffusion scales must be non-negative")
        if self.integration_substeps < 1:
            raise ValueError("integration_substeps must be positive")

    def drift(self, state, control):
        state, control = jnp.asarray(state), jnp.asarray(control)
        if state.shape[-1:] != (2,) or control.shape[-1:] != (1,):
            raise ValueError("Duffing states/controls must end in dimensions 2/1")
        position, velocity = state[..., 0], state[..., 1]
        acceleration = (
            self.control_gain * control[..., 0]
            - self.damping * velocity
            - self.linear_stiffness * position
            - self.cubic_stiffness * position**3
        )
        return jnp.stack((velocity, acceleration), axis=-1)

    def sample_step(self, key, state, control, delta_t):
        """Advance one observation interval using independent EM substep noise."""
        state = jnp.asarray(state)
        control = jnp.asarray(control, dtype=state.dtype)
        dt = jnp.asarray(delta_t, dtype=state.dtype)
        if not isinstance(dt, jax.core.Tracer) and np.any(np.asarray(dt) <= 0):
            raise ValueError("delta_t must be strictly positive")
        sub_dt = dt / self.integration_substeps
        diffusion = jnp.asarray(
            (self.position_diffusion, self.velocity_diffusion), dtype=state.dtype
        )
        noises = jr.normal(
            key,
            (self.integration_substeps,) + state.shape,
            dtype=state.dtype,
        )

        def substep(current, noise):
            following = (
                current
                + self.drift(current, control) * sub_dt
                + diffusion * jnp.sqrt(sub_dt) * noise
            )
            return following, None

        following, _ = jax.lax.scan(substep, state, noises)
        return following

    def sample_rollout(self, key, initial, controls, delta_t):
        controls = jnp.asarray(controls)
        if controls.ndim < 2 or controls.shape[-1] != self.control_dim:
            raise ValueError("controls must have shape (..., H, 1)")
        horizon = controls.shape[-2]
        gaps = jnp.asarray(delta_t)
        if gaps.ndim == 0:
            gaps = jnp.broadcast_to(gaps, (horizon,))
        if gaps.shape != (horizon,):
            raise ValueError(f"delta_t must be scalar or have shape {(horizon,)}")
        keys = jr.split(key, horizon)

        def step(state, inputs):
            command, gap, step_key = inputs
            following = self.sample_step(step_key, state, command, gap)
            return following, following

        _, states = jax.lax.scan(step, initial, (controls, gaps, keys))
        return states


@dataclass(frozen=True)
class ControlledStochasticPendulum:
    """Torque-controlled stochastic pendulum in unwrapped-angle coordinates.

    ``dtheta = omega dt + sigma_theta dW_theta``
    ``domega = (torque/(m*l^2) - damping*omega
                - gravity*sin(theta)/length)dt + sigma_omega dW_omega``.

    Keeping the angle unwrapped avoids an artificial density discontinuity at
    ``+/-pi``. Consumers may wrap it only for visualization.
    """

    gravity: float = 9.81
    length: float = 1.0
    mass: float = 1.0
    damping: float = 0.2
    angle_diffusion: float = 0.01
    velocity_diffusion: float = 0.2
    integration_substeps: int = 8
    noise_model: str = "gaussian"
    gmm_separation: float = 0.8
    state_dim: int = 2
    control_dim: int = 1

    def __post_init__(self):
        if self.gravity <= 0 or self.length <= 0 or self.mass <= 0:
            raise ValueError("gravity, length, and mass must be positive")
        if self.damping < 0 or self.angle_diffusion < 0 or self.velocity_diffusion < 0:
            raise ValueError("damping and diffusion scales must be non-negative")
        if self.integration_substeps < 1:
            raise ValueError("integration_substeps must be positive")
        if self.noise_model not in {"gaussian", "gmm"}:
            raise ValueError("noise_model must be 'gaussian' or 'gmm'")
        if not 0 <= self.gmm_separation < 1:
            raise ValueError("gmm_separation must lie in [0, 1)")

    def drift(self, state, control):
        state, control = jnp.asarray(state), jnp.asarray(control)
        if state.shape[-1:] != (2,) or control.shape[-1:] != (1,):
            raise ValueError("pendulum states/controls must end in dimensions 2/1")
        angle, velocity = state[..., 0], state[..., 1]
        inertia = self.mass * self.length**2
        acceleration = (
            control[..., 0] / inertia
            - self.damping * velocity
            - (self.gravity / self.length) * jnp.sin(angle)
        )
        return jnp.stack((velocity, acceleration), axis=-1)

    def sample_step(self, key, state, control, delta_t):
        state = jnp.asarray(state)
        control = jnp.asarray(control, dtype=state.dtype)
        dt = jnp.asarray(delta_t, dtype=state.dtype)
        if not isinstance(dt, jax.core.Tracer) and np.any(np.asarray(dt) <= 0):
            raise ValueError("delta_t must be strictly positive")
        sub_dt = dt / self.integration_substeps
        diffusion = jnp.asarray(
            (self.angle_diffusion, self.velocity_diffusion), dtype=state.dtype
        )
        noise_key, mixture_key = jr.split(key)
        noises = jr.normal(
            noise_key,
            (self.integration_substeps,) + state.shape,
            dtype=state.dtype,
        )
        if self.noise_model == "gmm":
            # One symmetric mixture label per transition is shared over the
            # integration substeps. After summation, the standardized noise is
            # 0.5 N(-m, (1-m^2)I) + 0.5 N(m, (1-m^2)I), preserving zero mean
            # and unit marginal variance while making transitions bimodal.
            label_shape = state.shape[:-1] + (1,)
            labels = 2.0 * jr.bernoulli(mixture_key, shape=label_shape) - 1.0
            separation = jnp.asarray(self.gmm_separation, dtype=state.dtype)
            component_std = jnp.sqrt(1.0 - separation**2)
            mode_per_substep = labels * separation / jnp.sqrt(self.integration_substeps)
            noises = component_std * noises + mode_per_substep

        def substep(current, noise):
            following = (
                current
                + self.drift(current, control) * sub_dt
                + diffusion * jnp.sqrt(sub_dt) * noise
            )
            return following, None

        following, _ = jax.lax.scan(substep, state, noises)
        return following

    def sample_rollout(self, key, initial, controls, delta_t):
        controls = jnp.asarray(controls)
        if controls.ndim < 2 or controls.shape[-1] != self.control_dim:
            raise ValueError("controls must have shape (..., H, 1)")
        horizon = controls.shape[-2]
        gaps = jnp.asarray(delta_t)
        if gaps.ndim == 0:
            gaps = jnp.broadcast_to(gaps, (horizon,))
        if gaps.shape != (horizon,):
            raise ValueError(f"delta_t must be scalar or have shape {(horizon,)}")
        keys = jr.split(key, horizon)

        def step(state, inputs):
            command, gap, step_key = inputs
            following = self.sample_step(step_key, state, command, gap)
            return following, following

        _, states = jax.lax.scan(step, initial, (controls, gaps, keys))
        return states


def sample_controlled_duffing_data(
    key,
    process: ControlledDuffingOscillator,
    num_trajectories,
    num_steps,
    *,
    delta_t=0.05,
    control_scale=1.5,
    control_correlation=0.85,
    initial_std=(0.8, 0.5),
):
    """Generate reproducible trajectories and smooth bounded commands."""
    if num_trajectories < 1 or num_steps < 1 or delta_t <= 0 or control_scale <= 0:
        raise ValueError("trajectory counts, steps, delta_t, and scale must be positive")
    if not 0 <= control_correlation < 1:
        raise ValueError("control_correlation must lie in [0,1)")
    initial_key, control_key, rollout_key = jr.split(key, 3)
    initial_scale = jnp.asarray(initial_std)
    if initial_scale.shape != (process.state_dim,) or jnp.any(initial_scale <= 0):
        raise ValueError("initial_std must contain two positive values")
    initial = initial_scale * jr.normal(
        initial_key, (num_trajectories, process.state_dim)
    )
    innovations = jr.normal(
        control_key, (num_steps, num_trajectories, process.control_dim)
    )

    def control_step(previous, innovation):
        raw = (
            control_correlation * previous
            + jnp.sqrt(1.0 - control_correlation**2) * innovation
        )
        command = control_scale * jnp.tanh(raw)
        return raw, command

    _, control_time_major = jax.lax.scan(
        control_step,
        jnp.zeros((num_trajectories, process.control_dim)),
        innovations,
    )
    controls = jnp.swapaxes(control_time_major, 0, 1)
    trajectory_keys = jr.split(rollout_key, num_trajectories)
    following = jax.vmap(process.sample_rollout, in_axes=(0, 0, 0, None))(
        trajectory_keys, initial, controls, delta_t
    )
    states = jnp.concatenate((initial[:, None], following), axis=1)
    times = jnp.broadcast_to(
        jnp.arange(num_steps + 1, dtype=states.dtype) * delta_t,
        (num_trajectories, num_steps + 1),
    )
    return {
        "states": states,
        "controls": controls,
        "times": times,
    }


def sample_controlled_pendulum_data(
    key,
    process: ControlledStochasticPendulum,
    num_trajectories,
    num_steps,
    *,
    delta_t=0.05,
    control_scale=3.0,
    control_correlation=0.9,
    initial_velocity_std=0.6,
):
    """Generate pendulum trajectories with smooth bounded torque commands."""
    if num_trajectories < 1 or num_steps < 1 or delta_t <= 0:
        raise ValueError("trajectory counts, steps, and delta_t must be positive")
    if control_scale <= 0 or initial_velocity_std <= 0:
        raise ValueError("control and initial velocity scales must be positive")
    if not 0 <= control_correlation < 1:
        raise ValueError("control_correlation must lie in [0,1)")
    angle_key, velocity_key, control_key, rollout_key = jr.split(key, 4)
    angle = jr.uniform(
        angle_key, (num_trajectories, 1), minval=-jnp.pi, maxval=jnp.pi
    )
    velocity = initial_velocity_std * jr.normal(
        velocity_key, (num_trajectories, 1)
    )
    initial = jnp.concatenate((angle, velocity), axis=-1)
    innovations = jr.normal(
        control_key, (num_steps, num_trajectories, process.control_dim)
    )

    def control_step(previous, innovation):
        raw = (
            control_correlation * previous
            + jnp.sqrt(1.0 - control_correlation**2) * innovation
        )
        return raw, control_scale * jnp.tanh(raw)

    _, control_time_major = jax.lax.scan(
        control_step,
        jnp.zeros((num_trajectories, process.control_dim)),
        innovations,
    )
    controls = jnp.swapaxes(control_time_major, 0, 1)
    trajectory_keys = jr.split(rollout_key, num_trajectories)
    following = jax.vmap(process.sample_rollout, in_axes=(0, 0, 0, None))(
        trajectory_keys, initial, controls, delta_t
    )
    states = jnp.concatenate((initial[:, None], following), axis=1)
    times = jnp.broadcast_to(
        jnp.arange(num_steps + 1, dtype=states.dtype) * delta_t,
        (num_trajectories, num_steps + 1),
    )
    return {"states": states, "controls": controls, "times": times}


__all__ = (
    "ControlledDuffingOscillator",
    "ControlledStochasticPendulum",
    "sample_controlled_duffing_data",
    "sample_controlled_pendulum_data",
)
