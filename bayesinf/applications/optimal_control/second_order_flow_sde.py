"""Velocity-augmented, structured second-order controlled RealNVP FlowSDE."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from .scoring import energy_score, squared_mean_error

from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPDiffusionNet,
    SinusoidalEmb,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    StepNoisePath,
    get_solver,
)


class AccelerationNet(eqx.Module):
    net: eqx.nn.MLP
    time_emb: SinusoidalEmb

    def __init__(self, position_dim, control_dim, hidden_size, depth,
                 time_emb_dim, key):
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            2 * position_dim + control_dim + time_emb_dim, position_dim,
            hidden_size, depth, activation=jax.nn.gelu, key=key,
        )

    def __call__(self, state, control, time):
        return self.net(jnp.concatenate([state, control, self.time_emb(time)]))


class SecondOrderControlledRealNVPFlowSDE(eqx.Module):
    """Fixed-grid kinetic transition with transformed velocity innovations.

    Integrates dq=v dt and dv=a dt+g(eps)*sqrt(dt) as a discrete model.
    Nonlinear g is neither assumed centered nor a Brownian coefficient.
    """

    acceleration: AccelerationNet
    diffusion: RealNVPDiffusionNet
    position_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    diffusion_condition_on_control: bool = eqx.field(static=True, default=True)
    velocity_to_position_scale: jnp.ndarray
    velocity_to_position_bias: jnp.ndarray

    def __init__(self, position_dim, control_dim, key, *, hidden_size=64,
                 depth=2, diffusion_hidden=64, flow_layers=4,
                 time_emb_dim=16, realnvp_alpha=3.0, dt=0.02,
                 velocity_to_position_scale=None,
                 velocity_to_position_bias=None,
                 diffusion_condition_on_control=True):
        self.position_dim = position_dim
        self.state_dim = 2 * position_dim
        self.control_dim = control_dim
        self.dt = float(dt)
        self.diffusion_condition_on_control = diffusion_condition_on_control
        if velocity_to_position_scale is None:
            velocity_to_position_scale = jnp.ones(position_dim)
        if velocity_to_position_bias is None:
            velocity_to_position_bias = jnp.zeros(position_dim)
        self.velocity_to_position_scale = jnp.asarray(velocity_to_position_scale)
        self.velocity_to_position_bias = jnp.asarray(velocity_to_position_bias)
        acceleration_key, diffusion_key = jr.split(key)
        self.acceleration = AccelerationNet(
            position_dim, control_dim, hidden_size, depth,
            time_emb_dim, acceleration_key,
        )
        # q is passed as z_state and [v,u] as context:
        # g(epsilon; q, v, u, t).
        self.diffusion = RealNVPDiffusionNet(
            position_dim,
            position_dim + (control_dim if diffusion_condition_on_control else 0),
            diffusion_hidden, flow_layers,
            time_emb_dim, diffusion_key, alpha=realnvp_alpha,
        )

    def _diffusion_context(self, velocity, control):
        if self.diffusion_condition_on_control:
            return jnp.concatenate([velocity, control])
        return velocity

    def drift(self, state, control, time, state_history=None):
        q_rate = self.velocity_to_position_scale * state[self.position_dim:] + self.velocity_to_position_bias
        return jnp.concatenate((q_rate, self.acceleration(state, control, time)))

    def sequence_histories(self, states):
        return None

    def transition_nll(self, z0, u0, z1, time, state_history=None):
        """Velocity-only Euler NLL; positions have no full-dimensional density."""
        dt = jnp.asarray(self.dt, dtype=z0.dtype)
        d = self.position_dim

        def one(a, control, b, t):
            q0, v0 = a[:d], a[d:]
            v1 = b[d:]
            residual = (v1 - v0 - self.acceleration(a, control, t) * dt) / jnp.sqrt(dt)
            context = self._diffusion_context(v0, control)
            centre = self.diffusion.apply(jnp.zeros(d), q0, context, t)
            epsilon, log_det = self.diffusion.inverse_and_log_det(
                residual + centre, q0, context, t
            )
            base = 0.5 * jnp.sum(epsilon ** 2) + 0.5 * d * jnp.log(2 * jnp.pi)
            return (base + log_det + 0.5 * d * jnp.log(dt)) / d

        return jnp.mean(jax.vmap(one)(
            z0.reshape(-1, self.state_dim), u0.reshape(-1, self.control_dim),
            z1.reshape(-1, self.state_dim), time.reshape(-1),
        ))

    def noisy_transition_nll(self, z0, u0, z1, time, key,
                             state_noise_std=0.0, control_noise_std=0.0, state_history=None):
        kx, ku = jr.split(key)
        noisy_z = z0 + state_noise_std * jr.normal(kx, z0.shape)
        noisy_u = u0 + control_noise_std * jr.normal(ku, u0.shape)
        return self.transition_nll(noisy_z, noisy_u, z1, time)

    def deterministic_rollout(self, z0, controls, start_step=0, solver_name="heun"):
        """The same integration scheme as sampling, with innovations disabled."""
        return self._sample_path(z0, controls, jr.PRNGKey(0), start_step,
                                 diffusion_temperature=0.0, solver_name=solver_name)

    def _sample_path(self, z0, controls, key, start_step=0,
                     diffusion_temperature=1.0, solver_name="heun"):
        dt = jnp.asarray(self.dt, dtype=z0.dtype)
        d = self.position_dim
        noises = jr.normal(key, (len(controls), d))
        noise_path = StepNoisePath(noises, start_step * dt, dt)
        if solver_name in ("milstein", "adaptive_euler", "adaptive_milstein"):
            raise ValueError("nonlinear flow innovations require a fixed-grid transition solver")
        solver = get_solver(solver_name)

        def step(z, k):
            t0 = (start_step + k) * dt
            control = controls[k]

            def drift_fn(t, value):
                q_rate = (self.velocity_to_position_scale * value[d:]
                          + self.velocity_to_position_bias)
                return jnp.concatenate([q_rate, self.acceleration(value, control, t)])

            def diffusion_fn(t, value, epsilon):
                q, v = value[:d], value[d:]
                context = self._diffusion_context(v, control)
                noise = (self.diffusion.apply(epsilon * diffusion_temperature, q, context, t)
                         - self.diffusion.apply(jnp.zeros_like(epsilon), q, context, t))
                return jnp.concatenate([jnp.zeros(d, dtype=value.dtype), noise])

            z1 = solver.step(z, t0, t0 + dt, drift_fn, diffusion_fn, noise_path)
            return z1, z1
        _, path = jax.lax.scan(step, z0, jnp.arange(len(controls)))
        return path

    def sample_rollout(self, z0, controls, key, n_particles=64, start_step=0,
                       diffusion_temperature=1.0, solver_name="heun", state_history=None,
                       rollout_mode="normflowsde", ito_moment_samples=64, ito_jitter=1e-6):
        if rollout_mode != "normflowsde":
            raise ValueError("second-order flow supports fixed-step normflowsde transitions")
        return jax.vmap(lambda particle_key: self._sample_path(
            z0, controls, particle_key, start_step,
            diffusion_temperature, solver_name,
        ))(jr.split(key, n_particles))

    def multiple_shooting_loss(self, states, controls, start_steps, key, *,
                               segment_length=16, n_particles=4,
                               solver_name="heun", shooting_noise_std=0.01,
                               control_noise_std=0.01,
                               path_weight=1.0,
                               uncertainty_weight=0.5,
                               continuity_weight=0.1,
                               state_loss_weights=None):
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments, d = window_steps // segment_length, self.position_dim
        batch_keys = jr.split(key, len(states))

        def trajectory_loss(z, u, first_step, trajectory_key):
            control_key, segment_root = jr.split(trajectory_key)
            if control_noise_std > 0:
                u = u + control_noise_std * jr.normal(control_key, u.shape)
            starts = z[:-1].reshape(n_segments, segment_length, self.state_dim)[:, 0]
            targets = z[1:].reshape(n_segments, segment_length, self.state_dim)
            u_segments = u.reshape(n_segments, segment_length, self.control_dim)
            times = first_step + jnp.arange(n_segments) * segment_length
            keys = jr.split(segment_root, n_segments)

            def segment_loss(z0, us, target, segment_start, segment_key):
                init_key, path_key = jr.split(segment_key)
                if shooting_noise_std > 0:
                    z0 = z0 + shooting_noise_std * jr.normal(init_key, z0.shape)
                paths = self.sample_rollout(
                    z0, us, path_key, n_particles, segment_start, 1.0, solver_name
                )
                weights = (jnp.concatenate([jnp.ones(d), .01 * jnp.ones(d)])
                           if state_loss_weights is None else jnp.asarray(state_loss_weights))
                score = jnp.mean(energy_score(paths, target, weights=weights,
                                              spread_weight=uncertainty_weight))
                continuity = squared_mean_error(paths[:, -1], target[-1], weights=weights)
                return score, continuity

            scores, continuity = jax.vmap(segment_loss)(
                starts, u_segments, targets, times, keys
            )
            return (path_weight * jnp.mean(scores)
                    + continuity_weight * jnp.mean(continuity))

        return jnp.mean(jax.vmap(trajectory_loss)(
            states, controls, start_steps, batch_keys
        ))
