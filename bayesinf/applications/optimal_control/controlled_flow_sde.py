"""Control-conditioned RealNVP FlowSDE for forward dynamics.

The normalized-state, fixed-grid transition model is

    x[k+1] - x[k] = f(x[k], u[k], t[k]) dt + G(epsilon; x[k], u[k], t[k]) sqrt(dt),
    epsilon ~ Normal(0, I).

Nonlinear transformed noise need not be Gaussian or mean zero. This kernel
does not claim an Itô limit under grid refinement. The optional ``itosde``
route moment-matches at the configured dt and uses Euler-Maruyama.

Unlike the state-estimation FlowSDE, this model is causal: the conditioning
variable is the command at the current step, rather than a bidirectional
encoding of an observation trajectory.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from collections.abc import Callable

from .scoring import energy_score, squared_mean_error

from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPDiffusionNet,
    SDEDriftNet,
    SinusoidalEmb,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    StepNoisePath,
    get_solver,
)
ARM_INDICES = jnp.asarray((0, 1, 2, 6, 7, 8))
OBJECT_INDICES = jnp.asarray((3, 4, 5, 9, 10, 11))


def _mean_covariance(values, jitter):
    """Shared finite-sample moment convention for rollout and distillation."""
    mean = jnp.mean(values, axis=0)
    centered = values - mean
    covariance = centered.T @ centered / (values.shape[0] - 1)
    return mean, covariance + jitter * jnp.eye(values.shape[-1], dtype=values.dtype)


class DistilledItoSDE(eqx.Module):
    """Compact MLP Itô surrogate distilled from a controlled NormFlowSDE."""

    drift_net: SDEDriftNet
    factor_net: eqx.nn.MLP
    time_emb: SinusoidalEmb
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    diag_floor: float = eqx.field(static=True)
    max_factor_norm: float = eqx.field(static=True)

    def __init__(self, state_dim, control_dim, key, *, hidden_size=64, depth=2,
                 time_emb_dim=16, dt=0.02, diag_floor=1e-5,
                 max_factor_norm=5.0):
        if dt <= 0:
            raise ValueError("dt must be positive")
        if diag_floor <= 0:
            raise ValueError("diag_floor must be positive")
        if max_factor_norm <= 0:
            raise ValueError("max_factor_norm must be positive")
        drift_key, factor_key = jr.split(key)
        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.dt = float(dt)
        self.diag_floor = float(diag_floor)
        self.max_factor_norm = float(max_factor_norm)
        self.drift_net = SDEDriftNet(
            state_dim, control_dim, hidden_size, depth, time_emb_dim, drift_key
        )
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.factor_net = eqx.nn.MLP(
            state_dim + control_dim + time_emb_dim,
            state_dim * state_dim,
            hidden_size,
            depth,
            activation=jax.nn.gelu,
            key=factor_key,
        )

    def drift(self, state, control, time):
        return self.drift_net(state, control, time)

    def diffusion_factor(self, state, control, time):
        raw = self.factor_net(
            jnp.concatenate([state, control, self.time_emb(time)])
        ).reshape(self.state_dim, self.state_dim)
        lower = jnp.tril(raw)
        diag = jax.nn.softplus(jnp.diag(lower)) + self.diag_floor
        factor = lower.at[jnp.diag_indices(self.state_dim)].set(diag)
        # Local-moment students are queried recursively by MPC. Bound the
        # factor outside their tiny calibration support to prevent covariance
        # extrapolation from producing numerically explosive rollouts.
        norm = jnp.linalg.norm(factor)
        return factor / jnp.maximum(1.0, norm / self.max_factor_norm)

    def covariance(self, state, control, time):
        factor = self.diffusion_factor(state, control, time)
        return factor @ factor.T

    def sample_rollout(self, x0, controls, key, n_particles=64, start_step=0):
        dt = jnp.asarray(self.dt, dtype=x0.dtype)
        keys = jr.split(key, n_particles)

        def one_path(path_key):
            noises = jr.normal(path_key, (len(controls), self.state_dim))

            def step(state, item):
                k, control, eps = item
                time = (start_step + k) * dt
                drift = self.drift(state, control, time)
                factor = self.diffusion_factor(state, control, time)
                next_state = state + drift * dt + factor @ eps * jnp.sqrt(dt)
                return next_state, next_state

            _, path = jax.lax.scan(
                step, x0, (jnp.arange(len(controls)), controls, noises)
            )
            return path

        return jax.vmap(one_path)(keys)


class SplitRealNVPDiffusionNet(eqx.Module):
    """Independent invertible noise heads for arm and object coordinates."""

    arm: RealNVPDiffusionNet
    object: RealNVPDiffusionNet
    feature_dim: int = eqx.field(static=True)
    history_dim: int = eqx.field(static=True)
    d_x: int = eqx.field(static=True, default=12)

    def __init__(self, control_dim, feature_dim, history_dim, hidden, layers,
                 time_dim, key, alpha):
        arm_key, object_key = jr.split(key)
        arm_context_dim = control_dim + 6 + feature_dim
        object_context_dim = arm_context_dim + history_dim
        self.arm = RealNVPDiffusionNet(
            6, arm_context_dim, hidden, layers, time_dim, arm_key, alpha=alpha
        )
        self.object = RealNVPDiffusionNet(
            6, object_context_dim, hidden, layers, time_dim, object_key, alpha=alpha
        )
        self.feature_dim = feature_dim
        self.history_dim = history_dim

    def _parts(self, state, context):
        base_end = context.shape[0] - self.history_dim
        base = context[:base_end]
        history = context[base_end:]
        control = base[:-self.feature_dim] if self.feature_dim else base
        features = base[-self.feature_dim:] if self.feature_dim else base[:0]
        arm_state, object_state = state[ARM_INDICES], state[OBJECT_INDICES]
        return (arm_state, object_state,
                jnp.concatenate((control, object_state, features)),
                jnp.concatenate((control, arm_state, features, history)))

    @staticmethod
    def _merge(arm, object):
        out = jnp.zeros((12,), dtype=arm.dtype)
        return out.at[ARM_INDICES].set(arm).at[OBJECT_INDICES].set(object)

    def apply(self, eps, state, context, time):
        arm_state, object_state, arm_context, object_context = self._parts(
            state, context
        )
        return self._merge(
            self.arm.apply(eps[ARM_INDICES], arm_state, arm_context, time),
            self.object.apply(
                eps[OBJECT_INDICES], object_state, object_context, time
            ),
        )

    def inverse_and_log_det(self, value, state, context, time):
        arm_state, object_state, arm_context, object_context = self._parts(
            state, context
        )
        arm, arm_logdet = self.arm.inverse_and_log_det(
            value[ARM_INDICES], arm_state, arm_context, time
        )
        obj, obj_logdet = self.object.inverse_and_log_det(
            value[OBJECT_INDICES], object_state, object_context, time
        )
        return self._merge(arm, obj), arm_logdet + obj_logdet


class ControlledRealNVPFlowSDE(eqx.Module):
    """Euler--Maruyama controlled FlowSDE with exact transition density."""

    drift_full: SDEDriftNet | None
    drift_arm: SDEDriftNet | None
    drift_object: SDEDriftNet | None
    drift_object_contact: SDEDriftNet | None
    diffusion: RealNVPDiffusionNet | SplitRealNVPDiffusionNet
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    diffusion_condition_on_control: bool = eqx.field(static=True)
    diffusion_condition_on_geometry: bool = eqx.field(static=True)
    object_drift_no_control: bool = eqx.field(static=True)
    separate_dynamics_heads: bool = eqx.field(static=True)
    coupled_six_state_flows: bool = eqx.field(static=True)
    use_contact_geometry: bool = eqx.field(static=True)
    geometry_feature_fn: Callable | None = eqx.field(static=True)
    geometry_temperature: float = eqx.field(static=True)
    object_history_steps: int = eqx.field(static=True)
    kinematic_position_drift: bool = eqx.field(static=True)
    geometry_feature_dim: int = eqx.field(static=True)
    contact_gated_object_drift: bool = eqx.field(static=True)
    state_mean: jax.Array
    state_std: jax.Array
    geometry_mean: jax.Array
    geometry_std: jax.Array

    def __init__(self, state_dim, control_dim, key, *, hidden_size=64, depth=2,
                 diffusion_hidden=64, flow_layers=4, time_emb_dim=16,
                 realnvp_alpha=3.0, dt=0.02,
                 diffusion_condition_on_control=True,
                 diffusion_condition_on_geometry=True,
                 object_drift_no_control=False, separate_dynamics_heads=False,
                 coupled_six_state_flows=False,
                 use_contact_geometry=False, geometry_temperature=0.02,
                 object_history_steps=0,
                 kinematic_position_drift=False,
                 geometry_feature_dim=5,
                 contact_gated_object_drift=False,
                 geometry_feature_fn=None,
                 state_mean=None, state_std=None, geometry_mean=None,
                 geometry_std=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = float(dt)
        self.diffusion_condition_on_control = bool(diffusion_condition_on_control)
        self.diffusion_condition_on_geometry = bool(
            diffusion_condition_on_geometry
        )
        self.object_drift_no_control = bool(object_drift_no_control)
        self.separate_dynamics_heads = bool(separate_dynamics_heads)
        self.coupled_six_state_flows = bool(coupled_six_state_flows)
        self.use_contact_geometry = bool(use_contact_geometry)
        self.geometry_feature_fn = geometry_feature_fn
        self.geometry_temperature = float(geometry_temperature)
        self.object_history_steps = int(object_history_steps)
        self.kinematic_position_drift = bool(kinematic_position_drift)
        self.geometry_feature_dim = int(geometry_feature_dim)
        self.contact_gated_object_drift = bool(contact_gated_object_drift)
        if self.geometry_feature_dim not in (5, 9):
            raise ValueError("geometry_feature_dim must be 5 or 9")
        if self.contact_gated_object_drift and not self.use_contact_geometry:
            raise ValueError("contact-gated object drift requires contact geometry")
        if self.object_history_steps < 0:
            raise ValueError("object_history_steps must be non-negative")
        if self.object_history_steps and not self.separate_dynamics_heads:
            raise ValueError(
                "object history requires separate_dynamics_heads"
            )
        if self.kinematic_position_drift and state_dim != 12:
            raise ValueError("kinematic position drift requires a 12-D [q,v] state")
        self.state_mean = jnp.zeros(state_dim) if state_mean is None else jnp.asarray(state_mean)
        self.state_std = jnp.ones(state_dim) if state_std is None else jnp.asarray(state_std)
        self.geometry_mean = (jnp.zeros(self.geometry_feature_dim) if geometry_mean is None
                              else jnp.asarray(geometry_mean))
        self.geometry_std = (jnp.ones(self.geometry_feature_dim) if geometry_std is None
                             else jnp.asarray(geometry_std))
        if (self.separate_dynamics_heads or self.coupled_six_state_flows
                or self.use_contact_geometry) and state_dim != 12:
            raise ValueError("split heads/contact geometry require the 12-D pusher state")
        if self.coupled_six_state_flows and self.use_contact_geometry:
            raise ValueError("coupled six-state flows use explicit cross-state coupling")
        if self.use_contact_geometry and self.geometry_feature_fn is None:
            raise ValueError("use_contact_geometry requires geometry_feature_fn")
        drift_key, diffusion_key = jr.split(key)
        feature_dim = self.geometry_feature_dim if self.use_contact_geometry else 0
        history_dim = self.object_history_steps * state_dim
        if self.coupled_six_state_flows:
            arm_key, object_key = jr.split(drift_key)
            self.drift_full = None
            # Robot flow: r_t plus (u_t, o_t). Object flow: o_t plus
            # (u_t, r_t, predicted robot drift). Both remain jointly trained.
            self.drift_arm = SDEDriftNet(
                6, control_dim + 6, hidden_size, depth, time_emb_dim,
                arm_key, output_dim=6,
            )
            self.drift_object = SDEDriftNet(
                6, control_dim + 12, hidden_size, depth, time_emb_dim,
                object_key, output_dim=6,
            )
            self.drift_object_contact = None
        elif self.object_drift_no_control or self.separate_dynamics_heads:
            if state_dim != 12:
                raise ValueError(
                    "object_drift_no_control requires the 12-D pusher state"
                )
            arm_key, object_key, contact_key = jr.split(drift_key, 3)
            self.drift_full = None
            self.drift_arm = SDEDriftNet(
                state_dim + feature_dim, control_dim, hidden_size, depth, time_emb_dim,
                arm_key, output_dim=6,
            )
            self.drift_object = SDEDriftNet(
                state_dim + feature_dim + history_dim,
                0 if self.object_drift_no_control else control_dim,
                hidden_size, depth, time_emb_dim,
                object_key, output_dim=6,
            )
            self.drift_object_contact = (
                SDEDriftNet(
                    state_dim + feature_dim + history_dim,
                    0 if self.object_drift_no_control else control_dim,
                    hidden_size, depth, time_emb_dim,
                    contact_key, output_dim=6,
                ) if self.contact_gated_object_drift else None
            )
        else:
            self.drift_full = SDEDriftNet(
                state_dim + feature_dim, control_dim, hidden_size, depth, time_emb_dim,
                drift_key, output_dim=state_dim,
            )
            self.drift_arm = None
            self.drift_object = None
            self.drift_object_contact = None
        diffusion_control_dim = control_dim if self.diffusion_condition_on_control else 0
        diffusion_feature_dim = (
            feature_dim if self.diffusion_condition_on_geometry else 0
        )
        if self.separate_dynamics_heads or self.coupled_six_state_flows:
            self.diffusion = SplitRealNVPDiffusionNet(
                diffusion_control_dim, diffusion_feature_dim, history_dim,
                diffusion_hidden,
                flow_layers, time_emb_dim, diffusion_key, realnvp_alpha,
            )
        else:
            self.diffusion = RealNVPDiffusionNet(
                state_dim, diffusion_control_dim + diffusion_feature_dim,
                diffusion_hidden, flow_layers,
                time_emb_dim, diffusion_key, alpha=realnvp_alpha,
            )

    def geometry_features(self, state):
        physical = state * self.state_std + self.state_mean
        raw = self.geometry_feature_fn(
            physical, temperature=self.geometry_temperature
        )[:self.geometry_feature_dim]
        return (raw - self.geometry_mean) / self.geometry_std

    def _augmented_state(self, state):
        if not self.use_contact_geometry:
            return state
        return jnp.concatenate((state, self.geometry_features(state)))

    def _history_context(self, state, state_history=None):
        if self.object_history_steps == 0:
            return state[:0]
        if state_history is None:
            history = jnp.broadcast_to(
                state, (self.object_history_steps, self.state_dim)
            )
        else:
            history = jnp.asarray(state_history)[-self.object_history_steps:]
        return history.reshape(-1)

    def drift(self, state, control, time, state_history=None):
        result = self._learned_drift(state, control, time, state_history)
        if self.kinematic_position_drift:
            q_rate = (self.state_mean[6:] + self.state_std[6:] * state[6:]) / self.state_std[:6]
            result = result.at[:6].set(q_rate)
        return result

    def _learned_drift(self, state, control, time, state_history=None):
        augmented = self._augmented_state(state)
        if self.coupled_six_state_flows:
            robot_state = state[ARM_INDICES]
            object_state = state[OBJECT_INDICES]
            robot = self.drift_arm(
                robot_state, jnp.concatenate((control, object_state)), time
            )
            obj = self.drift_object(
                object_state,
                jnp.concatenate((control, robot_state, robot)), time,
            )
            return jnp.zeros_like(state).at[ARM_INDICES].set(robot).at[
                OBJECT_INDICES
            ].set(obj)
        if not (self.object_drift_no_control or self.separate_dynamics_heads):
            return self.drift_full(augmented, control, time)
        arm = self.drift_arm(augmented, control, time)
        object_control = (jnp.empty((0,), dtype=control.dtype)
                          if self.object_drift_no_control else control)
        object_input = jnp.concatenate(
            (augmented, self._history_context(state, state_history))
        )
        obj = self.drift_object(object_input, object_control, time)
        if self.drift_object_contact is not None:
            contact_obj = self.drift_object_contact(
                object_input, object_control, time
            )
            # Proximity is feature 2 in both basic and extended geometry.
            proximity = jnp.clip(
                self.geometry_features(state)[2] * self.geometry_std[2]
                + self.geometry_mean[2], 0.0, 1.0
            )
            obj = (1.0 - proximity) * obj + proximity * contact_obj
        # State order is [q_arm, q_object, v_arm, v_object].
        result = jnp.concatenate([arm[:3], obj[:3], arm[3:], obj[3:]])
        return result

    def _diffusion_context(self, control):
        context = (control if self.diffusion_condition_on_control
                   else jnp.empty((0,), dtype=control.dtype))
        return context

    def _diffusion_context_for_state(self, state, control, state_history=None):
        context = self._diffusion_context(control)
        if self.use_contact_geometry and self.diffusion_condition_on_geometry:
            context = jnp.concatenate((context, self.geometry_features(state)))
        if self.separate_dynamics_heads:
            context = jnp.concatenate(
                (context, self._history_context(state, state_history))
            )
        return context

    def transition_log_prob(self, x1, x0, u0, time, state_history=None):
        """Exact normalized-coordinate Euler transition log density."""
        x0, x1 = jnp.asarray(x0), jnp.asarray(x1)
        if x0.shape != x1.shape or x0.shape[-1:] != (self.state_dim,):
            raise ValueError("x0 and x1 must have matching final state dimensions")
        batch_shape = x0.shape[:-1]
        u0 = jnp.broadcast_to(
            jnp.asarray(u0, dtype=x0.dtype), batch_shape + (self.control_dim,)
        )
        time = jnp.broadcast_to(jnp.asarray(time, dtype=x0.dtype), batch_shape or ())
        dt = jnp.asarray(self.dt, dtype=x0.dtype)
        if state_history is None:
            history_steps = max(1, self.object_history_steps)
            state_history = jnp.broadcast_to(
                x0[..., None, :], x0.shape[:-1] +
                (history_steps, self.state_dim)
            )
        def one(a, control, b, t, history):
            residual = (b - a - self.drift(a, control, t, history) * dt) / jnp.sqrt(dt)
            diffusion_context = self._diffusion_context_for_state(a, control, history)
            # Anchor innovations at epsilon=0. This is a nominal trajectory,
            # not mean centering: nonlinear flows can still have nonzero means.
            centre = self.diffusion.apply(
                jnp.zeros_like(residual), a, diffusion_context, t
            )
            epsilon, forward_log_det = self.diffusion.inverse_and_log_det(
                residual + centre, a, diffusion_context, t
            )
            base = 0.5 * jnp.sum(epsilon ** 2)
            base += 0.5 * self.state_dim * jnp.log(2.0 * jnp.pi)
            # p(x1|x0,u) includes both the flow and sqrt(dt) Jacobians.
            return -(base + forward_log_det
                     + 0.5 * self.state_dim * jnp.log(dt))

        flat_shape = (-1, self.state_dim)
        values = jax.vmap(one)(
            x0.reshape(flat_shape), u0.reshape((-1, self.control_dim)),
            x1.reshape(flat_shape), time.reshape((-1,)),
            state_history.reshape((-1, state_history.shape[-2], self.state_dim)),
        )
        return values.reshape(batch_shape) if batch_shape else values[0]

    def transition_nll(self, x0, u0, x1, time, state_history=None):
        """Mean per-state-dimension exact Euler transition NLL."""
        return -jnp.mean(
            self.transition_log_prob(x1, x0, u0, time, state_history)
        ) / self.state_dim

    def noisy_transition_nll(self, x0, u0, x1, time, key,
                             state_noise_std=0.0, control_noise_std=0.0,
                             state_history=None):
        """Denoising likelihood: perturb inputs while keeping labels clean."""
        kx, ku = jr.split(key)
        noisy_x = x0 + state_noise_std * jr.normal(kx, x0.shape)
        noisy_u = u0 + control_noise_std * jr.normal(ku, u0.shape)
        return self.transition_nll(noisy_x, noisy_u, x1, time, state_history)

    def deterministic_rollout(self, x0, controls, start_step=0,
                              state_history=None):
        """Roll out the learned drift (diffusion disabled)."""
        dt = jnp.asarray(self.dt, dtype=x0.dtype)

        history = self._initial_history(x0, state_history)
        def step(carry, item):
            x, history = carry
            k, u = item
            t = (start_step + k) * dt
            next_x = x + self.drift(x, u, t, history) * dt
            next_history = self._advance_history(history, x)
            return (next_x, next_history), next_x

        _, path = jax.lax.scan(
            step, (x0, history), (jnp.arange(len(controls)), controls)
        )
        return path

    def _initial_history(self, state, state_history=None):
        steps = max(1, self.object_history_steps)
        if state_history is None or state_history.shape[-2] == 0:
            return jnp.broadcast_to(state, (steps, self.state_dim))
        supplied = jnp.asarray(state_history)
        supplied = supplied[-steps:]
        pad = jnp.broadcast_to(supplied[0], (steps - supplied.shape[0], self.state_dim))
        return jnp.concatenate((pad, supplied), axis=0)

    def _advance_history(self, history, state):
        return jnp.concatenate((history[1:], state[None]), axis=0)

    def sequence_histories(self, states):
        """Prior-state windows aligned with ``states[..., :-1, :]``."""
        if self.object_history_steps == 0:
            return None
        steps = self.object_history_steps
        width = states.shape[-2] - 1
        pad = jnp.repeat(states[..., :1, :], steps, axis=-2)
        source = jnp.concatenate((pad, states[..., :-1, :]), axis=-2)
        return jnp.stack(
            [source[..., offset:offset + width, :]
             for offset in range(steps)],
            axis=-2,
        )

    def _centered_diffusion_values(
        self, eps_values, state, control, time, state_history=None,
        diffusion_temperature=1.0,
    ):
        context = self._diffusion_context_for_state(
            state, control, state_history
        )
        zero = jnp.zeros((self.state_dim,), dtype=state.dtype)
        centre = self.diffusion.apply(zero, state, context, time)
        return jax.vmap(
            lambda eps: self.diffusion.apply(
                eps * diffusion_temperature, state, context, time
            ) - centre
        )(eps_values)

    def _ito_flow_moments(
        self, state, control, time, moment_eps, state_history=None,
        diffusion_temperature=1.0, jitter=1e-6,
    ):
        values = self._centered_diffusion_values(
            moment_eps, state, control, time, state_history,
            diffusion_temperature,
        )
        mean, covariance = _mean_covariance(values, jitter)
        factor = jnp.linalg.cholesky(covariance)
        return mean, factor

    def ito_moments(self, state, control, time, key, *, samples=64,
                    state_history=None, diffusion_temperature=1.0,
                    jitter=1e-6):
        """Match transition moments at this dt, including G(epsilon)-G(0)'s mean.

        The resulting Itô coefficients depend on dt; they are not a claim
        that the nonlinear-noise kernel has a resolution-independent SDE limit.
        """
        if samples < 2:
            raise ValueError("Itô moment estimation requires at least two samples")
        eps = jr.normal(key, (samples, self.state_dim), dtype=state.dtype)
        mean, factor = self._ito_flow_moments(
            state, control, time, eps, state_history,
            diffusion_temperature, jitter,
        )
        dt = jnp.asarray(self.dt, dtype=state.dtype)
        return self.drift(state, control, time, state_history) + mean / jnp.sqrt(dt), factor

    def _sample_path(self, x0, controls, key, start_step=0,
                     diffusion_temperature=1.0, solver_name="heun",
                     state_history=None, rollout_mode="normflowsde",
                     ito_moment_samples=64, ito_jitter=1e-6):
        """Integrate one path with the same custom solvers as state FlowSDE."""
        dt = jnp.asarray(self.dt, dtype=x0.dtype)
        if rollout_mode == "normflowsde":
            noises = jr.normal(key, (len(controls), self.state_dim))
            moment_noises = None
        elif rollout_mode == "itosde":
            noises = jr.normal(jr.fold_in(key, 17), (len(controls), self.state_dim))
            moment_noises = jr.normal(
                jr.fold_in(key, 23),
                (len(controls), ito_moment_samples, self.state_dim),
            )
        else:
            raise ValueError("rollout_mode must be 'normflowsde' or 'itosde'")
        noise_path = StepNoisePath(noises, start_step * dt, dt)
        if solver_name in ("milstein", "adaptive_euler", "adaptive_milstein"):
            raise ValueError("nonlinear flow innovations require a fixed-grid transition solver")
        solver = get_solver(solver_name)

        # The history is held fixed within one solver step and advanced only
        # after that step, so all Heun stages use the same causal context.
        def step(carry, k):
            x, history = carry
            t0 = (start_step + k) * dt
            u = controls[k]

            def drift_fn(t, value):
                if rollout_mode == "itosde":
                    mean, _ = self._ito_flow_moments(
                        value, u, t, moment_noises[k], history,
                        diffusion_temperature, ito_jitter,
                    )
                    return self.drift(value, u, t, history) + mean / jnp.sqrt(dt)
                return self.drift(value, u, t, history)

            def diffusion_fn(t, value, epsilon):
                if rollout_mode == "itosde":
                    _, factor = self._ito_flow_moments(
                        value, u, t, moment_noises[k], history,
                        diffusion_temperature, ito_jitter,
                    )
                    return factor @ epsilon
                context = self._diffusion_context_for_state(
                    value, u, history
                )
                zero = jnp.zeros_like(epsilon)
                return (self.diffusion.apply(
                            epsilon * diffusion_temperature, value, context, t
                        ) - self.diffusion.apply(zero, value, context, t))

            x1 = solver.step(x, t0, t0 + dt, drift_fn, diffusion_fn, noise_path)
            return (x1, self._advance_history(history, x)), x1

        history = self._initial_history(x0, state_history)
        _, path = jax.lax.scan(
            step, (x0, history), jnp.arange(len(controls))
        )
        return path

    def sample_rollout(self, x0, controls, key, n_particles=64, start_step=0,
                       diffusion_temperature=1.0, solver_name=None,
                       state_history=None, rollout_mode="normflowsde",
                       ito_moment_samples=64, ito_jitter=1e-6):
        """Return stochastic paths with shape (particles, time, state_dim)."""
        if solver_name is None:
            solver_name = "euler" if rollout_mode == "itosde" else "heun"
        if rollout_mode == "itosde" and solver_name not in ("euler", "euler_maruyama"):
            raise ValueError("Itô moment rollouts require Euler; Heun is a different Stratonovich law")
        if rollout_mode == "itosde" and ito_moment_samples < 2:
            raise ValueError("Itô rollout requires at least two moment samples")
        return jax.vmap(lambda particle_key: self._sample_path(
            x0, controls, particle_key, start_step,
            diffusion_temperature, solver_name, state_history,
            rollout_mode, ito_moment_samples, ito_jitter,
        ))(jr.split(key, n_particles))

    def coupled_multiple_shooting_losses(
            self, states, controls, start_steps, key, *, segment_length=10,
            n_particles=2, solver_name="heun", shooting_noise_std=0.01,
            control_noise_std=0.01, path_weight=1.0, object_path_weight=1.0,
            uncertainty_weight=0.5,
            continuity_weight=0.1, object_temporal_weights=None):
        """Robot/object energy scores from one shared stochastic rollout."""
        if self.state_dim != 12:
            raise ValueError("coupled losses require the 12-D robot/object state")
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments = window_steps // segment_length
        if object_temporal_weights is None:
            object_temporal_weights = jnp.ones(
                controls.shape[:2], dtype=states.dtype
            )
        object_temporal_weights = (
            object_temporal_weights
            / (jnp.mean(object_temporal_weights, axis=1, keepdims=True) + 1e-6)
        )
        batch_keys = jr.split(key, states.shape[0])

        def trajectory_losses(x, u, first_step, trajectory_key, object_weights):
            control_key, shooting_key = jr.split(trajectory_key)
            if control_noise_std > 0:
                u = u + control_noise_std * jr.normal(control_key, u.shape)
            state_starts = x[:-1].reshape(
                n_segments, segment_length, self.state_dim
            )[:, 0]
            targets = x[1:].reshape(
                n_segments, segment_length, self.state_dim
            )
            control_segments = u.reshape(
                n_segments, segment_length, self.control_dim
            )
            segment_starts = first_step + jnp.arange(n_segments) * segment_length
            object_weights = object_weights.reshape(n_segments, segment_length)

            def segment_losses(x0, segment_controls, target, segment_start,
                               object_time_weights, segment_key):
                init_key, path_key = jr.split(segment_key)
                if shooting_noise_std > 0:
                    x0 = x0 + shooting_noise_std * jr.normal(init_key, x0.shape)
                paths = self.sample_rollout(
                    x0, segment_controls, path_key, n_particles,
                    segment_start, 1.0, solver_name,
                )

                def group_score(indices, time_weights, group_path_weight):
                    group_paths = paths[..., indices]
                    group_target = target[..., indices]
                    score = jnp.mean(time_weights * energy_score(
                        group_paths, group_target, spread_weight=uncertainty_weight))
                    continuity = squared_mean_error(group_paths[:, -1], group_target[-1])
                    return group_path_weight * score + continuity_weight * continuity

                return (
                    group_score(ARM_INDICES, jnp.ones_like(object_time_weights),
                                path_weight),
                    group_score(OBJECT_INDICES, object_time_weights,
                                object_path_weight),
                )

            robot, obj = jax.vmap(segment_losses)(
                state_starts, control_segments, targets, segment_starts,
                object_weights, jr.split(shooting_key, n_segments),
            )
            return jnp.mean(robot), jnp.mean(obj)

        robot, obj = jax.vmap(trajectory_losses)(
            states, controls, start_steps, batch_keys, object_temporal_weights
        )
        return jnp.mean(robot), jnp.mean(obj)

    def multiple_shooting_loss(self, states, controls, start_steps, key, *,
                               segment_length=16, n_particles=4,
                               solver_name="heun", shooting_noise_std=0.01,
                               control_noise_std=0.01,
                               path_weight=1.0,
                               uncertainty_weight=0.5,
                               continuity_weight=0.1,
                               state_loss_weights=None,
                               temporal_loss_weights=None,
                               contact_weight=1.0,
                               contact_center=0.5,
                               contact_width=0.15):
        """Time-marginal energy scores over multiple-shooting segments.

        At uncertainty_weight=.5 this is proper for each time marginal;
        it alone does not identify temporal dependence. Other weights are
        explicit task regularizers, not calibration scores.

        ``states`` has shape (B, W+1, dx), ``controls`` (B, W, du), and W
        must be divisible by ``segment_length``. Each segment is integrated by
        the selected stochastic solver instead of evaluating one-step NLL.
        Optional ``temporal_loss_weights`` has shape (B, W) and is loss-only
        supervision; it is not passed to drift or diffusion networks.
        """
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments = window_steps // segment_length
        weights = (
            jnp.ones((self.state_dim,), dtype=states.dtype)
            if state_loss_weights is None
            else jnp.asarray(state_loss_weights, dtype=states.dtype)
        )
        batch_keys = jr.split(key, states.shape[0])
        supplied_temporal_weights = temporal_loss_weights is not None
        if temporal_loss_weights is None:
            temporal_loss_weights = jnp.ones(
                controls.shape[:2], dtype=states.dtype
            )

        def trajectory_loss(x, u, first_step, trajectory_key,
                            trajectory_temporal_weights):
            control_key, shooting_key = jr.split(trajectory_key)
            if control_noise_std > 0:
                u = u + control_noise_std * jr.normal(control_key, u.shape)
            segment_keys = jr.split(shooting_key, n_segments)
            state_starts = x[:-1].reshape(
                n_segments, segment_length, self.state_dim
            )[:, 0]
            all_histories = self.sequence_histories(x[None])
            segment_histories = (
                None if all_histories is None else
                all_histories[0, ::segment_length]
            )
            targets = x[1:].reshape(
                n_segments, segment_length, self.state_dim
            )
            control_segments = u.reshape(
                n_segments, segment_length, self.control_dim
            )
            segment_starts = first_step + jnp.arange(n_segments) * segment_length
            relative_time = (
                jnp.arange(window_steps, dtype=states.dtype) + 1
            ) / window_steps
            temporal_weights = 1.0 + (contact_weight - 1.0) * jnp.exp(
                -0.5 * ((relative_time - contact_center) / contact_width) ** 2
            )
            if supplied_temporal_weights:
                # Privileged labels may shape the training loss, but are never
                # part of the learned dynamics interface or inference inputs.
                temporal_weights *= trajectory_temporal_weights
            temporal_weights /= jnp.mean(temporal_weights)
            temporal_weights = temporal_weights.reshape(
                n_segments, segment_length
            )

            if segment_histories is None:
                segment_histories = jnp.broadcast_to(
                    state_starts[:, None, :],
                    (n_segments, 1, self.state_dim),
                )

            def segment_loss(x0, history, segment_controls, target, segment_start,
                             time_weights, seg_key):
                init_key, path_key = jr.split(seg_key)
                if shooting_noise_std > 0:
                    x0 = x0 + shooting_noise_std * jr.normal(init_key, x0.shape)
                paths = self.sample_rollout(
                    x0, segment_controls, path_key, n_particles,
                    segment_start, 1.0, solver_name, history,
                )
                score = jnp.mean(time_weights * energy_score(
                    paths, target, weights=weights, spread_weight=uncertainty_weight))
                continuity = squared_mean_error(paths[:, -1], target[-1], weights=weights)
                return score, continuity

            scores, continuities = jax.vmap(segment_loss)(
                state_starts, segment_histories, control_segments, targets, segment_starts,
                temporal_weights, segment_keys,
            )
            return (
                path_weight * jnp.mean(scores)
                + continuity_weight * jnp.mean(continuities)
            )

        return jnp.mean(jax.vmap(trajectory_loss)(
            states, controls, start_steps, batch_keys, temporal_loss_weights
        ))


def normflowsde_ito_targets(
    teacher: ControlledRealNVPFlowSDE,
    state,
    control,
    time,
    key,
    *,
    samples=64,
    state_history=None,
    diffusion_temperature=1.0,
    jitter=1e-6,
):
    """Match fixed-dt transition moments, not a grid-independent SDE limit."""
    if samples < 2:
        raise ValueError("moment estimation requires at least two samples")
    eps = jr.normal(key, (samples, teacher.state_dim), dtype=state.dtype)
    values = teacher._centered_diffusion_values(
        eps, state, control, time, state_history, diffusion_temperature
    )
    mean, covariance = _mean_covariance(values, jitter)
    dt = jnp.asarray(teacher.dt, dtype=state.dtype)
    drift = teacher.drift(state, control, time, state_history) + mean / jnp.sqrt(dt)
    return drift, covariance


def distilled_ito_loss(
    student: DistilledItoSDE,
    teacher: ControlledRealNVPFlowSDE,
    states,
    controls,
    times,
    key,
    *,
    moment_samples=64,
    lambda_mu=1.0,
    lambda_cov=1.0,
    diffusion_temperature=1.0,
    jitter=1e-6,
):
    """Distill FlowSDE local moments into a compact MLP Itô surrogate."""
    keys = jr.split(key, states.shape[0])
    dt = jnp.asarray(teacher.dt, dtype=states.dtype)

    def one(state, control, time, subkey):
        target_mu, target_cov = normflowsde_ito_targets(
            teacher, state, control, time, subkey,
            samples=moment_samples,
            diffusion_temperature=diffusion_temperature,
            jitter=jitter,
        )
        pred_mu = student.drift(state, control, time)
        pred_factor = student.diffusion_factor(state, control, time)
        pred_cov = pred_factor @ pred_factor.T
        # Compare mean state increments for scale stability, and covariance in
        # diffusion coordinates before the outer sqrt(dt) factor is applied.
        mean_loss = jnp.mean(((pred_mu - target_mu) * dt) ** 2)
        cov_loss = jnp.mean((pred_cov - target_cov) ** 2)
        return lambda_mu * mean_loss + lambda_cov * cov_loss

    return jnp.mean(jax.vmap(one)(states, controls, times, keys))


@eqx.filter_jit
def make_step_distilled_ito(
    student: DistilledItoSDE,
    teacher: ControlledRealNVPFlowSDE,
    states,
    controls,
    times,
    key,
    opt_state,
    optimizer,
    *,
    moment_samples=64,
    lambda_mu=1.0,
    lambda_cov=1.0,
    diffusion_temperature=1.0,
    jitter=1e-6,
):
    """One optimizer step for the MLP Itô distillation objective."""
    def loss_fn(candidate):
        return distilled_ito_loss(
            candidate, teacher, states, controls, times, key,
            moment_samples=moment_samples,
            lambda_mu=lambda_mu,
            lambda_cov=lambda_cov,
            diffusion_temperature=diffusion_temperature,
            jitter=jitter,
        )

    loss, grads = eqx.filter_value_and_grad(loss_fn)(student)
    updates, opt_state = optimizer.update(grads, opt_state, student)
    return eqx.apply_updates(student, updates), opt_state, loss
