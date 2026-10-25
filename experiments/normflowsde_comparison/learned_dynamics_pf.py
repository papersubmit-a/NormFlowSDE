"""NSF-PF and FlowSDE-PF: learned-dynamics bootstrap particle filters.

The observation-conditioned NSF/FlowSDE generators feed measurements only
through an encoder context and never correct a generated state against the
current measurement, so their error grows like an open-loop forecast. Here the
same dynamics components become the prediction step of a particle filter:

    predict   x[k+1] ~ p_theta(. | x[k], u[k])      (NSF flow / FlowSDE Euler-Maruyama)
    update    w     *= q_phi(y[k] | x[k+1])         (conditional normalizing flow)
    resample  systematic, when ESS < threshold * N

The dynamics see controls and the physical clock only, never observations, so
every measurement enters through the Bayesian update. Index conventions follow
the comparison caches: observations[k] measures states[k+1]; controls[k]
drives states[k] -> states[k+1]; states[0] is the known initial state. All
arrays are in the cache's training-normalized coordinates.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import logsumexp

from bayesinf.applications.state_estimation.flow_sde import RealNVPDiffusionNet, SDEDriftNet
from bayesinf.applications.state_estimation.fm_flow_sde import FlowMatchingSDE, FlowSDEConfig, flow_sde_loss
from bayesinf.applications.state_estimation.neural_stochastic_flow import NeuralStochasticFlow
from bayesinf.applications.state_estimation.normalizing_flow_dpf import ConditionalNormalizingFlow


class NSFDynamics(eqx.Module):
    """Exact-density NSF transition p(x[k+1] | x[k], u[k], t_k), trained by one-step NLL."""

    transition: NeuralStochasticFlow
    dt: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(self, state_dim, control_dim, dt, *, hidden_size, hidden_depth, flow_layers, key):
        self.dt, self.state_dim = float(dt), state_dim
        self.transition = NeuralStochasticFlow(
            state_dim, control_dim, hidden_size=hidden_size, hidden_depth=hidden_depth,
            num_coupling_layers=flow_layers, include_absolute_time=True, key=key)

    def loss(self, states, controls, key):
        del key
        times = jnp.broadcast_to(self.dt * jnp.arange(controls.shape[1], dtype=states.dtype), controls.shape[:2])
        log_prob = self.transition.log_prob(states[:, 1:], states[:, :-1], self.dt, controls, t_current=times)
        return -jnp.mean(log_prob) / self.state_dim

    def step(self, key, particles, control, index):
        count = particles.shape[0]
        return self.transition.sample(
            key, particles, self.dt, jnp.broadcast_to(control, (count, control.shape[-1])), num_samples=1,
            t_current=jnp.full((count,), self.dt * index, particles.dtype))[0]


class FlowSDEDynamics(eqx.Module):
    """FlowSDE drift with fixed isotropic diffusion D, trained by bridge flow matching."""

    dynamics: FlowMatchingSDE
    dt: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    config: FlowSDEConfig = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)

    def __init__(self, state_dim, control_dim, dt, *, hidden_size, hidden_depth, diffusion, substeps,
                 epr_weight=0., hjb_weight=1e-3, orthogonality_weight=1e-3, key):
        self.dt, self.substeps, self.state_dim = float(dt), int(substeps), state_dim
        # As in the observation-conditioned baseline, the physical gap is an extra control feature.
        self.dynamics = FlowMatchingSDE(state_dim, control_dim=control_dim + 1, width=hidden_size,
                                        depth=hidden_depth, diffusion=diffusion, use_potential=True, key=key)
        self.config = FlowSDEConfig(diffusion=diffusion, epr_weight=epr_weight, hjb_weight=hjb_weight,
                                    orthogonality_weight=orthogonality_weight,
                                    regularizer_measure="boltzmann_importance")

    def _conditions(self, controls):
        return jnp.concatenate((controls, jnp.full((*controls.shape[:-1], 1), self.dt, controls.dtype)), axis=-1)

    def loss(self, states, controls, key):
        times = jnp.broadcast_to(self.dt * jnp.arange(controls.shape[1], dtype=states.dtype),
                                 controls.shape[:2]).reshape(-1)
        parts = flow_sde_loss(self.dynamics, states[:, :-1].reshape(-1, self.state_dim),
                              states[:, 1:].reshape(-1, self.state_dim), times, times + self.dt, key=key,
                              controls=self._conditions(controls).reshape(-1, self.dynamics.control_dim),
                              config=self.config)
        return parts.total

    def step(self, key, particles, control, index):
        condition = self._conditions(control)
        sub_dt = self.dt / self.substeps
        scale = jnp.sqrt(2 * self.dynamics.diffusion * sub_dt)

        def substep(i, state):
            time = (index * self.substeps + i) * sub_dt
            drift = jax.vmap(lambda x: self.dynamics(x, time, control=condition))(state)
            return state + sub_dt * drift + scale * jr.normal(jr.fold_in(key, i), state.shape, state.dtype)

        return jax.lax.fori_loop(0, self.substeps, substep, particles)


LOG_SCALE_BOUNDS = (-10., 3.)


class NormFlowSDEDynamics(eqx.Module):
    """NormFlowSDE's Euler transition z' = z + dt f(z,u,t) + sqrt(dt) T(eps; z,u,t), eps ~ N(0,I).

    Same drift network and conditional RealNVP noise flow as the amortized method, but
    conditioned on control and time only (no measurement encoder), and trained by the
    exact Euler transition NLL of RealNVPFlowSDE.transition_nll.

    ``learned_scale=True`` puts a learned per-coordinate scale in front of the flow,
    z' = z + dt f + sqrt(dt) sigma(z,u,t) * T(eps), with log sigma smoothly bounded to
    LOG_SCALE_BOUNDS and initialized at 0 (sigma = 1). The coupling layers' log-scales
    are bounded to +-alpha each, which limits how small the noise can be; this Gaussian
    stage (as in NSF's candidate) frees the flow to model shape and correlation only.
    """

    drift: SDEDriftNet
    diffusion: RealNVPDiffusionNet
    scale: eqx.nn.MLP | None
    dt: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    use_time: bool = eqx.field(static=True)

    def __init__(self, state_dim, control_dim, dt, *, hidden_size, hidden_depth, flow_layers, time_emb_dim=32,
                 alpha=1.0, use_time=True, learned_scale=False, key):
        if control_dim < 1:
            raise ValueError("NormFlowSDEDynamics expects at least one control channel as conditioning")
        drift_key, diffusion_key = jr.split(key)  # unchanged split: default runs keep their initialization
        scale_key = jr.fold_in(key, 2)
        scale = None
        if learned_scale:
            scale = eqx.nn.MLP(state_dim + control_dim + time_emb_dim, state_dim, hidden_size, hidden_depth,
                               activation=jax.nn.gelu, key=scale_key)
            # Zero output layer: log sigma = 0 at initialization, so training starts from the unscaled flow.
            last = scale.layers[-1]
            scale = eqx.tree_at(lambda m: (m.layers[-1].weight, m.layers[-1].bias), scale,
                                (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
        self.scale = scale
        self.dt, self.state_dim, self.use_time = float(dt), state_dim, bool(use_time)
        self.drift = SDEDriftNet(state_dim, control_dim, hidden_size, hidden_depth, time_emb_dim, drift_key)
        # alpha bounds each coupling layer's log-scale to +-alpha (RealNVPFlowSDE's realnvp_alpha).
        self.diffusion = RealNVPDiffusionNet(state_dim, control_dim, hidden_size, flow_layers, time_emb_dim,
                                             diffusion_key, alpha=alpha)

    def _time(self, t):
        """Absolute physical time, or a constant 0 for time-invariant dynamics (use_time=False)."""
        return t if self.use_time else jnp.zeros_like(t)

    def _log_scale(self, z, u, t):
        if self.scale is None:
            return jnp.zeros_like(z)
        low, high = LOG_SCALE_BOUNDS
        offset = jnp.log(-low / high)  # sigmoid(offset) = -low / (high - low), so log sigma = 0 at a zero output
        raw = self.scale(jnp.concatenate((z, u, self.drift.time_emb(t))))
        return low + (high - low) * jax.nn.sigmoid(raw + offset)

    def _nll(self, x0, x1, u, t):
        t = self._time(t)
        log_scale = self._log_scale(x0, u, t)
        residual = (x1 - x0 - self.dt * self.drift(x0, u, t)) / jnp.sqrt(self.dt) * jnp.exp(-log_scale)
        eps, log_det = self.diffusion.inverse_and_log_det(residual, x0, u, t)
        # As in RealNVPFlowSDE.transition_nll, the inverse routine returns the forward log determinant;
        # the learned scale adds sum(log sigma) to it.
        return (0.5 * jnp.sum(eps ** 2) + 0.5 * self.state_dim * jnp.log(2 * jnp.pi) + log_det + jnp.sum(log_scale)
                + 0.5 * self.state_dim * jnp.log(self.dt)) / self.state_dim

    def loss(self, states, controls, key):
        del key
        times = jnp.broadcast_to(self.dt * jnp.arange(controls.shape[1], dtype=states.dtype), controls.shape[:2])
        nll = jax.vmap(jax.vmap(self._nll))(states[:, :-1], states[:, 1:], controls, times)
        return jnp.mean(nll)

    def step(self, key, particles, control, index):
        time = self._time(jnp.asarray(self.dt * index, particles.dtype))
        noise = jr.normal(key, particles.shape, particles.dtype)

        def one(z, eps):
            noise = jnp.exp(self._log_scale(z, control, time)) * self.diffusion.apply(eps, z, control, time)
            return z + self.dt * self.drift(z, control, time) + jnp.sqrt(self.dt) * noise

        return jax.vmap(one)(particles, noise)


class LearnedDynamicsPF(eqx.Module):
    """Bootstrap particle filter with learned dynamics and a learned measurement density."""

    dynamics: NSFDynamics | FlowSDEDynamics | NormFlowSDEDynamics
    measurement: ConditionalNormalizingFlow

    def __init__(self, dynamics, observation_dim, state_dim, *, hidden_size=64, hidden_depth=2, flow_layers=4,
                 key):
        self.dynamics = dynamics
        self.measurement = ConditionalNormalizingFlow(observation_dim, state_dim, hidden_size=hidden_size,
                                                      hidden_depth=hidden_depth, num_flow_layers=flow_layers,
                                                      base_scale=1.0, key=key)

    def measurement_loss(self, states, observations):
        """Mean -log q(y[k] | x[k+1]) per observation coordinate."""
        return -jnp.mean(self.measurement.log_prob(observations, states[:, 1:])) / observations.shape[-1]

    def loss(self, states, observations, controls, key):
        dynamics = self.dynamics.loss(states, controls, key)
        measurement = self.measurement_loss(states, observations)
        return dynamics + measurement, dict(dynamics=dynamics, measurement=measurement)

    def filter(self, observations, x0, controls, *, key, num_particles, resampling_threshold=0.5, jitter=0.):
        """Filtering particles (T,N,D), normalized weights (T,N) and ESS (T,) for one window.

        ``jitter`` adds N(0, jitter^2 I) process noise to every predicted particle (normalized
        units): process-noise inflation for model error, selected on validation data.
        """
        count = num_particles
        uniform = jnp.full((count,), -jnp.log(count), x0.dtype)

        def step(carry, inputs):
            particles, log_weights = carry
            observation, control, index = inputs
            resample_key, predict_key = jr.split(jr.fold_in(key, index))
            ess = jnp.exp(-logsumexp(2 * log_weights))
            positions = (jr.uniform(resample_key, (), x0.dtype) + jnp.arange(count)) / count
            ancestors = jnp.searchsorted(jnp.cumsum(jnp.exp(log_weights)).at[-1].set(1.), positions, side="right")
            resample = ess < resampling_threshold * count
            particles = jnp.where(resample, particles[ancestors], particles)
            log_weights = jnp.where(resample, uniform, log_weights)
            particles = self.dynamics.step(predict_key, particles, control, index)
            particles = particles + jitter * jr.normal(jr.fold_in(predict_key, 1), particles.shape, particles.dtype)
            log_weights = log_weights + self.measurement.log_prob(observation, particles)
            normalizer = logsumexp(log_weights)
            log_weights = jnp.where(jnp.isfinite(normalizer), log_weights - normalizer, uniform)
            return (particles, log_weights), (particles, jnp.exp(log_weights), jnp.exp(-logsumexp(2 * log_weights)))

        initial = jnp.broadcast_to(x0, (count, x0.shape[-1]))
        _, (particles, weights, ess) = jax.lax.scan(
            step, (initial, uniform), (observations, controls, jnp.arange(observations.shape[0])))
        return particles, weights, ess

    def sample(self, observations, x0, controls, *, key, num_particles, num_samples, jitter=0.):
        """Posterior draws (M,T,D): per time, categorical draws from the weighted filtering particles.

        As for NF-DPF, these are filtering marginals; draws at different times are not coherent paths.
        """
        filter_key, draw_key = jr.split(key)
        particles, weights, _ = self.filter(observations, x0, controls, key=filter_key, num_particles=num_particles,
                                            jitter=jitter)
        keys = jr.split(draw_key, observations.shape[0])
        draws = jax.vmap(lambda k, p, w: p[jr.categorical(k, jnp.log(w), shape=(num_samples,))])(
            keys, particles, weights)
        return draws.swapaxes(0, 1)


__all__ = ["NSFDynamics", "FlowSDEDynamics", "NormFlowSDEDynamics", "LearnedDynamicsPF"]
