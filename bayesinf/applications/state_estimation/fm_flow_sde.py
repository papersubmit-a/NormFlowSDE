"""Flow-matching SDE dynamics for state-estimation applications.

This is an independent JAX implementation of the method described by Lou et
al., *FlowSDE: a flow-matching-based SDE framework for predicting state
transitions* (2026), DOI: 10.1098/rsta.2024.0492.  It deliberately keeps the
artificial bridge coordinate ``flow_time`` separate from physical SDE time.

The learned physical dynamics are

    dX_t = f_theta(t, X_t, u_t) dt + sqrt(2 D) I dW_t.

Training states are sampled from Gaussian bridges between observed transition
pairs.  The optional potential network is used only for the force-energy/EPR, HJB,
and Helmholtz--Hodge regularisers; the drift is not forced to be conservative.
The filtering layer can therefore use :meth:`FlowMatchingSDE.transition`
directly as a particle transition kernel.

The paper does not describe Bayesian filtering. Observation conditioning is
intentionally absent here: doing so without a density correction would not be
a valid particle-filter proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

Array = jax.Array


def _as_vector(value: Array | None, size: int, dtype: jnp.dtype) -> Array:
    if size == 0:
        return jnp.empty((0,), dtype=dtype)
    if value is None:
        return jnp.zeros((size,), dtype=dtype)
    value = jnp.asarray(value, dtype=dtype)
    if value.shape != (size,):
        raise ValueError(f"Expected a vector of shape ({size},), got {value.shape}.")
    return value


class FourierTimeEmbedding(eqx.Module):
    """Deterministic sine/cosine embedding of one scalar."""

    frequencies: Array
    output_size: int = eqx.field(static=True)

    def __init__(self, output_size: int, max_frequency: float = 100.0):
        if output_size < 1:
            raise ValueError("output_size must be positive")
        self.output_size = output_size
        pairs = output_size // 2
        self.frequencies = (
            jnp.geomspace(1.0, max_frequency, pairs) if pairs else jnp.empty((0,))
        )

    def __call__(self, time: Array) -> Array:
        angles = jnp.asarray(time) * self.frequencies
        result = jnp.concatenate((jnp.sin(angles), jnp.cos(angles)))
        if self.output_size % 2:
            result = jnp.concatenate((result, jnp.asarray(time)[None]))
        return result


class DriftNetwork(eqx.Module):
    """Physical drift ``f(t, x, u)``.

    ``flow_time`` is accepted to make the two clocks explicit at call sites,
    but is deliberately not a model feature: the SDE drift must remain a
    function of physical time, not of the training-only bridge coordinate.
    """

    network: eqx.nn.MLP
    physical_embedding: FourierTimeEmbedding
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        *,
        control_dim: int = 0,
        width: int = 128,
        depth: int = 3,
        time_embedding_dim: int = 16,
        key: Array,
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.physical_embedding = FourierTimeEmbedding(time_embedding_dim)
        self.network = eqx.nn.MLP(
            state_dim + control_dim + time_embedding_dim,
            state_dim,
            width,
            depth,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(
        self,
        state: Array,
        physical_time: Array,
        flow_time: Array = jnp.asarray(0.0),
        control: Array | None = None,
    ) -> Array:
        del flow_time
        control = _as_vector(control, self.control_dim, state.dtype)
        features = jnp.concatenate(
            (
                state,
                control,
                self.physical_embedding(physical_time),
            )
        )
        return self.network(features)


class PotentialNetwork(eqx.Module):
    """Time-independent scalar mean-field potential ``U(x)``."""

    network: eqx.nn.MLP
    state_dim: int = eqx.field(static=True)

    def __init__(
        self, state_dim: int, *, width: int = 128, depth: int = 3, key: Array
    ):
        self.state_dim = state_dim
        self.network = eqx.nn.MLP(
            state_dim,
            1,
            width,
            depth,
            activation=jax.nn.silu,
            final_activation=lambda x: x,
            key=key,
        )

    def __call__(self, state: Array) -> Array:
        return self.network(state)[0]


@dataclass(frozen=True)
class GaussianBridge:
    """Differentiable bridge used to construct conditional FM targets.

    ``noise_scale`` is the standard deviation at the middle of a unit-time
    bridge up to the factor ``sqrt(tau * (1-tau))``. If omitted by the loss,
    the SDE-consistent value ``sqrt(2 * D * delta_time)`` is used.
    """

    endpoint_epsilon: float = 1e-4

    def coefficients(self, flow_time: Array) -> tuple[Array, Array, Array]:
        tau = jnp.clip(flow_time, 0.0, 1.0)
        variance_shape = tau * (1.0 - tau)
        sigma_shape = jnp.sqrt(variance_shape)
        safe_sigma = jnp.sqrt(jnp.maximum(variance_shape, self.endpoint_epsilon**2))
        derivative = (1.0 - 2.0 * tau) / (2.0 * safe_sigma)
        return tau, sigma_shape, derivative

    def sample_and_velocity(
        self,
        start: Array,
        end: Array,
        flow_time: Array,
        noise: Array,
        *,
        delta_time: Array,
        noise_scale: Array,
    ) -> tuple[Array, Array]:
        """Return bridge state and target velocity in *physical-time units*."""
        tau, sigma_shape, derivative = self.coefficients(flow_time)
        dt = jnp.maximum(jnp.asarray(delta_time), jnp.finfo(start.dtype).eps)
        state = (1.0 - tau) * start + tau * end + noise_scale * sigma_shape * noise
        velocity = ((end - start) + noise_scale * derivative * noise) / dt
        return state, velocity

    def sample_and_drift(
        self, start, end, flow_time, noise, *, delta_time, noise_scale, diffusion
    ):
        """Gaussian-bridge forward Ito drift, in physical-time units.

        A differentiable bridge velocity solves a continuity equation. Its
        forward SDE with diffusion sqrt(2D) has drift velocity + D*score.
        Averaging this conditional target gives the matching marginal drift.
        Interior bridge times and positive bridge noise are required.
        """
        state, velocity = self.sample_and_velocity(
            start, end, flow_time, noise,
            delta_time=delta_time, noise_scale=noise_scale,
        )
        _, sigma_shape, _ = self.coefficients(flow_time)
        sigma = noise_scale * sigma_shape
        sigma = eqx.error_if(sigma, sigma <= 0, "SDE bridge noise must be positive at an interior time")
        return state, velocity - diffusion * noise / sigma


@dataclass(frozen=True)
class FlowSDEConfig:
    """Loss and solver choices not fixed unambiguously by the article."""

    diffusion: float = 0.1
    flow_weight: float = 1.0
    epr_weight: float = 0.0
    hjb_weight: float = 0.0
    orthogonality_weight: float = 0.0
    bridge_noise_scale: float | None = None
    bridge_endpoint_epsilon: float = 1e-4
    boltzmann_clip: float = 30.0  # Legacy field; importance weights are not clipped.
    solver: str = "euler"
    dt0: float = 1e-2
    rtol: float = 1e-5
    atol: float = 1e-6
    max_steps: int = 100_000
    regularizer_measure: str = "empirical"

    def __post_init__(self) -> None:
        if self.regularizer_measure not in {"empirical", "boltzmann_importance"}:
            raise ValueError("regularizer_measure must be empirical or boltzmann_importance")
        if self.diffusion <= 0:
            raise ValueError("diffusion D must be strictly positive")
        if self.dt0 <= 0:
            raise ValueError("dt0 must be strictly positive")
        if self.bridge_noise_scale is not None and self.bridge_noise_scale <= 0:
            raise ValueError("bridge_noise_scale must be positive for SDE drift matching")
        if not 0 < self.bridge_endpoint_epsilon < 0.5:
            raise ValueError("bridge_endpoint_epsilon must lie in (0, 0.5)")


class FlowMatchingSDE(eqx.Module):
    """Learned drift, fixed isotropic diffusion, and optional potential."""

    drift: DriftNetwork
    potential: PotentialNetwork | None
    diffusion: float = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)

    def __init__(
        self,
        state_dim: int,
        *,
        control_dim: int = 0,
        width: int = 128,
        depth: int = 3,
        time_embedding_dim: int = 16,
        diffusion: float = 0.1,
        use_potential: bool = True,
        key: Array,
    ):
        if diffusion <= 0:
            raise ValueError("diffusion D must be strictly positive")
        drift_key, potential_key = jr.split(key)
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.diffusion = float(diffusion)
        self.drift = DriftNetwork(
            state_dim,
            control_dim=control_dim,
            width=width,
            depth=depth,
            time_embedding_dim=time_embedding_dim,
            key=drift_key,
        )
        self.potential = (
            PotentialNetwork(state_dim, width=width, depth=depth, key=potential_key)
            if use_potential
            else None
        )

    def __call__(
        self,
        state: Array,
        physical_time: Array,
        flow_time: Array = jnp.asarray(0.0),
        control: Array | None = None,
    ) -> Array:
        return self.drift(state, physical_time, flow_time, control)

    def diffusion_matrix(self, state: Array) -> Array:
        return jnp.sqrt(2.0 * self.diffusion) * jnp.eye(
            self.state_dim, dtype=state.dtype
        )

    def potential_gradient(self, state: Array) -> Array:
        if self.potential is None:
            raise ValueError("This model was constructed without a potential network.")
        return jax.grad(self.potential)(state)

    def nonconservative_force(
        self, state: Array, physical_time: Array, control: Array | None = None
    ) -> Array:
        return self(state, physical_time, jnp.asarray(0.0), control) + self.potential_gradient(
            state
        )

    def transition(
        self,
        state: Array,
        t0: float | Array,
        t1: float | Array,
        *,
        key: Array,
        control: Array | None = None,
        config: FlowSDEConfig | None = None,
    ) -> Array:
        """Draw one transition using Diffrax and an explicit Brownian key."""
        config = config or FlowSDEConfig(diffusion=self.diffusion)
        if config.diffusion != self.diffusion:
            raise ValueError("config.diffusion must equal model.diffusion")
        t0, t1 = jnp.asarray(t0), jnp.asarray(t1)
        control = _as_vector(control, self.control_dim, state.dtype)

        def drift_fn(t, y, args):
            del args
            return self(y, t, jnp.asarray(0.0, dtype=y.dtype), control)

        def diffusion_fn(t, y, args):
            del t, args
            return self.diffusion_matrix(y)

        brownian = diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, tol=max(config.atol, 1e-5), shape=(self.state_dim,), key=key
        )
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(drift_fn), diffrax.ControlTerm(diffusion_fn, brownian)
        )
        solvers = {
            "euler": diffrax.Euler(),
            "euler_heun": diffrax.EulerHeun(),
            "reversible_heun": diffrax.ReversibleHeun(),
        }
        if config.solver not in solvers:
            raise ValueError(f"Unknown SDE solver {config.solver!r}; choose from {tuple(solvers)}")
        adaptive = config.solver == "reversible_heun"
        controller = (
            diffrax.PIDController(rtol=config.rtol, atol=config.atol)
            if adaptive
            else diffrax.ConstantStepSize()
        )
        solution = diffrax.diffeqsolve(
            terms,
            solvers[config.solver],
            t0=t0,
            t1=t1,
            dt0=jnp.minimum(jnp.asarray(config.dt0), t1 - t0),
            y0=state,
            saveat=diffrax.SaveAt(t1=True),
            stepsize_controller=controller,
            max_steps=config.max_steps,
        )
        return solution.ys[0]

    def transition_particles(
        self,
        particles: Array,
        t0: float | Array,
        t1: float | Array,
        *,
        key: Array,
        control: Array | None = None,
        config: FlowSDEConfig | None = None,
    ) -> Array:
        """Independently propagate ``(n_particles, state_dim)`` particles."""
        keys = jr.split(key, particles.shape[0])
        return jax.vmap(
            lambda particle, particle_key: self.transition(
                particle, t0, t1, key=particle_key, control=control, config=config
            )
        )(particles, keys)


# Concise alias matching the article name.
FMFlowSDE = FlowMatchingSDE


class FlowSDELosses(NamedTuple):
    total: Array
    flow_matching: Array
    epr: Array
    hjb: Array
    orthogonality: Array


def _single_regularisers(
    model: FlowMatchingSDE,
    state: Array,
    physical_time: Array,
    control: Array,
) -> tuple[Array, Array, Array, Array]:
    if model.potential is None:
        zero = jnp.asarray(0.0, dtype=state.dtype)
        return zero, zero, zero, zero

    potential = model.potential(state)
    grad_u = jax.grad(model.potential)(state)
    laplacian_u = jnp.trace(jax.hessian(model.potential)(state))

    def drift_at(x):
        return model(x, physical_time, jnp.asarray(0.0, dtype=x.dtype), control)

    drift = drift_at(state)
    divergence = jnp.trace(jax.jacfwd(drift_at)(state))
    force = drift + grad_u
    epr_integrand = jnp.vdot(force, force) / model.diffusion
    hjb_residual = (
        -jnp.vdot(drift, grad_u)
        + model.diffusion * laplacian_u
        + model.diffusion * divergence
        - jnp.vdot(grad_u, grad_u)
    )
    orth_integrand = jnp.vdot(grad_u, force) ** 2
    return potential, epr_integrand, hjb_residual**2, orth_integrand


def flow_sde_loss(
    model: FlowMatchingSDE,
    starts: Array,
    ends: Array,
    t0: Array,
    t1: Array,
    *,
    key: Array,
    controls: Array | None = None,
    config: FlowSDEConfig | None = None,
) -> FlowSDELosses:
    """Simulation-free matching of the forward Ito drift of Gaussian bridges.

    Shapes are ``starts, ends: (batch, state_dim)``, ``t0, t1: (batch,)``,
    and optionally ``controls: (batch, control_dim)``. The target is divided by
    each physical interval, so irregularly sampled transitions are supported.

    By default, regularizers are empirical bridge-state penalties; ``epr`` is
    then a force-energy penalty, not a physical stationary entropy-production
    integral. ``regularizer_measure="boltzmann_importance"`` estimates that
    integral using self-normalized importance weights, including the known
    conditional Gaussian bridge proposal density. This unbiased proposal
    correction can have high variance for narrow endpoint bridges; finite-batch
    self-normalized estimates remain biased. No clipping changes the measure.
    """
    config = config or FlowSDEConfig(diffusion=model.diffusion)
    if config.diffusion != model.diffusion:
        raise ValueError("config.diffusion must equal model.diffusion")
    if starts.shape != ends.shape or starts.ndim != 2:
        raise ValueError("starts and ends must have equal (batch, state_dim) shapes")
    if starts.shape[1] != model.state_dim:
        raise ValueError("Transition state dimension does not match the model")
    batch = starts.shape[0]
    if t0.shape != (batch,) or t1.shape != (batch,):
        raise ValueError("t0 and t1 must both have shape (batch,)")
    if controls is None:
        controls = jnp.zeros((batch, model.control_dim), dtype=starts.dtype)
    if controls.shape != (batch, model.control_dim):
        raise ValueError(f"controls must have shape ({batch}, {model.control_dim})")

    tau_key, noise_key = jr.split(key)
    eps = config.bridge_endpoint_epsilon
    flow_times = jr.uniform(tau_key, (batch,), minval=eps, maxval=1.0 - eps)
    noises = jr.normal(noise_key, starts.shape)
    delta_times = t1 - t0
    delta_times = eqx.error_if(
        delta_times,
        jnp.any(delta_times <= 0),
        "Every t1 must be strictly greater than t0",
    )
    if config.bridge_noise_scale is None:
        noise_scales = jnp.sqrt(2.0 * model.diffusion * delta_times)
    else:
        noise_scales = jnp.full_like(delta_times, config.bridge_noise_scale)
    bridge = GaussianBridge(config.bridge_endpoint_epsilon)
    bridge_states, targets = jax.vmap(
        lambda x0, x1, tau, noise, dt, scale: bridge.sample_and_drift(
            x0, x1, tau, noise, delta_time=dt, noise_scale=scale,
            diffusion=model.diffusion,
        )
    )(starts, ends, flow_times, noises, delta_times, noise_scales)
    physical_times = t0 + flow_times * delta_times
    predictions = jax.vmap(model)(bridge_states, physical_times, flow_times, controls)
    fm = jnp.mean(jnp.sum((predictions - targets) ** 2, axis=-1))

    potentials, epr_values, hjb_values, orth_values = jax.vmap(
        lambda x, t, u: _single_regularisers(model, x, t, u)
    )(bridge_states, physical_times, controls)
    if model.potential is None:
        epr = hjb = orth = jnp.asarray(0.0, dtype=starts.dtype)
    else:
        if config.regularizer_measure == "boltzmann_importance":
            sigmas = noise_scales * jnp.sqrt(flow_times * (1.0 - flow_times))
            log_proposal = (-0.5 * jnp.sum(noises ** 2, axis=-1)
                            - model.state_dim * (jnp.log(sigmas) + 0.5 * jnp.log(2 * jnp.pi)))
            weights = jax.nn.softmax(-potentials / model.diffusion - log_proposal)
        else:
            weights = jnp.full((batch,), 1.0 / batch, dtype=starts.dtype)
        epr = jnp.sum(weights * epr_values)
        hjb = jnp.mean(hjb_values)
        orth = jnp.sum(weights * orth_values)
    total = (
        config.flow_weight * fm
        + config.epr_weight * epr
        + config.hjb_weight * hjb
        + config.orthogonality_weight * orth
    )
    return FlowSDELosses(total, fm, epr, hjb, orth)


def make_train_step(optimizer: optax.GradientTransformation):
    """Create a filtered, JIT-compiled Optax update with gradient diagnostics."""

    @eqx.filter_jit
    def step(
        model: FlowMatchingSDE,
        opt_state: optax.OptState,
        starts: Array,
        ends: Array,
        t0: Array,
        t1: Array,
        key: Array,
        controls: Array | None,
        config: FlowSDEConfig,
    ) -> tuple[FlowMatchingSDE, optax.OptState, FlowSDELosses, Array, Array]:
        def objective(candidate):
            losses = flow_sde_loss(
                candidate,
                starts,
                ends,
                t0,
                t1,
                key=key,
                controls=controls,
                config=config,
            )
            return losses.total, losses

        (_, losses), grads = eqx.filter_value_and_grad(objective, has_aux=True)(model)
        leaves = [x for x in jax.tree.leaves(grads) if eqx.is_array(x)]
        grad_norm = optax.global_norm(grads)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(x)) for x in leaves]))
        parameters = eqx.filter(model, eqx.is_inexact_array)
        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        updates = jax.tree.map(
            lambda update: jnp.where(finite, update, jnp.zeros_like(update))
            if eqx.is_array(update)
            else update,
            updates,
        )
        model = eqx.apply_updates(model, updates)
        return model, opt_state, losses, grad_norm, finite

    return step


def initialise_optimizer(
    model: FlowMatchingSDE,
    *,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
) -> tuple[optax.GradientTransformation, optax.OptState]:
    """AdamW with global-norm clipping, matching the supplement's optimizer family."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adamw(learning_rate, weight_decay=weight_decay),
    )
    return optimizer, optimizer.init(eqx.filter(model, eqx.is_inexact_array))


def make_transition_pairs(
    states: Array, times: Array
) -> tuple[Array, Array, Array, Array]:
    """Flatten ``(batch, time, state)`` trajectories into adjacent transitions."""
    if states.ndim != 3:
        raise ValueError("states must have shape (batch, time, state_dim)")
    if times.ndim == 1:
        times = jnp.broadcast_to(times, states.shape[:2])
    if times.shape != states.shape[:2]:
        raise ValueError("times must have shape (time,) or (batch, time)")
    return (
        states[:, :-1].reshape((-1, states.shape[-1])),
        states[:, 1:].reshape((-1, states.shape[-1])),
        times[:, :-1].reshape((-1,)),
        times[:, 1:].reshape((-1,)),
    )


__all__: Sequence[str] = (
    "FMFlowSDE",
    "FlowMatchingSDE",
    "FlowSDEConfig",
    "FlowSDELosses",
    "GaussianBridge",
    "DriftNetwork",
    "PotentialNetwork",
    "flow_sde_loss",
    "initialise_optimizer",
    "make_train_step",
    "make_transition_pairs",
)
