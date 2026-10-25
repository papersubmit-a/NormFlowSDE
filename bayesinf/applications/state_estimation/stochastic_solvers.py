"""Fixed-step stochastic transition schemes and Gaussian SDE integration.

All step functions consume standardized interval noise eps=dW/sqrt(dt).
Euler/Heun/midpoint/RK4 also accept nonlinear maps G(eps) as *fixed-grid
transition kernels*. Such maps do not generally define a continuous-time SDE;
changing their grid changes their law. Classical SDE convergence statements
apply only to linear centered Gaussian diffusion with the stated convention.
"""

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Callable, Optional, Tuple, NamedTuple
from functools import partial


def _positive_dt(t0: float, t1: float) -> jnp.ndarray:
    """Return a strictly positive step size for stable sqrt(dt)."""
    dt = jnp.asarray(t1 - t0)
    return eqx.error_if(dt, dt <= 0, "Solver intervals must be strictly increasing")


# -----------------------------------------------------------------------------
# Solver Noise Utilities
# -----------------------------------------------------------------------------

class BrownianIncrement(eqx.Module):
    """
    Generate independent standard-Gaussian solver noise.
    
    For SDE integration over time steps [t0, t1]:
        eps ~ N(0, I), and the solver applies sqrt(dt).
    """
    shape: Tuple[int, ...] = eqx.field(static=True)
    
    def __init__(self, shape: Tuple[int, ...]):
        self.shape = shape
    
    def sample(self, key: jr.PRNGKey, dt: float) -> jnp.ndarray:
        """Sample eps ~ N(0, I)."""
        del dt
        return jr.normal(key, self.shape)
    
    def sample_multiple(self, key: jr.PRNGKey, dt: float, n: int) -> jnp.ndarray:
        """Sample multiple independent standard-Gaussian increments."""
        keys = jr.split(key, n)
        return jnp.stack([self.sample(k, dt) for k in keys])


class VirtualBrownianTree(eqx.Module):
    """A genuine Brownian tree exposing standardized interval increments.

    sqrt(h)*evaluate(t,t+h) are additive over subdivisions of the same path.
    Unlike independent endpoint hashes this is suitable for adaptive integration.
    """
    path: diffrax.VirtualBrownianTree

    def __init__(self, t0, t1, shape, key, tol=1e-6):
        self.path = diffrax.VirtualBrownianTree(t0, t1, tol, shape, key)

    def evaluate(self, t0, t1):
        return self.path.evaluate(t0, t1) / jnp.sqrt(_positive_dt(t0, t1))


def _noise(bm, t0, t1):
    if not hasattr(bm, "evaluate"):
        raise TypeError("A solver needs an interval-indexed noise path; pre-sample BrownianIncrement into StepNoisePath")
    return bm.evaluate(t0, t1)


def _gaussian_matrix(diffuse_fn, t, z):
    """Extract and validate the local linear centered Gaussian noise map."""
    zero = jnp.zeros_like(z)
    matrix = jax.jacfwd(lambda eps: diffuse_fn(t, z, eps))(zero)
    probe = jnp.arange(1, z.size + 1, dtype=z.dtype) / (z.size + 1)
    residual = jnp.max(jnp.abs(diffuse_fn(t, z, probe) - matrix @ probe))
    residual = jnp.maximum(residual, jnp.max(jnp.abs(diffuse_fn(t, z, zero))))
    return eqx.error_if(matrix, residual > 1e-5,
        "This solver requires diffusion linear in centered Gaussian noise")


class StepNoisePath(eqx.Module):
    """
    Array-backed standard-Gaussian step noise for JAX scans.

    ``eps_steps[i]`` is returned for the interval starting at
    ``t_start + i * dt``. The solvers multiply the transformed diffusion by
    ``sqrt(dt)``, so these are standard normal epsilons, not Brownian-scaled
    increments.
    """
    eps_steps: jnp.ndarray
    t_start: jnp.ndarray
    dt: jnp.ndarray

    def __init__(self, eps_steps: jnp.ndarray, t_start: float, dt: float):
        self.eps_steps = eps_steps
        self.t_start = jnp.asarray(t_start)
        self.dt = jnp.asarray(dt)

    def evaluate(self, t0: float, t1: float) -> jnp.ndarray:
        position = (t0 - self.t_start) / self.dt
        idx = jnp.rint(position).astype(jnp.int32)
        invalid = ((idx < 0) | (idx >= self.eps_steps.shape[0])
                   | (jnp.abs(position - idx) > 1e-4 + 8*jnp.finfo(position.dtype).eps*jnp.abs(position))
                   | (jnp.abs((t1 - t0) / self.dt - 1) > 1e-4 + 8*jnp.finfo(position.dtype).eps*jnp.abs(t0/self.dt)))
        idx = eqx.error_if(idx, invalid, "StepNoisePath only supports its original fixed grid")
        return self.eps_steps[idx]


# -----------------------------------------------------------------------------
# Solver State
# -----------------------------------------------------------------------------

class SolverState(eqx.Module):
    """
    State container for adaptive solvers.
    """
    t: float
    y: jnp.ndarray
    dt: float
    error: Optional[jnp.ndarray] = None


# -----------------------------------------------------------------------------
# Base Solver Classes
# -----------------------------------------------------------------------------

class StochasticSolver(eqx.Module):
    """
    Base class for stochastic solvers.
    
    All solvers follow the signature:
        step(z, t0, t1, drift_fn, diffuse_fn, bm) -> z_new
        
    where:
        - z: current state (d_x,)
        - t0, t1: time interval [t0, t1]
        - drift_fn(t, z): returns drift f(z, t) → (d_x,)
        - diffuse_fn(t, z, dW): returns G_φ(dW, z, t) → (d_x,)
        - bm: step-noise source with evaluate(t0, t1) -> eps
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        """Perform one solver step."""
        raise NotImplementedError("Subclasses must implement step()")
    
    def name(self) -> str:
        """Return solver name."""
        raise NotImplementedError("Subclasses must implement name()")


# -----------------------------------------------------------------------------
# Euler-Maruyama Solver
# -----------------------------------------------------------------------------

class EulerMaruyama(StochasticSolver):
    """
    Euler-Maruyama solver (strong order 0.5).
    
    Update:
        z_{t+1} = z_t + f(z_t, t) * dt + G_φ(eps_t, z_t, t) * sqrt(dt)
    
    where eps_t ~ N(0, I) and G_φ is the RealNVP flow.
    
    This is the simplest and most widely used SDE solver.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        f0 = drift_fn(t0, z)
        dw = _noise(bm, t0, t1)
        g0 = diffuse_fn(t0, z, dw)
        return z + f0 * dt + g0 * dt_sqrt
    
    def name(self) -> str:
        return "EulerMaruyama"


class EulerMaruyamaWithGradient(EulerMaruyama):
    """Legacy alias of EulerMaruyama; use Milstein for a derivative correction.

    No additional numerical-order or gradient-correction claim is made.
    """


# -----------------------------------------------------------------------------
# Heun Solver (Stochastic)
# -----------------------------------------------------------------------------

class StochasticHeun(StochasticSolver):
    """
    Heun transition; Stratonovich convention for linear Gaussian diffusion.
    
    Update:
        z_p = z + f(z, t) * dt + G_φ(dW, z, t) * sqrt(dt)
        z_{t+1} = z + 0.5 * (f0 + f1) * dt
                    + 0.5 * (g0 + g1) * sqrt(dt)
    
    The same dW is used for both predictor and corrector steps.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        # Evaluate at t0
        f0 = drift_fn(t0, z)
        dw = _noise(bm, t0, t1)
        g0 = diffuse_fn(t0, z, dw)
        
        # Predictor step
        z_p = z + f0 * dt + g0 * dt_sqrt
        
        # Evaluate at t1 using predicted state
        f1 = drift_fn(t1, z_p)
        g1 = diffuse_fn(t1, z_p, dw)
        
        # Corrector step (average of start and end)
        return z + 0.5 * (f0 + f1) * dt + 0.5 * (g0 + g1) * dt_sqrt
    
    def name(self) -> str:
        return "StochasticHeun"


class ReversibleHeun(StochasticHeun):
    """Compatibility alias for the stateless Heun transition kernel.

    This API has no reversible auxiliary state and makes no memory-saving claim.
    For reversible Gaussian SDE integration use diffrax.ReversibleHeun directly.
    """
    def name(self):
        return "Heun (legacy reversible_heun alias)"


# -----------------------------------------------------------------------------
# Milstein Solver
# -----------------------------------------------------------------------------

class Milstein(StochasticSolver):
    """Ito Milstein for diagonal noise b_i(z_i,t) dW_i.

    Each diagonal coefficient must depend only on its own state coordinate.
    General matrix/noncommutative or nonlinear transformed noise requires a
    different integrator and is explicitly rejected.
    """
    use_gradient_correction: bool = eqx.field(static=True)

    def __init__(self, use_gradient_correction=True):
        self.use_gradient_correction = use_gradient_correction

    def step(self, z, t0, t1, drift_fn, diffuse_fn, bm):
        dt = _positive_dt(t0, t1)
        eps = _noise(bm, t0, t1)
        matrix = _gaussian_matrix(diffuse_fn, t0, z)
        diagonal = jnp.diag(matrix)
        matrix = eqx.error_if(matrix, jnp.max(jnp.abs(matrix-jnp.diag(diagonal))) > 1e-6,
                             "Milstein requires diagonal diffusion")
        result = z + drift_fn(t0, z)*dt + (matrix @ eps)*jnp.sqrt(dt)
        if self.use_gradient_correction:
            derivative = jax.jacfwd(lambda state: jnp.diag(
                _gaussian_matrix(diffuse_fn, t0, state)))(z)
            own = jnp.diag(derivative)
            own = eqx.error_if(own, jnp.max(jnp.abs(derivative-jnp.diag(own))) > 1e-6,
                              "Diagonal Milstein requires b_i to depend only on z_i")
            result = result + .5*diagonal*own*(eps**2-1)*dt
        return result

    def name(self):
        return "Ito diagonal Milstein"


# -----------------------------------------------------------------------------
# Stochastic Midpoint Solver
# -----------------------------------------------------------------------------

class StochasticMidpoint(StochasticSolver):
    """
    Stochastic Midpoint solver.
    
    Uses midpoint approximation for both drift and diffusion:
        z_{t+1} = z_t + f(z_mid, t_mid) * dt + G_φ(dW, z_mid, t_mid) * sqrt(dt)
    
    where z_mid = z_t + 0.5 * (f(z_t, t) * dt + G_φ(dW, z_t, t) * sqrt(dt))
    
    For linear multiplicative Gaussian noise this uses a Stratonovich convention;
    no general stiff-stability guarantee is made.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        t_mid = 0.5 * (t0 + t1)
        
        # Evaluate at t0
        f0 = drift_fn(t0, z)
        dw = _noise(bm, t0, t1)
        g0 = diffuse_fn(t0, z, dw)
        
        # Midpoint approximation
        z_mid = z + 0.5 * (f0 * dt + g0 * dt_sqrt)
        
        # Evaluate at midpoint
        f_mid = drift_fn(t_mid, z_mid)
        g_mid = diffuse_fn(t_mid, z_mid, dw)
        
        return z + f_mid * dt + g_mid * dt_sqrt
    
    def name(self) -> str:
        return "StochasticMidpoint"


# -----------------------------------------------------------------------------
# Adaptive Step Size Solvers
# -----------------------------------------------------------------------------

class AdaptiveEulerMaruyama(StochasticSolver):
    """Compatibility name for adaptive diagonal Ito Milstein via Diffrax.

    Naive adaptive Euler can bias an SDE through step rejection; step doubling
    therefore uses the convergent Ito Milstein method instead. Requires a
    VirtualBrownianTree and diagonal diffusion b_i(z_i,t)*eps_i. Arbitrary
    nonlinear flow-noise kernels cannot be subdivided consistently.
    """
    safety_factor: float = eqx.field(static=True)
    min_dt: float = eqx.field(static=True)
    max_dt: float = eqx.field(static=True)
    error_tol: float = eqx.field(static=True)

    def __init__(self, safety_factor=.9, min_dt=1e-6, max_dt=.1, error_tol=1e-5):
        if not (0 < min_dt <= max_dt and error_tol > 0 and 0 < safety_factor < 1):
            raise ValueError("Invalid adaptive solver settings")
        self.safety_factor = safety_factor
        self.min_dt = min_dt
        self.max_dt = max_dt
        self.error_tol = error_tol

    def step(self, z, t0, t1, drift_fn, diffuse_fn, bm):
        if not isinstance(bm, VirtualBrownianTree):
            raise ValueError("Adaptive Euler requires a subdivision-consistent VirtualBrownianTree")
        def diagonal_matrix(t, y, args):
            matrix = _gaussian_matrix(diffuse_fn, t, y)
            derivative = jax.jacfwd(lambda state: jnp.diag(
                _gaussian_matrix(diffuse_fn, t, state)))(y)
            invalid = ((jnp.max(jnp.abs(matrix-jnp.diag(jnp.diag(matrix)))) > 1e-6)
                       | (jnp.max(jnp.abs(derivative-jnp.diag(jnp.diag(derivative)))) > 1e-6))
            return eqx.error_if(matrix, invalid,
                "Adaptive Ito Milstein requires diagonal b_i(z_i,t) diffusion")
        terms = diffrax.MultiTerm(
            diffrax.ODETerm(lambda t, y, args: drift_fn(t, y)),
            diffrax.ControlTerm(diagonal_matrix, bm.path),
        )
        solution = diffrax.diffeqsolve(
            terms, diffrax.HalfSolver(diffrax.ItoMilstein()), t0=t0, t1=t1,
            dt0=jnp.minimum(self.max_dt, t1-t0), y0=z,
            stepsize_controller=diffrax.PIDController(
                rtol=self.error_tol, atol=self.error_tol, safety=self.safety_factor,
                dtmin=self.min_dt, dtmax=self.max_dt),
            saveat=diffrax.SaveAt(t1=True), max_steps=100000,
        )
        return solution.ys[0]

    def name(self):
        return "Adaptive Ito Milstein (legacy adaptive_euler alias)"


# -----------------------------------------------------------------------------
# Higher Order Solvers
# -----------------------------------------------------------------------------

class StochasticRK4(StochasticSolver):
    """
    Four-stage Runge-Kutta transition with one realized random increment.

    For nonlinear flow noise this defines a fixed-step kernel, not an SDE
    method of claimed strong order. For linear multiplicative noise it follows
    a Stratonovich convention; it must not be interchanged with Ito Euler.
    """
    
    def step(
        self,
        z: jnp.ndarray,
        t0: float,
        t1: float,
        drift_fn: Callable[[float, jnp.ndarray], jnp.ndarray],
        diffuse_fn: Callable[[float, jnp.ndarray, jnp.ndarray], jnp.ndarray],
        bm,
    ) -> jnp.ndarray:
        dt = _positive_dt(t0, t1)
        dt_sqrt = jnp.sqrt(dt)
        
        # Get Brownian increment
        dw = _noise(bm, t0, t1)
        
        # Stage 1
        f1 = drift_fn(t0, z)
        g1 = diffuse_fn(t0, z, dw)
        
        # Stage 2
        z2 = z + 0.5 * (f1 * dt + g1 * dt_sqrt)
        f2 = drift_fn(t0 + dt * 0.5, z2)
        g2 = diffuse_fn(t0 + dt * 0.5, z2, dw)
        
        # Stage 3 (uses same dW as stages 1-2 for consistency)
        z3 = z + 0.5 * (f2 * dt + g2 * dt_sqrt)
        f3 = drift_fn(t0 + dt * 0.5, z3)
        g3 = diffuse_fn(t0 + dt * 0.5, z3, dw)
        
        # Stage 4
        z4 = z + f3 * dt + g3 * dt_sqrt
        f4 = drift_fn(t1, z4)
        g4 = diffuse_fn(t1, z4, dw)
        
        # Combine stages with drift scaled by dt and diffusion by sqrt(dt).
        z_new = z + (f1 + 2 * f2 + 2 * f3 + f4) * dt / 6
        z_new = z_new + (g1 + 2 * g2 + 2 * g3 + g4) * dt_sqrt / 6
        
        return z_new
    
    def name(self) -> str:
        return "StochasticRK4"


# -----------------------------------------------------------------------------
# Solver Registry
# -----------------------------------------------------------------------------

SOLVER_REGISTRY = {
    "euler": EulerMaruyama,
    "euler_maruyama": EulerMaruyama,
    "heun": StochasticHeun,
    "stochastic_heun": StochasticHeun,
    "reversible_heun": ReversibleHeun,
    "milstein": Milstein,
    "midpoint": StochasticMidpoint,
    "stochastic_midpoint": StochasticMidpoint,
    "adaptive_euler": AdaptiveEulerMaruyama,
    "adaptive_milstein": AdaptiveEulerMaruyama,
    "rk4": StochasticRK4,
}


def get_solver(name: str, **kwargs) -> StochasticSolver:
    """
    Get a stochastic solver by name.
    
    Args:
        name: Solver name (case-insensitive)
        **kwargs: Additional arguments for solver initialization
    
    Returns:
        StochasticSolver instance
    
    Raises:
        ValueError: If solver name is not recognized
    """
    name_lower = name.lower()
    if name_lower not in SOLVER_REGISTRY:
        raise ValueError(
            f"Unknown solver '{name}'. Available: {list(SOLVER_REGISTRY.keys())}"
        )
    return SOLVER_REGISTRY[name_lower](**kwargs)


def list_solvers() -> list:
    """Return list of available solver names."""
    return list(SOLVER_REGISTRY.keys())


# -----------------------------------------------------------------------------
# Convenience Functions
# -----------------------------------------------------------------------------

@eqx.filter_jit
def integrate_sde(
    z0: jnp.ndarray,
    ts: jnp.ndarray,
    drift_fn: Callable,
    diffuse_fn: Callable,
    solver: StochasticSolver,
    bm: Optional[eqx.Module] = None,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> jnp.ndarray:
    """
    Integrate SDE over given time points.
    
    Args:
        z0: Initial state (d_x,)
        ts: Time points (T,)
        drift_fn: Drift function f(t, z) -> (d_x,)
        diffuse_fn: Diffusion function G(t, z, dW) -> (d_x,)
        solver: Stochastic solver instance
        bm: Brownian motion (if None, creates new VirtualBrownianTree)
        key: Random key for Brownian motion
    
    Returns:
        Trajectory (T, d_x)
    """
    if ts.ndim != 1 or ts.shape[0] < 2:
        raise ValueError("ts must contain at least two increasing time points")
    ts = eqx.error_if(ts, jnp.any(jnp.diff(ts) <= 0), "ts must be strictly increasing")
    if bm is None:
        bm = VirtualBrownianTree(
            t0=ts[0], 
            t1=ts[-1], 
            shape=z0.shape, 
            key=key
        )
    
    def scan_fn(carry, t_pair):
        z_prev, t_prev = carry
        t_curr = t_pair
        z_next = solver.step(z_prev, t_prev, t_curr, drift_fn, diffuse_fn, bm)
        return (z_next, t_curr), z_next
    
    # Pair consecutive time points
    t_pairs = ts[1:]
    _, zs = jax.lax.scan(scan_fn, (z0, ts[0]), t_pairs)
    
    # Prepend initial state
    return jnp.concatenate([z0[None], zs], axis=0)


@eqx.filter_jit
def integrate_sde_multiple_particles(
    z0: jnp.ndarray,
    ts: jnp.ndarray,
    drift_fn: Callable,
    diffuse_fn: Callable,
    solver: StochasticSolver,
    n_particles: int,
    key: jr.PRNGKey = jr.PRNGKey(0),
) -> jnp.ndarray:
    """
    Integrate SDE for multiple particles with independent Brownian paths.
    
    Args:
        z0: Initial state (d_x,) or (n_particles, d_x,)
        ts: Time points (T,)
        drift_fn: Drift function f(t, z) -> (d_x,) or (n_particles, d_x,)
        diffuse_fn: Diffusion function G(t, z, dW) -> (d_x,) or (n_particles, d_x,)
        solver: Stochastic solver instance
        n_particles: Number of particles
        key: Random key
    
    Returns:
        Trajectories (n_particles, T, d_x)
    """
    if not isinstance(n_particles, int) or n_particles < 1:
        raise ValueError("n_particles must be a positive Python integer")
    if z0.ndim == 1:
        initial = jnp.broadcast_to(z0, (n_particles,) + z0.shape)
    elif z0.ndim == 2 and z0.shape[0] == n_particles:
        initial = z0
    else:
        raise ValueError("z0 must be (D,) or (n_particles,D)")
    keys = jr.split(key, n_particles)
    return jax.vmap(lambda state, draw_key: integrate_sde(
        state, ts, drift_fn, diffuse_fn, solver, key=draw_key
    ))(initial, keys)
