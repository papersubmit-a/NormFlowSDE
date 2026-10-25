"""
Lorenz-96 chaotic dataset for training and evaluating AmortizedStateFlow.

The Lorenz-96 model (Lorenz 1996) is a chaotic ODE with N variables and forcing F:
  dX_i/dt = (X_{i+1} - X_{i-2}) * X_{i-1} - X_i + F   (indices mod N).

Typical: N=40, F=8 for chaotic dynamics. Observations are often every 4th state
plus Gaussian noise. Data shapes match AmortizedStateFlow: y (M,H,D_y), u (M,H,D_u),
x0 (M,D_x), x (M,H,D_x).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Callable, Dict, Optional, Tuple


# -----------------------------------------------------------------------------
# Lorenz-96 dynamics
# -----------------------------------------------------------------------------

def lorenz96_rhs(x: jnp.ndarray, F: float) -> jnp.ndarray:
    """
    Right-hand side of Lorenz-96: dX_i/dt = (X_{i+1} - X_{i-2})*X_{i-1} - X_i + F.

    x: (N,) state vector. Indices are periodic (mod N).
    Returns: (N,) derivative.
    """
    N = x.shape[0]
    # Roll for periodic BC: x_{i+1}, x_{i-1}, x_{i-2}
    x_ip1 = jnp.roll(x, -1)   # x_{i+1}
    x_im1 = jnp.roll(x,  1)   # x_{i-1}
    x_im2 = jnp.roll(x,  2)   # x_{i-2}
    return (x_ip1 - x_im2) * x_im1 - x + F


def rk4_step(
    x: jnp.ndarray,
    dt: float,
    F: float,
) -> jnp.ndarray:
    """Single RK4 step for Lorenz-96."""
    f = lorenz96_rhs
    k1 = f(x, F)
    k2 = f(x + 0.5 * dt * k1, F)
    k3 = f(x + 0.5 * dt * k2, F)
    k4 = f(x + dt * k3, F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_lorenz96(
    x0: jnp.ndarray,
    n_steps: int,
    dt: float,
    F: float,
) -> jnp.ndarray:
    """
    Integrate Lorenz-96 from x0 for n_steps with step size dt.

    x0: (N,) initial state.
    Returns: (n_steps + 1, N) trajectory (including x0).
    """
    def step(carry, _):
        x = carry
        x_next = rk4_step(x, dt, F)
        return x_next, x_next

    _, trajectory = jax.lax.scan(step, x0, None, length=n_steps)
    return jnp.concatenate([x0[None, :], trajectory], axis=0)


# -----------------------------------------------------------------------------
# Observation model
# -----------------------------------------------------------------------------

def observation_matrix_every_kth(N: int, obs_interval: int = 4) -> jnp.ndarray:
    """
    Observation matrix H: observe every obs_interval-th state.

    H has shape (D_y, N) with D_y = N // obs_interval (or ceil).
    y = H @ x, so D_y = number of observed components.
    """
    indices = jnp.arange(0, N, obs_interval)
    D_y = len(indices)
    H = jnp.zeros((D_y, N))
    H = H.at[jnp.arange(D_y), indices].set(1.0)
    return H


def observe(
    x: jnp.ndarray,
    H: jnp.ndarray,
    key: Optional[jr.PRNGKey] = None,
    obs_std: float = 0.0,
) -> jnp.ndarray:
    """
    Linear observation with optional Gaussian noise: y = H @ x + N(0, obs_std^2 I).

    x: (..., N).  H: (D_y, N).  Returns (..., D_y).
    """
    y = jnp.dot(x, H.T)  # (..., D_y)
    if obs_std > 0 and key is not None:
        y = y + obs_std * jr.normal(key, y.shape)
    return y


def _observe_batch(
    x: jnp.ndarray,
    H: jnp.ndarray,
    key: jr.PRNGKey,
    obs_std: float,
    obs_df: Optional[float] = None,
    obs_noise_type: str = "gaussian",
    gmm_weights: Optional[Tuple] = None,
    gmm_offsets: Optional[Tuple] = None,
    gmm_stds: Optional[Tuple] = None,
) -> jnp.ndarray:
    """
    Batched observe: x (M, T, N) -> y (M, T, D_y).

    Noise models (obs_noise_type):
      'gaussian'   :  y = H x + N(0, obs_std² I)
      'student_t'  :  y = H x + t(obs_df, 0, obs_std² I)  via Gamma scale mixture
      'gmm'        :  y = H x + GMM noise (multi-modal, complex)
                       gmm_weights, gmm_offsets, gmm_stds are in units of obs_std.
    """
    y = jnp.dot(x, H.T)
    if obs_std <= 0:
        return y

    if obs_noise_type == "gmm":
        # Gaussian mixture: noise ~ sum_k w_k N(mu_k, sigma_k^2)
        # Parameters are given in units of obs_std for scale-independence.
        weights = tuple(gmm_weights) if gmm_weights is not None else (0.70, 0.15, 0.15)
        offsets = tuple(gmm_offsets) if gmm_offsets is not None else (0.0,  2.5,  -2.5)
        stds    = tuple(gmm_stds)    if gmm_stds    is not None else (0.50, 1.50,  1.50)

        w_arr  = jnp.array(weights)         # (K,)
        mu_arr = jnp.array(offsets) * obs_std   # (K,) absolute offsets
        s_arr  = jnp.array(stds)   * obs_std    # (K,) absolute stds

        key_comp, key_eps = jr.split(key)
        # Categorical sampling via inverse-CDF (JIT-safe, no dynamic shapes)
        cumw = jnp.cumsum(w_arr[:-1])           # (K-1,) thresholds
        u    = jr.uniform(key_comp, y.shape)    # (M, T, D_y) in [0, 1)
        comp = jnp.sum(u[..., None] >= cumw, axis=-1)   # (M, T, D_y) int in [0, K-1]

        eps   = jr.normal(key_eps, y.shape)
        noise = mu_arr[comp] + s_arr[comp] * eps
        return y + noise

    elif obs_noise_type == "student_t" or (obs_df is not None and obs_df < 1e6):
        # Student-t via Gaussian scale mixture
        key_tau, key_eps = jr.split(key)
        alpha = obs_df / 2.0
        tau   = jr.gamma(key_tau, alpha, shape=y.shape[:-1] + (1,)) / alpha
        eps   = jr.normal(key_eps, y.shape)
        return y + obs_std * eps / jnp.sqrt(tau)

    else:
        # Gaussian
        return y + obs_std * jr.normal(key, y.shape)


# -----------------------------------------------------------------------------
# Dataset generation
# -----------------------------------------------------------------------------

def generate_lorenz96_trajectory(
    key: jr.PRNGKey,
    N: int,
    n_steps: int,
    dt: float = 0.01,
    F: float = 8.0,
    spinup_steps: int = 0,
    x0: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Generate a single Lorenz-96 trajectory.

    If x0 is None, sample initial state from N(0, small) and optionally spin up.
    Returns: (n_steps + 1, N) trajectory.
    """
    if x0 is None:
        key_init, key = jr.split(key)
        x0 = 0.01 * jr.normal(key_init, (N,))
    total = spinup_steps + n_steps
    traj = integrate_lorenz96(x0, total, dt, F)
    if spinup_steps > 0:
        traj = traj[spinup_steps:]  # (n_steps + 1, N)
    return traj


def generate_lorenz96_dataset(
    key: jr.PRNGKey,
    n_samples: int,
    horizon: int,
    N: int = 40,
    obs_interval: int = 4,
    obs_std: float = 1.0,
    obs_df: Optional[float] = None,
    obs_noise_type: str = "gaussian",
    gmm_weights: Optional[Tuple] = None,
    gmm_offsets: Optional[Tuple] = None,
    gmm_stds: Optional[Tuple] = None,
    dt: float = 0.01,
    F: float = 8.0,
    spinup_steps: int = 1000,
    trajectory_length: Optional[int] = None,
    d_u: int = 1,
    u_zeros: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Generate a dataset of (y, u, x0, x) windows.

    Parameters
    ----------
    key : PRNG key
    n_samples : number of windows (batch M)
    horizon : H steps per window
    N : Lorenz-96 state dimension
    obs_interval : observe every obs_interval-th state → D_y = N // obs_interval
    obs_std : observation noise scale
    obs_df : Student-t degrees of freedom (None = Gaussian)
    obs_noise_type : 'gaussian' | 'student_t' | 'gmm'
    gmm_weights : GMM component weights (K-tuple, sum=1); units = probability
    gmm_offsets : GMM component means in units of obs_std (K-tuple)
    gmm_stds    : GMM component stds  in units of obs_std (K-tuple)
    dt, F, spinup_steps : Lorenz-96 integration parameters
    trajectory_length : if set, generate one long trajectory and slice windows
    d_u : control dimension

    Returns
    -------
    dict: "y" (M,H,D_y), "u" (M,H,d_u), "x0" (M,N), "x" (M,H,N), "H" (D_y,N)
    """
    D_y   = N // obs_interval
    H_obs = observation_matrix_every_kth(N, obs_interval)

    _noise_kwargs = dict(
        obs_noise_type=obs_noise_type,
        gmm_weights=gmm_weights,
        gmm_offsets=gmm_offsets,
        gmm_stds=gmm_stds,
    )

    if trajectory_length is not None and trajectory_length >= spinup_steps + n_samples * horizon:
        key_traj, key_noise = jr.split(key)
        traj = generate_lorenz96_trajectory(
            key_traj, N, trajectory_length, dt, F, spinup_steps=spinup_steps,
        )
        starts   = jnp.arange(n_samples) * horizon
        step_idx = starts[:, None] + jnp.arange(1, horizon + 1)[None, :]
        x0_arr   = traj[starts]
        x_arr    = traj[step_idx]
        y_arr    = _observe_batch(x_arr, H_obs, key_noise, obs_std, obs_df, **_noise_kwargs)
        u_arr    = jnp.zeros((n_samples, horizon, d_u))
        return {"y": y_arr, "u": u_arr, "x0": x0_arr, "x": x_arr, "H": H_obs}

    # Independent trajectories via vmap
    keys      = jr.split(key, n_samples)
    key_noise = jr.fold_in(key, 1)

    def one_trajectory(k):
        traj = generate_lorenz96_trajectory(k, N, horizon, dt, F, spinup_steps=spinup_steps)
        return traj[0], traj[1 : horizon + 1]

    x0_arr, x_arr = jax.vmap(one_trajectory)(keys)
    y_arr = _observe_batch(x_arr, H_obs, key_noise, obs_std, obs_df, **_noise_kwargs)
    u_arr = jnp.zeros((n_samples, horizon, d_u))

    return {"y": y_arr, "u": u_arr, "x0": x0_arr, "x": x_arr, "H": H_obs}


def jit_generate_lorenz96_dataset(
    n_samples: int,
    horizon: int,
    N: int = 40,
    obs_interval: int = 4,
    obs_std: float = 1.0,
    dt: float = 0.01,
    F: float = 8.0,
    spinup_steps: int = 1000,
    trajectory_length: Optional[int] = None,
    d_u: int = 1,
) -> Callable[[jr.PRNGKey], Dict[str, jnp.ndarray]]:
    """
    Return a JIT-compiled dataset generator with fixed config.

    First call compiles; subsequent calls with different keys are fast.
    Use when generating many batches with the same shape (e.g. training loop).
    """
    def _generator(key: jr.PRNGKey) -> Dict[str, jnp.ndarray]:
        return generate_lorenz96_dataset(
            key, n_samples, horizon, N=N, obs_interval=obs_interval,
            obs_std=obs_std, dt=dt, F=F, spinup_steps=spinup_steps,
            trajectory_length=trajectory_length, d_u=d_u,
        )
    return jax.jit(_generator)


def segment_sequences(
    data: Dict[str, jnp.ndarray],
    seg_len: int,
    forecast_horizon: int,
) -> Dict[str, jnp.ndarray]:
    """
    Split (M, T, ...) sequences into non-overlapping (context, forecast) segments.

    Each segment covers seg_len + forecast_horizon consecutive steps:
      y_ctx  : (seg_len, D_y)         observation context  → encoder input
      u_ctx  : (seg_len, d_u)         control context
      x0     : (N,)                   first state of context  → encoder anchor
      x_last : (N,)                   last state of context   → flow initialisation
      x_fct  : (forecast_horizon, N)  future states           → training target

    Returns a dict with leading dimension M * n_seg (all segments from all
    sequences concatenated), ready to be shuffled and batched.
    """
    x = data["x"]   # (M, T, N)
    y = data["y"]   # (M, T, D_y)
    u = data["u"]   # (M, T, d_u)
    M, T = x.shape[:2]

    seg_total = seg_len + forecast_horizon
    n_seg     = T // seg_total
    assert n_seg > 0, (
        f"Sequence length {T} is too short for "
        f"seg_len={seg_len} + forecast_horizon={forecast_horizon}"
    )

    # Vectorised indexing — no Python loop over segments
    seg_starts = jnp.arange(n_seg) * seg_total          # (n_seg,)
    ctx_idx    = seg_starts[:, None] + jnp.arange(seg_len)            # (n_seg, W)
    fct_idx    = seg_starts[:, None] + seg_len + jnp.arange(forecast_horizon)  # (n_seg, H)

    # Index into each sequence: shapes (M, n_seg, ...)
    y_ctx  = y[:, ctx_idx]                    # (M, n_seg, W, D_y)
    u_ctx  = u[:, ctx_idx]                    # (M, n_seg, W, d_u)
    x0     = x[:, seg_starts]                 # (M, n_seg, N)
    x_last = x[:, seg_starts + seg_len - 1]   # (M, n_seg, N)
    x_fct  = x[:, fct_idx]                    # (M, n_seg, H, N)

    N_out = M * n_seg
    return {
        "y_ctx":  y_ctx.reshape(N_out, seg_len, -1),
        "u_ctx":  u_ctx.reshape(N_out, seg_len, -1),
        "x0":     x0.reshape(N_out, -1),
        "x_last": x_last.reshape(N_out, -1),
        "x_fct":  x_fct.reshape(N_out, forecast_horizon, -1),
    }


def train_val_test_split(
    data: Dict[str, jnp.ndarray],
    key: jr.PRNGKey,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    Split dataset into train / val / test by random permutation.

    data: dict with arrays having same leading dimension M.
    Returns (train_data, val_data, test_data) each with same keys.
    """
    M = data["y"].shape[0]
    perm = jr.permutation(key, M)
    n_train = int(M * train_frac)
    n_val = int(M * val_frac)
    n_test = M - n_train - n_val

    def split(arr):
        arr = arr[perm]
        return arr[:n_train], arr[n_train : n_train + n_val], arr[n_train + n_val :]

    train_data = {}
    val_data = {}
    test_data = {}
    for k, v in data.items():
        if isinstance(v, jnp.ndarray) and v.shape[0] == M:
            t, va, te = split(v)
            train_data[k] = t
            val_data[k] = va
            test_data[k] = te
        else:
            train_data[k] = v
            val_data[k] = v
            test_data[k] = v
    return train_data, val_data, test_data


# -----------------------------------------------------------------------------
# Evaluation metrics (for AmortizedStateFlow samples)
# -----------------------------------------------------------------------------

def state_rmse(x_true: jnp.ndarray, x_pred: jnp.ndarray) -> jnp.ndarray:
    """
    Per-sample RMSE between true and predicted state sequences.

    x_true: (M, H, D_x).  x_pred: (M, H, D_x) or (S, M, H, D_x) (posterior samples).
    If x_pred is (S, M, H, D_x), returns RMSE of mean over S vs x_true, shape (M,) or scalar.
    """
    if x_pred.ndim == 4:
        x_pred = jnp.mean(x_pred, axis=0)
    diff = x_true - x_pred
    mse = jnp.mean(diff ** 2, axis=(-2, -1))
    return jnp.sqrt(mse)


def state_mae(x_true: jnp.ndarray, x_pred: jnp.ndarray) -> jnp.ndarray:
    """Per-sample MAE. Same shape convention as state_rmse."""
    if x_pred.ndim == 4:
        x_pred = jnp.mean(x_pred, axis=0)
    return jnp.mean(jnp.abs(x_true - x_pred), axis=(-2, -1))


# -----------------------------------------------------------------------------
# Example: generate data and run one training / eval step
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    from bayesinf.applications.state_estimation.amortized_state_flow import (
        AmortizedStateFlow,
        state_flow_loss,
    )

    key = jr.PRNGKey(0)
    N, horizon = 8, 6  # small for quick check
    obs_interval = 2
    D_y = N // obs_interval
    d_u = 1

    key, k1, k2 = jr.split(key, 3)
    data = generate_lorenz96_dataset(
        k1, n_samples=20, horizon=horizon, N=N,
        obs_interval=obs_interval, obs_std=0.5, dt=0.02, F=8.0,
        spinup_steps=200, d_u=d_u,
    )
    train_data, val_data, test_data = train_val_test_split(data, k2, train_frac=0.6, val_frac=0.2)

    d_x = N
    d_enc, d_out = 16, 24
    key, k_model = jr.split(key)
    model = AmortizedStateFlow(
        d_x=d_x, d_y=D_y, d_u=d_u, d_enc=d_enc, d_out=d_out, key=k_model,
        flow_type="realnvp", n_coupling=2, hidden_size=32, n_hidden=2,
        encoder_type="gru", gru_hidden=16,
    )

    x, y, u, x0 = train_data["x"], train_data["y"], train_data["u"], train_data["x0"]
    loss = state_flow_loss(model, x, y, u, x0)
    print(f"Train NLL loss (one batch): {float(loss):.4f}")

    key, k_sample = jr.split(key)
    samples = model.sample(y, u, x0, key=k_sample, num_samples=5, prior="wiener_scaled_t")
    rmse = state_rmse(x, samples)
    print(f"State RMSE (posterior mean vs true): {float(jnp.mean(rmse)):.4f}")
    print("Lorenz-96 dataset and AmortizedStateFlow pipeline OK.")
