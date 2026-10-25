"""Conditional fixed-grid flow-noise trajectory model.

At the trained physical step h, x[i+1]=x[i]+h*f(x[i],c[i],t[i])
+sqrt(h)*G(eps[i];x[i],c[i],t[i]), with independent standard Gaussian eps.
A nonlinear or noncentered G defines a discrete transition law, not generally
an Ito SDE with a grid-refinement limit. The observation encoder sees only a
bounded window through observation i when conditioning step i. Its two GRU
directions operate inside that past window, never on future observations.
``paper_loss`` implements the manuscript's reconstruction, sampled transition
entropy, shooting consistency, and full-horizon energy objective. The legacy
transition-NLL and ensemble objectives remain explicitly selectable.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


def energy_score(
    preds: jnp.ndarray,
    target: jnp.ndarray,
    beta: float = 1.0,
    uncertainty_weight: float = 0.5,
    eps: float = 0.0,
) -> jnp.ndarray:
    """Unbiased ensemble energy score of the *joint* trajectory distribution.

    The time and state axes form one vector, so temporal dependence is scored.
    The proper-score coefficient is 1/2; another uncertainty_weight explicitly
    defines a heuristic spread penalty. At least two independent draws are
    needed to estimate the pair expectation without self-pair bias.
    """
    if preds.shape[0] < 2:
        raise ValueError("Energy scoring requires at least two ensemble members")
    if not 0 < beta < 2:
        raise ValueError("energy beta must be in (0,2)")
    flat = preds.reshape((preds.shape[0], -1))
    truth = target.reshape((-1,))
    if eps < 0:
        raise ValueError("energy norm smoothing eps must be nonnegative")

    def norm(delta):
        squared = jnp.sum(delta**2, axis=-1)
        # Exact unsquared norms are nondifferentiable at zero. Select the zero
        # subgradient without ever differentiating sqrt(0), including self-pairs.
        positive = squared + eps > 0
        safe = jnp.where(positive, squared + eps, 1.0)
        return jnp.where(positive, safe ** (beta / 2), 0.0) - eps ** (beta / 2)
    accuracy = jnp.mean(norm(flat - truth))
    pair = norm(flat[:, None] - flat[None, :])
    n = preds.shape[0]
    spread = jnp.sum(pair) / (n * (n - 1))
    return accuracy - uncertainty_weight * spread


def gaussian_nll(preds: jnp.ndarray, target: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Diagonal Gaussian NLL fitted to an ensemble of paths."""
    mu = jnp.mean(preds, axis=0)
    var = jnp.var(preds, axis=0) + eps
    nll = 0.5 * ((target - mu) ** 2) / var + 0.5 * jnp.log(2.0 * jnp.pi * var)
    return jnp.mean(nll)


from bayesinf.applications.state_estimation.stochastic_solvers import (
    StepNoisePath, get_solver as get_stochastic_solver,
)


class SinusoidalEmb(eqx.Module):
    """Sinusoidal positional embedding for a scalar time value -> (dim,)."""
    dim: int = eqx.field(static=True)

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        half  = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / float(max(1, half - 1)))
        args  = jnp.asarray(t, dtype=jnp.float32) * freqs
        emb   = jnp.concatenate([jnp.sin(args), jnp.cos(args)])
        if self.dim % 2 == 1:
            emb = jnp.concatenate([emb, jnp.zeros(1)])
        return emb


# ---------------------------------------------------------------------------
# Bidirectional GRU encoder  y_{1:T} -> c_{1:T}
# ---------------------------------------------------------------------------

class BiGRUEncoder(eqx.Module):
    """
    Bidirectional GRU encoder producing per-step context vectors.

    Forward and backward hidden states are concatenated then projected.
    ``__call__`` is the generic sequence encoder. NormFlowSDE instead calls
    ``window_context`` independently on each preceding observation window.

    Input : y  (T, d_y)
    Output: c  (T, d_out)
    """
    fwd_cell: eqx.nn.GRUCell
    bwd_cell: eqx.nn.GRUCell
    proj:     eqx.nn.Linear

    def __init__(self, d_y: int, d_hidden: int, d_out: int, key: jr.PRNGKey):
        k1, k2, k3 = jr.split(key, 3)
        self.fwd_cell = eqx.nn.GRUCell(d_y, d_hidden, key=k1)
        self.bwd_cell = eqx.nn.GRUCell(d_y, d_hidden, key=k2)
        self.proj     = eqx.nn.Linear(2 * d_hidden, d_out, key=k3)

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        d_h = self.fwd_cell.hidden_size
        h0  = jnp.zeros(d_h)

        def fwd_step(h, y_t):
            h_new = self.fwd_cell(y_t, h)
            return h_new, h_new

        def bwd_step(h, y_t):
            h_new = self.bwd_cell(y_t, h)
            return h_new, h_new

        _, h_fwd = jax.lax.scan(fwd_step, h0, y)           # (T, d_h)
        _, h_bwd = jax.lax.scan(bwd_step, h0, y[::-1])     # (T, d_h), reversed
        h_bwd    = h_bwd[::-1]                              # restore time order

        h_bi = jnp.concatenate([h_fwd, h_bwd], axis=-1)    # (T, 2*d_h)
        return jax.vmap(self.proj)(h_bi)                    # (T, d_out)

    def window_context(self, y: jnp.ndarray, valid: jnp.ndarray) -> jnp.ndarray:
        """Encode one past window, skipping unavailable entries near its start.

        Both final GRU states summarize the same available window. Masked
        padding never updates either hidden state, so a short prefix is exactly
        equivalent to encoding that prefix alone.
        """
        h0 = jnp.zeros(self.fwd_cell.hidden_size, dtype=y.dtype)

        def scan(cell, values, mask):
            def step(hidden, item):
                value, available = item
                updated = cell(value, hidden)
                hidden = jnp.where(available, updated, hidden)
                return hidden, None
            hidden, _ = jax.lax.scan(step, h0, (values, mask))
            return hidden

        forward = scan(self.fwd_cell, y, valid)
        backward = scan(self.bwd_cell, y[::-1], valid[::-1])
        return self.proj(jnp.concatenate((forward, backward)))


# ---------------------------------------------------------------------------
# Drift network  f_θ(z_t, c_t, t)
# ---------------------------------------------------------------------------

class SDEDriftNet(eqx.Module):
    """
    Parameterises the SDE drift  f_θ(z_t, c_t, t) → (d_x,).
    Input:  z_t  (d_x,)  |  c_t  (d_c,)  |  t  scalar physical time
    """
    net:      eqx.nn.MLP
    time_emb: SinusoidalEmb
    use_time: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        depth:        int,
        time_emb_dim: int,
        key:          jr.PRNGKey,
        output_dim:    int | None = None,
        use_time:      bool = True,
    ):
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.use_time = bool(use_time)
        self.net = eqx.nn.MLP(
            in_size    = d_x + d_c + time_emb_dim,
            out_size   = d_x if output_dim is None else output_dim,
            width_size = d_hidden,
            depth      = depth,
            activation = jax.nn.gelu,
            key        = key,
        )

    def __call__(self, z: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # use_time=False feeds the constant embedding of t=0: time-invariant, same parameter count.
        t_emb = self.time_emb(t if self.use_time else jnp.zeros_like(t))
        return self.net(jnp.concatenate([z, c, t_emb]))


# ---------------------------------------------------------------------------
# RealNVP transition-noise network  T_φ(eps_t; z_t, c_t, t)
# ---------------------------------------------------------------------------

class RealNVPDiffusionLayer(eqx.Module):
    """
    Conditional affine coupling layer. The transformed half is
    y_b = x_b * exp(alpha * tanh(s_raw)) + shift.
    """
    net:     eqx.nn.MLP
    time_emb: SinusoidalEmb
    d_x:     int  = eqx.field(static=True)
    d_half:  int  = eqx.field(static=True)
    d_pass:  int  = eqx.field(static=True)
    d_shift: int  = eqx.field(static=True)
    flip:    bool = eqx.field(static=True)
    alpha:   float = eqx.field(static=True)
    use_time: bool = eqx.field(static=True, default=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        time_emb_dim: int,
        flip:         bool,
        alpha:        float,
        key:          jr.PRNGKey,
        use_time:     bool = True,
    ):
        self.d_x    = d_x
        self.d_half = d_x // 2
        self.flip   = flip
        self.alpha  = alpha
        self.use_time = bool(use_time)
        if flip:
            self.d_pass  = d_x - d_x // 2   # lower half of state
            self.d_shift = d_x // 2          # upper half
        else:
            self.d_pass  = d_x // 2          # upper half of state
            self.d_shift = d_x - d_x // 2    # lower half
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size    = self.d_pass + d_x + d_c + time_emb_dim,
            out_size   = 2 * self.d_shift,
            width_size = d_hidden,
            depth      = 2,
            activation = jax.nn.tanh,
            key        = key,
        )

    def _params(
        self, x_pass: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        t_emb  = self.time_emb(t if self.use_time else jnp.zeros_like(t))
        raw = self.net(jnp.concatenate([x_pass, z_state, c, t_emb]))
        shift, raw_scale = jnp.split(raw, 2)
        return shift, self.alpha * jnp.tanh(raw_scale)

    def apply(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Forward affine coupling transform."""
        if self.flip:
            eps_pass  = eps[self.d_half:]
            eps_shift = eps[:self.d_half]
            shift, log_scale = self._params(eps_pass, z_state, c, t)
            return jnp.concatenate([eps_shift * jnp.exp(log_scale) + shift, eps_pass])
        else:
            eps_pass  = eps[:self.d_half]
            eps_shift = eps[self.d_half:]
            shift, log_scale = self._params(eps_pass, z_state, c, t)
            return jnp.concatenate([eps_pass, eps_shift * jnp.exp(log_scale) + shift])

    def apply_inverse(self, v: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Inverse affine coupling transform."""
        if self.flip:
            v_pass  = v[self.d_half:]
            v_shift = v[:self.d_half]
            shift, log_scale = self._params(v_pass, z_state, c, t)
            return jnp.concatenate([(v_shift - shift) * jnp.exp(-log_scale), v_pass])
        else:
            v_pass  = v[:self.d_half]
            v_shift = v[self.d_half:]
            shift, log_scale = self._params(v_pass, z_state, c, t)
            return jnp.concatenate([v_pass, (v_shift - shift) * jnp.exp(-log_scale)])

    def forward_log_det(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        eps_pass = eps[self.d_half:] if self.flip else eps[:self.d_half]
        _, log_scale = self._params(eps_pass, z_state, c, t)
        return jnp.sum(log_scale)


class RealNVPDiffusionNet(eqx.Module):
    """
    Conditional RealNVP transform for standard-Gaussian transition noise.
    """
    layers:        tuple
    d_x:   int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_c:          int,
        d_hidden:     int,
        n_layers:     int,
        time_emb_dim: int,
        key:          jr.PRNGKey,
        alpha:         float = 1.0,
        use_time:      bool = True,
    ):
        if alpha <= 0:
            raise ValueError(f"RealNVP alpha must be positive; got {alpha}.")
        self.d_x = d_x
        self.alpha = alpha
        keys = jr.split(key, n_layers)
        self.layers = tuple(
            RealNVPDiffusionLayer(
                d_x, d_c, d_hidden, time_emb_dim, flip=(i % 2 == 1),
                alpha=alpha, key=keys[i], use_time=use_time,
            )
            for i in range(n_layers)
        )

    def apply(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """T_φ(eps; z_t, c_t, t)."""
        for layer in self.layers:
            eps = layer.apply(eps, z_state, c, t)
        return eps

    def apply_inverse(self, v: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """T_φ^{-1}(v; z_t, c_t, t)."""
        for layer in reversed(self.layers):
            v = layer.apply_inverse(v, z_state, c, t)
        return v

    def inverse_and_log_det(self, v, z_state, c, t):
        log_det = jnp.array(0.0, dtype=v.dtype)
        current = v
        for layer in reversed(self.layers):
            eps = layer.apply_inverse(current, z_state, c, t)
            log_det += layer.forward_log_det(eps, z_state, c, t)  # ← minus
            current = eps
        return current, log_det

    def log_det(self, eps: jnp.ndarray, z_state: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        log_det = jnp.array(0.0, dtype=eps.dtype)
        current = eps
        for layer in self.layers:
            log_det = log_det + layer.forward_log_det(current, z_state, c, t)
            current = layer.apply(current, z_state, c, t)
        return log_det


LEARNED_LOG_SCALE_BOUNDS = (-10.0, 3.0)


class LearnedScaleDiffusionNet(eqx.Module):
    """Per-coordinate scale in front of a RealNVP noise flow: v = sigma(z, c, t) * T(eps; z, c, t).

    log sigma is smoothly bounded to LEARNED_LOG_SCALE_BOUNDS and is 0 at initialization
    (zero output layer), so training starts from the unscaled flow. The coupling log-scales
    stay bounded by +-alpha; this Gaussian stage frees the flow from also setting the noise
    magnitude. Same transform as NormFlowSDEDynamics(learned_scale=True) in
    experiments/normflowsde_comparison/learned_dynamics_pf.py.
    """
    flow:     RealNVPDiffusionNet
    scale:    eqx.nn.MLP
    time_emb: SinusoidalEmb
    d_x:      int = eqx.field(static=True)
    alpha:    float = eqx.field(static=True)
    use_time: bool = eqx.field(static=True, default=True)

    def __init__(self, flow, d_c, d_hidden, time_emb_dim, key, use_time=True):
        self.flow = flow
        self.d_x = flow.d_x
        self.alpha = flow.alpha
        self.use_time = bool(use_time)
        self.time_emb = SinusoidalEmb(time_emb_dim)
        scale = eqx.nn.MLP(self.d_x + d_c + time_emb_dim, self.d_x, d_hidden, 2,
                           activation=jax.nn.gelu, key=key)
        last = scale.layers[-1]
        self.scale = eqx.tree_at(lambda m: (m.layers[-1].weight, m.layers[-1].bias), scale,
                                 (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))

    @property
    def layers(self):
        return self.flow.layers

    def log_scale(self, z_state, c, t):
        low, high = LEARNED_LOG_SCALE_BOUNDS
        offset = jnp.log(-low / high)  # sigmoid(offset) = -low / (high - low): log sigma = 0 at a zero output
        t_emb = self.time_emb(t if self.use_time else jnp.zeros_like(t))
        raw = self.scale(jnp.concatenate([z_state, c, t_emb]))
        return low + (high - low) * jax.nn.sigmoid(raw + offset)

    def apply(self, eps, z_state, c, t):
        return jnp.exp(self.log_scale(z_state, c, t)) * self.flow.apply(eps, z_state, c, t)

    def apply_inverse(self, v, z_state, c, t):
        return self.flow.apply_inverse(v * jnp.exp(-self.log_scale(z_state, c, t)), z_state, c, t)

    def inverse_and_log_det(self, v, z_state, c, t):
        log_scale = self.log_scale(z_state, c, t)
        eps, log_det = self.flow.inverse_and_log_det(v * jnp.exp(-log_scale), z_state, c, t)
        return eps, log_det + jnp.sum(log_scale)

    def log_det(self, eps, z_state, c, t):
        return self.flow.log_det(eps, z_state, c, t) + jnp.sum(self.log_scale(z_state, c, t))


class NICEDiffusionLayer(eqx.Module):
    """Conditional additive coupling with unit Jacobian determinant."""
    net: eqx.nn.MLP
    time_emb: SinusoidalEmb
    d_x: int = eqx.field(static=True)
    d_half: int = eqx.field(static=True)
    flip: bool = eqx.field(static=True)

    def __init__(self, d_x, d_c, d_hidden, time_emb_dim, flip, key):
        self.d_x = d_x
        self.d_half = d_x // 2
        self.flip = flip
        d_pass = d_x - self.d_half if flip else self.d_half
        d_shift = self.d_half if flip else d_x - self.d_half
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            in_size=d_pass + d_x + d_c + time_emb_dim,
            out_size=d_shift,
            width_size=d_hidden,
            depth=2,
            activation=jax.nn.tanh,
            key=key,
        )

    def _shift(self, x_pass, z_state, c, t):
        return self.net(jnp.concatenate([x_pass, z_state, c, self.time_emb(t)]))

    def apply(self, x, z_state, c, t):
        if self.flip:
            x_pass, x_shift = x[self.d_half:], x[:self.d_half]
            return jnp.concatenate([x_shift + self._shift(x_pass, z_state, c, t), x_pass])
        x_pass, x_shift = x[:self.d_half], x[self.d_half:]
        return jnp.concatenate([x_pass, x_shift + self._shift(x_pass, z_state, c, t)])

    def apply_inverse(self, y, z_state, c, t):
        if self.flip:
            y_pass, y_shift = y[self.d_half:], y[:self.d_half]
            return jnp.concatenate([y_shift - self._shift(y_pass, z_state, c, t), y_pass])
        y_pass, y_shift = y[:self.d_half], y[self.d_half:]
        return jnp.concatenate([y_pass, y_shift - self._shift(y_pass, z_state, c, t)])


class NICEDiffusionNet(eqx.Module):
    """Volume-preserving NICE transform with a learnable global scale."""
    layers: tuple
    log_noise_scale: jnp.ndarray
    d_x: int = eqx.field(static=True)

    def __init__(self, d_x, d_c, d_hidden, n_layers, time_emb_dim, key,
                 noise_scale=0.5):
        if noise_scale <= 0:
            raise ValueError(f"NICE noise_scale must be positive; got {noise_scale}.")
        self.d_x = d_x
        self.log_noise_scale = jnp.asarray(jnp.log(noise_scale), dtype=jnp.float32)
        keys = jr.split(key, n_layers)
        self.layers = tuple(
            NICEDiffusionLayer(
                d_x, d_c, d_hidden, time_emb_dim, i % 2 == 1, keys[i],
            )
            for i in range(n_layers)
        )

    def noise_scale(self):
        return jnp.exp(self.log_noise_scale)

    def apply(self, eps, z_state, c, t):
        value = eps
        for layer in self.layers:
            value = layer.apply(value, z_state, c, t)
        return self.noise_scale() * value

    def apply_inverse(self, v, z_state, c, t):
        value = v / self.noise_scale()
        for layer in reversed(self.layers):
            value = layer.apply_inverse(value, z_state, c, t)
        return value

    def inverse_and_log_det(self, v, z_state, c, t):
        eps = self.apply_inverse(v, z_state, c, t)
        log_det = self.d_x * self.log_noise_scale.astype(v.dtype)
        return eps, log_det

    def log_det(self, eps, z_state, c, t):
        del z_state, c, t
        return self.d_x * self.log_noise_scale.astype(eps.dtype)

class RealNVPFlowSDE(eqx.Module):
    """Conditional transition-noise model on a fixed physical grid.

    Euler has an exact transition density. Heun/midpoint/RK4 are alternative
    discrete kernels whose scores must be learned with the selected sampler.
    Their classical SDE orders do not apply to arbitrary nonlinear flow noise.
    ``reversible_heun`` is a compatibility name for the same Heun kernel.
    """
    encoder:    BiGRUEncoder
    drift:      SDEDriftNet
    diffusion:  RealNVPDiffusionNet | LearnedScaleDiffusionNet | NICEDiffusionNet
    d_x:        int    = eqx.field(static=True)
    d_y:        int    = eqx.field(static=True)
    d_u:        int    = eqx.field(static=True)
    exclude_control_from_diffusion: bool = eqx.field(static=True)
    physics_fn: object = eqx.field(static=True, default=None)
    flow_type:  str = eqx.field(static=True, default="realnvp")
    observation_stride: int = eqx.field(static=True, default=1)
    observation_window: int = eqx.field(static=True, default=1)
    direct_control: bool = eqx.field(static=True, default=False)
    time_step: float = eqx.field(static=True, default=1.0)

    def __init__(
        self,
        d_x:           int,
        d_y:           int,
        d_u:           int   = 0,
        exclude_control_from_diffusion: bool = False,
        d_c:           int   = 64,
        d_hidden_enc:  int   = 64,
        d_hidden:      int   = 128,
        d_hidden_diffusion: int = 64,
        depth:         int   = 3,
        time_emb_dim:  int   = 32,
        n_realnvp_layers: int = 2,
        realnvp_alpha: float = 1.0,
        flow_type:      str = "realnvp",
        n_nice_layers:  int = 2,
        nice_noise_scale: float = 0.5,
        observation_stride: int = 1,
        time_step:      float = 1.0,
        physics_fn            = None,
        key:           jr.PRNGKey = jr.PRNGKey(0),
        observation_window: int = 1,
        direct_control: bool = False,
        use_time:       bool = True,
        learned_scale:  bool = False,
    ):
        """``use_time=False`` makes drift and noise flow time-invariant (constant time
        embedding; parameter count unchanged). ``learned_scale=True`` adds a learned
        per-coordinate noise scale in front of the RealNVP flow (extra parameters,
        initialized to sigma = 1; default models keep their exact initialization)."""
        self.d_x        = d_x
        self.d_y        = d_y
        self.d_u        = d_u
        self.exclude_control_from_diffusion = exclude_control_from_diffusion
        if d_u < 0 or d_u > d_y:
            raise ValueError(f"d_u must be between 0 and d_y; got d_u={d_u}, d_y={d_y}.")
        self.physics_fn = physics_fn
        self.direct_control = direct_control
        if direct_control and d_u == d_y:
            raise ValueError("direct_control requires at least one observation channel")
        self.flow_type = flow_type.lower()
        if observation_stride < 1:
            raise ValueError("observation_stride must be at least 1.")
        if observation_window < 1:
            raise ValueError("observation_window must be at least 1.")
        if time_step <= 0:
            raise ValueError("time_step must be positive.")
        self.observation_stride = observation_stride
        self.observation_window = observation_window
        self.time_step = float(time_step)
        k_enc, k_drift, k_diff = jr.split(key, 3)
        encoder_dim = d_y - d_u if direct_control else d_y
        context_dim = d_c + d_u if direct_control else d_c
        self.encoder   = BiGRUEncoder(encoder_dim, d_hidden_enc, d_c, k_enc)
        self.drift     = SDEDriftNet(d_x, context_dim, d_hidden, depth, time_emb_dim, k_drift,
                                     use_time=use_time)
        if learned_scale and self.flow_type != "realnvp":
            raise ValueError("learned_scale requires flow_type='realnvp'")
        if not use_time and self.flow_type != "realnvp":
            raise ValueError("use_time=False requires flow_type='realnvp'")
        if self.flow_type == "realnvp":
            self.diffusion = RealNVPDiffusionNet(
                d_x, context_dim, d_hidden_diffusion, n_realnvp_layers, time_emb_dim, k_diff,
                alpha=realnvp_alpha, use_time=use_time,
            )
            if learned_scale:
                # Separate key: the drift/flow initialization of default models is unchanged.
                self.diffusion = LearnedScaleDiffusionNet(
                    self.diffusion, context_dim, d_hidden_diffusion, time_emb_dim,
                    jr.fold_in(key, 3), use_time=use_time,
                )
        elif self.flow_type == "nice":
            self.diffusion = NICEDiffusionNet(
                d_x, context_dim, d_hidden_diffusion, n_nice_layers, time_emb_dim, k_diff,
                noise_scale=nice_noise_scale,
            )
        else:
            raise ValueError(f"Unknown flow_type '{flow_type}'; choose 'realnvp' or 'nice'.")

    def encode(self, y: jnp.ndarray) -> jnp.ndarray:
        """Encode c[k] from y[max(0,k-w+1):k+1], with no future access.

        With ``direct_control``, trailing control channels are excluded from
        the GRU and the current u[k] is appended to the encoded observation.
        """
        if y.ndim != 2 or y.shape[1] != self.d_y or y.shape[0] < 1:
            raise ValueError(f"Expected a nonempty observation array (T,{self.d_y})")
        observations = y[:, :self.d_y - self.d_u] if self.direct_control and self.d_u else y
        width = min(self.observation_window, y.shape[0])

        def one_window(index):
            indices = index - jnp.arange(width - 1, -1, -1)
            valid = indices >= 0
            window = observations[jnp.maximum(indices, 0)]
            if isinstance(self.encoder, BiGRUEncoder):
                return self.encoder.window_context(window, valid)
            # Permit lightweight pointwise encoders used in analytic checks.
            return self.encoder(jnp.where(valid[:, None], window, 0.0))[-1]

        context = jax.vmap(one_window)(jnp.arange(y.shape[0]))
        if self.direct_control and self.d_u:
            context = jnp.concatenate((context, y[:, -self.d_u:]), axis=-1)
        return context

    def encode_observation_stride(self, y: jnp.ndarray, stride: int) -> jnp.ndarray:
        """Encode observations every ``stride`` steps and hold context between them."""
        if stride < 1:
            raise ValueError(f"observation stride must be >= 1; got {stride}.")
        sparse_context = self.encode(y[::stride])
        context_idx = jnp.minimum(
            jnp.arange(y.shape[0]) // stride, sparse_context.shape[0] - 1
        )
        held = sparse_context[context_idx]
        if self.direct_control and self.d_u:
            # Controls remain current even when observation updates are sparse.
            held = jnp.concatenate((held[:, :-self.d_u], y[:, -self.d_u:]), axis=-1)
        return held

    def encode_diffusion_stride(self, y: jnp.ndarray, stride: int) -> jnp.ndarray:
        """Encode stochastic context with control channels masked when requested."""
        if self.exclude_control_from_diffusion and self.d_u:
            y = y.at[..., -self.d_u:].set(0.0)
        return self.encode_observation_stride(y, stride)

    def transition_nll(self, x_seg, c_drift_seg, c_diffusion_seg, t_start, dt):
        times = t_start + jnp.arange(x_seg.shape[0] - 1) * dt

        def one_step(x0, x1, c_drift0, c_diffusion0, t0):
            drift = self.drift(x0, c_drift0, t0)
            if self.physics_fn is not None:
                drift = drift + self.physics_fn(x0)
            normalized_residual = (x1 - x0 - drift * dt) / jnp.sqrt(dt)
            eps, log_det = self.diffusion.inverse_and_log_det(
                normalized_residual, x0, c_diffusion0, t0
            )
            base_nll = 0.5 * jnp.sum(eps**2) + 0.5 * self.d_x * jnp.log(2.0 * jnp.pi)
            time_log_det = 0.5 * self.d_x * jnp.log(dt)
            # The inverse routine returns the forward transform log determinant.
            return (base_nll + log_det + time_log_det) / self.d_x

        losses = jax.vmap(one_step)(
            x_seg[:-1], x_seg[1:], c_drift_seg[:x_seg.shape[0] - 1], c_diffusion_seg[:x_seg.shape[0] - 1], times
        )
        return jnp.mean(losses)

    # ------------------------------------------------------------------
    # Pathwise training:  integrate SDE → minimise reconstruction loss
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_training_path(x, y):
        if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] + 1:
            raise ValueError(
                "Training requires x=(T+1,D) including x0 and y=(T,O) "
                "containing the observations for the T transitions."
            )
        if y.shape[0] < 1:
            raise ValueError("At least one transition is required")

    def paper_loss_components_single(
        self, x, y, key, *, seg_length, K=8, state_sigma=1.0,
        include_rollout=True, rollout_K=None,
    ):
        """Monte Carlo estimates of the four manuscript objective terms.

        ``seg_length`` counts state points, including each supplied anchor.
        All transitions, including a partial final segment, are scored once.
        Independent segment rollouts retain the global observation window and
        physical clock. Norms and log densities are summed over time and state
        coordinates, then averaged over draws, as in method.tex.
        """
        self._validate_training_path(x, y)
        if seg_length < 2:
            raise ValueError("Manuscript seg_length counts state points and must be >= 2")
        M = K if rollout_K is None else rollout_K
        if K < 1 or (include_rollout and M < 2):
            raise ValueError("K must be positive, and the rollout ensemble must have at least 2 draws")
        if state_sigma <= 0:
            raise ValueError("state_sigma must be positive")
        if self.flow_type != "realnvp":
            raise ValueError("The manuscript objective requires bounded RealNVP coupling scales")
        if self.observation_stride != 1:
            raise ValueError("The manuscript observation window requires observation_stride=1")
        if self.d_u and (not self.direct_control or self.exclude_control_from_diffusion):
            raise ValueError("The manuscript requires direct_control=True and controls in both drift and flow")

        context = self.encode(y)
        transitions = y.shape[0]
        span = min(seg_length - 1, transitions)
        starts = jnp.arange(0, transitions, span)
        offsets = jnp.arange(span)
        dt = jnp.asarray(self.time_step, dtype=x.dtype)
        state_constant = 0.5 * transitions * self.d_x * jnp.log(2 * jnp.pi * state_sigma**2)
        base_constant = 0.5 * self.d_x * jnp.log(2 * jnp.pi)
        time_log_det = 0.5 * self.d_x * jnp.log(dt)
        segment_keys = jr.split(jr.fold_in(key, 0), starts.shape[0])

        def segment(start, segment_key):
            global_indices = start + offsets
            valid = global_indices < transitions
            truth = x[jnp.minimum(global_indices + 1, transitions)]

            def draw(draw_key):
                eps = jr.normal(draw_key, (span, self.d_x), dtype=x.dtype)

                def step(state, item):
                    index, noise, available = item
                    c = context[jnp.minimum(index, transitions - 1)]
                    time = index * dt
                    drift = self.drift(state, c, time)
                    if self.physics_fn is not None:
                        drift = drift + self.physics_fn(state)
                    transformed = self.diffusion.apply(noise, state, c, time)
                    next_state = state + dt * drift + jnp.sqrt(dt) * transformed
                    next_state = jnp.where(available, next_state, state)
                    # This is -log q at the generated transition, not an NLL
                    # evaluated on the reference transition. Its expectation
                    # is the entropy term specified by the manuscript.
                    nll = (0.5 * jnp.sum(noise**2) + base_constant
                           + self.diffusion.log_det(noise, state, c, time)
                           + time_log_det)
                    return next_state, (next_state, jnp.where(available, nll, 0.0))

                endpoint, (states, nlls) = jax.lax.scan(
                    step, x[start], (global_indices, eps, valid)
                )
                state_error = jnp.sum(jnp.where(valid[:, None], (states - truth)**2, 0.0))
                next_anchor = x[jnp.minimum(start + span, transitions)]
                boundary = jnp.where(start + span < transitions,
                                     jnp.sum((endpoint - next_anchor)**2), 0.0)
                return state_error / (2 * state_sigma**2), jnp.sum(nlls), boundary

            values = jax.vmap(draw)(jr.split(segment_key, K))
            return tuple(jnp.mean(value) for value in values)

        state, entropy, continuity = jax.vmap(segment)(starts, segment_keys)
        rollout = jnp.asarray(0.0, dtype=x.dtype)
        if include_rollout:
            ts = jnp.arange(transitions + 1, dtype=x.dtype) * dt
            draws = jax.vmap(lambda draw_key: self._integrate_custom(
                x[0], ts, context, draw_key, "euler"
            )[1:])(jr.split(jr.fold_in(key, 1), M))
            rollout = energy_score(draws, x[1:])
        return {
            "state": jnp.sum(state) + state_constant,
            "transition_entropy": jnp.sum(entropy),
            "continuity": jnp.sum(continuity),
            "rollout": rollout,
        }

    def paper_loss(
        self, y, x_true, ts, key, *, seg_length, K=8,
        continuity_weight=0.0, rollout_weight=0.0, state_sigma=1.0,
        include_rollout=True, rollout_K=None,
    ):
        """Differentiable manuscript objective for one path or a batch of paths.

        ``y[k]`` is the observation at the source time of transition k. The
        shared ``ts`` must contain all T+1 state times starting at zero, on the
        trained physical grid. Call with ``include_rollout=(update % 5 == 0)``
        to reproduce the paper's every-five-updates rollout schedule.
        """
        if x_true.ndim not in {2, 3} or y.ndim != x_true.ndim:
            raise ValueError("Expected x=(T+1,D), y=(T,O), or their batched versions")
        if x_true.shape[-1] != self.d_x or y.shape[-1] != self.d_y:
            raise ValueError("State and observation feature dimensions must match the model")
        ts = jnp.asarray(ts, dtype=x_true.dtype)
        if ts.ndim != 1 or ts.shape[0] != x_true.shape[-2]:
            raise ValueError("ts must contain the T+1 state times, including the supplied initial time")
        expected = jnp.arange(ts.shape[0], dtype=ts.dtype) * self.time_step
        tolerance = 1e-4 * self.time_step + 8 * jnp.finfo(ts.dtype).eps * jnp.max(jnp.abs(expected))
        x_true = eqx.error_if(x_true, jnp.any(~jnp.isfinite(ts) | (jnp.abs(ts - expected) > tolerance)),
                              "Manuscript loss requires state times starting at zero on the trained fixed grid")
        active_rollout = include_rollout and rollout_weight != 0.0

        def one(xi, yi, draw_key):
            terms = self.paper_loss_components_single(
                xi, yi, draw_key, seg_length=seg_length, K=K,
                state_sigma=state_sigma, include_rollout=active_rollout,
                rollout_K=rollout_K,
            )
            return (terms["state"] + terms["transition_entropy"]
                    + continuity_weight * terms["continuity"]
                    + rollout_weight * terms["rollout"])

        if x_true.ndim == 2:
            return one(x_true, y, key)
        if x_true.shape[0] != y.shape[0]:
            raise ValueError("State and observation batch sizes must match")
        return jnp.mean(jax.vmap(one)(x_true, y, jr.split(key, x_true.shape[0])))

    def nll_pathwise_single(
        self, x, y, key, seg_length, solver_name="euler",
        use_custom_solver=True, shooting_noise_std=0.0, kl_weight=0.0,
        continuity_weight=0.0, objective="transition_nll", num_ensemble=8,
        energy_beta=1.0, energy_uncertainty_weight=0.5, state_sigma=1.0,
    ):
        """Score a conditional trajectory on the model's fixed physical grid.

        ``x`` includes the supplied initial state; ``y[i]`` conditions transition
        x[i] -> x[i+1]. The default is exact Euler transition NLL. Other fixed
        step kernels use the proper ensemble energy score. Sample reconstruction
        MSE remains an explicitly selected auxiliary objective, not a likelihood.
        ``kl_weight`` is a deprecated compatibility argument; no KL is claimed.
        """
        del use_custom_solver, kl_weight
        self._validate_training_path(x, y)
        if objective == "paper":
            if solver_name.lower() not in {"euler", "euler_maruyama"}:
                raise ValueError("The manuscript NormFlowSDE update requires the Euler kernel")
            if shooting_noise_std != 0.0:
                raise ValueError("The manuscript uses supplied segment anchors without shooting noise")
            return self.paper_loss(
                y, x, jnp.arange(x.shape[0], dtype=x.dtype) * self.time_step,
                key, seg_length=seg_length, K=num_ensemble,
                continuity_weight=continuity_weight, state_sigma=state_sigma,
                include_rollout=False,
            )
        if seg_length < 1:
            raise ValueError("seg_length must be positive")
        c = self.encode_observation_stride(y, self.observation_stride)
        c_diff = self.encode_diffusion_stride(y, self.observation_stride)
        if objective == "transition_nll":
            if solver_name.lower() not in {"euler", "euler_maruyama"}:
                raise ValueError("Exact transition NLL requires the Euler kernel; use objective='energy_score' for other fixed-step kernels")
            return self.transition_nll(x, c, c_diff, 0.0, self.time_step)
        if objective not in {"energy_score", "gaussian_nll", "mse"}:
            raise ValueError("objective must be transition_nll, energy_score, gaussian_nll, or mse")
        if objective == "energy_score" and num_ensemble < 2:
            raise ValueError("Energy scoring requires at least two independent samples")
        # Include the final partial segment. Each is scored conditional on its
        # true initial state; this is a composite score when seg_length < T.
        total = jnp.asarray(0.0, x.dtype)
        transitions = y.shape[0]
        for segment_index, start in enumerate(range(0, transitions, seg_length)):
            stop = min(start + seg_length, transitions)
            segment_key = jr.fold_in(key, segment_index)
            init_key, draw_key = jr.split(segment_key)
            z0 = x[start] + shooting_noise_std * jr.normal(init_key, x[start].shape)
            ts = jnp.arange(start, stop + 1, dtype=x.dtype) * self.time_step
            count = num_ensemble if objective != "mse" else 1
            draws = jax.vmap(lambda k: self._integrate_custom(
                z0, ts, c, k, solver_name, c_diffusion=c_diff
            )[1:])(jr.split(draw_key, count))
            truth = x[start + 1:stop + 1]
            if objective == "energy_score":
                value = energy_score(draws, truth, energy_beta, energy_uncertainty_weight)
            elif objective == "gaussian_nll":
                value = gaussian_nll(draws, truth)
            else:
                value = jnp.mean((draws[0] - truth) ** 2)
            # Extra endpoint reconstruction is explicitly an auxiliary penalty.
            if continuity_weight:
                value = value + continuity_weight * jnp.mean((jnp.mean(draws[:, -1], axis=0) - truth[-1]) ** 2)
            total = total + (stop - start) * value
        return total / transitions

    def _context_at(self, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        """Nearest-neighbour context lookup on the physical-time grid."""
        T = c.shape[0]
        # c[i] conditions the transition starting at physical time i*time_step.
        idx = jnp.rint(t / self.time_step).astype(jnp.int32)
        idx = jnp.clip(idx, 0, T - 1)
        return c[idx]

    def _custom_drift(self, z: jnp.ndarray, c: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        c_t = self._context_at(c, t)
        f = self.drift(z, c_t, t)
        if self.physics_fn is not None:
            f = f + self.physics_fn(z)
        return f

    def _custom_diffusion(
        self,
        eps:            jnp.ndarray,
        z:              jnp.ndarray,
        c:              jnp.ndarray,
        t:              jnp.ndarray,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        c_t = self._context_at(c, t)
        return self.diffusion.apply(eps * diffusion_temp, z, c_t, t)

    def _integrate_custom(
        self,
        z0:             jnp.ndarray,
        ts:             jnp.ndarray,
        c:              jnp.ndarray,
        key:            jr.PRNGKey,
        solver_name:    str,
        diffusion_temp: float = 1.0,
        c_diffusion:    jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Backward-compatible alias for the shared stochastic solver path."""
        return self._integrate_stochastic_solver(
            z0, ts, c, key, solver_name, diffusion_temp, c_diffusion
        )

    def _integrate_stochastic_solver(
        self,
        z0:             jnp.ndarray,
        ts:             jnp.ndarray,
        c:              jnp.ndarray,
        key:            jr.PRNGKey,
        solver_name:    str,
        diffusion_temp: float = 1.0,
        c_diffusion:    jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Integrate using solver implementations from stochastic_solvers.py."""
        solver = get_stochastic_solver(solver_name)
        if c_diffusion is None:
            c_diffusion = c
        eps_steps = jr.normal(key, (ts.shape[0] - 1, self.d_x))
        dt0 = ts[1] - ts[0]
        bm = StepNoisePath(eps_steps, ts[0], dt0)

        gaps = ts[1:] - ts[:-1]
        ts = eqx.error_if(ts, jnp.any(jnp.abs(gaps - self.time_step) > 1e-4 * self.time_step + 8 * jnp.finfo(ts.dtype).eps * jnp.max(jnp.abs(ts))),
                          "Flow-noise kernels require their trained fixed time_step; changing dt changes the model")
        if solver_name.lower() in {"milstein", "adaptive_euler", "adaptive_milstein"}:
            raise ValueError("Milstein/adaptive Euler require linear Gaussian diffusion, not a nonlinear flow-noise kernel")

        def step(z, t_pair):
            t0, t1 = t_pair
            c_t = self._context_at(c, t0)
            cd_t = self._context_at(c_diffusion, t0)
            def drift_fn(t, state):
                drift = self.drift(state, c_t, t)
                return drift if self.physics_fn is None else drift + self.physics_fn(state)
            def diffuse_fn(t, state, eps):
                return self.diffusion.apply(eps * diffusion_temp, state, cd_t, t)
            z1 = solver.step(z, t0, t1, drift_fn, diffuse_fn, bm)
            return z1, z1

        _, states = jax.lax.scan(step, z0, (ts[:-1], ts[1:]))
        return jnp.concatenate([z0[None], states], axis=0)

    def nll_pathwise(
        self, x, y, key, seg_length, solver_name="euler",
        use_custom_solver=True, shooting_noise_std=0.0, kl_weight=0.0,
        continuity_weight=0.0, objective="transition_nll", num_ensemble=8,
        energy_beta=1.0, energy_uncertainty_weight=0.5, state_sigma=1.0,
    ):
        """Batch conditional score; x=(B,T+1,D), y=(B,T,O)."""
        keys = jr.split(key, x.shape[0])
        return jnp.mean(jax.vmap(lambda xi, yi, ki: self.nll_pathwise_single(
            xi, yi, ki, seg_length, solver_name, use_custom_solver,
            shooting_noise_std, kl_weight, continuity_weight, objective,
            num_ensemble, energy_beta, energy_uncertainty_weight, state_sigma,
        ))(x, y, keys))

    def rollout_loss_single(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        solver_name:      str = "euler",
        use_custom_solver: bool = True,
        num_ensemble: int = 20,
        loss_type: str = "gaussian_nll",
        energy_beta: float = 1.0,
        energy_uncertainty_weight: float = 0.5,
    ) -> jnp.ndarray:
        """Full-horizon rollout loss from x[0].

        ``loss_type="mse"`` keeps the old one-sample reconstruction objective.
        ``"energy_score"`` and ``"gaussian_nll"`` evaluate an ensemble of SDE
        paths. Energy scoring targets their joint law; gaussian_nll is a
        diagonal Gaussian moment approximation, not the flow likelihood.
        """
        self._validate_training_path(x, y)
        T = x.shape[0]
        dt0 = self.time_step
        ts = jnp.arange(T, dtype=x.dtype) * dt0
        c = self.encode_observation_stride(y, self.observation_stride)
        c_diffusion = self.encode_diffusion_stride(y, self.observation_stride)

        sde_key = key
        if loss_type == "mse":
            z0 = x[0]
            pred = self._integrate_stochastic_solver(
                z0, ts, c, sde_key, solver_name, c_diffusion=c_diffusion
            )
            return jnp.mean((pred[1:] - x[1:]) ** 2)

        z0s = jnp.broadcast_to(x[0], (num_ensemble, self.d_x))

        sde_keys = jr.split(sde_key, num_ensemble)

        def integrate_one(z0, k):
            return self._integrate_stochastic_solver(
                z0, ts, c, k, solver_name, c_diffusion=c_diffusion
            )

        preds = jax.vmap(integrate_one)(z0s, sde_keys)[:, 1:]
        x = x[1:]
        if loss_type == "energy_score":
            return energy_score(
                preds,
                x,
                beta=energy_beta,
                uncertainty_weight=energy_uncertainty_weight,
            )
        if loss_type == "gaussian_nll":
            return gaussian_nll(preds, x)
        raise ValueError(
            f"Unknown rollout loss_type '{loss_type}'. "
            "Choose 'mse', 'energy_score', or 'gaussian_nll'."
        )

    def rollout_loss(
        self,
        x:                jnp.ndarray,
        y:                jnp.ndarray,
        key:              jr.PRNGKey,
        solver_name:      str = "euler",
        use_custom_solver: bool = True,
        num_ensemble: int = 20,
        loss_type: str = "energy_score",
        energy_beta: float = 1.0,
        energy_uncertainty_weight: float = 0.5,
        rollout_window: int = 0,
    ) -> jnp.ndarray:
        """Batch full-horizon pathwise rollout loss."""
        if rollout_window > 0 and rollout_window < y.shape[1]:
            # Composite proper scoring uses true anchors without resetting the
            # model's clock or truncating its observation encoder context.
            return self.nll_pathwise(
                x, y, key, rollout_window, solver_name, use_custom_solver,
                objective=loss_type, num_ensemble=num_ensemble,
                energy_beta=energy_beta,
                energy_uncertainty_weight=energy_uncertainty_weight,
            )
        B = x.shape[0]
        keys = jr.split(key, B)
        return jnp.mean(jax.vmap(
            lambda xi, yi, ki: self.rollout_loss_single(
                xi, yi, ki, solver_name, use_custom_solver,
                num_ensemble, loss_type, energy_beta, energy_uncertainty_weight
            )
        )(x, y, keys))

    # ------------------------------------------------------------------
    # Posterior sampling
    # ------------------------------------------------------------------

    def sample_posterior(
        self,
        y:              jnp.ndarray,   # (T, d_y)
        x0:             jnp.ndarray,   # (d_x,)
        key:            jr.PRNGKey,
        n_particles:    int   = 100,
        solver_name:    str   = "euler",
        dt_factor:      float | None = None,
        diffusion_temp: float = 1.0,
    ) -> jnp.ndarray:
        """
        Sample the learned conditional law of x[1:T+1] given y[0:T] and x0.

        Every particle starts at the supplied ``x0``.
        G_φ(z_t, c_t, t) shapes standard Gaussian eps into structured diffusion.
        Returns (n_particles, T, d_x).
        """
        return self.sample_posterior_custom_solver(
            y, x0, key, n_particles, solver_name, dt_factor, diffusion_temp
        )

    def sample_posterior_custom_solver(
        self,
        y:              jnp.ndarray,   # (T, d_y)
        x0:             jnp.ndarray,   # (d_x,)
        key:            jr.PRNGKey,
        n_particles:    int   = 100,
        solver_name:    str   = "euler",
        dt_factor:      float | None = None,
        diffusion_temp: float = 1.0,
        observation_stride: int | None = None,
    ) -> jnp.ndarray:
        """
        Sample x[1:T+1] using source-time observations y[0:T].
        
        The selected fixed-step kernel is applied on the trained physical grid.
        Milstein and adaptive SDE solvers require linear Gaussian diffusion and
        are not valid for this nonlinear transition-noise model.
        
        The custom solver form is:
            dz_t = f_θ(z_t, c_t, t) dt + G_φ(eps_t, z_t, c_t, t) sqrt(dt)

        where G_φ is the RealNVP flow that transforms eps_t ~ N(0, I).
        
        Args:
            y: Observations (T, d_y)
            x0: Initial state (d_x,)
            key: Random key
            n_particles: Number of particles to sample
            solver_name: Fixed-step kernel ("euler", "heun", "midpoint", "rk4").
            dt_factor: Legacy duration argument; must equal ``T * self.time_step``.
            diffusion_temp: Temperature scaling for diffusion
        
        Returns:
            Samples (n_particles, T, d_x)
        """
        T   = y.shape[0]
        if T < 1 or n_particles < 1:
            raise ValueError("At least one observation and particle are required")
        stride = self.observation_stride if observation_stride is None else observation_stride
        c = self.encode_observation_stride(y, stride)
        c_diffusion = self.encode_diffusion_stride(y, stride)
        # y[k] is at the source time of x[k] -> x[k+1]. Integrate T
        # intervals and discard the supplied initial state from the result.
        duration = T * self.time_step if dt_factor is None else dt_factor
        if T < 1:
            raise ValueError("At least one observation is required")
        duration = eqx.error_if(jnp.asarray(duration), jnp.abs(duration - T * self.time_step) > 1e-5 * self.time_step,
                               "dt_factor must preserve the trained fixed time_step")
        ts = jnp.arange(T + 1) * (duration / T)
        
        def simulate_one(particle_key):
            k_sde = particle_key
            z0 = x0
            path = self._integrate_custom(
                z0, ts, c, k_sde, solver_name, diffusion_temp,
                c_diffusion=c_diffusion,
            )
            return path[1:]
        
        return jax.vmap(simulate_one)(jr.split(key, n_particles))

    def sample_posterior_batch(
        self,
        y:           jnp.ndarray,   # (B, T, d_y)
        x0:          jnp.ndarray,   # (B, d_x)
        key:         jr.PRNGKey,
        n_particles: int = 50,
        solver_name: str = "euler",
    ) -> jnp.ndarray:
        """Returns (n_particles, B, T, d_x)."""
        B    = y.shape[0]
        keys = jr.split(key, B)
        samples = jax.vmap(
            lambda yi, x0i, ki: self.sample_posterior(
                yi, x0i, ki, n_particles, solver_name
            )
        )(y, x0, keys)
        return jnp.swapaxes(samples, 0, 1)

# ---------------------------------------------------------------------------
# RealNVPFlowSDE training utilities
# ---------------------------------------------------------------------------

@eqx.filter_jit
def make_step_realnvp_pathwise(
    model:            RealNVPFlowSDE,
    x_batch:          jnp.ndarray,
    y_batch:          jnp.ndarray,
    key:              jr.PRNGKey,
    seg_length:       int,
    opt_state,
    optimizer,
    solver_name:      str = "euler",
    use_custom_solver: bool = True,
    shooting_noise_std: float = 0.0,
    rollout_loss_weight: float = 0.0,
    kl_weight: float = 0.0,
    continuity_weight: float = 0.0,
    rollout_num_ensemble: int = 20,
    rollout_loss_type: str = "energy_score",
    rollout_energy_beta: float = 1.0,
    rollout_uncertainty_weight: float = 0.5,
    rollout_window: int = 0,
    transition_loss_weight: float = 0.0,
    objective: str = "transition_nll",
    objective_num_ensemble: int = 8,
    state_sigma: float = 1.0,
):
    """Optimize the manuscript loss or an explicit legacy conditional score.

    ``use_custom_solver`` and ``kl_weight`` are legacy compatibility arguments.
    There is no KL term. An auxiliary Euler transition loss is not duplicated
    when transition NLL is already the primary objective. Other selected
    kernels are scored by their simulated path ensembles.
    """
    def loss_fn(m):
        if objective == "paper":
            if solver_name.lower() not in {"euler", "euler_maruyama"}:
                raise ValueError("The manuscript NormFlowSDE update requires the Euler kernel")
            if shooting_noise_std != 0.0 or transition_loss_weight != 0.0 or rollout_window != 0:
                raise ValueError("The manuscript uses exact anchors, sampled transition entropy, and a full-horizon rollout")
            if (rollout_loss_type != "energy_score" or rollout_energy_beta != 1.0
                    or rollout_uncertainty_weight != 0.5):
                raise ValueError("The manuscript rollout requires the standard energy score")
            return m.paper_loss(
                y_batch, x_batch,
                jnp.arange(x_batch.shape[1], dtype=x_batch.dtype) * m.time_step,
                key, seg_length=seg_length, K=objective_num_ensemble,
                continuity_weight=continuity_weight,
                rollout_weight=rollout_loss_weight, state_sigma=state_sigma,
                rollout_K=rollout_num_ensemble,
            )
        path_loss = m.nll_pathwise(
            x_batch, y_batch, key, seg_length, solver_name, use_custom_solver,
            shooting_noise_std, kl_weight, continuity_weight, objective,
            objective_num_ensemble, state_sigma=state_sigma,
        )
        total_loss = path_loss
        if transition_loss_weight != 0.0 and objective != "transition_nll":
            def one_transition_nll(xi, yi):
                drift_context = m.encode_observation_stride(
                    yi, m.observation_stride
                )
                diffusion_context = m.encode_diffusion_stride(
                    yi, m.observation_stride
                )
                return m.transition_nll(
                    xi, drift_context, diffusion_context,
                    jnp.asarray(0.0, xi.dtype), m.time_step
                )

            transition_loss = jnp.mean(
                jax.vmap(one_transition_nll)(x_batch, y_batch)
            )
            total_loss = total_loss + transition_loss_weight * transition_loss
        if rollout_loss_weight != 0.0:
            rollout_loss = m.rollout_loss(
                x_batch, y_batch, jr.fold_in(key, 29), solver_name,
                use_custom_solver, rollout_num_ensemble,
                rollout_loss_type, rollout_energy_beta,
                rollout_uncertainty_weight, rollout_window
            )
            total_loss = total_loss + rollout_loss_weight * rollout_loss
        return total_loss

    loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
    updates, opt_state_new = optimizer.update(grads, opt_state, model)
    return eqx.apply_updates(model, updates), opt_state_new, loss
