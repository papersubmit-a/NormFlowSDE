"""
Windowed pointwise amortized state estimation.

For each timestep t, estimate x_t conditioned on a causal window of
recent observations:

    cond_t = Encoder([y_{t-W+1}, ..., y_t], [u_{t-W+1}, ..., u_t], x_0)

The encoder is a GRU or Transformer (same architectures as AmortizedStateFlow),
but it only sees a fixed-length window of length W = seg_len rather than the
full trajectory.  Its output at the **last** position is used as the flow
condition for x_t.

Windowing is done entirely inside JAX — no Python loop over time:
  1. Pad (y, u) left with W-1 zeros → (M, H+W-1, *)
  2. Extract all windows via integer indexing → (M, H, W, *)
  3. Reshape to (M*H, W, *) and feed into the encoder
  4. Take last encoder output → (M*H, D_enc), project to (M*H, D_out)
  5. Flow maps x_t <-> z_t conditioned on cond_t, one token at a time

This is fully causal (no future observations leak in) and online-capable.
Window length W is a static hyperparameter (seg_len from the training script).

The public API (log_prob / nll_loss / sample) mirrors AmortizedStateFlow so
training and evaluation scripts are interchangeable.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import List, Optional, Tuple

from bayesinf.applications.state_estimation.amortized_state_flow import (
    _apply_linear,
    GRUSequenceEncoder,
    TransformerSequenceEncoder,
)
from bayesinf.models.normalizing_flow import (
    ActNorm, RandomPermutation,
    AffineCoupling, SplineCoupling, MAFLayer,
)
from bayesinf.models.flow_matching import VectorField as FlowMatchingVectorField


# ---------------------------------------------------------------------------
# Windowed encoder
# ---------------------------------------------------------------------------

class WindowedEncoder(eqx.Module):
    """
    Causal windowed encoder: for each position t, encodes the window
    [y_{t-W+1}, ..., y_t] and [u_{t-W+1}, ..., u_t] through a GRU or
    Transformer, then takes the last output as the conditioning signal.

    Input shapes:
        y  : (M, H, D_y)
        u  : (M, H, D_u)
        x0 : (M, D_x)

    Output shape:
        cond : (M, H, D_out)

    Pipeline
    --------
    1. Concatenate [y, u] -> (M, H, D_y+D_u)
    2. Pad left with W-1 zeros -> (M, H+W-1, D_y+D_u)
    3. Extract causal windows via index trick -> (M, H, W, D_y+D_u)
    4. Reshape -> (M*H, W, D_y+D_u)
    5. Encoder (GRU/Transformer) -> (M*H, W, D_enc), take [:, -1, :] -> (M*H, D_enc)
    6. Project x0 -> (M, D_enc), broadcast to (M, H, D_enc), reshape -> (M*H, D_enc)
    7. Concatenate [enc_win, enc_x0] -> (M*H, 2*D_enc), project -> (M*H, D_out)
    8. Reshape -> (M, H, D_out)
    """

    seq_encoder:  eqx.Module        # GRUSequenceEncoder | TransformerSequenceEncoder
    x0_proj:      eqx.nn.Linear     # D_x  -> D_enc
    out_proj:     eqx.nn.Linear     # 2*D_enc -> D_out
    window_size:  int  = eqx.field(static=True)
    d_enc:        int  = eqx.field(static=True)
    d_out:        int  = eqx.field(static=True)
    encoder_type: str  = eqx.field(static=True)

    def __init__(
        self,
        d_y:          int,
        d_u:          int,
        d_x:          int,
        d_enc:        int,
        d_out:        int,
        window_size:  int,
        key:          jr.PRNGKey,
        encoder_type: str = "gru",
        gru_hidden:   int = 64,
        d_model:      int = 64,
        n_heads:      int = 4,
        d_ff:         int = 256,
        n_layers:     int = 3,
        max_len:      int = 512,
    ):
        self.window_size  = window_size
        self.d_enc        = d_enc
        self.d_out        = d_out
        self.encoder_type = encoder_type.lower()

        k_enc, k_x0, k_out = jr.split(key, 3)

        # Encoder input dim is D_y + D_u (y and u are concatenated per timestep)
        d_in = d_y + d_u
        if self.encoder_type == "transformer":
            self.seq_encoder = TransformerSequenceEncoder(
                d_y=d_in, d_enc=d_enc, key=k_enc,
                d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, n_layers=n_layers, max_len=max_len,
            )
        else:
            self.seq_encoder = GRUSequenceEncoder(
                d_y=d_in, d_hidden=gru_hidden, d_enc=d_enc, key=k_enc,
            )

        self.x0_proj  = eqx.nn.Linear(d_x,        d_enc,   key=k_x0)
        self.out_proj = eqx.nn.Linear(2 * d_enc,   d_out,   key=k_out)

    def encode_with_hidden(
        self,
        y:  jnp.ndarray,   # (M, H, D_y)
        u:  jnp.ndarray,   # (M, H, D_u)
        x0: jnp.ndarray,   # (M, D_x)
    ):
        """
        Full forward pass returning both the flow-conditioning context and the
        pre-projection shared backbone representation.

        Returns
        -------
        cond : (M, H, D_out)    context for flow conditioning  [= out_proj(h)]
        h    : (M, H, 2*D_enc)  shared backbone repr before out_proj
                                 — branch point for GaussianObsHead

        Dimension trace
        ---------------
        yu         : (M, H, D_y+D_u)
        yu_pad     : (M, H+W-1, D_y+D_u)
        windows    : (M, H, W, D_y+D_u)
        win_flat   : (M*H, W, D_y+D_u)
        enc_last   : (M*H, D_enc)          last GRU/Transformer position
        enc_x0     : (M*H, D_enc)          x0 projection, broadcast
        h_flat     : (M*H, 2*D_enc)        shared backbone output  ← branch here
        cond_flat  : (M*H, D_out)          head 1: flow conditioning
        """
        M, H, D_y = y.shape
        W = self.window_size

        yu    = jnp.concatenate([y, u], axis=-1)                        # (M, H, D_in)
        D_in  = yu.shape[-1]
        yu_pad = jnp.pad(yu, ((0, 0), (W - 1, 0), (0, 0)))             # (M, H+W-1, D_in)
        t_idx   = jnp.arange(H)[:, None] + jnp.arange(W)[None, :]      # (H, W)
        windows = yu_pad[:, t_idx, :]                                    # (M, H, W, D_in)
        win_flat = windows.reshape(M * H, W, D_in)                      # (M*H, W, D_in)

        enc_flat = self.seq_encoder(win_flat)                            # (M*H, W, D_enc)
        enc_last = enc_flat[:, -1, :]                                    # (M*H, D_enc)

        enc_x0_vec = jax.vmap(self.x0_proj)(x0)                         # (M, D_enc)
        enc_x0     = jnp.broadcast_to(
            enc_x0_vec[:, None, :], (M, H, self.d_enc)
        ).reshape(M * H, self.d_enc)                                     # (M*H, D_enc)

        h_flat    = jnp.concatenate([enc_last, enc_x0], axis=-1)        # (M*H, 2*D_enc)
        cond_flat = jax.vmap(self.out_proj)(h_flat)                     # (M*H, D_out)

        return (
            cond_flat.reshape(M, H, self.d_out),      # cond
            h_flat.reshape(M, H, 2 * self.d_enc),     # h
        )

    def __call__(
        self,
        y:  jnp.ndarray,   # (M, H, D_y)
        u:  jnp.ndarray,   # (M, H, D_u)
        x0: jnp.ndarray,   # (M, D_x)
    ) -> jnp.ndarray:
        """Returns cond (M, H, D_out). Thin wrapper around encode_with_hidden."""
        cond, _ = self.encode_with_hidden(y, u, x0)
        return cond


# ---------------------------------------------------------------------------
# Flow transform builder
# ---------------------------------------------------------------------------

def _mask(dim: int, parity: int) -> jnp.ndarray:
    return jnp.array([(i + parity) % 2 for i in range(dim)], dtype=jnp.float32)


def build_flow_transforms(
    input_dim:    int,
    cond_dim:     int,
    n_blocks:     int,
    hidden_size:  int,
    n_hidden:     int,
    flow_type:    str,
    key:          jr.PRNGKey,
    num_bins:     int = 8,
) -> List[eqx.Module]:
    flow_type    = flow_type.lower()
    transforms   = []
    hidden_sizes = [hidden_size] * n_hidden

    for i in range(n_blocks // 2):
        k_an, k_perm, k1, k2 = jr.split(jr.fold_in(key, i), 4)
        transforms.append(ActNorm(input_dim))
        transforms.append(RandomPermutation(input_dim, k_perm))
        if flow_type == "nsf":
            transforms += [
                SplineCoupling(input_dim, hidden_sizes, cond_dim,
                               _mask(input_dim, i % 2),       k1, num_bins=num_bins),
                SplineCoupling(input_dim, hidden_sizes, cond_dim,
                               _mask(input_dim, (i+1) % 2),   k2, num_bins=num_bins),
            ]
        elif flow_type == "maf":
            transforms += [
                MAFLayer(input_dim, hidden_sizes, cond_dim,
                         natural_ordering=False, key=k1),
                MAFLayer(input_dim, hidden_sizes, cond_dim,
                         natural_ordering=False, key=k2),
            ]
        else:  # realnvp
            transforms += [
                AffineCoupling(input_dim, hidden_sizes, cond_dim,
                               _mask(input_dim, i % 2),       k1),
                AffineCoupling(input_dim, hidden_sizes, cond_dim,
                               _mask(input_dim, (i+1) % 2),   k2),
            ]
    transforms.append(ActNorm(input_dim))
    return transforms


# ---------------------------------------------------------------------------
# PointwiseStateFlow
# ---------------------------------------------------------------------------

class PointwiseStateFlow(eqx.Module):
    """
    Amortized state flow with a causal windowed encoder.

    For each timestep t, the flow estimates p(x_t | y_{t-W+1:t}, u_{t-W+1:t}, x_0)
    using a GRU or Transformer encoder over the observation window of length W.

    Parameters
    ----------
    d_x           : state dimension
    d_y           : observation dimension
    d_u           : control dimension
    d_enc         : encoder internal dimension
    d_out         : flow conditioning dimension (encoder output)
    window_size   : causal window length W (= seg_len)
    encoder_type  : 'gru' | 'transformer'
    gru_hidden    : GRU hidden size
    d_model       : Transformer model dim
    n_heads       : Transformer attention heads
    d_ff          : Transformer FFN hidden dim
    n_layers      : Transformer depth
    flow_type     : 'realnvp' | 'nsf' | 'maf'
    n_coupling    : number of coupling block pairs
    hidden_size   : coupling net hidden width
    n_hidden      : coupling net hidden depth
    num_bins      : spline bins (NSF only)
    """

    encoder:      WindowedEncoder
    transforms:   List[eqx.Module]
    vector_field: Optional[FlowMatchingVectorField]

    state_dim:   int = eqx.field(static=True)
    d_out:       int = eqx.field(static=True)
    flow_type:   str = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,
        d_y:          int,
        d_u:          int,
        d_enc:        int         = 64,
        d_out:        int         = 96,
        window_size:  int         = 5,
        encoder_type: str         = "gru",
        gru_hidden:   int         = 64,
        d_model:      int         = 64,
        n_heads:      int         = 4,
        d_ff:         int         = 256,
        n_layers:     int         = 3,
        flow_type:    str         = "realnvp",
        n_coupling:   int         = 8,
        hidden_size:  int         = 128,
        n_hidden:     int         = 2,
        num_bins:     int         = 8,
        key:          jr.PRNGKey  = jr.PRNGKey(0),
    ):
        self.state_dim = d_x
        self.d_out     = d_out
        self.flow_type = flow_type.lower()

        k_enc, k_flow = jr.split(key)

        self.encoder = WindowedEncoder(
            d_y=d_y, d_u=d_u, d_x=d_x,
            d_enc=d_enc, d_out=d_out,
            window_size=window_size,
            key=k_enc,
            encoder_type=encoder_type,
            gru_hidden=gru_hidden,
            d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, n_layers=n_layers,
        )

        if self.flow_type == "fm":
            self.transforms   = []
            self.vector_field = FlowMatchingVectorField(
                d_x, d_out, hidden_size, n_hidden, k_flow
            )
        else:
            self.transforms = build_flow_transforms(
                input_dim=d_x, cond_dim=d_out,
                n_blocks=n_coupling,
                hidden_size=hidden_size, n_hidden=n_hidden,
                flow_type=self.flow_type, key=k_flow,
                num_bins=num_bins,
            )
            self.vector_field = None

    # ------------------------------------------------------------------
    # Internal flow helpers
    # ------------------------------------------------------------------

    def _flow_fwd(
        self,
        x:        jnp.ndarray,   # (N, D_x)   N = M*H
        cond_emb: jnp.ndarray,   # (N, D_out)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logdet = jnp.zeros(x.shape[0])
        z = x
        for t in self.transforms:
            if isinstance(t, (ActNorm, RandomPermutation)):
                z, ldj = t(z, reverse=False)
            elif isinstance(t, MAFLayer):
                z, ldj = t(z, cond=cond_emb, reverse=False)
            else:
                z, ldj = t(z, cond_emb, reverse=False)
            logdet = logdet + ldj
        return z, logdet

    def _flow_inv(
        self,
        z:        jnp.ndarray,   # (N, D_x)
        cond_emb: jnp.ndarray,   # (N, D_out)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logdet = jnp.zeros(z.shape[0])
        x = z
        for t in reversed(self.transforms):
            if isinstance(t, (ActNorm, RandomPermutation)):
                x, ldj = t(x, reverse=True)
            elif isinstance(t, MAFLayer):
                x, ldj = t(x, cond=cond_emb, reverse=True)
            else:
                x, ldj = t(x, cond_emb, reverse=True)
            logdet = logdet + ldj
        return x, logdet

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _encode(
        self,
        y:  jnp.ndarray,
        u:  jnp.ndarray,
        x0: jnp.ndarray,
    ) -> jnp.ndarray:
        """Returns cond (M, H, D_out). Convenience alias for FM loss."""
        return self.encoder(y, u, x0)

    def log_prob(
        self,
        x:  jnp.ndarray,   # (M, H, D_x)
        y:  jnp.ndarray,   # (M, H, D_y)
        u:  jnp.ndarray,   # (M, H, D_u)
        x0: jnp.ndarray,   # (M, D_x)
    ) -> jnp.ndarray:
        """
        Log p(x | y_{windowed}, u_{windowed}, x0).  Returns (M,).
        Not defined for FM — raises if called on an FM model.
        """
        assert self.flow_type != "fm", "log_prob not available for FM; use nll_loss."
        M, H, D_x = x.shape

        cond      = self.encoder(y, u, x0)               # (M, H, D_out)
        x_flat    = x.reshape(M * H, D_x)
        cond_flat = cond.reshape(M * H, self.d_out)

        z_flat, logdet_flat = self._flow_fwd(x_flat, cond_flat)

        z      = z_flat.reshape(M, H, D_x)
        logdet = logdet_flat.reshape(M, H).sum(axis=-1)  # (M,)

        log_p = -0.5 * (
            jnp.sum(z ** 2, axis=(-2, -1))
            + H * D_x * jnp.log(2 * jnp.pi)
        )

        log_prob = log_p + logdet
        log_prob = jnp.where(jnp.isfinite(log_prob), log_prob,
                             jnp.full_like(log_prob, -1e4))
        return log_prob

    def nll_loss(
        self,
        x:   jnp.ndarray,             # (M, H, D_x)
        y:   jnp.ndarray,             # (M, H, D_y)
        u:   jnp.ndarray,             # (M, H, D_u)
        x0:  jnp.ndarray,             # (M, D_x)
        key: Optional[jr.PRNGKey] = None,
    ) -> jnp.ndarray:
        """Scalar loss per dimension, normalised by H*D_x.
        NF types: NLL.  FM type: flow-matching MSE (key required).
        """
        _, H, D_x = x.shape
        if self.flow_type == "fm":
            return self._fm_loss(x, y, u, x0, key)
        return -jnp.mean(self.log_prob(x, y, u, x0)) / (H * D_x)

    def _fm_loss(
        self,
        x:   jnp.ndarray,   # (M, H, D_x)
        y:   jnp.ndarray,
        u:   jnp.ndarray,
        x0:  jnp.ndarray,
        key: jr.PRNGKey,
    ) -> jnp.ndarray:
        """Flow-matching MSE: z_t = (1-t)*z0 + t*x, target v = x - z0."""
        cond = self.encoder(y, u, x0)                    # (M, H, D_out)
        M, H, D_x = x.shape
        n = M * H
        x_flat = x.reshape(n, D_x)
        c_flat = cond.reshape(n, self.d_out)

        kt, kz = jr.split(key)
        t      = jr.uniform(kt, (n,))
        z0     = jr.normal(kz, (n, D_x))
        t_b    = t[:, None]
        z_t    = (1 - t_b) * z0 + t_b * x_flat
        v_pred = jax.vmap(self.vector_field)(t, z_t, c_flat)
        return jnp.mean((v_pred - (x_flat - z0)) ** 2)

    def sample(
        self,
        y:           jnp.ndarray,   # (M, H, D_y)
        u:           jnp.ndarray,   # (M, H, D_u)
        x0:          jnp.ndarray,   # (M, D_x)
        key:         jr.PRNGKey,
        num_samples: int = 1,
        fm_steps:    int = 100,
    ) -> jnp.ndarray:
        """
        Sample x_{1:H} ~ p(x | y_{windowed}, u_{windowed}, x0).
        Returns (S, M, H, D_x).
        """
        M, H, _ = y.shape
        S = num_samples
        cond = self.encoder(y, u, x0)                    # (M, H, D_out)

        if self.flow_type == "fm":
            return self._sample_fm(cond, key, S, fm_steps)

        cond_rep  = jnp.broadcast_to(cond[None], (S, M, H, self.d_out))
        cond_flat = cond_rep.reshape(S * M * H, self.d_out)
        z_flat    = jr.normal(key, (S * M * H, self.state_dim))
        x_flat, _ = self._flow_inv(z_flat, cond_flat)
        return x_flat.reshape(S, M, H, self.state_dim)

    def _sample_fm(
        self,
        cond:        jnp.ndarray,   # (M, H, D_out)
        key:         jr.PRNGKey,
        num_samples: int,
        steps:       int,
    ) -> jnp.ndarray:
        """Euler ODE integration for FM. Returns (S, M, H, D_x)."""
        M, H, D_out = cond.shape
        S = num_samples
        n = S * M * H
        cond_rep  = jnp.broadcast_to(cond[None], (S, M, H, D_out))
        cond_flat = cond_rep.reshape(n, D_out)
        z0        = jr.normal(key, (n, self.state_dim))
        dt        = 1.0 / steps
        vf        = self.vector_field

        def step(z_curr, i):
            t_val = i * dt
            t_b   = jnp.full((n,), t_val)
            v = jax.vmap(vf)(t_b, z_curr, cond_flat)
            return z_curr + v * dt, None

        z1, _ = jax.lax.scan(step, z0, jnp.arange(steps))
        return z1.reshape(S, M, H, self.state_dim)
