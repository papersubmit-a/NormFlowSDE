"""
Encoder architectures for amortized Bayesian state estimation.

Goal: estimate x_{1:H} given y_{1:H}, u_{1:H}, x_0.

Condition pipeline (shared by both GRU and Transformer variants):
  1. Encode y_{1:H}  -> enc_y  shape (M, H, D_enc)
  2. Project x_0     -> enc_x0 shape (M, D_enc), broadcast to (M, H, D_enc)
  3. Concatenate [enc_y, u_{1:H}, x_0_repeated] -> (M, H, D_enc + D_u + D_enc)
  4. Project to D_out -> cond  shape (M, H, D_out)

The flow receives cond (M, H, D_out) and transforms x_{1:H} <-> z_{0:H-1},
both of shape (M, H, D_x).  No changes to flow transforms.

Two encoder backends (selected via encoder_type='gru' | 'transformer'):
  - GRUSequenceEncoder : y (M,H,D_y) -> GRU scan -> Linear -> (M,H,D_enc)
  - TransformerSequenceEncoder: y (M,H,D_y) -> proj+PE+blocks -> proj -> (M,H,D_enc)
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Utility: apply eqx.nn.Linear over arbitrary leading dimensions
# ---------------------------------------------------------------------------

def _apply_linear(linear: eqx.nn.Linear, x: jnp.ndarray) -> jnp.ndarray:
    """
    Apply Linear to x of shape (..., in_features) -> (..., out_features).
    Flattens all leading dims, vmaps, reshapes back.
    Single vmap call -> efficient XLA fusion.
    """
    *leading, d_in = x.shape
    flat = x.reshape(-1, d_in)                      # (N, d_in)
    out  = jax.vmap(linear)(flat)                   # (N, d_out)
    return out.reshape(*leading, out.shape[-1])      # (..., d_out)


# ---------------------------------------------------------------------------
# GRU-based y-encoder
# ---------------------------------------------------------------------------

class GRUSequenceEncoder(eqx.Module):
    """
    Encode y_{1:H} with a GRU, then project each hidden state.

    Input : y  (M, H, D_y)
    Output: enc_y (M, H, D_enc)

    When local_window=True each GRU step receives either [y_{t-1}, y_t, y_{t+1}]
    or [y_{t-1}, y_t] when local_window_prev_only=True. Boundaries are
    zero-padded.

    Implementation notes
    --------------------
    * lax.scan over time axis (axis=1 transposed to axis=0) — O(H) steps,
      no Python loop, fully JIT-compiled.
    * vmap over batch inside scan step — one vmap per timestep (not M×H vmaps).
    * Single vmap over (M*H) for the output projection.
    """

    cell:         eqx.nn.GRUCell      # (D_in,) x (D_hidden,) -> (D_hidden,)
    proj:         eqx.nn.Linear       # D_hidden -> D_enc
    local_window: bool = eqx.field(static=True)
    local_window_prev_only: bool = eqx.field(static=True)

    def __init__(
        self,
        d_y:          int,
        d_hidden:     int,
        d_enc:        int,
        key:          jr.PRNGKey,
        local_window: bool = False,
        local_window_prev_only: bool = False,
    ):
        k_cell, k_proj = jr.split(key)
        if local_window:
            d_in = d_y * (2 if local_window_prev_only else 3)
        else:
            d_in = d_y
        self.cell         = eqx.nn.GRUCell(d_in, d_hidden, key=k_cell)
        self.proj         = eqx.nn.Linear(d_hidden, d_enc, key=k_proj)
        self.local_window = local_window
        self.local_window_prev_only = local_window_prev_only

    def __call__(self, y: jnp.ndarray) -> jnp.ndarray:
        """
        y: (M, H, D_y)  ->  enc_y: (M, H, D_enc)

        Dimension trace (local_window=True)
        ------------------------------------
        y_prev   : (M, H, D_y)  y shifted right by 1 (zero-padded at t=0)
        y_next   : (M, H, D_y)  y shifted left  by 1 (zero-padded at t=H-1)
        y_in     : (M, H, 2*D_y or 3*D_y) concatenated local window
        y_T      : (H, M, D_in) after transpose
        h_seq    : (H, M, D_hidden) from scan
        enc_y    : (M, H, D_enc)   after _apply_linear
        """
        M, H, D_y = y.shape
        d_hidden = self.cell.hidden_size

        if self.local_window:
            y_prev = jnp.concatenate(
                [jnp.zeros_like(y[:, :1, :]), y[:, :-1, :]], axis=1
            )  # (M, H, D_y)
            if self.local_window_prev_only:
                y_in = jnp.concatenate([y_prev, y], axis=-1)  # (M, H, 2*D_y)
            else:
                y_next = jnp.concatenate(
                    [y[:, 1:, :], jnp.zeros_like(y[:, -1:, :])], axis=1
                )  # (M, H, D_y)
                y_in = jnp.concatenate([y_prev, y, y_next], axis=-1)  # (M, H, 3*D_y)
        else:
            y_in = y  # (M, H, D_y)

        y_T = jnp.transpose(y_in, (1, 0, 2))  # (H, M, D_in)
        h0  = jnp.zeros((M, d_hidden))         # (M, D_hidden)

        # scan step: vmap GRUCell over batch dim
        cell = self.cell  # capture for closure

        def step(h: jnp.ndarray, y_t: jnp.ndarray):
            # h: (M, D_hidden), y_t: (M, D_in)
            h_next = jax.vmap(cell)(y_t, h)  # (M, D_hidden)
            return h_next, h_next

        _, h_seq = jax.lax.scan(step, h0, y_T)  # h_seq: (H, M, D_hidden)

        h_seq = jnp.transpose(h_seq, (1, 0, 2))  # (M, H, D_hidden)
        return _apply_linear(self.proj, h_seq)    # (M, H, D_enc)


# ---------------------------------------------------------------------------
# Transformer-based y-encoder
# ---------------------------------------------------------------------------

class _LearnedPosEnc(eqx.Module):
    """Learned positional embeddings. (M, H, D) -> (M, H, D)."""
    emb: jnp.ndarray  # (max_len, D)

    def __init__(self, max_len: int, d_model: int, key: jr.PRNGKey):
        self.emb = 0.02 * jr.normal(key, (max_len, d_model))

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        H = x.shape[1]
        return x + self.emb[:H]   # broadcast over M


class _MHSelfAttention(eqx.Module):
    """
    Multi-head self-attention.

    Input/output: (M, H, D_model)

    Dimension trace (inside __call__)
    ----------------------------------
    Q, K, V  : (M, H, D_model)  via _apply_linear
    Q, K, V  : (M, n_heads, H, d_k)  after reshape+transpose
    scores   : (M, n_heads, H, H)
    out      : (M, n_heads, H, d_k)  after softmax + matmul
    out      : (M, H, D_model)       after transpose+reshape
    output   : (M, H, D_model)       after W_o
    """

    W_q: eqx.nn.Linear
    W_k: eqx.nn.Linear
    W_v: eqx.nn.Linear
    W_o: eqx.nn.Linear
    n_heads: int = eqx.field(static=True)
    d_model: int = eqx.field(static=True)

    def __init__(self, d_model: int, n_heads: int, key: jr.PRNGKey):
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        kq, kk, kv, ko = jr.split(key, 4)
        self.W_q = eqx.nn.Linear(d_model, d_model, key=kq)
        self.W_k = eqx.nn.Linear(d_model, d_model, key=kk)
        self.W_v = eqx.nn.Linear(d_model, d_model, key=kv)
        self.W_o = eqx.nn.Linear(d_model, d_model, key=ko)

    def __call__(
        self,
        x: jnp.ndarray,                        # (M, H, D_model)
        mask: Optional[jnp.ndarray] = None,    # (M, n_heads, H, H) bool, True=keep
    ) -> jnp.ndarray:
        M, H, _ = x.shape
        d_k   = self.d_model // self.n_heads
        scale = d_k ** -0.5

        # Project: each _apply_linear does one vmap over (M*H,)
        Q = _apply_linear(self.W_q, x)          # (M, H, D_model)
        K = _apply_linear(self.W_k, x)
        V = _apply_linear(self.W_v, x)

        # Reshape to multi-head layout
        def split_heads(t):
            # (M, H, D) -> (M, n_heads, H, d_k)
            t = t.reshape(M, H, self.n_heads, d_k)
            return jnp.transpose(t, (0, 2, 1, 3))

        Q, K, V = split_heads(Q), split_heads(K), split_heads(V)

        scores = jnp.matmul(Q, jnp.swapaxes(K, -2, -1)) * scale  # (M, n, H, H)
        if mask is not None:
            scores = jnp.where(mask, scores, -1e9)
        w   = jax.nn.softmax(scores, axis=-1)                      # (M, n, H, H)
        out = jnp.matmul(w, V)                                     # (M, n, H, d_k)

        # Merge heads: (M, n_heads, H, d_k) -> (M, H, D_model)
        out = jnp.transpose(out, (0, 2, 1, 3)).reshape(M, H, self.d_model)
        return _apply_linear(self.W_o, out)                        # (M, H, D_model)


class _TransformerBlock(eqx.Module):
    """Pre-norm Transformer block: LN->MHA->residual, LN->FFN->residual."""

    ln1:     eqx.nn.LayerNorm
    mha:     _MHSelfAttention
    ln2:     eqx.nn.LayerNorm
    linear1: eqx.nn.Linear    # D_model -> D_ff
    linear2: eqx.nn.Linear    # D_ff -> D_model

    def __init__(self, d_model: int, n_heads: int, d_ff: int, key: jr.PRNGKey):
        k1, k2, km = jr.split(key, 3)
        self.ln1     = eqx.nn.LayerNorm((d_model,))
        self.mha     = _MHSelfAttention(d_model, n_heads, km)
        self.ln2     = eqx.nn.LayerNorm((d_model,))
        self.linear1 = eqx.nn.Linear(d_model, d_ff,    key=k1)
        self.linear2 = eqx.nn.Linear(d_ff,    d_model, key=k2)

    def __call__(
        self,
        x: jnp.ndarray,                        # (M, H, D_model)
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # LN applied over last dim; _apply_linear handles (..., d) shapes
        h  = _apply_linear(self.ln1, x)
        h  = self.mha(h, mask)
        x  = x + h
        h  = _apply_linear(self.ln2, x)
        h  = _apply_linear(self.linear1, h)
        h  = jax.nn.gelu(h)
        h  = _apply_linear(self.linear2, h)
        return x + h


class TransformerSequenceEncoder(eqx.Module):
    """
    Encode y_{1:H} with a Transformer.

    Input : y  (M, H, D_y)
    Output: enc_y (M, H, D_enc)

    When local_window=True each position t attends only to {t-1, t, t+1} via a
    boolean attention mask — the output at t therefore depends only on the
    immediately neighbouring observations.

    Dimension trace
    ---------------
    after input_proj : (M, H, D_model)
    after pos_enc    : (M, H, D_model)
    after n blocks   : (M, H, D_model)
    after output_proj: (M, H, D_enc)

    Efficiency notes
    ----------------
    * MHA uses batched matmul — no Python loop over heads or batch.
    * FFN via _apply_linear (single vmap over M*H).
    * All ops fuse well under XLA.
    """

    input_proj:   eqx.nn.Linear
    pos_enc:      _LearnedPosEnc
    blocks:       List[_TransformerBlock]
    output_proj:  eqx.nn.Linear
    local_window: bool = eqx.field(static=True)

    def __init__(
        self,
        d_y:          int,
        d_enc:        int,
        key:          jr.PRNGKey,
        d_model:      int  = 64,
        n_heads:      int  = 4,
        d_ff:         int  = 256,
        n_layers:     int  = 3,
        max_len:      int  = 512,
        local_window: bool = False,
    ):
        keys = jr.split(key, 3 + n_layers)
        self.input_proj  = eqx.nn.Linear(d_y,     d_model, key=keys[0])
        self.pos_enc     = _LearnedPosEnc(max_len, d_model, keys[1])
        self.blocks      = [
            _TransformerBlock(d_model, n_heads, d_ff, keys[2 + i])
            for i in range(n_layers)
        ]
        self.output_proj  = eqx.nn.Linear(d_model, d_enc, key=keys[-1])
        self.local_window = local_window

    def __call__(
        self,
        y: jnp.ndarray,                        # (M, H, D_y)
        mask: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        M, H, _ = y.shape
        h = _apply_linear(self.input_proj, y)  # (M, H, D_model)
        h = self.pos_enc(h)                    # (M, H, D_model)

        if self.local_window:
            # Build ±1 windowed mask: position t attends to {t-1, t, t+1} only.
            # Shape (1, 1, H, H) — broadcast over batch and heads.
            idx  = jnp.arange(H)
            win  = jnp.abs(idx[:, None] - idx[None, :]) <= 1  # (H, H) bool
            mask = win[None, None, :, :]                       # (1, 1, H, H)

        for block in self.blocks:
            h = block(h, mask)                 # (M, H, D_model)
        return _apply_linear(self.output_proj, h)  # (M, H, D_enc)


# ---------------------------------------------------------------------------
# Unified StateConditionEncoder
# ---------------------------------------------------------------------------

class StateConditionEncoder(eqx.Module):
    """
    Encode (y_{1:H}, u_{1:H}, x_0) -> condition (M, H, D_out) for the flow.

    Pipeline (shared by both backends):

        enc_y  = y_encoder(y)                    # (M, H, D_enc)
        enc_x0 = x0_proj(x_0)                   # (M, D_enc)
               = broadcast to (M, H, D_enc)
        cat    = [enc_y | u | enc_x0]            # (M, H, D_enc + D_u + D_enc)
        cond   = out_proj(cat)                   # (M, H, D_out)

    Dimension check
    ---------------
    enc_y    : (M, H, D_enc)
    u        : (M, H, D_u)          — passed in directly
    enc_x0   : (M, D_enc) -> broadcast -> (M, H, D_enc)
    cat      : (M, H, D_enc + D_u + D_enc)
    cond     : (M, H, D_out)        — flow condition

    Parameters
    ----------
    d_y           : D_y  (observation dim)
    d_u           : D_u  (input/control dim)
    d_x           : D_x  (state dim, size of x_0)
    d_enc         : D_enc (internal encoding dim)
    d_out         : D_out (flow condition dim)
    encoder_type  : 'gru' | 'transformer'
    gru_hidden    : hidden size for GRU backend
    d_model       : Transformer d_model
    n_heads       : Transformer attention heads
    d_ff          : Transformer FFN hidden dim
    n_layers      : Transformer depth
    max_len       : Transformer max sequence length
    """

    y_encoder:    eqx.Module       # GRUSequenceEncoder | TransformerSequenceEncoder
    x0_proj:      eqx.nn.Linear    # D_x -> D_enc
    out_proj:     eqx.nn.Linear    # (D_enc + D_u + D_enc) -> D_out
    d_enc:        int  = eqx.field(static=True)
    d_out:        int  = eqx.field(static=True)
    encoder_type: str  = eqx.field(static=True)
    local_window: bool = eqx.field(static=True)
    local_window_prev_only: bool = eqx.field(static=True)

    def __init__(
        self,
        d_y:          int,
        d_u:          int,
        d_x:          int,
        d_enc:        int,
        d_out:        int,
        key:          jr.PRNGKey,
        encoder_type: str  = "gru",
        local_window: bool = False,
        local_window_prev_only: bool = False,
        # GRU kwargs
        gru_hidden:   int  = 64,
        # Transformer kwargs
        d_model:      int  = 64,
        n_heads:      int  = 4,
        d_ff:         int  = 256,
        n_layers:     int  = 3,
        max_len:      int  = 512,
    ):
        self.d_enc        = d_enc
        self.d_out        = d_out
        self.encoder_type = encoder_type.lower()
        self.local_window = local_window
        self.local_window_prev_only = local_window_prev_only

        k_enc, k_x0, k_out = jr.split(key, 3)

        if self.encoder_type == "transformer":
            self.y_encoder = TransformerSequenceEncoder(
                d_y=d_y, d_enc=d_enc, key=k_enc,
                d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, n_layers=n_layers, max_len=max_len,
                local_window=local_window,
            )
        else:
            self.y_encoder = GRUSequenceEncoder(
                d_y=d_y, d_hidden=gru_hidden, d_enc=d_enc, key=k_enc,
                local_window=local_window,
                local_window_prev_only=local_window_prev_only,
            )

        self.x0_proj  = eqx.nn.Linear(d_x,                   d_enc, key=k_x0)
        self.out_proj = eqx.nn.Linear(d_enc + d_u + d_enc,   d_out, key=k_out)

    def __call__(
        self,
        y:  jnp.ndarray,   # (M, H, D_y)
        u:  jnp.ndarray,   # (M, H, D_u)
        x0: jnp.ndarray,   # (M, D_x)
    ) -> jnp.ndarray:
        """
        Returns cond of shape (M, H, D_out).

        Dimension trace
        ---------------
        enc_y      : (M, H, D_enc)
        enc_x0_vec : (M, D_enc)
        enc_x0     : (M, H, D_enc)  via broadcast
        cat        : (M, H, D_enc + D_u + D_enc)
        cond       : (M, H, D_out)
        """
        M, H, _ = y.shape

        enc_y      = self.y_encoder(y)                    # (M, H, D_enc)
        enc_x0_vec = jax.vmap(self.x0_proj)(x0)          # (M, D_enc)
        enc_x0     = jnp.broadcast_to(
            enc_x0_vec[:, None, :], (M, H, self.d_enc)
        )                                                  # (M, H, D_enc)

        cat  = jnp.concatenate([enc_y, u, enc_x0], axis=-1)   # (M, H, D_enc+D_u+D_enc)
        cond = _apply_linear(self.out_proj, cat)               # (M, H, D_out)
        return cond


# ---------------------------------------------------------------------------
# AmortizedStateFlow  (flow transforms unchanged; encoder replaced)
# ---------------------------------------------------------------------------

# Import the flow building blocks that are kept as-is.
from bayesinf.models.normalizing_flow import (
    ActNorm,
    RandomPermutation,
    AffineCoupling,
    SplineCoupling,
    MAFLayer,
)
from bayesinf.models.flow_matching import VectorField as FlowMatchingVectorField
from bayesinf.models.score_based_diffusion_model import (
    SinusoidalPosEmb,
    NoiseNet as ScoreNoiseNet,
)


def _mask(dim: int, parity: int) -> jnp.ndarray:
    return jnp.array([(i + parity) % 2 for i in range(dim)], dtype=jnp.float32)


def build_cond_flow_transforms(
    input_dim:         int,
    cond_dim:          int,
    n_blocks:          int,
    hidden_layer_size: int,
    n_hidden:          int,
    flow_type:         str,
    key:               jr.PRNGKey,
    num_bins:          int = 4,
) -> List[eqx.Module]:
    """Unchanged flow transform builder (RealNVP / NSF / MAF)."""
    flow_type  = flow_type.lower()
    transforms = []
    hidden_sizes = [hidden_layer_size] * n_hidden

    for i in range(n_blocks // 2):
        k_an, k_perm, k1, k2 = jr.split(jr.fold_in(key, i), 4)
        transforms.append(ActNorm(input_dim))
        transforms.append(RandomPermutation(input_dim, k_perm))
        if flow_type == "nsf":
            transforms += [
                SplineCoupling(input_dim, hidden_sizes, cond_dim,
                               _mask(input_dim, i % 2),        k1, num_bins=num_bins),
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
                               _mask(input_dim, i % 2),        k1),
                AffineCoupling(input_dim, hidden_sizes, cond_dim,
                               _mask(input_dim, (i+1) % 2),   k2),
            ]
    transforms.append(ActNorm(input_dim))
    return transforms


def _sbm_marginal_prob(
    x: jnp.ndarray, t: jnp.ndarray,
    beta_min: float, beta_max: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    log_mean_coeff = (
        -0.25 * t**2 * (beta_max - beta_min) - 0.5 * t * beta_min
    )
    log_mean_coeff = log_mean_coeff[..., None]
    mean = x * jnp.exp(log_mean_coeff)
    std  = jnp.sqrt(jnp.maximum(1.0 - jnp.exp(2.0 * log_mean_coeff), 1e-8))
    return mean, std


class AmortizedStateFlow(eqx.Module):
    """
    Amortized Bayesian state estimator.

    Goal: estimate x_{1:H} given y_{1:H}, u_{1:H}, x_0.

    Shapes
    ------
    y    : (M, H, D_y)
    u    : (M, H, D_u)
    x0   : (M, D_x)
    x    : (M, H, D_x)   — states to infer / flow input
    cond : (M, H, D_out) — flow condition produced by encoder
    z    : (M, H, D_x)   — latent (flow output / prior sample)

    Flow: x_{1:H} <-> z_{0:H-1}, both (M, H, D_x).
    Transforms are applied independently per (M*H) token (flat batch).
    """

    encoder:     StateConditionEncoder
    transforms:  List[eqx.Module]
    vector_field: Optional[FlowMatchingVectorField]
    time_emb:    Optional[SinusoidalPosEmb]
    score_net:   Optional[ScoreNoiseNet]

    state_dim:   int   = eqx.field(static=True)
    d_out:       int   = eqx.field(static=True)
    flow_type:   str   = eqx.field(static=True)
    dt:          float = eqx.field(static=True)
    sbm_beta_min: float = eqx.field(static=True)
    sbm_beta_max: float = eqx.field(static=True)

    def __init__(
        self,
        d_x:          int,          # state dim
        d_y:          int,          # observation dim
        d_u:          int,          # control/input dim
        d_enc:        int,          # internal encoder dim
        d_out:        int,          # flow condition dim (D_out)
        key:          jr.PRNGKey,
        # Flow options
        flow_type:    str   = "realnvp",
        n_coupling:   int   = 4,
        hidden_size:  int   = 128,
        n_hidden:     int   = 2,
        dt:           float = 0.01,
        sbm_beta_min: float = 0.1,
        sbm_beta_max: float = 20.0,
        sbm_time_dim: int   = 32,
        # Encoder options
        encoder_type: str   = "gru",
        local_window: bool  = False,
        gru_hidden:   int   = 64,
        d_model:      int   = 64,
        n_heads:      int   = 4,
        d_ff:         int   = 256,
        n_layers:     int   = 3,
        max_len:      int   = 512,
    ):
        self.state_dim    = d_x
        self.d_out        = d_out
        self.flow_type    = flow_type.lower()
        self.dt           = dt
        self.sbm_beta_min = sbm_beta_min
        self.sbm_beta_max = sbm_beta_max

        k_enc, k_flow = jr.split(key)

        self.encoder = StateConditionEncoder(
            d_y=d_y, d_u=d_u, d_x=d_x,
            d_enc=d_enc, d_out=d_out, key=k_enc,
            encoder_type=encoder_type,
            local_window=local_window,
            gru_hidden=gru_hidden,
            d_model=d_model, n_heads=n_heads,
            d_ff=d_ff, n_layers=n_layers, max_len=max_len,
        )

        if self.flow_type == "fm":
            self.transforms   = []
            self.vector_field = FlowMatchingVectorField(
                d_x, d_out, hidden_size, n_hidden, k_flow
            )
            self.time_emb  = None
            self.score_net = None
        elif self.flow_type == "sbm":
            self.transforms   = []
            self.vector_field = None
            k_te, k_net = jr.split(k_flow)
            self.time_emb  = SinusoidalPosEmb(dim=sbm_time_dim)
            self.score_net = ScoreNoiseNet(
                d_x, d_out, sbm_time_dim,
                [hidden_size] * n_hidden, key=k_net,
            )
        else:
            self.transforms   = build_cond_flow_transforms(
                d_x, d_out, n_coupling, hidden_size, n_hidden,
                self.flow_type, k_flow,
            )
            self.vector_field = None
            self.time_emb     = None
            self.score_net    = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(
        self,
        y:  jnp.ndarray,   # (M, H, D_y)
        u:  jnp.ndarray,   # (M, H, D_u)
        x0: jnp.ndarray,   # (M, D_x)
    ) -> jnp.ndarray:
        """Returns cond (M, H, D_out)."""
        return self.encoder(y, u, x0)

    def _flow_fwd(
        self,
        x:        jnp.ndarray,   # (M*H, D_x)
        cond_emb: jnp.ndarray,   # (M*H, D_out)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Forward pass: x -> z, accumulate log|det J|.
        Both x and cond_emb are flat (M*H, *) — one token per row.

        Dimension trace
        ---------------
        x, z     : (M*H, D_x)
        logdet   : (M*H,)
        """
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
        z:        jnp.ndarray,   # (M*H, D_x)
        cond_emb: jnp.ndarray,   # (M*H, D_out)
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Inverse pass: z -> x."""
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

    def _flat(
        self,
        x:    jnp.ndarray,   # (M, H, D_x)
        cond: jnp.ndarray,   # (M, H, D_out)
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Tuple[int,int,int]]:
        """
        Flatten (M, H, *) -> (M*H, *) for flow transforms.
        Returns flat_x, flat_cond, (M, H, D_x) for later reshape.
        """
        M, H, D_x   = x.shape
        D_out        = cond.shape[-1]
        return (
            x.reshape(M * H, D_x),
            cond.reshape(M * H, D_out),
            (M, H, D_x),
        )

    # ------------------------------------------------------------------
    # Prior log-probabilities
    # ------------------------------------------------------------------

    def _log_prior_iid(self, z: jnp.ndarray) -> jnp.ndarray:
        """z (M, H, D_x) iid N(0,I). Returns (M,)."""
        return -0.5 * (
            jnp.sum(z ** 2, axis=(-2, -1))
            + z.shape[-2] * z.shape[-1] * jnp.log(2 * jnp.pi)
        )

    def _log_prior_wiener_increment(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z_0 ~ N(0, dt I),  z_t - z_{t-1} ~ N(0, dt I).
        z: (M, H, D_x) -> (M,).
        """
        M, H, D_x = z.shape
        dt = self.dt
        log_p0 = -0.5 * (
            jnp.sum(z[:, 0, :] ** 2, axis=-1) / dt
            + D_x * jnp.log(2 * jnp.pi * dt)
        )
        if H == 1:
            return log_p0
        incr   = z[:, 1:, :] - z[:, :-1, :]          # (M, H-1, D_x)
        log_pi = -0.5 * (
            jnp.sum(incr ** 2, axis=-1) / dt
            + D_x * jnp.log(2 * jnp.pi * dt)
        )                                              # (M, H-1)
        return log_p0 + jnp.sum(log_pi, axis=-1)      # (M,)

    def _log_prior_wiener_scaled_t(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        z_t ~ N(0, (t+1)^2 dt I) per timestep.
        z: (M, H, D_x) -> (M,).

        Uses lax.scan over H — efficient for large H.
        """
        M, H, D_x = z.shape
        dt = self.dt

        def body(log_p, t):
            z_t      = z[:, t, :]                         # (M, D_x)
            scale_sq = (t + 1) ** 2 * dt
            lp_t = -0.5 * (
                jnp.sum(z_t ** 2, axis=-1) / scale_sq
                + D_x * jnp.log(2 * jnp.pi * scale_sq)
            )                                              # (M,)
            return log_p + lp_t, None

        log_p, _ = jax.lax.scan(body, jnp.zeros(M), jnp.arange(H))
        return log_p  # (M,)

    # ------------------------------------------------------------------
    # Public API: log_prob, sample
    # ------------------------------------------------------------------

    def log_prob(
        self,
        x:    jnp.ndarray,   # (M, H, D_x)   — observed/target states
        y:    jnp.ndarray,   # (M, H, D_y)
        u:    jnp.ndarray,   # (M, H, D_u)
        x0:   jnp.ndarray,   # (M, D_x)
        prior: str = "wiener_scaled_t",
    ) -> jnp.ndarray:
        """
        Log p(x | y, u, x0).  Returns (M,).

        prior options: 'wiener_scaled_t' | 'wiener_increment' | 'iid'

        Dimension trace
        ---------------
        cond      : (M, H, D_out)
        x_flat    : (M*H, D_x)
        cond_flat : (M*H, D_out)
        z_flat    : (M*H, D_x)
        logdet_flat: (M*H,)
        z         : (M, H, D_x)
        logdet    : (M, H)  -> sum -> (M,)
        log_prior : (M,)
        log_prob  : (M,)
        """
        cond = self._encode(y, u, x0)                      # (M, H, D_out)
        x_flat, cond_flat, (M, H, D_x) = self._flat(x, cond)

        z_flat, logdet_flat = self._flow_fwd(x_flat, cond_flat)

        z      = z_flat.reshape(M, H, D_x)                 # (M, H, D_x)
        logdet = logdet_flat.reshape(M, H).sum(axis=-1)    # (M,)

        if prior == "wiener_scaled_t":
            log_p = self._log_prior_wiener_scaled_t(z)
        elif prior == "wiener_increment":
            log_p = self._log_prior_wiener_increment(z)
        else:
            log_p = self._log_prior_iid(z)

        log_prob = log_p + logdet   # (M,)
        # Guard: inf - inf = NaN when ActNorm scales blow up during early training.
        # Replace non-finite samples with a large negative value (high NLL, zero gradient)
        # so they do not corrupt parameters.
        log_prob = jnp.where(jnp.isfinite(log_prob), log_prob, jnp.full_like(log_prob, -1e4))
        return log_prob

    def nll_loss(
        self,
        x:     jnp.ndarray,   # (M, H, D_x)
        y:     jnp.ndarray,   # (M, H, D_y)
        u:     jnp.ndarray,   # (M, H, D_u)
        x0:    jnp.ndarray,   # (M, D_x)
        prior: str = "wiener_scaled_t",
    ) -> jnp.ndarray:
        """Scalar NLL per dimension: -mean(log_prob) / (H * D_x).

        Normalising by H * D_x keeps the loss scale invariant to horizon length
        and state dimension, making loss values comparable across runs with
        different --horizon, --seg_len, or --N settings.  The iid Gaussian
        baseline is then 0.5 * log(2*pi) ≈ 0.919 nats/dim regardless of H or N.
        """
        _, H, D_x = x.shape
        return -jnp.mean(self.log_prob(x, y, u, x0, prior)) / (H * D_x)

    def sample(
        self,
        y:           jnp.ndarray,   # (M, H, D_y)
        u:           jnp.ndarray,   # (M, H, D_u)
        x0:          jnp.ndarray,   # (M, D_x)
        key:         jr.PRNGKey,
        num_samples: int   = 1,
        prior:       str   = "wiener_scaled_t",
        fm_steps:    int   = 100,
        sbm_steps:   int   = 1000,
    ) -> jnp.ndarray:
        """
        Sample x_{1:H} ~ p(x | y, u, x0).

        Returns (num_samples, M, H, D_x).

        Dimension trace
        ---------------
        cond        : (M, H, D_out)
        cond_rep    : (num_samples, M, H, D_out)
        cond_flat   : (num_samples*M*H, D_out)
        z_flat      : (num_samples*M*H, D_x)
        x_flat      : (num_samples*M*H, D_x)
        x           : (num_samples, M, H, D_x)
        """
        M, H, _ = y.shape
        cond = self._encode(y, u, x0)                       # (M, H, D_out)

        if self.flow_type == "fm":
            return self._sample_fm(cond, key, num_samples, fm_steps)
        if self.flow_type == "sbm":
            return self._sample_sbm(cond, key, num_samples, sbm_steps)

        # --- draw z from prior ---
        if prior == "wiener_scaled_t":
            keys  = jr.split(key, H)
            z_list = [
                (t + 1) * jnp.sqrt(self.dt)
                * jr.normal(keys[t], (num_samples, M, self.state_dim))
                for t in range(H)
            ]
            z = jnp.stack(z_list, axis=2)                  # (S, M, H, D_x)
        elif prior == "wiener_increment":
            k0, ki = jr.split(key)
            z0   = jnp.sqrt(self.dt) * jr.normal(k0, (num_samples, M, self.state_dim))
            if H > 1:
                incr = jnp.sqrt(self.dt) * jr.normal(
                    ki, (num_samples, M, H - 1, self.state_dim)
                )
                z = jnp.concatenate(
                    [z0[:, :, None, :],
                     z0[:, :, None, :] + jnp.cumsum(incr, axis=2)],
                    axis=2,
                )
            else:
                z = z0[:, :, None, :]
        else:
            z = jr.normal(key, (num_samples, M, H, self.state_dim))

        # --- broadcast cond and invert flow ---
        # cond (M, H, D_out) -> (S, M, H, D_out)
        cond_rep  = jnp.broadcast_to(cond[None], (num_samples, M, H, self.d_out))
        n_total   = num_samples * M * H
        z_flat    = z.reshape(n_total, self.state_dim)
        cond_flat = cond_rep.reshape(n_total, self.d_out)

        x_flat, _ = self._flow_inv(z_flat, cond_flat)      # (num_samples*M*H, D_x)
        return x_flat.reshape(num_samples, M, H, self.state_dim)

    # ------------------------------------------------------------------
    # FM / SBM sampling
    # ------------------------------------------------------------------

    def _sample_fm(
        self,
        cond:        jnp.ndarray,   # (M, H, D_out)
        key:         jr.PRNGKey,
        num_samples: int,
        steps:       int,
    ) -> jnp.ndarray:
        """ODE integration (Euler). Returns (S, M, H, D_x)."""
        M, H, D_out = cond.shape
        S = num_samples
        n = S * M * H
        cond_rep  = jnp.broadcast_to(cond[None], (S, M, H, D_out))
        cond_flat = cond_rep.reshape(n, D_out)              # (n, D_out)
        z0        = jr.normal(key, (n, self.state_dim))     # (n, D_x)
        dt        = 1.0 / steps
        vf        = self.vector_field

        def step(z_curr, i):
            t_val = i * dt
            t_b   = jnp.full((n,), t_val)
            v = jax.vmap(vf)(t_b, z_curr, cond_flat)       # (n, D_x)
            return z_curr + v * dt, None

        z1, _ = jax.lax.scan(step, z0, jnp.arange(steps))
        return z1.reshape(S, M, H, self.state_dim)

    def _sample_sbm(
        self,
        cond:        jnp.ndarray,   # (M, H, D_out)
        key:         jr.PRNGKey,
        num_samples: int,
        steps:       int,
    ) -> jnp.ndarray:
        """Reverse SDE (Euler-Maruyama). Returns (S, M, H, D_x)."""
        M, H, D_out = cond.shape
        S = num_samples
        n = S * M * H
        cond_rep  = jnp.broadcast_to(cond[None], (S, M, H, D_out))
        cond_flat = cond_rep.reshape(n, D_out)
        z         = jr.normal(key, (n, self.state_dim))
        abs_dt    = 1.0 / steps
        sn        = self.score_net
        te        = self.time_emb

        def step_fn(carry, i):
            z_val, k = carry
            t_val = 1.0 - i / steps
            t_b   = jnp.full((n,), t_val)
            t_emb = te(t_b)
            _, std = _sbm_marginal_prob(
                jnp.ones_like(z_val), t_b,
                self.sbm_beta_min, self.sbm_beta_max,
            )
            pred  = jax.vmap(sn)(z_val, t_emb, cond_flat)
            score = pred / (std + 1e-6)
            beta  = self.sbm_beta_min + t_val * (self.sbm_beta_max - self.sbm_beta_min)
            rev_drift = -0.5 * beta * z_val - beta * score
            k, k_dw = jr.split(k)
            dw    = jr.normal(k_dw, z_val.shape) * jnp.sqrt(abs_dt)
            z_next = z_val - rev_drift * abs_dt + jnp.sqrt(beta) * dw
            return (z_next, k), None

        (z_final, _), _ = jax.lax.scan(
            step_fn, (z, key), jnp.arange(steps)
        )
        return z_final.reshape(S, M, H, self.state_dim)


# ---------------------------------------------------------------------------
# FM / SBM training loss functions
# ---------------------------------------------------------------------------

def fm_loss(
    model:    AmortizedStateFlow,
    x:        jnp.ndarray,   # (M, H, D_x)
    y:        jnp.ndarray,   # (M, H, D_y)
    u:        jnp.ndarray,   # (M, H, D_u)
    x0:       jnp.ndarray,   # (M, D_x)
    key:      jr.PRNGKey,
) -> jnp.ndarray:
    """
    Flow matching MSE loss.
    z_t = (1-t)*z0 + t*x,  target v = x - z0.

    Dimension trace
    ---------------
    cond  : (M, H, D_out)
    *_flat: (M*H, *)
    """
    cond     = model._encode(y, u, x0)              # (M, H, D_out)
    M, H, D_x = x.shape
    n         = M * H
    x_flat    = x.reshape(n, D_x)
    c_flat    = cond.reshape(n, model.d_out)

    kt, kz    = jr.split(key)
    t         = jr.uniform(kt, (n,))
    z0        = jr.normal(kz, (n, D_x))
    t_b       = t[:, None]
    z_t       = (1 - t_b) * z0 + t_b * x_flat
    v_target  = x_flat - z0

    v_pred = jax.vmap(model.vector_field)(t, z_t, c_flat)
    return jnp.mean((v_pred - v_target) ** 2)


def sbm_loss(
    model:    AmortizedStateFlow,
    x:        jnp.ndarray,   # (M, H, D_x)
    y:        jnp.ndarray,   # (M, H, D_y)
    u:        jnp.ndarray,   # (M, H, D_u)
    x0:       jnp.ndarray,   # (M, D_x)
    key:      jr.PRNGKey,
) -> jnp.ndarray:
    """
    Denoising score matching loss (VP-SDE).

    Dimension trace
    ---------------
    cond  : (M, H, D_out)
    *_flat: (M*H, *)
    """
    cond     = model._encode(y, u, x0)
    M, H, D_x = x.shape
    n         = M * H
    x_flat    = x.reshape(n, D_x)
    c_flat    = cond.reshape(n, model.d_out)

    kt, kn    = jr.split(key)
    t         = jr.uniform(kt, (n,), minval=1e-4, maxval=1.0)
    noise     = jr.normal(kn, (n, D_x))
    mean, std = _sbm_marginal_prob(x_flat, t, model.sbm_beta_min, model.sbm_beta_max)
    z_t       = mean + std * noise
    t_emb     = model.time_emb(t)

    pred = jax.vmap(model.score_net)(z_t, t_emb, c_flat)
    return jnp.mean((pred - (-noise)) ** 2)


def state_flow_loss(
    model:    AmortizedStateFlow,
    x:        jnp.ndarray,   # (M, H, D_x)
    y:        jnp.ndarray,   # (M, H, D_y)
    u:        jnp.ndarray,   # (M, H, D_u)
    x0:       jnp.ndarray,   # (M, D_x)
    key:      Optional[jr.PRNGKey] = None,
    prior:    str = "wiener_scaled_t",
) -> jnp.ndarray:
    """
    Unified training loss:
      NF  -> NLL
      FM  -> flow matching MSE
      SBM -> denoising score matching MSE
    """
    if model.flow_type == "fm":
        return fm_loss(model, x, y, u, x0, key)
    if model.flow_type == "sbm":
        return sbm_loss(model, x, y, u, x0, key)
    return model.nll_loss(x, y, u, x0, prior)
