"""Shared deterministic transformations for conditional and stochastic flows."""
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Any, List, Tuple

class ActNorm(eqx.Module):
    bias: jnp.ndarray
    logs: jnp.ndarray
    logscale_factor: float = eqx.field(static=True)
    eps: float = eqx.field(static=True)
    initialized: bool

    def __init__(self, dim: int, logscale_factor=1.0, eps=1e-6):
        self.bias = jnp.zeros((dim,))
        self.logs = jnp.zeros((dim,))
        self.logscale_factor = logscale_factor
        self.eps = eps
        self.initialized = False

    def initialize(self, x: jnp.ndarray):
        """
        Data-dependent initialization.
        Returns a NEW ActNorm instance with updated parameters.
        """
        # x: (batch, dim)
        mean = jnp.mean(x, axis=0)
        std = jnp.std(x, axis=0) + self.eps

        new_bias = -mean
        new_logs = -jnp.log(std) / self.logscale_factor

        # Return a copy of self with updated fields
        return eqx.tree_at(
            lambda t: (t.bias, t.logs, t.initialized),
            self,
            (new_bias, new_logs, True)
        )

    def __call__(self, x: jnp.ndarray, reverse: bool = False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # x: (batch, dim)
        # Note: In JAX/Equinox, we cannot check "if not self.initialized: self.initialize(x)"
        # inside the forward pass easily if JIT compiling.
        # The user should call .initialize() explicitly before training.

        # The notebook treats data-dependent ActNorm statistics as fixed.
        bias = jax.lax.stop_gradient(self.bias)
        logs = jax.lax.stop_gradient(self.logs) * self.logscale_factor

        if not reverse:
            z = (x + bias) * jnp.exp(logs)
            # ldj = sum(logs) for each batch element
            ldj_val = jnp.sum(logs)
            ldj = jnp.full(x.shape[:-1], ldj_val)
            return z, ldj
        else:
            z = x * jnp.exp(-logs) - bias
            ldj_val = -jnp.sum(logs)
            ldj = jnp.full(x.shape[:-1], ldj_val)
            return z, ldj


class RandomPermutation(eqx.Module):
    dim: int = eqx.field(static=True)
    perm: jnp.ndarray
    inv_perm: jnp.ndarray

    def __init__(self, dim: int, key: jax.Array):
        self.dim = dim
        self.perm = jax.random.permutation(key, jnp.arange(dim))
        self.inv_perm = jnp.argsort(self.perm)

    def __call__(self, x: jnp.ndarray, reverse: bool = False):
        if not reverse:
            return x[..., self.perm], jnp.zeros(x.shape[:-1])
        else:
            return x[..., self.inv_perm], jnp.zeros(x.shape[:-1])


class AffineCoupling(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    idx_a: Any = eqx.field(static=True)
    idx_b: Any = eqx.field(static=True)
    net: eqx.nn.MLP
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_sizes: List[int],
        cond_dim: int,
        mask: jnp.ndarray,
        key: jax.Array,
        dropout_rate: float = 0.0,
    ):
        self.dim = input_dim
        self.cond_dim = cond_dim
        self.dropout_rate = dropout_rate

        self.idx_a = tuple(np.flatnonzero(np.asarray(mask)).tolist())
        self.idx_b = tuple(np.flatnonzero(1 - np.asarray(mask)).tolist())

        in_dim = int(jnp.sum(mask)) + cond_dim
        out_dim = input_dim - int(jnp.sum(mask))

        # Equinox MLP
        # We want the final layer initialized to near-zero.
        # Equinox doesn't allow easy custom init per layer in the constructor
        # simply, so we construct, then modify.
        self.net = eqx.nn.MLP(
            in_size=in_dim,
            out_size=out_dim * 2,
            width_size=hidden_sizes[0] if hidden_sizes else 16, # simplify for list
            depth=len(hidden_sizes)+1,  # preserve the established checkpoint architecture
            activation=jax.nn.relu,
            key=key
        )

        # Initialize last layer to zeros for stability (identity init)
        final_layer_idx = len(self.net.layers) - 1
        w_zeros = jnp.zeros_like(self.net.layers[final_layer_idx].weight)
        b_zeros = jnp.zeros_like(self.net.layers[final_layer_idx].bias)

        new_final = eqx.tree_at(
            lambda l: (l.weight, l.bias),
            self.net.layers[final_layer_idx],
            (w_zeros, b_zeros)
        )
        self.net = eqx.tree_at(
            lambda n: n.layers[final_layer_idx],
            self.net,
            new_final
        )

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, reverse: bool = False,
                 key: jax.Array | None = None, inference: bool = True):
        # x: (batch, dim), cond: (batch, cond_dim)

        # Function to process a single batch element
        keys = jr.split(key if key is not None else jr.PRNGKey(0), len(x))

        def _step(x_i, c_i, key_i):
            x_a_flat = x_i[..., jnp.asarray(self.idx_a, dtype=jnp.int32)]
            inp = jnp.concatenate([x_a_flat, c_i], axis=-1)
            if self.dropout_rate and not inference:
                if key is None:
                    raise ValueError("Training with coupling dropout requires a PRNG key")
                keep = jr.bernoulli(key_i, 1.0 - self.dropout_rate, inp.shape)
                inp = jnp.where(keep, inp / (1.0 - self.dropout_rate), 0.0)
            st = self.net(inp)
            s, t = jnp.split(st, 2, axis=-1)
            s = jnp.tanh(s)

            x_b_flat = x_i[..., jnp.asarray(self.idx_b, dtype=jnp.int32)]

            if not reverse:
                z_b_flat = x_b_flat * jnp.exp(s) + t
                # Reconstruct full z
                z = jnp.zeros_like(x_i)
                z = z.at[..., jnp.asarray(self.idx_a, dtype=jnp.int32)].set(x_a_flat)
                z = z.at[..., jnp.asarray(self.idx_b, dtype=jnp.int32)].set(z_b_flat)
                ldj = jnp.sum(s)
                return z, ldj
            else:
                x_b_rec = (x_b_flat - t) * jnp.exp(-s)
                z = jnp.zeros_like(x_i)
                z = z.at[..., jnp.asarray(self.idx_a, dtype=jnp.int32)].set(x_a_flat)
                z = z.at[..., jnp.asarray(self.idx_b, dtype=jnp.int32)].set(x_b_rec)
                ldj = -jnp.sum(s)
                return z, ldj

        return jax.vmap(_step)(x, cond, keys)


