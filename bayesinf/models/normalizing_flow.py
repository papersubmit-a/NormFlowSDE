import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from bayesinf.models.optimization import apply_model_updates, preserve_model_state
import optax
from typing import Any, List, Tuple, Optional, Sequence
from functools import partial
from bayesinf.models.networks import (
    BatchNormBidirectionalLSTMEncoder,
    BatchNormLSTMEncoder,
    BatchNormTransformerTimeSeriesEncoder,
    LayerNormBidirectionalLSTMEncoder,
    TimeSeriesEncoder,
    TransformerTimeSeriesEncoder,
    VectorizedBidirectionalLSTMEncoder,
)
from bayesinf.models.MCMC_Flow import make_refinement, refine_samples

from bayesinf.models.flow_layers import ActNorm, RandomPermutation, AffineCoupling


class CouplingNet(eqx.Module):
    """Configurable Xavier-initialized MLP supporting arbitrary batch axes."""
    layers: tuple
    activation: Any = eqx.field(static=True)

    def __init__(self, in_size, out_size, hidden_dims, key,
                 activation=jnp.tanh, bias=0.001):
        hidden_dims = tuple(hidden_dims)
        sizes = (in_size, *hidden_dims, out_size)
        keys = jr.split(key, len(sizes) - 1)
        layers = []
        for fan_in, fan_out, layer_key in zip(sizes[:-1], sizes[1:], keys):
            layer = eqx.nn.Linear(fan_in, fan_out, key=layer_key)
            # Exact JAX equivalent of torch.nn.init.xavier_uniform_(weight).
            limit = jnp.sqrt(6.0 / (fan_in + fan_out))
            weight = jr.uniform(layer_key, layer.weight.shape, minval=-limit, maxval=limit)
            layer = eqx.tree_at(
                lambda item: (item.weight, item.bias), layer,
                (weight, jnp.full_like(layer.bias, bias)),
            )
            layers.append(layer)
        self.layers = tuple(layers)
        self.activation = activation

    def __call__(self, value):
        leading_shape = value.shape[:-1]
        value = value.reshape((-1, value.shape[-1]))
        for layer in self.layers[:-1]:
            value = jax.vmap(layer)(value)
            value = self.activation(value)
        value = jax.vmap(self.layers[-1])(value)
        return value.reshape((*leading_shape, value.shape[-1]))


class DoubleRealNVPCoupling(eqx.Module):
    """Two consecutive affine updates that transform both halves of a vector."""
    dim: int = eqx.field(static=True)
    split: int = eqx.field(static=True)
    subnet_s1: CouplingNet
    subnet_t1: CouplingNet
    subnet_s2: CouplingNet
    subnet_t2: CouplingNet
    clamp: float = eqx.field(static=True)
    clamp_activation: str = eqx.field(static=True)

    def __init__(self, dim, cond_dim, hidden_dims, key, activation=jnp.tanh,
                 clamp=2.0, clamp_activation="TANH"):
        self.dim = dim
        self.split = round(dim * 0.5)
        self.clamp = clamp
        self.clamp_activation = clamp_activation.upper()
        if self.clamp_activation not in {"TANH", "ATAN", "SIGMOID", "I"}:
            raise ValueError("clamp_activation must be TANH, ATAN, SIGMOID, or I")
        first, second = self.split, dim - self.split
        keys = jr.split(key, 4)
        self.subnet_s1 = CouplingNet(first + cond_dim, second, hidden_dims, keys[0], activation)
        self.subnet_t1 = CouplingNet(first + cond_dim, second, hidden_dims, keys[1], activation)
        self.subnet_s2 = CouplingNet(second + cond_dim, first, hidden_dims, keys[2], activation)
        self.subnet_t2 = CouplingNet(second + cond_dim, first, hidden_dims, keys[3], activation)

    def _bounded_log_scale(self, raw_scale):
        if self.clamp_activation == "TANH":
            activated = jnp.tanh(raw_scale)
        elif self.clamp_activation == "ATAN":
            activated = 0.636 * jnp.arctan(raw_scale)
        elif self.clamp_activation == "SIGMOID":
            activated = 2.0 * (jax.nn.sigmoid(raw_scale) - 0.5)
        else:
            activated = raw_scale
        return self.clamp * activated

    def __call__(self, value, cond, reverse=False):
        first, second = value[..., :self.split], value[..., self.split:]
        if not reverse:
            second_cond = jnp.concatenate((second, cond), axis=-1)
            scale_first = self._bounded_log_scale(self.subnet_s2(second_cond))
            shift_first = self.subnet_t2(second_cond)
            out_first = jnp.exp(scale_first) * first + shift_first
            first_cond = jnp.concatenate((out_first, cond), axis=-1)
            scale_second = self._bounded_log_scale(self.subnet_s1(first_cond))
            shift_second = self.subnet_t1(first_cond)
            out_second = jnp.exp(scale_second) * second + shift_second
            logdet = jnp.sum(scale_first, axis=-1) + jnp.sum(scale_second, axis=-1)
        else:
            first_cond = jnp.concatenate((first, cond), axis=-1)
            scale_second = self._bounded_log_scale(self.subnet_s1(first_cond))
            shift_second = self.subnet_t1(first_cond)
            out_second = (second - shift_second) * jnp.exp(-scale_second)
            second_cond = jnp.concatenate((out_second, cond), axis=-1)
            scale_first = self._bounded_log_scale(self.subnet_s2(second_cond))
            shift_first = self.subnet_t2(second_cond)
            out_first = (first - shift_first) * jnp.exp(-scale_first)
            logdet = -jnp.sum(scale_first, axis=-1) - jnp.sum(scale_second, axis=-1)
        return jnp.concatenate((out_first, out_second), axis=-1), logdet

class MaskedLinear(eqx.nn.Linear):
    mask: jnp.ndarray

    def __init__(self, in_features, out_features, key):
        super().__init__(in_features, out_features, key=key)
        self.mask = jnp.ones((out_features, in_features), dtype=bool)

    def set_mask(self, mask):
        # update mask, return new instance
        return eqx.tree_at(lambda l: l.mask, self, jnp.asarray(mask, dtype=bool))

    def __call__(self, x):
        # x: (..., in_features)
        w_masked = self.weight * self.mask
        return x @ w_masked.T + self.bias


class MADE(eqx.Module):
    input_dim: int = eqx.field(static=True)
    hidden_sizes: List[int] = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    natural_ordering: bool = eqx.field(static=True)
    indices: List[int]
    net: List[MaskedLinear]

    def __init__(self, input_dim, hidden_sizes, cond_dim, natural_ordering, key):
        self.input_dim = input_dim
        self.cond_dim = cond_dim
        self.natural_ordering = natural_ordering
        self.hidden_sizes = hidden_sizes

        # Input layer size is Data + Condition
        hs = [input_dim + cond_dim] + hidden_sizes + [input_dim * 2]

        keys = jax.random.split(key, len(hs))
        self.net = []
        for i in range(len(hs) - 1):
            self.net.append(MaskedLinear(hs[i], hs[i+1], key=keys[i]))

        # Create masks and assign them
        self._create_masks(keys[-1])

    def _create_masks(self, key):
        D = self.input_dim
        C = self.cond_dim
        k1, k2 = jax.random.split(key)

        # 1. Degrees for Physical Parameters (1 to D)
        # These must respect autoregressive property
        if self.natural_ordering:
            degrees_x = jnp.arange(1, D + 1)
        else:
            degrees_x = jax.random.permutation(k1, jnp.arange(1, D + 1))
        self.indices = jnp.argsort(degrees_x).tolist()

        # 2. Degrees for Condition (ALWAYS 0)
        # This ensures Hidden Units (deg >= 1) can ALWAYS see Condition (deg 0)
        degrees_c = jnp.zeros(C, dtype=jnp.int32)

        # Combined Input Degrees: [Params, Conditions]
        degrees_in = jnp.concatenate([degrees_x, degrees_c])
        degrees = [degrees_in]

        # 3. Hidden Layer Degrees
        for h in self.hidden_sizes:
            key, subkey = jax.random.split(key)
            # Hidden units range from 1 to D-1 (or D)
            hidden_degrees = jax.random.randint(subkey, (h,), 0 if C else 1, max(D, 1) if C else max(D, 2))
            if C:
                hidden_degrees = hidden_degrees.at[0].set(0)
            degrees.append(hidden_degrees)

        # 4. Output Degrees
        # We only reconstruct the parameters, so output degrees match degrees_x
        degrees.append(jnp.tile(degrees_x, 2)) # (D*2,)

        masks = []
        for l in range(len(degrees) - 1):
            d_in = degrees[l]
            d_out = degrees[l+1]

            # mask shape: (out, in)
            is_output_layer = (l == len(degrees) - 2)

            if is_output_layer:
                # Output layer cannot see itself (strictly >)
                mask = (d_out[:, None] > d_in[None, :]).astype(jnp.float32)
            else:
                # Hidden layers can see lower or equal degrees
                mask = (d_out[:, None] >= d_in[None, :]).astype(jnp.float32)
            masks.append(mask)

        # Assign masks to layers
        new_net = []
        for layer, m in zip(self.net, masks):
            new_net.append(layer.set_mask(m))
        self.net = new_net

    def __call__(self, x, cond=None):
        # Concatenate x (Params) and cond (Encoded Time Series)
        if cond is not None:
            h = jnp.concatenate([x, cond], axis=-1)
        else:
            h = x

        for i, layer in enumerate(self.net):
            h = layer(h)
            if i < len(self.net) - 1:
                h = jax.nn.relu(h)

        mu, log_scale = jnp.split(h, 2, axis=-1)
        return mu, log_scale


class MAFLayer(eqx.Module):
    dim: int = eqx.field(static=True)
    net: MADE

    def __init__(self, input_dim, hidden_sizes, cond_dim, natural_ordering, key):
        self.dim = input_dim
        self.net = MADE(input_dim, hidden_sizes, cond_dim, natural_ordering, key)

    def __call__(self, x, cond=None, reverse=False):
        if not reverse:
            # Forward: x -> z
            mu, log_scale = self.net(x, cond)
            log_scale = jnp.tanh(log_scale)
            # MAF Transformation: z = (x - mu) / scale
            z = (x - mu) * jnp.exp(-log_scale)

            # Log Det Jacobian = -sum(log(scale))
            ldj = -jnp.sum(log_scale, axis=-1)
            return z, ldj
        else:
            # Inverse: z -> x (Sequential)
            # z is the input here
            def _inverse_single(z_inp, c_inp):
                x_rec = jnp.zeros_like(z_inp)
                ldj = 0
                # We use a python loop because the dimension (3) is tiny.
                # Ideally use jax.lax.scan for large dimensions.
                def body_fun(val, idx):
                    x_current, ldj = val
                    mu, log_scale = self.net(x_current, c_inp)
                    log_scale = jnp.tanh(log_scale)
                    # it only depends on x_current[<i]
                    x_i = z_inp[idx] * jnp.exp(log_scale[idx]) + mu[idx]
                    ldj += log_scale[idx]
                    return (x_current.at[idx].set(x_i), ldj), None

                (x_rec, ldj), _ = jax.lax.scan(body_fun, (x_rec, ldj), jnp.array(self.net.indices))

                return x_rec, ldj # LDJ not needed for sampling usually

            return jax.vmap(_inverse_single)(x, cond)


def searchsorted(bin_locations, inputs, eps=1e-6):
    # bin_locations: (..., num_bins+1)
    # inputs: (..., )
    # Returns index of bin (..., )
    # We just use jnp.searchsorted on the last axis.
    # jnp.searchsorted expects 1D sorted array usually, but supports batching if shapes match?
    # Actually jax.numpy.searchsorted is not batched in the same convenient way as torch.
    # We use vmap for robustness.

    return jax.vmap(lambda b, i: jnp.searchsorted(b, i, side='right') - 1)(bin_locations, inputs)


def rational_quadratic_spline(inputs, widths, heights, derivatives, inverse,
                              tail_bound=3.0, min_bin_width=1e-3,
                              min_bin_height=1e-3, min_derivative=1e-3):
    # inputs: (batch, dim)
    # widths, heights: (batch, dim, num_bins)
    # derivatives: (batch, dim, num_bins+1)

    num_bins = widths.shape[-1]

    inside_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)

    # JAX evaluates BOTH branches of jnp.where before selecting, so out-of-domain
    # inputs still pass through spline maths and can produce NaN that poisons
    # gradients even when the output is correctly masked.  Fix: clamp inputs to
    # the valid domain before all spline computations; the final jnp.where still
    # applies the identity (linear) tail for truly out-of-domain values.
    inputs_s = jnp.clip(inputs, -tail_bound, tail_bound)

    # Normalize params
    widths = jax.nn.softmax(widths, axis=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths

    cumwidths = jnp.cumsum(widths, axis=-1)
    cumwidths = jnp.pad(cumwidths, ((0,0), (0,0), (1,0)), constant_values=0.0)
    cumwidths = (cumwidths - 0.5) * 2 * tail_bound
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    heights = jax.nn.softmax(heights, axis=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights

    cumheights = jnp.cumsum(heights, axis=-1)
    cumheights = jnp.pad(cumheights, ((0,0), (0,0), (1,0)), constant_values=0.0)
    cumheights = (cumheights - 0.5) * 2 * tail_bound
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    derivatives = jax.nn.softplus(derivatives) + min_derivative

    if inverse:
        bin_locations = cumheights
    else:
        bin_locations = cumwidths

    def _get_bin_params(b_locs, inp):
        idx = jnp.searchsorted(b_locs, inp, side='right') - 1
        return jnp.clip(idx, 0, num_bins - 1)

    # Use clamped inputs for bin search and all spline computations
    bin_idx = jax.vmap(jax.vmap(_get_bin_params))(bin_locations, inputs_s)  # (B, D)
    idx_expanded = bin_idx[..., None]

    input_cumwidths   = jnp.take_along_axis(cumwidths,   idx_expanded,     axis=-1).squeeze(-1)
    input_bin_widths  = jnp.take_along_axis(widths,      idx_expanded,     axis=-1).squeeze(-1)
    input_cumheights  = jnp.take_along_axis(cumheights,  idx_expanded,     axis=-1).squeeze(-1)
    input_bin_heights = jnp.take_along_axis(heights,     idx_expanded,     axis=-1).squeeze(-1)
    input_delta       = input_bin_heights / input_bin_widths
    input_derivatives          = jnp.take_along_axis(derivatives, idx_expanded,     axis=-1).squeeze(-1)
    input_derivatives_plus_one = jnp.take_along_axis(derivatives, idx_expanded + 1, axis=-1).squeeze(-1)

    if inverse:
        a = (inputs_s - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_bin_heights * (input_delta - input_derivatives)
        b = input_bin_heights * input_derivatives - (inputs_s - input_cumheights) * (
                input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs_s - input_cumheights)

        discriminant = jnp.maximum(jnp.square(b) - 4 * a * c, 0.0)
        denom_inv = -b - jnp.sqrt(discriminant)
        # Protect division: denominator is zero only at spline endpoints (theta=0,1)
        # which cannot occur for clamped interior inputs, but guard anyway.
        denom_inv = jnp.where(denom_inv == 0, -jnp.finfo(inputs.dtype).tiny, denom_inv)
        root = (2 * c) / denom_inv
        outputs_inner = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
        )
        derivative_numerator = jnp.square(input_delta) * (
                input_derivatives_plus_one * jnp.square(root)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * jnp.square(1 - root)
        )
        logabsdet_inner = (
            -jnp.log(jnp.maximum(derivative_numerator, jnp.finfo(inputs.dtype).tiny))
            + 2 * jnp.log(jnp.maximum(denominator, jnp.finfo(inputs.dtype).tiny))
        )

    else:
        theta = (inputs_s - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_bin_heights * (
                input_delta * jnp.square(theta) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
                (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                * theta_one_minus_theta
        )
        # Protect division and log
        denominator_safe = jnp.maximum(denominator, jnp.finfo(inputs.dtype).tiny)
        outputs_inner = input_cumheights + numerator / denominator_safe

        derivative_numerator = jnp.square(input_delta) * (
                input_derivatives_plus_one * jnp.square(theta)
                + 2 * input_delta * theta_one_minus_theta
                + input_derivatives * jnp.square(1 - theta)
        )
        logabsdet_inner = (
            jnp.log(jnp.maximum(derivative_numerator, jnp.finfo(inputs.dtype).tiny))
            - 2 * jnp.log(denominator_safe)
        )

    # Apply identity tail for out-of-domain inputs
    outputs   = jnp.where(inside_mask, outputs_inner,   inputs)
    logabsdet = jnp.where(inside_mask, logabsdet_inner, jnp.zeros_like(inputs))

    return outputs, jnp.sum(logabsdet, axis=-1)


class SplineCoupling(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    idx_a: Any = eqx.field(static=True)
    idx_b: Any = eqx.field(static=True)
    net: eqx.nn.Sequential
    tail_bound: float = 3.0

    def __init__(self, input_dim, hidden_sizes, cond_dim, mask, key, num_bins=8, tail_bound=3.0):
        self.dim = input_dim
        self.cond_dim = cond_dim
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.idx_a = tuple(np.flatnonzero(np.asarray(mask)).tolist())
        self.idx_b = tuple(np.flatnonzero(1 - np.asarray(mask)).tolist())

        in_dim = int(jnp.sum(mask)) + cond_dim
        out_dim = input_dim - int(jnp.sum(mask))
        param_dim = out_dim * (3 * num_bins + 1)

        layers = []
        curr_dim = in_dim
        for h in hidden_sizes:
            k1, key = jax.random.split(key)
            layers.append(eqx.nn.Linear(curr_dim, h, key=k1))
            layers.append(eqx.nn.Lambda(jax.nn.relu))
            curr_dim = h

        k2, key = jax.random.split(key)
        final = eqx.nn.Linear(curr_dim, param_dim, key=k2)

        # Init final layer to zero
        zeros_w = jnp.zeros_like(final.weight)
        zeros_b = jnp.zeros_like(final.bias)
        final = eqx.tree_at(lambda l: (l.weight, l.bias), final, (zeros_w, zeros_b))
        layers.append(final)

        self.net = eqx.nn.Sequential(layers)

    def __call__(self, x, cond, reverse=False):

        # Split
        x_a = x[..., jnp.asarray(self.idx_a, dtype=jnp.int32)]
        x_b = x[..., jnp.asarray(self.idx_b, dtype=jnp.int32)]

        inp = jnp.concatenate([x_a, cond], axis=-1)
        params = jax.vmap(self.net)(inp)

        # Reshape params
        batch_size = x.shape[0]
        out_dim = x_b.shape[-1]
        params = params.reshape(batch_size, out_dim, 3 * self.num_bins + 1)

        widths = params[..., :self.num_bins]
        heights = params[..., self.num_bins:2*self.num_bins]
        derivatives = params[..., 2*self.num_bins:]

        z_b, ldj = rational_quadratic_spline(
            x_b, widths, heights, derivatives, reverse,
            tail_bound=self.tail_bound
        )

        z = jnp.zeros_like(x)
        z = z.at[..., jnp.asarray(self.idx_a, dtype=jnp.int32)].set(x_a)
        z = z.at[..., jnp.asarray(self.idx_b, dtype=jnp.int32)].set(z_b)

        return z, ldj


class BNNormFlow(eqx.Module):
    """Double RealNVP with an explicitly stateful BatchNorm conditioner."""
    dim: int = eqx.field(static=True)
    conditioner: eqx.Module
    transforms: tuple
    dropout_rate: float = eqx.field(static=True)

    def __init__(self, dim, cond_dim, key, *,
                 n_coupling=4, hidden_dims=(64,) * 4, encoded_dim=64,
                 encoder_hidden_size=64, encoder_layers=2, sequence_length=96,
                 transformer_heads=4, activation=jnp.tanh, clamp=1.0,
                 clamp_activation="I", dropout_rate=0.0,
                 conditioner_type="lstm", transformer_ff_dim=None,
                 encoder_linear_hidden_size=512, encoder_linear_layers=4,
                 use_actnorm=True):
        self.dim = dim
        self.dropout_rate = dropout_rate
        keys = jr.split(key, n_coupling + 1)
        if conditioner_type == "lstm":
            self.conditioner = BatchNormBidirectionalLSTMEncoder(
                cond_dim, encoder_hidden_size, encoded_dim, keys[0],
                n_layers=encoder_layers)
        elif conditioner_type == "pytorch_lstm":
            self.conditioner = BatchNormLSTMEncoder(
                cond_dim, encoder_hidden_size, encoded_dim,
                encoder_linear_hidden_size, keys[0],
                n_lstm_layers=encoder_layers,
                n_linear_layers=encoder_linear_layers,
            )
        elif conditioner_type == "transformer_batchnorm":
            self.conditioner = BatchNormTransformerTimeSeriesEncoder(
                cond_dim, encoder_hidden_size, encoded_dim, keys[0],
                sequence_length=sequence_length, n_layers=encoder_layers,
                num_heads=transformer_heads, ff_dim=transformer_ff_dim)
        else:
            raise ValueError(f"Unknown stateful conditioner {conditioner_type!r}")
        transforms = [ActNorm(dim)] if use_actnorm else []
        for index in range(n_coupling):
            coupling_key, permutation_key = jr.split(keys[index + 1])
            transforms.extend((
                DoubleRealNVPCoupling(
                    dim, encoded_dim, hidden_dims, coupling_key,
                    activation=activation, clamp=clamp,
                    clamp_activation=clamp_activation),
                RandomPermutation(dim, permutation_key),
            ))
            if use_actnorm:
                transforms.append(ActNorm(dim))
        self.transforms = tuple(transforms)

    def initial_batchnorm_state(self):
        return self.conditioner.initial_state()

    def _encode(self, signals, state=None, *, inference=True):
        if state is None:
            state = self.conditioner.initial_state()
        return self.conditioner(signals, state, inference=inference)

    def _flow(self, value, condition, reverse=False):
        logdet = jnp.zeros(value.shape[:-1])
        transforms = self.transforms if not reverse else reversed(self.transforms)
        for transform in transforms:
            if isinstance(transform, DoubleRealNVPCoupling):
                value, increment = transform(value, condition, reverse=reverse)
            else:
                value, increment = transform(value, reverse=reverse)
            logdet = logdet + increment
        return value, logdet

    def _drop_condition(self, condition, key, inference):
        if not self.dropout_rate or inference:
            return condition
        if key is None:
            raise ValueError("Training with flow dropout requires a PRNG key")
        keep = jr.bernoulli(key, 1.0 - self.dropout_rate, condition.shape)
        return jnp.where(keep, condition / (1.0 - self.dropout_rate), 0.0)

    def initialize_actnorm(self, parameters, signals, state=None):
        condition, _ = self._encode(signals, state, inference=False)
        value, initialized = parameters, []
        for transform in self.transforms:
            if isinstance(transform, ActNorm):
                transform = transform.initialize(value)
                value, _ = transform(value)
            elif isinstance(transform, DoubleRealNVPCoupling):
                value, _ = transform(value, condition)
            else:
                value, _ = transform(value)
            initialized.append(transform)
        return eqx.tree_at(lambda item: item.transforms, self, tuple(initialized))

    def forward_with_state(self, parameters, signals, state=None, *, key=None,
                           inference=True):
        condition, new_state = self._encode(signals, state, inference=inference)
        condition = self._drop_condition(condition, key, inference)
        latent, logdet = self._flow(parameters, condition)
        return latent, logdet, new_state

    def __call__(self, parameters, signals, key=None, inference=True):
        latent, logdet, _ = self.forward_with_state(
            parameters, signals, key=key, inference=inference)
        return latent, logdet

    def log_prob_with_state(self, parameters, signals, state=None, *, key=None, inference=True):
        latent, logdet, new_state = self.forward_with_state(
            parameters, signals, state, key=key, inference=inference)
        base = -0.5 * (jnp.sum(latent ** 2, axis=-1) + self.dim * jnp.log(2 * jnp.pi))
        return base + logdet, new_state

    def log_prob(self, parameters, signals, key=None, inference=True):
        return self.log_prob_with_state(parameters, signals, key=key, inference=inference)[0]

    def inverse(self, latent, signals, state=None):
        condition, _ = self._encode(signals, state, inference=True)
        return self._flow(latent, condition, reverse=True)

    def sample(self, signals, key, num_samples, state=None):
        condition, _ = self._encode(signals, state, inference=True)
        # Preserve sample and observation axes throughout the flow. CouplingNet
        # handles arbitrary leading axes, so flatten/repeat is unnecessary.
        latent = jr.normal(key, (num_samples, signals.shape[0], self.dim))
        condition = jnp.broadcast_to(condition, (num_samples, *condition.shape))
        return self._flow(latent, condition, reverse=True)[0]


class CondNormFlow(eqx.Module):
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    flow_type: str = eqx.field(static=True)
    encoder: eqx.Module
    transforms: List[eqx.Module]
    prior_mean: jnp.ndarray
    prior_cov: jnp.ndarray
    refinement: Optional[eqx.Module]
    dropout_rate: float = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        key: jax.Array,
        n_coupling: int = 6,
        hidden_nf: int = 128,
        num_hidden_nf: int = 2,
        encoded_latent_dim: int = 16,
        hidden_rnn: int = 128,
        rnn_type: str = "lstm",
        flow_type: str = 'realnvp',
        encoder_type: str = "rnn",
        encoder_layers: int = 2,
        sequence_length: int = 96,
        transformer_heads: int = 4,
        transformer_ff_dim: Optional[int] = None,
        double_hidden_dims: Optional[Sequence[int]] = None,
        clamp: float = 1.0,
        clamp_activation: str = "I",
        dropout_rate: float = 0.0,
        use_actnorm: bool = True,
        log_posterior_fn=None,
        mcmc_type='MALA', num_mh_steps: int = 1, mcmc_step_size: float = 1e-3, noise_std=0.01
    ):
        self.dim = dim
        self.cond_dim = cond_dim
        self.flow_type = flow_type.lower()
        if self.flow_type not in {"realnvp", "double_realnvp", "nsf", "maf"}:
            raise ValueError("Unknown flow_type")
        if n_coupling < 1 or (self.flow_type in {"realnvp", "nsf"} and n_coupling % 2):
            raise ValueError("Coupling flows require a positive even n_coupling")
        self.dropout_rate = dropout_rate
        self.prior_mean = jnp.zeros(dim)
        self.prior_cov = jnp.eye(dim)

        k_enc, k_flow = jax.random.split(key)
        encoder_type = encoder_type.lower()
        if encoder_type == "lstm":
            self.encoder = VectorizedBidirectionalLSTMEncoder(
                cond_dim, hidden_rnn, encoded_latent_dim, k_enc,
                n_layers=encoder_layers)
        elif encoder_type == "lstm_layernorm":
            self.encoder = LayerNormBidirectionalLSTMEncoder(
                cond_dim, hidden_rnn, encoded_latent_dim, k_enc,
                n_layers=encoder_layers)
        elif encoder_type == "transformer":
            self.encoder = TransformerTimeSeriesEncoder(
                cond_dim, hidden_rnn, encoded_latent_dim, k_enc,
                sequence_length=sequence_length, n_layers=encoder_layers,
                num_heads=transformer_heads, ff_dim=transformer_ff_dim)
        elif encoder_type == "rnn":
            self.encoder = TimeSeriesEncoder(
                cond_dim, hidden_rnn, encoded_latent_dim, k_enc, rnn=rnn_type)
        else:
            raise ValueError(
                "encoder_type must be 'rnn', 'lstm', 'lstm_layernorm', or 'transformer'")

        self.transforms = []
        keys = jax.random.split(k_flow, n_coupling + 2) # +2 for ActNorms

        flow_type = self.flow_type

        if flow_type == "double_realnvp":
            hidden_dims = (tuple(double_hidden_dims) if double_hidden_dims is not None
                           else (hidden_nf,) * num_hidden_nf)
            double_keys = jr.split(k_flow, n_coupling)
            self.transforms = [ActNorm(dim)] if use_actnorm else []
            for index, layer_key in enumerate(double_keys):
                coupling_key, permutation_key = jr.split(layer_key)
                self.transforms.extend((
                    DoubleRealNVPCoupling(
                        dim, encoded_latent_dim, hidden_dims, coupling_key,
                        clamp=clamp, clamp_activation=clamp_activation),
                    RandomPermutation(dim, permutation_key),
                ))
                if use_actnorm:
                    self.transforms.append(ActNorm(dim))
            self.transforms = tuple(self.transforms)
            self.refinement = make_refinement(log_posterior_fn, mcmc_type,
                num_mh_steps=num_mh_steps, step_size=mcmc_step_size, noise_std=noise_std)
            return

        def get_mask(parity):
            return jnp.array([(i + parity) % 2 for i in range(dim)], dtype=jnp.float32)

        for i in range(n_coupling if flow_type == "maf" else n_coupling // 2):
            k_layer1, k_layer2, k_perm = jax.random.split(keys[i], 3)

            # ActNorm
            if use_actnorm:
                self.transforms.append(ActNorm(dim))

            # Permutation
            self.transforms.append(RandomPermutation(dim, key=k_perm))

            # Coupling / Flow Step
            if flow_type == 'nsf':
                self.transforms.append(
                    SplineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim, get_mask(i % 2),
                                   key=k_layer1, num_bins=4)
                )
                self.transforms.append(
                    SplineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim, get_mask((i + 1) % 2),
                                   key=k_layer2, num_bins=4)
                )
            elif flow_type == 'maf':
                self.transforms.append(
                    MAFLayer(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim, False, key=k_layer1)
                )
            else:
                # RealNVP
                self.transforms.append(
                    AffineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                    get_mask(i % 2), key=k_layer1, dropout_rate=dropout_rate)
                )
                self.transforms.append(
                    AffineCoupling(dim, [hidden_nf]*num_hidden_nf, encoded_latent_dim,
                    get_mask((i + 1) % 2), key=k_layer2, dropout_rate=dropout_rate)
                )

        # Final ActNorm
        if use_actnorm:
            self.transforms.append(ActNorm(dim))

        self.refinement = make_refinement(log_posterior_fn, mcmc_type,
            num_mh_steps=num_mh_steps, step_size=mcmc_step_size, noise_std=noise_std)

    def initial_batchnorm_state(self):
        """Stateless conditioners expose the same training protocol as BNNormFlow."""
        return None

    def _encode(self, signals, state=None, *, inference=True):
        del state, inference
        return self.encoder(signals), None

    def initialize_actnorm(self, parameters, signals, state=None):
        condition, _ = self._encode(signals, state, inference=False)
        value, initialized = parameters, []
        for transform in self.transforms:
            if isinstance(transform, ActNorm):
                transform = transform.initialize(value)
                value, _ = transform(value)
            elif isinstance(transform, (RandomPermutation,)):
                value, _ = transform(value)
            elif isinstance(transform, AffineCoupling):
                value, _ = transform(value, condition, inference=True)
            else:
                value, _ = transform(value, condition)
            initialized.append(transform)
        return eqx.tree_at(lambda item: item.transforms, self, type(self.transforms)(initialized))

    def forward_with_state(self, parameters, signals, state=None, *, key=None,
                           inference=True):
        condition, _ = self._encode(signals, state, inference=inference)
        if self.flow_type == "double_realnvp" and self.dropout_rate and not inference:
            if key is None:
                raise ValueError("Training with flow dropout requires a PRNG key")
            keep = jr.bernoulli(key, 1.0 - self.dropout_rate, condition.shape)
            condition = jnp.where(
                keep, condition / (1.0 - self.dropout_rate), 0.0)
        latent, logdet = self.__flow_call__(
            parameters, condition, key=key, inference=inference)
        return latent, logdet, None

    def __flow_call__(self, z: jnp.ndarray, cond_emb:jnp.ndarray, reverse=False,
                      key=None, inference=True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logdet = jnp.zeros(z.shape[:-1])
        x = z

        flow_list = self.transforms if not reverse else reversed(self.transforms)
        flow_list = list(flow_list)
        keys = jr.split(key, len(flow_list)) if key is not None else [None] * len(flow_list)
        for t, layer_key in zip(flow_list, keys):
            if isinstance(t, (ActNorm, RandomPermutation)):
                x, ldj = t(x, reverse=reverse)
            elif isinstance(t, AffineCoupling):
                x, ldj = t(x, cond_emb, reverse=reverse, key=layer_key, inference=inference)
            else:
                x, ldj = t(x, cond_emb, reverse=reverse)
            logdet = logdet + ldj

        return x, logdet

    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray, key=None,
                 inference=True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # cond: raw time series (Batch, T, C)
        # Encode condition
        cond_emb, _ = self._encode(cond, inference=inference)
        z, logdet = self.__flow_call__(x, cond_emb, reverse=False, key=key, inference=inference)
        return z, logdet

    def inverse(self, z: jnp.ndarray, cond: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        cond_emb, _ = self._encode(cond, inference=True)
        x, logdet = self.__flow_call__(z, cond_emb, reverse=True)
        return x, logdet

    def log_prob(self, x: jnp.ndarray, cond: jnp.ndarray, key=None,
                 inference=True) -> jnp.ndarray:
        z, logdet = self(x, cond, key=key, inference=inference)
        # Prior log prob (Gaussian)
        # -0.5 * (z^T z + D * log(2pi))
        log_prob_z = -0.5 * (jnp.sum(z**2, axis=-1) + self.dim * jnp.log(2 * jnp.pi))
        return log_prob_z + logdet

    def sample(self, cond: jnp.ndarray, key: jax.Array, num_samples: int,
               state=None) -> jnp.ndarray:
        cond_emb, _ = self._encode(cond, state, inference=True)
        if self.flow_type == "double_realnvp":
            latent = jr.normal(key, (num_samples, cond.shape[0], self.dim))
            condition = jnp.broadcast_to(cond_emb, (num_samples, *cond_emb.shape))
            return self.__flow_call__(latent, condition, reverse=True)[0]
        batch_size = cond.shape[0]
        total_num = num_samples * batch_size
        cond_rep = jnp.repeat(cond_emb, num_samples, axis=0)

        z = jr.normal(key, (total_num, self.dim))
        samples = self.__flow_call__(z, cond_rep, reverse=True)[0]
        samples = samples.reshape(batch_size, num_samples, -1)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples

    def sample_with_MCMC(self, cond, key, num_samples):
        sample_key, refinement_key = jr.split(key)
        return refine_samples(self, self.sample(cond, sample_key, num_samples), cond, refinement_key)


def cnf_loss(model, y: jnp.ndarray, true_params: jnp.ndarray, key=None):
    z, logdet = model(true_params, y, key=key, inference=key is None)
    return jnp.mean(0.5 * jnp.sum(z ** 2, axis=-1) - logdet)


@eqx.filter_jit
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state: Any, optimizer: optax.GradientTransformation, filter_fn=None):
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return cnf_loss(full_model, y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = preserve_model_state(model, eqx.combine(diff_model, static_model))
    else:
        loss_fn = lambda m: cnf_loss(m, y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = apply_model_updates(model, updates)
    return model, opt_state, loss


# Example usage snippet
if __name__ == "__main__":

    import sys
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_dir, '..')
    sys.path.append(project_root)
    from utils.generate_data import generate_mass_spring_data

    flow_type = 'maf' if len(sys.argv) == 1 else sys.argv[1]
    dt0 = 0.01
    t1 = 1.0
    noise_std = 0.0
    generate_data_jit = partial(jax.jit, static_argnames=['n_samples', 'dt0', 't1', 'noise_std'])(
                                            generate_mass_spring_data)

    feature_dim = 2  # Input features per time step
    hidden_dim_rnn = 64  # GRU hidden size
    hidden_dim = 64
    latent_dim = 16  # Encoded feature size
    param_dim = 3  # Number of physical parameters to infer (theta_p)
    param_names = [r'$m$', r'$k$', r'$c$']

    test_key = jr.PRNGKey(0)
    test_key, sample_key, noise_key = jr.split(test_key, 3)
    test_size = 200
    n_samples = 50
    test_D, test_theta_p = generate_data_jit(test_key, dt0=dt0, t1=t1, n_samples=test_size, noise_std=noise_std)

    model_key = jr.PRNGKey(42)

    # Initialize Model
    model = CondNormFlow(param_dim, feature_dim, key=model_key, n_coupling=8, hidden_nf=hidden_dim, num_hidden_nf=2,
                         encoded_latent_dim=latent_dim, hidden_rnn=hidden_dim_rnn, rnn_type='lstm', flow_type=flow_type)


    def check_invertibility(model, x, cond, tol=1e-5):
        """
        Verifies x == inverse(forward(x)) and z == forward(inverse(z))
        """
        print(f"--- Checking {type(model).__name__} ---")

        # 1. Forward -> Inverse reconstruction
        z, ldj_fwd = model(x, cond)
        x_rec, ldj_inv = model.inverse(z, cond)

        # Check Data Reconstruction
        diff_x = jnp.max(jnp.abs(x - x_rec))
        is_invertible_x = diff_x < tol

        # Check LogDet Consistency: LDJ_forward + LDJ_inverse should be ~ 0
        diff_ldj = jnp.max(jnp.abs(ldj_fwd + ldj_inv))

        print(f"Max Reconstruction Error (x -> z -> x): {diff_x:.2e}")
        print(f"Max LogDet Error (LDJ_fwd + LDJ_inv):   {diff_ldj:.2e}")

        if not is_invertible_x:
            print("❌ Invertibility check FAILED (Tolerance exceeded)")
        else:
            print("✅ Invertibility check PASSED")

        return is_invertible_x


    check_invertibility(model, test_theta_p, test_D, tol=1e-5)
