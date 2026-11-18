import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import NamedTuple, Union, Sequence


class BatchNormState(NamedTuple):
    """Running statistics for the MEMS LSTM summary network."""
    mean1: jax.Array
    var1: jax.Array
    mean2: jax.Array
    var2: jax.Array


class BatchNormLSTMState(NamedTuple):
    """Running statistics for every projection of ``BatchNormLSTMEncoder``."""
    means: tuple
    variances: tuple


class BatchNormTransformerBlockState(NamedTuple):
    """Running statistics for both normalizations in a Transformer block."""
    mean1: jax.Array
    var1: jax.Array
    mean2: jax.Array
    var2: jax.Array


class BatchNormTransformerState(NamedTuple):
    """Running statistics for every BatchNorm Transformer block."""
    blocks: tuple


class BatchNorm1d(eqx.Module):
    """PyTorch-compatible BatchNorm over the leading batch dimension."""
    weight: jax.Array
    bias: jax.Array
    eps: float = eqx.field(static=True)
    momentum: float = eqx.field(static=True)

    def __init__(self, size, eps=1e-5, momentum=0.1):
        self.weight = jnp.ones(size)
        self.bias = jnp.zeros(size)
        self.eps = eps
        self.momentum = momentum

    def __call__(self, value, running_mean, running_var, *, inference):
        if inference:
            mean, variance = running_mean, running_var
            new_mean, new_var = running_mean, running_var
        else:
            mean = jnp.mean(value, axis=0)
            variance = jnp.var(value, axis=0)
            n = value.shape[0]
            unbiased = jnp.where(n > 1, variance * n / (n - 1), variance)
            new_mean = (1 - self.momentum) * running_mean + self.momentum * mean
            new_var = (1 - self.momentum) * running_var + self.momentum * unbiased
        value = (value - mean) * jax.lax.rsqrt(variance + self.eps)
        return value * self.weight + self.bias, new_mean, new_var


def xavier_linear(in_size, out_size, key, bias=0.001):
    """Equinox linear layer with PyTorch Xavier-uniform initialization."""
    layer = eqx.nn.Linear(in_size, out_size, key=key)
    limit = jnp.sqrt(6.0 / (in_size + out_size))
    weight = jr.uniform(key, layer.weight.shape, minval=-limit, maxval=limit)
    return eqx.tree_at(
        lambda item: (item.weight, item.bias), layer,
        (weight, jnp.full_like(layer.bias, bias)),
    )


class TorchCompatibleLSTMCell(eqx.Module):
    """LSTM cell with ``torch.nn.LSTM`` parameter layout and initialization."""
    weight_ih: jax.Array
    weight_hh: jax.Array
    bias_ih: jax.Array
    bias_hh: jax.Array
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)

    def __init__(self, input_size, hidden_size, key):
        keys = jr.split(key, 4)
        limit = 1.0 / jnp.sqrt(hidden_size)
        self.weight_ih = jr.uniform(
            keys[0], (4 * hidden_size, input_size), minval=-limit, maxval=limit)
        self.weight_hh = jr.uniform(
            keys[1], (4 * hidden_size, hidden_size), minval=-limit, maxval=limit)
        self.bias_ih = jr.uniform(
            keys[2], (4 * hidden_size,), minval=-limit, maxval=limit)
        self.bias_hh = jr.uniform(
            keys[3], (4 * hidden_size,), minval=-limit, maxval=limit)
        self.input_size, self.hidden_size = input_size, hidden_size

    def __call__(self, value, state):
        hidden, cell = state
        gates = (self.weight_ih @ value + self.bias_ih
                 + self.weight_hh @ hidden + self.bias_hh)
        input_gate, forget_gate, candidate, output_gate = jnp.split(gates, 4)
        input_gate, forget_gate = jax.nn.sigmoid(input_gate), jax.nn.sigmoid(forget_gate)
        candidate, output_gate = jnp.tanh(candidate), jax.nn.sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * candidate
        return output_gate * jnp.tanh(cell), cell

class TimeSeriesEncoder(eqx.Module):
    """Encodes a time series into a single feature vector."""
    cell: eqx.Module
    l1: eqx.Module
    l2: eqx.Module

    def __init__(self, in_size: int, hidden_size: int, out_size: int, key: jr.PRNGKey, rnn='gru'):
        key_cell, key_1, key_2 = jr.split(key, 3)
        if rnn == 'gru':
            self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key_cell)
        else:
            self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key_cell)
        self.l1 = eqx.nn.Linear(hidden_size, hidden_size, key=key_1)
        self.l2 = eqx.nn.Linear(hidden_size, out_size, key=key_2)

    def __rnn_call__(self, x: jax.Array):
        # x is (Time, Features)
        # h0 initialization: jnp.zeros is fine for recurrent models
        def f(carry, inp):
            return self.cell(inp, carry), None

        h0 = jnp.zeros(self.cell.hidden_size)
        if isinstance(self.cell, eqx.nn.LSTMCell):
            c0 = jnp.zeros(self.cell.hidden_size)
            final_states, _ = jax.lax.scan(f, (h0, c0), x)
            final_state = final_states[0]
        else:
            final_state, _ = jax.lax.scan(f, h0, x)
        return final_state

    def __call__(self, x: jax.Array):

        h = jax.vmap(self.__rnn_call__)(x)
        h = jax.nn.elu(h)
        h = jax.vmap(self.l1)(h)
        h = jax.nn.elu(h)
        h = jax.vmap(self.l2)(h)
        return h


class StackedBidirectionalLSTMEncoder(eqx.Module):
    """Two-direction stacked LSTM summary encoder."""
    forward_cells: tuple
    backward_cells: tuple
    projection1: eqx.nn.Linear
    projection2: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, out_size, key, n_layers=2):
        keys = jr.split(key, 2 * n_layers + 2)
        forward, backward = [], []
        layer_in = in_size
        for layer in range(n_layers):
            forward.append(eqx.nn.LSTMCell(layer_in, hidden_size, key=keys[2 * layer]))
            backward.append(eqx.nn.LSTMCell(layer_in, hidden_size, key=keys[2 * layer + 1]))
            layer_in = 2 * hidden_size
        self.forward_cells, self.backward_cells = tuple(forward), tuple(backward)
        self.projection1 = eqx.nn.Linear(2 * hidden_size, out_size, key=keys[-2])
        self.projection2 = eqx.nn.Linear(out_size, out_size, key=keys[-1])
        self.hidden_size = hidden_size

    def _direction(self, cell, sequence):
        initial = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        def step(carry, value):
            carry = cell(value, carry)
            return carry, carry[0]
        final, outputs = jax.lax.scan(step, initial, sequence)
        return final[0], outputs

    def _encode_one(self, sequence):
        values = sequence
        final_forward = final_backward = None
        for forward, backward in zip(self.forward_cells, self.backward_cells):
            final_forward, out_forward = self._direction(forward, values)
            final_backward, out_backward_reversed = self._direction(backward, values[::-1])
            values = jnp.concatenate((out_forward, out_backward_reversed[::-1]), axis=-1)
        summary = jnp.concatenate((final_forward, final_backward), axis=-1)
        return self.projection2(jax.nn.elu(self.projection1(summary)))

    def __call__(self, x):
        return jax.vmap(self._encode_one)(x)


class StackedBidirectionalLSTMAutoencoder(eqx.Module):
    """Bidirectional sequence encoder with an explicit recurrent decoder."""
    encoder: StackedBidirectionalLSTMEncoder
    initial: eqx.nn.Linear
    decoder: eqx.nn.LSTMCell
    output: eqx.nn.Linear
    sequence_length: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, latent_size, key, n_layers=2, sequence_length=96):
        if n_layers < 1 or sequence_length < 1:
            raise ValueError("n_layers and sequence_length must be positive")
        keys = jr.split(key, 4)
        self.encoder = StackedBidirectionalLSTMEncoder(in_size, hidden_size, latent_size, keys[0], n_layers)
        self.initial = eqx.nn.Linear(latent_size, 2*hidden_size, key=keys[1])
        self.decoder = eqx.nn.LSTMCell(latent_size, hidden_size, key=keys[2])
        self.output = eqx.nn.Linear(hidden_size, in_size, key=keys[3])
        self.sequence_length = sequence_length

    def __call__(self, signals):
        if signals.shape[1] != self.sequence_length:
            raise ValueError("Input sequence length differs from the configured decoder length")
        def decode(latent):
            hidden, cell = jnp.split(self.initial(latent), 2)
            def step(carry, _):
                carry = self.decoder(latent, carry)
                return carry, self.output(carry[0])
            return jax.lax.scan(step, (hidden, cell), None, length=self.sequence_length)[1]
        return jax.vmap(decode)(self.encoder(signals))


class BatchNormBidirectionalLSTMEncoder(eqx.Module):
    """Notebook-equivalent stacked BiLSTM with whole-batch recurrent scans."""
    forward_cells: tuple
    backward_cells: tuple
    bn1: BatchNorm1d
    projection1: eqx.nn.Linear
    bn2: BatchNorm1d
    projection2: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, out_size, key, n_layers=2):
        keys = jr.split(key, 2 * n_layers + 2)
        forward, backward = [], []
        layer_in = in_size
        for layer in range(n_layers):
            forward.append(TorchCompatibleLSTMCell(layer_in, hidden_size, keys[2 * layer]))
            backward.append(TorchCompatibleLSTMCell(layer_in, hidden_size, keys[2 * layer + 1]))
            layer_in = 2 * hidden_size
        self.forward_cells, self.backward_cells = tuple(forward), tuple(backward)
        self.bn1 = BatchNorm1d(hidden_size)
        self.projection1 = xavier_linear(hidden_size, out_size, keys[-2])
        self.bn2 = BatchNorm1d(out_size)
        self.projection2 = xavier_linear(out_size, out_size, keys[-1])
        self.hidden_size, self.out_size = hidden_size, out_size

    def initial_state(self):
        return BatchNormState(
            jnp.zeros(self.hidden_size), jnp.ones(self.hidden_size),
            jnp.zeros(self.out_size), jnp.ones(self.out_size),
        )

    def _direction(self, cell, sequence):
        batch = sequence.shape[0]
        initial = (jnp.zeros((batch, self.hidden_size)),
                   jnp.zeros((batch, self.hidden_size)))
        def step(carry, value):
            carry = jax.vmap(cell)(value, carry)
            return carry, carry[0]
        final, outputs = jax.lax.scan(step, initial, jnp.swapaxes(sequence, 0, 1))
        return final[0], jnp.swapaxes(outputs, 0, 1)

    def __call__(self, sequences, state, *, inference):
        values = sequences
        final_backward = None
        # Preserve the notebook's backward-summary convention and saved weights.
        # The last forward direction has no consumer, so do not execute it.
        for index, (forward, backward) in enumerate(zip(self.forward_cells, self.backward_cells)):
            final_backward, out_backward = self._direction(backward, values[:, ::-1])
            if index + 1 < len(self.backward_cells):
                _, out_forward = self._direction(forward, values)
                values = jnp.concatenate((out_forward, out_backward[:, ::-1]), axis=-1)
        summary, mean1, var1 = self.bn1(
            final_backward, state.mean1, state.var1, inference=inference)
        hidden = jax.vmap(self.projection1)(summary)
        hidden, mean2, var2 = self.bn2(
            hidden, state.mean2, state.var2, inference=inference)
        encoded = jax.vmap(self.projection2)(jax.nn.elu(hidden))
        return encoded, BatchNormState(mean1, var1, mean2, var2)


class BatchNormLSTMEncoder(eqx.Module):
    """One-directional LSTM followed by Linear-BatchNorm-ELU projections."""
    cells: tuple
    projections: tuple
    batch_norms: tuple
    output: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, out_size, linear_hidden_size, key,
                 n_lstm_layers=1, n_linear_layers=4):
        if n_lstm_layers < 1 or n_linear_layers < 1:
            raise ValueError("n_lstm_layers and n_linear_layers must be positive")
        hidden_widths = tuple(linear_hidden_size // (2 ** index)
                              for index in range(max(1, n_linear_layers - 1)))
        keys = jr.split(key, n_lstm_layers + len(hidden_widths) + 1)
        cells, current = [], in_size
        for index in range(n_lstm_layers):
            cells.append(TorchCompatibleLSTMCell(current, hidden_size, keys[index]))
            current = hidden_size
        self.cells = tuple(cells)
        projections, batch_norms = [], []
        for index, width in enumerate(hidden_widths):
            projections.append(xavier_linear(current, width, keys[n_lstm_layers + index]))
            batch_norms.append(BatchNorm1d(width))
            current = width
        self.projections = tuple(projections)
        self.batch_norms = tuple(batch_norms)
        self.output = xavier_linear(current, out_size, keys[-1])
        self.hidden_size = hidden_size

    def initial_state(self):
        widths = tuple(layer.weight.shape[0] for layer in self.projections)
        return BatchNormLSTMState(
            tuple(jnp.zeros(width) for width in widths),
            tuple(jnp.ones(width) for width in widths),
        )

    def _layer(self, cell, sequences):
        batch = sequences.shape[0]
        initial = (jnp.zeros((batch, self.hidden_size)),
                   jnp.zeros((batch, self.hidden_size)))
        def step(carry, values):
            carry = jax.vmap(cell)(values, carry)
            return carry, carry[0]
        final, outputs = jax.lax.scan(step, initial, jnp.swapaxes(sequences, 0, 1))
        return final[0], jnp.swapaxes(outputs, 0, 1)

    def __call__(self, sequences, state, *, inference):
        values, final_hidden = sequences, None
        for cell in self.cells:
            final_hidden, values = self._layer(cell, values)
        means, variances = [], []
        values = final_hidden
        for projection, batch_norm, mean, variance in zip(
                self.projections, self.batch_norms, state.means, state.variances):
            values = jax.vmap(projection)(values)
            values, mean, variance = batch_norm(
                values, mean, variance, inference=inference)
            values = jax.nn.elu(values)
            means.append(mean)
            variances.append(variance)
        return jax.vmap(self.output)(values), BatchNormLSTMState(tuple(means), tuple(variances))


class VectorizedBidirectionalLSTMEncoder(eqx.Module):
    """Stacked BiLSTM without BatchNorm and without per-observation scans."""
    forward_cells: tuple
    backward_cells: tuple
    projection1: eqx.nn.Linear
    projection2: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, out_size, key, n_layers=2):
        keys = jr.split(key, 2 * n_layers + 2)
        forward, backward = [], []
        layer_in = in_size
        for layer in range(n_layers):
            forward.append(TorchCompatibleLSTMCell(layer_in, hidden_size, keys[2 * layer]))
            backward.append(TorchCompatibleLSTMCell(layer_in, hidden_size, keys[2 * layer + 1]))
            layer_in = 2 * hidden_size
        self.forward_cells, self.backward_cells = tuple(forward), tuple(backward)
        self.projection1 = xavier_linear(hidden_size, out_size, keys[-2])
        self.projection2 = xavier_linear(out_size, out_size, keys[-1])
        self.hidden_size = hidden_size

    def _direction(self, cell, sequence):
        batch = sequence.shape[0]
        initial = (jnp.zeros((batch, self.hidden_size)),
                   jnp.zeros((batch, self.hidden_size)))
        def step(carry, value):
            carry = jax.vmap(cell)(value, carry)
            return carry, carry[0]
        final, outputs = jax.lax.scan(step, initial, jnp.swapaxes(sequence, 0, 1))
        return final[0], jnp.swapaxes(outputs, 0, 1)

    def __call__(self, sequences):
        values = sequences
        final_backward = None
        # Preserve the notebook's backward-summary convention and saved weights.
        # The last forward direction has no consumer, so do not execute it.
        for index, (forward, backward) in enumerate(zip(self.forward_cells, self.backward_cells)):
            final_backward, out_backward = self._direction(backward, values[:, ::-1])
            if index + 1 < len(self.backward_cells):
                _, out_forward = self._direction(forward, values)
                values = jnp.concatenate((out_forward, out_backward[:, ::-1]), axis=-1)
        hidden = jnp.tanh(jax.vmap(self.projection1)(final_backward))
        return jax.vmap(self.projection2)(hidden)


class LayerNormBidirectionalLSTMEncoder(eqx.Module):
    """BiLSTM summary network with LayerNorm replacing BatchNorm."""
    forward_cells: tuple
    backward_cells: tuple
    norm1: eqx.nn.LayerNorm
    projection1: eqx.nn.Linear
    norm2: eqx.nn.LayerNorm
    projection2: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, out_size, key, n_layers=2):
        keys = jr.split(key, 2 * n_layers + 2)
        forward, backward = [], []
        layer_in = in_size
        for layer in range(n_layers):
            forward.append(TorchCompatibleLSTMCell(layer_in, hidden_size, keys[2 * layer]))
            backward.append(TorchCompatibleLSTMCell(layer_in, hidden_size, keys[2 * layer + 1]))
            layer_in = 2 * hidden_size
        self.forward_cells, self.backward_cells = tuple(forward), tuple(backward)
        self.norm1 = eqx.nn.LayerNorm(hidden_size)
        self.projection1 = xavier_linear(hidden_size, out_size, keys[-2])
        self.norm2 = eqx.nn.LayerNorm(out_size)
        self.projection2 = xavier_linear(out_size, out_size, keys[-1])
        self.hidden_size = hidden_size

    def _direction(self, cell, sequence):
        batch = sequence.shape[0]
        initial = (jnp.zeros((batch, self.hidden_size)),
                   jnp.zeros((batch, self.hidden_size)))
        def step(carry, value):
            carry = jax.vmap(cell)(value, carry)
            return carry, carry[0]
        final, outputs = jax.lax.scan(step, initial, jnp.swapaxes(sequence, 0, 1))
        return final[0], jnp.swapaxes(outputs, 0, 1)

    def __call__(self, sequences):
        values = sequences
        final_backward = None
        # Preserve the notebook's backward-summary convention and saved weights.
        # The last forward direction has no consumer, so do not execute it.
        for index, (forward, backward) in enumerate(zip(self.forward_cells, self.backward_cells)):
            final_backward, out_backward = self._direction(backward, values[:, ::-1])
            if index + 1 < len(self.backward_cells):
                _, out_forward = self._direction(forward, values)
                values = jnp.concatenate((out_forward, out_backward[:, ::-1]), axis=-1)
        summary = jax.vmap(self.norm1)(final_backward)
        hidden = jax.vmap(self.projection1)(summary)
        hidden = jax.vmap(self.norm2)(hidden)
        return jax.vmap(self.projection2)(jax.nn.elu(hidden))


class TransformerEncoderBlock(eqx.Module):
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    model_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(self, model_dim, num_heads, ff_dim, key):
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        keys = jr.split(key, 6)
        self.query = xavier_linear(model_dim, model_dim, keys[0], bias=0.0)
        self.key = xavier_linear(model_dim, model_dim, keys[1], bias=0.0)
        self.value = xavier_linear(model_dim, model_dim, keys[2], bias=0.0)
        self.output = xavier_linear(model_dim, model_dim, keys[3], bias=0.0)
        self.ff1 = xavier_linear(model_dim, ff_dim, keys[4])
        self.ff2 = xavier_linear(ff_dim, model_dim, keys[5])
        self.norm1, self.norm2 = eqx.nn.LayerNorm(model_dim), eqx.nn.LayerNorm(model_dim)
        self.model_dim, self.num_heads = model_dim, num_heads

    def __call__(self, values):
        project = lambda layer: jax.vmap(jax.vmap(layer))(values)
        q, k, v = project(self.query), project(self.key), project(self.value)
        batch, time, _ = q.shape
        head_dim = self.model_dim // self.num_heads
        split = lambda x: x.reshape(batch, time, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        weights = jax.nn.softmax(jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(head_dim), axis=-1)
        attended = jnp.einsum("bhts,bhsd->bhtd", weights, v).transpose(0, 2, 1, 3)
        attended = attended.reshape(batch, time, self.model_dim)
        attended = jax.vmap(jax.vmap(self.output))(attended)
        values = jax.vmap(jax.vmap(self.norm1))(values + attended)
        hidden = jax.nn.gelu(jax.vmap(jax.vmap(self.ff1))(values))
        hidden = jax.vmap(jax.vmap(self.ff2))(hidden)
        return jax.vmap(jax.vmap(self.norm2))(values + hidden)


class TransformerTimeSeriesEncoder(eqx.Module):
    """Batch-first Transformer summary encoder without BatchNorm."""
    input_projection: eqx.nn.Linear
    position: jax.Array
    blocks: tuple
    output_projection: eqx.nn.Linear
    sequence_length: int = eqx.field(static=True)

    def __init__(self, in_size, model_dim, out_size, key, *, sequence_length=96,
                 n_layers=2, num_heads=4, ff_dim=None):
        keys = jr.split(key, n_layers + 3)
        ff_dim = ff_dim or 4 * model_dim
        self.input_projection = xavier_linear(in_size, model_dim, keys[0])
        self.position = 0.02 * jr.normal(keys[1], (sequence_length, model_dim))
        self.blocks = tuple(TransformerEncoderBlock(model_dim, num_heads, ff_dim, keys[i + 2])
                            for i in range(n_layers))
        self.output_projection = xavier_linear(model_dim, out_size, keys[-1])
        self.sequence_length = sequence_length

    def __call__(self, sequences):
        if sequences.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {sequences.shape[1]}")
        values = jax.vmap(jax.vmap(self.input_projection))(sequences) + self.position
        for block in self.blocks:
            values = block(values)
        return jax.vmap(self.output_projection)(jnp.mean(values, axis=1))


class BatchNormTransformerEncoderBlock(eqx.Module):
    """Transformer block with BatchNorm replacing both LayerNorm operations."""
    query: eqx.nn.Linear
    key: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear
    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear
    norm1: BatchNorm1d
    norm2: BatchNorm1d
    model_dim: int = eqx.field(static=True)
    num_heads: int = eqx.field(static=True)

    def __init__(self, model_dim, num_heads, ff_dim, key):
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        keys = jr.split(key, 6)
        self.query = xavier_linear(model_dim, model_dim, keys[0], bias=0.0)
        self.key = xavier_linear(model_dim, model_dim, keys[1], bias=0.0)
        self.value = xavier_linear(model_dim, model_dim, keys[2], bias=0.0)
        self.output = xavier_linear(model_dim, model_dim, keys[3], bias=0.0)
        self.ff1 = xavier_linear(model_dim, ff_dim, keys[4])
        self.ff2 = xavier_linear(ff_dim, model_dim, keys[5])
        self.norm1, self.norm2 = BatchNorm1d(model_dim), BatchNorm1d(model_dim)
        self.model_dim, self.num_heads = model_dim, num_heads

    def initial_state(self):
        zeros, ones = jnp.zeros(self.model_dim), jnp.ones(self.model_dim)
        return BatchNormTransformerBlockState(zeros, ones, zeros, ones)

    @staticmethod
    def _normalize(norm, values, mean, variance, inference):
        shape = values.shape
        flat, mean, variance = norm(
            values.reshape(-1, shape[-1]), mean, variance, inference=inference)
        return flat.reshape(shape), mean, variance

    def __call__(self, values, state, *, inference):
        project = lambda layer: jax.vmap(jax.vmap(layer))(values)
        q, k, v = project(self.query), project(self.key), project(self.value)
        batch, time, _ = q.shape
        head_dim = self.model_dim // self.num_heads
        split = lambda x: x.reshape(batch, time, self.num_heads, head_dim).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        weights = jax.nn.softmax(
            jnp.einsum("bhtd,bhsd->bhts", q, k) / jnp.sqrt(head_dim), axis=-1)
        attended = jnp.einsum("bhts,bhsd->bhtd", weights, v).transpose(0, 2, 1, 3)
        attended = attended.reshape(batch, time, self.model_dim)
        attended = jax.vmap(jax.vmap(self.output))(attended)
        values, mean1, var1 = self._normalize(
            self.norm1, values + attended, state.mean1, state.var1, inference)
        hidden = jax.nn.gelu(jax.vmap(jax.vmap(self.ff1))(values))
        hidden = jax.vmap(jax.vmap(self.ff2))(hidden)
        values, mean2, var2 = self._normalize(
            self.norm2, values + hidden, state.mean2, state.var2, inference)
        return values, BatchNormTransformerBlockState(mean1, var1, mean2, var2)


class BatchNormTransformerTimeSeriesEncoder(eqx.Module):
    """Batch-first Transformer whose blocks use BatchNorm instead of LayerNorm."""
    input_projection: eqx.nn.Linear
    position: jax.Array
    blocks: tuple
    output_projection: eqx.nn.Linear
    sequence_length: int = eqx.field(static=True)

    def __init__(self, in_size, model_dim, out_size, key, *, sequence_length=96,
                 n_layers=2, num_heads=4, ff_dim=None):
        keys = jr.split(key, n_layers + 3)
        ff_dim = ff_dim or 4 * model_dim
        self.input_projection = xavier_linear(in_size, model_dim, keys[0])
        self.position = 0.02 * jr.normal(keys[1], (sequence_length, model_dim))
        self.blocks = tuple(
            BatchNormTransformerEncoderBlock(model_dim, num_heads, ff_dim, keys[i + 2])
            for i in range(n_layers))
        self.output_projection = xavier_linear(model_dim, out_size, keys[-1])
        self.sequence_length = sequence_length

    def initial_state(self):
        return BatchNormTransformerState(tuple(block.initial_state() for block in self.blocks))

    def __call__(self, sequences, state, *, inference):
        if sequences.shape[1] != self.sequence_length:
            raise ValueError(f"Expected sequence length {self.sequence_length}, got {sequences.shape[1]}")
        values = jax.vmap(jax.vmap(self.input_projection))(sequences) + self.position
        new_states = []
        for block, block_state in zip(self.blocks, state.blocks):
            values, block_state = block(values, block_state, inference=inference)
            new_states.append(block_state)
        encoded = jax.vmap(self.output_projection)(jnp.mean(values, axis=1))
        return encoded, BatchNormTransformerState(tuple(new_states))


class FrozenBatchNormEncoder(eqx.Module):
    """Use a pretrained BatchNorm encoder with its frozen inference state."""

    encoder: eqx.Module
    state: eqx.nn.State

    def __init__(self, encoder, state):
        self.encoder = encoder
        self.state = state

    def __call__(self, sequences):
        encoded, _ = self.encoder(sequences, self.state, inference=True)
        return jax.lax.stop_gradient(encoded)


class TrainableBatchNormEncoder(eqx.Module):
    """Train a stateful encoder while carrying its running statistics in-tree."""

    encoder: eqx.Module
    state: object
    inference: bool = eqx.field(static=True)

    def __init__(self, encoder, state, *, inference=False):
        self.encoder = encoder
        self.state = state
        self.inference = inference

    def __call__(self, sequences):
        encoded, _ = self.encoder(sequences, self.state, inference=self.inference)
        return encoded
