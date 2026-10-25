import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from typing import List, NamedTuple

from bayesinf.models.networks import BatchNorm1d


class PELSState(NamedTuple):
    encoder_means: tuple
    encoder_variances: tuple
    regressor_means: tuple
    regressor_variances: tuple

# linear block pattern: Linear -> BN -> ReLU -> Dropout
class LinearBlock(eqx.Module):
    linear: eqx.nn.Linear
    bn: BatchNorm1d
    dropout: eqx.nn.Dropout
    use_dropout: bool

    def __init__(self, in_features, out_features, dropout_rate, key):
        self.linear = eqx.nn.Linear(in_features, out_features, key=key)
        self.bn = BatchNorm1d(out_features)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)
        self.use_dropout = dropout_rate > 0

    def __call__(self, x, mean, variance, key=None, inference=False):
        x = jax.vmap(self.linear)(x)
        x, mean, variance = self.bn(
            x, mean, variance, inference=inference)
        x = jax.nn.relu(x)
        if self.use_dropout and key is not None:
            x = self.dropout(x, key=key, inference=inference)
        return x, mean, variance

class Encoder(eqx.Module):
    lstm_cell: eqx.nn.LSTMCell
    linear_layers: List[LinearBlock]
    final_linear: eqx.nn.Linear
    linear_mu: eqx.nn.Linear
    linear_std: eqx.nn.Linear

    latent_dim_z: int
    theta_dim: int
    dropout_rate: float

    def __init__(self, latent_dim_z, n_time_series, lstm_hidden_dim, linear_hidden_dim, n_linear_layers=4, theta_dim=0, dropout=0.0, key=None):
        super().__init__()
        keys = jr.split(key, 5 + n_linear_layers)

        self.lstm_cell = eqx.nn.LSTMCell(n_time_series, lstm_hidden_dim, key=keys[0])

        self.linear_layers = []
        # First Linear Block
        self.linear_layers.append(
            LinearBlock(lstm_hidden_dim + theta_dim, linear_hidden_dim, dropout, keys[1])
        )

        # Subsequent Linear Blocks
        current_dim = linear_hidden_dim
        if n_linear_layers > 1:
            for i in range(1, n_linear_layers - 1):
                next_dim = linear_hidden_dim // (2**i)
                self.linear_layers.append(
                    LinearBlock(current_dim, next_dim, dropout, keys[2+i])
                )
                current_dim = next_dim

        self.final_linear = eqx.nn.Linear(current_dim, latent_dim_z * 2, key=keys[-3])
        self.linear_mu = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-2])
        self.linear_std = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-1])

        self.theta_dim = theta_dim
        self.latent_dim_z = latent_dim_z
        self.dropout_rate = dropout

    def __call__(self, x, means, variances, theta=None, key=None, inference=False):
        # x shape: (batch, seq_len, n_time_series)

        # LSTM Scan
        # Initial hidden state (zeros)
        h0 = jnp.zeros((x.shape[0], self.lstm_cell.hidden_size))
        c0 = jnp.zeros((x.shape[0], self.lstm_cell.hidden_size))

        # Run LSTM
        def batch_scan(carry, input_t):
            h_new, c_new = jax.vmap(self.lstm_cell)(input_t, carry)
            return (h_new, c_new), h_new
        (h_final, _), _ = jax.lax.scan(
            batch_scan, (h0, c0), jnp.swapaxes(x, 0, 1))

        eh1 = h_final # Take last hidden state

        if self.theta_dim > 0 and theta is not None:
            eh1 = jnp.concatenate([eh1, theta], axis=-1)

        # Linear Block Forward
        out = eh1
        keys = jr.split(key, len(self.linear_layers)) if key is not None else [None] * len(self.linear_layers)

        new_means, new_variances = [], []
        for layer, mean, variance, k in zip(
                self.linear_layers, means, variances, keys):
            out, mean, variance = layer(
                out, mean, variance, key=k, inference=inference)
            new_means.append(mean)
            new_variances.append(variance)

        eh2 = jax.vmap(self.final_linear)(out)
        mu = jax.vmap(self.linear_mu)(eh2)
        logvar = jax.vmap(self.linear_std)(eh2)

        return mu, logvar, tuple(new_means), tuple(new_variances)

class Regressor(eqx.Module):
    linear_layers: List[LinearBlock]
    final_linear: eqx.nn.Linear
    linear_mu: eqx.nn.Linear
    linear_std: eqx.nn.Linear

    def __init__(self, latent_dim_z, theta_dim, linear_hidden_dim, n_linear_layers=4, dropout=0.0, key=None):
        super().__init__()
        keys = jr.split(key, 5 + n_linear_layers)

        self.linear_layers = []
        # First Linear Block
        self.linear_layers.append(
            LinearBlock(theta_dim, linear_hidden_dim, dropout, keys[0])
        )

        current_dim = linear_hidden_dim
        if n_linear_layers > 1:
            for i in range(1, n_linear_layers - 1):
                next_dim = linear_hidden_dim // (2**i)
                self.linear_layers.append(
                    LinearBlock(current_dim, next_dim, dropout, keys[1+i])
                )
                current_dim = next_dim

        self.final_linear = eqx.nn.Linear(current_dim, latent_dim_z * 2, key=keys[-3])
        self.linear_mu = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-2])
        self.linear_std = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-1])

    def __call__(self, theta, means, variances, key=None, inference=False):
        out = theta
        keys = jr.split(key, len(self.linear_layers)) if key is not None else [None] * len(self.linear_layers)

        new_means, new_variances = [], []
        for layer, mean, variance, k in zip(
                self.linear_layers, means, variances, keys):
            out, mean, variance = layer(
                out, mean, variance, key=k, inference=inference)
            new_means.append(mean)
            new_variances.append(variance)

        out = jax.vmap(self.final_linear)(out)
        mu = jax.vmap(self.linear_mu)(out)
        logvar = jax.vmap(self.linear_std)(out)
        return mu, logvar, tuple(new_means), tuple(new_variances)

class Decoder(eqx.Module):
    latent_expand: eqx.nn.Linear
    lstm_cell: eqx.nn.LSTMCell
    extra_cells: tuple
    dropout: eqx.nn.Dropout = eqx.field(static=True)
    linear: eqx.nn.Linear
    theta_dim: int
    latent_dim_z: int
    seq_len: int

    def __init__(self, latent_dim_z, seq_len, n_time_series, lstm_hidden_dim, n_lstm_layers, theta_dim=0, dropout=0.0, key=None):
        super().__init__()
        if n_lstm_layers < 1:
            raise ValueError("n_lstm_layers must be positive")
        k1, k2, k3 = jr.split(key, 3)
        self.extra_cells = tuple(eqx.nn.LSTMCell(lstm_hidden_dim, lstm_hidden_dim, key=k)
            for k in jr.split(jr.fold_in(key, 101), n_lstm_layers-1))
        self.dropout = eqx.nn.Dropout(dropout)

        self.latent_expand = eqx.nn.Linear(latent_dim_z + theta_dim, (latent_dim_z + theta_dim) * seq_len, key=k1)
        # Using LSTMCell for manual loop
        self.lstm_cell = eqx.nn.LSTMCell(latent_dim_z + theta_dim, lstm_hidden_dim, key=k2)
        self.linear = eqx.nn.Linear(lstm_hidden_dim, n_time_series, key=k3)

        self.theta_dim = theta_dim
        self.latent_dim_z = latent_dim_z
        self.seq_len = seq_len

    def _one(self, z, theta=None, key=None, inference=True):
        if self.theta_dim > 0 and theta is not None:
            z = jnp.concatenate([z, theta], axis=-1)

        # Expand and reshape to (seq_len, input_dim)
        z_expanded = self.latent_expand(z)
        z_expanded = jax.nn.relu(z_expanded)

        # View/Reshape: (input_dim * seq_len) -> (seq_len, input_dim)
        z_seq = z_expanded.reshape(self.seq_len, self.latent_dim_z + self.theta_dim)

        # LSTM Scan
        def scan_fn(carry, input_t):
            h, c = carry
            h_new, c_new = self.lstm_cell(input_t, (h, c))
            return (h_new, c_new), h_new # Return hidden state at each step

        h0 = jnp.zeros((self.lstm_cell.hidden_size,))
        c0 = jnp.zeros((self.lstm_cell.hidden_size,))

        _, dh1_seq = jax.lax.scan(scan_fn, (h0, c0), z_seq)

        for index, cell in enumerate(self.extra_cells):
            dh1_seq = self.dropout(dh1_seq, key=None if key is None else jr.fold_in(key, index), inference=inference)
            def stacked_step(carry, value):
                carry = cell(value, carry)
                return carry, carry[0]
            _, dh1_seq = jax.lax.scan(stacked_step, (h0, c0), dh1_seq)

        # Apply final linear layer to all time steps.
        # vmap over the time dimension (axis 0)
        xhat = jax.vmap(self.linear)(dh1_seq)

        return xhat

    def __call__(self, z, theta=None, key=None, inference=True):
        if key is None and not inference and self.extra_cells and self.dropout.p > 0:
            raise ValueError("Decoder dropout requires a PRNG key during training")
        keys = jr.split(key if key is not None else jr.PRNGKey(0), z.shape[0])
        if theta is None:
            return jax.vmap(lambda latent,k: self._one(latent,key=k,inference=inference))(z,keys)
        return jax.vmap(lambda latent,parameters,k: self._one(latent,parameters,k,inference))(z,theta,keys)

class PELS_VAE(eqx.Module):
    encoder: Encoder
    regressor: Regressor
    decoder: Decoder

    def __init__(self, theta_dim, latent_dim_z, seq_len, n_time_series, lstm_hidden_dim, linear_hidden_dim,
                 n_lstm_layers=1, n_linear_layers=4, dropout=0.0, key=None):
        super().__init__()
        k_enc, k_reg, k_dec = jr.split(key, 3)

        self.encoder = Encoder(
            latent_dim_z, n_time_series, lstm_hidden_dim, linear_hidden_dim,
            n_linear_layers=n_linear_layers, theta_dim=theta_dim,
            dropout=dropout, key=k_enc,
        )
        self.regressor = Regressor(latent_dim_z, theta_dim, linear_hidden_dim,
                                   n_linear_layers, dropout, key=k_reg)
        self.decoder = Decoder(latent_dim_z, seq_len, n_time_series, lstm_hidden_dim, n_lstm_layers,
                               theta_dim, dropout, key=k_dec)

    def reparameterize(self, mu, logvar, key):
        std = jnp.exp(0.5 * logvar)
        eps = jr.normal(key, mu.shape)
        return mu + eps * std

    def initial_state(self):
        encoder_widths = tuple(layer.linear.out_features for layer in self.encoder.linear_layers)
        regressor_widths = tuple(layer.linear.out_features for layer in self.regressor.linear_layers)
        return PELSState(
            tuple(jnp.zeros(width) for width in encoder_widths),
            tuple(jnp.ones(width) for width in encoder_widths),
            tuple(jnp.zeros(width) for width in regressor_widths),
            tuple(jnp.ones(width) for width in regressor_widths),
        )

    def __call__(self, x, state, theta=None, key=None, inference=False):
        """
        Forward a completed batch so BatchNorm sees the full leading axis.
        """
        k_enc, k_reg, k_sample, k_dec = jr.split(key, 4)

        mu, logvar, encoder_means, encoder_variances = self.encoder(
            x, state.encoder_means, state.encoder_variances,
            theta=theta, key=k_enc, inference=inference)
        muhat, logvarhat, regressor_means, regressor_variances = self.regressor(
            theta, state.regressor_means, state.regressor_variances,
            key=k_reg, inference=inference)
        state = PELSState(encoder_means, encoder_variances,
                          regressor_means, regressor_variances)

        z = self.reparameterize(mu, logvar, k_sample)
        xhat = self.decoder(z, theta=theta, key=k_dec, inference=inference)

        return xhat, mu, logvar, muhat, logvarhat, z, state

    def predict(self, theta, state, key=None, passes=1):
        """
        Predict xhat from theta.
        """
        muhat, logvarhat, _, _ = self.regressor(
            theta, state.regressor_means, state.regressor_variances,
            key=key, inference=True)

        def single_pass(k):
            z = self.reparameterize(muhat, logvarhat, k)
            return self.decoder(z, theta=theta)

        keys = jr.split(key, passes)
        xhats = jax.vmap(single_pass)(keys)
        xhat = jnp.mean(xhats, axis=0)

        return xhat

# Example of how to initialize and use
if __name__ == '__main__':
    # Mock config variables
    theta_dim = 5
    latent_dim_z = 10
    seq_len = 50
    n_time_series = 3
    lstm_hidden = 64
    linear_hidden = 128

    key = jr.PRNGKey(0)
    model = PELS_VAE(theta_dim, latent_dim_z, seq_len, n_time_series, lstm_hidden,
                     linear_hidden, key=key)

    state = model.initial_state()
    x_batch = jnp.zeros((32, seq_len, n_time_series))
    theta_batch = jnp.zeros((32, theta_dim))
    out = model(x_batch, state, theta=theta_batch, key=jr.PRNGKey(1))
    print(f"Output shape: {out[0].shape}")
