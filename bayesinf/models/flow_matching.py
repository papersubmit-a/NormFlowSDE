import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from bayesinf.models.optimization import apply_model_updates, preserve_model_state
from bayesinf.models.networks import (
    TimeSeriesEncoder,
    BatchNormBidirectionalLSTMEncoder,
    BatchNormLSTMEncoder,
    BatchNormTransformerTimeSeriesEncoder,
)
from typing import Optional
from bayesinf.models.MCMC_Flow import make_refinement, refine_samples


class VectorField(eqx.Module):
    """
    Approximates the time-dependent vector field v_t(z|cond).
    Input: t (scalar), z (param_dim), cond (latent_dim)
    Output: dz/dt (param_dim)
    """
    net: eqx.nn.MLP

    def __init__(self, param_dim, cond_dim, hidden_dim, depth, key):
        # Input size: time(1) + z(param_dim) + cond(cond_dim)
        input_dim = 1 + param_dim + cond_dim
        self.net = eqx.nn.MLP(
            in_size=input_dim,
            out_size=param_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.gelu,  # GELU usually works better for Flows
            key=key
        )

    def __call__(self, t, z, cond):
        t_arr = jnp.array([t])
        x = jnp.concatenate([t_arr, z, cond])
        return self.net(x)


class ConditionalFlowMatching(eqx.Module):
    encoder: TimeSeriesEncoder
    vector_field: VectorField
    param_dim: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    refinement: Optional[eqx.Module]

    def __init__(self, feature_dim, param_dim, latent_dim, hidden_sizes, hidden_size_rnn, key, log_posterior_fn=None,
                 mcmc_type='MALA', num_mh_steps: int = 1, mcmc_step_size: float = 1e-3, noise_std=0.01):
        key_enc, key_vf = jr.split(key)
        self.param_dim = param_dim
        self.latent_dim = latent_dim

        self.encoder = TimeSeriesEncoder(feature_dim, hidden_size_rnn, latent_dim, key_enc)
        self.vector_field = VectorField(param_dim, latent_dim, hidden_sizes[0], depth=len(hidden_sizes), key=key_vf)
        self.refinement = make_refinement(log_posterior_fn, mcmc_type,
            num_mh_steps=num_mh_steps, step_size=mcmc_step_size, noise_std=noise_std)


    def __call__(self, t, z, x_ts):
        cond = self.encoder(x_ts)
        return jax.vmap(self.vector_field)(t, z, cond)

    def sample(self, x_ts, key, num_samples=1, steps=100,
               return_trajectory=False):

        cond = self.encoder(x_ts)
        batch_size = x_ts.shape[0]
        total_num = batch_size * num_samples
        cond_rep = jnp.repeat(cond, num_samples, axis=0)

        z0 = jr.normal(key, (total_num, self.param_dim))

        # Simple Euler Integration (or RK4)
        # Conditional interpolation paths are straight; marginal trajectories generally are not.
        # Euler error should be checked by increasing the number of integration steps.
        dt = 1.0 / steps

        def body_fn(i, z_curr):
            t = i * dt
            v = jax.vmap(self.vector_field, in_axes=(None, 0, 0))(t, z_curr, cond_rep)
            return z_curr + v * dt

        if return_trajectory:
            def scan_fn(z_curr, i):
                z_next = body_fn(i, z_curr)
                return z_next, z_next

            z1, path = jax.lax.scan(scan_fn, z0, jnp.arange(steps))
            path = jnp.concatenate((z0[None], path), axis=0)
            path = path.reshape(
                steps + 1, batch_size, num_samples, self.param_dim)
            return jnp.swapaxes(path, 1, 2)
        z1 = jax.lax.fori_loop(0, steps, body_fn, z0)

        samples = z1.reshape(batch_size, num_samples, self.param_dim)
        samples = jnp.swapaxes(samples, 0, 1)
        return samples

    def sample_with_MCMC(self, cond, key, num_samples, steps=100):
        sample_key, refinement_key = jr.split(key)
        return refine_samples(self, self.sample(cond, sample_key, num_samples, steps=steps), cond, refinement_key)


def cfm_loss(model, x_ts_batch, theta_batch, key):
    B = theta_batch.shape[0]
    key_t, key_z0 = jr.split(key)
    t = jr.uniform(key_t, (B,))
    z0 = jr.normal(key_z0, theta_batch.shape)
    z1 = theta_batch
    t_b = t[:, None]
    z_t = (1 - t_b) * z0 + t_b * z1
    v_target = z1 - z0
    v_pred = model(t, z_t, x_ts_batch)
    return jnp.mean((v_pred - v_target) ** 2)


class StatefulConditionalFlowMatching(eqx.Module):
    """Conditional OT flow matching with the calibration BN conditioner.

    This deliberately mirrors the small state/sample interface used by
    ``BNNormFlow`` so calibration can exchange only its posterior head.
    """
    dim: int = eqx.field(static=True)
    conditioner: eqx.Module
    vector_field: VectorField
    integration_steps: int = eqx.field(static=True)

    def __init__(self, dim, cond_dim, key, *, hidden_dim=64, depth=4,
                 encoded_dim=64, encoder_hidden_size=64, encoder_layers=2,
                 encoder_linear_hidden_size=512, encoder_linear_layers=4,
                 conditioner_type="lstm", sequence_length=96,
                 transformer_heads=4, transformer_ff_dim=None,
                 integration_steps=100):
        self.dim = dim
        self.integration_steps = integration_steps
        encoder_key, field_key = jr.split(key)
        if conditioner_type == "lstm":
            self.conditioner = BatchNormBidirectionalLSTMEncoder(
                cond_dim, encoder_hidden_size, encoded_dim, encoder_key,
                n_layers=encoder_layers)
        elif conditioner_type == "pytorch_lstm":
            self.conditioner = BatchNormLSTMEncoder(
                cond_dim, encoder_hidden_size, encoded_dim,
                encoder_linear_hidden_size, encoder_key,
                n_lstm_layers=encoder_layers,
                n_linear_layers=encoder_linear_layers)
        elif conditioner_type == "transformer_batchnorm":
            self.conditioner = BatchNormTransformerTimeSeriesEncoder(
                cond_dim, encoder_hidden_size, encoded_dim, encoder_key,
                sequence_length=sequence_length, n_layers=encoder_layers,
                num_heads=transformer_heads, ff_dim=transformer_ff_dim)
        else:
            raise ValueError(f"Unknown stateful conditioner {conditioner_type!r}")
        self.vector_field = VectorField(
            dim, encoded_dim, hidden_dim, depth=depth, key=field_key)

    def initial_batchnorm_state(self):
        return self.conditioner.initial_state()

    def _encode(self, signals, state=None, *, inference=True):
        if state is None:
            state = self.conditioner.initial_state()
        return self.conditioner(signals, state, inference=inference)

    def initialize_actnorm(self, parameters, signals, state=None):
        del parameters, signals, state
        return self

    def loss_with_state(self, parameters, signals, key, state=None, *, inference=True,
                        sample_weights=None):
        condition, new_state = self._encode(signals, state, inference=inference)
        time_key, noise_key = jr.split(key)
        time = jr.uniform(time_key, (parameters.shape[0],))
        noise = jr.normal(noise_key, parameters.shape)
        time_column = time[:, None]
        path = (1.0 - time_column) * noise + time_column * parameters
        target_velocity = parameters - noise
        velocity = jax.vmap(self.vector_field)(time, path, condition)
        per_example = jnp.mean((velocity - target_velocity) ** 2, axis=-1)
        if sample_weights is None:
            loss = jnp.mean(per_example)
        else:
            sample_weights = jnp.asarray(sample_weights)
            if sample_weights.shape != per_example.shape:
                raise ValueError("sample_weights must have one weight per example")
            sample_weights = eqx.error_if(sample_weights,
                jnp.any(~jnp.isfinite(sample_weights)) | jnp.any(sample_weights < 0) | (jnp.sum(sample_weights) <= 0),
                "sample_weights must be finite, nonnegative and have positive mass")
            loss = jnp.sum(sample_weights * per_example) / jnp.sum(sample_weights)
        return loss, new_state

    def sample(self, signals, key, num_samples, state=None):
        condition, _ = self._encode(signals, state, inference=True)
        value = jr.normal(key, (num_samples, signals.shape[0], self.dim))
        condition = jnp.broadcast_to(condition, (num_samples, *condition.shape))
        dt = 1.0 / self.integration_steps

        def step(index, current):
            time = index * dt
            velocity = jax.vmap(
                jax.vmap(self.vector_field, in_axes=(None, 0, 0)),
                in_axes=(None, 0, 0),
            )(time, current, condition)
            return current + dt * velocity

        return jax.lax.fori_loop(0, self.integration_steps, step, value)


@eqx.filter_jit
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state, optimizer, filter_fn=None):
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return cfm_loss(full_model, y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = preserve_model_state(model, eqx.combine(diff_model, static_model))
    else:
        loss_fn = lambda m: cfm_loss(m, y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = apply_model_updates(model, updates)
    return model, opt_state, loss
