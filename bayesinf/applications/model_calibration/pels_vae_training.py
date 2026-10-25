"""Training utilities for the thermal-model PELS-VAE surrogate."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from .data import CalibrationNormalization


@dataclass(frozen=True)
class PELSTrainingConfig:
    epochs: int = 10_000
    batch_size: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 1e6
    patience: int = 1_000
    min_delta: float = 1e-4
    beta_max: float = 0.5
    beta_start: float = 0.1
    beta_step_epochs: int = 50
    beta_increment: float = 0.1
    lr_step_epochs: int = 2_000
    lr_decay: float = 0.5
    seed: int = 0


def beta_at_epoch(epoch: int, config: PELSTrainingConfig) -> float:
    return min(config.beta_start + config.beta_increment * (epoch // config.beta_step_epochs),
               config.beta_max)


def pels_loss(outputs, target, beta):
    reconstruction, mu, logvar, predicted_mu, predicted_logvar = outputs[:5]
    mse = jnp.mean((reconstruction - target) ** 2)
    mse_mu = jnp.mean((predicted_mu - mu) ** 2)
    mse_logvar = jnp.mean((predicted_logvar - logvar) ** 2)
    kl = jnp.mean(-0.5 * jnp.sum(1.0 + logvar - mu ** 2 - jnp.exp(logvar), axis=-1))
    return mse + mse_mu + mse_logvar + beta * kl, (mse, mse_mu, mse_logvar, kl)


def _batch_forward(model, state, signals, parameters, key, inference):
    return model(signals, state, parameters, key, inference)


@eqx.filter_jit
def _train_step(model, state, signals, parameters, key, beta, optimizer_state, optimizer):
    def objective(candidate, current_state):
        outputs = _batch_forward(candidate, current_state, signals, parameters, key, False)
        new_state = outputs[-1]
        loss, components = pels_loss(outputs, signals, beta)
        return loss, (components, new_state)

    (loss, (components, state)), gradients = eqx.filter_value_and_grad(
        objective, has_aux=True
    )(model, state)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    return eqx.apply_updates(model, updates), state, optimizer_state, loss, components


@eqx.filter_jit
def _validation_step(model, state, signals, parameters, key, beta):
    outputs = _batch_forward(model, state, signals, parameters, key, True)
    return pels_loss(outputs, signals, beta)


def train_pels_vae(model, state, data, output_dir: Path | str, config: PELSTrainingConfig):
    """Train PELS-VAE and save the best weights and BatchNorm state separately."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    normalization = CalibrationNormalization.fit(data.train)
    normalization.save(output / "normalization.npz")
    steps_per_epoch = max(1, (len(data.train.signals) + config.batch_size - 1) // config.batch_size)
    boundaries = [config.lr_step_epochs * steps_per_epoch * i for i in range(1, 5)]
    schedule = optax.piecewise_constant_schedule(
        config.learning_rate, {boundary: config.lr_decay for boundary in boundaries}
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.gradient_clip),
        optax.adamw(schedule, weight_decay=config.weight_decay),
    )
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_x = jnp.asarray(normalization.normalize_signals(data.train.signals))
    train_theta = jnp.asarray(normalization.normalize_parameters(data.train.parameters))
    validation_x = jnp.asarray(normalization.normalize_signals(data.validation.signals))
    validation_theta = jnp.asarray(normalization.normalize_parameters(data.validation.parameters))
    key = jr.PRNGKey(config.seed)
    best_loss, stale, history = np.inf, 0, []
    validation_key = jr.fold_in(key, 9173)

    for epoch in range(config.epochs):
        key, permutation_key, epoch_key = jr.split(key, 3)
        permutation = np.asarray(jr.permutation(permutation_key, len(train_x)))
        losses, component_rows = [], []
        beta = beta_at_epoch(epoch + 1, config)
        for begin in range(0, len(permutation), config.batch_size):
            ids = permutation[begin:begin + config.batch_size]
            epoch_key, batch_key = jr.split(epoch_key)
            model, state, optimizer_state, loss, components = _train_step(
                model, state, train_x[ids], train_theta[ids], batch_key,
                beta, optimizer_state, optimizer,
            )
            losses.append(float(loss))
            component_rows.append(np.asarray(components, dtype=np.float64))
        validation_loss, validation_components = _validation_step(
            model, state, validation_x, validation_theta, validation_key, config.beta_max
        )
        validation_loss = float(validation_loss)
        components = np.mean(component_rows, axis=0)
        history.append((epoch + 1, float(np.mean(losses)), validation_loss, beta, *components,
                        *np.asarray(validation_components, dtype=np.float64), config.beta_max))
        if np.isfinite(validation_loss) and validation_loss < best_loss - config.min_delta:
            best_loss, stale = validation_loss, 0
            eqx.tree_serialise_leaves(output / "pels_vae.eqx", model)
            eqx.tree_serialise_leaves(output / "pels_vae_state.eqx", state)
        else:
            stale += 1
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"PELS-VAE epoch {epoch + 1:5d} train={history[-1][1]:.6f} "
                  f"validation={validation_loss:.6f} beta={beta:.2f}", flush=True)
        if stale >= config.patience:
            print(f"Early stopping after {epoch + 1} epochs; best validation={best_loss:.6f}", flush=True)
            break

    with (output / "pels_vae_history.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("epoch", "train_loss", "validation_loss", "beta", "train_mse",
                         "train_mu_mse", "train_logvar_mse", "train_kl", "validation_mse",
                         "validation_mu_mse", "validation_logvar_mse", "validation_kl",
                         "validation_beta"))
        writer.writerows(history)
    return best_loss, history
