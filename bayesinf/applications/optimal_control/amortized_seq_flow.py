"""Amortized normalizing flow for bounded control sequences.

This is the optimal-control counterpart of
``state_estimation.amortized_state_flow``.  The flow acts on a complete action
chunk and is conditioned by two independent sequence-encoder heads: one for
the observation history and one for the future reference/preview.
"""

from typing import List, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.amortized_state_flow import (
    GRUSequenceEncoder,
    TransformerSequenceEncoder,
    build_cond_flow_transforms,
)
from bayesinf.models.normalizing_flow import (
    ActNorm,
    AffineCoupling,
    MAFLayer,
    RandomPermutation,
    SplineCoupling,
)


class SequenceConditioner(eqx.Module):
    """Encode a sequence into a single flow-conditioning vector.

    Keeping this as a separate module is important: the observation and
    reference inputs have different semantics, lengths, and feature sizes, so
    they do not share parameters.  The final sequence token is used as the
    summary, matching the final hidden state used by the previous GRU-only
    implementation.
    """

    encoder: eqx.Module
    encoder_type: str = eqx.field(static=True)

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        output_dim: int,
        key: jr.PRNGKey,
        *,
        encoder_type: str = "gru",
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
        max_len: int = 512,
    ):
        self.encoder_type = encoder_type.lower()
        if self.encoder_type == "gru":
            self.encoder = GRUSequenceEncoder(
                d_y=input_dim,
                d_hidden=hidden_size,
                d_enc=output_dim,
                key=key,
            )
        elif self.encoder_type == "transformer":
            self.encoder = TransformerSequenceEncoder(
                d_y=input_dim,
                d_enc=output_dim,
                key=key,
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                n_layers=n_layers,
                max_len=max_len,
            )
        else:
            raise ValueError(
                f"Unknown encoder_type {encoder_type!r}; expected 'gru' or "
                "'transformer'."
            )

    def __call__(self, sequence: jnp.ndarray) -> jnp.ndarray:
        """Map ``(batch, time, features)`` to ``(batch, output_dim)``."""
        if sequence.ndim != 3:
            raise ValueError("sequence must have shape (batch, time, features)")
        return self.encoder(sequence)[:, -1, :]


class AmortizedSeqFlow(eqx.Module):
    """Model ``p(action[t:t+H] | observation history, reference preview)``.

    The public API remains compatible with the original lane-keeping model.
    Internally it uses the same conditional flow transforms and encoder
    backends as :class:`AmortizedStateFlow`, with two conditioning heads.
    """

    observation_conditioner: SequenceConditioner
    reference_conditioner: SequenceConditioner
    transforms: List[eqx.Module]

    horizon: int = eqx.field(static=True)
    delta_max: float = eqx.field(static=True)
    saturation_factor: float = eqx.field(static=True)
    condition_dim: int = eqx.field(static=True)
    flow_type: str = eqx.field(static=True)
    density_version: int = eqx.field(static=True)
    action_resolution: float = eqx.field(static=True)

    def __init__(
        self,
        horizon: int,
        delta_max: float,
        key: jr.PRNGKey,
        hidden_size: int = 64,
        condition_dim: int = 48,
        n_coupling: int = 6,
        *,
        n_hidden: int = 2,
        flow_type: str = "realnvp",
        encoder_type: str = "gru",
        observation_dim: int = 1,
        reference_dim: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers: int = 3,
        max_len: int = 512,
        num_bins: int = 4,
        saturation_factor: float = 1.0,
        density_version: int = 2,
        action_resolution: float = 1e-4,
    ):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if delta_max <= 0:
            raise ValueError("delta_max must be positive")
        if saturation_factor < 1.0:
            raise ValueError("saturation_factor must be at least 1.0")

        if density_version not in (1, 2):
            raise ValueError("density_version must be 1 (legacy sampling) or 2")
        if not 0 < action_resolution < 1:
            raise ValueError("action_resolution must be a relative bin width in (0,1)")
        self.density_version = density_version
        self.action_resolution = float(action_resolution)
        self.horizon = horizon
        self.delta_max = float(delta_max)
        self.saturation_factor = float(saturation_factor)
        self.condition_dim = condition_dim
        self.flow_type = flow_type.lower()

        ko, kr, kf = jr.split(key, 3)
        encoder_kwargs = dict(
            encoder_type=encoder_type,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            max_len=max_len,
        )
        # The history head receives [measurement, previous control] at each
        # history step.  This makes the latent state observable under known
        # actuation instead of asking the encoder to explain all motion from y.
        self.observation_conditioner = SequenceConditioner(
            observation_dim + 1, hidden_size, condition_dim, ko, **encoder_kwargs
        )
        self.reference_conditioner = SequenceConditioner(
            reference_dim, hidden_size, condition_dim, kr, **encoder_kwargs
        )
        self.transforms = build_cond_flow_transforms(
            input_dim=horizon,
            cond_dim=2 * condition_dim + 1,
            n_blocks=n_coupling,
            hidden_layer_size=hidden_size,
            n_hidden=n_hidden,
            flow_type=self.flow_type,
            key=kf,
            num_bins=num_bins,
        )

    def condition(
        self,
        observations: jnp.ndarray,
        reference: jnp.ndarray,
        past_controls: jnp.ndarray | None = None,
        previous_steering: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return the concatenated outputs of the two condition heads."""
        if observations.shape[0] != reference.shape[0]:
            raise ValueError("observations and reference must have the same batch size")
        batch, history_len, _ = observations.shape
        if past_controls is None:
            past_controls = jnp.zeros((batch, history_len, 1), dtype=observations.dtype)
        elif past_controls.ndim == 2:
            past_controls = past_controls[..., None]
        if past_controls.shape != (batch, history_len, 1):
            raise ValueError(
                "past_controls must have shape (batch, history_length, 1)"
            )
        if previous_steering is None:
            previous_steering = past_controls[:, -1, 0]
        previous_steering = jnp.asarray(previous_steering, dtype=observations.dtype)
        if previous_steering.ndim == 2 and previous_steering.shape[-1] == 1:
            previous_steering = previous_steering[:, 0]
        if previous_steering.shape != (batch,):
            raise ValueError("previous_steering must have shape (batch,) or (batch, 1)")

        history = jnp.concatenate(
            [observations, past_controls / self.delta_max], axis=-1
        )
        return jnp.concatenate(
            [
                self.observation_conditioner(history),
                self.reference_conditioner(reference),
                (previous_steering / self.delta_max)[:, None],
            ],
            axis=-1,
        )

    def _flow(
        self, x: jnp.ndarray, cond: jnp.ndarray, reverse: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        logdet = jnp.zeros(x.shape[0], dtype=x.dtype)
        layers = reversed(self.transforms) if reverse else self.transforms
        for layer in layers:
            if isinstance(layer, (ActNorm, RandomPermutation)):
                x, ldj = layer(x, reverse=reverse)
            elif isinstance(layer, MAFLayer):
                x, ldj = layer(x, cond=cond, reverse=reverse)
            elif isinstance(layer, (AffineCoupling, SplineCoupling)):
                x, ldj = layer(x, cond, reverse=reverse)
            else:  # Keep failures clear if a new transform is added upstream.
                raise TypeError(f"Unsupported flow transform: {type(layer).__name__}")
            logdet = logdet + ldj
        return x, logdet

    def log_prob(
        self,
        actions: jnp.ndarray,
        observations: jnp.ndarray,
        reference: jnp.ndarray,
        past_controls: jnp.ndarray | None = None,
        previous_steering: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Exact version-2 density on the OPEN physical action interval.

        Version 1 hard-clipped an enlarged-support flow and has boundary
        atoms. Its old continuous score is intentionally not exposed as a
        density; it remains available for explicit legacy sampling only.
        """
        if self.density_version != 2:
            raise ValueError("legacy clipped policies have no continuous action density; retrain version 2")
        if actions.ndim != 2 or actions.shape[-1] != self.horizon:
            raise ValueError(f"actions must have shape (batch, {self.horizon})")
        inside = jnp.all(jnp.abs(actions) < self.delta_max, axis=-1)
        interior_limit = jnp.nextafter(jnp.asarray(1., actions.dtype), jnp.asarray(0., actions.dtype))
        scaled = jnp.clip(actions / self.delta_max, -interior_limit, interior_limit)
        unconstrained = jnp.arctanh(scaled) / self.saturation_factor
        z, flow_ldj = self._flow(
            unconstrained,
            self.condition(
                observations, reference, past_controls, previous_steering
            ),
        )
        base = -0.5 * jnp.sum(z**2 + jnp.log(2.0 * jnp.pi), axis=-1)
        # du/da for u=atanh(a/delta_max)/saturation_factor.
        bound_ldj = -jnp.sum(
            jnp.log(self.delta_max * self.saturation_factor * (1.0 - scaled**2)), axis=-1
        )
        result = base + flow_ldj + bound_ldj
        return jnp.where(inside, result, -jnp.inf)

    def loss(
        self,
        actions,
        observations,
        reference,
        past_controls=None,
        previous_steering=None,
        *,
        key=None,
    ) -> jnp.ndarray:
        """Density NLL with explicit uniform dequantization of saturated labels.

        Exact bound labels are interpreted as the last physical action bin
        (relative width ``action_resolution``), not continuous observations
        at a zero-mass point. The expectation over uniform dequantization is
        a lower bound on the log bin probability. Pass a fresh key during
        training; the default gives reproducible validation. Interior labels
        retain exact density NLL.
        """
        width = self.delta_max * self.action_resolution
        edge = jnp.abs(actions) == self.delta_max
        key = jr.PRNGKey(0) if key is None else key
        noise = jr.uniform(key, actions.shape, minval=jnp.finfo(actions.dtype).eps, maxval=1.0)
        interior = jnp.minimum(self.delta_max - width * noise,
                               jnp.nextafter(jnp.asarray(self.delta_max, actions.dtype), jnp.asarray(0., actions.dtype)))
        dequantized = jnp.where(edge, jnp.sign(actions) * interior, actions)
        log_density = self.log_prob(dequantized, observations, reference,
                                    past_controls, previous_steering)
        log_bin_volume = jnp.sum(edge, axis=-1) * jnp.log(width)
        return -jnp.mean(log_density + log_bin_volume) / self.horizon

    def sample(
        self,
        observations: jnp.ndarray,
        reference: jnp.ndarray,
        key: jr.PRNGKey,
        num_samples: int = 1,
        deterministic: bool = False,
        past_controls: jnp.ndarray | None = None,
        previous_steering: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Return bounded samples with shape ``(samples, batch, horizon)``."""
        cond = self.condition(
            observations, reference, past_controls, previous_steering
        )
        batch = cond.shape[0]
        shape = (num_samples, batch, self.horizon)
        z = jnp.zeros(shape) if deterministic else jr.normal(key, shape)
        cond = jnp.broadcast_to(
            cond[None], (num_samples, batch, 2 * self.condition_dim + 1)
        ).reshape(-1, 2 * self.condition_dim + 1)
        raw, _ = self._flow(z.reshape(-1, self.horizon), cond, reverse=True)
        raw = raw.reshape(num_samples, batch, self.horizon)
        if self.density_version == 1:
            internal = self.delta_max * self.saturation_factor * jnp.tanh(raw)
            return jnp.clip(internal, -self.delta_max, self.delta_max)
        # A bijection into physical bounds: no clipping atoms. The factor now
        # controls tanh sharpness, not an enlarged statistical support.
        return self.delta_max * jnp.tanh(self.saturation_factor * raw)
