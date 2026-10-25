"""Historical Lorenz reconstruction objective with causal Heun simulation.

This is an explicitly separate experiment from the manuscript objective. The
``transition`` component is the historical reference-state *Euler* transition
NLL, used as an auxiliary surrogate while the pathwise losses use Heun. The
affine-flow Jacobian is not the Jacobian of the full Heun update, so this term
must not be described as an exact Heun likelihood or generated-path entropy.

The caller supplies preceding-history observations: y[k] contains information
available before predicting x[k+1]. The model's encoder keeps its past window,
and its custom integrator freezes that context in both Heun stages.
"""

import math

import jax
import jax.numpy as jnp
import jax.random as jr


def historical_energy_score(predictions, target):
    """Historical time-averaged energy estimate, including ensemble self pairs.

    Distances are over state coordinates at each time, rather than over the
    joint flattened trajectory. The small norm smoothing and finite-ensemble
    self-pair convention deliberately reproduce the historical objective.
    """
    if predictions.ndim != 3 or target.shape != predictions.shape[1:]:
        raise ValueError("Expected predictions=(draws,time,state) and target=(time,state)")
    if predictions.shape[0] < 2:
        raise ValueError("Historical energy scoring needs at least two draws")
    distance = jnp.sqrt(jnp.sum((predictions - target[None]) ** 2, axis=-1) + 1e-6)
    pair_distance = jnp.sqrt(jnp.sum(
        (predictions[:, None] - predictions[None, :]) ** 2, axis=-1
    ) + 1e-6)
    return jnp.mean(distance) - 0.5 * jnp.mean(pair_distance)


def loss_components_single(
    model, x, y, key, *, segment_length=5,
    state_sigma=1.0 / math.sqrt(2.0), shooting_noise_std=0.025,
    rollout_active=False, rollout_draws=8,
):
    """Return four unweighted historical-style losses for one trajectory.

    x=(T+1,D) includes the supplied initial state and y=(T,O) is already
    shifted to preceding observations by the runner. ``segment_length`` counts
    state points including the anchor; five means four transitions. Each
    segment uses one independent Heun path. State NLL is averaged over valid
    time points (including its anchor) and coordinates, then equally over
    segments, as in the old reconstruction loss. A final partial segment is
    included with its own valid-point mean, rather than silently dropped.

    With the default sigma=1/sqrt(2), state_nll equals the historical sampled
    MSE plus 0.5*log(pi), preserving its parameter gradients exactly. Only
    noninitial segment anchors receive the historical start perturbation.
    Continuity averages endpoint errors against the next true anchor over
    nonfinal segments. The optional whole-path rollout starts at exact x[0],
    runs without resets, and uses the historical per-time energy estimate.
    """
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] + 1:
        raise ValueError("Expected x=(T+1,D), y=(T,O), including the supplied x0")
    if y.shape[0] < 1:
        raise ValueError("At least one transition is required")
    if segment_length < 2:
        raise ValueError("segment_length counts state points and must be at least 2")
    if state_sigma <= 0 or shooting_noise_std < 0:
        raise ValueError("state_sigma must be positive and shooting_noise_std nonnegative")
    if rollout_active and rollout_draws < 2:
        raise ValueError("rollout_draws must be at least two when rollout is active")
    if model.physics_fn is not None:
        raise ValueError("This learned-dynamics experiment requires physics_fn=None")

    context = model.encode_observation_stride(y, model.observation_stride)
    diffusion_context = model.encode_diffusion_stride(y, model.observation_stride)
    transitions, dim = y.shape[0], x.shape[1]
    span = min(segment_length - 1, transitions)
    starts = jnp.arange(0, transitions, span)
    offsets = jnp.arange(span + 1)
    dt = jnp.asarray(model.time_step, dtype=x.dtype)
    normalizer = 0.5 * jnp.log(2.0 * jnp.pi * state_sigma ** 2)
    segment_keys = jr.split(jr.fold_in(key, 0), starts.shape[0])

    def segment(start, segment_key):
        init_key, integration_key = jr.split(segment_key)
        perturbation = shooting_noise_std * jr.normal(init_key, (dim,), dtype=x.dtype)
        initial = x[start] + jnp.where(start > 0, perturbation, jnp.zeros_like(perturbation))
        times = (start + offsets).astype(x.dtype) * dt
        prediction = model._integrate_custom(
            initial, times, context, integration_key, "heun",
            c_diffusion=diffusion_context,
        )
        valid = start + offsets <= transitions
        truth = x[jnp.minimum(start + offsets, transitions)]
        squared_error = jnp.sum(jnp.where(valid[:, None], (prediction - truth) ** 2, 0.0))
        state_nll = squared_error / (2.0 * state_sigma ** 2 * dim * jnp.sum(valid)) + normalizer
        endpoint_index = jnp.minimum(span, transitions - start)
        boundary = jnp.mean((prediction[endpoint_index] - x[jnp.minimum(start + span, transitions)]) ** 2)
        return state_nll, jnp.where(start + span < transitions, boundary, 0.0)

    state_terms, continuity_terms = jax.vmap(segment)(starts, segment_keys)
    transition = model.transition_nll(x, context, diffusion_context, jnp.asarray(0.0, x.dtype), dt)
    rollout = jnp.asarray(0.0, dtype=x.dtype)
    if rollout_active:
        times = jnp.arange(transitions + 1, dtype=x.dtype) * dt
        paths = jax.vmap(lambda draw_key: model._integrate_custom(
            x[0], times, context, draw_key, "heun", c_diffusion=diffusion_context,
        ))(jr.split(jr.fold_in(key, 1), rollout_draws))
        rollout = historical_energy_score(paths, x)
    return {
        "state_nll": jnp.mean(state_terms),
        "transition": transition,
        "continuity": jnp.sum(continuity_terms) / max(starts.shape[0] - 1, 1),
        "rollout": rollout,
    }
