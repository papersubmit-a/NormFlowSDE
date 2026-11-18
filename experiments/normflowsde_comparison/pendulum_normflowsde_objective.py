"""Explicit proper-score variants for causal pendulum NormFlowSDE training.

These opt-in variants preserve the model and information access. They are
distinct from the manuscript's default summed reconstruction/entropy loss.
All coordinates below are the frozen training-normalized coordinates.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from bayesinf.applications.state_estimation.flow_sde import energy_score


PROFILES = ("proper_rollout", "reference_nll")


def objective_definition(profile, *, rollout_every=5):
    """Serializable provenance for the opt-in loss, including its units."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown pendulum NormFlowSDE objective: {profile}")
    return dict(
        profile=profile, manuscript_default=False,
        coordinate_space="frozen training-normalized states",
        normalization=dict(
            state="summed Gaussian reconstruction / (T * D)",
            transition_entropy="summed generated-transition entropy / (T * D)",
            reference_transition_nll="exact Euler reference-transition NLL averaged over T and D",
            continuity="summed boundary discrepancies / max(1, internal_boundaries * D)",
            rollout="unbiased joint trajectory energy score / sqrt(T * D)",
            marginal_crps="unbiased distinct-pair CRPS averaged equally over time and coordinates",
        ),
        weights=dict(state=1., transition_entropy=.01 if profile == "proper_rollout" else 0.,
                     reference_transition_nll=1. if profile == "reference_nll" else 0.,
                     continuity=.1, rollout=1., marginal_crps=5.),
        rollout_every=rollout_every,
        rollout_terms=["rollout", "marginal_crps"],
        pair_denominator="M * (M - 1), with independent draws and no self-pairs",
        solver="euler", entropy_reported_when_unweighted=True,
    )


def marginal_crps(draws, truth):
    """Unbiased marginal CRPS, averaged over time and state coordinates.

    ``draws`` has shape (M,T,D). The pair expectation excludes self-pairs,
    unlike the finite empirical-distribution CRPS used to evaluate forecasts.
    """
    if draws.ndim != 3 or truth.shape != draws.shape[1:]:
        raise ValueError("Expected draws=(M,T,D) and truth=(T,D)")
    count = draws.shape[0]
    if count < 2:
        raise ValueError("Unbiased CRPS requires at least two independent draws")
    accuracy = jnp.mean(jnp.abs(draws - truth[None]))
    differences = jnp.abs(draws[:, None] - draws[None, :])
    # Diagonal differences vanish, but the U-statistic denominator remains
    # M(M-1), so the spread expectation has no self-pair bias.
    spread = jnp.sum(differences, axis=(0, 1)) / (count * (count - 1))
    return accuracy - 0.5 * jnp.mean(spread)


def loss_parts(model, states, inputs, key, *, segment_length, training_draws,
               rollout_draws, state_sigma, profile, include_rollout):
    """Return one batch objective and its consistently normalized terms."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown pendulum NormFlowSDE objective: {profile}")
    transitions, dimensions = inputs.shape[1], states.shape[-1]
    if transitions < 1 or states.shape[1] != transitions + 1:
        raise ValueError("Expected states=(B,T+1,D) and inputs=(B,T,O)")
    if segment_length < 2:
        raise ValueError("Segment length must include at least two state points")
    scale = transitions * dimensions
    boundary_count = (transitions - 1) // (segment_length - 1)
    continuity_scale = max(1, boundary_count * dimensions)
    count = training_draws if rollout_draws is None else rollout_draws
    if include_rollout and count < 2:
        raise ValueError("Proper rollout scores require at least two draws")

    def one(path, observations, draw_key):
        raw = model.paper_loss_components_single(
            path, observations, jax.random.fold_in(draw_key, 0),
            seg_length=segment_length, K=training_draws,
            state_sigma=state_sigma, include_rollout=False,
        )
        state = raw["state"] / scale
        entropy = raw["transition_entropy"] / scale
        continuity = raw["continuity"] / continuity_scale
        reference = jnp.asarray(0., dtype=path.dtype)
        joint = jnp.asarray(0., dtype=path.dtype)
        marginal = jnp.asarray(0., dtype=path.dtype)
        context = model.encode(observations)
        if profile == "reference_nll":
            # transition_nll already averages over T transitions and D states.
            reference = model.transition_nll(
                path, context, context, 0., model.time_step,
            )
        if include_rollout:
            times = jnp.arange(transitions + 1, dtype=path.dtype) * model.time_step
            draws = jax.vmap(lambda sample_key: model._integrate_custom(
                path[0], times, context, sample_key, "euler",
            )[1:])(jax.random.split(jax.random.fold_in(draw_key, 1), count))
            joint = energy_score(draws, path[1:]) / jnp.sqrt(float(scale))
            marginal = marginal_crps(draws, path[1:])
        entropy_weight = .01 if profile == "proper_rollout" else 0.
        total = (state + entropy_weight * entropy + reference
                 + .1 * continuity + joint + 5. * marginal)
        return total, dict(state=state, transition_entropy=entropy,
                           reference_transition_nll=reference,
                           continuity=continuity, rollout=joint,
                           marginal_crps=marginal)

    totals, components = jax.vmap(one)(
        states, inputs, jax.random.split(key, states.shape[0]),
    )
    return jnp.mean(totals), {name: jnp.mean(value) for name, value in components.items()}
