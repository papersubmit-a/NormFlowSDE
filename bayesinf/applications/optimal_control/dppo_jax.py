"""JAX adaptation of the official DPPO update for stochastic interpolants.

Formula provenance:
``third_party/dppo/model/diffusion/diffusion_ppo.py`` and
``third_party/dppo/agent/finetune/train_ppo_diffusion_agent.py``.

The generative parameterization differs, but the outer GAE computation and
inner-MDP PPO objective intentionally follow DPPO: likelihoods are averaged
over executed action coordinates, advantages are normalized, discounted over
inner steps, and optimized with step-dependent clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .cond_stoch_interpolant import CondStochInterpolantPolicy
from .dppo_diffusion_jax import DPPODiffusionPolicy


@dataclass(frozen=True)
class DPPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    gamma_denoising: float = 0.99
    clip_ploss_coef: float = 0.01
    clip_ploss_coef_base: float = 0.01
    clip_ploss_coef_rate: float = 3.0
    vf_coef: float = 0.5
    target_kl: float = 1.0
    reward_horizon: int = 4
    ft_denoising_steps: int = 10
    bc_reference_coef: float = 0.0
    normalize_advantage: bool = True
    logprob_min: float = -5.0
    logprob_max: float = 2.0


class RolloutBatch(NamedTuple):
    states: jnp.ndarray
    chains: jnp.ndarray
    old_log_probs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    old_values: jnp.ndarray


class DPPOMetrics(NamedTuple):
    total_loss: jnp.ndarray
    policy_loss: jnp.ndarray
    value_loss: jnp.ndarray
    approx_kl: jnp.ndarray
    clip_fraction: jnp.ndarray
    ratio: jnp.ndarray
    bc_reference_loss: jnp.ndarray


class ValueCritic(eqx.Module):
    """DPPO's state-only 3x256 Mish value network."""

    layers: tuple[eqx.nn.Linear, ...]

    def __init__(
        self,
        obs_dim: int,
        cond_steps: int,
        key: jr.PRNGKey,
        hidden_size: int = 256,
        depth: int = 3,
    ):
        if depth < 1:
            raise ValueError("depth must be positive")
        keys = jr.split(key, depth + 1)
        sizes = [obs_dim * cond_steps] + [hidden_size] * depth + [1]
        self.layers = tuple(
            eqx.nn.Linear(sizes[i], sizes[i + 1], key=keys[i])
            for i in range(len(sizes) - 1)
        )

    def __call__(self, state: jnp.ndarray) -> jnp.ndarray:
        value = state.reshape(-1)
        for layer in self.layers[:-1]:
            value = jax.nn.mish(layer(value))
        return self.layers[-1](value).squeeze()

    def batch(self, states: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(self)(states)


def generalized_advantage_estimate(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    next_value: jnp.ndarray,
    terminated: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """DPPO-compatible GAE over ``(time, environments)`` arrays."""
    if rewards.shape != values.shape or rewards.shape != terminated.shape:
        raise ValueError("rewards, values, and terminated must have equal shapes")

    def reverse_step(carry, inputs):
        reward, value, following_value, terminal = inputs
        nonterminal = 1.0 - terminal.astype(value.dtype)
        delta = reward + gamma * following_value * nonterminal - value
        advantage = delta + gamma * gae_lambda * nonterminal * carry
        return advantage, advantage

    following = jnp.concatenate([values[1:], next_value[None]], axis=0)
    _, reversed_advantages = jax.lax.scan(
        reverse_step,
        jnp.zeros_like(next_value),
        (rewards[::-1], values[::-1], following[::-1], terminated[::-1]),
    )
    advantages = reversed_advantages[::-1]
    return advantages, advantages + values


def _clip_coefficients(steps: int, config: DPPOConfig, dtype) -> jnp.ndarray:
    fraction = jnp.arange(steps, dtype=dtype) / max(steps - 1, 1)
    if steps == 1:
        return jnp.zeros((1,), dtype=dtype)
    rate = config.clip_ploss_coef_rate
    return config.clip_ploss_coef_base + (
        config.clip_ploss_coef - config.clip_ploss_coef_base
    ) * (jnp.exp(rate * fraction) - 1.0) / (jnp.exp(rate) - 1.0)


def dppo_loss(
    policy: CondStochInterpolantPolicy | DPPODiffusionPolicy,
    critic: ValueCritic,
    batch: RolloutBatch,
    config: DPPOConfig,
    base_policy=None,
) -> tuple[jnp.ndarray, DPPOMetrics]:
    """Joint actor/critic objective matching official DPPO conventions."""
    reference_log_probs = None
    if isinstance(policy, DPPODiffusionPolicy) and base_policy is not None:
        new_log_probs = policy.hybrid_transition_log_prob(
            batch.states, batch.chains, base_policy, config.ft_denoising_steps
        )
        first = policy.denoising_steps - config.ft_denoising_steps
        reference_log_probs = base_policy.transition_log_prob(
            batch.states, batch.chains,
            jnp.arange(first, policy.denoising_steps),
        )
    else:
        new_log_probs = policy.transition_log_prob(batch.states, batch.chains)
        if base_policy is not None:
            reference_log_probs = base_policy.transition_log_prob(
                batch.states, batch.chains
            )
    new_log_probs = jnp.clip(
        new_log_probs, config.logprob_min, config.logprob_max
    )
    old_log_probs = jnp.clip(
        batch.old_log_probs, config.logprob_min, config.logprob_max
    )
    reward_horizon = min(config.reward_horizon, policy.horizon)
    new_log_probs = jnp.mean(
        new_log_probs[:, :, :reward_horizon], axis=(-1, -2)
    )
    old_log_probs = jnp.mean(
        old_log_probs[:, :, :reward_horizon], axis=(-1, -2)
    )
    if reference_log_probs is None or config.bc_reference_coef == 0:
        bc_reference_loss = jnp.asarray(0.0, dtype=new_log_probs.dtype)
    else:
        reference_log_probs = jnp.clip(
            reference_log_probs, config.logprob_min, config.logprob_max
        )
        reference_log_probs = jnp.mean(
            reference_log_probs[:, :, :reward_horizon], axis=(-1, -2)
        )
        # A sampled transition log-ratio penalty anchors the actor to the
        # frozen BC transition kernel without requiring model-specific KL code.
        bc_reference_loss = jnp.mean(
            (new_log_probs - jax.lax.stop_gradient(reference_log_probs)) ** 2
        )
    advantages = batch.advantages
    if config.normalize_advantage:
        advantages = (advantages - jnp.mean(advantages)) / (
            jnp.std(advantages) + 1e-8
        )
    inner_steps = new_log_probs.shape[1]
    discounts = config.gamma_denoising ** jnp.arange(
        inner_steps - 1, -1, -1, dtype=advantages.dtype
    )
    inner_advantages = advantages[:, None] * discounts[None]
    log_ratio = new_log_probs - old_log_probs
    ratio = jnp.exp(log_ratio)
    clip = _clip_coefficients(inner_steps, config, ratio.dtype)
    unclipped = -inner_advantages * ratio
    clipped = -inner_advantages * jnp.clip(ratio, 1.0 - clip, 1.0 + clip)
    policy_loss = jnp.mean(jnp.maximum(unclipped, clipped))

    predicted_values = critic.batch(batch.states)
    value_loss = 0.5 * jnp.mean((predicted_values - batch.returns) ** 2)
    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_fraction = jnp.mean(jnp.abs(ratio - 1.0) > clip[None])
    total = (
        policy_loss + config.vf_coef * value_loss
        + config.bc_reference_coef * bc_reference_loss
    )
    return total, DPPOMetrics(
        total, policy_loss, value_loss, approx_kl, clip_fraction,
        jnp.mean(ratio), bc_reference_loss,
    )


class BCTrainState(NamedTuple):
    policy: CondStochInterpolantPolicy
    optimizer_state: optax.OptState
    step: jnp.ndarray


def _bc_networks(policy):
    """Train learned modules only; analytical schedules stay immutable."""
    if isinstance(policy, DPPODiffusionPolicy):
        return policy.network
    return policy.encoder, policy.vel_net, policy.score_net


def _replace_bc_networks(policy, networks):
    if isinstance(policy, DPPODiffusionPolicy):
        return eqx.tree_at(lambda p: p.network, policy, networks)
    return eqx.tree_at(lambda p: (p.encoder, p.vel_net, p.score_net),
                       policy, networks, is_leaf=lambda x: x is None)


def make_bc_state(policy, learning_rate=1e-3, weight_decay=1e-6):
    optimizer = optax.adamw(learning_rate, weight_decay=weight_decay)
    # The DDPM schedule and posterior coefficients are analytical constants,
    # but are stored as JAX arrays on DPPODiffusionPolicy.  Initialising AdamW
    # over the complete policy would therefore train (and independently
    # corrupt) those constants.  Diffusion BC optimises only the noise network;
    # SI BC likewise optimises its encoder/velocity/score networks only.
    optimized = _bc_networks(policy)
    return BCTrainState(
        policy, optimizer.init(eqx.filter(optimized, eqx.is_array)), jnp.array(0)
    ), optimizer


@eqx.filter_jit
def bc_update(
    state: BCTrainState,
    optimizer: optax.GradientTransformation,
    actions: jnp.ndarray,
    observations: jnp.ndarray,
    key: jr.PRNGKey,
) -> tuple[BCTrainState, jnp.ndarray]:
    optimized = _bc_networks(state.policy)

    def objective(model):
        policy = _replace_bc_networks(state.policy, model)
        return policy.loss(actions, observations, key)

    loss, gradients = eqx.filter_value_and_grad(objective)(optimized)
    updates, optimizer_state = optimizer.update(
        gradients, state.optimizer_state, optimized
    )
    optimized = eqx.apply_updates(optimized, updates)
    policy = _replace_bc_networks(state.policy, optimized)
    return BCTrainState(policy, optimizer_state, state.step + 1), loss


class DPPOTrainState(NamedTuple):
    policy: CondStochInterpolantPolicy | DPPODiffusionPolicy
    base_policy: CondStochInterpolantPolicy | DPPODiffusionPolicy
    critic: ValueCritic
    actor_optimizer_state: optax.OptState
    critic_optimizer_state: optax.OptState
    step: jnp.ndarray


def make_dppo_state(
    policy,
    critic,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    max_grad_norm=None,
):
    actor_optimizer = optax.chain(
        optax.clip_by_global_norm(max_grad_norm) if max_grad_norm else optax.identity(),
        optax.adamw(actor_learning_rate, weight_decay=0.0),
    )
    critic_optimizer = optax.adamw(critic_learning_rate, weight_decay=0.0)
    # SI fine-tunes only the velocity field, freezing its informative source.
    # Diffusion fine-tunes its noise-prediction network; its Gaussian source
    # has no parameters. This is the matched trainable actor split.
    actor = policy.vel_net if hasattr(policy, "vel_net") else policy.network
    actor_state = actor_optimizer.init(eqx.filter(actor, eqx.is_array))
    critic_state = critic_optimizer.init(eqx.filter(critic, eqx.is_array))
    return (
        DPPOTrainState(policy, policy, critic, actor_state, critic_state, jnp.array(0)),
        actor_optimizer,
        critic_optimizer,
    )


@eqx.filter_jit
def dppo_update(
    state: DPPOTrainState,
    actor_optimizer: optax.GradientTransformation,
    critic_optimizer: optax.GradientTransformation,
    batch: RolloutBatch,
    config: DPPOConfig,
) -> tuple[DPPOTrainState, DPPOMetrics]:
    is_interpolant = hasattr(state.policy, "vel_net")
    current_actor = (
        state.policy.vel_net if is_interpolant else state.policy.network
    )

    def replace_actor(policy, actor):
        if is_interpolant:
            return eqx.tree_at(lambda model: model.vel_net, policy, actor)
        return eqx.tree_at(lambda model: model.network, policy, actor)

    def objective(models):
        actor, critic = models
        policy = replace_actor(state.policy, actor)
        return dppo_loss(policy, critic, batch, config, state.base_policy)

    (_, metrics), (actor_grad, critic_grad) = eqx.filter_value_and_grad(
        objective, has_aux=True
    )((current_actor, state.critic))
    actor_updates, actor_optimizer_state = actor_optimizer.update(
        actor_grad, state.actor_optimizer_state, current_actor
    )
    critic_updates, critic_optimizer_state = critic_optimizer.update(
        critic_grad, state.critic_optimizer_state, state.critic
    )
    actor = eqx.apply_updates(current_actor, actor_updates)
    critic = eqx.apply_updates(state.critic, critic_updates)
    policy = replace_actor(state.policy, actor)
    return (
        DPPOTrainState(
            policy,
            state.base_policy,
            critic,
            actor_optimizer_state,
            critic_optimizer_state,
            state.step + 1,
        ),
        metrics,
    )
