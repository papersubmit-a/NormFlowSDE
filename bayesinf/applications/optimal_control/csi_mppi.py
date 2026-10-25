"""CSI path proposals, discrete path selection, and receding-horizon MPPI.

The generative model predicts *geometric reference paths*, not controls or
forward dynamics. Only the tracker produces executable controls. Independent
route modes are ranked, never averaged across obstacles.
"""
from dataclasses import dataclass
from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .cond_stoch_interpolant import CondStochInterpolantPolicy


class CSITrajectoryPolicy(eqx.Module):
    """Adapt the existing CSI network to normalized interior path residuals.

    ``conditions`` are (B, history, observation), ``paths`` are (B, H+2, D).
    The H learned points are residuals from the start-to-goal line. Start and
    goal are inserted exactly at inference. Normalization is fitted on training
    data only and saved with the model; physical coordinates are never clipped.
    """

    policy: CondStochInterpolantPolicy
    condition_mean: jax.Array
    condition_scale: jax.Array
    path_scale: jax.Array

    def __init__(self, policy, condition_mean, condition_scale, path_scale):
        self.policy = policy
        self.condition_mean = jnp.asarray(condition_mean)
        self.condition_scale = jnp.asarray(condition_scale)
        self.path_scale = jnp.asarray(path_scale)
        if self.condition_mean.shape != (policy.obs_dim,):
            raise ValueError("condition_mean must match policy.obs_dim")
        if self.condition_scale.shape != (policy.obs_dim,):
            raise ValueError("condition_scale must match policy.obs_dim")
        if self.path_scale.shape != (policy.action_dim,):
            raise ValueError("path_scale must match policy.action_dim")
        for scale in (self.condition_scale, self.path_scale):
            if not np.all(np.isfinite(scale)) or not np.all(np.asarray(scale) > 0):
                raise ValueError("normalization scales must be finite and positive")
        if not np.all(np.isfinite(self.condition_mean)):
            raise ValueError("condition_mean must be finite")

    @staticmethod
    def baseline(start, goal, horizon):
        fraction = jnp.linspace(0, 1, horizon + 2)[1:-1]
        return start[..., None, :] + fraction[:, None] * (goal - start)[..., None, :]

    @classmethod
    def fit_normalization(cls, policy, conditions, paths):
        conditions, paths = jnp.asarray(conditions), jnp.asarray(paths)
        if conditions.ndim != 3 or conditions.shape[1:] != (policy.cond_steps, policy.obs_dim):
            raise ValueError("conditions must have shape (batch, cond_steps, obs_dim)")
        if paths.shape != (len(conditions), policy.horizon + 2, policy.action_dim) or not len(paths):
            raise ValueError("paths must have shape (batch, horizon+2, path_dim)")
        if not np.all(np.isfinite(conditions)) or not np.all(np.isfinite(paths)):
            raise ValueError("training arrays must be finite")
        residuals = paths[:, 1:-1] - cls.baseline(paths[:, 0], paths[:, -1], policy.horizon)
        return cls(policy, jnp.mean(conditions, axis=(0, 1)),
                   jnp.maximum(jnp.std(conditions, axis=(0, 1)), .1),
                   jnp.maximum(jnp.sqrt(jnp.mean(residuals ** 2, axis=(0, 1))), .2))

    def loss(self, paths, conditions, key):
        if paths.shape != (len(conditions), self.policy.horizon + 2, self.policy.action_dim):
            raise ValueError("paths must have shape (batch, horizon+2, path_dim)")
        targets = (paths[:, 1:-1] - self.baseline(paths[:, 0], paths[:, -1], self.policy.horizon)) / self.path_scale
        normalized = (conditions - self.condition_mean) / self.condition_scale
        return self.policy.loss(targets, normalized, key)

    @eqx.filter_jit
    def sample_paths(self, condition, start, goal, key, *, num_samples=32, sampler="sde"):
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if condition.shape != (self.policy.cond_steps, self.policy.obs_dim):
            raise ValueError("condition must have shape (cond_steps, obs_dim)")
        if start.shape != (self.policy.action_dim,) or goal.shape != start.shape:
            raise ValueError("start and goal must match the path dimension")
        conditions = jnp.broadcast_to((condition - self.condition_mean) / self.condition_scale,
                                      (num_samples,) + condition.shape)
        if sampler == "sde":
            samples = self.policy.sample_sde(conditions, key, clip=None).actions
        elif sampler == "ode":
            samples = self.policy.sample_ode(conditions, key, clip=None).actions
        else:
            raise ValueError("sampler must be 'sde' or 'ode'")
        interior = self.baseline(start, goal, self.policy.horizon) + samples * self.path_scale
        starts = jnp.broadcast_to(start, (num_samples, 1, start.size))
        goals = jnp.broadcast_to(goal, (num_samples, 1, goal.size))
        return jnp.concatenate((starts, interior, goals), axis=1)


@dataclass(frozen=True)
class CSIPlannerConfig:
    num_samples: int = 32
    sampler: str = "sde"
    replan_interval: int = 8
    switch_penalty: float = .25

    def __post_init__(self):
        if self.num_samples < 1 or self.replan_interval < 1:
            raise ValueError("num_samples and replan_interval must be positive")
        if self.sampler not in ("ode", "sde"):
            raise ValueError("sampler must be 'sde' or 'ode'")
        if not np.isfinite(self.switch_penalty) or self.switch_penalty < 0:
            raise ValueError("switch_penalty must be finite and nonnegative")


@dataclass
class PathSelection:
    path: jax.Array
    candidates: jax.Array
    costs: jax.Array
    valid: jax.Array
    selected_index: int
    source: str

    @property
    def has_valid_path(self):
        return self.selected_index >= 0


class CSITrajectoryPlanner:
    """Rank samples with an application-specific geometric cost and validity.

    ``score_paths(paths, state, goal, context) -> (costs, valid)`` must check
    entire segments, not just vertices. An optional previous path competes with
    new samples and receives a switching preference, but passes the same checks.
    If every proposal is invalid, return a stationary path and source ``stop``.
    """

    def __init__(self, policy, score_paths: Callable, config=CSIPlannerConfig()):
        self.policy, self.score_paths, self.config = policy, score_paths, config

    def select(self, condition, state, start, goal, key, context=None, previous_path=None):
        cfg = self.config
        paths = self.policy.sample_paths(condition, start, goal, key,
                                         num_samples=cfg.num_samples, sampler=cfg.sampler)
        if previous_path is not None:
            if previous_path.shape != paths.shape[1:]:
                raise ValueError("previous_path shape must match sampled paths")
            paths = jnp.concatenate((paths, previous_path[None]), axis=0)
        costs, valid = self.score_paths(paths, state, goal, context)
        costs, valid = jnp.asarray(costs), jnp.asarray(valid, dtype=bool)
        if costs.shape != (len(paths),) or valid.shape != costs.shape:
            raise ValueError("score_paths must return one cost and validity per path")
        valid = valid & jnp.isfinite(costs) & jnp.all(jnp.isfinite(paths), axis=(1, 2))
        if previous_path is not None:
            costs = costs.at[:cfg.num_samples].add(cfg.switch_penalty)
        costs = jnp.where(valid, costs, jnp.inf)
        if not bool(jnp.any(valid)):
            return PathSelection(jnp.broadcast_to(start, paths.shape[1:]), paths,
                                 costs, valid, -1, "stop")
        index = int(jnp.argmin(costs))
        source = "previous" if index == cfg.num_samples else "csi"
        return PathSelection(paths[index], paths, costs, valid, index, source)


def remaining_polyline(path, position):
    """Keep the nearest segment's suffix, preserving vertices and array size.

    Repeated terminal points pad the tail. Each new connection from the actual
    position is subsequently checked by the planner's geometric validity test.
    Calling this on the retained suffix gives monotone progress along a route.
    """
    path, position = np.asarray(path), np.asarray(position)
    delta = np.diff(path, axis=0)
    fraction = np.clip(np.sum((position - path[:-1]) * delta, axis=1) /
                       np.maximum(np.sum(delta * delta, axis=1), 1e-12), 0, 1)
    projected = path[:-1] + fraction[:, None] * delta
    index = int(np.argmin(np.sum((projected - position) ** 2, axis=1)))
    suffix = np.concatenate((position[None], path[index + 1:]), axis=0)
    return jnp.asarray(np.concatenate((suffix, np.repeat(path[-1:], len(path) - len(suffix), axis=0))))


class CSIMPPIController:
    """Slow CSI replanning with fast MPPI feedback and checked path retention.

    Application callbacks: condition_fn(state, goal, context),
    reference_fn(path, state, tracker_horizon), and optionally
    nominal_fn(reference, state). ``context`` reaches both geometric ranking
    and MPPI's running cost. ``command`` returns the applied action and plan.
    A stop fallback assumes that clipped zero control stops the application.
    """

    def __init__(self, planner, tracker, condition_fn, reference_fn, *,
                 position_fn=lambda state: state[:2], nominal_fn=None):
        self.planner, self.tracker = planner, tracker
        self.condition_fn, self.reference_fn = condition_fn, reference_fn
        self.position_fn, self.nominal_fn = position_fn, nominal_fn
        self.reset()

    def reset(self):
        self.tracker.reset()
        self.path = None
        self.last_selection = None
        self.last_reference = None
        self.step = 0
        self.replan_count = 0
        self.stop_count = 0

    def command(self, state, goal, key, *, context=None):
        position = self.position_fn(state)
        previous = None
        # Reject corruption before nearest-segment projection: projection can
        # otherwise discard a nonfinite vertex or choose a segment using NaNs.
        if self.path is not None and bool(jnp.all(jnp.isfinite(self.path))):
            previous = remaining_polyline(self.path, position)
        previous_valid = False
        if previous is not None:
            costs, valid = self.planner.score_paths(previous[None], state, goal, context)
            previous_valid = bool(valid[0] & jnp.isfinite(costs[0]) & jnp.all(jnp.isfinite(previous)))
            # Goal changes invalidate an otherwise collision-free retained route.
            previous_valid &= bool(jnp.allclose(previous[-1], goal))
        replan = self.step % self.planner.config.replan_interval == 0 or not previous_valid
        planner_key, tracker_key = jax.random.split(key)
        if replan:
            self.last_selection = self.planner.select(
                self.condition_fn(state, goal, context), state, position, goal,
                planner_key, context, previous if previous_valid else None)
            self.replan_count += 1
            self.path = self.last_selection.path if self.last_selection.has_valid_path else None
        else:
            self.path = previous
        self.step += 1
        if self.path is None:
            self.stop_count += 1
            self.tracker.reset()
            self.last_reference = None
            stop = jnp.clip(jnp.zeros_like(self.tracker.low), self.tracker.low, self.tracker.high)
            return stop, jnp.broadcast_to(stop, self.tracker.plan.shape)
        reference = self.reference_fn(self.path, state, self.tracker.config.horizon)
        self.last_reference = reference
        nominal = self.nominal_fn(reference, state) if self.nominal_fn is not None and replan else None
        return self.tracker.command(state, reference, tracker_key, context=context, nominal=nominal)
