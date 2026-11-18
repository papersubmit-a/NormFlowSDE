"""Parameter partitions for controlled dynamics with fixed physical metadata."""

import equinox as eqx
import jax


def control_trainable_filter(model):
    """Exclude normalization and coordinate conversion constants from AdamW."""
    selected = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    for name in ("state_mean", "state_std", "geometry_mean", "geometry_std",
                 "velocity_to_position_scale", "velocity_to_position_bias"):
        if hasattr(model, name):
            selected = eqx.tree_at(lambda item: getattr(item, name), selected,
                                    replace=False)
    return selected
