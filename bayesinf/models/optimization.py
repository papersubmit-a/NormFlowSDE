"""Apply parameter updates while retaining immutable encoder and normalization state."""
import jax
import equinox as eqx
from bayesinf.models.networks import FrozenBatchNormEncoder, TrainableBatchNormEncoder
from bayesinf.models.flow_layers import ActNorm


def apply_model_updates(model, updates):
    updated = eqx.apply_updates(model, updates)
    return preserve_model_state(model, updated)


def preserve_model_state(model, updated):
    """Retain explicitly frozen parameters and nonlearned statistics after an update."""
    def preserve(before, after):
        if isinstance(before, FrozenBatchNormEncoder):
            return before
        if isinstance(before, TrainableBatchNormEncoder):
            return eqx.tree_at(lambda value: value.state, after, before.state)
        if isinstance(before, ActNorm):
            return eqx.tree_at(lambda value: (value.bias, value.logs), after,
                               (before.bias, before.logs))
        return after
    return jax.tree.map(preserve, model, updated,
        is_leaf=lambda value: isinstance(value,(FrozenBatchNormEncoder,TrainableBatchNormEncoder,ActNorm)))
