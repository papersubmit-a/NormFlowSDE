"""Backward-compatible controlled LatentSSMFlow interface."""

from .gru_latent_ssm_flow import GRULatentSSMFlow


class ControlledLatentSSMFlow(GRULatentSSMFlow):
    """Alias using the original constructor vocabulary."""

    def __init__(self, state_dim, control_dim, history_len=1, key=None, **kwargs):
        if key is None:
            import jax.random as jr
            key = jr.PRNGKey(0)
        super().__init__(
            state_dim, control_dim, history_len=history_len, key=key, **kwargs
        )
