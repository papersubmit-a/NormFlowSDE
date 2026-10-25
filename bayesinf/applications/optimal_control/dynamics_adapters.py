"""Physical-coordinate adapters independent of optional simulator packages."""

import jax
import jax.numpy as jnp


class FlowSDEDynamicsAdapter:
    """Normalize histories and controls around a fixed-step flow transition."""

    def __init__(self, model, x_mean, x_std, u_mean, u_std, solver_name='heun'):
        self.model, self.solver_name = model, solver_name
        self.x_mean, self.x_std = jnp.asarray(x_mean), jnp.asarray(x_std)
        self.u_mean, self.u_std = jnp.asarray(u_mean), jnp.asarray(u_std)

    def normalize_state(self, state):
        return (state - self.x_mean) / self.x_std

    def inverse_normalize_state(self, state):
        return state * self.x_std + self.x_mean

    def predict(self, state, action):
        """Nominal solver transition; this is not the nonlinear flow mean."""
        def one(x, u):
            path = self.model.sample_rollout(
                self.normalize_state(x), ((u-self.u_mean)/self.u_std)[None],
                jax.random.PRNGKey(0), n_particles=1, diffusion_temperature=0.,
                solver_name=self.solver_name)
            return self.inverse_normalize_state(path[0, 0])
        return one(state, action) if state.ndim == 1 else jax.vmap(one)(state, action)

    def sample_rollout(self, state_history, previous_controls, future_controls, key,
                       n_particles=1, start_step=0, diffusion_temperature=1.0):
        history = self.normalize_state(jnp.asarray(state_history))
        controls = (jnp.asarray(future_controls) - self.u_mean) / self.u_std
        paths = self.model.sample_rollout(
            history[-1], controls, key, n_particles=n_particles,
            start_step=start_step, diffusion_temperature=diffusion_temperature,
            solver_name=self.solver_name, state_history=history[:-1])
        return self.inverse_normalize_state(paths)
