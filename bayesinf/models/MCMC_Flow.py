"""Conditional Metropolis kernels and stochastic-flow path-space training.

Callbacks return one log density per row: ``log_posterior(x, condition,
 gradient_scale) -> (batch,)``. ``gradient_scale`` is retained for callback
compatibility; exact training/refinement uses 1. A kernel returns its change in
energy, a path-density ratio, not the Jacobian of a deterministic map.
"""
from typing import Callable, Optional
from numbers import Integral
from bayesinf.models.optimization import preserve_model_state
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from bayesinf.models.networks import TimeSeriesEncoder
from bayesinf.models.flow_layers import ActNorm, RandomPermutation, AffineCoupling


def gradient_schedule(x, gradient_scale):
    """Explicit surrogate-gradient utility; not used by exact SNF training."""
    return jax.lax.stop_gradient(x) + gradient_scale * (x - jax.lax.stop_gradient(x))


def _accept(x, proposal, log_ratio, key):
    """Independent MH choice and differentiable log probability of that choice."""
    log_a = jnp.minimum(log_ratio, 0.0)
    accepted = jnp.log(jr.uniform(key, (x.shape[0],))) < log_a
    # At acceptance probability one, the rejected branch is impossible. Avoid
    # differentiating log(0) in that unused branch.
    safe_log_a = jnp.where(log_a < 0, log_a, -1.0)
    log_reject = jnp.log(-jnp.expm1(safe_log_a))
    log_choice = jnp.where(accepted, log_a, log_reject)
    return jnp.where(accepted[:, None], proposal, x), log_choice


class _MetropolisKernel(eqx.Module):
    log_posterior: Callable = eqx.field(static=True)
    lambd: float = eqx.field(static=True)
    num_mh_steps: int = eqx.field(static=True)
    apply_gradient_schedule: bool = eqx.field(static=True)

    def _initialize(self, log_posterior, lambd, num_mh_steps, apply_gradient_schedule):
        if not 0 <= lambd <= 1:
            raise ValueError("lambd must be in [0, 1]")
        if num_mh_steps < 0:
            raise ValueError("num_mh_steps must be nonnegative")
        if apply_gradient_schedule:
            raise ValueError("Energy-ratio gradient suppression is not valid SNF training")
        self.log_posterior = log_posterior
        self.lambd = float(lambd)
        self.num_mh_steps = int(num_mh_steps)
        self.apply_gradient_schedule = False

    def get_interpolated_energy(self, x, condition, gradient_scale=1.0):
        prior = .5 * jnp.sum(x ** 2, axis=-1)
        if self.lambd == 0:
            return prior
        log_density = self.log_posterior(x, condition, gradient_scale)
        if isinstance(log_density, tuple):
            raise TypeError("log_posterior must return a (batch,) log-density array, not an energy/auxiliary tuple")
        log_density = jnp.asarray(log_density)
        if log_density.shape != x.shape[:-1]:
            raise ValueError("log_posterior must return one log density per sample")
        # Do not form 0*inf for bounded-support targets at lambda=1.
        return -self.lambd * log_density + (1 - self.lambd) * prior

    def energy_grad(self, x, condition, gradient_scale=1.0):
        def single(value, cond):
            return self.get_interpolated_energy(value[None], cond[None], gradient_scale)[0]
        values, gradients = jax.vmap(jax.value_and_grad(single))(x, condition)
        return gradients, values

    def transition(self, x, condition, key, cond_emb=None, gradient_scale=1.0):
        del cond_emb
        initial = self.get_interpolated_energy(x, condition, gradient_scale)
        x = eqx.error_if(x, jnp.any(~jnp.isfinite(initial)),
            "MCMC initial states must have finite target energy; initialize chains inside support")
        current = x
        choices = jnp.zeros(x.shape[0])
        for _ in range(self.num_mh_steps):
            key, step_key = jr.split(key)
            current, log_choice = self._step(current, condition, step_key, gradient_scale)
            choices = choices + log_choice
        final = self.get_interpolated_energy(current, condition, gradient_scale)
        # This term must remain differentiable for path-space training.
        return current, final - initial, choices

    def __call__(self, x, condition, key, cond_emb=None, gradient_scale=1.0):
        value, ratio, _ = self.transition(x, condition, key, cond_emb, gradient_scale)
        return value, ratio


class MALALayer(_MetropolisKernel):
    """Metropolis-adjusted Langevin proposals, including auxiliary multi-step paths."""
    num_langevin_steps: int = eqx.field(static=True)
    step_size: float = eqx.field(static=True)

    def __init__(self, log_posterior, lambd=1., num_mh_steps=1,
                 num_langevin_steps=1, step_size=1e-3, apply_gradient_schedule=False):
        self._initialize(log_posterior, lambd, num_mh_steps, apply_gradient_schedule)
        if step_size <= 0 or num_langevin_steps < 1:
            raise ValueError("Positive step_size and num_langevin_steps required")
        self.step_size = float(step_size)
        self.num_langevin_steps = int(num_langevin_steps)

    def langevin_proposal(self, x, condition, key, gradient_scale=1.0):
        h = self.step_size
        initial = self.get_interpolated_energy(x, condition, gradient_scale)
        log_ratio = jnp.zeros(x.shape[0])
        current = x
        for _ in range(self.num_langevin_steps):
            key, noise_key = jr.split(key)
            grad, _ = self.energy_grad(current, condition, gradient_scale)
            noise = jr.normal(noise_key, x.shape)
            proposed = current - h * grad + jnp.sqrt(2 * h) * noise
            grad_reverse, _ = self.energy_grad(proposed, condition, gradient_scale)
            reverse_noise = (current - proposed + h * grad_reverse) / jnp.sqrt(2 * h)
            log_ratio += .5 * jnp.sum(noise ** 2 - reverse_noise ** 2, axis=-1)
            current = proposed
        return current, log_ratio, initial, self.get_interpolated_energy(current, condition, gradient_scale)

    def _step(self, x, condition, key, gradient_scale):
        proposal_key, accept_key = jr.split(key)
        proposal, ratio, initial, final = self.langevin_proposal(x, condition, proposal_key, gradient_scale)
        return _accept(x, proposal, initial - final + ratio, accept_key)

    def mh_step(self, x, condition, key, gradient_scale=1.0):
        value, _ = self._step(x, condition, key, gradient_scale)
        return value, self.get_interpolated_energy(value, condition, gradient_scale)


class MCMCLayer(_MetropolisKernel):
    """Gaussian random-walk Metropolis kernel."""
    noise_std: float = eqx.field(static=True)

    def __init__(self, log_posterior, lambd=1., num_mh_steps=3, noise_std=.1,
                 apply_gradient_schedule=False):
        self._initialize(log_posterior, lambd, num_mh_steps, apply_gradient_schedule)
        if noise_std <= 0:
            raise ValueError("noise_std must be positive")
        self.noise_std = float(noise_std)

    def _step(self, x, condition, key, gradient_scale):
        proposal_key, accept_key = jr.split(key)
        proposal = x + self.noise_std * jr.normal(proposal_key, x.shape)
        ratio = (self.get_interpolated_energy(x, condition, gradient_scale)
                 - self.get_interpolated_energy(proposal, condition, gradient_scale))
        return _accept(x, proposal, ratio, accept_key)


class RMALALayer(_MetropolisKernel):
    """Diagonal gradient-preconditioned MH; complete state-dependent Gaussian density."""
    base_step_size: float = eqx.field(static=True)
    eps: float = eqx.field(static=True, default=1e-6)

    def __init__(self, log_posterior, lambd=1., num_mh_steps=1, step_size=1e-3,
                 apply_gradient_schedule=False):
        self._initialize(log_posterior, lambd, num_mh_steps, apply_gradient_schedule)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.base_step_size = float(step_size)

    def get_geometry(self, x, condition, gradient_scale=1.0):
        grad, energy = self.energy_grad(x, condition, gradient_scale)
        return energy, grad, 1 / (jnp.abs(grad) + self.eps)

    def _step(self, x, condition, key, gradient_scale):
        proposal_key, accept_key = jr.split(key)
        h = self.base_step_size
        energy, grad, precond = self.get_geometry(x, condition, gradient_scale)
        mean, variance = x - h * precond * grad, 2 * h * precond
        proposal = mean + jnp.sqrt(variance) * jr.normal(proposal_key, x.shape)
        new_energy, new_grad, new_precond = self.get_geometry(proposal, condition, gradient_scale)
        reverse_mean, reverse_variance = proposal - h * new_precond * new_grad, 2 * h * new_precond
        log_forward = -.5 * jnp.sum((proposal - mean)**2 / variance + jnp.log(variance), axis=-1)
        log_reverse = -.5 * jnp.sum((x - reverse_mean)**2 / reverse_variance + jnp.log(reverse_variance), axis=-1)
        return _accept(x, proposal, energy - new_energy + log_reverse - log_forward, accept_key)


class BMCMCLayer(_MetropolisKernel):
    """Barker proposals with their required Metropolis-Hastings correction."""
    step_size: float = eqx.field(static=True)

    def __init__(self, log_posterior, lambd=1., num_mh_steps=1, step_size=.01,
                 apply_gradient_schedule=False):
        self._initialize(log_posterior, lambd, num_mh_steps, apply_gradient_schedule)
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.step_size = float(step_size)

    def _step(self, x, condition, key, gradient_scale):
        noise_key, sign_key, accept_key = jr.split(key, 3)
        grad, energy = self.energy_grad(x, condition, gradient_scale)
        noise = self.step_size * jr.normal(noise_key, x.shape)
        logits = -grad * noise
        positive = jr.uniform(sign_key, x.shape) < jax.nn.sigmoid(logits)
        displacement = jnp.where(positive, noise, -noise)
        sign_log_prob = jnp.sum(jax.nn.log_sigmoid(jnp.where(positive, logits, -logits)), axis=-1)
        proposal = x + displacement
        reverse_grad, reverse_energy = self.energy_grad(proposal, condition, gradient_scale)
        log_forward = jnp.sum(jax.nn.log_sigmoid(-grad * displacement), axis=-1)
        log_reverse = jnp.sum(jax.nn.log_sigmoid(reverse_grad * displacement), axis=-1)
        value, choice = _accept(x, proposal, energy - reverse_energy + log_reverse - log_forward, accept_key)
        return value, choice + sign_log_prob


STOCHASTIC_LAYERS = (MALALayer, RMALALayer, MCMCLayer, BMCMCLayer)


def make_refinement(log_posterior, mcmc_type='MALA', *, num_mh_steps=1,
                    step_size=1e-3, noise_std=.01, lambd=1.):
    name = mcmc_type.upper()
    classes = {'MALA': MALALayer, 'RMALA': RMALALayer, 'BMCMC': BMCMCLayer, 'MCMC': MCMCLayer}
    if name not in classes:
        raise ValueError(f"Unsupported MCMC kernel {mcmc_type!r}")
    if log_posterior is None:
        return None
    options = {'noise_std': noise_std} if name == 'MCMC' else {'step_size': step_size}
    return classes[name](log_posterior, lambd=lambd, num_mh_steps=num_mh_steps, **options)


class CondSNF(eqx.Module):
    """Conditional SNF. Training uses a path-space variational bound, not exact NLL."""
    dim: int = eqx.field(static=True)
    cond_dim: int = eqx.field(static=True)
    n_stochastic: tuple = eqx.field(static=True)
    encoder: TimeSeriesEncoder
    transforms: tuple
    refinement: Optional[eqx.Module]
    log_posterior_fn: Callable = eqx.field(static=True)
    apply_gradient_schedule: bool = eqx.field(static=True)

    def __init__(self, dim, cond_dim, log_posterior_fn, key, n_coupling=6,
                 n_stochastic=None, hidden_nf=128, num_hidden_nf=2,
                 encoded_latent_dim=16, hidden_rnn=128, rnn_type='lstm',
                 flow_type='realnvp', mcmc_type='MALA', num_mh_steps=1,
                 mcmc_step_size=1e-3, noise_std=.01, apply_gradient_schedule=False, refinement_log_posterior_fn=None):
        if n_coupling < 2 or n_coupling % 2:
            raise ValueError("SNF requires a positive even n_coupling")
        if flow_type.lower() != 'realnvp':
            raise ValueError("CondSNF currently supports only realnvp deterministic layers")
        if apply_gradient_schedule:
            raise ValueError("SNF path-ratio gradients cannot be suppressed")
        blocks = n_coupling // 2
        positions = tuple(range(1, blocks + 1)) if n_stochastic is None else ((n_stochastic,) if isinstance(n_stochastic, int) else tuple(n_stochastic))
        if any(not isinstance(position, Integral) or position < 0 or position > blocks for position in positions) or len(set(positions)) != len(positions):
            raise ValueError(f"Stochastic positions must be distinct integers in [0, {blocks}]; 0 means final refinement")
        self.dim, self.cond_dim = dim, cond_dim
        self.n_stochastic = positions
        self.log_posterior_fn = log_posterior_fn
        self.apply_gradient_schedule = False
        self.refinement = make_refinement(refinement_log_posterior_fn or log_posterior_fn, mcmc_type, num_mh_steps=num_mh_steps,
            step_size=mcmc_step_size, noise_std=noise_std) if 0 in positions else None
        encoder_key, flow_key = jr.split(key)
        self.encoder = TimeSeriesEncoder(cond_dim, hidden_rnn, encoded_latent_dim, encoder_key, rnn=rnn_type)
        transforms = []
        for index, layer_key in enumerate(jr.split(flow_key, blocks)):
            k1, k2, kp = jr.split(layer_key, 3)
            mask = (jnp.arange(dim) + index) % 2
            transforms.extend((RandomPermutation(dim, kp),
                AffineCoupling(dim, [hidden_nf] * num_hidden_nf, encoded_latent_dim, mask, k1),
                AffineCoupling(dim, [hidden_nf] * num_hidden_nf, encoded_latent_dim, 1-mask, k2)))
            if index + 1 in positions:
                transforms.append(make_refinement(log_posterior_fn, mcmc_type,
                    num_mh_steps=num_mh_steps, step_size=mcmc_step_size,
                    noise_std=noise_std, lambd=1-(index+1)/blocks))
        self.transforms = tuple(transforms)

    def _path(self, value, cond, cond_emb, key, reverse=False, gradient_scale=1.0):
        ratio = jnp.zeros(value.shape[0])
        choices = jnp.zeros(value.shape[0])
        layers = self.transforms if not reverse else tuple(reversed(self.transforms))
        for layer, layer_key in zip(layers, jr.split(key, len(layers))):
            if isinstance(layer, STOCHASTIC_LAYERS):
                value, increment, log_choices = layer.transition(value, cond, layer_key, cond_emb, gradient_scale)
                choices = choices + log_choices
            elif isinstance(layer, (ActNorm, RandomPermutation)):
                value, increment = layer(value, reverse=reverse)
            else:
                value, increment = layer(value, cond_emb, reverse=reverse)
            ratio = ratio + increment
        return value, ratio, choices

    def __flow_call__(self, z, cond, cond_emb, key, reverse=False, gradient_scale=1.0):
        value, ratio, _ = self._path(z, cond, cond_emb, key, reverse, gradient_scale)
        return value, ratio

    def __call__(self, x, cond, key, gradient_scale=1.0):
        return self.__flow_call__(x, cond, self.encoder(cond), key, gradient_scale=gradient_scale)

    def inverse(self, z, cond, key):
        return self.__flow_call__(z, cond, self.encoder(cond), key, reverse=True)

    def path_log_prob(self, x, cond, key, gradient_scale=1.0):
        """One path-bound draw and the log probability of its discrete choices."""
        z, ratio, choices = self._path(x, cond, self.encoder(cond), key, gradient_scale=gradient_scale)
        base = -.5 * (jnp.sum(z ** 2, axis=-1) + self.dim*jnp.log(2*jnp.pi))
        return base + ratio, choices

    def log_prob(self, x, cond, key, gradient_scale=1.0):
        if any(isinstance(layer, STOCHASTIC_LAYERS) for layer in self.transforms):
            raise ValueError("SNF marginal log density is intractable; use path_log_prob for its variational bound")
        return self.path_log_prob(x, cond, key, gradient_scale)[0]

    def sample(self, cond, key, num_samples):
        latent_key, path_key = jr.split(key)
        batch = cond.shape[0]
        signals = jnp.repeat(cond, num_samples, axis=0)
        conditions = jnp.repeat(self.encoder(cond), num_samples, axis=0)
        latent = jr.normal(latent_key, (batch * num_samples, self.dim))
        values = self.__flow_call__(latent, signals, conditions, path_key, reverse=True)[0]
        return values.reshape(batch, num_samples, self.dim).swapaxes(0, 1)

    def sample_with_MCMC(self, cond, key, num_samples):
        sample_key, refinement_key = jr.split(key)
        return refine_samples(self, self.sample(cond, sample_key, num_samples), cond, refinement_key)


def snf_loss(model, y, true_params, key, gradient_scale=1.0):
    """Negative path bound with score-function terms for all discrete choices.

    Continuous proposal noise is reparameterized. MH accept/reject and Barker
    sign choices require likelihood-ratio gradients in addition to pathwise
    derivatives. The correction has zero *value*, not zero gradient.
    """
    bound, choices = model.path_log_prob(true_params, y, key, gradient_scale)
    cost = -bound
    correction = jax.lax.stop_gradient(cost) * (choices - jax.lax.stop_gradient(choices))
    return jnp.mean(cost + correction)


@eqx.filter_jit
def make_step(model, y, true_params, key, opt_state, optimizer, filter_fn=None, gradient_scale=1.0):
    if filter_fn is not None:
        trainable, fixed = eqx.partition(model, filter_fn)
        objective = lambda value: snf_loss(eqx.combine(value, fixed), y, true_params, key, gradient_scale)
    else:
        trainable, fixed = eqx.partition(model, eqx.is_inexact_array)
        objective = lambda value: snf_loss(eqx.combine(value, fixed), y, true_params, key, gradient_scale)
    loss, gradients = eqx.filter_value_and_grad(objective)(trainable)
    updates, opt_state = optimizer.update(gradients, opt_state, trainable)
    return preserve_model_state(model, eqx.combine(eqx.apply_updates(trainable, updates), fixed)), opt_state, loss


def refine_samples(model, samples, conditions, key):
    """Refine (draw, observation, parameter) samples with independent randomness."""
    if model.refinement is None:
        return samples
    draws, batch, dim = samples.shape
    values = samples.swapaxes(0, 1).reshape(batch * draws, dim)
    signals = jnp.repeat(conditions, draws, axis=0)
    initialization_key, transition_key = jr.split(key)
    if hasattr(model.refinement.log_posterior, "initialize_chains"):
        values = model.refinement.log_posterior.initialize_chains(values, signals, initialization_key)
    values = model.refinement(values, signals, transition_key)[0]
    return values.reshape(batch, draws, dim).swapaxes(0, 1)
