"""Direct (amortized, observation-conditioned) state estimators: Latent NSF, Latent SDE, SDE-GAN.

Added 2026-11-09 to extend the benchmark from "NormFlowSDE vs. three particle filters" to four direct and
four indirect (particle-filter) estimators. Each model generates the state sequence without particle weights or
resampling, like NormFlowSDE.

Index conventions follow the comparison caches: observations[k] measures states[k+1]; controls[k] drives
states[k] -> states[k+1]; states[0] is the known initial state; all arrays are in the cache's training-
normalized coordinates.

Encoder (since 2026-11-12): every model reads the observations through NormFlowSDE's own encoder (WindowEncoder:
the bidirectional GRU of flow_sde.py over the last 10 source-time measurements, NormFlowSDE's widths and control
handling per dataset). h[k] summarizes y(x[k-9]), ..., y(x[k]) and conditions transition k, exactly the inputs
NormFlowSDE's transition k receives. The 2026-11-09/10 runs used a forward GRU (width 64) over the full history
y[0:k+1], i.e. a longer window and the destination measurement y(x[k+1]) that NormFlowSDE does not see.

The published methods learn a latent state from observations alone. Our datasets pair states with
measurements, so the latent space is fixed to the physical state and the paired states supervise it:

* LatentNSF (Kiyohara et al., NeurIPS 2025, Sec. 4): VSSM with an NSF prior transition, a measurement
  decoder p(y[k] | x[k+1]) (conditional normalizing flow, as in the particle filters), and the Gaussian
  mean-field GRU posterior q(x[k+1] | y[0:k+1]) of their Eq. (15). Loss: the beta-NELBO of their Eq. (16) and
  skip-ahead KL of Eq. (17), plus (state supervision) -log q(true state) and the NSF NLL of the true states at
  gaps 1..max_gap. With direct NLL supervision at every gap the Chapman--Kolmogorov flow loss is not needed.
  Inference draws the Gaussian posterior marginals (draws at different times are independent).
  posterior="markov" (default since 2026-11-10): the mean-field posterior never sees the known x[0], which
  leaves the unobserved Lorenz96 coordinates unidentifiable (validation RMSE 2.9 -> 15 in the 20261110 runs).
  The posterior is therefore made Markov in the state, q(x[k+1] | x[k], y[0:k+1]) = N(x[k] + m, s^2) with
  (m, s) from (x[k], h[k], u[k], t_k) (structured inference network, Krishnan et al. 2017), rolled out from
  x[0] like the other direct methods. Supervision is -log q(true x[k+1] | sampled x~[k], h[k]) along an
  ancestral posterior path (as Latent SDE scores its own rollout; teacher forcing drifted at inference), and
  the KL is evaluated along the same path. The transition network has the prior's width and depth (a width-64,
  one-layer head failed on Lorenz96: test RMSE 4.9, validation 5.0 -> 11.1), widths re-matched to the
  NormFlowSDE parameter count. The skip-ahead KL is dropped (the multi-step marginal of a
  Markov posterior has no closed form; every gap is supervised by the prior NLL instead).
  posterior="meanfield" keeps the Eq. (15) version.
* LatentSDE (Li et al., AISTATS 2020): prior SDE dx = f(x,u,t)dt + g(x,u,t)dW and posterior SDE
  dx = h(x,c,u,t)dt + g(x,u,t)dW with the shared diagonal diffusion; Euler--Maruyama in the physical step;
  ELBO = E[log N(true state; x, sigma_dec^2)] - KL(path), KL by Girsanov 0.5 |(h - f)/g|^2 dt. The decoder
  reconstructs the paired true state with a learned per-coordinate std, i.e. the Gaussian decoder with
  learned variance discussed in the introduction. Inference rolls the posterior SDE out from x[0].
* SDEGAN (Kidger et al., ICML 2021), conditional: generator SDE dx = mu(x,c,u,t)dt + sigma(x,c,u,t)dW from
  x[0]; the discriminator is a neural CDE over the joint path (t, x, y, u), so it scores the state path
  together with the measurements it must explain; Wasserstein loss with a gradient penalty on the state
  path (discretize-then-optimize, so the penalty's double backpropagation is available; Kidger et al.
  use weight clipping because of the adjoint). Inference rolls the generator out from x[0].
* SigKernelSDE (Issa et al., NeurIPS 2023, "Non-adversarial training of Neural SDEs with signature kernel
  scores"), conditional: the SDE-GAN generator trained with the strictly proper signature-kernel score of the
  generated state paths against the true path instead of a critic (added 2026-11-10; replaces SDE-GAN, whose
  conditional WGAN-GP training collapsed to near-zero spread on all datasets in the 2026-11-09 runs).
"""
from __future__ import annotations

import functools
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bayesinf.applications.state_estimation.flow_sde import BiGRUEncoder
from bayesinf.applications.state_estimation.neural_stochastic_flow import NeuralStochasticFlow
from bayesinf.applications.state_estimation.normalizing_flow_dpf import ConditionalNormalizingFlow

LOG_2PI = math.log(2 * math.pi)


def gaussian_log_prob(x, mean, std):
    return jnp.sum(-0.5 * (((x - mean) / std) ** 2 + LOG_2PI) - jnp.log(std), axis=-1)


def mlp(in_size, out_size, width, depth, key, final=None):
    return eqx.nn.MLP(in_size, out_size, width, depth, activation=jax.nn.silu,
                      final_activation=final or (lambda x: x), key=key)


class WindowEncoder(eqx.Module):
    """NormFlowSDE's observation encoder (flow_sde.BiGRUEncoder.window_context), shared by the direct baselines.

    Same module, window, widths and inputs as the amortized NormFlowSDE (run.py preceding_observations and
    RealNVPFlowSDE.encode): the transition x[k] -> x[k+1] is conditioned on h[k], the bidirectional GRU summary of
    the last `window` source-time measurements y(x[k-window+1]), ..., y(x[k]), where y(x[0]) is the clean observed
    coordinates of the known x[0] and y(x[j]) = observations[j-1] for j >= 1. Entries before x[0] are masked.
    direct_control=True keeps the controls out of the GRU (the models receive u[k] separately, as NormFlowSDE
    appends it to the context); direct_control=False feeds (y, u) to the GRU (NormFlowSDE's Lorenz-96 setting).
    """

    gru: BiGRUEncoder
    hidden_size: int = eqx.field(static=True)
    window: int = eqx.field(static=True)
    direct_control: bool = eqx.field(static=True)
    observation_index: tuple = eqx.field(static=True)
    state_mean: tuple = eqx.field(static=True)
    state_std: tuple = eqx.field(static=True)
    observation_mean: tuple = eqx.field(static=True)
    observation_std: tuple = eqx.field(static=True)

    def __init__(self, observation_dim, control_dim, hidden_size, *, window, direct_control, metadata, key):
        if window < 1:
            raise ValueError("encoder window must be at least 1")
        self.hidden_size, self.window, self.direct_control = int(hidden_size), int(window), bool(direct_control)
        self.observation_index = tuple(int(i) for i in metadata["observation_indices"])
        self.state_mean, self.state_std, self.observation_mean, self.observation_std = (
            tuple(float(v) for v in np.ravel(metadata[k]))
            for k in ("state_mean", "state_std", "observation_mean", "observation_std"))
        if len(self.observation_index) != observation_dim:
            raise ValueError("observation_indices do not match the observation dimension")
        input_dim = observation_dim if self.direct_control else observation_dim + control_dim
        # d_hidden_enc = d_c = hidden_size, as in every amortized NormFlowSDE run.
        self.gru = BiGRUEncoder(input_dim, self.hidden_size, self.hidden_size, key)

    def initial_observation(self, x0):
        """Noise-free observed coordinates of the known x[0], in observation-normalized units."""
        physical = x0 * jnp.asarray(self.state_std, x0.dtype) + jnp.asarray(self.state_mean, x0.dtype)
        selected = physical[jnp.asarray(self.observation_index)]
        return (selected - jnp.asarray(self.observation_mean, x0.dtype)) / jnp.asarray(self.observation_std, x0.dtype)

    def __call__(self, x0, observations, controls):
        """(T, observation_dim), (T, control_dim) -> (T, hidden_size); row k conditions x[k] -> x[k+1]."""
        preceding = jnp.concatenate((self.initial_observation(x0)[None], observations[:-1]))
        inputs = preceding if self.direct_control else jnp.concatenate((preceding, controls), -1)
        length = inputs.shape[0]
        width = min(self.window, length)

        def one_window(index):
            indices = index - jnp.arange(width - 1, -1, -1)
            return self.gru.window_context(inputs[jnp.maximum(indices, 0)], indices >= 0)

        return jax.vmap(one_window)(jnp.arange(length))


def _times(dt, length, dtype):
    return dt * jnp.arange(length, dtype=dtype)


# ----------------------------------------------------------------------------------------------- Latent NSF
class LatentNSF(eqx.Module):
    encoder: WindowEncoder
    head: eqx.nn.MLP
    prior: NeuralStochasticFlow
    decoder: ConditionalNormalizingFlow
    dt: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    beta_skip: float = eqx.field(static=True)
    recon_weight: float = eqx.field(static=True)
    kl_weight: float = eqx.field(static=True)
    max_gap: int = eqx.field(static=True)
    markov: bool = eqx.field(static=True)
    truncate: bool = eqx.field(static=True)
    euler: bool = eqx.field(static=True)

    def __init__(self, state_dim, observation_dim, control_dim, dt, *, hidden_size, hidden_depth, flow_layers,
                 encoder_hidden, measurement_hidden=64, measurement_depth=2, measurement_layers=4, beta=1.,
                 beta_skip=1., max_gap=4, posterior="markov", truncate=False, step="residual", recon_weight=1.,
                 kl_weight=1., encoder_options, key):
        keys = jr.split(key, 4)
        self.dt, self.beta, self.beta_skip, self.max_gap = float(dt), float(beta), float(beta_skip), int(max_gap)
        # Markov posterior only: separate multipliers of the reconstruction and KL terms inside beta (ablation).
        self.recon_weight, self.kl_weight = float(recon_weight), float(kl_weight)
        if posterior not in ("markov", "meanfield"):
            raise ValueError(f"unknown posterior {posterior!r}")
        self.markov = posterior == "markov"
        # truncate: no backpropagation into the previous sampled state (one-step BPTT); the rollout still visits
        # the model's own states. Full BPTT through a chaotic learned map explodes (Lorenz96 gradient norm ~1e3).
        self.truncate = bool(truncate)
        if step not in ("residual", "euler"):
            raise ValueError(f"unknown posterior step {step!r}")
        # step="euler": mean x + dt m, std sqrt(dt) s. The unscaled residual (step="residual") trains on the robot
        # datasets but its full-BPTT gradients explode on chaotic Lorenz96; truncating BPTT instead destabilizes
        # the rollout (ablations_direct/latent_nsf_truncated_bptt).
        self.euler = step == "euler"
        self.encoder = WindowEncoder(observation_dim, control_dim, encoder_hidden, key=keys[0], **encoder_options)
        if self.markov:   # the transition network carries the dynamics at inference: width/depth as Latent SDE's drift
            self.head = mlp(encoder_hidden + state_dim + control_dim + 1, 2 * state_dim, hidden_size, hidden_depth,
                            keys[1])
        else:
            self.head = mlp(encoder_hidden, 2 * state_dim, encoder_hidden, 1, keys[1])
        if self.markov:   # start as x[k+1] = x[k] + small noise; a random head diverges over the T-step rollout
            last = self.head.layers[-1]
            self.head = eqx.tree_at(lambda m: (m.layers[-1].weight, m.layers[-1].bias), self.head,
                                    (jnp.zeros_like(last.weight), jnp.zeros_like(last.bias)))
        self.prior = NeuralStochasticFlow(state_dim, control_dim, hidden_size=hidden_size, hidden_depth=hidden_depth,
                                          num_coupling_layers=flow_layers, include_absolute_time=True, key=keys[2])
        self.decoder = ConditionalNormalizingFlow(observation_dim, state_dim, hidden_size=measurement_hidden,
                                                  hidden_depth=measurement_depth, num_flow_layers=measurement_layers,
                                                  base_scale=1.0, key=keys[3])

    def posterior(self, x0, observations, controls):
        """Mean-field posterior marginals q(x[k+1] | h[k]) (their Eq. 15; h[k] from the shared window encoder)."""
        mean, raw = jnp.split(jax.vmap(self.head)(self.encoder(x0, observations, controls)), 2, -1)
        return mean, jax.nn.softplus(raw) + 1e-3

    def transition(self, x, context, u, t):
        """Markov posterior q(x[k+1] | x[k], y[0:k+1]) = N(x[k] + m, s^2), (m, s) from (x[k], h[k], u[k], t_k)."""
        delta, raw = jnp.split(self.head(jnp.concatenate((x, context, u, t[None]))), 2, -1)
        if self.euler:   # Euler--Maruyama form, as Latent SDE: per-step Jacobian I + dt df/dx keeps BPTT stable
            return x + self.dt * delta, math.sqrt(self.dt) * (jax.nn.softplus(raw - 1.) + 1e-3)
        return x + delta, jax.nn.softplus(raw - 3.) + 1e-3                 # initial std ~0.05

    def rollout(self, x0, context, controls, times, key):
        """Ancestral draw x~[1..T] from the Markov posterior started at the known x[0].

        Returns the path, log q of each draw, and the per-step mean / std given the *sampled* previous state."""
        def step(x, inputs):
            c, u, t, k = inputs
            mean, std = self.transition(jax.lax.stop_gradient(x) if self.truncate else x, c, u, t)
            nxt = mean + std * jr.normal(k, x.shape, x.dtype)
            return nxt, (nxt, gaussian_log_prob(nxt, mean, std), mean, std)
        _, (path, log_q, mean, std) = jax.lax.scan(step, x0, (context, controls, times,
                                                              jr.split(key, controls.shape[0])))
        return path, log_q, mean, std

    def _markov_window_loss(self, states, observations, controls, key, beta_scale):
        length, dim = observations.shape[0], states.shape[-1]
        times = _times(self.dt, length + 1, states.dtype)
        context = self.encoder(states[0], observations, controls)
        roll_key, gap_key = jr.split(key)
        cum_u = jnp.concatenate((jnp.zeros_like(controls[:1]), jnp.cumsum(controls, 0)))
        # One ancestral posterior path from the known x[0] carries every term.
        draw, log_q, mean, std = self.rollout(states[0], context, controls, times[:-1], roll_key)
        # State supervision on the model's own rollout: -log q(true x[k+1] | sampled x~[k], h[k]). (Teacher forcing,
        # i.e. conditioning on the true x[k], trained a posterior that trusted x[k] and drifted at inference:
        # 20261110 runs, Lorenz96 validation RMSE 4 -> 15 while every training term improved.)
        supervised = -jnp.sum(gaussian_log_prob(states[1:], mean, std))
        path = jnp.concatenate((states[:1], draw))
        reconstruction = -jnp.sum(self.decoder.log_prob(observations, draw))
        log_p = self.prior.log_prob(draw, path[:-1], self.dt, controls, t_current=times[:-1])
        kl = jnp.sum(log_q - log_p)
        i, j = self._gap_pairs(gap_key, length)
        mean_u = (cum_u[j] - cum_u[i]) / jnp.maximum(j - i, 1)[:, None]
        prior_nll = -jnp.sum(self.prior.log_prob(states[j], states[i], (j - i) * self.dt, mean_u,
                                                 t_current=times[i]))
        scale = 1. / (length * dim)
        parts = dict(supervised=supervised * scale, prior_nll=prior_nll * scale,
                     reconstruction=reconstruction / (length * observations.shape[-1]), kl=kl * scale)
        total = (parts["supervised"] + parts["prior_nll"]
                 + beta_scale * self.beta * (self.recon_weight * parts["reconstruction"]
                                             + self.kl_weight * parts["kl"]))
        return total, parts

    def _gap_pairs(self, key, length):
        """For every start i in 0..T-1 a random gap g in 1..max_gap (truncated at T); controls averaged."""
        starts = jnp.arange(length)
        gaps = jr.randint(key, (length,), 1, self.max_gap + 1)
        return starts, jnp.minimum(starts + gaps, length)

    def _window_loss(self, states, observations, controls, key, beta_scale):
        length, dim = observations.shape[0], states.shape[-1]
        dtype = states.dtype
        times = _times(self.dt, length + 1, dtype)
        mean, std = self.posterior(states[0], observations, controls)
        eps_key, gap_key, skip_key = jr.split(key, 3)
        draw = mean + std * jr.normal(eps_key, mean.shape, dtype)             # x~[1..T]
        path = jnp.concatenate((states[:1], draw))                            # x0 known
        cum_u = jnp.concatenate((jnp.zeros_like(controls[:1]), jnp.cumsum(controls, 0)))

        def mean_control(i, j):
            return (cum_u[j] - cum_u[i]) / jnp.maximum(j - i, 1)

        supervised = -jnp.sum(gaussian_log_prob(states[1:], mean, std))
        reconstruction = -jnp.sum(self.decoder.log_prob(observations, draw))
        log_q = gaussian_log_prob(draw, mean, std)
        log_p = self.prior.log_prob(draw, path[:-1], self.dt, controls, t_current=times[:-1])
        kl = jnp.sum(log_q - log_p)
        # NSF prior NLL on the true states at random gaps (direct supervision of every gap used below).
        i, j = self._gap_pairs(gap_key, length)
        prior_nll = -jnp.sum(self.prior.log_prob(states[j], states[i], (j - i) * self.dt,
                                                 jax.vmap(mean_control)(i, j), t_current=times[i]))
        # Skip-ahead KL (their Eq. 17): q(x_j | y<=j) against the NSF prediction from the posterior at t_i.
        i, j = self._gap_pairs(skip_key, length)
        valid = (j - i) >= 2
        log_q_j = gaussian_log_prob(path[j], mean[j - 1], std[j - 1])
        log_p_j = self.prior.log_prob(path[j], path[i], jnp.maximum(j - i, 1) * self.dt,
                                      jax.vmap(mean_control)(i, j), t_current=times[i])
        skip = jnp.sum(jnp.where(valid, log_q_j - log_p_j, 0.))
        scale = 1. / (length * dim)
        parts = dict(supervised=supervised * scale, prior_nll=prior_nll * scale,
                     reconstruction=reconstruction / (length * observations.shape[-1]), kl=kl * scale,
                     skip=skip * scale)
        total = (parts["supervised"] + parts["prior_nll"]
                 + beta_scale * (self.beta * (parts["reconstruction"] + parts["kl"]) + self.beta_skip * parts["skip"]))
        return total, parts

    def loss(self, states, observations, controls, key, progress):
        beta_scale = jnp.minimum(1., progress / 0.3)                          # KL annealing over 30 %
        keys = jr.split(key, states.shape[0])
        window_loss = self._markov_window_loss if self.markov else self._window_loss
        total, parts = jax.vmap(lambda *a: window_loss(*a, beta_scale))(states, observations, controls, keys)
        return jnp.mean(total), {k: jnp.mean(v) for k, v in parts.items()}

    def sample(self, observations, x0, controls, *, key, num_samples):
        if self.markov:
            context = self.encoder(x0, observations, controls)
            times = _times(self.dt, controls.shape[0], x0.dtype)
            return jax.vmap(lambda k: self.rollout(x0, context, controls, times, k)[0])(jr.split(key, num_samples))

        mean, std = self.posterior(x0, observations, controls)
        return mean + std * jr.normal(key, (num_samples, *mean.shape), mean.dtype)


# ----------------------------------------------------------------------------------------------- Latent SDE
class LatentSDE(eqx.Module):
    encoder: WindowEncoder
    prior_drift: eqx.nn.MLP
    posterior_drift: eqx.nn.MLP
    diffusion: eqx.nn.MLP
    log_decoder_std: jax.Array
    dt: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, state_dim, observation_dim, control_dim, dt, *, hidden_size, hidden_depth, encoder_hidden,
                 substeps=1, beta=1., encoder_options, key):
        keys = jr.split(key, 4)
        self.dt, self.substeps, self.beta = float(dt), int(substeps), float(beta)
        self.encoder = WindowEncoder(observation_dim, control_dim, encoder_hidden, key=keys[0], **encoder_options)
        self.prior_drift = mlp(state_dim + control_dim + 1, state_dim, hidden_size, hidden_depth, keys[1])
        self.posterior_drift = mlp(state_dim + encoder_hidden + control_dim + 1, state_dim, hidden_size,
                                   hidden_depth, keys[2])
        self.diffusion = mlp(state_dim + control_dim + 1, state_dim, hidden_size, hidden_depth, keys[3])
        self.log_decoder_std = jnp.full((state_dim,), math.log(0.1))

    def _diffusion(self, z, u, t):
        return jax.nn.softplus(self.diffusion(jnp.concatenate((z, u, t[None])))) + 1e-3

    def rollout(self, x0, context, controls, key, with_kl=True):
        """Posterior SDE path x[1..T] (T,D) from x0 and the summed path KL."""
        h = self.dt / self.substeps

        def interval(carry, inputs):
            z, kl = carry
            c, u, k, step_key = inputs
            for s in range(self.substeps):
                t = (k * self.substeps + s) * h * jnp.ones((), z.dtype)
                post = self.posterior_drift(jnp.concatenate((z, c, u, t[None])))
                g = self._diffusion(z, u, t)
                if with_kl:
                    prior = self.prior_drift(jnp.concatenate((z, u, t[None])))
                    kl = kl + 0.5 * jnp.sum(((post - prior) / g) ** 2) * h
                z = z + post * h + g * math.sqrt(h) * jr.normal(jr.fold_in(step_key, s), z.shape, z.dtype)
            return (z, kl), z

        length = controls.shape[0]
        inputs = (context, controls, jnp.arange(length), jr.split(key, length))
        (_, kl), path = jax.lax.scan(interval, (x0, jnp.zeros((), x0.dtype)), inputs)
        return path, kl

    def _window_loss(self, states, observations, controls, key, beta_scale):
        length, dim = observations.shape[0], states.shape[-1]
        path, kl = self.rollout(states[0], self.encoder(states[0], observations, controls), controls, key)
        reconstruction = -jnp.sum(gaussian_log_prob(states[1:], path, jnp.exp(self.log_decoder_std)))
        scale = 1. / (length * dim)
        parts = dict(reconstruction=reconstruction * scale, kl=kl * scale,
                     decoder_std=jnp.mean(jnp.exp(self.log_decoder_std)))
        return parts["reconstruction"] + beta_scale * self.beta * parts["kl"], parts

    def loss(self, states, observations, controls, key, progress):
        beta_scale = jnp.minimum(1., progress / 0.3)
        keys = jr.split(key, states.shape[0])
        total, parts = jax.vmap(lambda *a: self._window_loss(*a, beta_scale))(states, observations, controls, keys)
        return jnp.mean(total), {k: jnp.mean(v) for k, v in parts.items()}

    def sample(self, observations, x0, controls, *, key, num_samples):
        context = self.encoder(x0, observations, controls)
        path_key, noise_key = jr.split(key)
        paths = jax.vmap(lambda k: self.rollout(x0, context, controls, k, with_kl=False)[0])(
            jr.split(path_key, num_samples))
        return paths + jnp.exp(self.log_decoder_std) * jr.normal(noise_key, paths.shape, paths.dtype)


# -------------------------------------------------------------------------------------------------- SDE-GAN
class SDEGenerator(eqx.Module):
    encoder: WindowEncoder
    drift: eqx.nn.MLP
    diffusion: eqx.nn.MLP
    dt: float = eqx.field(static=True)
    substeps: int = eqx.field(static=True)

    def __init__(self, state_dim, observation_dim, control_dim, dt, *, hidden_size, hidden_depth, encoder_hidden,
                 encoder_options, substeps=1, key):
        keys = jr.split(key, 3)
        self.dt, self.substeps = float(dt), int(substeps)
        self.encoder = WindowEncoder(observation_dim, control_dim, encoder_hidden, key=keys[0], **encoder_options)
        size = state_dim + encoder_hidden + control_dim + 1
        self.drift = mlp(size, state_dim, hidden_size, hidden_depth, keys[1])
        self.diffusion = mlp(size, state_dim, hidden_size, hidden_depth, keys[2])

    def rollout(self, x0, observations, controls, key):
        context = self.encoder(x0, observations, controls)
        h = self.dt / self.substeps

        def interval(z, inputs):
            c, u, k, step_key = inputs
            for s in range(self.substeps):
                t = (k * self.substeps + s) * h * jnp.ones((), z.dtype)
                features = jnp.concatenate((z, c, u, t[None]))
                g = jax.nn.softplus(self.diffusion(features)) + 1e-3
                z = z + self.drift(features) * h + g * math.sqrt(h) * jr.normal(jr.fold_in(step_key, s), z.shape,
                                                                               z.dtype)
            return z, z

        length = controls.shape[0]
        _, path = jax.lax.scan(interval, x0, (context, controls, jnp.arange(length), jr.split(key, length)))
        return path

    def sample(self, observations, x0, controls, *, key, num_samples):
        return jax.vmap(lambda k: self.rollout(x0, observations, controls, k))(jr.split(key, num_samples))


class CDEDiscriminator(eqx.Module):
    """Neural CDE over Z[k] = (t[k], x[k], y[k-1], u[k-1]) (k = 0 repeats the first y, u), Euler-discretized."""

    initial: eqx.nn.MLP
    field: eqx.nn.MLP
    readout: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)
    path_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)

    def __init__(self, state_dim, observation_dim, control_dim, dt, *, hidden_size, hidden_depth, key):
        keys = jr.split(key, 3)
        self.hidden_size, self.dt = hidden_size, float(dt)
        self.path_dim = 1 + state_dim + observation_dim + control_dim
        self.initial = mlp(self.path_dim, hidden_size, hidden_size, hidden_depth, keys[0])
        self.field = mlp(hidden_size, hidden_size * self.path_dim, hidden_size, hidden_depth, keys[1],
                         final=jnp.tanh)
        self.readout = eqx.nn.Linear(hidden_size, 1, key=keys[2])

    def __call__(self, states, observations, controls):
        """states (T+1,D) including x0 -> scalar critic value."""
        length = observations.shape[0]
        y = jnp.concatenate((observations[:1], observations))
        u = jnp.concatenate((controls[:1], controls))
        t = (self.dt * jnp.arange(length + 1, dtype=states.dtype))[:, None] / (self.dt * length)
        path = jnp.concatenate((t, states, y, u), -1)
        increments = path[1:] - path[:-1]

        def step(hidden, dz):
            matrix = self.field(hidden).reshape(self.hidden_size, self.path_dim)
            return hidden + matrix @ dz, None

        hidden, _ = jax.lax.scan(step, self.initial(path[0]), increments)
        return self.readout(hidden)[0]


class SDEGAN(eqx.Module):
    generator: SDEGenerator
    critic: CDEDiscriminator
    gradient_penalty: float = eqx.field(static=True)

    def __init__(self, state_dim, observation_dim, control_dim, dt, *, hidden_size, hidden_depth, encoder_hidden,
                 encoder_options, critic_hidden=64, critic_depth=2, substeps=1, gradient_penalty=10., key):
        generator_key, critic_key = jr.split(key)
        self.gradient_penalty = float(gradient_penalty)
        self.generator = SDEGenerator(state_dim, observation_dim, control_dim, dt, hidden_size=hidden_size,
                                      hidden_depth=hidden_depth, encoder_hidden=encoder_hidden, substeps=substeps,
                                      encoder_options=encoder_options, key=generator_key)
        self.critic = CDEDiscriminator(state_dim, observation_dim, control_dim, dt, hidden_size=critic_hidden,
                                       hidden_depth=critic_depth, key=critic_key)

    def fake(self, states, observations, controls, key):
        keys = jr.split(key, states.shape[0])
        paths = jax.vmap(self.generator.rollout)(states[:, 0], observations, controls, keys)
        return jnp.concatenate((states[:, :1], paths), 1)

    @staticmethod
    def critic_loss(critic, gradient_penalty, real, fake, observations, controls, key):
        score = jax.vmap(critic)
        fake = jax.lax.stop_gradient(fake)
        weight = jr.uniform(key, (real.shape[0], 1, 1), real.dtype)
        mixed = weight * real + (1 - weight) * fake
        grads = jax.vmap(jax.grad(critic))(mixed, observations, controls)
        norm = jnp.sqrt(jnp.sum(grads ** 2, axis=(1, 2)) + 1e-12)
        wasserstein = jnp.mean(score(real, observations, controls)) - jnp.mean(score(fake, observations, controls))
        penalty = jnp.mean((norm - 1.) ** 2)
        return -wasserstein + gradient_penalty * penalty, dict(wasserstein=wasserstein, penalty=penalty)

    def generator_loss(self, generator, states, observations, controls, key):
        keys = jr.split(key, states.shape[0])
        paths = jax.vmap(generator.rollout)(states[:, 0], observations, controls, keys)
        fake = jnp.concatenate((states[:, :1], paths), 1)
        return -jnp.mean(jax.vmap(self.critic)(fake, observations, controls))

    def sample(self, observations, x0, controls, *, key, num_samples):
        return jax.vmap(lambda k: self.generator.rollout(x0, observations, controls, k))(jr.split(key, num_samples))


# ------------------------------------------------------------------------- Neural SDE, signature-kernel score
def _goursat(increments):
    """Signature kernel from the (Lx-1, Ly-1) matrix of static-kernel increments.

    Explicit second-order scheme of Salvi et al. (2021, SIAM J. Math. Data Sci.) for k_xy(s,t) with
    d^2k/dsdt = <dx_s, dy_t> k, k(0,.) = k(.,0) = 1:
        K[i+1,j+1] = (K[i+1,j] + K[i,j+1]) A[i,j] - K[i,j] C[i,j],  A = 1 + z/2 + z^2/12,  C = 1 - z^2/12.
    For a fixed row i the recursion in j is first-order linear, K[i+1,j+1] = A K[i+1,j] + B with
    B = A K[i,j+1] - C K[i,j], so each row is one associative scan (log depth) and the scan runs over rows.
    """
    a = 1. + 0.5 * increments + increments ** 2 / 12.
    c = 1. - increments ** 2 / 12.

    def combine(left, right):
        return right[0] * left[0], right[0] * left[1] + right[1]

    def row(previous, coefficients):
        a_i, c_i = coefficients
        b_i = a_i * previous[1:] - c_i * previous[:-1]
        prod, offset = jax.lax.associative_scan(combine, (a_i, b_i))
        current = jnp.concatenate((jnp.ones((1,), previous.dtype), prod + offset))   # K[i+1, 0] = 1
        return current, None

    first = jnp.ones((increments.shape[1] + 1,), increments.dtype)
    last, _ = jax.lax.scan(row, first, (a, c))
    return last[-1]


def rbf_signature_kernel(x, y, sigma):
    """Signature kernel of the RBF-lifted paths x (Lx,d), y (Ly,d) (Issa et al. 2023, Sec. 3)."""
    distance = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, -1)
    gram = jnp.exp(-0.5 * distance / sigma ** 2)
    increments = gram[1:, 1:] - gram[1:, :-1] - gram[:-1, 1:] + gram[:-1, :-1]
    return _goursat(increments)


def linear_signature_kernel(x, y):
    """Signature kernel with the linear static kernel (used by the unit check against the closed form)."""
    dx, dy = jnp.diff(x, axis=0), jnp.diff(y, axis=0)
    return _goursat(dx @ dy.T)


class SigKernelSDE(eqx.Module):
    """Conditional neural SDE trained with the signature-kernel scoring rule (Issa et al., NeurIPS 2023).

    Generator: the SDE-GAN generator above (drift and diagonal diffusion conditioned on the causal GRU context,
    Euler--Maruyama from the known x[0]). Loss per window, with m generated paths X_1..X_m and the true path y,
    each augmented with normalized time and including x[0]:
        S = 1/(m(m-1)) sum_{i!=j} k(X_i, X_j) - 2/m sum_i k(X_i, y)   (unbiased; + k(y,y), gradient-free, so
    the reported value is the MMD^2 estimate). k is the signature kernel of the RBF-lifted path, which is
    characteristic, so the score is strictly proper (their Prop. 3.3). Non-adversarial: no critic.
    """

    generator: SDEGenerator
    num_samples: int = eqx.field(static=True)
    sigma: float = eqx.field(static=True)
    stride: int = eqx.field(static=True)

    def __init__(self, state_dim, observation_dim, control_dim, dt, *, hidden_size, hidden_depth, encoder_hidden,
                 encoder_options, substeps=1, num_samples=8, sigma=None, stride=1, key):
        self.generator = SDEGenerator(state_dim, observation_dim, control_dim, dt, hidden_size=hidden_size,
                                      hidden_depth=hidden_depth, encoder_hidden=encoder_hidden, substeps=substeps,
                                      encoder_options=encoder_options, key=key)
        self.num_samples, self.stride = int(num_samples), int(stride)
        # Default bandwidth sqrt(D) in the training-normalized state coordinates (unit-variance coordinates).
        self.sigma = float(sigma) if sigma else math.sqrt(state_dim)

    def _lift(self, path):
        """(T+1,D) path -> (L,D+1) with normalized time; every stride-th point plus the endpoint."""
        length = path.shape[0]
        index = np.unique(np.append(np.arange(0, length, self.stride), length - 1))   # static shapes
        t = jnp.asarray(index / (length - 1), path.dtype)[:, None]
        return jnp.concatenate((t, path[index]), -1)

    def _window_loss(self, states, observations, controls, key):
        m = self.num_samples
        paths = jax.vmap(lambda k: self.generator.rollout(states[0], observations, controls, k))(jr.split(key, m))
        fakes = jax.vmap(self._lift)(jnp.concatenate((jnp.broadcast_to(states[:1], (m, 1, states.shape[-1])),
                                                      paths), 1))
        truth = self._lift(states)
        kernel = functools.partial(rbf_signature_kernel, sigma=self.sigma)
        i, j = jnp.triu_indices(m, 1)
        cross = jax.vmap(kernel)(fakes[i], fakes[j])                              # m(m-1)/2 pairs
        to_truth = jax.vmap(lambda f: kernel(f, truth))(fakes)
        self_truth = jax.lax.stop_gradient(kernel(truth, truth))
        score = 2. * jnp.sum(cross) / (m * (m - 1)) - 2. * jnp.mean(to_truth)
        return score, dict(mmd2=score + self_truth, k_cross=jnp.mean(cross), k_truth=jnp.mean(to_truth),
                           k_self=self_truth)

    def loss(self, states, observations, controls, key, progress):
        del progress
        keys = jr.split(key, states.shape[0])
        score, parts = jax.vmap(self._window_loss)(states, observations, controls, keys)
        return jnp.mean(score), {k: jnp.mean(v) for k, v in parts.items()}

    def sample(self, observations, x0, controls, *, key, num_samples):
        return self.generator.sample(observations, x0, controls, key=key, num_samples=num_samples)


__all__ = ["WindowEncoder", "LatentNSF", "LatentSDE", "SDEGAN", "SigKernelSDE", "rbf_signature_kernel",
           "linear_signature_kernel"]
