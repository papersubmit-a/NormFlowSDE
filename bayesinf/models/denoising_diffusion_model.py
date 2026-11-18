import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from bayesinf.models.optimization import apply_model_updates, preserve_model_state
from bayesinf.models.networks import TimeSeriesEncoder
from typing import Optional
from bayesinf.models.MCMC_Flow import make_refinement, refine_samples


class SinusoidalPosEmb(eqx.Module):
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, t):
        """
        t: array-like of shape (batch,) or scalar ints/floats
        returns: (batch, dim) float32
        """
        # Ensure float array
        t = jnp.asarray(t).astype(jnp.float32)
        if t.ndim == 0:
            t = t[None]
        half = self.dim // 2
        # log(10000) scale
        freqs = jnp.exp(-jnp.log(10000.0) * jnp.arange(half) / float(max(1, half - 1)))
        # outer product: (batch, half)
        args = t[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
        # If dim is odd, pad one zero column
        if emb.shape[-1] < self.dim:
            emb = jnp.pad(emb, ((0, 0), (0, 1)))
        return emb.astype(jnp.float32)


class FiLM(eqx.Module):
    """
    Feature-wise Linear Modulation (FiLM).

    Mathematically:
        Output = γ(cond) * Input + β(cond)

    Where:
        γ (Gamma) is the Scale/Gain
        β (Beta)  is the Shift/Bias
    """
    gamma_proj: eqx.nn.Linear
    beta_proj: eqx.nn.Linear

    def __init__(self, input_dim, cond_dim, key):
        """
        Args:
            input_dim: Dimension of the feature vector 'h' to modulate.
            cond_dim: Dimension of the conditioning vector 'c' (e.g., time embedding).
            key: PRNGKey.
        """
        k1, k2 = jr.split(key)
        # Projection for Scale (Gamma)
        self.gamma_proj = eqx.nn.Linear(cond_dim, input_dim, key=k1)
        # Projection for Shift (Beta)
        self.beta_proj = eqx.nn.Linear(cond_dim, input_dim, key=k2)

    def __call__(self, h: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            h: Input features (Batch, Dim)
            cond: Conditioning embedding (Batch, CondDim)
        """
        gamma = self.gamma_proj(cond)
        beta = self.beta_proj(cond)

        # Apply modulation
        # Note: Sometimes implemented as h * (1 + gamma) + beta for identity init stability
        return h * gamma + beta


class NoiseNet(eqx.Module):
    time_mlp: eqx.nn.Sequential
    input_proj: eqx.nn.Linear
    blocks: list
    film_layers: list
    output_proj: eqx.nn.Linear

    def __init__(self, param_dim, cond_dim, time_dim, hidden_sizes, key):
        k_time, k_in, k_out, *keys = jr.split(key, 3 + len(hidden_sizes)*2)
        width = hidden_sizes[0]

        # Time embedding MLP
        self.time_mlp = eqx.nn.Sequential([
            eqx.nn.Linear(time_dim, width, key=k_time),
            eqx.nn.Lambda(jax.nn.silu),
            eqx.nn.Linear(width, time_dim, key=jr.fold_in(k_time, 1)),
        ])

        # Initial projection
        in_dim = param_dim + time_dim + cond_dim
        self.input_proj = eqx.nn.Linear(in_dim, width, key=k_in)

        # MLP blocks with FiLM conditioning
        self.blocks = []
        self.film_layers = []
        for i in range(len(hidden_sizes)):
            k1, k2 = jr.split(keys[2*i])
            self.blocks.append(eqx.nn.Sequential([
                eqx.nn.Linear(width, width, key=k1),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(width, width, key=k2),
            ]))
            self.film_layers.append(FiLM(width, cond_dim + time_dim, keys[2*i+1]))

        # Output
        self.output_proj = eqx.nn.Linear(width, param_dim, key=k_out)

    def __call__(self, z_t, t_emb, cond):
        t_cond = self.time_mlp(t_emb)  # time embedding processed

        # Cross-attention injection of cond into z
        h0 = jnp.concatenate([z_t, t_cond, cond], axis=-1)
        h = self.input_proj(h0)

        # MLP + FiLM blocks
        for block, film in zip(self.blocks, self.film_layers):
            h_res = block(h)
            h = h + film(h_res, jnp.concatenate([t_cond, cond], axis=-1))

        return self.output_proj(h)


# --- Conditional DDPM module ---
class ConditionalDDPM(eqx.Module):
    encoder: object  # TimeSeriesEncoder-like module
    net: NoiseNet
    time_emb: SinusoidalPosEmb

    n_steps: int = eqx.field(static=True)
    param_dim: int = eqx.field(static=True)
    condition_dropout: float = eqx.field(static=True)
    refinement: Optional[eqx.Module]

    def __init__(
        self,
        feature_dim,
        param_dim,
        feature_latent_dim,
        hidden_dim_rnn,
        hidden_dim_time,
        hidden_sizes_noise,
        n_steps: int = 1000,
        *,
        key,
        log_posterior_fn=None,
        mcmc_type='MALA',
        num_mh_steps: int = 1,
        mcmc_step_size: float = 1e-3,
        noise_std=0.01,
        condition_dropout: float = 0.2,
    ):
        key_enc, key_net = jr.split(key, 2)
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        self.n_steps = int(n_steps)
        self.param_dim = int(param_dim)
        if not 0.0 <= condition_dropout <= 1.0:
            raise ValueError("condition_dropout must be in [0, 1]")
        self.condition_dropout = float(condition_dropout)

        # User-supplied encoder: must map x_ts -> (batch, feature_latent_dim)
        # Keep as a placeholder type; user passes TimeSeriesEncoder instance.
        self.encoder = TimeSeriesEncoder(feature_dim, hidden_dim_rnn, feature_latent_dim, key_enc)

        self.time_emb = SinusoidalPosEmb(dim=hidden_dim_time)
        self.net = NoiseNet(param_dim, feature_latent_dim, hidden_dim_time, hidden_sizes_noise, key=key_net)

        self.refinement = make_refinement(log_posterior_fn, mcmc_type,
            num_mh_steps=num_mh_steps, step_size=mcmc_step_size, noise_std=noise_std)

    @property
    def betas(self):
        time = jnp.linspace(0.0, 1.0, self.n_steps + 1)
        cumulative = jnp.cos((time + 0.008) / 1.008 * jnp.pi / 2) ** 2
        cumulative = cumulative / cumulative[0]
        return jnp.clip(1 - cumulative[1:] / cumulative[:-1], 1e-4, 0.9999)

    @property
    def alphas(self):
        return 1 - self.betas

    @property
    def alphas_cumprod(self):
        return jnp.cumprod(self.alphas)

    @property
    def sqrt_alphas_cumprod(self):
        return jnp.sqrt(self.alphas_cumprod)

    @property
    def sqrt_one_minus_alphas_cumprod(self):
        return jnp.sqrt(1 - self.alphas_cumprod)

    def loss(self, x_ts, theta_target, key):
        """
        x_ts: (batch, ... ) time-series input for encoder
        theta_target: (batch, param_dim) ground-truth parameters
        key: PRNGKey
        """
        batch_size = theta_target.shape[0]
        key_t, key_eps, key_drop = jr.split(key, 3)

        # timesteps uniform in [0, n_steps)
        t = jr.randint(key_t, (batch_size,), minval=0, maxval=self.n_steps)
        t = t.astype(jnp.int32)

        # noise
        eps = jr.normal(key_eps, theta_target.shape)

        # coefficients (gathered per-sample)
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t][:, None]         # (batch,1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t][:, None]

        # noisy latent at timestep t
        z_t = sqrt_alpha_bar * theta_target + sqrt_one_minus_alpha_bar * eps

        # condition encoding
        cond = self.encoder(x_ts)  # (batch, cond_dim)

        # small-prob dropout on condition (per-sample)
        # create dropout mask with same leading dimension as cond
        mask = jr.bernoulli(
            key_drop, p=self.condition_dropout, shape=(batch_size,))[:, None]
        cond_input = jnp.where(mask, jnp.zeros_like(cond), cond)

        # time embeddings
        t_emb = self.time_emb(t.astype(jnp.float32))  # (batch, time_dim)

        # predict noise
        eps_pred = jax.vmap(self.net)(z_t, t_emb, cond_input)

        # MSE loss
        return jnp.mean((eps - eps_pred) ** 2)

    @eqx.filter_jit
    def sample(self, x_ts, key, num_samples: int = 1, guidance_scale=1.0,
               return_trajectory=False):
        """
        x_ts: (batch, ...), encoder input
        returns: samples shaped (batch, num_samples, param_dim)
        """

        batch_size = x_ts.shape[0]
        cond = self.encoder(x_ts)
        cond_uncond = jnp.zeros_like(cond)
        # repeat condition for num_samples
        cond_rep = jnp.repeat(cond, num_samples, axis=0)
        uncond_rep = jnp.repeat(cond_uncond, num_samples, axis=0)
        total_num = batch_size * num_samples

        # initial noise (z_T)
        key, key_init, key_noise = jr.split(key, 3)
        z = jr.normal(key_init, (total_num, self.param_dim))

        # pre-sample all noise needed for reverse chain (for t > 0)
        # shape: (n_steps, total_num, param_dim)
        noise_stack = jr.normal(key_noise, (self.n_steps, total_num, self.param_dim))

        # prepare reversed timestep indices and reversed noise sequence
        t_seq = jnp.arange(self.n_steps - 1, -1, -1, dtype=jnp.int32)  # length n_steps
        noise_seq = noise_stack[t_seq]  # (n_steps, total_num, param_dim)

        # prepare t_emb for every step: (n_steps, time_dim) - using float timesteps
        t_emb_seq = self.time_emb(t_seq.astype(jnp.float32))  # (n_steps, time_dim)
        # replicate t_emb for the whole batch when in scan: we'll broadcast inside step

        def step_fn(z_carry, elems):
            """
            z_carry: (total_num, param_dim)
            elems: tuple (t_idx_scalar, noise_for_all_samples, t_emb_for_step)
                - t_idx_scalar is an int scalar (but we pass t_emb directly instead)
            """
            noise_t = elems[0]   # (total_num, param_dim)
            t_emb_step = elems[1]  # (time_dim,)

            # Broadcast t_emb to batch
            t_emb_batched = jnp.broadcast_to(t_emb_step[None, :], (total_num, t_emb_step.shape[-1]))

            # predict noise
            eps_cond = jax.vmap(self.net)(z_carry, t_emb_batched, cond_rep)

            def compute_guided(eps_c):
                eps_uncond = jax.vmap(self.net)(z_carry, t_emb_batched, uncond_rep)
                return eps_uncond + guidance_scale * (eps_c - eps_uncond)

            # 3. Use lax.cond to skip computation if scale == 1.0
            eps_pred = jax.lax.cond(
                guidance_scale != 1.0,
                compute_guided,
                lambda x: x, # Identity function if scale == 1.0
                eps_cond
            )

            t_idx = elems[2]  # scalar int (we will pass it)
            t_idx_int = t_idx

            # read schedule scalars for this timestep
            alpha = self.alphas[t_idx_int]
            alpha_bar = self.alphas_cumprod[t_idx_int]
            beta = self.betas[t_idx_int]

            sqrt_one_minus_alpha_bar = jnp.sqrt(1.0 - alpha_bar)
            pred_x0 = (z_carry - sqrt_one_minus_alpha_bar * eps_pred) / jnp.sqrt(alpha_bar)

            # compute posterior terms (vectorized constants)
            def body_when_tpos():
                alpha_bar_prev = self.alphas_cumprod[t_idx_int - 1]
                coef1 = beta * jnp.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
                coef2 = (1.0 - alpha_bar_prev) * jnp.sqrt(alpha) / (1.0 - alpha_bar)
                mean = coef1 * pred_x0 + coef2 * z_carry
                variance = beta * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
                z_next = mean + jnp.sqrt(variance) * noise_t
                return z_next

            def body_when_t0():
                return pred_x0

            z_next = jax.lax.cond(t_idx_int > 0, body_when_tpos, body_when_t0)
            return z_next, z_next if return_trajectory else None

        # Build elems tuple array: (noise_seq[i], t_emb_seq[i], t_seq[i]) for each step
        # For scan it's convenient to pass a tuple where each element has leading dimension n_steps
        elems = (noise_seq, t_emb_seq, t_seq)

        # run scan
        z_final, path = jax.lax.scan(step_fn, z, elems)

        if return_trajectory:
            path = jnp.concatenate((z[None], path), axis=0)
            path = path.reshape(
                self.n_steps + 1, batch_size, num_samples, self.param_dim)
            return jnp.swapaxes(path, 1, 2)

        # z_final is the carry after all steps
        samples = z_final.reshape(batch_size, num_samples, self.param_dim)
        return jnp.swapaxes(samples, 0, 1)

    @eqx.filter_jit
    def sample_with_MCMC(self, x_ts, key, num_samples=1, guidance_scale=1.0):
        sample_key, refinement_key = jr.split(key)
        return refine_samples(self, self.sample(x_ts, sample_key, num_samples, guidance_scale=guidance_scale), x_ts, refinement_key)


@eqx.filter_jit
def make_step(model, y: jnp.ndarray, true_params: jnp.ndarray, key, opt_state, optimizer, filter_fn=None):
    if filter_fn is not None:
        diff_model, static_model = eqx.partition(model, filter_fn)
        def loss_fn(diff_m):
            full_model = eqx.combine(diff_m, static_model)
            return full_model.loss(y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(diff_model)
        updates, opt_state = optimizer.update(grads, opt_state, diff_model)
        diff_model = eqx.apply_updates(diff_model, updates)
        model = preserve_model_state(model, eqx.combine(diff_model, static_model))
    else:
        loss_fn = lambda m: m.loss(y, true_params, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = apply_model_updates(model, updates)
    return model, opt_state, loss
