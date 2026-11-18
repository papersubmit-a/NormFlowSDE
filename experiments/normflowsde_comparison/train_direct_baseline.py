"""Train and evaluate the direct estimators Latent NSF, Latent SDE, SigKernel SDE and SDE-GAN (direct_baselines.py).

Same data, selection and outputs as train_state_filter.py (the particle-filter baselines): random training
windows, every --eval-every updates the estimator samples all validation windows and the checkpoint with the
lowest validation RMSE is kept; the test split is sampled once after training with the protocol's posterior
draw count. Outputs: posterior.npz, metrics.json, run_metadata.json, summary.json, model.eqx, history.jsonl.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.direct_baselines import SDEGAN, LatentNSF, LatentSDE, SigKernelSDE
from experiments.normflowsde_comparison.evaluation_metrics import metrics_from_samples
from experiments.normflowsde_comparison.run import score_samples

ACCESS = ("filtering: x[k] -> x[k+1] is conditioned on NormFlowSDE's encoder (bidirectional GRU over the last "
          "--encoder-window source-time measurements y(x[k-W+1..k]), y(x[0]) = clean observed coordinates of x[0]); "
          "direct sampling, no particle weights; no future observations")
VARIANTS = dict(latent_nsf="Latent NSF (state-supervised VSSM, NSF prior, Gaussian posterior with NormFlowSDE's window-"
                           "encoder context: Markov in the state and rolled out from x[0] unless --posterior meanfield)",
                latent_sde="Latent SDE (state-supervised, posterior SDE with NormFlowSDE's window-encoder context)",
                sde_gan="conditional SDE-GAN (generator SDE with window-encoder context, neural-CDE critic, WGAN-GP)",
                sigker_sde="Neural SDE, signature-kernel score (generator SDE with NormFlowSDE's window-encoder context, "
                           "RBF signature-kernel scoring rule, non-adversarial)")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=tuple(VARIANTS))
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4400)
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--hidden-depth", type=int, default=2)
    p.add_argument("--encoder-hidden", type=int, required=True,
                   help="GRU hidden size = context size (NormFlowSDE d_hidden_enc = d_c of the dataset).")
    p.add_argument("--encoder-window", type=int, default=10, help="Measurements per encoder window (NormFlowSDE o).")
    p.add_argument("--direct-control", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep controls out of the GRU (NormFlowSDE direct_control; off on Lorenz-96).")
    p.add_argument("--flow-layers", type=int, default=6, help="Latent NSF prior coupling layers.")
    p.add_argument("--measurement-hidden", type=int, default=64, help="Latent NSF emission-flow width.")
    p.add_argument("--measurement-depth", type=int, default=2, help="Latent NSF emission-flow depth.")
    p.add_argument("--measurement-layers", type=int, default=4, help="Latent NSF emission-flow coupling layers.")
    p.add_argument("--max-gap", type=int, default=4, help="Latent NSF: largest prior / skip-KL gap in steps.")
    p.add_argument("--truncate-bptt", action="store_true",
                   help="Latent NSF (Markov): one-step backpropagation through the posterior rollout.")
    p.add_argument("--posterior-step", choices=("residual", "euler"), default="residual",
                   help="Latent NSF (Markov) transition: x + m, s (residual) or x + dt m, sqrt(dt) s (euler).")
    p.add_argument("--beta", type=float, default=1.)
    p.add_argument("--recon-weight", type=float, default=1., help="Latent NSF (Markov): reconstruction term within beta.")
    p.add_argument("--kl-weight", type=float, default=1., help="Latent NSF (Markov): KL term within beta.")
    p.add_argument("--beta-skip", type=float, default=1.)
    p.add_argument("--posterior", choices=("markov", "meanfield"), default="markov",
                   help="Latent NSF posterior: Markov q(x[k+1]|x[k],h[k]) rolled out from x[0], or mean-field q(x[k+1]|h[k]).")
    p.add_argument("--substeps", type=int, default=1, help="Latent SDE / SDE-GAN Euler substeps per interval.")
    p.add_argument("--critic-hidden", type=int, default=64)
    p.add_argument("--critic-depth", type=int, default=2)
    p.add_argument("--critic-learning-rate", type=float, default=1e-3)
    p.add_argument("--critic-steps", type=int, default=2, help="SDE-GAN critic updates per generator update.")
    p.add_argument("--gradient-penalty", type=float, default=10.)
    p.add_argument("--sig-samples", type=int, default=8, help="SigKernelSDE: generated paths per window in the score.")
    p.add_argument("--sig-sigma", type=float, default=0., help="SigKernelSDE: RBF bandwidth (0 = sqrt(state dim)).")
    p.add_argument("--sig-stride", type=int, default=1, help="SigKernelSDE: time subsampling of paths in the kernel.")
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--validation-windows", type=int, default=10_000)
    p.add_argument("--validation-draws", type=int, default=64)
    p.add_argument("--eval-chunk", type=int, default=8)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def build(args, data):
    state_dim = data.train["states"].shape[-1]
    observation_dim = data.train["observations"].shape[-1]
    control_dim = data.train["controls"].shape[-1]
    encoder_options = dict(window=args.encoder_window, direct_control=args.direct_control, metadata=data.metadata)
    common = dict(hidden_size=args.hidden_size, hidden_depth=args.hidden_depth, encoder_hidden=args.encoder_hidden,
                  encoder_options=encoder_options, key=jr.PRNGKey(args.seed))
    if args.method == "latent_nsf":
        return LatentNSF(state_dim, observation_dim, control_dim, data.dt, flow_layers=args.flow_layers,
                         beta=args.beta, beta_skip=args.beta_skip, max_gap=args.max_gap, posterior=args.posterior,
                         truncate=args.truncate_bptt, step=args.posterior_step, recon_weight=args.recon_weight,
                         kl_weight=args.kl_weight,
                         # getattr: run_metadata.json of runs trained before these flags existed lacks them.
                         measurement_hidden=getattr(args, "measurement_hidden", 64),
                         measurement_depth=getattr(args, "measurement_depth", 2),
                         measurement_layers=getattr(args, "measurement_layers", 4),
                         **common)
    if args.method == "latent_sde":
        return LatentSDE(state_dim, observation_dim, control_dim, data.dt, substeps=args.substeps, beta=args.beta,
                         **common)
    if args.method == "sigker_sde":
        return SigKernelSDE(state_dim, observation_dim, control_dim, data.dt, substeps=args.substeps,
                            num_samples=args.sig_samples, sigma=args.sig_sigma or None, stride=args.sig_stride,
                            **common)
    return SDEGAN(state_dim, observation_dim, control_dim, data.dt, critic_hidden=args.critic_hidden,
                  critic_depth=args.critic_depth, substeps=args.substeps, gradient_penalty=args.gradient_penalty,
                  **common)


def count(tree):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))))


def estimator(model):
    return model.generator if isinstance(model, SDEGAN) else model


@functools.lru_cache(maxsize=None)
def _batched_inference(draws):
    """One compiled batched sampler per draw count (no jitted closures per validation; see memory note)."""
    @eqx.filter_jit
    def infer(candidate, states, observations, controls, keys):
        return jax.vmap(lambda x, y, u, k: candidate.sample(y, x[0], u, key=k, num_samples=draws))(
            states, observations, controls, keys)
    return infer


def posterior(model, split, windows, seed, draws, chunk):
    infer = _batched_inference(draws)
    windows = min(windows, len(split["states"]))
    outputs = []
    for begin in range(0, windows, chunk):
        index = np.arange(begin, min(begin + chunk, windows))
        keys = jnp.stack([jr.fold_in(jr.PRNGKey(seed), int(i)) for i in index])
        outputs.append(np.asarray(infer(model, *(jnp.asarray(split[k][index])
                                                 for k in ("states", "observations", "controls")), keys)))
    samples = np.concatenate(outputs)
    if not np.isfinite(samples).all():
        raise FloatingPointError("Nonfinite samples")
    return samples, np.asarray(split["states"][:windows, 1:])


def ensemble_crps(samples, truth, data):
    """Mean ensemble CRPS, E|X - y| - E|X - X'| / 2 (sorted-sample form), in score_samples' metric space.

    Logged at validation only (diagnostic for hyperparameter choice); checkpoints are still selected by RMSE."""
    if data.protocol["metric_space"] == "physical":
        mean, std = np.asarray(data.metadata["state_mean"]), np.asarray(data.metadata["state_std"])
        samples, truth = samples * std + mean, truth * std + mean
    m = samples.shape[1]
    ordered = np.sort(samples, axis=1)
    weights = (2. * np.arange(1, m + 1) - m - 1).reshape(1, m, *([1] * (samples.ndim - 2)))
    spread = 2. * np.sum(weights * ordered, axis=1) / m ** 2                      # E|X - X'|
    return float(np.mean(np.mean(np.abs(samples - truth[:, None]), axis=1) - 0.5 * spread))


def latency(model, split, draws, repeats=5):
    """Warmed batch-size-one full-window sampling time (ms), host transfer included."""
    infer = eqx.filter_jit(lambda m, y, x0, u, k: m.sample(y, x0, u, key=k, num_samples=draws))
    y, x0, u = (jnp.asarray(split[k][0]) for k in ("observations", "states", "controls"))
    np.asarray(infer(model, y, x0[0], u, jr.PRNGKey(0)))
    times = []
    for r in range(repeats):
        start = time.perf_counter()
        np.asarray(infer(model, y, x0[0], u, jr.PRNGKey(r + 1)))
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times))


def make_update(args, model, training):
    """Returns (update, optimizer_state); update(model, state, key, step) -> model, state, row-dict."""
    def schedule(lr):
        return optax.warmup_cosine_decay_schedule(0., lr, min(200, args.steps // 10 + 1), args.steps, lr * .1)

    def batch(key):
        ids = jr.randint(key, (args.batch_windows,), 0, training["states"].shape[0])
        return [training[k][ids] for k in ("states", "observations", "controls")]

    if not isinstance(model, SDEGAN):
        optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule(args.learning_rate),
                                                                            weight_decay=1e-5))
        state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        @eqx.filter_jit
        def update(candidate, opt_state, key, step):
            progress = step / args.steps
            (loss, parts), grads = eqx.filter_value_and_grad(
                lambda m: m.loss(*batch(jr.fold_in(key, 0)), jr.fold_in(key, 1), progress), has_aux=True)(candidate)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(candidate, eqx.is_inexact_array))
            return eqx.apply_updates(candidate, updates), opt_state, dict(loss=loss, gradient_norm=optax.global_norm(grads),
                                                                          **parts)
        return update, state

    g_opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(schedule(args.learning_rate), b1=0.5, b2=0.9))
    c_opt = optax.chain(optax.clip_by_global_norm(1.), optax.adam(schedule(args.critic_learning_rate), b1=0.5, b2=0.9))
    state = (g_opt.init(eqx.filter(model.generator, eqx.is_inexact_array)),
             c_opt.init(eqx.filter(model.critic, eqx.is_inexact_array)))

    @eqx.filter_jit
    def update(candidate, opt_state, key, step):
        del step
        g_state, c_state = opt_state
        critic = candidate.critic
        for c in range(args.critic_steps):
            states, observations, controls = batch(jr.fold_in(key, 10 + c))
            fake = candidate.fake(states, observations, controls, jr.fold_in(key, 20 + c))
            (c_loss, c_parts), c_grads = eqx.filter_value_and_grad(
                lambda cr: SDEGAN.critic_loss(cr, candidate.gradient_penalty, states, fake, observations, controls,
                                              jr.fold_in(key, 30 + c)), has_aux=True)(critic)
            updates, c_state = c_opt.update(c_grads, c_state, eqx.filter(critic, eqx.is_inexact_array))
            critic = eqx.apply_updates(critic, updates)
        candidate = eqx.tree_at(lambda m: m.critic, candidate, critic)
        states, observations, controls = batch(jr.fold_in(key, 1))
        g_loss, g_grads = eqx.filter_value_and_grad(
            lambda g: candidate.generator_loss(g, states, observations, controls, jr.fold_in(key, 2)))(candidate.generator)
        updates, g_state = g_opt.update(g_grads, g_state, eqx.filter(candidate.generator, eqx.is_inexact_array))
        candidate = eqx.tree_at(lambda m: m.generator, candidate, eqx.apply_updates(candidate.generator, updates))
        return candidate, (g_state, c_state), dict(loss=g_loss, gradient_norm=optax.global_norm(g_grads),
                                                   critic_loss=c_loss, **c_parts)
    return update, state


def main(argv=None):
    args = parse_args(argv)
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_cache(args.data_cache)
    if data.protocol.get("dataset") != args.dataset:
        raise ValueError("Data cache does not match --dataset")
    draws = data.protocol["posterior_samples"]
    if args.smoke:
        args.steps, args.eval_every, args.validation_windows, args.validation_draws, draws = 6, 3, 4, 8, 16
    model = build(args, data)
    training = {k: jnp.asarray(data.train[k]) for k in ("states", "observations", "controls")}
    update, state = make_update(args, model, training)

    def validate(candidate):
        samples, truth = posterior(estimator(candidate), data.validation, args.validation_windows, 77001,
                                   args.validation_draws, args.eval_chunk)
        scores = score_samples(samples, truth, data)
        return scores["rmse"], scores["coverage_90"], ensemble_crps(samples, truth, data)

    report = dict(status="running", method=args.method, dataset=args.dataset, seed=args.seed,
                  variant=VARIANTS[args.method], inference_access=ACCESS,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  posterior_samples=draws, parameter_count=count(estimator(model)),
                  training_parameter_count=count(model), devices=[str(d) for d in jax.devices()])
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    best, best_step, best_coverage = np.inf, 0, None
    with (args.output / "history.jsonl").open("w") as history:
        for step in range(args.steps):
            model, state, parts = update(model, state, jr.fold_in(jr.PRNGKey(args.seed + 500), step),
                                         jnp.asarray(step, jnp.float32))
            row = dict(step=step + 1, **{k: float(v) for k, v in parts.items()})
            if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
                raise FloatingPointError(f"Nonfinite loss/gradient at step {step + 1}")
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                rmse, coverage, crps = validate(model)
                row.update(validation_rmse=rmse, validation_coverage_90=coverage, validation_crps=crps,
                           elapsed_seconds=time.perf_counter() - started)
                if rmse < best:
                    best, best_step, best_coverage = rmse, step + 1, coverage
                    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
                row["best_validation_rmse"] = best
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + "\n")
            history.flush()
    report.update(training_seconds=time.perf_counter() - started, best_step=best_step,
                  best_validation_rmse=best, best_validation_coverage_90=best_coverage)

    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    windows = data.protocol["eval_windows"] if not args.smoke else 2
    samples, truth = posterior(estimator(model), data.test, windows, 99001, draws, args.eval_chunk)
    metrics = score_samples(samples, truth, data)
    proper = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                  metric_space=data.protocol["metric_space"], state_mean=data.metadata["state_mean"],
                                  state_std=data.metadata["state_std"], energy_method="exact")
    metrics.update(proper["metrics"])
    metrics.update(eval_windows=len(truth), posterior_samples=samples.shape[1], inference_access=ACCESS,
                   latency_ms_per_trajectory=latency(estimator(model), data.test, draws))
    np.savez_compressed(args.output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report["test"] = {k: metrics[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width",
                                              "latency_ms_per_trajectory") if k in metrics}
    report["status"] = "complete"
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(dict(
        method=args.method, seed=args.seed, best_step=best_step, best_validation_rmse=best,
        best_validation_coverage_90=best_coverage, test=report["test"]), indent=2) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.output), test=report["test"])), flush=True)


if __name__ == "__main__":
    main()
