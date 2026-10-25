"""NormFlowSDE filtering: full-horizon proper-score fine-tuning (pendulum first).

``--dataset`` extends the recipe to Lorenz96, ALOHA and REASSEMBLE, scoring in
each protocol's metric space. ``--allow-access-switch`` warm-starts from a
comparison-runner prediction checkpoint: the same network then receives the
current (unshifted) measurements as filtering inputs, so the reported access
changes from prediction to filtering. The pendulum description follows.

Warm-starts the same-seed checkpoint of the observation-access filtering run
(train_normflowsde_filter.py, segment-11 recipe) and replaces its objective,
which trained from true segment anchors with a squared-error term and an
entropy-minimizing term, by a proper score of the full 60-step filtering
ensemble generated exactly as at inference: every draw starts at the known x0,
receives actual noisy measurements through x[k+1] and is never reset to the
truth. The score is evaluated per time step in physical units, so its minimizer
is the filtering marginal of each state.

* ``--objective energy``: per-time multivariate energy score of the 2-D state;
* ``--objective crps``: per-time, per-coordinate CRPS (targets each marginal).

Batches hold real trajectories (default 16, versus 4). Checkpoints are selected
on all 48 validation windows with the filtering campaign's evaluator, seeds and
128 posterior draws; the step-0 warm start is scored and remains a candidate.
The test split is scored once at the end with the campaign protocol (seed
99001, 48 windows, 128 draws). Architecture and inference access are unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.applications.state_estimation.flow_sde import (
    LearnedScaleDiffusionNet, RealNVPDiffusionNet, RealNVPFlowSDE,
)
from .cache import load_cache
from .evaluation_metrics import metrics_from_samples
from .train_normflowsde_filter import ACCESS, filtering_inputs, make_evaluator


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--dataset", default="pendulum", choices=("pendulum", "lorenz96", "aloha", "reassemble"))
    p.add_argument("--allow-access-switch", action="store_true",
                   help="Warm-start from a preceding-observation prediction checkpoint; inputs become filtering inputs.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-cache", type=Path, required=True)
    p.add_argument("--init-run", type=Path, required=True,
                   help="Filtering run directory with run_metadata.json and model.eqx.")
    p.add_argument("--objective", choices=("energy", "crps"), default="energy")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-trajectories", type=int, default=16)
    p.add_argument("--draws", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--mean-mse-weight", type=float, default=0.,
                   help="Optional ensemble-mean squared error (physical units).")
    p.add_argument("--energy-beta", type=float, default=1.,
                   help="Opt-in exponent of the stage-2 energy score (strictly proper for 0 < beta < 2; default 1).")
    p.add_argument("--energy-spread-weight", type=float, default=.5,
                   help="Opt-in (2026-10-22): weight of the pairwise spread term of the stage-2 energy score. "
                        "0.5 is the proper energy score (default, original code path); larger values reward dispersion.")
    p.add_argument("--learned-scale", action="store_true",
                   help="Wrap the warm-started noise flow in a zero-initialized per-coordinate learned scale "
                        "(step-0 model unchanged); lifts the +-alpha coupling bound on the noise magnitude.")
    p.add_argument("--realnvp-alpha", type=float,
                   help="Opt-in (2026-10-21): override the RealNVP coupling log-scale bound of the warm-started flow. "
                        "No parameter change, but the step-0 model differs (log-scale = alpha * tanh(raw)).")
    p.add_argument("--validation-every", type=int, default=250)
    p.add_argument("--select-metric", choices=("rmse", "crps"), default="rmse",
                   help="Opt-in (2026-10-21): validation metric that selects the stage-2 checkpoint "
                        "(default rmse = original behaviour; crps is the proper score matching the energy objective).")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Tiny CPU check; not an accuracy run.")
    args = p.parse_args(argv)
    if args.draws < 2 or min(args.steps, args.batch_trajectories, args.validation_every) < 1:
        p.error("draws >= 2 and positive steps/batch/validation cadence required")
    if args.realnvp_alpha is not None and args.realnvp_alpha <= 0:
        p.error("realnvp-alpha must be positive")
    return args


def safe_norm(delta):
    squared = jnp.sum(delta ** 2, axis=-1)
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.)), 0.)


def powered_distance(delta, beta):
    """||delta||^beta with a smooth floor, so that the gradient stays finite for beta < 1."""
    return (jnp.sum(delta ** 2, axis=-1) + 1e-12) ** (beta / 2)


def per_time_energy(draws, truth, beta=1., spread_weight=.5):
    """Unbiased energy score of each time's state vector, averaged over time.

    ``beta`` (opt-in, 2026-10-21) is the exponent of the energy score, strictly proper for 0 < beta < 2;
    beta = 1 is the method's default and uses the original code path unchanged.
    ``spread_weight`` (opt-in, 2026-10-22) multiplies the pairwise spread term; 0.5 is the proper energy
    score and the default. Larger values are no longer strictly proper and reward dispersion.
    """
    count = draws.shape[0]
    if beta != 1.:
        accuracy = jnp.mean(powered_distance(draws - truth[None], beta), axis=0)
        pair = powered_distance(draws[:, None] - draws[None, :], beta)
        pair = jnp.where(jnp.eye(count, dtype=bool)[:, :, None], 0., pair)  # exclude self pairs
        spread = jnp.sum(pair, axis=(0, 1)) / (count * (count - 1))
        return jnp.mean(accuracy - spread_weight * spread)
    accuracy = jnp.mean(safe_norm(draws - truth[None]), axis=0)
    spread = jnp.sum(safe_norm(draws[:, None] - draws[None, :]), axis=(0, 1)) / (count * (count - 1))
    return jnp.mean(accuracy - spread_weight * spread)


def per_coordinate_crps(draws, truth):
    """Unbiased CRPS of every time/coordinate marginal, averaged."""
    count = draws.shape[0]
    accuracy = jnp.mean(jnp.abs(draws - truth[None]), axis=0)
    spread = jnp.sum(jnp.abs(draws[:, None] - draws[None, :]), axis=(0, 1)) / (count * (count - 1))
    return jnp.mean(accuracy - .5 * spread)


def loss_fn(model, states, observations, controls, key, state_std, args, solver="euler"):
    def one(x, y, u, k):
        draws = model.sample_posterior(filtering_inputs(y, u), x[0], k,
                                       n_particles=args.draws, solver_name=solver)
        draws, truth = draws * state_std, x[1:] * state_std
        score = (per_time_energy(draws, truth, args.energy_beta, args.energy_spread_weight) if args.objective == "energy"
                 else per_coordinate_crps(draws, truth))
        return score, jnp.mean((draws.mean(axis=0) - truth) ** 2)

    score, mse = jax.vmap(one)(states, observations, controls, jr.split(key, states.shape[0]))
    score, mse = jnp.mean(score), jnp.mean(mse)
    return score + args.mean_mse_weight * mse, dict(score=score, mean_mse=mse)


def main(argv=None):
    args = parse_args(argv)
    if not args.allow_cpu and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("GPU required; --allow-cpu is only for explicit smoke validation")
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    reference = json.loads((args.init_run / "run_metadata.json").read_text())
    access_switch = reference.get("inference_access") != ACCESS
    if (reference.get("dataset"), reference.get("method"), reference.get("training_seed")) != (
            args.dataset, "normflowsde", args.seed) or (access_switch and not (
            args.allow_access_switch and reference.get("inference_access", "").startswith("prediction:"))):
        raise ValueError("init-run must be the same-seed NormFlowSDE filtering run "
                         "(or a prediction run with --allow-access-switch)")
    data = load_cache(args.data_cache)
    reference_protocol = reference["data"]["protocol"]
    if (data.protocol.get("dataset") != args.dataset or data.protocol["seed"] != reference_protocol["seed"]
            or data.protocol.get("variant") != reference_protocol.get("variant")):
        raise ValueError("Data cache does not match the reference dataset protocol")
    model_config = dict(reference["model_config"])
    if args.realnvp_alpha is not None:
        if model_config.get("flow_type", "realnvp") != "realnvp":
            raise ValueError("--realnvp-alpha requires a RealNVP noise flow in the init run")
        model_config["realnvp_alpha"] = args.realnvp_alpha
    model = RealNVPFlowSDE(**model_config, key=jr.PRNGKey(args.seed))
    model = eqx.tree_deserialise_leaves(args.init_run / "model.eqx", model)
    count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))
    if count != reference["parameter_count"]:
        raise ValueError("Architecture differs from the reference checkpoint")
    learned_scale_source = None
    if args.learned_scale and isinstance(model.diffusion, LearnedScaleDiffusionNet):
        learned_scale_source = "init run already has a learned noise scale (kept, not re-wrapped)"
    elif args.learned_scale:
        if not isinstance(model.diffusion, RealNVPDiffusionNet):
            raise ValueError("--learned-scale requires a RealNVP noise flow in the init run")
        learned_scale_source = "zero-initialized wrapper added at warm start"
        cfg = reference["model_config"]
        wrapped = LearnedScaleDiffusionNet(model.diffusion, cfg["d_c"] + (cfg["d_u"] if cfg.get("direct_control") else 0),
                                           cfg["d_hidden_diffusion"], model.diffusion.layers[0].time_emb.dim,
                                           jr.fold_in(jr.PRNGKey(args.seed), 3), use_time=cfg.get("use_time", True))
        model = eqx.tree_at(lambda m: m.diffusion, model, wrapped)
        model_config["learned_scale"] = True
        count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array)))

    # Filtering runs store their recipe; comparison-runner checkpoints use run.py's fixed optimizer settings.
    recipe = reference.get("recipe") or dict(posterior_samples=data.protocol["posterior_samples"],
                                             eval_windows=data.protocol["eval_windows"], grad_norm=1.,
                                             weight_decay=1e-5)
    # Keep the init run's transition kernel (Heun for the Lorenz-96 recipe); comparison-runner runs are Euler.
    solver = recipe.get("solver", "euler")
    space = data.protocol["metric_space"]
    validation_windows, particles = len(data.validation["states"]), recipe["posterior_samples"]
    test_windows = recipe["eval_windows"]
    if args.smoke:
        validation_windows, particles, test_windows = 2, 4, 2
    state_std = (jnp.asarray(data.metadata["state_std"]) if space == "physical"
                 else jnp.ones(data.train["states"].shape[-1]))
    training = {name: jnp.asarray(data.train[name]) for name in ("states", "observations", "controls")}
    schedule = optax.warmup_cosine_decay_schedule(0., args.learning_rate, min(200, args.steps // 10 + 1),
                                                  args.steps, args.learning_rate * .1)
    optimizer = optax.chain(optax.clip_by_global_norm(recipe["grad_norm"]),
                            optax.adamw(schedule, weight_decay=recipe["weight_decay"]))
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(candidate, state, x, y, u, key):
        (loss, parts), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            candidate, x, y, u, key, state_std, args, solver)
        updates, state = optimizer.update(grads, state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), state, loss, optax.global_norm(grads), parts

    evaluate = make_evaluator(data, solver)

    def validate(candidate):
        scores, samples, truth, _ = evaluate(candidate, data.validation, validation_windows, particles, 77001)
        proper = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                      metric_space=space, state_mean=data.metadata["state_mean"],
                                      state_std=data.metadata["state_std"])["metrics"]
        return dict(rmse=scores["rmse"], crps=proper["crps"], coverage_90=proper["coverage_90"])

    report = dict(status="running", variant=f"{args.dataset}-normflowsde-filter-fullhorizon-{args.objective}-v1",
                  dataset=args.dataset, method="normflowsde", training_seed=args.seed,
                  inference_access=ACCESS, init_run=str(args.init_run), warm_start=True,
                  warm_start_inference_access=reference.get("inference_access"),
                  observation_access_switched=access_switch, metric_space=space, solver=solver,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  model_config=model_config, parameter_count=int(count),
                  devices=[str(d) for d in jax.devices()], post_calibration=False,
                  known_initial_state=True, intermediate_truth_resets_at_evaluation=False,
                  learned_scale=learned_scale_source)
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    initial = validate(model)
    best = initial[args.select_metric]
    report.update(initial_validation=initial, best_step=0, best_validation=initial)
    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
    print(json.dumps(dict(step=0, **{f"validation_{k}": v for k, v in initial.items()})), flush=True)
    with (args.output / "history.jsonl").open("w") as history:
        history.write(json.dumps(dict(step=0, **{f"validation_{k}": v for k, v in initial.items()})) + "\n")
        for step in range(args.steps):
            batch_key, loss_key = jr.split(jr.fold_in(jr.PRNGKey(args.seed + 700), step))
            ids = jr.randint(batch_key, (args.batch_trajectories,), 0, len(training["states"]))
            model, opt_state, loss, gradient_norm, parts = update(
                model, opt_state, *(training[n][ids] for n in ("states", "observations", "controls")), loss_key)
            row = dict(step=step + 1, loss=float(loss), gradient_norm=float(gradient_norm),
                       **{k: float(v) for k, v in parts.items()})
            if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
                raise FloatingPointError(f"Nonfinite loss/gradient at step {step + 1}")
            if (step + 1) % args.validation_every == 0 or step + 1 == args.steps:
                scores = validate(model)
                row.update({f"validation_{k}": v for k, v in scores.items()})
                if scores[args.select_metric] < best:
                    best = scores[args.select_metric]
                    report.update(best_step=step + 1, best_validation=scores)
                    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
                row.update({f"best_validation_{args.select_metric}": best}, elapsed_seconds=time.perf_counter() - started)
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + "\n")
            history.flush()
    report["training_seconds"] = time.perf_counter() - started

    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    _, samples, truth, elapsed = evaluate(model, data.test, test_windows, particles, 99001)
    metrics = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                   metric_space=space, state_mean=data.metadata["state_mean"],
                                   state_std=data.metadata["state_std"])
    np.savez_compressed(args.output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report.update(status="complete", test=metrics["metrics"])
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = dict(seed=args.seed, objective=args.objective, best_step=report["best_step"],
                   initial_validation=initial, best_validation=report["best_validation"],
                   test={k: metrics["metrics"][k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width")})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(dict(status="complete", **summary)), flush=True)


if __name__ == "__main__":
    main()
