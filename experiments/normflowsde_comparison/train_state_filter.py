"""Train and evaluate NSF-PF / FlowSDE-PF (learned-dynamics particle filters).

Dynamics (one-step NSF NLL or FlowSDE bridge flow matching, no observation
input) and the measurement flow q(y | x) are trained jointly from random
training windows. Every --eval-every updates the particle filter runs on all
validation windows; the checkpoint with the lowest validation RMSE (metric
space of the cache) is kept. The test split is filtered once, after training,
with the protocol's particle count and posterior draw count; outputs mirror
the comparison runs (posterior.npz, metrics.json, run_metadata.json,
summary.json, model.eqx) so build_results_paper can evaluate and plot them.
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

from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.evaluation_metrics import metrics_from_samples
from experiments.normflowsde_comparison.learned_dynamics_pf import (FlowSDEDynamics, LearnedDynamicsPF, NormFlowSDEDynamics,
                                                                        NSFDynamics)
from experiments.normflowsde_comparison.run import score_samples

ACCESS = ("filtering: x[k+1] uses observations through x[k+1] via a bootstrap particle filter "
          "(learned dynamics prediction, learned measurement-density update); no future observations")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=("nsf", "flowsde", "normflowsde"))
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-cache", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=4400)
    p.add_argument("--batch-windows", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--hidden-size", type=int, required=True, help="Dynamics width (match the comparison capacity).")
    p.add_argument("--hidden-depth", type=int, default=2)
    p.add_argument("--flow-layers", type=int, default=6, help="NSF / NormFlowSDE coupling layers.")
    p.add_argument("--diffusion", type=float, default=0.01, help="FlowSDE fixed diffusion D.")
    p.add_argument("--flow-substeps", type=int, default=4)
    p.add_argument("--epr-weight", type=float, default=0.)
    p.add_argument("--measurement-hidden", type=int, default=64)
    p.add_argument("--measurement-depth", type=int, default=2)
    p.add_argument("--measurement-layers", type=int, default=4)
    p.add_argument("--particles", type=int, help="Default: the protocol's num_particles.")
    p.add_argument("--eval-every", type=int, default=400)
    p.add_argument("--validation-windows", type=int, default=10_000)
    p.add_argument("--eval-chunk", type=int, default=8)
    p.add_argument("--selection-jitters",
                   help="Opt-in (2026-10-20): comma-separated process-noise jitter values scored at every "
                        "validation; the checkpoint is then selected jointly over (step, jitter) by validation "
                        "RMSE instead of at jitter 0. The test posterior uses the selected jitter. Default: "
                        "unset (jitter-0 selection, unchanged).")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args(argv)


def build(args, data):
    state_dim = data.train["states"].shape[-1]
    observation_dim = data.train["observations"].shape[-1]
    control_dim = data.train["controls"].shape[-1]
    dynamics_key, measurement_key = jr.split(jr.PRNGKey(args.seed))
    if args.method == "nsf":
        dynamics = NSFDynamics(state_dim, control_dim, data.dt, hidden_size=args.hidden_size,
                               hidden_depth=args.hidden_depth, flow_layers=args.flow_layers, key=dynamics_key)
    elif args.method == "normflowsde":
        dynamics = NormFlowSDEDynamics(state_dim, control_dim, data.dt, hidden_size=args.hidden_size,
                                       hidden_depth=args.hidden_depth, flow_layers=args.flow_layers, key=dynamics_key)
    else:
        dynamics = FlowSDEDynamics(state_dim, control_dim, data.dt, hidden_size=args.hidden_size,
                                   hidden_depth=args.hidden_depth, diffusion=args.diffusion,
                                   substeps=args.flow_substeps, epr_weight=args.epr_weight, key=dynamics_key)
    return LearnedDynamicsPF(dynamics, observation_dim, state_dim, hidden_size=args.measurement_hidden,
                             hidden_depth=args.measurement_depth, flow_layers=args.measurement_layers,
                             key=measurement_key)


def count(tree):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))))


def posterior(model, split, windows, seed, particles, draws, chunk, jitter=0.):
    @eqx.filter_jit
    def infer(candidate, states, observations, controls, keys):
        return jax.vmap(lambda x, y, u, k: candidate.sample(y, x[0], u, key=k, num_particles=particles,
                                                            num_samples=draws, jitter=jitter))(states, observations, controls, keys)

    windows = min(windows, len(split["states"]))
    outputs = []
    for begin in range(0, windows, chunk):
        index = np.arange(begin, min(begin + chunk, windows))
        keys = jnp.stack([jr.fold_in(jr.PRNGKey(seed), int(i)) for i in index])
        outputs.append(np.asarray(infer(model, *(jnp.asarray(split[k][index])
                                                 for k in ("states", "observations", "controls")), keys)))
    samples = np.concatenate(outputs)
    if not np.isfinite(samples).all():
        raise FloatingPointError("Nonfinite filtering samples")
    return samples, np.asarray(split["states"][:windows, 1:])


def latency(model, split, particles, draws, repeats=5, jitter=0.):
    """Warmed batch-size-one full-window filtering + draw time (ms), host transfer included."""
    infer = eqx.filter_jit(lambda m, y, x0, u, k: m.sample(y, x0, u, key=k, num_particles=particles,
                                                           num_samples=draws, jitter=jitter))
    y, x0, u = (jnp.asarray(split[k][0]) for k in ("observations", "states", "controls"))
    np.asarray(infer(model, y, x0[0], u, jr.PRNGKey(0)))
    times = []
    for r in range(repeats):
        start = time.perf_counter()
        np.asarray(infer(model, y, x0[0], u, jr.PRNGKey(r + 1)))
        times.append((time.perf_counter() - start) * 1e3)
    return float(np.median(times))


def main(argv=None):
    args = parse_args(argv)
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_cache(args.data_cache)
    if data.protocol.get("dataset") != args.dataset:
        raise ValueError("Data cache does not match --dataset")
    particles = args.particles or data.protocol["num_particles"]
    draws = data.protocol["posterior_samples"]
    if args.smoke:
        args.steps, args.eval_every, args.validation_windows, particles, draws = 6, 3, 4, 32, 16
    model = build(args, data)
    training = {k: jnp.asarray(data.train[k]) for k in ("states", "observations", "controls")}

    schedule = optax.warmup_cosine_decay_schedule(0., args.learning_rate, min(200, args.steps // 10 + 1),
                                                  args.steps, args.learning_rate * .1)
    optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-5))
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def update(candidate, opt_state, key):
        ids = jr.randint(jr.fold_in(key, 0), (args.batch_windows,), 0, training["states"].shape[0])
        batch = [training[k][ids] for k in ("states", "observations", "controls")]
        (loss, parts), grads = eqx.filter_value_and_grad(
            lambda m: m.loss(*batch, jr.fold_in(key, 1)), has_aux=True)(candidate)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), opt_state, loss, optax.global_norm(grads), parts

    jitters = [float(v) for v in args.selection_jitters.split(",")] if args.selection_jitters else [0.]

    def validate(candidate):
        rows = []
        for value in jitters:
            samples, truth = posterior(candidate, data.validation, args.validation_windows, 77001, particles,
                                       draws, args.eval_chunk, value)
            scores = score_samples(samples, truth, data)
            rows.append((scores["rmse"], scores["coverage_90"], value))
        rmse, coverage, value = min(rows, key=lambda row: row[0])
        return rmse, coverage, value, [dict(jitter=v, rmse=r, coverage_90=c) for r, c, v in rows]

    report = dict(status="running", method=args.method, dataset=args.dataset, seed=args.seed,
                  variant="learned-dynamics bootstrap particle filter", inference_access=ACCESS,
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  particles=particles, posterior_samples=draws,
                  parameter_count=count(model), dynamics_parameters=count(model.dynamics),
                  measurement_parameters=count(model.measurement), devices=[str(d) for d in jax.devices()])
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    started = time.perf_counter()
    best, best_step, best_coverage, best_jitter = np.inf, 0, None, 0.
    with (args.output / "history.jsonl").open("w") as history:
        for step in range(args.steps):
            model, state, loss, gradient_norm, parts = update(model, state, jr.fold_in(jr.PRNGKey(args.seed + 500), step))
            row = dict(step=step + 1, loss=float(loss), gradient_norm=float(gradient_norm),
                       **{k: float(v) for k, v in parts.items()})
            if not np.isfinite([row["loss"], row["gradient_norm"]]).all():
                raise FloatingPointError(f"Nonfinite loss/gradient at step {step + 1}")
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                rmse, coverage, jitter, by_jitter = validate(model)
                row.update(validation_rmse=rmse, validation_coverage_90=coverage,
                           elapsed_seconds=time.perf_counter() - started)
                if args.selection_jitters:
                    row.update(validation_jitter=jitter, validation_by_jitter=by_jitter)
                if rmse < best:
                    best, best_step, best_coverage, best_jitter = rmse, step + 1, coverage, jitter
                    eqx.tree_serialise_leaves(args.output / "model.eqx", model)
                row["best_validation_rmse"] = best
                print(json.dumps(row), flush=True)
            history.write(json.dumps(row) + "\n")
            history.flush()
    report.update(training_seconds=time.perf_counter() - started, best_step=best_step,
                  best_validation_rmse=best, best_validation_coverage_90=best_coverage)
    if args.selection_jitters:
        report.update(jitter=best_jitter, variant=report["variant"] + " + joint (checkpoint, process-noise "
                      "inflation) validation selection")

    model = eqx.tree_deserialise_leaves(args.output / "model.eqx", model)
    windows = data.protocol["eval_windows"] if not args.smoke else 2
    samples, truth = posterior(model, data.test, windows, 99001, particles, draws, args.eval_chunk, best_jitter)
    metrics = score_samples(samples, truth, data)
    proper = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                  metric_space=data.protocol["metric_space"], state_mean=data.metadata["state_mean"],
                                  state_std=data.metadata["state_std"], energy_method="exact")
    metrics.update(proper["metrics"])
    metrics.update(eval_windows=len(truth), posterior_samples=samples.shape[1], inference_access=ACCESS,
                   latency_ms_per_trajectory=latency(model, data.test, particles, draws, jitter=best_jitter))
    if args.selection_jitters:
        metrics["jitter"] = best_jitter
    np.savez_compressed(args.output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report["test"] = {k: metrics[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width",
                                              "latency_ms_per_trajectory") if k in metrics}
    report["status"] = "complete"
    (args.output / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(dict(
        method=args.method, seed=args.seed, best_step=best_step, best_validation_rmse=best,
        best_validation_coverage_90=best_coverage, test=report["test"],
        **(dict(jitter=best_jitter) if args.selection_jitters else {})), indent=2) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.output), test=report["test"])), flush=True)


if __name__ == "__main__":
    main()
