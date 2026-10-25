"""Explicit historical-style Lorenz variant; separate from the manuscript runs.

Heun is used for every generated path. ``transition`` is the historical
Euler reference-transition auxiliary NLL, not an exact Heun likelihood.
Preparation is the default; optimization requires --train.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
from . import run as common
from .lorenz_heun_objective import loss_components_single
from .tracking import JournalTracker


VARIANT = "legacy-causal-heun-v1"
EXTRA_SOURCES = (
    "experiments/normflowsde_comparison/lorenz_heun_objective.py",
    "experiments/normflowsde_comparison/train_lorenz_heun.py",
    "experiments/normflowsde_comparison/lorenz_heun_array.sh",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data-cache", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train", action="store_true")
    p.add_argument("--allow-cpu", action="store_true", help="Only for explicit small validation runs.")
    p.add_argument("--expected-manifest", type=Path)
    p.add_argument("--max-epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64, help="Actual complete trajectories per update.")
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-norm", type=float, default=1.)
    p.add_argument("--patience", type=int, default=40, help="Epochs since the best validation checkpoint.")
    p.add_argument("--validation-every-epochs", type=int, default=5)
    p.add_argument("--validation-particles", type=int, default=32)
    p.add_argument("--validation-windows", type=int, default=200)
    p.add_argument("--posterior-samples", type=int, default=200)
    p.add_argument("--eval-windows", type=int, default=100)
    p.add_argument("--segment-length", type=int, default=5, help="State points including anchor.")
    p.add_argument("--observation-window", type=int, default=10)
    p.add_argument("--shooting-noise-std", type=float, default=.025)
    p.add_argument("--state-sigma", type=float, default=math.sqrt(.5))
    p.add_argument("--realnvp-alpha", type=float, default=1.,
                   help="Positive bound on each coupling log-scale: alpha * tanh(raw_scale).")
    p.add_argument("--transition-weight", type=float, default=.25)
    p.add_argument("--continuity-weight", type=float, default=.1)
    p.add_argument("--rollout-weight", type=float, default=1.)
    p.add_argument("--rollout-every", type=int, default=5)
    p.add_argument("--rollout-draws", type=int, default=8)
    a = p.parse_args(argv)
    for name in ("max_epochs", "batch_size", "validation_every_epochs", "validation_particles",
                 "validation_windows", "posterior_samples", "eval_windows", "observation_window", "rollout_every"):
        if getattr(a, name) < 1:
            p.error(f"{name} must be positive")
    if a.segment_length < 2 or a.rollout_draws < 2:
        p.error("segment-length and rollout-draws must be at least two")
    if a.state_sigma <= 0 or a.learning_rate <= 0 or a.grad_norm <= 0:
        p.error("state-sigma, learning-rate and grad-norm must be positive")
    if not math.isfinite(a.realnvp_alpha) or a.realnvp_alpha <= 0:
        p.error("realnvp-alpha must be finite and positive")
    for name in ("warmup_steps", "weight_decay", "patience", "shooting_noise_std",
                 "transition_weight", "continuity_weight", "rollout_weight"):
        if getattr(a, name) < 0:
            p.error(f"{name} must be nonnegative")
    return a


def rollout_active(args, completed_step):
    return args.rollout_weight != 0 and completed_step % args.rollout_every == 0


def scientific_arguments(args):
    return {name: value for name, value in vars(args).items()
            if name not in {"output", "data_cache", "train", "expected_manifest"}}


def prepare(args):
    base = common.parse_args([
        "--dataset", "lorenz96", "--method", "normflowsde", "--noise-model", "gmm",
        "--output", str(args.output), "--data-cache", str(args.data_cache),
        "--seed", str(args.seed), "--hidden-size", "256", "--hidden-depth", "3",
        "--flow-layers", "2", "--context-dim", "128", "--diffusion-hidden-size", "64",
        "--segment-length", str(args.segment_length), "--observation-window", str(args.observation_window),
        "--batch-size", str(args.batch_size), "--learning-rate", str(args.learning_rate),
        "--posterior-samples", str(args.posterior_samples), "--eval-windows", str(args.eval_windows),
    ])
    data = common.load_data(base)
    n_batches = math.ceil(len(data.train["states"]) / args.batch_size)
    data.protocol["train_steps"] = args.max_epochs * n_batches
    report = common.prepare_report(data, base)
    config = common.model_config(data, base)
    # Reproduce reference A's 7-channel encoder (six measurements + zero control).
    config["direct_control"] = False
    config["realnvp_alpha"] = args.realnvp_alpha
    model = RealNVPFlowSDE(**config, physics_fn=None, key=jax.random.PRNGKey(args.seed))
    root = Path(__file__).resolve().parents[2]
    report["source_sha256"].update({name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                    for name in EXTRA_SOURCES})
    report.update(variant=VARIANT, model_config=config,
        arguments={name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        parameter_count=sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))),
        full_trajectory_training_batch=args.batch_size, normflowsde_rollout_every=args.rollout_every,
        solver="heun", known_physics=False,
        objective_definitions={
            "state_nll": "mean fixed-variance Gaussian reconstruction; sigma=sqrt(.5) matches historical MSE gradient",
            "transition": "Euler reference-transition NLL auxiliary; NOT exact Heun transition likelihood",
            "continuity": "mean squared endpoint error at internal segment boundaries",
            "rollout": "historical timewise energy, all ensemble pairs, full trajectory, no intermediate truth resets"},
        optimizer=dict(name="AdamW", actual_batch_trajectories=args.batch_size,
                       batches_per_epoch=n_batches, max_epochs=args.max_epochs,
                       warmup_steps=min(args.warmup_steps, max(0, args.max_epochs*n_batches-1)),
                       peak_learning_rate=args.learning_rate, decay="cosine", weight_decay=args.weight_decay,
                       global_gradient_clip=args.grad_norm, sampling="epoch shuffle without replacement"),
        validation=dict(windows=min(args.validation_windows,len(data.validation["states"])),
                        particles=args.validation_particles, every_epochs=args.validation_every_epochs,
                        selection="physical mean trajectory RMSE", patience_epochs=args.patience),
        scientific_arguments=scientific_arguments(args))
    if args.expected_manifest:
        expected = json.loads(args.expected_manifest.read_text())
        actual = {name: report[name] for name in ("source_sha256", "model_config", "scientific_arguments", "data")}
        actual["packages"] = report["environment"]["packages"]
        if actual != expected:
            raise ValueError("Frozen source, data, model or scientific arguments changed after preparation")
    return model, data, report


def train(model, data, args, report, tracker):
    if not args.allow_cpu and not any(d.platform == "gpu" for d in jax.devices()):
        raise RuntimeError("No GPU visible; use --allow-cpu only for an explicit small validation run")
    total_steps = report["data"]["protocol"]["train_steps"]
    warmup = report["optimizer"]["warmup_steps"]
    if warmup:
        schedule = optax.join_schedules([
            optax.linear_schedule(0., args.learning_rate, warmup),
            optax.cosine_decay_schedule(args.learning_rate, total_steps-warmup)], [warmup])
    else:
        schedule = optax.cosine_decay_schedule(args.learning_rate, total_steps)
    optimizer = optax.chain(optax.clip_by_global_norm(args.grad_norm),
                            optax.adamw(schedule, weight_decay=args.weight_decay))
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    training = {name: jnp.asarray(data.train[name]) for name in ("states", "observations", "controls")}

    @eqx.filter_jit
    def update(candidate, opt_state, x, observations, controls, key, include_rollout):
        preceding = common.preceding_observations(x, observations, data.metadata)
        y = jnp.concatenate((preceding, controls), axis=-1)
        def objective(m):
            parts = jax.vmap(lambda xi, yi, ki: loss_components_single(
                m, xi, yi, ki, segment_length=args.segment_length, state_sigma=args.state_sigma,
                shooting_noise_std=args.shooting_noise_std, rollout_active=include_rollout,
                rollout_draws=args.rollout_draws))(x, y, jax.random.split(key, x.shape[0]))
            parts = {name: jnp.mean(value) for name, value in parts.items()}
            return (parts["state_nll"] + args.transition_weight*parts["transition"]
                    + args.continuity_weight*parts["continuity"] + args.rollout_weight*parts["rollout"]), parts
        (loss, parts), gradients = eqx.filter_value_and_grad(objective, has_aux=True)(candidate)
        updates, opt_state = optimizer.update(gradients, opt_state, eqx.filter(candidate, eqx.is_inexact_array))
        return eqx.apply_updates(candidate, updates), opt_state, loss, optax.global_norm(gradients), parts

    @eqx.filter_jit
    def infer(candidate, x, observations, controls, key, particles):
        preceding = common.preceding_observations(x[None], observations[None], data.metadata)[0]
        y = jnp.concatenate((preceding, controls), axis=-1)
        return candidate.sample_posterior(y, x[0], key, n_particles=particles, solver_name="heun")

    def evaluate(candidate, split, count, particles, evaluation_seed):
        outputs = []
        inference_started = time.perf_counter()
        for i in range(min(count, len(split["states"]))):
            inputs = [jnp.asarray(split[name][i]) for name in ("states", "observations", "controls")]
            outputs.append(np.asarray(infer(candidate, *inputs,
                jax.random.fold_in(jax.random.PRNGKey(evaluation_seed), i), particles)))
        inference_seconds = time.perf_counter() - inference_started
        samples = np.stack(outputs)
        if not np.isfinite(samples).all():
            raise FloatingPointError("Nonfinite Heun posterior samples")
        truth = np.asarray(split["states"][:len(outputs), 1:])
        return common.score_samples(samples, truth, data), samples, truth, inference_seconds

    metadata = args.output/"run_metadata.json"
    def save_report():
        temporary = metadata.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(metadata)
    report.update(status="running", trained=True, started_at=datetime.now(timezone.utc).isoformat())
    save_report()
    started, best, best_epoch, step = time.perf_counter(), float("inf"), 0, 0
    with (args.output/"history.jsonl").open("x") as history:
        for epoch in range(1,args.max_epochs+1):
            order = jax.random.permutation(jax.random.fold_in(jax.random.PRNGKey(args.seed+100),epoch),
                                           len(training["states"]))
            for start in range(0,len(order),args.batch_size):
                indices = order[start:start+args.batch_size]
                batch = [training[name][indices] for name in ("states", "observations", "controls")]
                active = rollout_active(args,step+1)
                model,optimizer_state,loss,gradient_norm,parts = update(model,optimizer_state,*batch,
                    jax.random.fold_in(jax.random.PRNGKey(args.seed+200),step),active)
                step += 1
                row = dict(step=step,epoch=epoch,loss=float(loss),gradient_norm=float(gradient_norm),
                           learning_rate=float(schedule(step-1)),rollout_active=active,
                           loss_components={name:float(value) for name,value in parts.items()})
                if not np.isfinite([row["loss"],row["gradient_norm"],*row["loss_components"].values()]).all():
                    raise FloatingPointError(f"Nonfinite objective/gradient at update {step}")
                history.write(json.dumps(row)+"\n"); history.flush()
                values={"train/loss":row["loss"],"train/gradient_norm":row["gradient_norm"],
                        "train/learning_rate":row["learning_rate"],"train/rollout_active":int(active),"train/epoch":epoch}
                values.update({f"train/components/{name}":value for name,value in row["loss_components"].items()})
                tracker.metrics(values,step)
            if epoch % args.validation_every_epochs == 0 or epoch == args.max_epochs:
                validation,_,_,_ = evaluate(model,data.validation,args.validation_windows,args.validation_particles,77001)
                if validation["rmse"] < best:
                    best,best_epoch=validation["rmse"],epoch
                    report.update(best_step=step,best_epoch=epoch,best_validation_rmse=best)
                    eqx.tree_serialise_leaves(args.output/"model.eqx",model)
                validation_row=dict(step=step,epoch=epoch,validation_rmse=validation["rmse"],best_validation_rmse=best)
                with (args.output/"validation.jsonl").open("a") as vf:
                    vf.write(json.dumps(validation_row)+"\n")
                tracker.metrics({"validation/rmse":validation["rmse"],"validation/best_rmse":best},step)
                report.update(completed_steps=step,completed_epochs=epoch,
                              elapsed_training_seconds=time.perf_counter()-started)
                save_report()
                print(json.dumps({**row,**validation_row}),flush=True)
                if args.patience and epoch-best_epoch >= args.patience:
                    report["early_stopped"]=True
                    break
    report.update(training_seconds_including_compilation=time.perf_counter()-started,
                  completed_steps=step,completed_epochs=epoch)
    model=eqx.tree_deserialise_leaves(args.output/"model.eqx",model)
    first=[jnp.asarray(data.test[name][0]) for name in ("states","observations","controls")]
    np.asarray(infer(model,*first,jax.random.PRNGKey(99001),args.posterior_samples))
    metrics,samples,truth,inference_seconds=evaluate(model,data.test,args.eval_windows,args.posterior_samples,99001)
    metrics.update(eval_windows=len(truth),horizon=truth.shape[1],posterior_samples=args.posterior_samples,
                   inference_access=report["inference_access"],solver="heun",
                   milliseconds_per_trajectory=1000*inference_seconds/len(truth))
    np.savez_compressed(args.output/"posterior.npz",samples_normalized=samples,truth_normalized=truth)
    (args.output/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    report.update(status="complete",test=metrics,completed_at=datetime.now(timezone.utc).isoformat())
    tracker.metrics({**{f"test/{name}":value for name,value in metrics.items()},
                     "runtime/training_seconds_including_compilation":report["training_seconds_including_compilation"]},step)


def main(argv=None):
    args=parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Preserve existing run: {args.output}")
    model,data,report=prepare(args)
    args.output.mkdir(parents=True)
    metadata=args.output/"run_metadata.json"
    metadata.write_text(json.dumps(report,indent=2)+"\n")
    if args.train:
        tracker=JournalTracker(args.output,report,experiment="normflowsde-comparison")
        handlers={}
        def interrupted(signum,frame):
            raise KeyboardInterrupt(f"Received {signal.Signals(signum).name}")
        try:
            for signum in (signal.SIGINT,signal.SIGTERM):
                handlers[signum]=signal.signal(signum,interrupted)
            train(model,data,args,report,tracker)
        except BaseException as error:
            report.update(status="killed" if isinstance(error,(KeyboardInterrupt,SystemExit)) else "failed",
                          error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for signum,handler in handlers.items():
                signal.signal(signum,handler)
            metadata.write_text(json.dumps(report,indent=2)+"\n")
            tracker.finish({"complete":"FINISHED","killed":"KILLED"}.get(report["status"],"FAILED"),report)
    print(json.dumps(dict(status=report["status"],output=str(args.output),variant=VARIANT)),flush=True)


if __name__ == "__main__":
    main()
