"""Seconds per training update and per validation of a train_state_filter run, for sizing CPU allocations.

Rebuilds the model and optimizer of an existing run (its run_metadata.json arguments) and times, after compilation,
--updates training updates and --chunks validation chunk calls (8 windows, protocol particles and draws). One
validation of the trainer is jitters x ceil(validation_windows / 8) such calls. The core count is whatever the
process is allowed to use (taskset / cgroup).
"""
import argparse, json, os, sys, time
from argparse import Namespace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import equinox as eqx, jax, jax.numpy as jnp, jax.random as jr, numpy as np, optax
from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.train_state_filter import build, _batched_inference

p = argparse.ArgumentParser(); p.add_argument("--run", type=Path, required=True)
p.add_argument("--updates", type=int, default=12); p.add_argument("--chunks", type=int, default=2)
a = p.parse_args()
meta = json.loads((a.run / "run_metadata.json").read_text()); args = Namespace(**meta["arguments"])
data = load_cache(Path(args.data_cache)); model = build(args, data)
training = {k: jnp.asarray(data.train[k]) for k in ("states", "observations", "controls")}
schedule = optax.warmup_cosine_decay_schedule(0., args.learning_rate, min(200, args.steps // 10 + 1), args.steps, args.learning_rate * .1)
optimizer = optax.chain(optax.clip_by_global_norm(1.), optax.adamw(schedule, weight_decay=1e-5))
state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def update(candidate, opt_state, key):
    ids = jr.randint(jr.fold_in(key, 0), (args.batch_windows,), 0, training["states"].shape[0])
    batch = [training[k][ids] for k in ("states", "observations", "controls")]
    (loss, parts), grads = eqx.filter_value_and_grad(lambda m: m.loss(*batch, jr.fold_in(key, 1)), has_aux=True)(candidate)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(candidate, eqx.is_inexact_array))
    return eqx.apply_updates(candidate, updates), opt_state, loss

t = time.perf_counter(); model, state, loss = update(model, state, jr.PRNGKey(0)); float(loss); compile_u = time.perf_counter() - t
times = []
for i in range(a.updates):
    t = time.perf_counter(); model, state, loss = update(model, state, jr.PRNGKey(i + 1)); float(loss); times.append(time.perf_counter() - t)
infer = _batched_inference(meta["particles"], meta["posterior_samples"])
split = data.validation; vt = []
for c in range(a.chunks + 1):
    idx = np.arange(8 * c, 8 * c + 8); keys = jnp.stack([jr.fold_in(jr.PRNGKey(77001), int(i)) for i in idx])
    t = time.perf_counter()
    np.asarray(infer(model, *(jnp.asarray(split[k][idx]) for k in ("states", "observations", "controls")), keys, jnp.asarray(0.01, split["states"].dtype)))
    vt.append(time.perf_counter() - t)
jit = len(args.selection_jitters.split(",")) if args.selection_jitters else 1
windows = min(args.validation_windows, len(split["states"])); calls = jit * -(-windows // 8)
u, v = float(np.median(times)), float(np.median(vt[1:]))
evals = args.steps // args.eval_every
import resource
print(json.dumps(dict(run=str(a.run), cores=len(os.sched_getaffinity(0)), s_per_update=round(u, 3), s_per_validation_chunk=round(v, 3),
                      validation_calls=calls, s_per_validation=round(v * calls, 1), compile_update_s=round(compile_u, 1), compile_validation_s=round(vt[0], 1),
                      projected_hours=dict(updates=round(u * args.steps / 3600, 1), validations=round(v * calls * evals / 3600, 1),
                                           total=round((u * args.steps + v * calls * evals) / 3600, 1)),
                      peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))), flush=True)
