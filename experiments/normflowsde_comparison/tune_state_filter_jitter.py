"""Select particle-filter process-noise inflation on validation, then score the test split once.

Diagnosis (2026-10-19, Lorenz-96 NSF-PF): the learned NSF transition is accurate but
under-dispersed as a filter prior (one-step sample std 0.0023 versus one-step mean error
0.0042, normalized), so the particle cloud is ~5x narrower than the filtering error after
50 steps and 90% coverage is 0.40, while ESS stays high. Adding N(0, jitter^2 I) to each
predicted particle is standard process-noise inflation for model error. As for FlowSDE's
fixed diffusion, the value is a filter hyperparameter chosen on validation data; the grid
includes 0, so "no inflation" remains a candidate.

For a trained run (train_state_filter output) this evaluates every --grid value on the
validation windows (validation seed 77001, protocol particles and draws), keeps the lowest
validation RMSE, and writes a new run directory with the unchanged model, the selected
jitter, the test posterior (seed 99001) and metrics in the comparison format.
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import shutil

import equinox as eqx
import numpy as np

from experiments.normflowsde_comparison.cache import load_cache
from experiments.normflowsde_comparison.evaluation_metrics import metrics_from_samples
from experiments.normflowsde_comparison.run import score_samples
from experiments.normflowsde_comparison.train_state_filter import ACCESS, build, latency, posterior


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Completed train_state_filter run directory.")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--grid", default="0,0.005,0.01,0.02,0.03,0.05", help="Jitter std values (normalized units).")
    p.add_argument("--eval-chunk", type=int, default=8)
    args = p.parse_args(argv)
    if (args.output / "summary.json").exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    metadata = json.loads((args.run / "run_metadata.json").read_text())
    train_args = Namespace(**metadata["arguments"])
    data = load_cache(Path(train_args.data_cache))
    model = eqx.tree_deserialise_leaves(args.run / "model.eqx", build(train_args, data))
    particles, draws = metadata["particles"], metadata["posterior_samples"]
    windows = getattr(train_args, "validation_windows", 10_000)

    grid = [float(v) for v in args.grid.split(",")]
    validation = []
    for jitter in grid:
        samples, truth = posterior(model, data.validation, windows, 77001, particles, draws, args.eval_chunk, jitter)
        scores = score_samples(samples, truth, data)
        validation.append(dict(jitter=jitter, rmse=scores["rmse"], coverage_90=scores["coverage_90"]))
        print(json.dumps(validation[-1]), flush=True)
    chosen = min(validation, key=lambda row: row["rmse"])["jitter"]

    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.run / "model.eqx", args.output / "model.eqx")
    samples, truth = posterior(model, data.test, data.protocol["eval_windows"], 99001, particles, draws,
                               args.eval_chunk, chosen)
    metrics = score_samples(samples, truth, data)
    proper = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                  metric_space=data.protocol["metric_space"], state_mean=data.metadata["state_mean"],
                                  state_std=data.metadata["state_std"], energy_method="exact")
    metrics.update(proper["metrics"])
    metrics.update(eval_windows=len(truth), posterior_samples=samples.shape[1], inference_access=ACCESS,
                   jitter=chosen, latency_ms_per_trajectory=latency(model, data.test, particles, draws, jitter=chosen))
    np.savez_compressed(args.output / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    test = {k: metrics[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width", "latency_ms_per_trajectory")}
    metadata.update(status="complete", jitter=chosen, jitter_selection=validation, source_run=str(args.run),
                    variant=metadata.get("variant", "") + " + validation-selected process-noise inflation", test=test)
    (args.output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "summary.json").write_text(json.dumps(dict(
        method=metadata["method"], seed=metadata["seed"], jitter=chosen, jitter_selection=validation,
        best_validation_rmse=min(r["rmse"] for r in validation), test=test), indent=2) + "\n")
    for name in ("history.jsonl",):
        if (args.run / name).is_file():
            shutil.copy2(args.run / name, args.output / name)
    print(json.dumps(dict(status="complete", output=str(args.output), jitter=chosen, test=test)), flush=True)


if __name__ == "__main__":
    main()
