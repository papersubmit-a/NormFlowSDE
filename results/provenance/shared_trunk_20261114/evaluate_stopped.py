"""Score a train_state_filter run that was stopped before its last update (2026-11-15).

The trainer keeps the checkpoint with the lowest validation RMSE in model.eqx and scores the test split only after
the last update. For a run cancelled earlier, this performs that final step on the checkpoint it left: the best
validation row of history.jsonl gives the step, the validation scores and the selected process-noise jitter; the
test split is then filtered exactly as train_state_filter.main does (seed 99001, protocol particles and draws) and
the usual outputs are written into the run directory (posterior.npz, metrics.json, summary.json, run_metadata.json).
run_metadata.json / summary.json record under "stopped_early" how many of the planned updates were completed.

  python evaluate_stopped.py --run results/shared_trunk_20261114/<method>/<dataset>/seed<k>/<stage>
"""
from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import numpy as np  # noqa: E402

from experiments.normflowsde_comparison.cache import load_cache  # noqa: E402
from experiments.normflowsde_comparison.evaluation_metrics import metrics_from_samples  # noqa: E402
from experiments.normflowsde_comparison.run import score_samples  # noqa: E402
from experiments.normflowsde_comparison.train_state_filter import ACCESS, build, latency, posterior  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--reason", default="cancelled on a 2-4 core CPU allocation after about 13 h (10-30x slower than on a GPU)")
    args = p.parse_args()
    if (args.run / "summary.json").exists():
        raise FileExistsError(f"Already scored: {args.run}")
    report = json.loads((args.run / "run_metadata.json").read_text())
    if report.get("status") != "running":
        raise ValueError(f"Not an unfinished run: status={report.get('status')}")
    train = Namespace(**report["arguments"])
    best, row_best, last, unreadable = np.inf, None, None, 0
    for line in (args.run / "history.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # the update being written when the job was cancelled
            unreadable += 1
            continue
        last = row
        if "validation_rmse" in row and row["validation_rmse"] < best:  # as the trainer: first strict minimum
            best, row_best = row["validation_rmse"], row
    if row_best is None or unreadable > 1:
        raise ValueError(f"No usable validation row ({unreadable} unreadable lines): {args.run}")
    jitter = float(row_best.get("validation_jitter", 0.))
    data = load_cache(Path(train.data_cache))
    model = eqx.tree_deserialise_leaves(args.run / "model.eqx", build(train, data))
    particles, draws = report["particles"], report["posterior_samples"]
    samples, truth = posterior(model, data.test, data.protocol["eval_windows"], 99001, particles, draws,
                               train.eval_chunk, jitter)
    metrics = score_samples(samples, truth, data)
    proper = metrics_from_samples(samples, truth, observed_indices=data.protocol["observed_indices"],
                                  metric_space=data.protocol["metric_space"], state_mean=data.metadata["state_mean"],
                                  state_std=data.metadata["state_std"], energy_method="exact")
    metrics.update(proper["metrics"])
    metrics.update(eval_windows=len(truth), posterior_samples=samples.shape[1], inference_access=ACCESS,
                   latency_ms_per_trajectory=latency(model, data.test, particles, draws, jitter=jitter))
    if train.selection_jitters:
        metrics["jitter"] = jitter
    stopped = dict(completed_updates=last["step"], planned_updates=train.steps, last_validation_step=max(
        r for r in [row_best["step"]] + [json.loads(l)["step"] for l in (args.run / "history.jsonl").read_text().splitlines()
                                         if '"validation_rmse"' in l]), reason=args.reason,
                   note="learning-rate schedule (cosine to planned_updates) not completed; checkpoint = best validation "
                        "RMSE among the validations reached", test_scored_on=[str(d) for d in jax.devices()])
    np.savez_compressed(args.run / "posterior.npz", samples_normalized=samples, truth_normalized=truth)
    (args.run / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    report.update(training_seconds=row_best and max(json.loads(l).get("elapsed_seconds", 0) for l in
                                                   (args.run / "history.jsonl").read_text().splitlines() if '"elapsed_seconds"' in l),
                  best_step=row_best["step"], best_validation_rmse=best,
                  best_validation_coverage_90=row_best["validation_coverage_90"], stopped_early=stopped)
    if train.selection_jitters:
        report.update(jitter=jitter, variant=report["variant"] + " + joint (checkpoint, process-noise "
                      "inflation) validation selection")
    report["test"] = {k: metrics[k] for k in ("rmse", "crps", "coverage_90", "mean_interval_width",
                                              "latency_ms_per_trajectory") if k in metrics}
    report["status"] = "complete"
    (args.run / "run_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.run / "summary.json").write_text(json.dumps(dict(
        method=report["method"], seed=report["seed"], best_step=row_best["step"], best_validation_rmse=best,
        best_validation_coverage_90=row_best["validation_coverage_90"], test=report["test"],
        **(dict(jitter=jitter) if train.selection_jitters else {}), stopped_early=stopped), indent=2) + "\n")
    print(json.dumps(dict(status="complete", output=str(args.run), stopped_early=stopped, test=report["test"])), flush=True)


if __name__ == "__main__":
    main()
