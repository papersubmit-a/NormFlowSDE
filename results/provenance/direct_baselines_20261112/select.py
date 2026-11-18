"""Select each direct baseline's configuration per dataset from the seed-0 validation grid (GRID.md).

Candidates: the default run results/<method>/<dataset>/seed0/run and every results/<method>_sweep/<config>/<dataset>/
seed0/run with a summary.json (failed runs are listed as such). Selection: lowest best_validation_rmse (the benchmark's
checkpoint and hyperparameter rule). Test metrics are not read. Writes selected.json and tasks_selected.txt
(<method> <dataset> <seed> <TAG> <extra args> for seeds 1 and 2 of every selected sweep configuration; a selected
default needs no new runs).
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
R = ROOT / "results"


def grid_arguments():
    """TAG -> extra args, from every task list that launched a sweep run."""
    table = {}
    for path in HERE.glob("tasks_*sweep*.txt"):
        for line in path.read_text().splitlines():
            parts = line.split(maxsplit=1)
            if path.name == "tasks_nsf_lorenz_sweep.txt":
                table[("latent_nsf", parts[0])] = parts[1]
            else:
                method, rest = line.split(maxsplit=1)
                tag, extra = rest.split(maxsplit=1)
                table[(method, tag)] = extra
    for path in HERE.glob("tasks_grid_*.txt"):
        for line in path.read_text().splitlines():
            method, tag, extra = line.split(maxsplit=2)
            table[(method, tag)] = extra
    return table


arguments = grid_arguments()
selected, tasks = {}, []
for method in ("latent_nsf", "latent_sde", "sigker_sde"):
    for dataset in ("pendulum", "lorenz96", "aloha", "reassemble"):
        moved = R / f"{method}_sweep/default/{dataset}/seed0/run"      # assemble.py moved the default here
        candidates = [("default", moved if moved.is_dir() else R / method / dataset / "seed0/run", "")]
        for run in sorted((R / f"{method}_sweep").glob(f"*/{dataset}/seed0/run")):
            config = run.parents[2].name
            if config == "default":
                continue
            candidates.append((config, run, arguments[(method, f"_sweep/{config}")]))
        rows = []
        for config, run, extra in candidates:
            summary = run / "summary.json"
            if summary.is_file():
                s = json.loads(summary.read_text())
                rows.append(dict(config=config, extra=extra, best_validation_rmse=s["best_validation_rmse"],
                                 best_validation_coverage_90=s["best_validation_coverage_90"], best_step=s["best_step"]))
            else:
                rows.append(dict(config=config, extra=extra, status="no summary.json (failed or running)"))
        valid = [r for r in rows if "best_validation_rmse" in r]
        best = min(valid, key=lambda r: r["best_validation_rmse"])
        selected[f"{method}:{dataset}"] = dict(selected=best, candidates=sorted(
            rows, key=lambda r: r.get("best_validation_rmse", float("inf"))))
        print(f"{method:11}{dataset:11} {best['config']:28} val {best['best_validation_rmse']:.4f} "
              f"cov {best['best_validation_coverage_90']:.3f}  ({len(valid)}/{len(rows)} complete)")
        if best["config"] != "default":
            for seed in (1, 2):
                tasks.append(f"{method} {dataset} {seed} _sweep/{best['config']} {best['extra']}")
(HERE / "selected.json").write_text(json.dumps(selected, indent=2) + "\n")
(HERE / "tasks_selected.txt").write_text("\n".join(tasks) + ("\n" if tasks else ""))
print(len(tasks), "seed runs to launch")
