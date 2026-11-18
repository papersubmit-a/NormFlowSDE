"""Install the validation-selected configurations (selected.json) as the direct baselines' main results.

For every <method>:<dataset> whose selected config is not "default":
  1. the default runs results/<method>/<dataset>/seed{0,1,2}/run move to results/<method>_sweep/default/<dataset>/...
     (so the grid stays complete);
  2. results/<method>/<dataset>/seed0/run becomes a copy of the sweep's seed-0 run (the sweep keeps its own copy);
     seeds 1 and 2 move from results/<method>_sweep/<config>/<dataset>/seed{1,2}/run.
Each installed run gets selection.json (config, extra args, seed-0 validation RMSE). Idempotent: an already
installed dataset (selection.json present in seed0) is skipped. Run with --dry-run first.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[1]
dry = "--dry-run" in sys.argv
selected = json.loads((HERE / "selected.json").read_text())


def act(message, fn, *args):
    print(("DRY " if dry else "") + message)
    if not dry:
        fn(*args)


missing = []
for key, entry in selected.items():
    method, dataset = key.split(":")
    best = entry["selected"]
    if best["config"] == "default":
        continue
    sweep = R / f"{method}_sweep" / best["config"] / dataset
    for seed in (0, 1, 2):
        if not (sweep / f"seed{seed}/run/summary.json").is_file():
            missing.append(f"{sweep}/seed{seed}")
if missing:
    sys.exit("incomplete selected runs:\n" + "\n".join(missing))

for key, entry in selected.items():
    method, dataset = key.split(":")
    best = entry["selected"]
    if best["config"] == "default":
        print(f"{key}: default selected, nothing to do")
        continue
    main = R / method / dataset
    if (main / "seed0/run/selection.json").is_file():
        print(f"{key}: already installed")
        continue
    sweep = R / f"{method}_sweep" / best["config"] / dataset
    default = R / f"{method}_sweep" / "default" / dataset
    record = dict(config=best["config"], extra_args=best["extra"], seed0_best_validation_rmse=best["best_validation_rmse"],
                  rule="lowest seed-0 best_validation_rmse over GRID.md (select.py -> selected.json)")
    for seed in (0, 1, 2):
        src, dst = main / f"seed{seed}/run", default / f"seed{seed}/run"
        act(f"move {src} -> {dst}", lambda s=src, d=dst: (d.parent.mkdir(parents=True, exist_ok=True), shutil.move(s, d)))
    for seed in (0, 1, 2):
        src, dst = sweep / f"seed{seed}/run", main / f"seed{seed}/run"
        if seed == 0:
            act(f"copy {src} -> {dst}", lambda s=src, d=dst: shutil.copytree(s, d))
        else:
            act(f"move {src} -> {dst}", lambda s=src, d=dst: (d.parent.mkdir(parents=True, exist_ok=True), shutil.move(s, d)))
        act(f"write {dst}/selection.json",
            lambda d=dst: (d / "selection.json").write_text(json.dumps(record, indent=2) + "\n"))
