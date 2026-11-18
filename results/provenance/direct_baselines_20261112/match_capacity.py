"""Hidden widths of the direct baselines with NormFlowSDE's encoder, matched to the amortized NormFlowSDE.

Target = parameter_count of results/normflowsde/<dataset>/seed0/s2 (103600 / 365360 / 129656 / 120412), counted
exactly as train_direct_baseline.py reports it (count(estimator(model))). For each method and dataset the width
with the closest count is chosen; depth, flow layers and encoder settings are those of run_task.sh.
Writes capacity.json next to this file.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.normflowsde_comparison.cache import load_cache  # noqa: E402
from experiments.normflowsde_comparison import train_direct_baseline as tdb  # noqa: E402

SETTINGS = dict(pendulum=dict(depth=2, flows=6, encoder=16, direct=True),
                lorenz96=dict(depth=3, flows=2, encoder=128, direct=False),
                aloha=dict(depth=3, flows=2, encoder=64, direct=True),
                reassemble=dict(depth=3, flows=2, encoder=64, direct=True))
result = {}
for dataset, s in SETTINGS.items():
    target = json.loads((ROOT / f"results/normflowsde/{dataset}/seed0/s2/run_metadata.json").read_text())["parameter_count"]
    data = load_cache(ROOT / "results/datasets" / dataset)
    for method in ("latent_nsf", "latent_sde", "sigker_sde"):
        def params(width):
            argv = ["--method", method, "--dataset", dataset, "--data-cache", "x", "--output", "x",
                    "--hidden-size", str(width), "--hidden-depth", str(s["depth"]), "--flow-layers", str(s["flows"]),
                    "--encoder-hidden", str(s["encoder"]), "--direct-control" if s["direct"] else "--no-direct-control"]
            return tdb.count(tdb.estimator(tdb.build(tdb.parse_args(argv), data)))
        lo, hi = 8, 1024                                   # count is increasing in the width: bisection
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if params(mid) <= target else (lo, mid)
        best = min((lo, hi), key=lambda w: abs(params(w) - target))
        result[f"{method}:{dataset}"] = dict(hidden_size=best, parameters=params(best), target=target,
                                            relative_difference=params(best) / target - 1)
        print(method, dataset, result[f"{method}:{dataset}"], flush=True)
(Path(__file__).with_name("capacity.json")).write_text(json.dumps(result, indent=2) + "\n")
