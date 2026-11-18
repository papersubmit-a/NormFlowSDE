"""Width of the Markov Latent NSF that matches the amortized NormFlowSDE parameter count, per dataset."""
from experiments.normflowsde_comparison.train_direct_baseline import build, count, parse_args
from experiments.normflowsde_comparison.cache import load_cache

TARGET = dict(pendulum=103600, lorenz96=365360, aloha=129656, reassemble=120412)
SETUP = dict(pendulum=(2, 6), lorenz96=(3, 2), aloha=(3, 2), reassemble=(3, 2))   # depth, flow layers
for dataset, target in TARGET.items():
    data = load_cache(f"results/datasets/{dataset}")
    depth, flows = SETUP[dataset]
    def n(h):
        args = parse_args(["--method", "latent_nsf", "--dataset", dataset, "--data-cache", "x", "--output", "x",
                           "--hidden-size", str(h), "--hidden-depth", str(depth), "--flow-layers", str(flows)])
        return count(build(args, data))
    lo, hi = 8, 400
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if n(mid) <= target else (lo, mid)
    best = min((lo, hi), key=lambda h: abs(n(h) - target))
    print(f"{dataset}: H={best} params={n(best)} target={target}", flush=True)
