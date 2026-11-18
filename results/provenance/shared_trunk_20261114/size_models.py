"""Model sizes of the 2026-11-14 shared-trunk campaign (results/provenance/shared_trunk_20261114).

Pendulum: all eight methods at most BUDGET parameters, every flow with two coupling layers. NormFlowSDE is sized
first (largest shared trunk within the budget); every other method gets the width whose parameter count is closest
to NormFlowSDE's stage-2 count without exceeding the budget. All particle filters and latent NSF use the same
small measurement flow (MEASUREMENT).

Lorenz-96, ALOHA, REASSEMBLE: NormFlowSDE's shared trunk keeps the depth, encoder, window feature, number of
coupling layers and coupling-head width of the previous model and is widened until the stage-2 parameter count is
closest to the previous model's stage-2 count. The four particle filters are re-sized to that count as well
(the direct baselines already match it and are not retrained). All four particle filters use one measurement-flow
architecture per dataset (MEASUREMENT on the pendulum, MEASUREMENT_LARGE elsewhere); NF-DPF previously sized its
measurement flow like its transition flow.

Every model is built by the builder its training script uses; only array shapes of the data matter.
Usage (repository root on PYTHONPATH, any device):  python results/provenance/shared_trunk_20261114/size_models.py
Writes sizes.json next to this file.
"""
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.random as jr
import numpy as np

from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
from bayesinf.applications.state_estimation.normalizing_flow_dpf import NFDPFConfig, NormalizingFlowDPF
from experiments.normflowsde_comparison import train_direct_baseline as tdb
from experiments.normflowsde_comparison import train_state_filter as tsf

HERE = Path(__file__).resolve().parent
BUDGET = 10_000
MEASUREMENT = dict(measurement_hidden=16, measurement_depth=2, measurement_layers=2)
MEASUREMENT_LARGE = dict(measurement_hidden=64, measurement_depth=2, measurement_layers=4)  # as before for the bootstrap PFs


def count(tree):
    return int(sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))))


def fake_data(state_dim, observation_dim, control_dim, dt, observed):
    """Shapes and metadata only; the builders never look at the values."""
    zeros = lambda *shape: np.zeros(shape, np.float32)
    metadata = dict(observation_indices=list(observed), state_mean=zeros(state_dim), state_std=np.ones(state_dim),
                    observation_mean=zeros(observation_dim), observation_std=np.ones(observation_dim))
    return SimpleNamespace(train=dict(states=zeros(2, 5, state_dim), observations=zeros(2, 4, observation_dim),
                                      controls=zeros(2, 4, control_dim)), dt=dt, metadata=metadata)


def normflowsde(config, learned_scale=True):
    return count(RealNVPFlowSDE(**config, learned_scale=learned_scale, key=jr.PRNGKey(0)))


def closest(function, target, limit, widths=range(4, 700)):
    """Width whose count is closest to target among counts <= limit.

    Every count here is an exact quadratic in the width, so three builds determine it; the chosen width and its
    neighbours are then built again and must reproduce the prediction.
    """
    probe = (8, 16, 32)
    a, b, c = np.linalg.solve(np.array([[w * w, w, 1.] for w in probe]), np.array([function(w) for w in probe], float))
    predict = lambda w: int(round(a * w * w + b * w + c))
    best = None
    for width in widths:
        value = predict(width)
        if value <= limit and (best is None or abs(value - target) < abs(best[1] - target)):
            best = (width, value)
    for width in (best[0] - 1, best[0], best[0] + 1):
        assert function(width) == predict(width), (width, function(width), predict(width))
    return best


def particle_filters(data, target, limit, depth, measurement, pf_arch):
    """NSF-PF, FlowSDE-PF, NormFlowSDE-PF (shared trunk) and NF-DPF with a common measurement flow."""
    out = {}

    def state_filter(method, **extra):
        def build(width):
            args = Namespace(method=method, seed=0, hidden_size=width, hidden_depth=depth, flow_layers=2, alpha=1.0,
                             diffusion_hidden_size=None, learned_scale=False, shared_trunk=False, diffusion=0.01,
                             flow_substeps=4, epr_weight=0., **measurement)
            for name, value in extra.items():
                setattr(args, name, value)
            return tsf.build(args, data)
        width, value = closest(lambda w: count(build(w)), target, limit)
        model = build(width)
        return dict(hidden_size=width, total=value, dynamics=count(model.dynamics), measurement=count(model.measurement),
                    **extra)

    out["nsf"] = state_filter("nsf")
    out["flowsde"] = state_filter("flowsde")
    out["normflowsde_pf"] = state_filter("normflowsde", shared_trunk=True, learned_scale=True, **pf_arch)
    dims = (data.train["states"].shape[-1], data.train["observations"].shape[-1], data.train["controls"].shape[-1])

    def nf_dpf(width):
        return NormalizingFlowDPF(NFDPFConfig(*dims, width, depth, 2, measurement_hidden_size=measurement["measurement_hidden"],
                                              measurement_hidden_depth=measurement["measurement_depth"],
                                              measurement_flow_layers=measurement["measurement_layers"]), key=jr.PRNGKey(0))
    width, value = closest(lambda w: count(nf_dpf(w)), target, limit)
    out["nf_dpf"] = dict(hidden_size=width, hidden_depth=depth, flow_layers=2, total=value,
                         measurement=count(nf_dpf(width).measurement))
    return out


def pendulum():
    data = fake_data(2, 2, 1, 0.05, (0, 1))
    base = dict(d_x=2, d_y=3, d_u=1, d_c=16, d_hidden_enc=16, depth=2, n_realnvp_layers=2, observation_window=10,
                direct_control=True, time_step=0.05, flow_type="realnvp", shared_trunk=True, d_hidden_diffusion=16)
    width, total = closest(lambda w: normflowsde(dict(base, d_hidden=w)), BUDGET, BUDGET)
    out = dict(normflowsde=dict(config=dict(base, d_hidden=width), stage2=total,
                                stage1=normflowsde(dict(base, d_hidden=width), learned_scale=False)))
    target = total

    out.update(particle_filters(data, target, BUDGET, 2, MEASUREMENT, dict(diffusion_hidden_size=16)))

    for method in ("latent_nsf", "latent_sde", "sigker_sde"):
        def build(width):
            args = tdb.parse_args(["--method", method, "--dataset", "pendulum", "--data-cache", ".", "--output", ".",
                                   "--hidden-size", str(width), "--hidden-depth", "2", "--flow-layers", "2",
                                   "--encoder-hidden", "16", "--encoder-window", "10", "--direct-control",
                                   *[f"--{k.replace('_', '-')}={v}" for k, v in MEASUREMENT.items()]])
            return tdb.estimator(tdb.build(args, data))
        width, value = closest(lambda w: count(build(w)), target, BUDGET)
        out[method] = dict(hidden_size=width, total=value)
    return out


def widened(name, previous):
    """previous: (stage-1 model_config of the selected run, its stage-2 parameter count)."""
    config, target = previous
    base = dict(config, shared_trunk=True)
    width, total = closest(lambda w: normflowsde(dict(base, d_hidden=w)), target, 10 ** 9, range(64, 700))
    out = dict(normflowsde=dict(config=dict(base, d_hidden=width), stage2=total, previous_stage2=target,
                                stage1=normflowsde(dict(base, d_hidden=width), learned_scale=False)))
    data = fake_data(*DIMENSIONS[name], config["time_step"], range(DIMENSIONS[name][1]))
    # NormFlowSDE-PF: the transition of the amortized model (trunk depth 3, coupling-head width 64, its alpha).
    out.update(particle_filters(data, total, 10 ** 9, 3, MEASUREMENT_LARGE,
                                dict(diffusion_hidden_size=64, alpha=config.get("realnvp_alpha", 1.0))))
    return out


DIMENSIONS = dict(lorenz96=(12, 6, 1), aloha=(14, 14, 14), reassemble=(7, 7, 9))  # state, observation, control


PREVIOUS = dict(  # stage-1 model_config and stage-2 parameter_count of results/normflowsde/<dataset>/seed0
    lorenz96=(dict(d_x=12, d_y=7, d_u=1, d_c=128, d_hidden=256, depth=3, n_realnvp_layers=2, observation_window=10,
                   d_hidden_enc=128, d_hidden_diffusion=64, direct_control=False, observation_stride=1, time_step=0.02,
                   flow_type="realnvp", realnvp_alpha=1.5), 365360),
    aloha=(dict(d_x=14, d_y=28, d_u=14, d_c=64, d_hidden=128, depth=3, n_realnvp_layers=2, observation_window=10,
                d_hidden_enc=64, d_hidden_diffusion=64, direct_control=True, time_step=0.02, flow_type="realnvp"), 129656),
    reassemble=(dict(d_x=7, d_y=16, d_u=9, d_c=64, d_hidden=128, depth=3, n_realnvp_layers=2, observation_window=10,
                     d_hidden_enc=64, d_hidden_diffusion=64, direct_control=True, time_step=1 / 30, flow_type="realnvp"),
                120412),
)

if __name__ == "__main__":
    sizes = dict(budget=BUDGET, measurement_flow=MEASUREMENT, pendulum=pendulum(),
                 **{name: widened(name, previous) for name, previous in PREVIOUS.items()})
    (HERE / "sizes.json").write_text(json.dumps(sizes, indent=1) + "\n")
    for method, entry in sizes["pendulum"].items():
        print(f"pendulum   {method:15s} {json.dumps({k: v for k, v in entry.items() if k != 'config'})}"
              + (f"  d_hidden={entry['config']['d_hidden']}" if "config" in entry else ""))
    for name in PREVIOUS:
        for method, entry in sizes[name].items():
            print(f"{name:10s} {method:15s} {json.dumps({k: v for k, v in entry.items() if k != 'config'})}"
                  + (f"  d_hidden={entry['config']['d_hidden']}" if "config" in entry else ""))
