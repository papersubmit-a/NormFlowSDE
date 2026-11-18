"""Check that the direct baselines' WindowEncoder reproduces the amortized NormFlowSDE encoder exactly.

For each dataset: build NormFlowSDE from its seed-0 run_metadata.json model_config, load its trained model.eqx,
copy its BiGRU into a WindowEncoder with the dataset's settings, and compare, on validation windows,
    NormFlowSDE: model.encode(concat(preceding_observations(...), controls))   (run.py draw_samples inputs)
    baseline:    WindowEncoder(x0, observations, controls)
NormFlowSDE with direct_control appends u[k] to the context; the baselines receive u[k] as a separate input, so the
comparison drops those trailing columns. Writes encoder_check.json next to this file.
"""
import json
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE  # noqa: E402
from experiments.normflowsde_comparison.cache import load_cache  # noqa: E402
from experiments.normflowsde_comparison.direct_baselines import WindowEncoder  # noqa: E402
from experiments.normflowsde_comparison.run import preceding_observations  # noqa: E402

jax.config.update("jax_enable_x64", False)
report = {}
for dataset in ("pendulum", "lorenz96", "aloha", "reassemble"):
    run = ROOT / "results/normflowsde" / dataset / "seed0/s2"
    config = json.loads((run / "run_metadata.json").read_text())["model_config"]
    data = load_cache(ROOT / "results/datasets" / dataset)
    model = RealNVPFlowSDE(**config, key=jr.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(run / "model.eqx", model)
    observation_dim = data.validation["observations"].shape[-1]
    control_dim = data.validation["controls"].shape[-1]
    encoder = WindowEncoder(observation_dim, control_dim, config["d_c"], window=config["observation_window"],
                            direct_control=config["direct_control"], metadata=data.metadata, key=jr.PRNGKey(1))
    assert config["d_hidden_enc"] == config["d_c"]
    encoder = eqx.tree_at(lambda e: e.gru, encoder, model.encoder)
    states, observations, controls = (jnp.asarray(data.validation[k][:8]) for k in ("states", "observations", "controls"))
    inputs = jnp.concatenate((preceding_observations(states, observations, data.metadata), controls), -1)
    reference = jax.vmap(model.encode)(inputs)[..., :config["d_c"]]
    ours = jax.vmap(encoder)(states[:, 0], observations, controls)
    error = float(jnp.max(jnp.abs(reference - ours)))
    report[dataset] = dict(max_abs_difference=error, context_shape=list(ours.shape), window=encoder.window,
                           direct_control=encoder.direct_control, hidden=encoder.hidden_size)
    print(dataset, report[dataset], flush=True)
    assert error < 1e-5, dataset
(Path(__file__).with_name("encoder_check.json")).write_text(json.dumps(report, indent=2) + "\n")
print("OK")
