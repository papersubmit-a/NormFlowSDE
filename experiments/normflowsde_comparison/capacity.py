"""Verified parameter budgets for a new, capacity-matched comparison.

These profiles change network widths while preserving each method's model
family. Verification constructs the actual models; it performs no training,
data generation, checkpoint loading or evaluation. Historical run matrices
remain independent of these opt-in profiles.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np


PROFILE_NAME = "matched-v1"
LEGACY_PROFILE_NAME = "legacy-aligned-v1"
CAUSAL_PROFILE_NAME = "legacy-aligned-causal-v1"
PROFILE_NAMES = (PROFILE_NAME, LEGACY_PROFILE_NAME, CAUSAL_PROFILE_NAME)
METHODS = ("normflowsde", "nf_dpf", "nsf", "flowsde")
DATASETS = ("lorenz96", "pendulum", "aloha", "reassemble")
TARGET_TOLERANCE = 0.02
MAX_DATASET_RATIO = 1.05

# State, observation and control dimensions are the audited primary cases.
# The one Lorenz-96 control channel is the existing zero-valued dummy input.
DATASET_SETTINGS = {
    "pendulum": dict(state_dim=2, observation_dim=2, control_dim=1, dt=0.05,
                     context_dim=16, target_parameters=100_000),
    "lorenz96": dict(state_dim=12, observation_dim=6, control_dim=1, dt=0.02,
                     context_dim=64, target_parameters=1_000_000),
    "aloha": dict(state_dim=14, observation_dim=14, control_dim=14, dt=0.02,
                  context_dim=32, target_parameters=500_000),
    "reassemble": dict(state_dim=7, observation_dim=7, control_dim=9, dt=1 / 30,
                       context_dim=32, target_parameters=500_000),
}

# Each tuple is (hidden width, independently verified actual parameter count).
# All MLP depth settings are three; NormFlowSDE's coupling subnetworks retain
# their architecture's intrinsic depth of two. FlowSDE has no coupling flows,
# so its flow_layers setting is never used as a surrogate parameter count.
_WIDTHS_AND_COUNTS = {
    "pendulum": {
        "normflowsde": (88, 98_846), "nf_dpf": (46, 100_594),
        "nsf": (54, 101_260), "flowsde": (150, 99_445),
    },
    "lorenz96": {
        "normflowsde": (296, 1_002_468), "nf_dpf": (144, 1_011_156),
        "nsf": (164, 996_368), "flowsde": (476, 1_001_473),
    },
    "aloha": {
        "normflowsde": (202, 499_096), "nf_dpf": (92, 493_248),
        "nsf": (112, 503_128), "flowsde": (336, 500_407),
    },
    "reassemble": {
        "normflowsde": (210, 498_123), "nf_dpf": (100, 500_794),
        "nsf": (118, 493_230), "flowsde": (340, 501_480),
    },
}

# Historical state-estimation network sizes anchor NormFlowSDE's capacity.
# Audited current channel dimensions and the preceding-observation encoder
# remain in use; these are not replicas of checkpoints with future access or
# optional initial-state inference networks. Other method widths are selected
# to match the actual NormFlowSDE count for each dataset.
_LEGACY_SETTINGS = {
    "pendulum": dict(context_dim=16, hidden_depth=2, flow_layers=6,
                     diffusion_hidden_size=88, target_parameters=91_014),
    "lorenz96": dict(context_dim=128, hidden_depth=3, flow_layers=2,
                     diffusion_hidden_size=64, target_parameters=348_964),
    "aloha": dict(context_dim=64, hidden_depth=3, flow_layers=2,
                  diffusion_hidden_size=64, target_parameters=116_586),
    "reassemble": dict(context_dim=64, hidden_depth=3, flow_layers=2,
                       diffusion_hidden_size=64, target_parameters=108_565),
}
_LEGACY_WIDTHS_AND_COUNTS = {
    "pendulum": {"normflowsde": (88, 91_014), "nf_dpf": (61, 92_377),
                 "nsf": (69, 90_070), "flowsde": (200, 91_395)},
    "lorenz96": {"normflowsde": (256, 348_964), "nf_dpf": (129, 351_189),
                 "nsf": (106, 348_128), "flowsde": (208, 349_221)},
    "aloha": {"normflowsde": (128, 116_586), "nf_dpf": (65, 116_277),
              "nsf": (59, 117_063), "flowsde": (123, 116_971)},
    "reassemble": {"normflowsde": (128, 108_565), "nf_dpf": (69, 108_801),
                   "nsf": (62, 108_718), "flowsde": (121, 108_890)},
}

# legacy-aligned-causal-v1: NSF/FlowSDE with the causal (forward-only, filtering)
# observation encoder. Dropping the backward GRU removes encoder parameters, so the
# conditional-generator width is re-solved (bisection over actual models) to the same
# per-dataset NormFlowSDE target; all deviations are below 0.5%. Encoder width,
# depth and flow layers are unchanged; NormFlowSDE and NF-DPF keep legacy widths.
_CAUSAL_WIDTHS_AND_COUNTS = {
    "pendulum": {"nsf": (70, 91_136), "flowsde": (201, 91_058)},
    "lorenz96": {"nsf": (126, 349_856), "flowsde": (242, 348_495)},
    "aloha": {"nsf": (68, 117_004), "flowsde": (139, 116_715)},
    "reassemble": {"nsf": (71, 108_941), "flowsde": (136, 108_096)},
}


@dataclass(frozen=True)
class CapacityProfile:
    dataset: str
    method: str
    target_parameters: int
    expected_parameters: int
    hidden_size: int
    context_dim: int
    hidden_depth: int = 3
    flow_layers: int = 6
    name: str = PROFILE_NAME
    diffusion_hidden_size: int | None = None

    def to_dict(self):
        return asdict(self)


def capacity_profile(dataset: str, method: str, profile: str = PROFILE_NAME) -> CapacityProfile:
    """Return one explicit, immutable profile; unknown names are rejected."""
    if dataset not in DATASET_SETTINGS:
        raise ValueError(f"Unknown capacity dataset: {dataset}")
    if method not in METHODS:
        raise ValueError(f"Unknown capacity method: {method}")
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown capacity profile: {profile}")
    if profile in (LEGACY_PROFILE_NAME, CAUSAL_PROFILE_NAME):
        settings = _LEGACY_SETTINGS[dataset]
        hidden_size, expected = (_CAUSAL_WIDTHS_AND_COUNTS[dataset][method]
                                 if profile == CAUSAL_PROFILE_NAME and method in ("nsf", "flowsde")
                                 else _LEGACY_WIDTHS_AND_COUNTS[dataset][method])
        return CapacityProfile(
            dataset=dataset, method=method, hidden_size=hidden_size,
            context_dim=settings["context_dim"],
            target_parameters=settings["target_parameters"],
            expected_parameters=expected, hidden_depth=settings["hidden_depth"],
            flow_layers=settings["flow_layers"], name=profile,
            diffusion_hidden_size=(settings["diffusion_hidden_size"]
                                   if method == "normflowsde" else None),
        )
    hidden_size, expected = _WIDTHS_AND_COUNTS[dataset][method]
    settings = DATASET_SETTINGS[dataset]
    return CapacityProfile(
        dataset=dataset, method=method, hidden_size=hidden_size,
        context_dim=settings["context_dim"],
        target_parameters=settings["target_parameters"],
        expected_parameters=expected,
    )


def capacity_arguments(dataset: str, method: str, profile: str = PROFILE_NAME) -> list[str]:
    """Existing runner flags, suitable for appending to a command argv list."""
    profile = capacity_profile(dataset, method, profile)
    arguments = ["--hidden-size", str(profile.hidden_size),
                 "--hidden-depth", str(profile.hidden_depth),
                 "--flow-layers", str(profile.flow_layers),
                 "--context-dim", str(profile.context_dim)]
    if profile.diffusion_hidden_size is not None:
        arguments.extend(("--diffusion-hidden-size", str(profile.diffusion_hidden_size)))
    return arguments


def count_parameters(model) -> int:
    """Count the inexact arrays selected by the existing optimizer filter."""
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(
        eqx.filter(model, eqx.is_inexact_array)))


def validate_capacity(model, data, args) -> dict:
    """Reject changed dimensions, architecture settings or parameter counts.

    Data may be real benchmark arrays or canonical arrays used exclusively for
    constructing the model. The count is measured on the model passed here,
    rather than inferred from generic arguments that a method may not use.
    """
    from . import run

    profile = capacity_profile(args.dataset, args.method,
                               getattr(args, "capacity_profile", None) or PROFILE_NAME)
    dimensions = DATASET_SETTINGS[args.dataset]
    for array_name, dimension_name in (
        ("states", "state_dim"), ("observations", "observation_dim"),
        ("controls", "control_dim"),
    ):
        if data.train[array_name].shape[-1] != dimensions[dimension_name]:
            raise ValueError(f"Capacity profile {args.dataset}: {dimension_name} changed")
    config = run.model_config(data, args)
    expected = dict(hidden_size=profile.hidden_size, hidden_depth=profile.hidden_depth,
                    flow_layers=profile.flow_layers, context_dim=profile.context_dim)
    if args.method == "normflowsde":
        resolved = dict(hidden_size=config["d_hidden"], hidden_depth=config["depth"],
                        flow_layers=config["n_realnvp_layers"], context_dim=config["d_c"])
        diffusion_width = (profile.hidden_size if profile.diffusion_hidden_size is None
                           else profile.diffusion_hidden_size)
        if config["d_hidden_diffusion"] != diffusion_width or config["d_hidden_enc"] != profile.context_dim:
            raise ValueError("Capacity profile NormFlowSDE encoder/diffusion width changed")
    elif args.method == "nf_dpf":
        resolved = dict(hidden_size=config["hidden_size"], hidden_depth=config["hidden_depth"],
                        flow_layers=config["num_flow_layers"], context_dim=args.context_dim)
    else:
        resolved = {name: config[name] for name in expected}
        if config["encoder_hidden"] != profile.context_dim:
            raise ValueError("Capacity profile observation encoder width changed")
        causal = config.get("encoder_direction", "bidirectional") == "causal"
        if causal != (profile.name == CAUSAL_PROFILE_NAME):
            raise ValueError("The causal capacity profile requires --encoder-direction causal, and vice versa")
    if resolved != expected:
        raise ValueError(f"Capacity profile architecture changed: expected {expected}, got {resolved}")
    measured = count_parameters(model)
    if measured != profile.expected_parameters:
        raise ValueError(f"Capacity profile parameter count changed: expected "
                         f"{profile.expected_parameters}, got {measured}")
    deviation = measured / profile.target_parameters - 1
    if abs(deviation) > TARGET_TOLERANCE:
        raise ValueError(f"Capacity profile misses its target by {deviation:.2%}")
    return dict(**profile.to_dict(), parameter_count=measured,
                deviation_from_target=deviation,
                count_definition="optimizer-selected inexact array elements")


def validate_dataset_counts(dataset: str, counts: dict[str, int],
                            profile: str = PROFILE_NAME) -> float:
    """Require all four measured counts to share the declared dataset budget."""
    if set(counts) != set(METHODS):
        raise ValueError("Capacity comparison requires exactly all four methods")
    for method, measured in counts.items():
        method_profile = capacity_profile(dataset, method, profile)
        if measured != method_profile.expected_parameters:
            raise ValueError(f"Capacity profile parameter count changed for {dataset}/{method}")
        if abs(measured / method_profile.target_parameters - 1) > TARGET_TOLERANCE:
            raise ValueError(f"Capacity profile misses target for {dataset}/{method}")
    ratio = max(counts.values()) / min(counts.values())
    if ratio >= MAX_DATASET_RATIO:
        raise ValueError(f"Capacity spread exceeds {MAX_DATASET_RATIO}: {ratio}")
    return ratio


def canonical_model(dataset: str, method: str, profile: str = PROFILE_NAME):
    """Construct a real model using tiny arrays with audited channel counts."""
    from . import run

    profile_name = profile
    profile = capacity_profile(dataset, method, profile_name)
    settings = DATASET_SETTINGS[dataset]
    data = SimpleNamespace(
        protocol=dict(hidden_size=profile.hidden_size, hidden_depth=profile.hidden_depth,
                      flow_layers=profile.flow_layers),
        train={"states": np.zeros((1, 2, settings["state_dim"]), dtype=np.float32),
               "observations": np.zeros((1, 1, settings["observation_dim"]), dtype=np.float32),
               "controls": np.zeros((1, 1, settings["control_dim"]), dtype=np.float32)},
        dt=settings["dt"],
    )
    args = run.parse_args(["--dataset", dataset, "--method", method,
                           "--output", "/tmp/normflowsde-capacity-verification-unused",
                           "--capacity-profile", profile_name,
                           *capacity_arguments(dataset, method, profile_name)])
    return run.build_model(data, args), data, args


def verified_plan(datasets=DATASETS, profile: str = PROFILE_NAME) -> dict:
    """Actually build and count every selected dataset/method combination."""
    rows, ratios = [], {}
    for dataset in datasets:
        counts = {}
        for method in METHODS:
            model, data, args = canonical_model(dataset, method, profile)
            row = validate_capacity(model, data, args)
            rows.append(row)
            counts[method] = row["parameter_count"]
        ratios[dataset] = validate_dataset_counts(dataset, counts, profile)
    return dict(schema_version=1, profile=profile, trained=False,
                target_tolerance=TARGET_TOLERANCE, max_dataset_ratio=MAX_DATASET_RATIO,
                dataset_max_min_ratios=ratios, runs=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory for capacity_plan.json and .csv")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--profile", choices=PROFILE_NAMES, default=PROFILE_NAME)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a new capacity plan directory: {args.output}")
    plan = verified_plan(args.datasets, args.profile)
    if args.output is None:
        print(json.dumps(plan, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "capacity_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    with (args.output / "capacity_plan.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan["runs"][0]))
        writer.writeheader()
        writer.writerows(plan["runs"])
    print(f"Verified {len(plan['runs'])} model architectures in {args.output}; no training performed.")


if __name__ == "__main__":
    main()
