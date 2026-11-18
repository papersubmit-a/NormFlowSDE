"""Explicit, limited transfer of historical NormFlowSDE training settings.

The opt-in legacy profile preserves the audited model, observation access,
segment convention and standard energy score. It is a scientific variant,
not an exact replay of the older state-estimation experiments.
"""
from __future__ import annotations


LEGACY_PROFILE = "legacy-compatible-v1"
PROFILES = ("audited", LEGACY_PROFILE)
DATASETS = ("lorenz96", "pendulum", "aloha", "reassemble")
METHODS = ("normflowsde", "nf_dpf", "nsf", "flowsde")

# These are rollout ensemble sizes M, independent of the existing segment
# Monte Carlo size K (--training-draws). No architecture or budget is changed.
_SETTINGS = {
    "lorenz96": dict(continuity_weight=0.1, rollout_weight=1.0,
                     rollout_draws=8, rollout_every=2),
    "pendulum": dict(continuity_weight=1.0, rollout_weight=1.0,
                     rollout_draws=4, rollout_every=5),
    "aloha": dict(continuity_weight=1.0, rollout_weight=0.0,
                  rollout_draws=4, rollout_every=5),
    "reassemble": dict(continuity_weight=1.0, rollout_weight=0.5,
                       rollout_draws=4, rollout_every=2),
}

_EVIDENCE = {
    "lorenz96": dict(
        level="checkpoint_filename",
        source="results/results_state_estimation/lorenz96_N12/flowsde/"
               "RealNVPFlowSDE_lorenz96_N12_h100_obs2_std0.5_pwheun_custom_"
               "sn0.025_rw1_energyscore8_re2_rwnd100_cw0.1_tnll0.25_"
               "ms_curriculum_sched10to99.pt",
        establishes="Tags indicate continuity 0.1, rollout weight 1, "
                    "rollout ensemble 8 and cadence 2; no training log certifies them."),
    "pendulum": dict(
        level="direct_saved_metadata_different_objective",
        source="results/results_state_estimation/pendulum/gaussian/"
               "norm_flow_sde_pendulum_reconstruction_continuity/seed2026/metrics.json",
        establishes="Saved settings describe a staged controlled transition model. "
                    "They do not establish shooting-boundary or energy-score settings; "
                    "the audited paper-objective defaults are retained."),
    "aloha": dict(
        level="checkpoint_filename",
        source="results/results_state_estimation/Aloha_sim_insertion/dataset_noisy/"
               "RealNVPFlowSDE_aloha_insertion_h250_s25_std0.1_pwheun_custom_"
               "cw1_tnll1_seg25_cw1_noenergy.pt",
        establishes="Tags indicate continuity 1 and no auxiliary rollout energy. "
                    "Inactive ensemble/cadence values retain audited defaults."),
    "reassemble": dict(
        level="checkpoint_filename",
        source="results/results_state_estimation/REASSEMBLE/realnvp_flowsde_reweighted/"
               "RealNVPFlowSDE_reassemble_h250_s250_obsstd0.1_pwheun_custom_"
               "rw0.5_energyscore4uw0.1_re2_rwu5_rwnd25_cw1_tnll1.pt",
        establishes="Tags indicate continuity 1, rollout weight 0.5, "
                    "rollout ensemble 4 and cadence 2; no training log certifies them."),
}

_COMMON_UNTRANSFERRED = (
    "Architecture widths are handled separately by the selected capacity profile; "
    "old checkpoint scalar counts are not assumed.",
    "Whole-sequence future observations: preserve the preceding-observation encoder.",
    "Legacy optimizer budgets/LR: robot and Lorenz run settings are unrecorded; "
    "preserve the prepared budgets.",
)
_DATASET_UNTRANSFERRED = {
    "lorenz96": (
        "Heun: preserve the paper's Euler update and sampled transition density.",
        "Start perturbation 0.025: preserve exact supplied shooting anchors.",
        "Reference-transition NLL weight 0.25: preserve the generated-transition "
        "entropy term in the paper objective.",
        "Segment curriculum 10 to 99: preserve the explicitly requested segment length 5.",
        "Optional known-physics drift/initial-state network: preserve the audited "
        "architecture and known-initial-state protocol.",
    ),
    "pendulum": (
        "Drift pretraining 2400 updates at 5e-4, followed by 2000 joint updates at "
        "2e-4: the old staged transition objective differs from the paper objective.",
        "Joint drift weight 5 and kinematic continuity weight 2: these are different "
        "losses from reconstruction and shooting-boundary consistency.",
        "Removing the encoder: preserve the audited observation-conditioned model.",
    ),
    "aloha": (
        "Heun/reference-transition NLL weight 1: preserve the paper's Euler update "
        "and generated-transition entropy.",
        "Legacy segment length 25 and learned initial-state network: preserve segment "
        "length 5 and the known-initial-state protocol.",
    ),
    "reassemble": (
        "Heun/reference-transition NLL weight 1: preserve the paper's Euler update "
        "and generated-transition entropy.",
        "Energy-score spread coefficient 0.1: preserve the standard coefficient 0.5.",
        "Rollout window 25: preserve full-horizon distributional supervision.",
        "Warmup 5 epochs: the historical update-equivalent schedule is unrecorded.",
    ),
}


def _validate(dataset: str, method: str, profile: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown alignment dataset: {dataset}")
    if method not in METHODS:
        raise ValueError(f"Unknown alignment method: {method}")
    if profile not in PROFILES:
        raise ValueError(f"Unknown alignment profile: {profile}")


def alignment_arguments(dataset: str, method: str, profile: str = "audited") -> list[str]:
    """Return only opt-in training flags; never override other methods or K."""
    _validate(dataset, method, profile)
    if profile == "audited" or method != "normflowsde":
        return []
    return [token for name, value in _SETTINGS[dataset].items()
            for token in ("--" + name.replace("_", "-"), str(value))]


def alignment_metadata(dataset: str, method: str, profile: str = "audited") -> dict:
    """Describe evidence and exclusions without claiming historical replay."""
    _validate(dataset, method, profile)
    applied = profile == LEGACY_PROFILE and method == "normflowsde"
    result = dict(profile=profile, dataset=dataset, method=method, applied=applied,
                  exact_legacy_replay=False, settings={}, evidence=[], not_transferred=[])
    if not applied:
        result["note"] = "Existing audited training settings are retained."
        return result
    result.update(
        settings=dict(_SETTINGS[dataset]), evidence=[dict(_EVIDENCE[dataset])],
        not_transferred=list(_COMMON_UNTRANSFERRED + _DATASET_UNTRANSFERRED[dataset]),
        note="Limited historical-settings scientific variant; not an exact replay. "
             "Segment Monte Carlo draws K remain unchanged. Rollout cadence 2 differs "
             "from the paper's default cadence 5 where selected. Normalization and "
             "loss definitions differ from the historical objectives, so transferring "
             "a numerical weight does not establish an equivalent contribution.",
    )
    return result
