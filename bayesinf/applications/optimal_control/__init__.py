"""Reusable learned-dynamics, policy, and MPC architecture."""

from .amortized_seq_flow import AmortizedSeqFlow
from .sampling_mpc import (
    CEMStochasticMPC, MPPIController, SamplingMPCConfig,
)
from .gradient_based_mpc import BraxGradientMPC, GradientMPC, GradientMPCConfig
from .conditional_flow_sde import ConditionalRealNVP, FlowSDE, GaussianSDE
from .controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
    DistilledItoSDE,
    distilled_ito_loss,
    make_step_distilled_ito,
    normflowsde_ito_targets,
)
from .cond_stoch_interpolant import (
    AnalyticScorePolicy,
    CondStochInterpolantPolicy,
    ConditionalVelocity,
    FlowMatchingPolicy,
    InterpolantSample,
    LearnedScorePolicy,
    MLPContextEncoder,
)
from .csi_mppi import (
    CSIPlannerConfig, CSIMPPIController, CSITrajectoryPlanner,
    CSITrajectoryPolicy, PathSelection,
)
from .trajectory_mppi import TrackingMPPIConfig, TrajectoryMPPIController
from .dppo_jax import (
    BCTrainState,
    DPPOConfig,
    DPPOMetrics,
    DPPOTrainState,
    RolloutBatch,
    ValueCritic,
    bc_update,
    dppo_loss,
    dppo_update,
    generalized_advantage_estimate,
    make_bc_state,
    make_dppo_state,
)
from .dppo_diffusion_jax import (
    DPPODiffusionMLP,
    DPPODiffusionPolicy,
    DiffusionSample,
    ScoreBasedPolicy,
    cosine_beta_schedule,
)

# Some experiment modules are optional and are not present in lightweight
# checkouts. Keep the package importable so independent runners still work.
try:
    from .controlled_latent_ssm_flow import ControlledLatentSSMFlow
except ModuleNotFoundError:
    ControlledLatentSSMFlow = None
try:
    from .gru_latent_ssm_flow import GRULatentSSMFlow
except ModuleNotFoundError:
    GRULatentSSMFlow = None

__all__ = [
    "AmortizedSeqFlow",
    "ControlledLatentSSMFlow", "CEMStochasticMPC", "MPPIController",
    "SamplingMPCConfig", "GradientMPC", "GradientMPCConfig", "BraxGradientMPC",
    "ConditionalRealNVP", "FlowSDE", "GaussianSDE",
    "ControlledRealNVPFlowSDE", "DistilledItoSDE",
    "distilled_ito_loss", "make_step_distilled_ito",
    "normflowsde_ito_targets",
    "GRULatentSSMFlow",
    "CondStochInterpolantPolicy", "FlowMatchingPolicy", "AnalyticScorePolicy",
    "LearnedScorePolicy", "MLPContextEncoder", "ConditionalVelocity",
    "InterpolantSample",
    "BCTrainState", "DPPOConfig", "DPPOMetrics", "DPPOTrainState",
    "RolloutBatch", "ValueCritic", "bc_update", "dppo_loss", "dppo_update",
    "generalized_advantage_estimate", "make_bc_state", "make_dppo_state",
    "DPPODiffusionMLP", "DPPODiffusionPolicy", "DiffusionSample",
    "ScoreBasedPolicy", "cosine_beta_schedule",
    "CSIPlannerConfig", "CSIMPPIController", "CSITrajectoryPlanner",
    "CSITrajectoryPolicy", "PathSelection", "TrackingMPPIConfig",
    "TrajectoryMPPIController",
]
