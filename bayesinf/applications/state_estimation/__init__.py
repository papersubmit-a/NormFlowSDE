"""
State estimation with normalizing flows.

- AmortizedStateFlow: Amortized Bayesian state inference; encoder + flow over
  state sequences with standard or time-scaled Gaussian latent priors.
- NormFlowSDE (RealNVPFlowSDE): causal preceding-window conditioning, drift
  and invertible transition noise, with the manuscript composite objective.
- ObservationConditionedNSF/FlowSDE: direct state generators with full-sequence
  measurement encoding; historical transition-only models remain available.
- LatentSSMFlow: Linear Gaussian SSM in latent z-space with a conditional
  normalizing flow decoder; exact MLE training via change of variables.
- NeuralStochasticFlow: solver-free finite-time conditional transition flow.
"""

from bayesinf.applications.state_estimation.amortized_state_flow import (
    GRUSequenceEncoder,
    TransformerSequenceEncoder,
    StateConditionEncoder,
    AmortizedStateFlow,
    fm_loss,
    sbm_loss,
    state_flow_loss,
)
from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPFlowSDE,
    NICEDiffusionNet,
)
from bayesinf.applications.state_estimation.fm_flow_sde import (
    FMFlowSDE,
    FlowMatchingSDE,
    FlowSDEConfig,
    FlowSDELosses,
    GaussianBridge,
    flow_sde_loss,
    initialise_optimizer as initialise_fm_sde_optimizer,
    make_train_step as make_fm_sde_train_step,
    make_transition_pairs as make_fm_sde_transition_pairs,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    VirtualBrownianTree,
    get_solver,
    list_solvers,
    integrate_sde,
    integrate_sde_multiple_particles,
)
from bayesinf.applications.state_estimation.stochastic_interpolant import (
    StochasticInterpolantSE,
    make_step_si,
    make_step_si_ms,
)
from bayesinf.applications.state_estimation.latent_ssm_flow import (
    LatentSSMFlow,
    build_latent_ssm_flow,
    _flow_fwd_single,
    _flow_inv_single,
)
from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    ConsistencyLosses,
    NSF,
    NSFBridge,
    NSFConfig,
    NSFConsistencyConfig,
    NSFConsistencyTrainingResult,
    NSFTrainingConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    evaluate_ou_moments,
    fit_normalization,
    flow_consistency_loss,
    make_step_nsf,
    make_step_nsf_bridge,
    make_step_nsf_consistency,
    make_transition_pairs,
    sample_uniform_consistency_triplets,
    sample_ou_transition_data,
    train_nsf_nll,
    train_nsf_with_consistency,
)

__all__ = [
    "RealNVPFlowSDE",
    "NICEDiffusionNet",
    "FMFlowSDE",
    "FlowMatchingSDE",
    "FlowSDEConfig",
    "FlowSDELosses",
    "GaussianBridge",
    "flow_sde_loss",
    "initialise_fm_sde_optimizer",
    "make_fm_sde_train_step",
    "make_fm_sde_transition_pairs",
    "VirtualBrownianTree",
    "get_solver",
    "list_solvers",
    "integrate_sde",
    "integrate_sde_multiple_particles",
    "StochasticInterpolantSE",
    "make_step_si",
    "make_step_si_ms",
    "GRUSequenceEncoder",
    "TransformerSequenceEncoder",
    "StateConditionEncoder",
    "AmortizedStateFlow",
    "fm_loss",
    "sbm_loss",
    "state_flow_loss",
    "LatentSSMFlow",
    "build_latent_ssm_flow",
    "_flow_fwd_single",
    "_flow_inv_single",
    "ConsistencyLosses",
    "NSF",
    "NSFBridge",
    "NSFConfig",
    "NSFConsistencyConfig",
    "NSFConsistencyTrainingResult",
    "NSFTrainingConfig",
    "NeuralStochasticFlow",
    "OrnsteinUhlenbeck",
    "evaluate_ou_moments",
    "fit_normalization",
    "flow_consistency_loss",
    "make_step_nsf",
    "make_step_nsf_bridge",
    "make_step_nsf_consistency",
    "make_transition_pairs",
    "sample_uniform_consistency_triplets",
    "sample_ou_transition_data",
    "train_nsf_nll",
    "train_nsf_with_consistency",
]
from bayesinf.applications.state_estimation.filtering_baselines import (
    FilterResult,
    ParticleFilterConfig,
    bootstrap_nsf_particle_filter,
    bootstrap_particle_filter,
    bootstrap_particle_filter_nsf,
    extended_kalman_filter,
    reconstruct_particle_trajectories,
    unscented_kalman_filter,
)
from bayesinf.applications.state_estimation.observation_models import (
    IdentityGaussianObservation,
    IdentityGaussianObservationModel,
    LinearGaussianObservation,
    LinearGaussianObservationModel,
    NonlinearGaussianObservation,
    NonlinearGaussianObservationModel,
    ObservationModel,
)
from bayesinf.applications.state_estimation.transition_benchmarks import (
    BenchmarkTrainingConfig,
    NSFTransitionAdapter,
    NormFlowSDETransitionAdapter,
    TransitionKernel,
    count_trainable_parameters,
    ensemble_transition_metrics,
    train_transition_kernel_nll,
)
from bayesinf.applications.state_estimation.controlled_systems import (
    ControlledDuffingOscillator,
    ControlledStochasticPendulum,
    sample_controlled_duffing_data,
    sample_controlled_pendulum_data,
)

__all__ += [
    "FilterResult",
    "ParticleFilterConfig",
    "bootstrap_nsf_particle_filter",
    "bootstrap_particle_filter",
    "bootstrap_particle_filter_nsf",
    "extended_kalman_filter",
    "reconstruct_particle_trajectories",
    "unscented_kalman_filter",
    "IdentityGaussianObservation",
    "IdentityGaussianObservationModel",
    "LinearGaussianObservation",
    "LinearGaussianObservationModel",
    "NonlinearGaussianObservation",
    "NonlinearGaussianObservationModel",
    "ObservationModel",
    "NSFTransitionAdapter",
    "NormFlowSDETransitionAdapter",
    "TransitionKernel",
    "count_trainable_parameters",
    "ensemble_transition_metrics",
    "BenchmarkTrainingConfig",
    "train_transition_kernel_nll",
    "ControlledDuffingOscillator",
    "ControlledStochasticPendulum",
    "sample_controlled_duffing_data",
    "sample_controlled_pendulum_data",
]

from bayesinf.applications.state_estimation.normalizing_flow_dpf import (
    ConditionalNormalizingFlow,
    NFDPFConfig,
    NFDPFResult,
    NormalizingFlowDPF,
    nf_dpf_component_loss,
    nf_dpf_loss,
)

__all__ += [
    "ConditionalNormalizingFlow",
    "NFDPFConfig",
    "NFDPFResult",
    "NormalizingFlowDPF",
    "nf_dpf_component_loss",
    "nf_dpf_loss",
]

from bayesinf.applications.state_estimation.observation_conditioned_baselines import (
    ObservationConditionedConfig,
    ObservationConditionedNSF,
    ObservationConditionedFlowSDE,
)

__all__ += [
    "ObservationConditionedConfig",
    "ObservationConditionedNSF",
    "ObservationConditionedFlowSDE",
]
