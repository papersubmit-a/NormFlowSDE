# State-estimation framework

This package contains reusable flow estimators, stochastic solvers, and
classical filters. Dataset-specific training and evaluation entry points live
in direct use-case packages under `experiments`.

`normalizing_flow_dpf.py` adds `NormalizingFlowDPF`, an independent JAX/Equinox
implementation of [Chen and Li's NF-DPF](https://arxiv.org/abs/2403.01499).
It learns conditional flow densities for the transition, observation-informed
proposal, and measurement model; weights include the full transition/measurement
to proposal density ratio. Training supports differentiable regularized optimal
transport resampling and end-to-end supervised filtering loss. See
[`experiments/nf_dpf_state_estimation`](../../../experiments/nf_dpf_state_estimation/README.md)
for matched pendulum, Lorenz96:N12, Aloha, and Reassemble runs.

- `amortized_state_flow.py`, `pointwise_state_flow.py`, `latent_ssm_flow.py`,
  `flow_sde.py`, `neural_stochastic_flow.py`, and `stochastic_interpolant.py`:
  flow-based estimators. `neural_stochastic_flow.py` is the solver-free NSF
  finite-time transition model; unlike `flow_sde.py`, it does not integrate
  learned local SDE dynamics.
- `filtering_baselines.py`: model-agnostic EKF, UKF, and bootstrap particle
  filtering. Callers provide their transition function.
- `stochastic_solvers.py`: reusable stochastic integration.
- `utils/plots.py`: generic estimation diagnostics shared by experiments.

ALOHA insertion and Lorenz-96 datasets, trainers, evaluators, and plotting
utilities are under `experiments/aloha_state_estimation` and
`experiments/lorenz96_state_estimation`.

`experiments/contact_state_estimation` adds a contact-rich robot/object case:
intermittent object observations and hard-to-model impact dynamics make a
smooth physical-model Kalman filter fail, while a learned FlowSDE transition can
be used as the bootstrap particle-filter proposal.

## Neural Stochastic Flow transition API

`NeuralStochasticFlow` follows the state-estimation package's Equinox model
convention while exposing physical-unit finite-time transitions:

```python
from bayesinf.applications.state_estimation import NeuralStochasticFlow

model = NeuralStochasticFlow(state_dim=4, control_dim=2, key=model_key)
log_density = model.log_prob(x_next, x_current, delta_t, control)
samples = model.sample(sample_key, x_current, delta_t, control, num_samples=100)
```

Use `eqx.filter_jit` when compiling a bound model method. Inputs may be
unbatched or have arbitrary matching leading batch axes. `sample` adds a leading
sample axis. Controls represent a piecewise-constant input over one interval;
multi-control rollouts recursively apply one-step transitions.

NLL-only training uses `train_nsf_nll(model, transitions, NSFTrainingConfig())`,
where `transitions` contains `x_current`, `x_next`, and `delta_t`, plus optional
`control` and `t_current`. Reproducible OU reference workflows are available as:

```bash
python -m experiments.nsf_state_estimation.train_ou --output results/nsf_ou/seed0
python -m experiments.nsf_state_estimation.evaluate_ou results/nsf_ou/seed0
```

Milestone 4 adds `NSFBridge` and `train_nsf_with_consistency`. The consistency
trainer supports `one_to_two`, `two_to_one`, and `bidirectional` variational
bounds, exposes independent model/bridge optimizer states, performs a
configurable number of bridge inner updates, and logs every loss component. A
reproducible weight ablation is available as:

```bash
python -m experiments.nsf_state_estimation.ablate_consistency \
  --output results/nsf_ou/consistency --weights 0 0.1 1
```

Consistency training currently supports autonomous dynamics only. A controlled
multi-interval objective needs an explicit control-path representation and is
rejected until that Milestone 6 design is specified.

## NSF particle filtering

Milestone 5 provides identity, linear/partial, and callable nonlinear Gaussian
observation models in `observation_models.py`. `bootstrap_particle_filter_nsf`
uses the NSF transition directly as a batched bootstrap proposal and supports
irregular times, optional controls, missing-observation masks, log-space
weights, ESS-triggered systematic or stratified resampling, weighted posterior
particles, credible intervals, and optional ancestor reconstruction.

```python
from bayesinf.applications.state_estimation import (
    LinearGaussianObservation, ParticleFilterConfig,
    bootstrap_particle_filter_nsf,
)

observation_model = LinearGaussianObservation(H, noise_std=0.2)
posterior = bootstrap_particle_filter_nsf(
    observations, x0, times, filter_key, model, observation_model,
    mask=observation_mask,
    config=ParticleFilterConfig(num_particles=1000),
)
```

The analytic-OU noisy/full and partial-observation benchmark is runnable with:

```bash
python -m experiments.nsf_state_estimation.evaluate_filtering_ou \
  results/nsf_ou/seed0
```

The filter never inserts NSF transition density into bootstrap weights: because
the proposal is the transition prior itself, those proposal/transition factors
cancel and the incremental weight is only the observation likelihood.

## Controlled transition benchmark

`NeuralStochasticFlow.sample_rollout` recursively applies one finite-time NSF
transition per piecewise-constant command. Changing control sequences are never
compressed into a direct long-gap condition. `NSFTransitionAdapter` and
`NormFlowSDETransitionAdapter` expose the same physical-unit `sample`,
`log_prob`, and `sample_rollout` interface; the latter integrates the causal
controlled RealNVP FlowSDE at its configured solver step.

`NormFlowSDETransitionAdapter(..., rollout_mode="itosde")` samples from a
moment-matched Itô surrogate. Its `log_prob` is unavailable because that sampler
has a different law. `euler_reference_log_prob` explicitly scores the original
Euler flow-noise reference kernel; `log_prob` is exact only for the default
Euler `normflowsde` sampler. At each normalized state/control pair it estimates the conditional
mean and covariance of the centered RealNVP diffusion with `ito_moment_samples`,
adds the mean divided by `sqrt(dt)` as residual drift, and uses a Cholesky
covariance factor for Gaussian Euler-Maruyama noise. The default
`rollout_mode="normflowsde"` preserves the original flow-based rollout.

For offline deployment, `DistilledItoSDE` in
`bayesinf.applications.optimal_control.controlled_flow_sde` provides a simple
MLP drift plus MLP diffusion-factor surrogate. `distilled_ito_loss` and
`make_step_distilled_ito` train it from FlowSDE moment targets sampled offline,
so online MPPI can avoid repeated RealNVP moment estimation.

The matched nonlinear benchmark uses a controlled stochastic Duffing oscillator:

```bash
python -m experiments.nsf_state_estimation.benchmark_controlled_nsf_flowsde \
  --output results/nsf_controlled_duffing
```

Its default five-seed protocol uses trajectory-level splits, training-only
normalization, identical minibatches and optimizer budgets, reference simulator
ensembles, and compilation-excluded timing across horizons. It reports exact
one-step NLL, RMSE/MAE, CRPS, energy score, marginal Wasserstein distance,
covariance error, interval coverage/width, recursive accuracy, parameter counts,
and runtime per generated trajectory.

A matched pendulum reconstruction suite, including recursive trajectories and
finite-time conditional-mean vector-field diagnostics, is available as:

```bash
python -m experiments.nsf_state_estimation.run_pendulum_reconstruction \
  --output results/results_state_estimation/pendulum/nsf_pendulum_reconstruction/seed2026
python -m experiments.nsf_state_estimation.run_norm_flow_sde_pendulum_reconstruction \
  --output results/results_state_estimation/pendulum/norm_flow_sde_pendulum_reconstruction_staged/seed2026
python -m experiments.nsf_state_estimation.run_fm_sde_pendulum_reconstruction \
  --output results/results_state_estimation/pendulum/fm_flow_sde_pendulum_reconstruction/seed2026
```

The pendulum uses unwrapped angle coordinates to avoid an artificial density
discontinuity at `+/-pi`; the vector-field diagnostic is a finite-time mean
increment divided by `delta_t` for the finite-time NSF models, not an explicit
learned drift. The defaults are tuned on the validation trajectories for stable
60-step recursive reconstruction; the test trajectories remain evaluation-only.

These commands mark generated metadata private/not approved for redistribution.

Method contracts after the October 2026 audit
-------------------------------------------

- `RealNVPFlowSDE` models a fixed physical step, with transition observations
  `y=(B,T,O)` and training states `x=(B,T+1,D)` including the initial state.
  Euler transition NLL is the default objective; another fixed-step kernel
  requires explicit joint-trajectory energy scoring. A nonlinear transformed
  innovation generally has no grid-refinement SDE limit. `dt_factor` therefore
  cannot change the trained step size. Bidirectional contexts define smoothing.
- `FlowMatchingSDE` trains Gaussian-bridge forward Ito drift (velocity plus
  `D*score`), and simulates diffusion `sqrt(2D)`. This is a Markovian bridge
  projection; recovering arbitrary physical transition laws still depends on
  the bridge/model assumptions. Potential regularizers use an empirical bridge
  measure by default. Physical Boltzmann EPR estimation requires the explicit
  `boltzmann_importance` option, which corrects for the bridge proposal density
  and can have high importance-weight variance.
- `LatentSSMFlow` normalizing-flow variants retain the correlated Gaussian SSM;
  per-state conditional FM uses independent standard Gaussian innovations so
  its source conditioned on the generated predecessor matches training. All
  segment routines include the final partial segment, and the entropy term
  uses the same prior restarts as the likelihood and sampler.
- State stochastic-interpolant drift uses `v + eta * scaled_score` and diffusion
  `sqrt(2 * eta * sigma)`. Segment sampling draws the trained Gaussian source
  independently in each segment. It is explicitly an approximation that does
  not reproduce dependence across segment boundaries; use full-trajectory
  sampling for the model's joint law.
- Reusable solvers distinguish Ito Euler/diagonal Milstein from Stratonovich
  Heun/midpoint. `reversible_heun` is a legacy alias for stateless Heun, without
  a reversibility claim. `adaptive_euler` is a legacy alias for Brownian-tree
  adaptive diagonal Ito Milstein (`adaptive_milstein`), because naive adaptive
  Euler does not have the required SDE convergence guarantee.
