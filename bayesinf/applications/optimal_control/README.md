# Optimal-control framework

This package contains reusable architecture and controller implementations.
It intentionally contains no robot, vehicle, or dataset-specific workflow.

- `sampling_mpc.py`: MPPI and cross-entropy stochastic MPC.
- `flow_sde_mppi.py`: stochastic MPC rollout through controlled FlowSDEs.
- `gradient_based_mpc.py`: projected-gradient MPC, including Brax support.
- `controlled_flow_sde.py` and `second_order_flow_sde.py`: controlled
  probabilistic forward-dynamics architectures.
- `conditional_flow_sde.py`: compact conditional RealNVP FlowSDE and Gaussian
  dynamics architectures used by the interactive-pushing experiment.
- `gru_latent_ssm_flow.py` and `controlled_latent_ssm_flow.py`: controlled
  latent-state flow architectures.
- `amortized_seq_flow.py`: bounded amortized action-sequence flow.
- `cond_stoch_interpolant.py`, `dppo_jax.py`, and
  `dppo_diffusion_jax.py`: generative policies and DPPO optimization.
- `csi_mppi.py`: CSI geometric path proposals, validity-based route selection,
  and slower replanning around a fast feedback tracker.
- `trajectory_mppi.py`: bounded MPPI tracking with configurable dynamics,
  reference coordinates, and obstacle costs. See the self-contained
  [CSI + MPPI navigation experiment](../../../experiments/csi_mppi_navigation/README.md)
  for demonstration generation, training, checkpoints, and evaluation. The
  [real-map experiment](../../../experiments/csi_mppi_realmap/README.md) adds
  map-conditioned local proposals, shared A* routes, actuator lag and slip,
  held-out building geometry, and separate CPU/L40S latency measurements.

Concrete ALOHA, Jackal, lane-keeping, SO-100, and MuJoCo-pusher workflows are
direct packages under `experiments`.
