# Seed-0 validation grid of the direct baselines (2026-11-12)

Encoder fixed to NormFlowSDE's (WindowEncoder). One grid per method, identical on all four datasets; the selected
configuration per method and dataset is the one with the lowest best_validation_rmse (all grid points incl. the default
run results/<method>/<dataset>/seed0). Grid points run in the first pendulum / Lorenz-96 sweeps outside this grid
(latent NSF Euler step, SigKer bandwidth 0.3 and lr 1e-4) are also eligible.

| method | grid | default |
|---|---|---|
| latent_nsf | beta in {1, 0.1, 0.01, 0} x lr in {3e-4, 1e-3}, residual posterior step | beta 1, lr 3e-4 |
| latent_sde | beta in {1, 0.1, 0.01} x lr in {3e-4, 1e-3} | beta 1, lr 3e-4 |
| sigker_sde | RBF bandwidth in {sqrt(d), 1.0, 0.5} x lr in {3e-4, 1e-3} | sqrt(d), lr 3e-4 |

Launch: array_sweep.sbatch (CPU) / array_sweep_gpu.sbatch (A30; SigKer on Lorenz-96, ALOHA, REASSEMBLE) with
TASKS=tasks_grid_{cpu,gpu}_<dataset>.txt. Outputs results/<method>_sweep/<config>/<dataset>/seed0/run.
