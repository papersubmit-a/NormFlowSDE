#!/bin/bash
#SBATCH --job-name=direct_smoke_v2
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/smoke_v2_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/smoke_v2_%j.err
# Smoke: Latent NSF with rollout supervision, and SigKernel SDE at the smallest sweep bandwidth (0.1), all datasets.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110
cd /home/anon/NormFlowSDE
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
RUN=/home/anon/bayesinf_jax/container/run_in_container.sh
ENV=(env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu
     OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax)
rm -rf results/smoke_direct_20261110/latent_nsf results/smoke_direct_20261110/sigker_sde_sigma0.1
for d in pendulum lorenz96 aloha reassemble; do
  "$RUN" -- "${ENV[@]}" bash "$P/run_task.sh" latent_nsf $d 0 --smoke
  "$RUN" -- "${ENV[@]}" TAG=_sigma0.1 bash "$P/run_task.sh" sigker_sde $d 0 --smoke --sig-sigma 0.1
done
echo SMOKE_V2_OK
