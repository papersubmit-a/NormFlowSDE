#!/bin/bash
#SBATCH --job-name=nsf_beta0_seeds
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-1
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/%x_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/%x_%A_%a.err
# Latent NSF v3, Lorenz96, beta=0 (validation-selected), seeds 1-2.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110
cd /home/anon/NormFlowSDE
read -r METHOD DATASET SEED BETA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$P/tasks_nsfbeta0_seeds.txt")
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TAG=_beta$BETA \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash "$P/run_task.sh" "$METHOD" "$DATASET" "$SEED" --beta "$BETA"
