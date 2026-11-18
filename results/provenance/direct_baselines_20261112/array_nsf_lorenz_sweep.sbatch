#!/bin/bash
#SBATCH --job-name=dbenc_nsfsweep
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-5
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.err
# Latent NSF on Lorenz-96, seed 0: validation sweep of the beta-NELBO weight beta (Kiyohara et al. Eq. 16) and the
# posterior step form. The default (beta 1, residual) selected the first checkpoint in every seed (validation RMSE
# 4.5 at 500 updates -> 15-18 at the end while all training terms improved). Selection: lowest validation RMSE.
# One line of tasks_nsf_lorenz_sweep.txt per task: <TAG> <extra args>. Output results/latent_nsf<TAG>/lorenz96/seed0/run.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112
cd /home/anon/NormFlowSDE
read -r TAG EXTRA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$P/tasks_nsf_lorenz_sweep.txt")
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
# shellcheck disable=SC2086
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TAG="$TAG" \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash "$P/run_task.sh" latent_nsf lorenz96 0 $EXTRA
