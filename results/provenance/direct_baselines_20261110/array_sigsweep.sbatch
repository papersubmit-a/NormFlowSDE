#!/bin/bash
#SBATCH --job-name=sigker_sweep
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-11
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/%x_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/%x_%A_%a.err
# SigKernel SDE RBF-bandwidth sweep (seed 0; tasks_sigsweep.txt: method dataset seed sigma). Same budget, widths,
# samples and RMSE checkpoint selection as the sigma=sqrt(D) runs; only --sig-sigma changes.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110
cd /home/anon/NormFlowSDE
read -r METHOD DATASET SEED SIGMA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$P/tasks_sigsweep.txt")
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TAG=_sigma$SIGMA \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash "$P/run_task.sh" "$METHOD" "$DATASET" "$SEED" --sig-sigma "$SIGMA"
