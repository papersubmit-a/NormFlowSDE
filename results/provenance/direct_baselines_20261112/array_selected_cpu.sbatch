#!/bin/bash
#SBATCH --job-name=dbenc_selcpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.err
# Seeds 1 and 2 of the validation-selected configurations (select.py -> selected.json). One line of $TASKS per task:
# <method> <dataset> <seed> <TAG> <extra args>. Output results/<method><TAG>/<dataset>/seed<k>/run (seed 0 = the sweep run).
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112
cd /home/anon/NormFlowSDE
read -r METHOD DATASET SEED TAG EXTRA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$P/$TASKS")
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TAG="$TAG" \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash "$P/run_task.sh" "$METHOD" "$DATASET" "$SEED" $EXTRA
