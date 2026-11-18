#!/bin/bash
#SBATCH --job-name=dbenc_gpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-8
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.err
# SigKer SDE on REASSEMBLE, ALOHA, Lorenz-96 (tasks 0-8 of tasks.txt), moved from CPU array 35192 to one A30 each
# (CPU throughput 0.4 / 0.4 / 1 updates per second, i.e. 5-8 h). Same run_task.sh and settings; only the device differs.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112
cd /home/anon/NormFlowSDE
read -r METHOD DATASET SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$P/tasks_gpu.txt")
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash "$P/run_task.sh" "$METHOD" "$DATASET" "$SEED"
