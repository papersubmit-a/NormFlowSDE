#!/bin/bash
#SBATCH --job-name=dbenc_sweeppro
#SBATCH --account=ACCOUNT
#SBATCH --partition=pro6000
#SBATCH --qos=pro6000-gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112/logs/%x_%A_%a.err
# GPU (RTX Pro 6000) variant, 3 h limit (node reserved on weekdays 08-19), otherwise identical to the A30 variant for SigKer on Lorenz-96 and the robot datasets. Seed-0 validation sweeps of the direct baselines' own hyperparameters (encoder fixed to NormFlowSDE's).
#   sbatch --array=0-<n-1> --export=ALL,TASKS=<file>,DATASET=<dataset> array_sweep.sbatch
# One line of $TASKS per task: <method> <TAG> <extra args>. Output results/<method><TAG>/<dataset>/seed0/run.
# Selection: lowest best_validation_rmse among the sweep and the default run (results/<method>/<dataset>/seed0).
# Pendulum (tasks_pendulum_sweep.txt): with the shared 16-unit encoder, latent NSF degraded after ~3000 updates as
# the KL weight ramped up (val 0.168 -> 0.219), latent SDE plateaued at 0.112 with 0.98 coverage, and SigKer
# (bandwidth sqrt(d), never tuned) degraded after 3600 updates (0.182 -> 0.244).
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261112
cd /home/anon/NormFlowSDE
read -r METHOD TAG EXTRA < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$P/$TASKS")
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
# shellcheck disable=SC2086
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TAG="$TAG" \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash "$P/run_task.sh" "$METHOD" "$DATASET" 0 $EXTRA
