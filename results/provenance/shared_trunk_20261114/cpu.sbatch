#!/bin/bash
#SBATCH --job-name=trunk_cpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/cpu_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/cpu_%A_%a.out
# Shared-trunk campaign, CPU part (2026-11-14): the methods that need no GPU, one (method, dataset, seed) per array
# task; line N+1 of tasks_cpu.txt is array task N.
#   tasks 0-17   pendulum: NSF-PF, FlowSDE-PF, NormFlowSDE-PF, latent NSF, latent SDE, SigKer SDE (models of 10k
#                parameters; 4 CPUs as direct_baselines_20261112/array_selected_cpu.sbatch)
#   tasks 18-44  Lorenz-96, ALOHA, REASSEMBLE: NSF-PF, FlowSDE-PF, NormFlowSDE-PF (120k-365k parameters; submit with
#                more cores and time)
#   tasks 45-56  NormFlowSDE on the four datasets (CPU with --allow-cpu when the GPU slots are not available)
#   tasks 57-68  NF-DPF on the four datasets
#   sbatch --array=0-17 cpu.sbatch
#   sbatch --array=18-44 --time=3-00:00:00 cpu.sbatch
#   other CPU partitions: add --partition=a30 --qos=a30-cpu  or  --partition=pro6000 --qos=pro6000-cpu
#   SMOKE=1 sbatch --export=ALL,SMOKE --array=0,3,6,9,12,15 cpu.sbatch     one seed of every method, a few updates
# A finished stage is skipped, so a task can be resubmitted after a failure or a time limit.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; P=$ROOT/results/provenance/shared_trunk_20261114; cd $ROOT
read -r METHOD DATASET SEED < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $P/tasks_cpu.txt)
[ -n "${SEED:-}" ] || { echo "no task $SLURM_ARRAY_TASK_ID"; exit 2; }
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=1 SMOKE=${SMOKE:-0} \
  NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash $P/run_task.sh "$METHOD" "$DATASET" "$SEED"
