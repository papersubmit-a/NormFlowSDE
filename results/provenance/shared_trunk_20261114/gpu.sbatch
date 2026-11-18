#!/bin/bash
#SBATCH --job-name=trunk_gpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=24G
#SBATCH --time=14:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/gpu_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/gpu_%j.out
# Shared-trunk campaign, GPU part (2026-11-14): NormFlowSDE (stage 1 and stage 2 need a GPU) and NF-DPF.
# TASK_GROUPS is a space-separated list of method:dataset; each group runs its seeds side by side on the one A30
# (three processes, as in the earlier A30 sweeps), groups one after another. A finished stage is skipped, so the
# job can be resubmitted after a failure or a time limit.
#   sbatch --export=ALL,TASK_GROUPS="normflowsde:pendulum normflowsde:lorenz96 nf_dpf:pendulum normflowsde:aloha" gpu.sbatch
#   sbatch --export=ALL,TASK_GROUPS="normflowsde:reassemble" gpu.sbatch
#   SMOKE=1 SEEDS=0 sbatch --export=ALL,SMOKE,SEEDS,TASK_GROUPS="..." gpu.sbatch      a few updates per stage
# Times of the previous models on an A30 with three seeds side by side: pendulum 0.4 h, Lorenz-96 about 0.3 h (L40S),
# ALOHA about 2 h, REASSEMBLE about 2.5 h, NF-DPF about 0.7 h.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; P=$ROOT/results/provenance/shared_trunk_20261114; cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
task() {
  /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 SMOKE=${SMOKE:-0} \
    NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash $P/run_task.sh "$@"
}
failed=0
for group in $TASK_GROUPS; do
  method=${group%%:*}; dataset=${group##*:}
  echo "$(date -Is) group $method $dataset seeds ${SEEDS:-0 1 2}"
  pids=()
  for seed in ${SEEDS:-0 1 2}; do
    task $method $dataset $seed > $P/logs/gpu_${SLURM_JOB_ID}_${method}_${dataset}_seed$seed.out 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || { failed=1; echo "$(date -Is) a task of $group failed"; }; done
done
echo "$(date -Is) gpu job done failed=$failed"
exit $failed
