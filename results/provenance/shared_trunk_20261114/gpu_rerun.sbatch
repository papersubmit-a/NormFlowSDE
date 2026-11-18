#!/bin/bash
#SBATCH --job-name=trunk_rerun
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=24G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/rerun_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/rerun_%j.out
# Shared-trunk campaign, GPU rerun (2026-11-15) of the particle-filter tasks that could not finish on the CPU
# allocations of 2026-11-14 (cpu.sbatch, 2 cores / 8 GB per task):
#   * FlowSDE-PF on Lorenz-96 and REASSEMBLE: 10-17 s per update on 2 cores, three diffusion levels in sequence
#     = about 150 h against the 72 h limit (jobs 35640_21-23, 35658_39/41 cancelled at level 1; the earlier
#     selected runs took 1.7-1.8 h per level on a GPU).
#   * killed for memory at 8 GB: NSF-PF REASSEMBLE seeds 0-2, FlowSDE-PF REASSEMBLE seed 1, NormFlowSDE-PF
#     REASSEMBLE seed 1 (during training), NF-DPF stage 2 Lorenz-96 seeds 0, 2 and REASSEMBLE seeds 0-2 (in the
#     test evaluation after the last update; stage 1 is complete and is skipped).
#   * NSF-PF Lorenz-96 seed 2: array task 20 was submitted twice (35622_20, 35640_20) and both processes wrote
#     the same train/ directory (history.jsonl interleaved, model.eqx overwritten by either). Both cancelled, the
#     directory kept as train.two_writers.<time>, retrained here.
# Same run_task.sh, same recipes; only the device changes. TASK_SETS is a space-separated list of
# method:dataset:seed[+seed...]; the seeds of a group run side by side on the one GPU (as gpu.sbatch), groups one
# after another. A finished stage is skipped, so the job can be resubmitted after a failure or a time limit.
#   sbatch --export=ALL,TASK_SETS="flowsde:lorenz96:0+1+2" gpu_rerun.sbatch   (no commas: --export splits on them)
#   other GPU partition: add --partition=pro6000 --qos=pro6000-gpu
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; P=$ROOT/results/provenance/shared_trunk_20261114; cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
task() {
  /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 SMOKE=0 \
    NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax bash $P/run_task.sh "$@"
}
failed=0
for group in $TASK_SETS; do
  IFS=: read -r method dataset seeds <<< "$group"
  echo "$(date -Is) group $method $dataset seeds $seeds"
  pids=()
  for seed in ${seeds//+/ }; do
    task $method $dataset $seed > $P/logs/rerun_${SLURM_JOB_ID}_${method}_${dataset}_seed$seed.out 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait $pid || { failed=1; echo "$(date -Is) a task of $group failed"; }; done
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
done
echo "$(date -Is) rerun job done failed=$failed"
exit $failed
