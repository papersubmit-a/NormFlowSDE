#!/bin/bash
#SBATCH --job-name=trunk_eval
#SBATCH --account=ACCOUNT
#SBATCH --partition=pro6000
#SBATCH --qos=pro6000-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/eval_stopped_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/eval_stopped_%j.out
# Shared-trunk campaign (2026-11-15): the ten CPU particle-filter tasks still training after 13 h (jobs 35622_18/19,
# 35640_24-26, 35641_30-32, 35658_42/44; 4-32 h left at 3-14 s per update) were cancelled and are scored here from
# the best-validation checkpoint they had written (evaluate_stopped.py). Then run_task.sh completes what follows the
# stopped stage: the jitter stage of NSF-PF Lorenz-96 seeds 0-1, and for FlowSDE-PF ALOHA the third diffusion level
# (D=0.05, never started; trained in full here on the GPU) and the selection over the three levels.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; P=$ROOT/results/provenance/shared_trunk_20261114; B=results/shared_trunk_20261114; cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
incontainer() {
  /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 SMOKE=0 \
    NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax "$@"
}
failed=0
for run in nsf/lorenz96/seed0/train nsf/lorenz96/seed1/train \
           normflowsde_pf/lorenz96/seed0/learned_scale normflowsde_pf/lorenz96/seed1/learned_scale normflowsde_pf/lorenz96/seed2/learned_scale \
           normflowsde_pf/reassemble/seed0/learned_scale normflowsde_pf/reassemble/seed2/learned_scale \
           flowsde/aloha/seed0/D0.03 flowsde/aloha/seed1/D0.03 flowsde/aloha/seed2/D0.03; do
  echo "$(date -Is) score $run"
  incontainer /opt/venv/bin/python $P/evaluate_stopped.py --run $B/$run 2>&1 | grep -v -E "^W0|^I0|WARNING" || true
  [ -f $B/$run/summary.json ] || { failed=1; echo "$(date -Is) FAILED $run"; }
done
pids=()
for task in "nsf lorenz96 0" "nsf lorenz96 1" "flowsde aloha 0" "flowsde aloha 1" "flowsde aloha 2"; do
  set -- $task
  incontainer bash $P/run_task.sh $1 $2 $3 > $P/logs/eval_stopped_${SLURM_JOB_ID}_$1_$2_seed$3.out 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait $pid || { failed=1; echo "$(date -Is) a follow-up task failed"; }; done
echo "$(date -Is) eval job done failed=$failed"
exit $failed
