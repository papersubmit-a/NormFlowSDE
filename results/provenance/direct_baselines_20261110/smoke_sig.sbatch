#!/bin/bash
#SBATCH --job-name=sigker_smoke
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/smoke_sig_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/smoke_sig_%j.err
# SigKernelSDE: solver unit checks, smoke runs on all datasets, and timing of 20 real updates per dataset.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110
cd /home/anon/NormFlowSDE
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
RUN=/home/anon/bayesinf_jax/container/run_in_container.sh
ENV=(env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu
     OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax)
"$RUN" -- "${ENV[@]}" /opt/venv/bin/python "$P/check_sigkernel.py"
rm -rf results/smoke_direct_20261110/sigker_sde results/smoke_direct_20261110/timing_sigker
for d in pendulum lorenz96 aloha reassemble; do
  "$RUN" -- "${ENV[@]}" bash "$P/run_task.sh" sigker_sde $d 0 --smoke
  # timing: 20 real updates at the full batch / sample count (validation on 4 windows only)
  "$RUN" -- "${ENV[@]}" /opt/venv/bin/python -m experiments.normflowsde_comparison.train_direct_baseline \
    --method sigker_sde --dataset $d --data-cache results/datasets/$d \
    --output results/smoke_direct_20261110/timing_sigker/$d --hidden-size 140 --hidden-depth 3 \
    --steps 20 --eval-every 10 --validation-windows 4 --validation-draws 8 | grep -E '"step": (10|20)' | cut -c1-400 || true
done
echo SMOKE_SIG_OK
