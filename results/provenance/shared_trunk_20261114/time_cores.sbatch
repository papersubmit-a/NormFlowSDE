#!/bin/bash
#SBATCH --job-name=trunk_timing
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/timing_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/timing_%j.out
# Short CPU timing (2026-11-15) to size the full Lorenz-96 retrains of NSF-PF and NormFlowSDE-PF: seconds per update
# and per validation at 2, 4, 8 and 16 cores (taskset on the allocated cores), time_cores.py.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; P=$ROOT/results/provenance/shared_trunk_20261114; B=results/shared_trunk_20261114; cd $ROOT
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
CPUS=($(python3 -c "import os; print(*sorted(os.sched_getaffinity(0)))"))
echo "host $(hostname) allocated cores: ${CPUS[*]}"
for n in 16 8 4 2; do
  list=$(IFS=,; echo "${CPUS[*]:0:$n}")
  for run in normflowsde_pf/lorenz96/seed0/learned_scale nsf/lorenz96/seed0/train; do
    taskset -c $list /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 \
      PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=$n OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax \
      /opt/venv/bin/python $P/time_cores.py --run $B/$run 2>&1 | grep "^{"
  done
done
echo "$(date -Is) timing done"
