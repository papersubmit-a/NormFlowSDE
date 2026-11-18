#!/bin/bash
#SBATCH --job-name=nfsde_re_s2
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=8G
#SBATCH --time=07:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/explore_re_s2_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/explore_re_s2_%A_%a.out
# Exploration (2026-11-16), not part of the benchmark table: does a longer or faster stage 2 improve the shared-trunk
# NormFlowSDE on REASSEMBLE? Same stage-1 checkpoints (results/shared_trunk_20261114/normflowsde/reassemble/seed<k>/s1),
# same stage-2 recipe as run_task.sh (4000 updates, learning rate 1e-4) except for the one change of the variant:
#   long   8000 updates                array tasks 0-2 (seed = task)
#   lr2    learning rate 2e-4          array tasks 3-5 (seed = task - 3)
# To be judged on the validation RMSE only (the trainer also scores the test split at the end).
# The gradient guards of seed 1 (--skip-nonfinite --skip-gradient-above 20) are on for every seed; they never act on
# seeds 0 and 2. CPU sizing from the campaign's stage-2 runs: 3 cores, 6.3 GB peak, 2.2 h per 4000 updates.
#   sbatch --array=0-5 explore_reassemble_s2.sbatch
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; B=results/shared_trunk_20261114; cd $ROOT
T=$SLURM_ARRAY_TASK_ID; SEED=$((T % 3))
case $((T / 3)) in
  0) NAME=long; EXTRA="--steps 8000 --learning-rate 1e-4";;
  1) NAME=lr2;  EXTRA="--steps 4000 --learning-rate 2e-4";;
  *) echo "no task $T"; exit 2;;
esac
OUT=$B/explore_reassemble_s2/$NAME/seed$SEED
[ -f $OUT/summary.json ] && { echo "done already: $OUT"; exit 0; }
mkdir -p $(dirname $OUT)
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
# shellcheck disable=SC2086
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax \
  /opt/venv/bin/python -m experiments.normflowsde_comparison.improve_pendulum_normflowsde --dataset reassemble --seed $SEED \
  --data-cache results/datasets/reassemble --init-run $B/normflowsde/reassemble/seed$SEED/s1 --objective energy --learned-scale \
  --realnvp-alpha 2 --energy-spread-weight 0.52 $EXTRA --skip-nonfinite --skip-gradient-above 20 --allow-cpu --output $OUT
