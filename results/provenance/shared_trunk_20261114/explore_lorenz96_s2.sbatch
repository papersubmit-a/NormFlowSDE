#!/bin/bash
#SBATCH --job-name=nfsde_l96_s2
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=08:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/explore_l96_s2_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/explore_l96_s2_%A_%a.out
# Exploration (2026-11-16), not part of the benchmark table: the shared-trunk NormFlowSDE over-covers on Lorenz-96
# (test 90 % coverage 0.966-0.969). Its stage-1 model starts stage 2 at validation coverage 0.978 and the 4000
# updates at learning rate 3e-5 only bring it to 0.965, still falling at the last update (validation RMSE and CRPS
# too): Adam moves the free log-scale variables by about one learning rate per update, so at most ~0.12 in total.
# Same stage-1 checkpoints and stage-2 recipe as run_task.sh except for the one change of the variant:
#   lr1e-4  learning rate 1e-4            array tasks 0-2 (seed = task)
#   lr3e-4  learning rate 3e-4            array tasks 3-5
#   long    12000 updates at 3e-5         array tasks 6-8
# To be judged on the validation split only (CRPS, the selection metric of this recipe, and coverage).
# Sizing from the campaign runs (jobs 35620_48-50, 6 cores requested): 3.7 cores used on average, 3.3 GB peak,
# 1.13 h per 4000 stage-2 updates -> 4 cores, 4 GB; 8 h covers the 12000-update variant.
#   sbatch --array=0-8 explore_lorenz96_s2.sbatch
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; B=results/shared_trunk_20261114; cd $ROOT
T=$SLURM_ARRAY_TASK_ID; SEED=$((T % 3))
case $((T / 3)) in
  0) NAME=lr1e-4; EXTRA="--steps 4000 --learning-rate 1e-4";;
  1) NAME=lr3e-4; EXTRA="--steps 4000 --learning-rate 3e-4";;
  2) NAME=long;   EXTRA="--steps 12000 --learning-rate 3e-5";;
  *) echo "no task $T"; exit 2;;
esac
OUT=$B/explore_lorenz96_s2/$NAME/seed$SEED
[ -f $OUT/summary.json ] && { echo "done already: $OUT"; exit 0; }
mkdir -p $(dirname $OUT)
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
# shellcheck disable=SC2086
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax \
  /opt/venv/bin/python -m experiments.normflowsde_comparison.improve_pendulum_normflowsde --dataset lorenz96 --seed $SEED \
  --data-cache results/datasets/lorenz96 --init-run $B/normflowsde/lorenz96/seed$SEED/s1 --objective energy --learned-scale \
  --validation-every 250 --batch-trajectories 16 --draws 16 --energy-beta 1 --mean-mse-weight 0 --select-metric crps \
  $EXTRA --allow-cpu --output $OUT
