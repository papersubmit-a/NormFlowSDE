#!/usr/bin/env bash
# Direct-estimator baselines (2026-11-09): Latent NSF, Latent SDE and conditional SDE-GAN as amortized,
# observation-conditioned state estimators (experiments/normflowsde_comparison/direct_baselines.py), trained by
# train_direct_baseline.py with the particle-filter baselines' recipe (batch 16 windows, lr 3e-4, warmup-cosine,
# the dataset's update budget, validation-RMSE checkpoint selection) and capacity-matched to the amortized
# NormFlowSDE of each dataset (hidden width chosen so the estimator's parameter count matches; critic excluded).
#   usage: run_task.sh <method> <dataset> <seed> <cpu-list> [extra args]
# CPU-only host. Output: results/<method>/<dataset>/seed<k>/run.
set -uo pipefail
METHOD=$1; DATASET=$2; SEED=$3; CPUS=$4; shift 4
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax
PY=/home/anon/myenv_312/bin/python
case $DATASET in   # depth and NSF flow layers as NSF-PF; budget as NSF-PF / NormFlowSDE-PF
  pendulum)   DEPTH=2; FLOWS=6; BUDGET="--steps 8400 --eval-every 400";;
  lorenz96)   DEPTH=3; FLOWS=2; BUDGET="--steps 18500 --eval-every 500";;
  aloha)      DEPTH=3; FLOWS=2; BUDGET="--steps 9000 --eval-every 250";;
  reassemble) DEPTH=3; FLOWS=2; BUDGET="--steps 11000 --eval-every 500";;
  *) echo "unknown dataset $DATASET"; exit 2;;
esac
case $METHOD:$DATASET in   # capacity-matched widths (parameter count of the amortized NormFlowSDE: 103600 / 365360 / 129656 / 120412)
  latent_nsf:pendulum) H=88;;  latent_nsf:lorenz96) H=218;; latent_nsf:aloha) H=90;;  latent_nsf:reassemble) H=98;;
  latent_sde:pendulum) H=160;; latent_sde:lorenz96) H=230;; latent_sde:aloha) H=120;; latent_sde:reassemble) H=120;;
  sde_gan:pendulum)    H=180;; sde_gan:lorenz96)    H=274;; sde_gan:aloha)    H=142;; sde_gan:reassemble)    H=140;;
  *) echo "unknown method $METHOD"; exit 2;;
esac
OUT=results/$METHOD/$DATASET/seed$SEED/run
LOG=$ROOT/results/provenance/direct_baselines_20261109/logs/${METHOD}__${DATASET}__seed${SEED}.log
mkdir -p "$(dirname "$OUT")"
echo "$(date -Is) start $METHOD $DATASET seed$SEED cpus=$CPUS extra='$*'" >> "$LOG"
# shellcheck disable=SC2086
taskset -c "$CPUS" $PY -m experiments.normflowsde_comparison.train_direct_baseline --method $METHOD --dataset $DATASET \
  --data-cache results/datasets/$DATASET --output $OUT --seed $SEED --learning-rate 3e-4 --batch-windows 16 \
  --hidden-size $H --hidden-depth $DEPTH --flow-layers $FLOWS $BUDGET "$@" >> "$LOG" 2>&1
echo "$(date -Is) exit=$? $METHOD $DATASET seed$SEED" >> "$LOG"
