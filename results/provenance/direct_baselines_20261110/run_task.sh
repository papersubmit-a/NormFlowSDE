#!/usr/bin/env bash
# Direct-estimator baselines (2026-11-10 relaunch of direct_baselines_20261109 under SLURM + container).
# Same recipe as 20261109/run_task.sh: batch 16 windows, lr 3e-4, warmup-cosine, the dataset's update budget,
# validation-RMSE checkpoint selection, hidden width capacity-matched to the amortized NormFlowSDE.
#   usage (inside the container): run_task.sh <method> <dataset> <seed> [extra args]
#   optional env TAG: suffix of the output method directory (e.g. TAG=_sigma0.3 for the SigKernel bandwidth sweep)
# Output: results/<method>/<dataset>/seed<k>/run (results/smoke_direct_20261110/<method>/<dataset>/run with --smoke).
set -uo pipefail
METHOD=$1; DATASET=$2; SEED=$3; shift 3
ROOT=/home/anon/NormFlowSDE; cd "$ROOT"
case $DATASET in
  pendulum)   DEPTH=2; FLOWS=6; BUDGET="--steps 8400 --eval-every 400";;
  lorenz96)   DEPTH=3; FLOWS=2; BUDGET="--steps 18500 --eval-every 500";;
  aloha)      DEPTH=3; FLOWS=2; BUDGET="--steps 9000 --eval-every 250";;
  reassemble) DEPTH=3; FLOWS=2; BUDGET="--steps 11000 --eval-every 500";;
  *) echo "unknown dataset $DATASET"; exit 2;;
esac
case $METHOD:$DATASET in   # parameter count of the amortized NormFlowSDE: 103600 / 365360 / 129656 / 120412
  latent_nsf:pendulum) H=82;;  latent_nsf:lorenz96) H=185;; latent_nsf:aloha) H=77;;  latent_nsf:reassemble) H=84;;   # Markov posterior with full-width transition net (v3)
  latent_sde:pendulum) H=160;; latent_sde:lorenz96) H=230;; latent_sde:aloha) H=120;; latent_sde:reassemble) H=120;;
  sde_gan:pendulum)    H=180;; sde_gan:lorenz96)    H=274;; sde_gan:aloha)    H=142;; sde_gan:reassemble)    H=140;;
  # SigKernelSDE = the SDE-GAN generator (critic-free), so the SDE-GAN capacity-matched widths apply unchanged
  sigker_sde:pendulum) H=180;; sigker_sde:lorenz96) H=274;; sigker_sde:aloha) H=142;; sigker_sde:reassemble) H=140;;
  *) echo "unknown method $METHOD"; exit 2;;
esac
if [[ " $* " == *" --smoke "* ]]; then OUT=results/smoke_direct_20261110/$METHOD${TAG:-}/$DATASET/run; else OUT=results/$METHOD${TAG:-}/$DATASET/seed$SEED/run; fi
mkdir -p "$(dirname "$OUT")"
echo "$(date -Is) start $METHOD $DATASET seed$SEED host=$(hostname) job=${SLURM_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-} extra='$*'"
# shellcheck disable=SC2086
/opt/venv/bin/python -m experiments.normflowsde_comparison.train_direct_baseline --method $METHOD --dataset $DATASET \
  --data-cache results/datasets/$DATASET --output $OUT --seed $SEED --learning-rate 3e-4 --batch-windows 16 \
  --hidden-size $H --hidden-depth $DEPTH --flow-layers $FLOWS $BUDGET "$@"
rc=$?
echo "$(date -Is) exit=$rc $METHOD $DATASET seed$SEED"
exit $rc
