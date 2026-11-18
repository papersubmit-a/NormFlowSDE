#!/usr/bin/env bash
# Direct baselines with NormFlowSDE's encoder (2026-11-12 relaunch; supersedes direct_baselines_20261109/20261110).
# Change vs 20261110: the encoder is NormFlowSDE's bidirectional GRU over the last 10 source-time measurements with
# NormFlowSDE's widths (d_hidden_enc = d_c = 16 / 128 / 64 / 64) and control handling (controls kept out of the GRU
# except on Lorenz-96), instead of a width-64 forward GRU over the full history. check_encoder.py verifies it equals
# NormFlowSDE's encoder exactly; widths below re-matched to the NormFlowSDE parameter count by match_capacity.py
# (capacity.json). Everything else as 20261110: batch 16 windows, lr 3e-4, warmup-cosine, the dataset's update
# budget, validation-RMSE checkpoint selection, beta = 1, Markov latent-NSF posterior, SigKer bandwidth sqrt(d).
#   usage (inside the container): run_task.sh <method> <dataset> <seed> [extra args]
#   optional env TAG: suffix of the output method directory (validation sweeps, e.g. TAG=_sweep/beta0.1_euler)
# Output: results/<method>/<dataset>/seed<k>/run (results/smoke_direct_20261112/<method>/<dataset>/run with --smoke).
set -uo pipefail
METHOD=$1; DATASET=$2; SEED=$3; shift 3
ROOT=/home/anon/NormFlowSDE; cd "$ROOT"
case $DATASET in
  pendulum)   DEPTH=2; FLOWS=6; ENC="--encoder-hidden 16 --direct-control";     BUDGET="--steps 8400 --eval-every 400";;
  lorenz96)   DEPTH=3; FLOWS=2; ENC="--encoder-hidden 128 --no-direct-control"; BUDGET="--steps 18500 --eval-every 500";;
  aloha)      DEPTH=3; FLOWS=2; ENC="--encoder-hidden 64 --direct-control";     BUDGET="--steps 9000 --eval-every 250";;
  reassemble) DEPTH=3; FLOWS=2; ENC="--encoder-hidden 64 --direct-control";     BUDGET="--steps 11000 --eval-every 500";;
  *) echo "unknown dataset $DATASET"; exit 2;;
esac
case $METHOD:$DATASET in   # capacity.json; targets 103600 / 365360 / 129656 / 120412
  latent_nsf:pendulum) H=92;;  latent_nsf:lorenz96) H=138;; latent_nsf:aloha) H=63;;  latent_nsf:reassemble) H=70;;
  latent_sde:pendulum) H=177;; latent_sde:lorenz96) H=178;; latent_sde:aloha) H=107;; latent_sde:reassemble) H=107;;
  sigker_sde:pendulum) H=213;; sigker_sde:lorenz96) H=203;; sigker_sde:aloha) H=126;; sigker_sde:reassemble) H=124;;
  *) echo "unknown method $METHOD"; exit 2;;
esac
if [[ " $* " == *" --smoke "* ]]; then OUT=results/smoke_direct_20261112/$METHOD/$DATASET/run; else OUT=results/$METHOD${TAG:-}/$DATASET/seed$SEED/run; fi
mkdir -p "$(dirname "$OUT")"
echo "$(date -Is) start $METHOD $DATASET seed$SEED host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}_${SLURM_ARRAY_TASK_ID:-} extra='$*'"
# shellcheck disable=SC2086
/opt/venv/bin/python -m experiments.normflowsde_comparison.train_direct_baseline --method $METHOD --dataset $DATASET \
  --data-cache results/datasets/$DATASET --output $OUT --seed $SEED --learning-rate 3e-4 --batch-windows 16 \
  --hidden-size $H --hidden-depth $DEPTH --flow-layers $FLOWS --encoder-window 10 $ENC $BUDGET "$@"
rc=$?
echo "$(date -Is) exit=$rc $METHOD $DATASET seed$SEED"
exit $rc
