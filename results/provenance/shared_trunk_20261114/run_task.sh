#!/usr/bin/env bash
# Shared-trunk campaign (2026-11-14). One (method, dataset, seed) per call, every stage in order.
#
# What changes against the selected runs of the paper (results/SELECTED.md) and nothing else:
#   * NormFlowSDE (all four datasets): shared trunk (--shared-trunk). One feature network per step; the drift is a
#     linear head, each coupling layer a one-hidden-layer head, the learned scale one free variable per coordinate.
#     Pendulum: trunk 53 x 2, 2 coupling layers (was 6), head width 16 -> 9,940 parameters.
#     Lorenz-96 / ALOHA / REASSEMBLE: trunk widened to 266 / 153 / 150 so that the stage-2 parameter count matches
#     the previous model (365,382 / 129,567 / 120,032 against 365,360 / 129,656 / 120,412).
#   * Pendulum, the other seven methods: at most 10,000 parameters and 2 coupling layers in every flow (was 6).
#   * The four particle filters on all four datasets: sized to NormFlowSDE's parameter count (previously NSF-PF had
#     40-60 % fewer parameters), one measurement-flow architecture for all four per dataset (pendulum: width 16,
#     2 coupling layers; elsewhere width 64, 4 coupling layers, as the bootstrap filters had; NF-DPF previously
#     sized its measurement flow like its transition flow), and NormFlowSDE-PF with the shared-trunk transition.
#   The direct baselines on Lorenz-96 / ALOHA / REASSEMBLE already match NormFlowSDE and are not retrained.
# Widths from size_models.py (sizes.json). Recipes (losses, budgets, learning rates, selection) are those of the
# selected runs; no hyperparameter is re-tuned.
#
#   usage (inside the container): run_task.sh <method> <dataset> <seed>
#   env SMOKE=1: a few updates only, output under results/provenance/shared_trunk_20261114/smoke
# Output: results/shared_trunk_20261114/<method>/<dataset>/seed<k>/<stage>; log: logs/<method>__<dataset>__seed<k>.log
set -uo pipefail
METHOD=$1; DATASET=$2; SEED=$3
ROOT=/home/anon/NormFlowSDE; cd "$ROOT"
P=results/provenance/shared_trunk_20261114
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then BASE=$P/smoke; else BASE=results/shared_trunk_20261114; fi
OUT=$BASE/$METHOD/$DATASET/seed$SEED
PY=/opt/venv/bin/python; M=experiments.normflowsde_comparison
DATA=results/datasets/$DATASET
REF=experiments/normflowsde_comparison/filter_campaign_references/$DATASET/seed$SEED.json
JITTERS=0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2
case $DATASET in   # depth, update budget and validation schedule of the dataset; common measurement flow of the particle filters
  pendulum)   DEPTH=2; BUDGET="--steps 8400 --eval-every 400";  VAL=48;  MEAS="--measurement-hidden 16 --measurement-depth 2 --measurement-layers 2";;
  lorenz96)   DEPTH=3; BUDGET="--steps 18500 --eval-every 500"; VAL=200; MEAS="--measurement-hidden 64 --measurement-depth 2 --measurement-layers 4";;
  aloha)      DEPTH=3; BUDGET="--steps 9000 --eval-every 250";  VAL=50;  MEAS="--measurement-hidden 64 --measurement-depth 2 --measurement-layers 4";;
  reassemble) DEPTH=3; BUDGET="--steps 11000 --eval-every 500"; VAL=300; MEAS="--measurement-hidden 64 --measurement-depth 2 --measurement-layers 4";;
  *) echo "unknown dataset $DATASET"; exit 2;;
esac
# NormFlowSDE-PF on Lorenz-96 (2026-11-15): validate every 2500 updates instead of every 500. Measured on CPU
# (time_cores.py, job 35730): an update takes 0.04 s (18500 updates = 12 min) and one validation (10 jitters x 200
# windows, 1000 particles) 22-29 min, so 37 validations made a 14-18 h run. Same updates and schedule; the
# checkpoint is selected among 8 validations (2500, ..., 17500, 18500) instead of 37.
[ "$METHOD:$DATASET" = normflowsde_pf:lorenz96 ] && BUDGET="--steps 18500 --eval-every 2500"
[ "$SMOKE" = 1 ] && BUDGET="--smoke"
mkdir -p "$OUT" "$P/logs"
LOG=$P/logs/${METHOD}__${DATASET}__seed${SEED}$([ "$SMOKE" = 1 ] && echo __smoke).log
say() { echo "$(date -Is) $*" | tee -a "$LOG"; }
step() {  # step <name> <done-file> <command...>: skip a finished stage, stop the task at the first failure
  local name=$1 marker=$2; shift 2
  if [ -f "$marker" ]; then say "skip $name (exists)"; return 0; fi
  say "start $name: $*"
  "$@" >> "$LOG" 2>&1; local rc=$?
  say "exit=$rc $name"
  [ $rc = 0 ] || exit $rc
}
complete() { grep -q '"status": "complete"' "$1/run_metadata.json" 2>/dev/null; }
# The two NormFlowSDE trainers stop without a GPU unless --allow-cpu is given. On a CPU-only allocation
# (2026-11-14: the a30 GPU slots were needed elsewhere, CPU partitions a30 / l40s / pro6000) pass it explicitly.
CPUFLAG=""
[ "$($PY -c 'import jax; print(int(any(d.platform == "gpu" for d in jax.devices())))' 2>/dev/null)" = 1 ] || CPUFLAG="--allow-cpu"
fresh() { [ -d "$1" ] && mv "$1" "$1.failed.$(date +%s)"; return 0; }   # these trainers refuse an existing directory
say "task $METHOD $DATASET seed$SEED host=$(hostname) job=${SLURM_JOB_ID:-} smoke=$SMOKE devices=$($PY -c 'import jax; print(jax.devices())' 2>/dev/null)"

case $METHOD in
normflowsde)
  case $DATASET in
    pendulum)   ARCH="--shared-trunk --hidden-size 53 --flow-layers 2 --diffusion-hidden-size 16"
                S1="--train-steps 4400 --validation-windows 48 --allow-source-drift --allow-data-change --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 30"
                S2="--steps 4000 --learning-rate 3e-3";;
    lorenz96)   ARCH="--shared-trunk --hidden-size 266"
                S1="--train-steps 14500 --validation-windows 200 --learning-rate 2e-3 --patience 0 --solver euler --allow-source-drift"
                # 2026-11-16: stage-2 learning rate 3e-4 instead of the 3e-5 of the selected runs. At 3e-5 the free
                # log-scale variables move too little in 4000 updates and the model over-covers (validation 90 %
                # coverage 0.978 after stage 1, 0.965 after stage 2). Chosen on validation
                # (explore_lorenz96_s2.sbatch, job 35945), seeds 0 / 1 / 2: CRPS 0.1556 / 0.1559 / 0.1559 ->
                # 0.1462 / 0.1468 / 0.1477, coverage 0.965 -> 0.907 / 0.909 / 0.904. The table runs are those
                # exploration runs, copied from explore_lorenz96_s2/lr3e-4/seed<k>; the 3e-5 runs are kept as
                # s2.campaign_lr3e-5.*.
                S2="--steps 4000 --validation-every 250 --batch-trajectories 16 --draws 16 --learning-rate 3e-4 --energy-beta 1 --mean-mse-weight 0 --select-metric crps";;
    aloha)      ARCH="--shared-trunk --hidden-size 153"
                S1="--train-steps 5000 --validation-windows 50 --allow-source-drift --allow-data-change"
                S2="--steps 4000 --learning-rate 3e-4";;
    reassemble) ARCH="--shared-trunk --hidden-size 150"
                S1="--train-steps 7000 --validation-windows 300 --allow-source-drift --allow-data-change --batch-trajectories 4 --learning-rate 1e-3"
                # 2026-11-16: 8000 stage-2 updates instead of the 4000 of the selected runs, the one budget change of
                # the campaign. Chosen on validation RMSE (explore_reassemble_s2.sbatch, job 35896): 0.0395 / 0.0422 /
                # 0.0410 -> 0.0368 / 0.0394 / 0.0379 for seeds 0 / 1 / 2. The table runs are those exploration runs,
                # copied from explore_reassemble_s2/long/seed<k>; the 4000-update runs are kept as
                # s2.campaign_4000_updates.*. They ran with the gradient guards on for every seed (never acting on
                # seeds 0 and 2; 163 discarded updates on seed 1).
                S2="--steps 8000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.52 --skip-nonfinite --skip-gradient-above 20";;
  esac
  # Smoke: later flags win. 600 updates reach the first validation of every recipe (at most every 500 updates).
  if [ "$SMOKE" = 1 ]; then S1="$S1 --train-steps 600 --validation-windows 8"; S2="$S2 --steps 20 --validation-every 10 --smoke"; fi
  # REASSEMBLE seed 1 (2026-11-15): stage 2 stopped twice, on two CPU allocations, with a nonfinite loss at update
  # 801 after 800 healthy updates (jobs 35637_55, 35720_55; output kept as s2.failed_nonfinite_step801.*). That one
  # update is discarded instead (--skip-nonfinite; the count is recorded as skipped_nonfinite_updates). No other
  # task gets the flag, so their recorded arguments stay as run.
  # 2026-11-16: with --skip-nonfinite alone the run finished but discarded 31 updates and took 68 more with gradient
  # norms of 52 to 1.8e19 (clipped to norm 1): every one of the 31 had training window 1450 or 1983 in its batch,
  # and 28 of the 35 largest had 1060 or 1983. The stage-1 model of this seed already runs away on window 1983;
  # the clipped steps then broke 1450 and 1060 as well (output kept as s2.skip_nonfinite_only.*). Updates with a
  # gradient norm above 20 are therefore discarded too (seeds 0 and 2 never exceeded 4.6 in 4000 updates each).
  # (Since the 8000-update runs of 2026-11-16 both flags are part of the REASSEMBLE recipe above, for every seed.)
  if ! complete "$OUT/s1"; then
    fresh "$OUT/s1"
    # shellcheck disable=SC2086
    step s1 "$OUT/s1/never" $PY -m $M.train_normflowsde_filter --dataset $DATASET --seed $SEED --train --data-cache $DATA \
      --reference-metadata $REF $S1 $ARCH $CPUFLAG --output $OUT/s1
  else say "skip s1 (complete)"; fi
  # shellcheck disable=SC2086
  step s2 "$OUT/s2/summary.json" $PY -m $M.improve_pendulum_normflowsde --dataset $DATASET --seed $SEED --data-cache $DATA \
    --init-run $OUT/s1 --objective energy --learned-scale $S2 $CPUFLAG --output $OUT/s2
  ;;
nf_dpf)
  case $DATASET in   # width (sizes.json), context flag of the runner, stage-1 / stage-2 updates and dataset flags of the campaign plans
    pendulum)   H=32;  C=16;  N1=4400;  N2=4000; F1="--observation-noise gmm"; F2="--batches 32,16,8";;
    lorenz96)   H=155; C=128; N1=12500; N2=6000; F1=""; F2="";;
    aloha)      H=70;  C=64;  N1=3000;  N2=6000; F1="--variant pseudo_process_uncertainty_gmm"; F2="";;
    reassemble) H=77;  C=64;  N1=5000;  N2=6000; F1="--observation-noise gmm"; F2="";;
  esac
  V1=$VAL; V2=$VAL
  if [ "$SMOKE" = 1 ]; then N1=20; N2=20; V1=2; V2=2; fi
  if ! complete "$OUT/s1"; then
    fresh "$OUT/s1"
    # shellcheck disable=SC2086
    step s1 "$OUT/s1/never" $PY -m $M.run --dataset $DATASET --method nf_dpf --settings-profile legacy-compatible-v1 \
      --hidden-size $H --hidden-depth $DEPTH --flow-layers 2 --context-dim $C $MEAS $F1 --seed $SEED \
      --data-cache $DATA --train-steps $N1 --validation-windows $V1 --output $OUT/s1 --train
  else say "skip s1 (complete)"; fi
  # shellcheck disable=SC2086
  step s2 "$OUT/s2/summary.json" $PY -m $M.improve_lorenz_baselines --dataset $DATASET --method nf_dpf --seed $SEED \
    --anchor-weight 0 --crps-weight 1 --mean-mse-weight 0 --ess-weight 0.01 --remat --skip-nonfinite \
    --validation-windows $V2 --steps $N2 --data-cache $DATA --init-run $OUT/s1 --measurement-nll-weight 1 \
    --measurement-batch-windows 16 $F2 --output $OUT/s2
  ;;
nsf|flowsde|normflowsde_pf)
  COMMON="--dataset $DATASET --data-cache $DATA --seed $SEED --learning-rate 3e-4 --batch-windows 16 --hidden-depth $DEPTH --flow-layers 2 $BUDGET --selection-jitters $JITTERS $MEAS"
  case $METHOD:$DATASET in   # widths from sizes.json; NormFlowSDE-PF: coupling-head width and alpha of the amortized model
    nsf:pendulum) H=48;;  nsf:lorenz96) H=224;;  nsf:aloha) H=107;;  nsf:reassemble) H=113;;
    flowsde:pendulum) H=59;;  flowsde:lorenz96) H=282;;  flowsde:aloha) H=141;;  flowsde:reassemble) H=144;;
    normflowsde_pf:pendulum)   H=63;  PF="--diffusion-hidden-size 16 --alpha 1.0";;
    normflowsde_pf:lorenz96)   H=363; PF="--diffusion-hidden-size 64 --alpha 1.5";;
    normflowsde_pf:aloha)      H=165; PF="--diffusion-hidden-size 64 --alpha 1.0";;
    normflowsde_pf:reassemble) H=169; PF="--diffusion-hidden-size 64 --alpha 1.0";;
  esac
  case $METHOD in
  nsf)
    # shellcheck disable=SC2086
    step train "$OUT/train/summary.json" $PY -m $M.train_state_filter --method nsf --hidden-size $H $COMMON --output $OUT/train
    step jitter "$OUT/jitter/summary.json" $PY -m $M.tune_state_filter_jitter --run $OUT/train --grid $JITTERS --output $OUT/jitter
    ;;
  flowsde)
    for D in 0.01 0.03 0.05; do
      # shellcheck disable=SC2086
      step D$D "$OUT/D$D/summary.json" $PY -m $M.train_state_filter --method flowsde --hidden-size $H $COMMON --diffusion $D --output $OUT/D$D
    done
    # Diffusion level with the lowest validation RMSE (filter_campaign.run_selection: ties go to the first level).
    step select "$OUT/selected.json" $PY -c "
import json, sys
out = sys.argv[1]; levels = ['D0.01', 'D0.03', 'D0.05']
scores = {d: json.load(open(f'{out}/{d}/summary.json'))['best_validation_rmse'] for d in levels}
chosen = min(levels, key=lambda d: (scores[d], levels.index(d)))
json.dump(dict(selected=chosen, validation_rmse=scores), open(f'{out}/selected.json', 'w'), indent=1)
print(chosen, scores)" "$OUT"
    ;;
  normflowsde_pf)
    # shellcheck disable=SC2086
    step learned_scale "$OUT/learned_scale/summary.json" $PY -m $M.train_state_filter --method normflowsde --shared-trunk \
      --learned-scale --hidden-size $H $PF $COMMON --output $OUT/learned_scale
    ;;
  esac
  ;;
latent_nsf|latent_sde|sigker_sde)
  [ "$DATASET" = pendulum ] || { echo "$METHOD is retrained on the pendulum only"; exit 2; }
  case $METHOD in   # width from sizes.json; extra = the configuration selected on the validation grid (direct_baselines_20261112/selected.json)
    latent_nsf) H=32; EXTRA="--beta 1 --posterior-step residual --learning-rate 1e-3 $MEAS";;
    latent_sde) H=44; EXTRA="--beta 0.1 --learning-rate 1e-3";;
    sigker_sde) H=50; EXTRA="--sig-sigma 0.5 --learning-rate 1e-3";;
  esac
  # shellcheck disable=SC2086
  step run "$OUT/run/summary.json" $PY -m $M.train_direct_baseline --method $METHOD --dataset pendulum --data-cache $DATA \
    --output $OUT/run --seed $SEED --learning-rate 3e-4 --batch-windows 16 --hidden-size $H --hidden-depth 2 --flow-layers 2 \
    --encoder-window 10 --encoder-hidden 16 --direct-control $BUDGET $EXTRA
  ;;
*) echo "unknown method $METHOD"; exit 2;;
esac
say "done $METHOD $DATASET seed$SEED"
