#!/usr/bin/env bash
# NormFlowSDE seed-0 pilot of the remaining hyperparameter levers, 2026-10-21 (GMM observation noise).
# Rules (user): energy-score loss, total update budget (pendulum 8400, lorenz96 18500, reassemble 11000) and
# parameter count unchanged; hyperparameters only.  Targets: pendulum + Lorenz-96 RMSE, REASSEMBLE 90% coverage.
# Levers:
#   * Lorenz-96 stage-1 optimizer (lr / batch): never tested; stage 1 is still converging at 14500 updates.
#     18:15 update: lr 1e-3 early-stopped at 8500 (patience 40 epochs) and still gave stage-2 val .303 (campaign .389),
#     coverage .80; p6k_1 Lorenz runs now build on the new stage-1 lrs (patience 0 = full budget; s2 lr / draws; leftover budget to s2).
#   * Stage-2 optimizer and ensemble (lr, batch, draws) on every dataset; stage 2 is still improving at 4000.
#   * Budget split: fewer stage-1 updates, more stage-2 updates, same total.
#   * REASSEMBLE coverage: stage-2 RealNVP alpha x lr/draws combos, from the campaign and the b4lr1e-3 stage 1.
# Usage: sweep.sh <group> [parallel]     groups: a30_lorenz a30_reassemble p6k_1 p6k_2
set -uo pipefail
ROOT=/home/anon/bayesinf_jax; cd $ROOT
O=$ROOT/results/results_normflowsde_paper/nfsde_sweep2_20261021; PY=/home/anon/myenv_312/bin/python
G=$ROOT/results/results_arxiv/filter_campaign_20261020_gmm
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
group=$1; par=${2:-3}
REF=$ROOT/experiments/normflowsde_comparison/filter_campaign_references
# Per-dataset campaign settings (stage-1 updates, validation windows, data cache, extra flags).
declare -A S1STEPS=([pendulum]=4400 [lorenz96]=14500 [reassemble]=7000)
declare -A VAL=([pendulum]=48 [lorenz96]=200 [reassemble]=300)
declare -A DATA=([pendulum]=$G/data_cache/pendulum [lorenz96]=$ROOT/results/results_normflowsde_paper/datasets/lorenz96 [reassemble]=$G/data_cache/reassemble)
declare -A EXTRA=([pendulum]=--allow-data-change [lorenz96]="" [reassemble]=--allow-data-change)
# Current-best seed-0 stage-1 checkpoints (final 3-seed table) and alternatives.
LZ_S1=$ROOT/results/results_arxiv/filter_campaign_20261019_pro6000/runs/lorenz96/seed0/normflowsde/s1_reference/attempt0
PD_S1=$ROOT/results/results_arxiv/nfsde_stage1_optim_20261021/pendulum/b16lr6e-4/s1/seed0
RE_S1=$G/runs/reassemble/seed0/normflowsde/s1_reference/attempt0
RE_B4=$ROOT/results/results_normflowsde_paper/normflowsde/reassemble/seed0/s1

stage2() {  # ds tag init s2steps flags...   (stage 2 = energy, learned scale; default lr 3e-4); SEED env (default 0)
  local ds=$1 tag=$2 init=$3 steps=$4 seed=${SEED:-0}; shift 4
  [ -f $O/$ds/$tag/s2/seed$seed/summary.json ] && { echo "$ds $tag s2 seed$seed exists"; return; }
  for i in $(seq 1 180); do grep -q '"status": "complete"' $init/run_metadata.json 2>/dev/null && break; sleep 60; done
  grep -q '"status": "complete"' $init/run_metadata.json 2>/dev/null || { echo "$ds $tag init $init not complete"; return; }
  $PY -m experiments.normflowsde_comparison.improve_pendulum_normflowsde --dataset $ds --seed $seed --data-cache ${DATA[$ds]} \
    --init-run $init --objective energy --learned-scale --steps $steps --learning-rate 3e-4 "$@" \
    --output $O/$ds/$tag/s2/seed$seed > $O/logs/${ds}_${tag}_s2_seed$seed.out 2>&1; echo "$ds $tag s2 seed$seed exit $?"
}
full() {  # ds tag s1steps s2steps s1flags... -- s2flags...; SEED env (default 0)
  local ds=$1 tag=$2 s1steps=$3 s2steps=$4 seed=${SEED:-0}; shift 4
  local s1=(); while [ $# -gt 0 ] && [ "$1" != "--" ]; do s1+=("$1"); shift; done; [ $# -gt 0 ] && shift
  local out=$O/$ds/$tag/s1/seed$seed
  if [ ! -f $out/model.eqx ] || ! grep -q '"status": "complete"' $out/run_metadata.json 2>/dev/null; then
    $PY -m experiments.normflowsde_comparison.train_normflowsde_filter --dataset $ds --seed $seed --train --data-cache ${DATA[$ds]} \
      --reference-metadata $REF/$ds/seed$seed.json --train-steps $s1steps --validation-windows ${VAL[$ds]} \
      --allow-source-drift ${EXTRA[$ds]} "${s1[@]}" --output $out > $O/logs/${ds}_${tag}_s1_seed$seed.out 2>&1 \
      || { echo "$ds $tag s1 seed$seed FAILED"; return; }
  fi
  stage2 $ds $tag $out $s2steps "$@"
}
export -f stage2 full
queue() {  # run the queued commands with at most $par in parallel
  local running=0
  while IFS= read -r cmd; do
    [ -z "$cmd" ] && continue
    eval "$cmd" &
    running=$((running+1))
    if [ $running -ge $par ]; then wait -n; running=$((running-1)); fi
  done
  wait
}
case $group in
  a30_lorenz)  # stage-1 optimizer, Lorenz-96 (campaign: batch 64, lr 3e-4, cosine; 14500 + 4000)
    queue <<'Q'
full lorenz96 lr6e-4 14500 4000 --learning-rate 6e-4
full lorenz96 lr1e-3 14500 4000 --learning-rate 1e-3
full lorenz96 b32 14500 4000 --batch-trajectories 32
full lorenz96 b32lr6e-4 14500 4000 --batch-trajectories 32 --learning-rate 6e-4
Q
    ;;
  a30_reassemble)  # coverage: stage-2 alpha x lr/draws, stage 2 only
    queue <<Q
stage2 reassemble ref_a2_lr1e-4 $RE_S1 4000 --realnvp-alpha 2 --learning-rate 1e-4
stage2 reassemble ref_a2.5 $RE_S1 4000 --realnvp-alpha 2.5
stage2 reassemble ref_a2_d32 $RE_S1 4000 --realnvp-alpha 2 --draws 32
stage2 reassemble b4_a2_lr1e-4 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4
Q
    ;;
  p6k_1)  # pendulum stage-2 / stage-1 variants + Lorenz stage-2 variants and budget split
    queue <<Q
stage2 pendulum s2lr6e-4 $PD_S1 4000 --learning-rate 6e-4
stage2 pendulum s2lr1e-3 $PD_S1 4000 --learning-rate 1e-3
stage2 pendulum s2b32 $PD_S1 4000 --batch-trajectories 32
stage2 pendulum s2b32lr6e-4 $PD_S1 4000 --batch-trajectories 32 --learning-rate 6e-4
stage2 pendulum s2d32 $PD_S1 4000 --draws 32
full pendulum seg6 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --segment-length 6
full pendulum seg21 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --segment-length 21
full pendulum b16lr1e-3 4400 4000 --batch-trajectories 16 --learning-rate 1e-3
full pendulum w5 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --observation-window 5
full pendulum split2400_6000 2400 6000 --batch-trajectories 16 --learning-rate 6e-4
full lorenz96 lr1e-3_p0 14500 4000 --learning-rate 1e-3 --patience 0
full lorenz96 lr2e-3_p0 14500 4000 --learning-rate 2e-3 --patience 0
stage2 lorenz96 lr1e-3_p0_s2lr1e-4 $O/lorenz96/lr1e-3_p0/s1/seed0 4000 --learning-rate 1e-4
stage2 lorenz96 lr1e-3_s2_10000 $O/lorenz96/lr1e-3/s1/seed0 10000
stage2 lorenz96 lr1e-3_s2lr1e-4 $O/lorenz96/lr1e-3/s1/seed0 4000 --learning-rate 1e-4
stage2 lorenz96 lr6e-4_s2lr1e-4 $O/lorenz96/lr6e-4/s1/seed0 4000 --learning-rate 1e-4
stage2 lorenz96 lr6e-4_s2lr6e-4 $O/lorenz96/lr6e-4/s1/seed0 4000 --learning-rate 6e-4
stage2 lorenz96 lr6e-4_s2d32 $O/lorenz96/lr6e-4/s1/seed0 4000 --draws 32
full lorenz96 lr1e-3_p0_gn5 14500 4000 --learning-rate 1e-3 --patience 0 --grad-norm 5
full lorenz96 lr1e-3_p0_wd0 14500 4000 --learning-rate 1e-3 --patience 0 --weight-decay 0
full lorenz96 lr1e-3_p0_seg9 14500 4000 --learning-rate 1e-3 --patience 0 --segment-length 9
stage2 lorenz96 lr1e-3_p0_s2lr6e-4 $O/lorenz96/lr1e-3_p0/s1/seed0 4000 --learning-rate 6e-4
Q
    ;;
  p6k_2)  # REASSEMBLE: budget split and remaining alpha combos (stage 2 alpha 2)
    queue <<Q
full reassemble split5000_6000 5000 6000 -- --realnvp-alpha 2
full reassemble b4_split5000_6000 5000 6000 --batch-trajectories 4 --learning-rate 1e-3 -- --realnvp-alpha 2
stage2 reassemble b4_a2_d32 $RE_B4 4000 --realnvp-alpha 2 --draws 32
stage2 reassemble ref_a2_b32 $RE_S1 4000 --realnvp-alpha 2 --batch-trajectories 32
stage2 reassemble ref_a2_lr2e-4 $RE_S1 4000 --realnvp-alpha 2 --learning-rate 2e-4
SEED=1 full lorenz96 lr1e-3 14500 4000 --learning-rate 1e-3
SEED=2 full lorenz96 lr1e-3 14500 4000 --learning-rate 1e-3
Q
    ;;
  a30_reassemble12)  # seeds 1/2 of the REASSEMBLE pilot leader: stage 1 batch 4 lr 1e-3, stage 2 alpha 2 lr 1e-4 (seed 0: .0307 / cov .882)
    queue <<'Q'
SEED=1 full reassemble b4lr1e-3_a2_lr1e-4 7000 4000 --batch-trajectories 4 --learning-rate 1e-3 -- --realnvp-alpha 2 --learning-rate 1e-4
SEED=2 full reassemble b4lr1e-3_a2_lr1e-4 7000 4000 --batch-trajectories 4 --learning-rate 1e-3 -- --realnvp-alpha 2 --learning-rate 1e-4
Q
    ;;
  *) echo "unknown group $group"; exit 2 ;;
esac
echo "sweep2 $group done"
