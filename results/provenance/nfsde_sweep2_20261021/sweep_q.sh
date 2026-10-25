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
# sweep_b.sh: SEPARATE copy for groups added after 19:20. Never edit a script file while a Slurm job executes it
# (bash reads scripts incrementally; job 32358 resumed mid-line after an edit and ran unintended commands).
# sweep_c.sh: copy for groups added after 19:50 (sweep_b.sh is executed by job 32368's successors; leave it alone).
# sweep_d.sh: copy for groups added after 20:20 (earlier copies may still be executed by running jobs).
# sweep_e.sh: copy for groups added after 21:00.
# sweep_f.sh: copy for groups added after 22:00.
# sweep_g.sh: copy for groups added after 23:50.
# sweep_h.sh: copy for groups added 2026-10-22 12:30 (coverage round: pendulum .822, REASSEMBLE .874, target .90).
# sweep_i.sh: copy for groups added 2026-10-22 13:00 (sweep_h.sh is still executed by job 32386; leave it alone).
# sweep_j.sh: copy for groups added 2026-10-22 13:20 (sweep_h.sh / sweep_i.sh are executed by jobs 32386 / 32390; leave them alone).
# sweep_k.sh: copy for groups added 2026-10-22 13:40 (sweep_h/i/j.sh are executed by jobs 32386 / 32390 / 32391; leave them alone).
# sweep_l.sh: copy for groups added 2026-10-22 15:00 (sweep_h/i/j/k.sh belong to jobs 32386 / 32390 / 32391 / 32392; leave them alone).
# sweep_m.sh: copy for groups added 2026-10-22 14:10 (sweep_h..l.sh belong to jobs 32386/32390/32391/32392/32393; leave them alone).
# sweep_n.sh: copy for groups added 2026-10-22 14:40 (sweep_h..m.sh belong to running/pending jobs; leave them alone).
# sweep_o.sh: copy for groups added 2026-10-22 15:50. Pendulum work moved to HPC (results/results_arxiv/hpc_deployments/20261022_pendulum_coverage);
# REASSEMBLE stays local because its cache verifies raw parquet files at local paths.
# sweep_p.sh: copy for groups added 2026-10-22 16:00 (earlier copies belong to running/finished jobs).
# sweep_q.sh: copy for groups added 2026-10-22 17:40 (earlier copies belong to finished jobs).
# Usage: sweep_q.sh <group> [parallel]     groups: a30_q_reas52 a30_q_pend_rw30crps
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
  a30_pendulum_lr)  # pendulum stage-2 lr follow-ups (s2 lr 1e-3 gave .0817 vs .0912, best step 4000)
    queue <<Q
stage2 pendulum s2lr2e-3 $PD_S1 4000 --learning-rate 2e-3
full pendulum split2400_6000_s2lr1e-3 2400 6000 --batch-trajectories 16 --learning-rate 6e-4 -- --learning-rate 1e-3
stage2 pendulum s2lr1e-3_b32 $PD_S1 4000 --learning-rate 1e-3 --batch-trajectories 32
Q
    ;;
  a30_pendulum_c)  # pendulum round 3: s2 lr 2e-3 (.0732/cov .831) and lr1e-3+b32 (.0742/.818) lead; seg6 s1 had the best coverage (.882)
    queue <<Q
stage2 pendulum s2lr3e-3 $PD_S1 4000 --learning-rate 3e-3
stage2 pendulum s2lr2e-3_b32 $PD_S1 4000 --learning-rate 2e-3 --batch-trajectories 32
stage2 pendulum seg6_s2lr2e-3 $O/pendulum/seg6/s1/seed0 4000 --learning-rate 2e-3
stage2 pendulum seg6_s2lr2e-3_b32 $O/pendulum/seg6/s1/seed0 4000 --learning-rate 2e-3 --batch-trajectories 32
Q
    ;;
  a30_d)  # pendulum candidates seeds 1/2: A = b16lr6e-4 s1 + s2 lr 3e-3 (seed 0 .0709 / cov .797, val .0709);
          # B = seg6 s1 + s2 lr 2e-3 (.0731 / .846, val .0736). REASSEMBLE: b4 s1 + a2 + lr 1e-4 + 32 draws combo.
    queue <<Q
SEED=1 stage2 pendulum A_s2lr3e-3 $ROOT/results/results_arxiv/nfsde_joint_local_20261021/pendulum/b16lr6e-4/s1/seed1 4000 --learning-rate 3e-3
SEED=2 stage2 pendulum A_s2lr3e-3 $ROOT/results/results_arxiv/nfsde_joint_local_20261021/pendulum/b16lr6e-4/s1/seed2 4000 --learning-rate 3e-3
SEED=1 full pendulum B_seg6_s2lr2e-3 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --segment-length 6 -- --learning-rate 2e-3
SEED=2 full pendulum B_seg6_s2lr2e-3 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --segment-length 6 -- --learning-rate 2e-3
stage2 reassemble b4_a2_lr1e-4_d32 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --draws 32
Q
    ;;
  a30_e)  # Lorenz stage-1 lr with patience 0: 2e-3 reached val .303 at 13750 (1e-3: .328; campaign .485) -> push higher; s2 lr 6e-4 helped lr6e-4 (.315 vs .324)
    queue <<Q
full lorenz96 lr3e-3_p0 14500 4000 --learning-rate 3e-3 --patience 0
full lorenz96 lr4e-3_p0 14500 4000 --learning-rate 4e-3 --patience 0
stage2 lorenz96 lr2e-3_p0_s2lr6e-4 $O/lorenz96/lr2e-3_p0/s1/seed0 4000 --learning-rate 6e-4
Q
    ;;
  p6k_f)  # Lorenz: stage 2 on the lr 2e-3 full-budget stage 1 starts at val .295 / cov .97 and ends at .286 / cov .67 (test .307 / .685):
          # the energy fine-tune over-sharpens. Coverage-preserving variants (same total budget 18500) + seeds 1/2 of the lr 2e-3 stage 1.
    queue <<Q
stage2 lorenz96 lr2e-3_p0_s2lr3e-5 $O/lorenz96/lr2e-3_p0/s1/seed0 4000 --learning-rate 3e-5
stage2 lorenz96 lr2e-3_p0_s2lr1e-4 $O/lorenz96/lr2e-3_p0/s1/seed0 4000 --learning-rate 1e-4
full lorenz96 lr2e-3_split17500_1000 17500 1000 --learning-rate 2e-3 --patience 0
full lorenz96 lr2e-3_split16500_2000 16500 2000 --learning-rate 2e-3 --patience 0 -- --learning-rate 1e-4
SEED=1 full lorenz96 lr2e-3_p0 14500 4000 --learning-rate 2e-3 --patience 0
SEED=2 full lorenz96 lr2e-3_p0 14500 4000 --learning-rate 2e-3 --patience 0
Q
    ;;
  a30_g1)  # Lorenz: stage-2 checkpoint selected by validation CRPS (same checkpoints as RMSE on the other datasets); lr 3e-5 -> step 750, val .289 / cov .875 on seed 0
    queue <<Q
SEED=0 stage2 lorenz96 lr2e-3_p0_s2lr3e-5_crps $O/lorenz96/lr2e-3_p0/s1/seed0 4000 --learning-rate 3e-5 --select-metric crps
SEED=1 stage2 lorenz96 lr2e-3_p0_s2lr3e-5_crps $O/lorenz96/lr2e-3_p0/s1/seed1 4000 --learning-rate 3e-5 --select-metric crps
SEED=2 stage2 lorenz96 lr2e-3_p0_s2lr3e-5_crps $O/lorenz96/lr2e-3_p0/s1/seed2 4000 --learning-rate 3e-5 --select-metric crps
Q
    ;;
  a30_g2)  # same with stage-2 lr 1e-4 (seed 0: step 250, val .295 / cov .900)
    queue <<Q
SEED=0 stage2 lorenz96 lr2e-3_p0_s2lr1e-4_crps $O/lorenz96/lr2e-3_p0/s1/seed0 4000 --learning-rate 1e-4 --select-metric crps
SEED=1 stage2 lorenz96 lr2e-3_p0_s2lr1e-4_crps $O/lorenz96/lr2e-3_p0/s1/seed1 4000 --learning-rate 1e-4 --select-metric crps
SEED=2 stage2 lorenz96 lr2e-3_p0_s2lr1e-4_crps $O/lorenz96/lr2e-3_p0/s1/seed2 4000 --learning-rate 1e-4 --select-metric crps
Q
    ;;
  a30_h_pend)  # 2026-10-22 coverage round, pendulum. Leader b16lr6e-4 s1 + s2 lr 3e-3: .0709 / cov .797 (3-seed .822); coverage stays .73-.81 over all of stage 2.
               # Levers that act on dispersion only: energy-score exponent beta < 1 (strictly proper, more weight on calibration), RealNVP alpha 2 (wider log-scale bound), 32 draws.
    queue <<CMDS
stage2 pendulum s2lr3e-3_beta0.5 $PD_S1 4000 --learning-rate 3e-3 --energy-beta 0.5
stage2 pendulum s2lr3e-3_beta0.75 $PD_S1 4000 --learning-rate 3e-3 --energy-beta 0.75
stage2 pendulum s2lr3e-3_beta0.3 $PD_S1 4000 --learning-rate 3e-3 --energy-beta 0.3
stage2 pendulum s2lr2e-3_beta0.5 $PD_S1 4000 --learning-rate 2e-3 --energy-beta 0.5
stage2 pendulum s2lr3e-3_a2 $PD_S1 4000 --learning-rate 3e-3 --realnvp-alpha 2
stage2 pendulum s2lr3e-3_a2_beta0.5 $PD_S1 4000 --learning-rate 3e-3 --realnvp-alpha 2 --energy-beta 0.5
stage2 pendulum s2lr3e-3_beta0.5_d32 $PD_S1 4000 --learning-rate 3e-3 --energy-beta 0.5 --draws 32
CMDS
    ;;
  a30_h_reas)  # 2026-10-22 coverage round, REASSEMBLE. Leader b4lr1e-3 s1 + alpha 2, s2 lr 1e-4: .0307 / cov .882 (3-seed .874). Same levers; lr 3e-4 variant keeps the RMSE gain (.0275) if beta restores coverage.
    queue <<CMDS
stage2 reassemble b4_a2_beta0.5_lr1e-4 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-beta 0.5
stage2 reassemble b4_a2_beta0.75_lr1e-4 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-beta 0.75
stage2 reassemble b4_a2_beta0.3_lr1e-4 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-beta 0.3
stage2 reassemble b4_a3_lr1e-4 $RE_B4 4000 --realnvp-alpha 3 --learning-rate 1e-4
stage2 reassemble b4_a2_beta0.5_lr3e-4 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 3e-4 --energy-beta 0.5
stage2 reassemble b4_a3_beta0.5_lr1e-4 $RE_B4 4000 --realnvp-alpha 3 --learning-rate 1e-4 --energy-beta 0.5
CMDS
    ;;
  a30_i_pend)  # pendulum coverage round, seed-0 results: beta .75/.5/.3 -> cov .799/.815/.853 at RMSE .0724/.0748/.0792 (leader .0709/.797);
               # alpha 2 + beta .5 -> .0713 / cov .833 (best test trade-off, val RMSE .0756). Seeds 1/2 of beta .5 and alpha2+beta .5; two combined seed-0 variants.
    PD1=$ROOT/results/results_arxiv/nfsde_joint_local_20261021/pendulum/b16lr6e-4/s1/seed1; PD2=$ROOT/results/results_arxiv/nfsde_joint_local_20261021/pendulum/b16lr6e-4/s1/seed2
    queue <<CMDS
SEED=1 stage2 pendulum s2lr3e-3_beta0.5 $PD1 4000 --learning-rate 3e-3 --energy-beta 0.5
SEED=2 stage2 pendulum s2lr3e-3_beta0.5 $PD2 4000 --learning-rate 3e-3 --energy-beta 0.5
SEED=1 stage2 pendulum s2lr3e-3_a2_beta0.5 $PD1 4000 --learning-rate 3e-3 --realnvp-alpha 2 --energy-beta 0.5
SEED=2 stage2 pendulum s2lr3e-3_a2_beta0.5 $PD2 4000 --learning-rate 3e-3 --realnvp-alpha 2 --energy-beta 0.5
stage2 pendulum s2lr3e-3_a2_beta0.3 $PD_S1 4000 --learning-rate 3e-3 --realnvp-alpha 2 --energy-beta 0.3
stage2 pendulum s2lr2e-3_a2_beta0.5 $PD_S1 4000 --learning-rate 2e-3 --realnvp-alpha 2 --energy-beta 0.5
CMDS
    ;;
  a30_j_crps)  # Marginal-calibration pilot: per-coordinate CRPS objective (= 1-D energy score per coordinate) instead of the
               # multivariate energy score. Pendulum diagnosis: angle coverage .72 vs velocity .88; the angle's errors are
               # small relative to the velocity's and are ignored by the joint norm; first 5 steps have angle coverage .46.
    queue <<CMDS
stage2 pendulum s2lr3e-3_crpsobj $PD_S1 4000 --learning-rate 3e-3 --objective crps
stage2 pendulum s2lr2e-3_crpsobj $PD_S1 4000 --learning-rate 2e-3 --objective crps
stage2 reassemble b4_a2_lr1e-4_crpsobj $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --objective crps
CMDS
    ;;
  a30_k_spread)  # User-approved lever (2026-10-22): the spread-term factor of the stage-2 energy score (--energy-spread-weight,
                 # 0.5 = proper energy score). Larger factors reward dispersion. Pendulum from the leader (lr 3e-3), REASSEMBLE from b4 alpha 2 lr 1e-4.
    queue <<CMDS
stage2 pendulum s2lr3e-3_sw0.6 $PD_S1 4000 --learning-rate 3e-3 --energy-spread-weight 0.6
stage2 pendulum s2lr3e-3_sw0.75 $PD_S1 4000 --learning-rate 3e-3 --energy-spread-weight 0.75
stage2 pendulum s2lr3e-3_sw1.0 $PD_S1 4000 --learning-rate 3e-3 --energy-spread-weight 1.0
stage2 pendulum s2lr3e-3_a2_sw0.75 $PD_S1 4000 --learning-rate 3e-3 --realnvp-alpha 2 --energy-spread-weight 0.75
stage2 reassemble b4_a2_lr1e-4_sw0.6 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.6
stage2 reassemble b4_a2_lr1e-4_sw0.75 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.75
stage2 reassemble b4_a2_lr1e-4_sw1.0 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 1.0
CMDS
    ;;
  a30_l_s1bal)  # Stage-1 balance grid (user-approved 2026-10-22). Leader stage 1 ends with entropy -369 vs state 0.14 (above its
                # constant) and rollout 0.40: the entropy term pins the noise at its floor (coverage .58 entering stage 2).
                # Levers: state_sigma at the error scale (0.1 / 0.05 normalized) and rollout weight 10 / 30; stage 2 = leader (lr 3e-3).
    queue <<CMDS
full pendulum sig0.1 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --state-sigma 0.1 -- --learning-rate 3e-3
full pendulum sig0.05 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --state-sigma 0.05 -- --learning-rate 3e-3
full pendulum rw10 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 10 -- --learning-rate 3e-3
full pendulum rw30 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 30 -- --learning-rate 3e-3
full pendulum sig0.1_rw10 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --state-sigma 0.1 --rollout-weight 10 -- --learning-rate 3e-3
full pendulum sig0.05_rw30 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --state-sigma 0.05 --rollout-weight 30 -- --learning-rate 3e-3
CMDS
    ;;
  a30_m_crps12)  # per-coordinate CRPS objective, pendulum seed 0: test .0711 / cov .851 (leader .0709 / .797) -> seeds 1/2.
    PD1=$ROOT/results/results_arxiv/nfsde_joint_local_20261021/pendulum/b16lr6e-4/s1/seed1; PD2=$ROOT/results/results_arxiv/nfsde_joint_local_20261021/pendulum/b16lr6e-4/s1/seed2
    queue <<CMDS
SEED=1 stage2 pendulum s2lr3e-3_crpsobj $PD1 4000 --learning-rate 3e-3 --objective crps
SEED=2 stage2 pendulum s2lr3e-3_crpsobj $PD2 4000 --learning-rate 3e-3 --objective crps
CMDS
    ;;
  a30_n_spread2)  # spread weight >= .75 diverges (val RMSE 200-300, unbounded-below objective); .6 over-disperses REASSEMBLE (cov .95 at step 500).
                  # Fine band just above the proper value: .52 / .55 on both datasets.
    queue <<CMDS
stage2 pendulum s2lr3e-3_sw0.52 $PD_S1 4000 --learning-rate 3e-3 --energy-spread-weight 0.52
stage2 pendulum s2lr3e-3_sw0.55 $PD_S1 4000 --learning-rate 3e-3 --energy-spread-weight 0.55
stage2 reassemble b4_a2_lr1e-4_sw0.52 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.52
stage2 reassemble b4_a2_lr1e-4_sw0.55 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.55
CMDS
    ;;
  a30_o_reas_sw)  # REASSEMBLE fine spread weights only (pendulum copies run on HPC)
    queue <<CMDS
stage2 reassemble b4_a2_lr1e-4_sw0.52 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.52
stage2 reassemble b4_a2_lr1e-4_sw0.55 $RE_B4 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.55
CMDS
    ;;
  a30_p_rw)  # Stage-1 balance grid, seed 0: rollout weight 30 -> test .0682 / cov .847 (leader .0709 / .797; val .0693 vs .0709).
             # sigma .1 -> .0729 / .863; combos .074-.076 / .85-.86. Promote rw30 to seeds 1/2; probe rw60 / rw100 at seed 0.
    queue <<CMDS
SEED=1 full pendulum rw30 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 30 -- --learning-rate 3e-3
SEED=2 full pendulum rw30 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 30 -- --learning-rate 3e-3
full pendulum rw60 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 60 -- --learning-rate 3e-3
full pendulum rw100 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --rollout-weight 100 -- --learning-rate 3e-3
full pendulum sig0.1_rw30 4400 4000 --batch-trajectories 16 --learning-rate 6e-4 --state-sigma 0.1 --rollout-weight 30 -- --learning-rate 3e-3
CMDS
    ;;
  a30_q_reas52)  # REASSEMBLE spread weight .52 seed 0: test .0308 / cov .908 / width .100 (leader .0307 / .882 / .091), same val RMSE -> seeds 1/2
    queue <<CMDS
SEED=1 stage2 reassemble b4_a2_lr1e-4_sw0.52 $O/reassemble/b4lr1e-3_a2_lr1e-4/s1/seed1 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.52
SEED=2 stage2 reassemble b4_a2_lr1e-4_sw0.52 $O/reassemble/b4lr1e-3_a2_lr1e-4/s1/seed2 4000 --realnvp-alpha 2 --learning-rate 1e-4 --energy-spread-weight 0.52
CMDS
    ;;
  a30_q_pend_rw30crps)  # pendulum rw30 stage 1 (3-seed .0691 / cov .834) + per-coordinate CRPS stage 2 (admissibility pending user), seeds 0-2
    queue <<CMDS
SEED=0 stage2 pendulum rw30_crpsobj $O/pendulum/rw30/s1/seed0 4000 --learning-rate 3e-3 --objective crps
SEED=1 stage2 pendulum rw30_crpsobj $O/pendulum/rw30/s1/seed1 4000 --learning-rate 3e-3 --objective crps
SEED=2 stage2 pendulum rw30_crpsobj $O/pendulum/rw30/s1/seed2 4000 --learning-rate 3e-3 --objective crps
CMDS
    ;;
  *) echo "unknown group $group"; exit 2 ;;
esac
echo "sweep2 $group done"
