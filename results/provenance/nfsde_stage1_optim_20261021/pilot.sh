#!/usr/bin/env bash
# NormFlowSDE stage-1 optimizer pilot (seed 0), 2026-10-21. Usage: pilot.sh <dataset> <parallel> <tag:flags>...
# Each variant changes the stage-1 learning rate and/or trajectories per update; the loss terms and weights,
# update budget, architecture and the stage-2 recipe (energy score, learned noise scale, lr 3e-4, 4000 updates)
# are unchanged. Selection of the variant is by seed-0 validation RMSE.
set -uo pipefail
ROOT=/home/anon/bayesinf_jax; cd $ROOT
O=$ROOT/results/results_normflowsde_paper/nfsde_stage1_optim_20261021; PY=/home/anon/myenv_312/bin/python
ds=$1; par=$2; shift 2
case $ds in
  pendulum)   DATA=$ROOT/results/results_normflowsde_paper/datasets/pendulum; STEPS=4400; VAL=48; EXTRA=--allow-data-change ;;
  reassemble) DATA=$ROOT/results/results_normflowsde_paper/datasets/reassemble; STEPS=7000; VAL=300; EXTRA=--allow-data-change ;;
  lorenz96)   DATA=$ROOT/results/results_normflowsde_paper/datasets/lorenz96; STEPS=14500; VAL=200; EXTRA= ;;
esac
export ROOT O PY ds DATA STEPS VAL EXTRA PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
printf '%s\n' "$@" | xargs -P $par -I{} bash -c '
  tag=${1%%:*}; flags=${1#*:}; flags=${flags//,/ }
  $PY -m experiments.normflowsde_comparison.train_normflowsde_filter --dataset $ds --seed 0 --train --data-cache $DATA \
    --reference-metadata $ROOT/experiments/normflowsde_comparison/filter_campaign_references/$ds/seed0.json \
    --train-steps $STEPS --validation-windows $VAL --allow-source-drift $EXTRA $flags \
    --output $O/$ds/$tag/s1/seed0 > $O/logs/${ds}_${tag}_s1.out 2>&1 || { echo "$ds $tag s1 FAILED"; exit 0; }
  $PY -m experiments.normflowsde_comparison.improve_pendulum_normflowsde --dataset $ds --seed 0 --data-cache $DATA \
    --init-run $O/$ds/$tag/s1/seed0 --objective energy --learning-rate 3e-4 --learned-scale --steps 4000 \
    --output $O/$ds/$tag/s2/seed0 > $O/logs/${ds}_${tag}_s2.out 2>&1; echo "$ds $tag s2 exit $?"' _ {}
echo "pilot $ds done"
