#!/usr/bin/env bash
#SBATCH --job-name=gmm_lane_p6k
#SBATCH --account=Mitarbeiter
#SBATCH --partition=pro6000
#SBATCH --qos=pro6000-gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=11:20:00
# pro6000 lane (must end before the daily 08:00-19:00 daytime_damian reservation) of the GMM observation-noise campaign 20261020_gmm (shares plan/state with the a30/l40s lanes).
# GPUs 0/1 of gpunode142 may also run unscheduled training of another user: if the allocated GPU already
# holds > 20 GB, cap JAX at 10% memory per task and run at most 4 tasks (as on 2026-10-19).
set -uo pipefail
ROOT=/home/anon/bayesinf_jax
L=$ROOT/results/results_arxiv/filter_campaign_20261020_gmm
cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "${used:-0}" -gt 20000 ]; then
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10; ARGS="--slots 4 --capacity 4"; echo "shared GPU (${used} MiB used): 4 slots, 10% memory each"
else
  export XLA_PYTHON_CLIENT_PREALLOCATE=false; ARGS="--slots 8 --capacity 16"; echo "free GPU: 8 slots"
fi
exec /home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.filter_campaign lane --campaign $L $ARGS --fallback-hours 11.3
