#!/usr/bin/env bash
#SBATCH --job-name=gmm_lane_l40s
#SBATCH --account=Mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
# One GPU lane of the GMM observation-noise campaign 20261020_gmm (pendulum, ALOHA, REASSEMBLE; 99 training tasks).
# Lanes share the frozen plan/state in results/results_normflowsde_paper/provenance/filter_campaign_20261020_gmm/plan; nothing is deleted or overwritten.
set -uo pipefail
ROOT=/home/anon/bayesinf_jax
L=$ROOT/results/results_arxiv/filter_campaign_20261020_gmm
cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name,memory.total --format=csv
mkdir -p $L/slurm_logs
exec /home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.filter_campaign lane --campaign $L --slots 8 --capacity 14 --fallback-hours 72
