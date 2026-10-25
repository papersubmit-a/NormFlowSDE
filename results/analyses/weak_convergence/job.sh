#!/usr/bin/env bash
#SBATCH --job-name=weak_conv
#SBATCH --account=Mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# Numerical check of the paper's weak-convergence theorem (weak_convergence_check.py, default settings):
# fixed untrained SDEDriftNet / RealNVPDiffusionNet (alpha 1.5, coupling gain 8: skewed, heavy-tailed noise),
# centered scheme and uncentered control at h = h0/2^k (h0 = 0.1, k = 0..5) vs an EM reference at h0/128.
set -uo pipefail
ROOT=/home/anon/bayesinf_jax
cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name,memory.total --format=csv
/home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.weak_convergence_check \
  --output $ROOT/results/results_normflowsde_paper/analyses/weak_convergence/seed0
