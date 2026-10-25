#!/usr/bin/env bash
#SBATCH --job-name=weak_conv_ou
#SBATCH --account=Mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
# Trained NormFlowSDE vs a standard OU SDE (weak_convergence_ou.py): fit the transition to non-Gaussian h0-step
# increments with the OU first two moments, then centered / uncentered refinement vs the analytic OU law.

set -uo pipefail
ROOT=/home/anon/bayesinf_jax
cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false MPLBACKEND=Agg
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name,memory.total --format=csv
/home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.weak_convergence_ou \
  --output $ROOT/results/results_normflowsde_paper/analyses/weak_convergence/ou_seed0
