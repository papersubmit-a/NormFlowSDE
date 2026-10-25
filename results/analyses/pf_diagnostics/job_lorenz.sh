#!/usr/bin/env bash
#SBATCH --job-name=pf_diag_lz
#SBATCH --account=Mitarbeiter
#SBATCH --partition=pro6000
#SBATCH --qos=pro6000-gpu
#SBATCH --gres=gpu:pro6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=07:00:00
# PF diagnostics on the GMM campaign (evaluation only): particle count + ESS; learned vs true measurement density
# (pendulum). Jitter re-selection skipped: the campaign selected checkpoint and jitter jointly.
set -uo pipefail
ROOT=/home/anon/bayesinf_jax; cd $ROOT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
/home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.pf_diagnostics \
  --cells $ROOT/results/results_normflowsde_paper/analyses/pf_diagnostics/cells.json --output $ROOT/results/results_normflowsde_paper/analyses/pf_diagnostics \
  --data-root $ROOT/results/results_normflowsde_paper/datasets --parts particles measurement --datasets lorenz96 --chunk 8
