#!/usr/bin/env bash
#SBATCH --job-name=latency_filters
#SBATCH --account=Mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/analyses/latency_filters/job_%j.out
ROOT=/home/anon/bayesinf_jax; cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name --format=csv,noheader
/home/anon/myenv_312/bin/python results/results_normflowsde_paper/analyses/latency_filters/measure_reference_filters.py --repeats 30
