#!/bin/bash
#SBATCH --job-name=latency_fast
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/anon/NormFlowSDE/results/analyses/latency_gpu_20261116_shared_trunk/fast_sampler_%j.out
# Latency of a rewritten inference rollout of the shared-trunk NormFlowSDE (same weights), fast_sampler.py, on the
# L40S of the latency table. Sizing as job.sbatch (35895 used 1.33 cores and 3.9 GB for all eight methods).
set -euo pipefail
ROOT=/home/anon/NormFlowSDE; O=results/analyses/latency_gpu_20261116_shared_trunk; cd $ROOT
nvidia-smi --query-gpu=name --format=csv,noheader
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax \
  /opt/venv/bin/python $O/fast_sampler.py --sources $O/sources.json --output $O/fast_sampler.json \
  --specs pendulum:128 lorenz96:200 aloha:200 reassemble:500
