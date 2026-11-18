#!/bin/bash
#SBATCH --job-name=nsf_count
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/count_nsf_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/count_nsf_%j.err
set -euo pipefail
cd /home/anon/NormFlowSDE
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax \
  /opt/venv/bin/python /home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/count_nsf.py
