#!/usr/bin/env bash
#SBATCH --job-name=model_based
#SBATCH --account=Mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpunode254
# EKF / UKF / bootstrap PF with oracle and misspecified physics (model_based_baselines.py) on the campaign's
# verified pendulum and Lorenz-96 caches; same test windows, draws, particles and scoring as the learned methods.
set -uo pipefail
ROOT=/home/anon/bayesinf_jax
cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
for ds in pendulum lorenz96; do
  /home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.model_based_baselines --dataset $ds \
    --data-cache $ROOT/results/results_arxiv/filter_campaign_20261019_a30/data_cache/$ds \
    --output $ROOT/results/results_normflowsde_paper/model_based_baselines_20261019/$ds > $ROOT/results/results_normflowsde_paper/model_based_baselines_20261019/$ds.log 2>&1 &
done
wait
