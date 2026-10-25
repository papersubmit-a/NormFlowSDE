#!/usr/bin/env bash
#SBATCH --job-name=model_based_gmm
#SBATCH --account=Mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/model_based_baselines_20261022_gmm/%j.out
# EKF / UKF / bootstrap PF reference filters (model_based_baselines.py) on the GMM pendulum cache of the
# 2026-10-20 campaign (the 2026-10-19 run used the Gaussian cache). Lorenz-96 is unchanged: its 20261019 cache is
# byte-identical to results/results_normflowsde_paper/datasets/lorenz96 (sha256 e4c49ec7...).
set -uo pipefail
ROOT=/home/anon/bayesinf_jax
cd $ROOT
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu
/home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.model_based_baselines --dataset pendulum \
  --data-cache $ROOT/results/results_normflowsde_paper/datasets/pendulum \
  --output $ROOT/results/results_normflowsde_paper/filters/pendulum > $ROOT/results/results_normflowsde_paper/provenance/model_based_baselines_20261022_gmm/pendulum_gmmlik.log 2>&1
echo "model_based_gmm done exit $?"
# pendulum/ (first run, 15:1x) used the Gaussian std-0.1 likelihood on GMM data (bug, kept for reference as a likelihood-misspecified set);
# pendulum_gmmlik/ is the corrected run: GMM likelihood exact in the PF, moment-matched Gaussian in EKF/UKF/pf_gauss.
