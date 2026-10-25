#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --exclude=gpunode254
#SBATCH --job-name=sw2_a30_l_s1bal
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/nfsde_sweep2_20261021/job_a30_l_s1bal_%j.out
# 2026-10-22 coverage round: pendulum stage-1 balance grid (sweep_l.sh group a30_l_s1bal), 3 in parallel.
exec /home/anon/bayesinf_jax/results/results_normflowsde_paper/provenance/nfsde_sweep2_20261021/sweep_l.sh a30_l_s1bal 3
