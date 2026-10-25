#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=gpunode254
#SBATCH --job-name=sw2_a30_o_reas_sw
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/nfsde_sweep2_20261021/job_a30_o_reas_sw_%j.out
# 2026-10-22 coverage round: fine spread weights .52/.55 (sweep_o.sh group a30_o_reas_sw), 4 in parallel.
exec /home/anon/bayesinf_jax/results/results_normflowsde_paper/provenance/nfsde_sweep2_20261021/sweep_o.sh a30_o_reas_sw 2
