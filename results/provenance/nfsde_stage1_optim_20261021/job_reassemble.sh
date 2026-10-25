#!/usr/bin/env bash
#SBATCH --job-name=nfsde_opt_reas
#SBATCH --account=Mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --exclude=gpunode254
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/nfsde_stage1_optim_20261021/job_reassemble_%j.out
# Stage-1 optimizer pilot, seed 0 (GMM data). Diagnosis: pendulum velocity leaks 59% of measurement noise
# (NF-DPF 31%); REASSEMBLE posterior trails motion by ~2.6 steps. Both point to an under-fitted drift.
results/results_normflowsde_paper/provenance/nfsde_stage1_optim_20261021/pilot.sh reassemble 4 b4:--batch-trajectories,4 lr1e-3:--learning-rate,1e-3 b4lr1e-3:--batch-trajectories,4,--learning-rate,1e-3 b8:--batch-trajectories,8
