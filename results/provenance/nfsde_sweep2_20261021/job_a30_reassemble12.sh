#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH --exclude=gpunode254
#SBATCH --job-name=sw2_a30_reas12
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/nfsde_sweep2_20261021/job_a30_reassemble12_%j.out
# NormFlowSDE seed-0 pilot of remaining hyperparameter levers (see sweep.sh), group a30_reassemble12, 2 runs in parallel.
exec results/results_normflowsde_paper/provenance/nfsde_sweep2_20261021/sweep.sh a30_reassemble12 2
