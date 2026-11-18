#!/bin/bash
#SBATCH --job-name=nfpf_fixed
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=3G
#SBATCH --time=01:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/pendulum_fixed_scale_%A_%a.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/shared_trunk_20261114/logs/pendulum_fixed_scale_%A_%a.out
# Pendulum ablation of NormFlowSDE-PF without the learned per-coordinate scale (2026-11-16), for the shared-trunk
# model: the command of run_task.sh (normflowsde_pf, pendulum) without --learned-scale, written to
# normflowsde_pf/pendulum/seed<k>/fixed_scale as in the previous campaign (provenance/normflowsde_pf_20261031).
# The learned-scale runs took 15.5 min of training on CPU.   sbatch --array=0-2 pendulum_fixed_scale.sbatch
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; cd $ROOT; SEED=$SLURM_ARRAY_TASK_ID
OUT=results/shared_trunk_20261114/normflowsde_pf/pendulum/seed$SEED/fixed_scale
[ -f $OUT/summary.json ] && { echo "done already"; exit 0; }
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
/home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax \
  /opt/venv/bin/python -m experiments.normflowsde_comparison.train_state_filter --method normflowsde --shared-trunk \
  --hidden-size 63 --diffusion-hidden-size 16 --alpha 1.0 --dataset pendulum --data-cache results/datasets/pendulum --seed $SEED \
  --learning-rate 3e-4 --batch-windows 16 --hidden-depth 2 --flow-layers 2 --steps 8400 --eval-every 400 \
  --selection-jitters 0,0.005,0.01,0.02,0.03,0.05,0.075,0.1,0.15,0.2 --measurement-hidden 16 --measurement-depth 2 --measurement-layers 2 \
  --output $OUT
