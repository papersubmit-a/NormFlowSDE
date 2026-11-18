#!/bin/bash
#SBATCH --job-name=direct_smoke
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/smoke_%j.out
#SBATCH --error=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110/logs/smoke_%j.err
# Container smoke test: both new direct methods on the smallest (pendulum) and largest (lorenz96) dataset.
set -euo pipefail
P=/home/anon/NormFlowSDE/results/provenance/direct_baselines_20261110
cd /home/anon/NormFlowSDE
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
RUN=/home/anon/bayesinf_jax/container/run_in_container.sh
ENV=(env PYTHONPATH=/home/anon/NormFlowSDE PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg JAX_PLATFORMS=cpu
     OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax)
rm -rf results/smoke_direct_20261110
for m in latent_sde latent_nsf; do for d in pendulum lorenz96; do
  "$RUN" -- "${ENV[@]}" bash "$P/run_task.sh" $m $d 0 --smoke
done; done
echo SMOKE_OK
