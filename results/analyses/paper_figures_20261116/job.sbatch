#!/bin/bash
#SBATCH --job-name=paper_figs
#SBATCH --account=ACCOUNT
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/analyses/paper_figures_20261116/job_%j.out
# Paper figures for the shared-trunk campaign (2026-11-16), CPU only:
#  1. reliability curves of all table runs from their saved posteriors -> analyses/calibration/calibration_shared_trunk.*
#  2. REASSEMBLE measurement-density slices of the four retrained particle filters and implied residuals (seed 0)
#     -> analyses/noise_recovery_shared_trunk/reassemble_measurement_density.*
#  3. the 2x4 noise-recovery grid of the eight table methods -> analyses/noise_recovery_grid/reassemble_shared_trunk_*
# Memory: not measured before; one REASSEMBLE posterior is 300 x 500 x 250 x 7 float32 = 1.05 GB and step 2 holds
# one of them plus its mean, step 1 one plus sorted chunks -> 12 GB requested as a first run; see sacct for the peak.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; cd $ROOT
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
run() { /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax "$@"; }
S=results/analyses/latency_gpu_20261116_shared_trunk/sources.json
echo "== 2 measurement density"; mkdir -p results/analyses/noise_recovery_shared_trunk
run /opt/venv/bin/python -m experiments.normflowsde_comparison.plot_measurement_density_recovery --dataset reassemble --sources $S \
  --methods normflowsde nf_dpf nsf flowsde normflowsde_pf --out results/analyses/noise_recovery_shared_trunk/reassemble
echo "== 3 grid rc=$?"
run /opt/venv/bin/python experiments/normflowsde_comparison/plot_noise_recovery_grid.py --root . --dataset reassemble --sources $S \
  --saved results/analyses/noise_recovery_shared_trunk/reassemble_measurement_density.json \
  --out results/analyses/noise_recovery_grid/reassemble_shared_trunk
echo "== 1 calibration rc=$?"
run /opt/venv/bin/python results/analyses/calibration/plot_calibration_shared_trunk.py
echo "rc=$? $(date -Is) done"
