#!/bin/bash
#SBATCH --job-name=latency_trunk
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --output=/home/anon/NormFlowSDE/results/analyses/latency_gpu_20261116_shared_trunk/job_%j.out
# GPU latency of the shared-trunk campaign models, all eight learned methods on the four datasets, one method at a
# time on one NVIDIA L40S (2026-11-16). Protocol as ../latency_gpu_20261113/job.sbatch (paper Appendix "Latency"):
# selected seed-0 checkpoint, jitted batch-size-one full-window calls with inputs on the device, the first
# (compiling) call excluded, median of 30 further calls on different test windows; direct estimators draw the
# protocol M, particle filters run with the protocol particle count. inference_cost.py also re-scores the test
# windows with those draws (rows.jsonl "accuracy"), as a load check.
# Sized from the previous full run of the same script (job 35428, same four datasets, L40S): 18 min 46 s,
# 1.33 cores on average, 3.2 GB peak RSS -> 2 cores, 4 GB, 40 min.
set -euo pipefail
ROOT=/home/anon/NormFlowSDE; O=results/analyses/latency_gpu_20261116_shared_trunk
cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
run() {
  /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax "$@"
}
for spec in "pendulum 128 500" "lorenz96 200 1000" "aloha 200 200" "reassemble 500 500"; do
  set -- $spec
  run /opt/venv/bin/python -m experiments.normflowsde_comparison.inference_cost --sources $O/sources.json --output $O \
    --datasets $1 --methods normflowsde latent_nsf latent_sde sigker_sde --counts $2 --seeds 0 --repeats 30
  run /opt/venv/bin/python -m experiments.normflowsde_comparison.inference_cost --sources $O/sources.json --output $O \
    --datasets $1 --methods nf_dpf nsf flowsde normflowsde_pf --counts $3 --seeds 0 --repeats 30
done
echo "$(date -Is) latency done"
