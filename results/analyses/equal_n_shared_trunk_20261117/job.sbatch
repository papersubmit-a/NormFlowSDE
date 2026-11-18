#!/bin/bash
#SBATCH --job-name=equal_n_trunk
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/analyses/equal_n_shared_trunk_20261117/job_%j.out
# Equal sample / particle count sweep for the models of the current paper tables (2026-11-17): all eight learned
# methods, four datasets, three seeds, n in {32, 64, 128, 200, 500, 1000, 2000}. At each n a direct estimator draws
# n posterior paths and a particle filter runs n particles and returns n draws, so every method is compared at the
# same count (the table protocol gives the particle filters 500 / 1000 particles on the pendulum / Lorenz-96 but
# scores 128 / 200 draws). Replaces ../equal_n (2026-10-22, four methods, superseded checkpoints).
# sources.json is a copy of ../latency_gpu_20261116_shared_trunk/sources.json (the run directory of every table cell).
# Counts are the outer loop, ascending, so the small counts finish first; rows.jsonl is appended per cell and the
# job is resumable (finished cells are skipped). Latency is timed for seed 0 only.
# Sized from job 35895 (same script, eight methods, four datasets, seed 0, protocol counts: 18 min, 1.33 cores,
# 3.9 GB peak at n = 500 on REASSEMBLE): memory scales with n -> ~16 GB at n = 2000, 32 GB requested; 672 cells at
# an estimated 1 to 2 min each -> 11 to 22 h, 36 h requested. Check sacct afterwards.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; O=results/analyses/equal_n_shared_trunk_20261117
cd $ROOT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
run() {
  /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
    NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax "$@"
}
for n in 32 64 128 200 500 1000 2000; do
  echo "== n=$n $(date -Is)"
  run /opt/venv/bin/python -m experiments.normflowsde_comparison.inference_cost --sources $O/sources.json --output $O \
    --methods normflowsde latent_nsf latent_sde sigker_sde nf_dpf nsf flowsde normflowsde_pf \
    --counts $n --seeds 0 1 2 --repeats 30 --chunk 4
  echo "== n=$n rc=$?"
done
echo "$(date -Is) equal_n sweep done"
