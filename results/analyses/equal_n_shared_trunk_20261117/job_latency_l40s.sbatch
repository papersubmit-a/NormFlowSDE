#!/bin/bash
#SBATCH --job-name=equal_n_lat
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/analyses/equal_n_shared_trunk_20261117/latency_l40s/job_%j.out
# GPU latency at equal sample / particle counts (2026-11-17), the timing counterpart of job_cpu.sbatch: the CPU
# sweep gives accuracy and coverage at each count for three seeds, this job gives the latency on one NVIDIA L40S.
# Protocol of the latency table (../latency_gpu_20261116_shared_trunk/job.sbatch): seed-0 checkpoint, jitted
# batch-size-one full-window calls, compiling call excluded, median of 30 further calls. Here n sets both the
# particle count and the number of returned draws, n in {32, 64, 128, 200, 500, 1000, 2000}, all eight learned
# methods on the four datasets. Rows go to latency_l40s/rows.jsonl, separate from the CPU rows (<dataset>/rows.jsonl).
# Only "latency" of these rows is meant for the paper; "accuracy" is the seed-0 load check.
# Submitted with --dependency=afterany:<CPU array job> so it starts after the CPU sweep.
# Sized from job 35895 (same script, 32 cells at the protocol counts on an L40S: 18 min, 3.9 GB peak at n = 500 on
# REASSEMBLE): 224 cells, memory scales with n -> ~16 GB at n = 2000, 32 GB and 12 h requested. Resumable.
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
  run /opt/venv/bin/python -m experiments.normflowsde_comparison.inference_cost --sources $O/sources.json \
    --output $O/latency_l40s \
    --methods normflowsde latent_nsf latent_sde sigker_sde nf_dpf nsf flowsde normflowsde_pf \
    --counts $n --seeds 0 --repeats 30 --chunk 4
  echo "== n=$n rc=$?"
done
echo "$(date -Is) equal_n L40S latency done"
