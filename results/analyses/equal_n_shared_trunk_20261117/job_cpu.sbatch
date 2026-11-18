#!/bin/bash
#SBATCH --job-name=equal_n_cpu
#SBATCH --account=ACCOUNT
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --array=0-3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/anon/NormFlowSDE/results/analyses/equal_n_shared_trunk_20261117/job_cpu_%A_%a.out
# CPU version of job.sbatch (2026-11-17): the same equal sample / particle count sweep, without a GPU, because the
# accuracy and coverage at each count do not depend on the device and the L40S queue was three days long.
# One array task per dataset, each writing to its own subdirectory (<dataset>/rows.jsonl) so the tasks never append
# to the same file. The "latency" entries of these rows are CPU timings and must not be compared with the GPU
# latencies of the paper; only "accuracy" is used.
# Counts ascending, appended per cell, resumable. CPU cost not measured before: the GPU run of the protocol counts
# took 18 min for all datasets (job 35895); 8 cores, 32 GB (as job.sbatch, ~16 GB expected at n = 2000 on
# REASSEMBLE) and the partition limit of 3 days are requested. Check sacct afterwards.
set -uo pipefail
ROOT=/home/anon/NormFlowSDE; O=results/analyses/equal_n_shared_trunk_20261117
DATASETS=(pendulum lorenz96 aloha reassemble); DS=${DATASETS[$SLURM_ARRAY_TASK_ID]}
cd $ROOT
export CONTAINER_IMAGE=/home/anon/containers/bayesinf-jax-20261102.sqsh
run() {
  /home/anon/bayesinf_jax/container/run_in_container.sh -- env PYTHONPATH=$ROOT PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NORMFLOWSDE_RAW_ROOT=/home/anon/bayesinf_jax "$@"
}
for n in 32 64 128 200 500 1000 2000; do
  echo "== $DS n=$n $(date -Is)"
  run /opt/venv/bin/python -m experiments.normflowsde_comparison.inference_cost --sources $O/sources.json --output $O/$DS \
    --datasets $DS --methods normflowsde latent_nsf latent_sde sigker_sde nf_dpf nsf flowsde normflowsde_pf \
    --counts $n --seeds 0 1 2 --repeats 5 --chunk 4
  echo "== $DS n=$n rc=$?"
done
echo "$(date -Is) $DS equal_n sweep done"
