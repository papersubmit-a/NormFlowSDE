#!/usr/bin/env bash
# Round-5 refresh (2026-10-22): recompute only the pendulum and REASSEMBLE NormFlowSDE cells for the new settings; all other rows carried over from equal_n_gmm_final_20261021.
#SBATCH --job-name=equal_n_final
#SBATCH --account=Mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/anon/bayesinf_jax/results/results_normflowsde_paper/analyses/equal_n/job_%j.out
# Equal sample/particle count sweep matching the final 3-seed table (results_gmm_current_best_20261021).
# Seeds rows.jsonl with the still-valid rows of equal_n_gmm_20261020 (all ALOHA and Lorenz-96 rows from job 32321,
# pendulum baselines), dropping the pendulum/REASSEMBLE NormFlowSDE rows that used superseded checkpoints, then runs
# pendulum and REASSEMBLE (missing REASSEMBLE baselines; NormFlowSDE with the final checkpoints). Resumable.
set -uo pipefail
ROOT=/home/anon/bayesinf_jax; cd $ROOT
O=$ROOT/results/results_normflowsde_paper/analyses/equal_n; OLD=$ROOT/results/results_arxiv/equal_n_gmm_final_20261021
export PYTHONPATH=$ROOT PYTHONUNBUFFERED=1 MPLBACKEND=Agg XLA_FLAGS="--xla_gpu_autotune_level=0"
CUDNN=$ROOT/results/results_arxiv/lorenz96_segment_finetune_20261019/runtime/cudnn_cu12
export LD_LIBRARY_PATH=$CUDNN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export LD_PRELOAD=$(ls $CUDNN/libcudnn*.so.9 | paste -sd:)
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
[ -f $O/rows.jsonl ] || /home/anon/myenv_312/bin/python - <<'PY'
import json
old=[json.loads(l) for l in open('/home/anon/bayesinf_jax/results/results_arxiv/equal_n_gmm_final_20261021/rows.jsonl')]
keep=[r for r in old if not (r['dataset'] in ('pendulum','reassemble') and r['method']=='normflowsde')]
with open('/home/anon/bayesinf_jax/results/results_normflowsde_paper/analyses/equal_n/rows.jsonl','w') as h:
    for r in keep: h.write(json.dumps(r)+'\n')
print('seeded', len(keep), 'rows; dropped', len(old)-len(keep))
PY
/home/anon/myenv_312/bin/python -m experiments.normflowsde_comparison.inference_cost \
  --sources $O/sources.json --output $O --counts 32 64 128 200 500 1000 2000 --seeds 0 1 2 --repeats 30 --chunk 4 \
  --datasets pendulum reassemble --methods normflowsde
echo "equal_n final done"
