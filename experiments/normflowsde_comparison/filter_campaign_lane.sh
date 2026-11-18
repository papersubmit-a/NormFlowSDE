#!/usr/bin/env bash
# One GPU lane of the all-filtering campaign (filter_campaign.py lane). Lanes share one task
# state; each lane is a chain of jobs (afterany), so a follow-up job resumes interrupted tasks.
set -euo pipefail

: "${FC_ROOT:?Missing campaign source root}"
: "${FC_PYTHON:?Missing training interpreter}"
: "${FC_CAMPAIGN:?Missing campaign directory}"

cd "$FC_ROOT"
export PYTHONPATH="$FC_ROOT"
export MPLCONFIGDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/filter-campaign-mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
unset MLFLOW_TRACKING_URI
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
# shellcheck disable=SC2086  # FC_LANE_ARGS is a frozen, space-separated option list.
exec "$FC_PYTHON" -m experiments.normflowsde_comparison.filter_campaign lane --campaign "$FC_CAMPAIGN" ${FC_LANE_ARGS:-}
