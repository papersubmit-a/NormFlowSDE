#!/bin/bash
# Export the small robot-view bundle (scripts/export_robot_view.py) from the frozen NormFlowSDE runs. CPU only.
#   cd ~/NormFlowSDE && sbatch scripts/slurm/export_robot_view.sbatch
#SBATCH --job-name=robot_view_export
#SBATCH --account=mitarbeiter
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:40:00
#SBATCH --output=logs/robot_view_export_%j.out
#SBATCH --error=logs/robot_view_export_%j.out
set -euo pipefail
cd "$HOME/NormFlowSDE"
export CONTAINER_IMAGE="$HOME/containers/bayesinf-jax-20261102.sqsh"
"$HOME/bayesinf_jax/container/run_in_container.sh" -- env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 \
  /opt/venv/bin/python scripts/export_robot_view.py --out results/robot_view --draws 30
