#!/bin/bash
#SBATCH --job-name=truecost-v2-pilot
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=v2-pilot-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
export PYTHONDONTWRITEBYTECODE=1
: "${TRUECOST_PYTHON:?Run submission through pilot.py}"
"$TRUECOST_PYTHON" pilot.py collect --index "${SLURM_ARRAY_TASK_ID:?}"
