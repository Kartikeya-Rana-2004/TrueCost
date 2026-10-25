#!/bin/bash
#SBATCH --job-name=truecost-12000
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=large12000-%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
: "${ROAR_OLLAMA_BIN:?Set staged Ollama binary}"
: "${ROAR_TOOL_IMAGE:?Set staged SIF}"
: "${OLLAMA_MODELS:?Set model directory}"
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
export PYTHONDONTWRITEBYTECODE=1
python campaign.py collect-shard --index "${SLURM_ARRAY_TASK_ID:?}"
