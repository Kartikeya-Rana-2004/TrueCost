#!/bin/bash
#SBATCH --job-name=truecost-12k-smoke
#SBATCH --partition=standard
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=large12000-smoke-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
unset APPTAINER_BIND APPTAINER_BINDPATH SINGULARITY_BIND SINGULARITY_BINDPATH
export PYTHONDONTWRITEBYTECODE=1
python campaign.py run-smoke
