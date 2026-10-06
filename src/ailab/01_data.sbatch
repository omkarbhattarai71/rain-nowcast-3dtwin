#!/bin/bash
# Job 1 - data: steps 01-05 (download, ground truth, channels, radar, tabular features). CPU only.
#   sbatch src/ailab/01_data.sbatch                              # all data steps
#   sbatch src/ailab/01_data.sbatch --from-step step04_radar     # resume at a step
#SBATCH --job-name=rainnow-data
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p logs
SIF=${SIF:-rainnow.sif}
# AWS credentials come from .env in the repository root (never commit it)
singularity exec --env-file .env "$SIF" python src/run.py \
    step01_inventory step02_truth step03_channels step04_radar step05_tabular --workers 16 "$@"
