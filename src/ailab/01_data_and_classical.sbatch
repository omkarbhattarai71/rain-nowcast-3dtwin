#!/bin/bash
# Steps 01-08 (data, radar, tabular, baselines, LightGBM, classical SSMs). CPU only.
# Submit from the repository root:   sbatch src/ailab/01_data_and_classical.sbatch
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
    step01_inventory step02_truth step03_channels step04_radar step05_tabular \
    step06_baselines step07_gbm step08_ssm --workers 16 "$@"
