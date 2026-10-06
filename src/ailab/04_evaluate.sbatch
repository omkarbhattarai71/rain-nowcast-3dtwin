#!/bin/bash
# Job 4 - steps 10-13 after all models: hybrids, evaluation, attenuation, leakage audit.
# Runs even if some model jobs failed (afterany); missing models are reported, not fatal.
#   sbatch --dependency=afterany:<model job ids> src/ailab/04_evaluate.sbatch
#SBATCH --job-name=rainnow-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00

set -euo pipefail
mkdir -p logs
SIF=${SIF:-rainnow.sif}
singularity exec --env-file .env "$SIF" python src/run.py \
    step10_hybrids step11_evaluate step12_attenuation step13_leakage_audit "$@"
