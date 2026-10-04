#!/bin/bash
# Steps 10-13 after all models are trained: hybrids, evaluation, attenuation, leakage audit.
#   sbatch --dependency=afterok:<deep_job_id> src/ailab/03_evaluate.sbatch
#SBATCH --job-name=rainnow-eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=06:00:00

set -euo pipefail
mkdir -p logs
SIF=${SIF:-rainnow.sif}
singularity exec --env-file .env "$SIF" python src/run.py \
    step10_hybrids step11_evaluate step12_attenuation step13_leakage_audit "$@"
