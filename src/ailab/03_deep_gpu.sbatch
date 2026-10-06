#!/bin/bash
# Job 3 - step 09, deep models, one GPU per model. One array task per model, so they train in parallel:
#   sbatch src/ailab/03_deep_gpu.sbatch                 # all models (array 0-7) on benchmark A
#   sbatch --array=0 src/ailab/03_deep_gpu.sbatch       # SAMBA only
#   BENCH=B sbatch --array=0-2 src/ailab/03_deep_gpu.sbatch   # benchmark B models (samba, mamba, gru)
#SBATCH --job-name=rainnow-deep
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --array=0-7

set -euo pipefail
mkdir -p logs
SIF=${SIF:-rainnow.sif}
BENCH=${BENCH:-A}
if [ "$BENCH" = "A" ]; then
  MODELS=(samba samba_nonlinear samba_temporal_only mamba s4d gru tcn patchtst)
else
  MODELS=(samba mamba gru)
fi
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
echo "training $MODEL on benchmark $BENCH"
singularity exec --nv --env-file .env "$SIF" python src/run.py step09_deep --bench "$BENCH" --models "$MODEL" "$@"
