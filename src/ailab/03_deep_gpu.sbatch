#!/bin/bash
# Job 3 - step 09, deep models, one GPU per model (benchmark B: one GPU per model AND variant).
#   sbatch src/ailab/03_deep_gpu.sbatch                        # benchmark A, all 8 models (array 0-7)
#   sbatch --array=0 src/ailab/03_deep_gpu.sbatch              # SAMBA only
#   BENCH=B sbatch --array=0-5 src/ailab/03_deep_gpu.sbatch    # benchmark B: 3 models x {station, radar}
#
# Deadline-aware and resumable: training plans its steps to fit the time limit and keeps time for the
# predictions; state is checkpointed every epoch and predictions are saved per station-year. If a task
# still runs out of time it exits with an error, and submitting it again (run_all.sh does this
# automatically as "attempt 2") continues where it stopped. Finished models are skipped in seconds.
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
# the job must end 10 min before the Slurm limit: the code reserves its own margin on top
LIMIT_S=${LIMIT_S:-43200}
export RAINNOW_DEADLINE=$(( $(date +%s) + LIMIT_S - 600 ))
if [ "$BENCH" = "A" ]; then
  MODELS=(samba samba_nonlinear samba_temporal_only mamba s4d gru tcn patchtst)
  MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
  unset RAINNOW_VARIANT
else
  MODELS=(samba samba mamba mamba gru gru)
  VARIANTS=(_B_station _B_radar _B_station _B_radar _B_station _B_radar)
  MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
  export RAINNOW_VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
fi
echo "training $MODEL${RAINNOW_VARIANT:-} on benchmark $BENCH, deadline $(date -d @$RAINNOW_DEADLINE)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
singularity exec --nv --env-file .env "$SIF" python src/run.py step09_deep --bench "$BENCH" --models "$MODEL" "$@"
