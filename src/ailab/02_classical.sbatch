#!/bin/bash
# Job 2 - classical models, 5 array tasks running in parallel (CPU):
#   0  step06  baselines + logistic hurdle                          (benchmarks A and B)
#   1  step07  LightGBM main models                                 (A: lgbm_hurdle, lgbm_tweedie)
#   2  step07  LightGBM supervisor features + feature ablations     (A)
#   3  step07  LightGBM radar variants                              (B: lgbm_B_station/radar/nowcast/all)
#   4  step08  Kalman (MLE) + regime-switching HMM                  (fit on A, predict A and B)
#   sbatch src/ailab/02_classical.sbatch               # all five
#   sbatch --array=4 src/ailab/02_classical.sbatch     # only the state-space models
#SBATCH --job-name=rainnow-classical
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --cpus-per-task=32
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-4

set -euo pipefail
mkdir -p logs
SIF=${SIF:-rainnow.sif}
case "$SLURM_ARRAY_TASK_ID" in
  0) ARGS=(step06_baselines) ;;
  1) ARGS=(step07_gbm --bench A --models lgbm_hurdle lgbm_tweedie) ;;
  2) ARGS=(step07_gbm --bench A --models lgbm_supervisor_features gbm_abl_P gbm_abl_PT gbm_abl_PM gbm_abl_PN) ;;
  3) ARGS=(step07_gbm --bench B --models lgbm_B_station lgbm_B_radar lgbm_B_nowcast lgbm_B_all) ;;
  4) ARGS=(step08_ssm) ;;
  *) echo "unknown array task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
echo "classical task $SLURM_ARRAY_TASK_ID: ${ARGS[*]}"
singularity exec --env-file .env "$SIF" python src/run.py "${ARGS[@]}" "$@"
