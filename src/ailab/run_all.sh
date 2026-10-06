#!/bin/bash
# Submit the complete pipeline on AI-Lab as dependent Slurm jobs:
#
#   1 data (steps 01-05) ──┬─> 2 classical models (5 CPU tasks in parallel) ──┐
#                          ├─> 3 deep models, benchmark A (8 GPU tasks)  ──────┼─> 4 evaluation (steps 10-13)
#                          └─> 3 deep models, benchmark B (3 GPU tasks)  ──────┘
#
#   bash src/ailab/run_all.sh                              # everything
#   bash src/ailab/run_all.sh --from-step step04_radar     # resume the data job at a step
#   SKIP_DATA=1 bash src/ailab/run_all.sh                  # data (steps 01-05) already done: models + evaluation
#
# Model jobs only start if the data job succeeded (afterok). Evaluation starts when all model jobs have
# ended, even if one of them failed (afterany): it scores the models that exist and lists the missing ones.
# Jobs whose dependency can never be met are cancelled automatically (--kill-on-invalid-dep).
set -euo pipefail
mkdir -p logs
DEP=()
if [ "${SKIP_DATA:-0}" != "1" ]; then
  J1=$(sbatch --parsable src/ailab/01_data.sbatch "$@")
  DEP=(--dependency=afterok:$J1 --kill-on-invalid-dep=yes)
  echo "data job:       $J1"
fi
J2=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} src/ailab/02_classical.sbatch)
J3=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} src/ailab/03_deep_gpu.sbatch)
J4=$(BENCH=B sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=0-2 src/ailab/03_deep_gpu.sbatch)
J5=$(sbatch --parsable --dependency=afterany:$J2:$J3:$J4 --kill-on-invalid-dep=yes src/ailab/04_evaluate.sbatch)
echo "classical:      $J2   (array 0-4)"
echo "deep A / B:     $J3 / $J4"
echo "evaluation:     $J5"
echo "watch with: squeue --me      logs in logs/"
