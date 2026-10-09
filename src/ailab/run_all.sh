#!/bin/bash
# Submit the complete pipeline on AI-Lab as dependent Slurm jobs:
#
#   1 data (steps 01-05) ──┬─> 2 classical models (5 CPU tasks in parallel) ─────────────────┐
#                          ├─> 3 deep A (8 GPU tasks) ──> 3 deep A, attempt 2 (resumes) ──────┼─> 4 evaluation
#                          └─> 3 deep B (6 GPU tasks) ──> 3 deep B, attempt 2 (resumes) ──────┘   (steps 10-13)
#
#   bash src/ailab/run_all.sh                                   # everything
#   bash src/ailab/run_all.sh --from-step step04_radar          # resume the data job at a step
#   SKIP_DATA=1 bash src/ailab/run_all.sh                       # data (steps 01-05) already done
#   SKIP_DATA=1 SKIP_CLASSICAL=1 bash src/ailab/run_all.sh      # only deep models + evaluation
#   ATTEMPTS=3 ...                                              # more resume attempts for deep models
#
# Model jobs start only if the data job succeeded (afterok). A deep attempt starts after the previous
# attempt ended (afterany) and finishes in seconds for models that are already done. Evaluation starts
# when everything has ended, even if a model failed: it scores the models that exist and lists the rest.
# Jobs whose dependency can never be met are cancelled automatically (--kill-on-invalid-dep).
set -euo pipefail
mkdir -p logs
ATTEMPTS=${ATTEMPTS:-2}
DEP=()
if [ "${SKIP_DATA:-0}" != "1" ]; then
  J1=$(sbatch --parsable src/ailab/01_data.sbatch "$@")
  DEP=(--dependency=afterok:$J1 --kill-on-invalid-dep=yes)
  echo "data job:        $J1"
fi
WAIT=()
if [ "${SKIP_CLASSICAL:-0}" != "1" ]; then
  J2=$(sbatch --parsable ${DEP[@]+"${DEP[@]}"} src/ailab/02_classical.sbatch)
  WAIT+=("$J2")
  echo "classical:       $J2   (array 0-4)"
fi
for BENCH in A B; do
  if [ "$BENCH" = "A" ]; then ARR=0-7; else ARR=0-5; fi
  PREV=$(BENCH=$BENCH sbatch --parsable ${DEP[@]+"${DEP[@]}"} --array=$ARR src/ailab/03_deep_gpu.sbatch)
  echo "deep $BENCH attempt 1: $PREV   (array $ARR)"
  for ((a = 2; a <= ATTEMPTS; a++)); do
    PREV=$(BENCH=$BENCH sbatch --parsable --dependency=afterany:$PREV --kill-on-invalid-dep=yes \
           --array=$ARR src/ailab/03_deep_gpu.sbatch)
    echo "deep $BENCH attempt $a: $PREV   (resumes unfinished models)"
  done
  WAIT+=("$PREV")
done
J5=$(sbatch --parsable --dependency=afterany:$(IFS=:; echo "${WAIT[*]}") --kill-on-invalid-dep=yes \
     src/ailab/04_evaluate.sbatch)
echo "evaluation:      $J5"
echo "watch with: squeue --me      logs in logs/"
