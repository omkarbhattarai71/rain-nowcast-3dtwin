#!/bin/bash
# Submit the complete pipeline on AI-Lab as three dependent Slurm jobs (A and B benchmarks).
#   bash src/ailab/run_all.sh
set -euo pipefail
mkdir -p logs
J1=$(sbatch --parsable src/ailab/01_data_and_classical.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 src/ailab/02_deep_gpu.sbatch)
J3=$(BENCH=B sbatch --parsable --dependency=afterok:$J1 --array=0-2 src/ailab/02_deep_gpu.sbatch)
J4=$(sbatch --parsable --dependency=afterok:$J2:$J3 src/ailab/03_evaluate.sbatch)
echo "submitted: data=$J1 deepA=$J2 deepB=$J3 eval=$J4   (watch with: squeue --me)"
