#!/usr/bin/env bash
#SBATCH --job-name=csv_asmloc19
#SBATCH --partition=quick
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline19_asm_loc/logs/pipeline19-%j.out
#SBATCH --error=/data/KolmogorovLab/srinivasanbd/results/pipeline19_asm_loc/logs/pipeline19-%j.out

set -euo pipefail

PROJECT_DIR=/data/KolmogorovLab/srinivasanbd/complex-SV
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

cd "$PROJECT_DIR"
echo "[$(date --iso-8601=seconds)] Starting pipeline19 on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"

DEVICE=cuda \
TEACHER_PREDICTIONS="$PROJECT_DIR/../results/pipeline18/candidate_region_classifier_general/cross_fold_predictions.tsv" \
scripts/run_pipeline19_asm_loc.sh

echo "[$(date --iso-8601=seconds)] Pipeline19 completed"
