#!/usr/bin/env bash
#SBATCH --job-name=csv_pipeline18
#SBATCH --partition=norm
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline18/logs/pipeline18-%j.out
#SBATCH --error=/data/KolmogorovLab/srinivasanbd/results/pipeline18/logs/pipeline18-%j.out

set -euo pipefail

PROJECT_DIR=/data/KolmogorovLab/srinivasanbd/complex-SV
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

cd "$PROJECT_DIR"
echo "[$(date --iso-8601=seconds)] Starting pipeline18 on $(hostname)"
echo "Job ID: ${SLURM_JOB_ID:-unknown}"
echo "CPU allocation: ${SLURM_CPUS_PER_TASK:-16} threads"

DEVICE=cpu PLOT_GENOMES=1 scripts/run_pipeline18_v3_neural.sh

echo "[$(date --iso-8601=seconds)] Pipeline18 completed"
