#!/bin/bash
#SBATCH --job-name=p27_chr_5fold
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-4%5
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline27/logs/fivefold-%A_%a.out

set -euo pipefail
root=/data/KolmogorovLab/srinivasanbd
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
out="$root/results/pipeline27/fivefold/runs/fold_$SLURM_ARRAY_TASK_ID"
mkdir -p "$out"
cd "$root"
python results/pipeline27/chromosome_model.py \
  --split grouped5 \
  --fold "$SLURM_ARRAY_TASK_ID" \
  --unlabeled-weight 0.10 \
  --output "$out"
