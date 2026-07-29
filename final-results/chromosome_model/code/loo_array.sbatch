#!/bin/bash
#SBATCH --job-name=p27_chr_loo
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-36%12
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline27/logs/loo-%A_%a.out

set -euo pipefail
root=/data/KolmogorovLab/srinivasanbd
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
sample=$(awk -F '	' -v row="$((SLURM_ARRAY_TASK_ID + 2))" 'NR == row {print $1}' "$root/results/pipeline27/labeled_samples.tsv")
out="$root/results/pipeline27/loo/runs/$sample"
mkdir -p "$out"
cd "$root"
python results/pipeline27/chromosome_model.py \
  --split loo \
  --test-sample "$sample" \
  --unlabeled-weight 0.10 \
  --output "$out"
