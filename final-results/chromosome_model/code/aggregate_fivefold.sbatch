#!/bin/bash
#SBATCH --job-name=p27_agg_5fold
#SBATCH --partition=norm
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline27/logs/aggregate-fivefold-%j.out
set -euo pipefail
cd /data/KolmogorovLab/srinivasanbd
python results/pipeline27/aggregate.py --split fivefold
