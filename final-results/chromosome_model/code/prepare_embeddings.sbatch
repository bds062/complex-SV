#!/bin/bash
#SBATCH --job-name=p27_chr_embed
#SBATCH --partition=norm
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline27/logs/embed-%j.out

set -euo pipefail
root=/data/KolmogorovLab/srinivasanbd
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export LD_LIBRARY_PATH="$root/envs/env2/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
python_cmd="$root/envs/env2/bin/python"
mkdir -p "$root/results/pipeline27/logs"
cd "$root"
"$python_cmd" results/pipeline27/build_labels.py
"$python_cmd" results/pipeline27/build_chromosome_tabular.py
"$python_cmd" complex-SV/discovery/embed_corpus.py \
  --manifest results/pipeline18/complex_sv_manifest.tsv \
  --cn_checkpoint results/pipeline3/cn_pretrain_chrom/cn_encoder.pt \
  --graph_checkpoint results/pipeline3/sv3/graph_encoder.pt \
  --output_dir results/pipeline27/chromosome_embeddings \
  --candidate_source chromosomes \
  --embedding_normalization none \
  --report_scope all
