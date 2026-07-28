#!/usr/bin/env bash
#SBATCH --job-name=csv_asmloc19_cpu
#SBATCH --partition=norm
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/KolmogorovLab/srinivasanbd/results/pipeline19_asm_loc/logs/pipeline19-cpu-%j.out
#SBATCH --error=/data/KolmogorovLab/srinivasanbd/results/pipeline19_asm_loc/logs/pipeline19-cpu-%j.out

set -euo pipefail

PROJECT_DIR=/data/KolmogorovLab/srinivasanbd/complex-SV
OUTPUT_DIR="$PROJECT_DIR/../results/pipeline19_asm_loc"
PIPELINE18="$PROJECT_DIR/../results/pipeline18"
PYTHON_BIN="$PROJECT_DIR/../envs/env2/bin/python"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

cd "$PROJECT_DIR"
echo "[$(date --iso-8601=seconds)] Resuming pipeline19 on $(hostname)"

"$PYTHON_BIN" scripts/asm_loc_pipeline.py train \
    --dataset "$OUTPUT_DIR/chromosome_bins.tsv.gz" \
    --output "$OUTPUT_DIR/asm_loc_model.pt" \
    --epochs 15 \
    --patience 4 \
    --hidden_dim 64 \
    --num_layers 2 \
    --batch_size 32 \
    --device cpu

"$PYTHON_BIN" scripts/asm_loc_pipeline.py predict \
    --dataset "$OUTPUT_DIR/chromosome_bins.tsv.gz" \
    --checkpoint "$OUTPUT_DIR/asm_loc_model.pt" \
    --output "$OUTPUT_DIR/localized_proposals.tsv" \
    --bin_predictions "$OUTPUT_DIR/bin_predictions.tsv.gz" \
    --bin_size 1000000 \
    --batch_size 64 \
    --device cpu

"$PYTHON_BIN" scripts/asm_loc_pipeline.py evaluate \
    --proposals "$OUTPUT_DIR/localized_proposals.tsv" \
    --regions "$PIPELINE18/external_regions.tsv" \
    --output "$OUTPUT_DIR/localization_recall.tsv"

"$PYTHON_BIN" scripts/asm_loc_pipeline.py materialize \
    --manifest "$PIPELINE18/complex_sv_manifest.tsv" \
    --proposals "$OUTPUT_DIR/localized_proposals.tsv" \
    --output "$OUTPUT_DIR/merged_candidate_regions.csv"

"$PYTHON_BIN" training/train_candidate_region_classifier.py \
    --manifest "$PIPELINE18/complex_sv_manifest.tsv" \
    --candidate_regions "$OUTPUT_DIR/merged_candidate_regions.csv" \
    --cn_checkpoint "$PROJECT_DIR/../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt" \
    --graph_checkpoint "$PROJECT_DIR/../results/pipeline3/sv3/graph_encoder.pt" \
    --output_dir "$OUTPUT_DIR/candidate_embeddings" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --embedding_normalization none \
    --embedding_features full \
    --tabular_features safe \
    --device cpu \
    --unlabeled_candidates \
    --embeddings_only

"$PYTHON_BIN" training/apply_candidate_region_classifier.py \
    --checkpoint "$PIPELINE18/candidate_region_classifier_general/candidate_region_classifier.pt" \
    --embedding_dir "$OUTPUT_DIR/candidate_embeddings" \
    --output_dir "$OUTPUT_DIR/final_predictions" \
    --device cpu

echo "[$(date --iso-8601=seconds)] Pipeline19 CPU resume completed"
