#!/usr/bin/env bash
set -euo pipefail

# Chromosome-scale localization followed by the trained pipeline18 event typer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
PIPELINE18="${PIPELINE18:-$PROJECT_DIR/../results/pipeline18}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/../results/pipeline19_asm_loc}"
MANIFEST="${MANIFEST:-$PIPELINE18/complex_sv_manifest.tsv}"
REGIONS="${REGIONS:-$PIPELINE18/external_regions.tsv}"
TEACHER_PREDICTIONS="${TEACHER_PREDICTIONS:-$PIPELINE18/candidate_region_classifier_general/cross_fold_predictions.tsv}"
TYPE_CHECKPOINT="${TYPE_CHECKPOINT:-$PIPELINE18/candidate_region_classifier_general/candidate_region_classifier.pt}"
CN_CHECKPOINT="${CN_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/sv3/graph_encoder.pt}"
DEVICE="${DEVICE:-auto}"
BIN_SIZE="${BIN_SIZE:-1000000}"

mkdir -p "$OUTPUT_DIR"

BUILD_ARGS=(
    build-dataset
    --manifest "$MANIFEST"
    --regions "$REGIONS"
    --output "$OUTPUT_DIR/chromosome_bins.tsv.gz"
    --bin_size "$BIN_SIZE"
)
if [[ -f "$TEACHER_PREDICTIONS" ]]; then
    BUILD_ARGS+=(--teacher_predictions "$TEACHER_PREDICTIONS")
fi

"$PYTHON_BIN" "$SCRIPT_DIR/asm_loc_pipeline.py" "${BUILD_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/asm_loc_pipeline.py" train \
    --dataset "$OUTPUT_DIR/chromosome_bins.tsv.gz" \
    --output "$OUTPUT_DIR/asm_loc_model.pt" \
    --device "$DEVICE"

"$PYTHON_BIN" "$SCRIPT_DIR/asm_loc_pipeline.py" predict \
    --dataset "$OUTPUT_DIR/chromosome_bins.tsv.gz" \
    --checkpoint "$OUTPUT_DIR/asm_loc_model.pt" \
    --output "$OUTPUT_DIR/localized_proposals.tsv" \
    --bin_predictions "$OUTPUT_DIR/bin_predictions.tsv.gz" \
    --bin_size "$BIN_SIZE" \
    --device "$DEVICE"

"$PYTHON_BIN" "$SCRIPT_DIR/asm_loc_pipeline.py" evaluate \
    --proposals "$OUTPUT_DIR/localized_proposals.tsv" \
    --regions "$REGIONS" \
    --output "$OUTPUT_DIR/localization_recall.tsv"

"$PYTHON_BIN" "$SCRIPT_DIR/asm_loc_pipeline.py" materialize \
    --manifest "$MANIFEST" \
    --proposals "$OUTPUT_DIR/localized_proposals.tsv" \
    --output "$OUTPUT_DIR/merged_candidate_regions.csv"

"$PYTHON_BIN" "$PROJECT_DIR/training/train_candidate_region_classifier.py" \
    --manifest "$MANIFEST" \
    --candidate_regions "$OUTPUT_DIR/merged_candidate_regions.csv" \
    --cn_checkpoint "$CN_CHECKPOINT" \
    --graph_checkpoint "$GRAPH_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/candidate_embeddings" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --embedding_normalization none \
    --embedding_features full \
    --tabular_features safe \
    --device "$DEVICE" \
    --unlabeled_candidates \
    --embeddings_only

"$PYTHON_BIN" "$PROJECT_DIR/training/apply_candidate_region_classifier.py" \
    --checkpoint "$TYPE_CHECKPOINT" \
    --embedding_dir "$OUTPUT_DIR/candidate_embeddings" \
    --output_dir "$OUTPUT_DIR/final_predictions" \
    --device "$DEVICE"

echo "Pipeline19 ASM-Loc localization and pipeline18 classification complete: $OUTPUT_DIR"

