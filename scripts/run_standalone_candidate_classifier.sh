#!/usr/bin/env bash
set -euo pipefail

# Discover candidates from CNA+SV inputs only, embed them, and apply a trained model.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MANIFEST="${MANIFEST:?Set MANIFEST to a genome manifest TSV}"
OUTPUT_DIR="${OUTPUT_DIR:?Set OUTPUT_DIR for standalone results}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a trained candidate_region_classifier.pt}"
CN_CHECKPOINT="${CN_CHECKPOINT:-$PROJECT_DIR/models/pretrained/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$PROJECT_DIR/models/pretrained/sv_graph_encoder.pt}"
CLASS_NAMES="${CLASS_NAMES:-ecDNA,chromothripsis,BFB,seismic_amplification}"
PROFILE="${PROFILE:-balanced}"
DEVICE="${DEVICE:-auto}"
EMBEDDING_NORMALIZATION="${EMBEDDING_NORMALIZATION:-none}"
EMBEDDING_FEATURES="${EMBEDDING_FEATURES:-full}"
TABULAR_FEATURES="${TABULAR_FEATURES:-safe}"

mkdir -p "$OUTPUT_DIR"

echo "[standalone] No external caller files are read by this workflow."
echo "[standalone] checkpoint=$CHECKPOINT profile=$PROFILE output=$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/generate_standalone_candidate_regions.py" "$MANIFEST" \
    --output_dir "$OUTPUT_DIR/candidates" \
    --centromeres "$PROJECT_DIR/data/grch38.cen_coord.curated.bed" \
    --profile "$PROFILE" \
    --keep_going

"$PYTHON_BIN" "$PROJECT_DIR/training/train_candidate_region_classifier.py" \
    --manifest "$MANIFEST" \
    --candidate_regions "$OUTPUT_DIR/candidates/merged_candidate_regions.csv" \
    --cn_checkpoint "$CN_CHECKPOINT" \
    --graph_checkpoint "$GRAPH_CHECKPOINT" \
    --output_dir "$OUTPUT_DIR/embeddings" \
    --class_names "$CLASS_NAMES" \
    --embedding_normalization "$EMBEDDING_NORMALIZATION" \
    --embedding_features "$EMBEDDING_FEATURES" \
    --tabular_features "$TABULAR_FEATURES" \
    --device "$DEVICE" \
    --unlabeled_candidates \
    --embeddings_only

"$PYTHON_BIN" "$PROJECT_DIR/training/apply_candidate_region_classifier.py" \
    --checkpoint "$CHECKPOINT" \
    --embedding_dir "$OUTPUT_DIR/embeddings" \
    --output_dir "$OUTPUT_DIR/predictions" \
    --device "$DEVICE"

echo "Standalone candidate discovery and classification complete: $OUTPUT_DIR"
