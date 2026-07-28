#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/../results/unsupervised1}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/../results/pipeline16/complex_sv_manifest.tsv}"
EXTERNAL_REGIONS="${EXTERNAL_REGIONS:-$PROJECT_DIR/../results/pipeline15/external_regions.tsv}"
CENTROMERES="${CENTROMERES:-$PROJECT_DIR/../results/grch38.cen_coord.curated.bed}"
CN_CHECKPOINT="${CN_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/sv3/graph_encoder.pt}"
DEVICE="${DEVICE:-auto}"
REUSE_EMBEDDINGS="${REUSE_EMBEDDINGS:-1}"

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" "$SCRIPT_DIR/build_exact_method_region_corpus.py" \
    --manifest "$MANIFEST" \
    --external_regions "$EXTERNAL_REGIONS" \
    --centromeres "$CENTROMERES" \
    --output_dir "$OUTPUT_DIR"

EMBEDDING_DIR="$OUTPUT_DIR/embeddings"
EXTRA_ARGS=()
if [[ "$REUSE_EMBEDDINGS" == "1" && -f "$EMBEDDING_DIR/embeddings.npz" ]]; then
    EXTRA_ARGS+=(--reuse_embeddings)
fi

"$PYTHON_BIN" "$PROJECT_DIR/training/train_candidate_region_classifier.py" \
    --manifest "$MANIFEST" \
    --candidate_regions "$OUTPUT_DIR/method_regions.csv" \
    --cn_checkpoint "$CN_CHECKPOINT" \
    --graph_checkpoint "$GRAPH_CHECKPOINT" \
    --output_dir "$EMBEDDING_DIR" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --embedding_normalization none \
    --embedding_features full \
    --tabular_features safe \
    --subtype_targets general \
    --subtype_thresholding off \
    --cluster_aggregation off \
    --embeddings_only \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" "$SCRIPT_DIR/analyze_unsupervised_method_regions.py" \
    --embedding_dir "$EMBEDDING_DIR" \
    --method_regions "$OUTPUT_DIR/method_regions.csv" \
    --output_dir "$OUTPUT_DIR"

echo "Exact method-region unsupervised analysis complete: $OUTPUT_DIR"
