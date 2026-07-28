#!/usr/bin/env bash
set -euo pipefail

# Train and evaluate neural classifiers on caller-free standalone proposals.
# External calls are used only here, as offline supervision and evaluation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
BASE="${BASE:-$PROJECT_DIR/../results/pipeline16}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/../results/pipeline12/complex_sv_manifest.tsv}"
STANDALONE_DIR="${STANDALONE_DIR:-$PROJECT_DIR/../results/standalone_candidate_generator_v2}"
EXTERNAL_REGIONS="${EXTERNAL_REGIONS:-$PROJECT_DIR/../results/pipeline15/external_regions.tsv}"
CN_CHECKPOINT="${CN_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/sv3/graph_encoder.pt}"
CLASS_NAMES="${CLASS_NAMES:-ecDNA,chromothripsis,BFB}"
FOLD_SIZE="${FOLD_SIZE:-6}"
DEVICE="${DEVICE:-auto}"
REUSE_EMBEDDINGS="${REUSE_EMBEDDINGS:-1}"
FAST_THRESHOLDS="${FAST_THRESHOLDS:-0}"

SENSITIVE="$BASE/sensitive"
BALANCED="$BASE/balanced"
mkdir -p "$SENSITIVE" "$BALANCED"
cp "$MANIFEST" "$BASE/complex_sv_manifest.tsv"

"$PYTHON_BIN" "$SCRIPT_DIR/label_standalone_candidates.py" \
    --candidates "$STANDALONE_DIR/merged_candidate_regions_sensitive.csv" \
    --external_regions "$EXTERNAL_REGIONS" \
    --output_dir "$SENSITIVE"

"$PYTHON_BIN" "$SCRIPT_DIR/label_standalone_candidates.py" \
    --candidates "$STANDALONE_DIR/merged_candidate_regions_balanced.csv" \
    --external_regions "$EXTERNAL_REGIONS" \
    --output_dir "$BALANCED"

if [[ "$REUSE_EMBEDDINGS" != "1" || ! -f "$SENSITIVE/embeddings.npz" ]]; then
    "$PYTHON_BIN" "$PROJECT_DIR/training/train_candidate_region_classifier.py" \
        --manifest "$BASE/complex_sv_manifest.tsv" \
        --candidate_regions "$SENSITIVE/merged_candidate_regions.csv" \
        --cn_checkpoint "$CN_CHECKPOINT" \
        --graph_checkpoint "$GRAPH_CHECKPOINT" \
        --output_dir "$SENSITIVE" \
        --class_names "$CLASS_NAMES" \
        --embedding_normalization none \
        --embedding_features full \
        --tabular_features safe \
        --subtype_targets general \
        --subtype_thresholding off \
        --cluster_aggregation max \
        --embeddings_only \
        --device "$DEVICE"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/subset_candidate_embedding_bundle.py" \
    --source_dir "$SENSITIVE" \
    --candidate_regions "$BALANCED/merged_candidate_regions.csv" \
    --output_dir "$BALANCED"

EXTRA_ARGS=()
if [[ "$FAST_THRESHOLDS" == "1" ]]; then
    EXTRA_ARGS+=(--fast_thresholds)
fi
if [[ -n "${EPOCHS:-}" ]]; then
    EXTRA_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "${PATIENCE:-}" ]]; then
    EXTRA_ARGS+=(--patience "$PATIENCE")
fi

for PROFILE_DIR in "$SENSITIVE" "$BALANCED"; do
    "$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_study.py" \
        --source_dir "$PROFILE_DIR" \
        --manifest "$BASE/complex_sv_manifest.tsv" \
        --output_dir "$PROFILE_DIR" \
        --class_names "$CLASS_NAMES" \
        --fold_size "$FOLD_SIZE" \
        --device "$DEVICE" \
        "${EXTRA_ARGS[@]}"

    "$PYTHON_BIN" "$PROJECT_DIR/training/train_final_candidate_region_model.py" \
        --source_dir "$PROFILE_DIR" \
        --cross_fold_predictions "$PROFILE_DIR/cross_fold_predictions.tsv" \
        --output_dir "$PROFILE_DIR" \
        --class_names "$CLASS_NAMES" \
        --device "$DEVICE" \
        "${EXTRA_ARGS[@]}"
done

echo "Pipeline16 standalone neural experiments complete: $BASE"
