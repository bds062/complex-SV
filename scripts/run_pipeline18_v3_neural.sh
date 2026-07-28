#!/usr/bin/env bash
set -euo pipefail

# Train the neural caller-supervised model on standalone v3 candidate regions.
# Candidate discovery is caller-free; external calls are used only for labels and evaluation.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
BASE="${BASE:-$PROJECT_DIR/../results/pipeline18}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/../results/pipeline17/complex_sv_manifest.tsv}"
CANDIDATES="${CANDIDATES:-$PROJECT_DIR/../results/standalone_candidate_generator_v3/merged_candidate_regions.csv}"
EXTERNAL_REGIONS="${EXTERNAL_REGIONS:-$PROJECT_DIR/../results/pipeline17/external_regions.tsv}"
CN_CHECKPOINT="${CN_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/sv3/graph_encoder.pt}"
CLASS_NAMES="${CLASS_NAMES:-ecDNA,chromothripsis,BFB}"
FOLD_SIZE="${FOLD_SIZE:-6}"
DEVICE="${DEVICE:-auto}"
FAST_THRESHOLDS="${FAST_THRESHOLDS:-0}"
PLOT_GENOMES="${PLOT_GENOMES:-1}"
REUSE_EMBEDDINGS="${REUSE_EMBEDDINGS:-1}"

MODEL_DIR="$BASE/candidate_region_classifier_general"
mkdir -p "$BASE" "$MODEL_DIR"
cp "$MANIFEST" "$BASE/complex_sv_manifest.tsv"
cp "$EXTERNAL_REGIONS" "$BASE/external_regions.tsv"

"$PYTHON_BIN" "$SCRIPT_DIR/label_standalone_candidates.py" \
    --candidates "$CANDIDATES" \
    --external_regions "$BASE/external_regions.tsv" \
    --output_dir "$BASE"

if [[ "$REUSE_EMBEDDINGS" != "1" || ! -f "$MODEL_DIR/embeddings.npz" ]]; then
    "$PYTHON_BIN" "$PROJECT_DIR/training/train_candidate_region_classifier.py" \
        --manifest "$BASE/complex_sv_manifest.tsv" \
        --candidate_regions "$BASE/merged_candidate_regions.csv" \
        --cn_checkpoint "$CN_CHECKPOINT" \
        --graph_checkpoint "$GRAPH_CHECKPOINT" \
        --output_dir "$MODEL_DIR" \
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

"$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_study.py" \
    --source_dir "$MODEL_DIR" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --output_dir "$MODEL_DIR" \
    --class_names "$CLASS_NAMES" \
    --bfb_label "Cervical-panel BFB" \
    --fold_size "$FOLD_SIZE" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/training/train_final_candidate_region_model.py" \
    --source_dir "$MODEL_DIR" \
    --cross_fold_predictions "$MODEL_DIR/cross_fold_predictions.tsv" \
    --output_dir "$MODEL_DIR" \
    --class_names "$CLASS_NAMES" \
    --device "$DEVICE" \
    "${EXTRA_ARGS[@]}"

if [[ "$PLOT_GENOMES" == "1" ]]; then
    rm -rf "$MODEL_DIR/test_genome_plots"
    "$PYTHON_BIN" "$PROJECT_DIR/discovery/plot_predicted_chromosomes.py" \
        --manifest "$BASE/complex_sv_manifest.tsv" \
        --prototype_distances "$MODEL_DIR/cross_fold_predictions.tsv" \
        --output_dir "$MODEL_DIR/test_genome_plots" \
        --plot_scope all \
        --dpi 180
fi

echo "Pipeline18 v3 neural caller-supervised experiment complete: $BASE"
