#!/usr/bin/env bash
set -euo pipefail

# Apply a trained multi-label classifier head to a new manifest's chromosome-arm embeddings.
# Default example: pipeline5-trained head applied to pipeline6.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE="${BASE:-../results/pipeline8}"
MANIFEST="${MANIFEST:-$BASE/complex_sv_manifest.tsv}"
LABELS="${LABELS:-$BASE/complex_sv_labels.tsv}"
USE_LABELS="${USE_LABELS:-1}"
PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-$BASE/prototype_chrom_arm_sample_norm}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$PROTOTYPE_ROOT/inference}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTOTYPE_ROOT/multilabel_classification_head_applied}"
PLOT_DIR="${PLOT_DIR:-$OUTPUT_DIR/predicted_chromosome_arm_plots}"
CHECKPOINT="${CHECKPOINT:-../results/pipeline7/prototype_chrom_arm_sample_norm/multilabel_classification_head/multilabel_classification_head.pt}"
EMBEDDINGS_NPZ="${EMBEDDINGS_NPZ:-$EMBEDDING_DIR/embeddings.npz}"
METADATA_TSV="${METADATA_TSV:-$EMBEDDING_DIR/candidate_embeddings.tsv}"

CN_CHECKPOINT="${CN_CHECKPOINT:-../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-/data/KolmogorovLab/srinivasanbd/results/pipeline3/sv3/graph_encoder.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-chromosome-arms}"
CANDIDATE_RESOLUTION="${CANDIDATE_RESOLUTION:-chromosome-arm}"
EMBEDDING_NORMALIZATION="${EMBEDDING_NORMALIZATION:-sample_residual}"
SAMPLE_BASELINE_MIN_CANDIDATES="${SAMPLE_BASELINE_MIN_CANDIDATES:-3}"
EMBEDDING_TAU="${EMBEDDING_TAU:-0.5562}"
FORCE_EMBED="${FORCE_EMBED:-0}"
DEVICE="${DEVICE:-auto}"
PLOT_PREDICTED_CHROMS="${PLOT_PREDICTED_CHROMS:-1}"
PLOT_DPI="${PLOT_DPI:-180}"
PLOT_MAX_PLOTS="${PLOT_MAX_PLOTS:-}"
SKIP_VISUALIZATIONS="${SKIP_VISUALIZATIONS:-0}"

cd "$PROJECT_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "../envs/env2/bin/python" ]]; then
    PYTHON_BIN="../envs/env2/bin/python"
fi

if ! "$PYTHON_BIN" -c "import pandas, torch, matplotlib, torch_geometric" >/dev/null 2>&1; then
    echo "[apply_multilabel_classifier_head] ERROR: $PYTHON_BIN cannot import pandas, torch, matplotlib, and torch_geometric." >&2
    exit 1
fi

for required_file in "$MANIFEST" "$CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[apply_multilabel_classifier_head] ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

LABEL_ARGS=()
if [[ "$USE_LABELS" == "1" && -n "$LABELS" && -f "$LABELS" ]]; then
    LABEL_ARGS=(--labels "$LABELS")
fi

if [[ "$FORCE_EMBED" == "1" || ! -f "$EMBEDDINGS_NPZ" || ! -f "$METADATA_TSV" ]]; then
    for required_file in "$CN_CHECKPOINT"; do
        if [[ ! -f "$required_file" ]]; then
            echo "[apply_multilabel_classifier_head] ERROR: required file not found for embedding: $required_file" >&2
            exit 1
        fi
    done
    mkdir -p "$EMBEDDING_DIR"
    EMBED_CMD=(
        "$PYTHON_BIN" infer.py
        --manifest "$MANIFEST"
        "${LABEL_ARGS[@]}"
        --cn_checkpoint "$CN_CHECKPOINT"
        --output_dir "$EMBEDDING_DIR"
        --candidate_source "$CANDIDATE_SOURCE"
        --tau "$EMBEDDING_TAU"
        --embedding_normalization "$EMBEDDING_NORMALIZATION"
        --sample_baseline_min_candidates "$SAMPLE_BASELINE_MIN_CANDIDATES"
    )
    if [[ -n "$GRAPH_CHECKPOINT" && -f "$GRAPH_CHECKPOINT" ]]; then
        EMBED_CMD+=(--graph_checkpoint "$GRAPH_CHECKPOINT")
    fi
    echo "[apply_multilabel_classifier_head] Building embeddings:"
    printf '  %q' "${EMBED_CMD[@]}"
    printf '\n'
    "${EMBED_CMD[@]}"
else
    echo "[apply_multilabel_classifier_head] Reusing existing embeddings: $EMBEDDINGS_NPZ"
fi

mkdir -p "$OUTPUT_DIR"
if [[ "$PLOT_PREDICTED_CHROMS" == "1" ]]; then
    rm -rf "$PLOT_DIR"
fi

APPLY_CMD=(
    "$PYTHON_BIN" training/apply_multilabel_classifier_head.py
    --checkpoint "$CHECKPOINT"
    --embeddings_npz "$EMBEDDINGS_NPZ"
    --metadata_tsv "$METADATA_TSV"
    --output_dir "$OUTPUT_DIR"
    --candidate_resolution "$CANDIDATE_RESOLUTION"
    --device "$DEVICE"
)
if [[ -n "${TAU:-}" ]]; then
    APPLY_CMD+=(--tau "$TAU")
fi
if [[ -n "${TYPE_TAU:-}" ]]; then
    APPLY_CMD+=(--type_tau "$TYPE_TAU")
fi
if [[ "$SKIP_VISUALIZATIONS" == "1" ]]; then
    APPLY_CMD+=(--skip_visualizations)
fi

echo "[apply_multilabel_classifier_head] Applying classifier head:"
printf '  %q' "${APPLY_CMD[@]}"
printf '\n'
"${APPLY_CMD[@]}"

if [[ "$PLOT_PREDICTED_CHROMS" == "1" ]]; then
    PLOT_CMD=(
        "$PYTHON_BIN" discovery/plot_predicted_chromosomes.py
        --manifest "$MANIFEST"
        --prototype_distances "$OUTPUT_DIR/prototype_distances.tsv"
        --output_dir "$PLOT_DIR"
        --dpi "$PLOT_DPI"
    )
    if [[ -n "$PLOT_MAX_PLOTS" ]]; then
        PLOT_CMD+=(--max_plots "$PLOT_MAX_PLOTS")
    fi
    echo "[apply_multilabel_classifier_head] Plotting predicted chromosome arms:"
    printf '  %q' "${PLOT_CMD[@]}"
    printf '\n'
    "${PLOT_CMD[@]}"
fi

for expected_output in \
    "$OUTPUT_DIR/classification_predictions.tsv" \
    "$OUTPUT_DIR/predicted_complex_sv.tsv" \
    "$OUTPUT_DIR/predictions.tsv" \
    "$OUTPUT_DIR/prototype_distances.tsv" \
    "$OUTPUT_DIR/type_thresholds.tsv" \
    "$OUTPUT_DIR/inference_summary.json"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[apply_multilabel_classifier_head] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

if [[ "$SKIP_VISUALIZATIONS" != "1" ]]; then
    for expected_output in \
        "$OUTPUT_DIR/prototype_distances.png" \
        "$OUTPUT_DIR/embedding_projection_predicted.png"; do
        if [[ ! -f "$expected_output" ]]; then
            echo "[apply_multilabel_classifier_head] ERROR: expected output not found: $expected_output" >&2
            exit 1
        fi
    done
fi

if [[ "$PLOT_PREDICTED_CHROMS" == "1" && ! -f "$PLOT_DIR/selected_predictions.tsv" ]]; then
    echo "[apply_multilabel_classifier_head] ERROR: expected output not found: $PLOT_DIR/selected_predictions.tsv" >&2
    exit 1
fi

echo "[apply_multilabel_classifier_head] Done."
