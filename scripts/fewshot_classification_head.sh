#!/usr/bin/env bash
set -euo pipefail

# Train a few-shot prototypical classifier on pipeline5 chromosome-arm embeddings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE="${BASE:-../results/pipeline7}"
MANIFEST="${MANIFEST:-$BASE/complex_sv_manifest.tsv}"
LABELS="${LABELS:-$BASE/complex_sv_labels.tsv}"
USE_LABELS="${USE_LABELS:-1}"
PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-$BASE/prototype_chrom_arm_sample_norm}"
EMBEDDING_DIR="${EMBEDDING_DIR:-$PROTOTYPE_ROOT/inference}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTOTYPE_ROOT/fewshot_classification_head}"
PLOT_DIR="${PLOT_DIR:-$OUTPUT_DIR/predicted_chromosome_arm_plots}"
EMBEDDINGS_NPZ="${EMBEDDINGS_NPZ:-$EMBEDDING_DIR/embeddings.npz}"
METADATA_TSV="${METADATA_TSV:-$EMBEDDING_DIR/candidate_embeddings.tsv}"

CN_CHECKPOINT="${CN_CHECKPOINT:-../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-../data/KolmogorovLab/srinivasanbd/results/pipeline3/sv3/graph_encoder.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CANDIDATE_SOURCE="${CANDIDATE_SOURCE:-chromosome-arms}"
CANDIDATE_RESOLUTION="${CANDIDATE_RESOLUTION:-chromosome-arm}"
EMBEDDING_NORMALIZATION="${EMBEDDING_NORMALIZATION:-sample_residual}"
SAMPLE_BASELINE_MIN_CANDIDATES="${SAMPLE_BASELINE_MIN_CANDIDATES:-3}"
EMBEDDING_TAU="${EMBEDDING_TAU:-0.5562}"
FORCE_EMBED="${FORCE_EMBED:-0}"
DEVICE="${DEVICE:-auto}"

PROJECTION_DIM="${PROJECTION_DIM:-64}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
DROPOUT="${DROPOUT:-0.2}"
TEMPERATURE="${TEMPERATURE:-0.1}"
OBJECTNESS_SCALE="${OBJECTNESS_SCALE:-12.0}"
EPOCHS="${EPOCHS:-400}"
PATIENCE="${PATIENCE:-80}"
CV_EPOCHS="${CV_EPOCHS:-220}"
CV_PATIENCE="${CV_PATIENCE:-50}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-inverse_sqrt}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.02}"
TAU_SELECTION_METRIC="${TAU_SELECTION_METRIC:-typed_f1}"
DISTANCE_TAU_MIN="${DISTANCE_TAU_MIN:-0.0}"
DISTANCE_TAU_MAX="${DISTANCE_TAU_MAX:-4.0}"
DISTANCE_TAU_STEPS="${DISTANCE_TAU_STEPS:-161}"

PLOT_PREDICTED_CHROMS="${PLOT_PREDICTED_CHROMS:-1}"
PLOT_DPI="${PLOT_DPI:-180}"
PLOT_MAX_PLOTS="${PLOT_MAX_PLOTS:-}"

cd "$PROJECT_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "../envs/env2/bin/python" ]]; then
    PYTHON_BIN="../envs/env2/bin/python"
fi

if ! "$PYTHON_BIN" -c "import pandas, torch, matplotlib, torch_geometric" >/dev/null 2>&1; then
    echo "[fewshot_classification_head] ERROR: $PYTHON_BIN cannot import pandas, torch, matplotlib, and torch_geometric." >&2
    exit 1
fi

for required_file in "$MANIFEST"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[fewshot_classification_head] ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

LABEL_ARGS=()
if [[ "$USE_LABELS" == "1" && -n "$LABELS" && -f "$LABELS" ]]; then
    LABEL_ARGS=(--labels "$LABELS")
fi

if [[ "$FORCE_EMBED" == "1" || ! -f "$EMBEDDINGS_NPZ" || ! -f "$METADATA_TSV" ]]; then
    if [[ ! -f "$CN_CHECKPOINT" ]]; then
        echo "[fewshot_classification_head] ERROR: required file not found for embedding: $CN_CHECKPOINT" >&2
        exit 1
    fi
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
    echo "[fewshot_classification_head] Building embeddings:"
    printf '  %q' "${EMBED_CMD[@]}"
    printf '\n'
    "${EMBED_CMD[@]}"
else
    echo "[fewshot_classification_head] Reusing existing embeddings: $EMBEDDINGS_NPZ"
fi

mkdir -p "$OUTPUT_DIR"
TRAIN_CMD=(
    "$PYTHON_BIN" training/train_fewshot_classifier_head.py
    --embeddings_npz "$EMBEDDINGS_NPZ"
    --metadata_tsv "$METADATA_TSV"
    --output_dir "$OUTPUT_DIR"
    --candidate_resolution "$CANDIDATE_RESOLUTION"
    --projection_dim "$PROJECTION_DIM"
    --hidden_dim "$HIDDEN_DIM"
    --dropout "$DROPOUT"
    --temperature "$TEMPERATURE"
    --objectness_scale "$OBJECTNESS_SCALE"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --cv_epochs "$CV_EPOCHS"
    --cv_patience "$CV_PATIENCE"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --class_weighting "$CLASS_WEIGHTING"
    --label_smoothing "$LABEL_SMOOTHING"
    --tau_selection_metric "$TAU_SELECTION_METRIC"
    --distance_tau_min "$DISTANCE_TAU_MIN"
    --distance_tau_max "$DISTANCE_TAU_MAX"
    --distance_tau_steps "$DISTANCE_TAU_STEPS"
    --device "$DEVICE"
)
if [[ -n "${DISTANCE_TAU:-}" ]]; then
    TRAIN_CMD+=(--distance_tau "$DISTANCE_TAU")
fi
if [[ "${SKIP_LOSO:-0}" == "1" ]]; then
    TRAIN_CMD+=(--skip_loso)
fi

echo "[fewshot_classification_head] Training few-shot classifier:"
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

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
    echo "[fewshot_classification_head] Plotting predicted chromosome arms:"
    printf '  %q' "${PLOT_CMD[@]}"
    printf '\n'
    "${PLOT_CMD[@]}"
fi

for expected_output in \
    "$OUTPUT_DIR/fewshot_classification_head.pt" \
    "$OUTPUT_DIR/classification_predictions.tsv" \
    "$OUTPUT_DIR/predicted_complex_sv.tsv" \
    "$OUTPUT_DIR/predictions.tsv" \
    "$OUTPUT_DIR/prototype_distances.tsv" \
    "$OUTPUT_DIR/training_summary.json" \
    "$OUTPUT_DIR/prototype_distances.png" \
    "$OUTPUT_DIR/embedding_projection_predicted.png"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[fewshot_classification_head] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

if [[ "$PLOT_PREDICTED_CHROMS" == "1" && ! -f "$PLOT_DIR/selected_predictions.tsv" ]]; then
    echo "[fewshot_classification_head] ERROR: expected output not found: $PLOT_DIR/selected_predictions.tsv" >&2
    exit 1
fi

echo "[fewshot_classification_head] Done."
