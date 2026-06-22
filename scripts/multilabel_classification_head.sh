#!/usr/bin/env bash
set -euo pipefail

# Train a multi-label complex-SV classifier head on frozen prototype-mode embeddings.
# This keeps the objectness head, but predicts each complex-SV type with an
# independent sigmoid and per-class threshold. Non-canonical BFB/chromothripsis
# labels are collapsed into their canonical class during this experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE="${BASE:-../results/pipeline7}"
MANIFEST="${MANIFEST:-$BASE/complex_sv_manifest.tsv}"
PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-$BASE/prototype_chrom_arm_sample_norm}"
INPUT_DIR="${INPUT_DIR:-$PROTOTYPE_ROOT/inference}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTOTYPE_ROOT/multilabel_classification_head}"
PLOT_DIR="${PLOT_DIR:-$OUTPUT_DIR/predicted_chromosome_arm_plots}"
EMBEDDINGS_NPZ="${EMBEDDINGS_NPZ:-$INPUT_DIR/embeddings.npz}"
METADATA_TSV="${METADATA_TSV:-$INPUT_DIR/candidate_embeddings.tsv}"
CANDIDATE_RESOLUTION="${CANDIDATE_RESOLUTION:-chromosome-arm}"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLASS_NAMES="${CLASS_NAMES:-BFB,chromothripsis,seismic_amplification,TIC}"
HIDDEN_DIMS="${HIDDEN_DIMS:-128}"
ACTIVATION="${ACTIVATION:-relu}"
DROPOUT="${DROPOUT:-0.2}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-60}"
CV_EPOCHS="${CV_EPOCHS:-$EPOCHS}"
CV_PATIENCE="${CV_PATIENCE:-$PATIENCE}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-0.1}"
TYPE_LOSS_WEIGHT="${TYPE_LOSS_WEIGHT:-1.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.02}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-inverse_sqrt}"
TAU_SELECTION_METRIC="${TAU_SELECTION_METRIC:-f1}"
TYPE_TAU_MIN="${TYPE_TAU_MIN:-0.05}"
TYPE_TAU_MAX="${TYPE_TAU_MAX:-0.95}"
TYPE_TAU_STEPS="${TYPE_TAU_STEPS:-91}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
PLOT_PREDICTED_CHROMS="${PLOT_PREDICTED_CHROMS:-1}"
PLOT_DPI="${PLOT_DPI:-180}"
PLOT_MAX_PLOTS="${PLOT_MAX_PLOTS:-}"

cd "$PROJECT_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "../envs/env2/bin/python" ]]; then
    PYTHON_BIN="../envs/env2/bin/python"
fi

if ! "$PYTHON_BIN" -c "import pandas, torch, matplotlib" >/dev/null 2>&1; then
    echo "[multilabel_classification_head] ERROR: $PYTHON_BIN cannot import pandas, torch, and matplotlib." >&2
    exit 1
fi

for required_file in "$EMBEDDINGS_NPZ" "$METADATA_TSV" "$MANIFEST"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[multilabel_classification_head] ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

mkdir -p "$OUTPUT_DIR"
if [[ "$PLOT_PREDICTED_CHROMS" == "1" ]]; then
    rm -rf "$PLOT_DIR"
fi

echo "[multilabel_classification_head] project_dir=$PROJECT_DIR"
echo "[multilabel_classification_head] manifest=$MANIFEST"
echo "[multilabel_classification_head] embeddings_npz=$EMBEDDINGS_NPZ"
echo "[multilabel_classification_head] metadata_tsv=$METADATA_TSV"
echo "[multilabel_classification_head] output_dir=$OUTPUT_DIR"
echo "[multilabel_classification_head] plot_dir=$PLOT_DIR"
echo "[multilabel_classification_head] class_names=$CLASS_NAMES hidden_dims=$HIDDEN_DIMS activation=$ACTIVATION epochs=$EPOCHS"
echo "[multilabel_classification_head] background_weight=$BACKGROUND_WEIGHT tau_metric=$TAU_SELECTION_METRIC seed=$SEED device=$DEVICE"
echo "[multilabel_classification_head] candidate_resolution=$CANDIDATE_RESOLUTION"

CMD=(
    "$PYTHON_BIN" training/train_multilabel_classifier_head.py
    --embeddings_npz "$EMBEDDINGS_NPZ"
    --metadata_tsv "$METADATA_TSV"
    --output_dir "$OUTPUT_DIR"
    --candidate_resolution "$CANDIDATE_RESOLUTION"
    --class_names "$CLASS_NAMES"
    --hidden_dims "$HIDDEN_DIMS"
    --activation "$ACTIVATION"
    --dropout "$DROPOUT"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --cv_epochs "$CV_EPOCHS"
    --cv_patience "$CV_PATIENCE"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --background_weight "$BACKGROUND_WEIGHT"
    --type_loss_weight "$TYPE_LOSS_WEIGHT"
    --label_smoothing "$LABEL_SMOOTHING"
    --class_weighting "$CLASS_WEIGHTING"
    --tau_selection_metric "$TAU_SELECTION_METRIC"
    --type_tau_min "$TYPE_TAU_MIN"
    --type_tau_max "$TYPE_TAU_MAX"
    --type_tau_steps "$TYPE_TAU_STEPS"
    --seed "$SEED"
    --device "$DEVICE"
)

if [[ -n "${TAU:-}" ]]; then
    CMD+=(--tau "$TAU")
fi
if [[ -n "${TYPE_TAU:-}" ]]; then
    CMD+=(--type_tau "$TYPE_TAU")
fi
if [[ "${SKIP_LOSO:-0}" == "1" ]]; then
    CMD+=(--skip_loso)
fi

echo "[multilabel_classification_head] Training multi-label classifier head:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

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

    echo "[multilabel_classification_head] Plotting predicted unlabeled chromosome arms:"
    printf '  %q' "${PLOT_CMD[@]}"
    printf '\n'
    "${PLOT_CMD[@]}"
fi

for expected_output in \
    "$OUTPUT_DIR/multilabel_classification_head.pt" \
    "$OUTPUT_DIR/classification_predictions.tsv" \
    "$OUTPUT_DIR/predicted_complex_sv.tsv" \
    "$OUTPUT_DIR/prototype_distances.tsv" \
    "$OUTPUT_DIR/prototype_distances.png" \
    "$OUTPUT_DIR/embedding_projection_predicted.png" \
    "$OUTPUT_DIR/training_metrics.tsv" \
    "$OUTPUT_DIR/type_thresholds.tsv" \
    "$OUTPUT_DIR/training_summary.json"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[multilabel_classification_head] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

if [[ "$PLOT_PREDICTED_CHROMS" == "1" && ! -f "$PLOT_DIR/selected_predictions.tsv" ]]; then
    echo "[multilabel_classification_head] ERROR: expected output not found: $PLOT_DIR/selected_predictions.tsv" >&2
    exit 1
fi

if [[ "${SKIP_LOSO:-0}" != "1" ]]; then
    for expected_output in \
        "$OUTPUT_DIR/leave_one_sample_out.tsv" \
        "$OUTPUT_DIR/objectness_tau_sweep.tsv" \
        "$OUTPUT_DIR/type_threshold_sweep.tsv"; do
        if [[ ! -f "$expected_output" ]]; then
            echo "[multilabel_classification_head] ERROR: expected output not found: $expected_output" >&2
            exit 1
        fi
    done
fi

echo "[multilabel_classification_head] Done."
