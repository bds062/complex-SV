#!/usr/bin/env bash
set -euo pipefail

# Train a candidate-region complex-SV classifier from generated candidate regions.
# The model predicts empty vs complex-SV and independent multi-label event types.
# Run from anywhere with:
#   bash complex-SV/scripts/candidate_region_classifier.sh
# or from the repo with:
#   bash scripts/candidate_region_classifier.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE="${BASE:-../results/pipeline9}"
MANIFEST="${MANIFEST:-$BASE/complex_sv_manifest.tsv}"
CANDIDATE_REGIONS="${CANDIDATE_REGIONS:-$BASE/merged_candidate_regions.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/candidate_region_classifier}"

CN_CHECKPOINT="${CN_CHECKPOINT:-../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-/data/KolmogorovLab/srinivasanbd/results/pipeline3/sv3/graph_encoder.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"
PLOT_TEST_GENOMES="${PLOT_TEST_GENOMES:-1}"
PLOT_DIR="${PLOT_DIR:-$OUTPUT_DIR/test_genome_plots}"
PLOT_DPI="${PLOT_DPI:-180}"
PLOT_MAX_PLOTS="${PLOT_MAX_PLOTS:-}"

CLASS_NAMES="${CLASS_NAMES:-ecDNA,Seismic_Amplification,chromothripsis,BFB}"
REQUIRED_TEST_SAMPLE="${REQUIRED_TEST_SAMPLE:-H1395}"
TEST_SAMPLES="${TEST_SAMPLES:-}"
N_TEST_SAMPLES="${N_TEST_SAMPLES:-5}"
SEED="${SEED:-42}"
REUSE_EMBEDDINGS="${REUSE_EMBEDDINGS:-0}"

EMBEDDING_NORMALIZATION="${EMBEDDING_NORMALIZATION:-sample_residual}"
EMBEDDING_FEATURES="${EMBEDDING_FEATURES:-full}"
SAMPLE_BASELINE_MIN_CANDIDATES="${SAMPLE_BASELINE_MIN_CANDIDATES:-3}"
HIDDEN_DIMS="${HIDDEN_DIMS:-128}"
TABULAR_FEATURES="${TABULAR_FEATURES:-safe}"
TABULAR_HIDDEN_DIM="${TABULAR_HIDDEN_DIM:-32}"
ACTIVATION="${ACTIVATION:-relu}"
DROPOUT="${DROPOUT:-0.2}"
EPOCHS="${EPOCHS:-300}"
PATIENCE="${PATIENCE:-60}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.001}"
BACKGROUND_WEIGHT="${BACKGROUND_WEIGHT:-1.0}"
TYPE_LOSS_WEIGHT="${TYPE_LOSS_WEIGHT:-1.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.02}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-inverse_sqrt}"
THRESHOLD_CALIBRATION="${THRESHOLD_CALIBRATION:-logo}"
THRESHOLD_TIE_BREAK="${THRESHOLD_TIE_BREAK:-low}"
LOGO_EPOCHS="${LOGO_EPOCHS:-}"
LOGO_PATIENCE="${LOGO_PATIENCE:-}"
TAU_SELECTION_METRIC="${TAU_SELECTION_METRIC:-f1}"
TAU_MIN="${TAU_MIN:-0.05}"
TAU_MAX="${TAU_MAX:-0.95}"
TAU_STEPS="${TAU_STEPS:-91}"
TYPE_TAU_MIN="${TYPE_TAU_MIN:-0.05}"
TYPE_TAU_MAX="${TYPE_TAU_MAX:-0.95}"
TYPE_TAU_STEPS="${TYPE_TAU_STEPS:-91}"
RESCUE_THRESHOLDING="${RESCUE_THRESHOLDING:-optimize}"
RESCUE_TYPE_TAU="${RESCUE_TYPE_TAU:-0.85}"
RESCUE_OBJECTNESS_FLOOR="${RESCUE_OBJECTNESS_FLOOR:-0.0}"
RESCUE_MARGIN="${RESCUE_MARGIN:-0.0}"
RESCUE_TYPE_TAU_MIN="${RESCUE_TYPE_TAU_MIN:-0.60}"
RESCUE_TYPE_TAU_MAX="${RESCUE_TYPE_TAU_MAX:-0.98}"
RESCUE_TYPE_TAU_STEPS="${RESCUE_TYPE_TAU_STEPS:-20}"
RESCUE_OBJECTNESS_FLOOR_GRID="${RESCUE_OBJECTNESS_FLOOR_GRID:-0,0.001,0.005,0.01,0.02,0.05,0.10}"
RESCUE_MARGIN_MIN="${RESCUE_MARGIN_MIN:-0.0}"
RESCUE_MARGIN_MAX="${RESCUE_MARGIN_MAX:-0.30}"
RESCUE_MARGIN_STEPS="${RESCUE_MARGIN_STEPS:-7}"
RESCUE_MIN_RECALL="${RESCUE_MIN_RECALL:-0.85}"
RESCUE_MIN_PRECISION="${RESCUE_MIN_PRECISION:-0.60}"
RESCUE_MAX_EMPTY_FP_RATE="${RESCUE_MAX_EMPTY_FP_RATE:-0.75}"
SECONDARY_THRESHOLDING="${SECONDARY_THRESHOLDING:-optimize}"
SECONDARY_MIN="${SECONDARY_MIN:-0.55}"
SECONDARY_DELTA="${SECONDARY_DELTA:-0.15}"
SECONDARY_MIN_MIN="${SECONDARY_MIN_MIN:-0.30}"
SECONDARY_MIN_MAX="${SECONDARY_MIN_MAX:-0.90}"
SECONDARY_MIN_STEPS="${SECONDARY_MIN_STEPS:-13}"
SECONDARY_DELTA_MIN="${SECONDARY_DELTA_MIN:-0.0}"
SECONDARY_DELTA_MAX="${SECONDARY_DELTA_MAX:-0.50}"
SECONDARY_DELTA_STEPS="${SECONDARY_DELTA_STEPS:-11}"
SECONDARY_MIN_RECALL="${SECONDARY_MIN_RECALL:-0.85}"
SECONDARY_MIN_PRECISION="${SECONDARY_MIN_PRECISION:-0.60}"

cd "$PROJECT_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "../envs/env2/bin/python" ]]; then
    PYTHON_BIN="../envs/env2/bin/python"
fi

if ! "$PYTHON_BIN" -c "import pandas, torch, matplotlib, torch_geometric" >/dev/null 2>&1; then
    echo "[candidate_region_classifier] ERROR: $PYTHON_BIN cannot import pandas, torch, matplotlib, and torch_geometric." >&2
    exit 1
fi

for required_file in "$MANIFEST" "$CANDIDATE_REGIONS" "$CN_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[candidate_region_classifier] ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done
if [[ -n "$GRAPH_CHECKPOINT" && ! -f "$GRAPH_CHECKPOINT" ]]; then
    echo "[candidate_region_classifier] ERROR: graph checkpoint not found: $GRAPH_CHECKPOINT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[candidate_region_classifier] project_dir=$PROJECT_DIR"
echo "[candidate_region_classifier] manifest=$MANIFEST"
echo "[candidate_region_classifier] candidate_regions=$CANDIDATE_REGIONS"
echo "[candidate_region_classifier] output_dir=$OUTPUT_DIR"
echo "[candidate_region_classifier] cn_checkpoint=$CN_CHECKPOINT"
echo "[candidate_region_classifier] graph_checkpoint=$GRAPH_CHECKPOINT"
echo "[candidate_region_classifier] class_names=$CLASS_NAMES"
echo "[candidate_region_classifier] required_test_sample=$REQUIRED_TEST_SAMPLE test_samples=$TEST_SAMPLES n_test_samples=$N_TEST_SAMPLES seed=$SEED"
echo "[candidate_region_classifier] embedding_normalization=$EMBEDDING_NORMALIZATION embedding_features=$EMBEDDING_FEATURES reuse_embeddings=$REUSE_EMBEDDINGS"
echo "[candidate_region_classifier] epochs=$EPOCHS patience=$PATIENCE hidden_dims=$HIDDEN_DIMS tabular_features=$TABULAR_FEATURES tabular_hidden_dim=$TABULAR_HIDDEN_DIM device=$DEVICE"
echo "[candidate_region_classifier] threshold_calibration=$THRESHOLD_CALIBRATION threshold_tie_break=$THRESHOLD_TIE_BREAK logo_epochs=$LOGO_EPOCHS logo_patience=$LOGO_PATIENCE"
echo "[candidate_region_classifier] rescue_thresholding=$RESCUE_THRESHOLDING type_tau=$RESCUE_TYPE_TAU floor=$RESCUE_OBJECTNESS_FLOOR margin=$RESCUE_MARGIN recall_constraint=$RESCUE_MIN_RECALL precision_constraint=$RESCUE_MIN_PRECISION max_empty_fp_rate=$RESCUE_MAX_EMPTY_FP_RATE"
echo "[candidate_region_classifier] secondary_thresholding=$SECONDARY_THRESHOLDING min=$SECONDARY_MIN delta=$SECONDARY_DELTA recall_constraint=$SECONDARY_MIN_RECALL precision_constraint=$SECONDARY_MIN_PRECISION"
echo "[candidate_region_classifier] plot_test_genomes=$PLOT_TEST_GENOMES plot_dir=$PLOT_DIR"

CMD=(
    "$PYTHON_BIN" training/train_candidate_region_classifier.py
    --manifest "$MANIFEST"
    --candidate_regions "$CANDIDATE_REGIONS"
    --cn_checkpoint "$CN_CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
    --class_names "$CLASS_NAMES"
    --required_test_sample "$REQUIRED_TEST_SAMPLE"
    --n_test_samples "$N_TEST_SAMPLES"
    --embedding_normalization "$EMBEDDING_NORMALIZATION"
    --embedding_features "$EMBEDDING_FEATURES"
    --sample_baseline_min_candidates "$SAMPLE_BASELINE_MIN_CANDIDATES"
    --hidden_dims "$HIDDEN_DIMS"
    --tabular_features "$TABULAR_FEATURES"
    --tabular_hidden_dim "$TABULAR_HIDDEN_DIM"
    --activation "$ACTIVATION"
    --dropout "$DROPOUT"
    --epochs "$EPOCHS"
    --patience "$PATIENCE"
    --lr "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --background_weight "$BACKGROUND_WEIGHT"
    --type_loss_weight "$TYPE_LOSS_WEIGHT"
    --label_smoothing "$LABEL_SMOOTHING"
    --class_weighting "$CLASS_WEIGHTING"
    --tau_selection_metric "$TAU_SELECTION_METRIC"
    --threshold_calibration "$THRESHOLD_CALIBRATION"
    --threshold_tie_break "$THRESHOLD_TIE_BREAK"
    --tau_min "$TAU_MIN"
    --tau_max "$TAU_MAX"
    --tau_steps "$TAU_STEPS"
    --type_tau_min "$TYPE_TAU_MIN"
    --type_tau_max "$TYPE_TAU_MAX"
    --type_tau_steps "$TYPE_TAU_STEPS"
    --rescue_thresholding "$RESCUE_THRESHOLDING"
    --rescue_type_tau "$RESCUE_TYPE_TAU"
    --rescue_objectness_floor "$RESCUE_OBJECTNESS_FLOOR"
    --rescue_margin "$RESCUE_MARGIN"
    --rescue_type_tau_min "$RESCUE_TYPE_TAU_MIN"
    --rescue_type_tau_max "$RESCUE_TYPE_TAU_MAX"
    --rescue_type_tau_steps "$RESCUE_TYPE_TAU_STEPS"
    --rescue_objectness_floor_grid "$RESCUE_OBJECTNESS_FLOOR_GRID"
    --rescue_margin_min "$RESCUE_MARGIN_MIN"
    --rescue_margin_max "$RESCUE_MARGIN_MAX"
    --rescue_margin_steps "$RESCUE_MARGIN_STEPS"
    --rescue_min_recall "$RESCUE_MIN_RECALL"
    --rescue_min_precision "$RESCUE_MIN_PRECISION"
    --rescue_max_empty_fp_rate "$RESCUE_MAX_EMPTY_FP_RATE"
    --secondary_thresholding "$SECONDARY_THRESHOLDING"
    --secondary_min "$SECONDARY_MIN"
    --secondary_delta "$SECONDARY_DELTA"
    --secondary_min_min "$SECONDARY_MIN_MIN"
    --secondary_min_max "$SECONDARY_MIN_MAX"
    --secondary_min_steps "$SECONDARY_MIN_STEPS"
    --secondary_delta_min "$SECONDARY_DELTA_MIN"
    --secondary_delta_max "$SECONDARY_DELTA_MAX"
    --secondary_delta_steps "$SECONDARY_DELTA_STEPS"
    --secondary_min_recall "$SECONDARY_MIN_RECALL"
    --secondary_min_precision "$SECONDARY_MIN_PRECISION"
    --seed "$SEED"
    --device "$DEVICE"
)
if [[ -n "$GRAPH_CHECKPOINT" ]]; then
    CMD+=(--graph_checkpoint "$GRAPH_CHECKPOINT")
fi
if [[ -n "$TEST_SAMPLES" ]]; then
    CMD+=(--test_samples "$TEST_SAMPLES")
fi
if [[ -n "$LOGO_EPOCHS" ]]; then
    CMD+=(--logo_epochs "$LOGO_EPOCHS")
fi
if [[ -n "$LOGO_PATIENCE" ]]; then
    CMD+=(--logo_patience "$LOGO_PATIENCE")
fi
if [[ -n "${TAU:-}" ]]; then
    CMD+=(--tau "$TAU")
fi
if [[ -n "${TYPE_TAU:-}" ]]; then
    CMD+=(--type_tau "$TYPE_TAU")
fi
if [[ "$REUSE_EMBEDDINGS" == "1" ]]; then
    CMD+=(--reuse_embeddings)
fi
if [[ "${STRICT:-0}" == "1" ]]; then
    CMD+=(--strict)
fi

echo "[candidate_region_classifier] Training candidate-region classifier:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [[ "$PLOT_TEST_GENOMES" == "1" ]]; then
    rm -rf "$PLOT_DIR"
    PLOT_CMD=(
        "$PYTHON_BIN" discovery/plot_predicted_chromosomes.py
        --manifest "$MANIFEST"
        --prototype_distances "$OUTPUT_DIR/test_predictions.tsv"
        --output_dir "$PLOT_DIR"
        --plot_scope all
        --dpi "$PLOT_DPI"
    )
    if [[ -n "$PLOT_MAX_PLOTS" ]]; then
        PLOT_CMD+=(--max_plots "$PLOT_MAX_PLOTS")
    fi
    echo "[candidate_region_classifier] Plotting held-out candidate-region predictions:"
    printf '  %q' "${PLOT_CMD[@]}"
    printf '\n'
    "${PLOT_CMD[@]}"
fi

for expected_output in \
    "$OUTPUT_DIR/candidate_region_classifier.pt" \
    "$OUTPUT_DIR/candidate_embeddings.tsv" \
    "$OUTPUT_DIR/embeddings.npz" \
    "$OUTPUT_DIR/classification_predictions.tsv" \
    "$OUTPUT_DIR/train_predictions.tsv" \
    "$OUTPUT_DIR/test_predictions.tsv" \
    "$OUTPUT_DIR/metrics_summary.tsv" \
    "$OUTPUT_DIR/per_class_metrics.tsv" \
    "$OUTPUT_DIR/sample_splits.tsv" \
    "$OUTPUT_DIR/training_summary.json" \
    "$OUTPUT_DIR/type_thresholds.tsv" \
    "$OUTPUT_DIR/tabular_features.tsv" \
    "$OUTPUT_DIR/tabular_feature_names.txt" \
    "$OUTPUT_DIR/tabular_features.npz" \
    "$OUTPUT_DIR/selected_embedding_features.npz" \
    "$OUTPUT_DIR/embedding_features.txt" \
    "$OUTPUT_DIR/objectness_tau_sweep_calibration.tsv" \
    "$OUTPUT_DIR/type_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/rescue_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/secondary_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/objectness_tau_sweep_in_sample_train.tsv" \
    "$OUTPUT_DIR/type_threshold_sweep_in_sample_train.tsv" \
    "$OUTPUT_DIR/training_curves.png" \
    "$OUTPUT_DIR/split_metrics.png" \
    "$OUTPUT_DIR/per_class_metrics.png" \
    "$OUTPUT_DIR/objectness_tau_sweep.png" \
    "$OUTPUT_DIR/type_thresholds.png" \
    "$OUTPUT_DIR/prototype_distances.png" \
    "$OUTPUT_DIR/embedding_projection_predicted.png"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[candidate_region_classifier] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

if [[ "$THRESHOLD_CALIBRATION" == "logo" ]]; then
    for expected_logo_output in         "$OUTPUT_DIR/logo_calibration_raw.tsv"         "$OUTPUT_DIR/logo_calibration_predictions.tsv"         "$OUTPUT_DIR/logo_training_metrics.tsv"         "$OUTPUT_DIR/logo_metrics_summary.tsv"         "$OUTPUT_DIR/logo_per_class_metrics.tsv"         "$OUTPUT_DIR/objectness_tau_sweep_logo.tsv"         "$OUTPUT_DIR/type_threshold_sweep_logo.tsv"         "$OUTPUT_DIR/rescue_threshold_sweep_logo.tsv"         "$OUTPUT_DIR/secondary_threshold_sweep_logo.tsv"; do
        if [[ ! -f "$expected_logo_output" ]]; then
            echo "[candidate_region_classifier] ERROR: expected LOGO output not found: $expected_logo_output" >&2
            exit 1
        fi
    done
fi

if [[ "$PLOT_TEST_GENOMES" == "1" && ! -f "$PLOT_DIR/selected_predictions.tsv" ]]; then
    echo "[candidate_region_classifier] ERROR: expected plot index not found: $PLOT_DIR/selected_predictions.tsv" >&2
    exit 1
fi

echo "[candidate_region_classifier] Done."
