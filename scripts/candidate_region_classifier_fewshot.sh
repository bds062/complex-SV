#!/usr/bin/env bash
set -euo pipefail

# Train a prototypical few-shot candidate-region complex-SV classifier.
# This is a parallel experiment to candidate_region_classifier.sh:
# it uses the same candidate-region embeddings/features/splits, but replaces
# the neural classifier head with class prototypes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE="${BASE:-../results/pipeline11}"
MANIFEST="${MANIFEST:-$BASE/complex_sv_manifest.tsv}"
CANDIDATE_REGIONS="${CANDIDATE_REGIONS:-$BASE/merged_candidate_regions.csv}"

SUBTYPE_TARGETS="${SUBTYPE_TARGETS:-general}"
if [[ -z "${SUBTYPE_THRESHOLDING+x}" ]]; then
    if [[ "$SUBTYPE_TARGETS" == "specific" ]]; then
        SUBTYPE_THRESHOLDING="optimize"
    else
        SUBTYPE_THRESHOLDING="off"
    fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-$BASE/candidate_region_classifier_fewshot_${SUBTYPE_TARGETS}}"
PLOT_TEST_GENOMES="${PLOT_TEST_GENOMES:-1}"
PLOT_POST_MODEL_UMAPS="${PLOT_POST_MODEL_UMAPS:-1}"
PLOT_DIR="${PLOT_DIR:-$OUTPUT_DIR/test_genome_plots}"
POST_MODEL_UMAP_DIR="${POST_MODEL_UMAP_DIR:-$OUTPUT_DIR/class_specific_umaps_post_model}"
PLOT_DPI="${PLOT_DPI:-180}"
PLOT_MAX_PLOTS="${PLOT_MAX_PLOTS:-}"

CN_CHECKPOINT="${CN_CHECKPOINT:-../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-/data/KolmogorovLab/srinivasanbd/results/pipeline3/sv3/graph_encoder.pt}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"

CLASS_NAMES="${CLASS_NAMES:-ecDNA,Seismic_Amplification,chromothripsis,BFB}"
REQUIRED_TEST_SAMPLE="${REQUIRED_TEST_SAMPLE:-H1395,H1437,ME180,SCC154,ZR7530,SNU1245}"
TEST_SAMPLES="${TEST_SAMPLES:-}"
N_TEST_SAMPLES="${N_TEST_SAMPLES:-6}"
SEED="${SEED:-42}"

# Optional source embedding directory. If present, this avoids recomputing the
# expensive region/context embeddings and trains the prototype head from the
# existing embeddings + candidate_embeddings.tsv.
EMBEDDING_DIR="${EMBEDDING_DIR:-$BASE/candidate_region_classifier_general}"
EMBEDDINGS_NPZ="${EMBEDDINGS_NPZ:-$EMBEDDING_DIR/embeddings.npz}"
METADATA_TSV="${METADATA_TSV:-$EMBEDDING_DIR/candidate_embeddings.tsv}"
FORCE_EMBED="${FORCE_EMBED:-0}"

EMBEDDING_NORMALIZATION="${EMBEDDING_NORMALIZATION:-none}"
EMBEDDING_FEATURES="${EMBEDDING_FEATURES:-full}"
SAMPLE_BASELINE_MIN_CANDIDATES="${SAMPLE_BASELINE_MIN_CANDIDATES:-3}"
TABULAR_FEATURES="${TABULAR_FEATURES:-safe}"
TABULAR_WEIGHT="${TABULAR_WEIGHT:-1.0}"
FINAL_L2_NORMALIZE="${FINAL_L2_NORMALIZE:-1}"

CONTAINING_PROTOTYPES="${CONTAINING_PROTOTYPES:-1}"
MIN_PROTOTYPE_MEMBERS="${MIN_PROTOTYPE_MEMBERS:-1}"
MIN_CLUSTER_MEMBERS="${MIN_CLUSTER_MEMBERS:-2}"
PROTOTYPE_SUBTYPE_WEIGHTING="${PROTOTYPE_SUBTYPE_WEIGHTING:-auto}"
PROTOTYPE_TEMPERATURE="${PROTOTYPE_TEMPERATURE:-0.25}"
SCORE_TRANSFORM="${SCORE_TRANSFORM:-exp}"

THRESHOLD_CALIBRATION="${THRESHOLD_CALIBRATION:-logo}"
THRESHOLD_TIE_BREAK="${THRESHOLD_TIE_BREAK:-low}"
TAU_SELECTION_METRIC="${TAU_SELECTION_METRIC:-f1}"
TAU_MIN="${TAU_MIN:-0.05}"
TAU_MAX="${TAU_MAX:-0.95}"
TAU_STEPS="${TAU_STEPS:-91}"
TYPE_TAU_MIN="${TYPE_TAU_MIN:-0.05}"
TYPE_TAU_MAX="${TYPE_TAU_MAX:-0.95}"
TYPE_TAU_STEPS="${TYPE_TAU_STEPS:-91}"
CLUSTER_AGGREGATION="${CLUSTER_AGGREGATION:-max}"
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

if ! "$PYTHON_BIN" -c "import pandas, torch, matplotlib, sklearn" >/dev/null 2>&1; then
    echo "[candidate_region_classifier_fewshot] ERROR: $PYTHON_BIN cannot import pandas, torch, matplotlib, and sklearn." >&2
    exit 1
fi

USE_EXTERNAL_EMBEDDINGS=0
if [[ "$FORCE_EMBED" != "1" && -f "$EMBEDDINGS_NPZ" && -f "$METADATA_TSV" ]]; then
    USE_EXTERNAL_EMBEDDINGS=1
fi

if [[ ! -f "$MANIFEST" ]]; then
    echo "[candidate_region_classifier_fewshot] ERROR: required file not found: $MANIFEST" >&2
    exit 1
fi
if [[ "$USE_EXTERNAL_EMBEDDINGS" != "1" ]]; then
    for required_file in "$CANDIDATE_REGIONS" "$CN_CHECKPOINT"; do
        if [[ ! -f "$required_file" ]]; then
            echo "[candidate_region_classifier_fewshot] ERROR: required file not found: $required_file" >&2
            exit 1
        fi
    done
    if [[ -n "$GRAPH_CHECKPOINT" && ! -f "$GRAPH_CHECKPOINT" ]]; then
        echo "[candidate_region_classifier_fewshot] ERROR: graph checkpoint not found: $GRAPH_CHECKPOINT" >&2
        exit 1
    fi
fi

mkdir -p "$OUTPUT_DIR"

echo "[candidate_region_classifier_fewshot] project_dir=$PROJECT_DIR"
echo "[candidate_region_classifier_fewshot] manifest=$MANIFEST"
echo "[candidate_region_classifier_fewshot] candidate_regions=$CANDIDATE_REGIONS"
echo "[candidate_region_classifier_fewshot] output_dir=$OUTPUT_DIR"
echo "[candidate_region_classifier_fewshot] class_names=$CLASS_NAMES"
echo "[candidate_region_classifier_fewshot] required_test_sample=$REQUIRED_TEST_SAMPLE test_samples=$TEST_SAMPLES n_test_samples=$N_TEST_SAMPLES seed=$SEED"
echo "[candidate_region_classifier_fewshot] use_external_embeddings=$USE_EXTERNAL_EMBEDDINGS embeddings_npz=$EMBEDDINGS_NPZ metadata_tsv=$METADATA_TSV"
echo "[candidate_region_classifier_fewshot] embedding_normalization=$EMBEDDING_NORMALIZATION embedding_features=$EMBEDDING_FEATURES tabular_features=$TABULAR_FEATURES tabular_weight=$TABULAR_WEIGHT"
echo "[candidate_region_classifier_fewshot] subtype_targets=$SUBTYPE_TARGETS subtype_thresholding=$SUBTYPE_THRESHOLDING cluster_aggregation=$CLUSTER_AGGREGATION"
echo "[candidate_region_classifier_fewshot] containing_prototypes=$CONTAINING_PROTOTYPES min_members=$MIN_PROTOTYPE_MEMBERS min_cluster_members=$MIN_CLUSTER_MEMBERS subtype_weighting=$PROTOTYPE_SUBTYPE_WEIGHTING temperature=$PROTOTYPE_TEMPERATURE score_transform=$SCORE_TRANSFORM"
echo "[candidate_region_classifier_fewshot] threshold_calibration=$THRESHOLD_CALIBRATION threshold_tie_break=$THRESHOLD_TIE_BREAK"
echo "[candidate_region_classifier_fewshot] rescue_thresholding=$RESCUE_THRESHOLDING type_tau=$RESCUE_TYPE_TAU floor=$RESCUE_OBJECTNESS_FLOOR margin=$RESCUE_MARGIN"
echo "[candidate_region_classifier_fewshot] secondary_thresholding=$SECONDARY_THRESHOLDING min=$SECONDARY_MIN delta=$SECONDARY_DELTA"
echo "[candidate_region_classifier_fewshot] plot_test_genomes=$PLOT_TEST_GENOMES plot_dir=$PLOT_DIR"
echo "[candidate_region_classifier_fewshot] plot_post_model_umaps=$PLOT_POST_MODEL_UMAPS post_model_umap_dir=$POST_MODEL_UMAP_DIR"

CMD=(
    "$PYTHON_BIN" training/train_candidate_region_fewshot_classifier.py
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
    --tabular_features "$TABULAR_FEATURES"
    --tabular_weight "$TABULAR_WEIGHT"
    --containing_prototypes "$CONTAINING_PROTOTYPES"
    --min_prototype_members "$MIN_PROTOTYPE_MEMBERS"
    --min_cluster_members "$MIN_CLUSTER_MEMBERS"
    --prototype_subtype_weighting "$PROTOTYPE_SUBTYPE_WEIGHTING"
    --prototype_temperature "$PROTOTYPE_TEMPERATURE"
    --score_transform "$SCORE_TRANSFORM"
    --tau_selection_metric "$TAU_SELECTION_METRIC"
    --threshold_calibration "$THRESHOLD_CALIBRATION"
    --threshold_tie_break "$THRESHOLD_TIE_BREAK"
    --tau_min "$TAU_MIN"
    --tau_max "$TAU_MAX"
    --tau_steps "$TAU_STEPS"
    --type_tau_min "$TYPE_TAU_MIN"
    --type_tau_max "$TYPE_TAU_MAX"
    --type_tau_steps "$TYPE_TAU_STEPS"
    --subtype_targets "$SUBTYPE_TARGETS"
    --subtype_thresholding "$SUBTYPE_THRESHOLDING"
    --cluster_aggregation "$CLUSTER_AGGREGATION"
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
if [[ "$FINAL_L2_NORMALIZE" != "1" ]]; then
    CMD+=(--no-final_l2_normalize)
fi
if [[ -n "$GRAPH_CHECKPOINT" ]]; then
    CMD+=(--graph_checkpoint "$GRAPH_CHECKPOINT")
fi
if [[ -n "$TEST_SAMPLES" ]]; then
    CMD+=(--test_samples "$TEST_SAMPLES")
fi
if [[ "$USE_EXTERNAL_EMBEDDINGS" == "1" ]]; then
    CMD+=(--embeddings_npz "$EMBEDDINGS_NPZ" --metadata_tsv "$METADATA_TSV")
fi
if [[ -n "${TAU:-}" ]]; then
    CMD+=(--tau "$TAU")
fi
if [[ -n "${TYPE_TAU:-}" ]]; then
    CMD+=(--type_tau "$TYPE_TAU")
fi
if [[ "${STRICT:-0}" == "1" ]]; then
    CMD+=(--strict)
fi

echo "[candidate_region_classifier_fewshot] Training few-shot candidate-region classifier:"
printf '  %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

if [[ "$PLOT_POST_MODEL_UMAPS" == "1" ]]; then
    POST_UMAP_CMD=(
        "$PYTHON_BIN" scripts/plot_class_specific_umaps.py
        "$OUTPUT_DIR/post_model_embeddings.npz"
        --metadata "$OUTPUT_DIR/candidate_embeddings.tsv"
        --output-dir "$POST_MODEL_UMAP_DIR"
        --title-prefix "fewshot_${SUBTYPE_TARGETS}_post_model"
    )
    echo "[candidate_region_classifier_fewshot] Plotting post-model class-specific UMAPs:"
    printf '  %q' "${POST_UMAP_CMD[@]}"
    printf '\n'
    "${POST_UMAP_CMD[@]}"
fi

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
    echo "[candidate_region_classifier_fewshot] Plotting held-out few-shot predictions:"
    printf '  %q' "${PLOT_CMD[@]}"
    printf '\n'
    "${PLOT_CMD[@]}"
fi

for expected_output in \
    "$OUTPUT_DIR/candidate_region_classifier_fewshot.pt" \
    "$OUTPUT_DIR/fewshot_prototypes.tsv" \
    "$OUTPUT_DIR/fewshot_prototypes.npz" \
    "$OUTPUT_DIR/fewshot_class_summary.tsv" \
    "$OUTPUT_DIR/post_model_embeddings.npz" \
    "$OUTPUT_DIR/candidate_embeddings.tsv" \
    "$OUTPUT_DIR/embeddings.npz" \
    "$OUTPUT_DIR/classification_predictions.tsv" \
    "$OUTPUT_DIR/train_predictions.tsv" \
    "$OUTPUT_DIR/test_predictions.tsv" \
    "$OUTPUT_DIR/cluster_predictions.tsv" \
    "$OUTPUT_DIR/row_raw_predictions.tsv" \
    "$OUTPUT_DIR/cluster_aggregated_raw_predictions.tsv" \
    "$OUTPUT_DIR/metrics_summary.tsv" \
    "$OUTPUT_DIR/per_class_metrics.tsv" \
    "$OUTPUT_DIR/sample_splits.tsv" \
    "$OUTPUT_DIR/training_summary.json" \
    "$OUTPUT_DIR/type_thresholds.tsv" \
    "$OUTPUT_DIR/subtype_thresholds.tsv" \
    "$OUTPUT_DIR/tabular_features.tsv" \
    "$OUTPUT_DIR/tabular_feature_names.txt" \
    "$OUTPUT_DIR/tabular_features.npz" \
    "$OUTPUT_DIR/selected_embedding_features.npz" \
    "$OUTPUT_DIR/embedding_features.txt" \
    "$OUTPUT_DIR/objectness_tau_sweep_calibration.tsv" \
    "$OUTPUT_DIR/type_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/subtype_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/rescue_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/secondary_threshold_sweep_calibration.tsv" \
    "$OUTPUT_DIR/objectness_tau_sweep_in_sample_train.tsv" \
    "$OUTPUT_DIR/type_threshold_sweep_in_sample_train.tsv" \
    "$OUTPUT_DIR/subtype_threshold_sweep_in_sample_train.tsv" \
    "$OUTPUT_DIR/split_metrics.png" \
    "$OUTPUT_DIR/per_class_metrics.png" \
    "$OUTPUT_DIR/objectness_tau_sweep.png" \
    "$OUTPUT_DIR/type_thresholds.png" \
    "$OUTPUT_DIR/prototype_distances.png" \
    "$OUTPUT_DIR/embedding_projection_predicted.png"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[candidate_region_classifier_fewshot] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

if [[ "$THRESHOLD_CALIBRATION" == "logo" ]]; then
    for expected_logo_output in \
        "$OUTPUT_DIR/logo_calibration_raw.tsv" \
        "$OUTPUT_DIR/logo_calibration_predictions.tsv" \
        "$OUTPUT_DIR/logo_training_metrics.tsv" \
        "$OUTPUT_DIR/logo_metrics_summary.tsv" \
        "$OUTPUT_DIR/logo_per_class_metrics.tsv" \
        "$OUTPUT_DIR/objectness_tau_sweep_logo.tsv" \
        "$OUTPUT_DIR/type_threshold_sweep_logo.tsv" \
        "$OUTPUT_DIR/subtype_threshold_sweep_logo.tsv" \
        "$OUTPUT_DIR/rescue_threshold_sweep_logo.tsv" \
        "$OUTPUT_DIR/secondary_threshold_sweep_logo.tsv"; do
        if [[ ! -f "$expected_logo_output" ]]; then
            echo "[candidate_region_classifier_fewshot] ERROR: expected LOGO output not found: $expected_logo_output" >&2
            exit 1
        fi
    done
fi

if [[ "$PLOT_POST_MODEL_UMAPS" == "1" && ! -f "$POST_MODEL_UMAP_DIR/class_specific_umap_summary.tsv" ]]; then
    echo "[candidate_region_classifier_fewshot] ERROR: expected post-model UMAP summary not found: $POST_MODEL_UMAP_DIR/class_specific_umap_summary.tsv" >&2
    exit 1
fi

if [[ "$PLOT_TEST_GENOMES" == "1" && ! -f "$PLOT_DIR/selected_predictions.tsv" ]]; then
    echo "[candidate_region_classifier_fewshot] ERROR: expected plot index not found: $PLOT_DIR/selected_predictions.tsv" >&2
    exit 1
fi

echo "[candidate_region_classifier_fewshot] Done."
