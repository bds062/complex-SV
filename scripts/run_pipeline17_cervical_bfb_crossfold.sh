#!/usr/bin/env bash
set -euo pipefail

# Rebuild pipeline15 with curated cervical-panel BFB coordinates from CSV column 2.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
BASE="${BASE:-$PROJECT_DIR/../results/pipeline17}"
SOURCE_BASE="${SOURCE_BASE:-$PROJECT_DIR/../results/pipeline12}"
MANIFEST="${MANIFEST:-$SOURCE_BASE/complex_sv_manifest.tsv}"
BASE_CANDIDATES="${BASE_CANDIDATES:-$SOURCE_BASE/merged_candidate_regions.csv}"
BFB_CSV="${BFB_CSV:-$PROJECT_DIR/../labels/BFB in Cervical Panel Data - BFB.csv}"
SHATTERSEEK_CALLS="${SHATTERSEEK_CALLS:-$PROJECT_DIR/../results/shatterseek2/chromothripsis_calls.tsv}"
CORAL_CALLS="${CORAL_CALLS:-$PROJECT_DIR/../results/coral/coral_ecDNA_candidate_calls.tsv}"
CN_CHECKPOINT="${CN_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$PROJECT_DIR/../results/pipeline3/sv3/graph_encoder.pt}"
FOLD_SIZE="${FOLD_SIZE:-6}"
PLOT_GENOMES="${PLOT_GENOMES:-1}"
DEVICE="${DEVICE:-auto}"
FAST_THRESHOLDS="${FAST_THRESHOLDS:-0}"

mkdir -p "$BASE"
cp "$MANIFEST" "$BASE/complex_sv_manifest.tsv"

"$PYTHON_BIN" "$SCRIPT_DIR/prepare_cervical_panel_bfb_calls.py" \
    --input_csv "$BFB_CSV" \
    --output_tsv "$BASE/cervical_panel_bfb_calls.tsv" \
    --audit_tsv "$BASE/cervical_panel_bfb_calls_audit.tsv" \
    --summary_json "$BASE/cervical_panel_bfb_calls_summary.json"

"$PYTHON_BIN" "$SCRIPT_DIR/identify_caller_aware_candidate_regions.py" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --base_candidates "$BASE_CANDIDATES" \
    --bfb_calls "$BASE/cervical_panel_bfb_calls.tsv" \
    --shatterseek_calls "$SHATTERSEEK_CALLS" \
    --coral_calls "$CORAL_CALLS" \
    --output_dir "$BASE"

"$PYTHON_BIN" "$SCRIPT_DIR/label_pipeline15_candidates.py" \
    --candidates "$BASE/candidate_regions_unlabeled.csv" \
    --members "$BASE/candidate_region_members.tsv" \
    --external_regions "$BASE/external_regions.tsv" \
    --output_dir "$BASE"

GENERAL_DIR="$BASE/candidate_region_classifier_general"
FEWSHOT_DIR="$BASE/candidate_region_classifier_fewshot_general"
HYBRID_DIR="$BASE/candidate_region_classifier_hybrid_general_fewshot"
mkdir -p "$GENERAL_DIR"

"$PYTHON_BIN" "$PROJECT_DIR/training/train_candidate_region_classifier.py" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --candidate_regions "$BASE/merged_candidate_regions.csv" \
    --cn_checkpoint "$CN_CHECKPOINT" \
    --graph_checkpoint "$GRAPH_CHECKPOINT" \
    --output_dir "$GENERAL_DIR" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --embedding_normalization none \
    --embedding_features full \
    --tabular_features safe \
    --subtype_targets general \
    --subtype_thresholding off \
    --cluster_aggregation max \
    --embeddings_only

threshold_args=()
if [[ "$FAST_THRESHOLDS" == "1" ]]; then
    threshold_args+=(--fast_thresholds)
fi

"$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_study.py" \
    --source_dir "$GENERAL_DIR" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --output_dir "$GENERAL_DIR" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --bfb_label "Cervical-panel BFB" \
    --fold_size "$FOLD_SIZE" \
    --device "$DEVICE" \
    "${threshold_args[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_fewshot_study.py" \
    --source_dir "$GENERAL_DIR" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --output_dir "$FEWSHOT_DIR" \
    --fold_assignments "$GENERAL_DIR/fold_assignments.tsv" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --bfb_label "Cervical-panel BFB" \
    --fold_size "$FOLD_SIZE" \
    --containing_prototypes 1 \
    --min_prototype_members 1 \
    --min_cluster_members 2 \
    "${threshold_args[@]}"

"$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_hybrid_study.py" \
    --source_dir "$GENERAL_DIR" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --output_dir "$HYBRID_DIR" \
    --fold_assignments "$GENERAL_DIR/fold_assignments.tsv" \
    --fewshot_source_dir "$FEWSHOT_DIR" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --bfb_label "Cervical-panel BFB" \
    --fold_size "$FOLD_SIZE" \
    "${threshold_args[@]}"

if [[ "$PLOT_GENOMES" == "1" ]]; then
    for model_dir in "$GENERAL_DIR" "$FEWSHOT_DIR" "$HYBRID_DIR"; do
        rm -rf "$model_dir/test_genome_plots"
        "$PYTHON_BIN" "$PROJECT_DIR/discovery/plot_predicted_chromosomes.py" \
            --manifest "$BASE/complex_sv_manifest.tsv" \
            --prototype_distances "$model_dir/cross_fold_predictions.tsv" \
            --output_dir "$model_dir/test_genome_plots" \
            --plot_scope all \
            --dpi 180
    done
fi

echo "Pipeline17 curated-BFB caller-aware cross-fold comparison complete: $BASE"
