#!/usr/bin/env bash
set -euo pipefail

# Compare the two-centroid few-shot model with pipeline14's neural classifier
# on the same caller labels and grouped six-genome folds.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
BASE="${BASE:-$PROJECT_DIR/../results/pipeline14}"
SOURCE_DIR="${SOURCE_DIR:-$BASE/candidate_region_classifier_general}"
OUTPUT_DIR="${OUTPUT_DIR:-$BASE/candidate_region_classifier_fewshot_general}"
FOLD_ASSIGNMENTS="${FOLD_ASSIGNMENTS:-$SOURCE_DIR/fold_assignments.tsv}"

"$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_fewshot_study.py" \
    --source_dir "$SOURCE_DIR" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --output_dir "$OUTPUT_DIR" \
    --fold_assignments "$FOLD_ASSIGNMENTS" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --fold_size 6 \
    --containing_prototypes 1 \
    --min_prototype_members 1 \
    --min_cluster_members 2

rm -rf "$OUTPUT_DIR/test_genome_plots"
"$PYTHON_BIN" "$PROJECT_DIR/discovery/plot_predicted_chromosomes.py" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --prototype_distances "$OUTPUT_DIR/cross_fold_predictions.tsv" \
    --output_dir "$OUTPUT_DIR/test_genome_plots" \
    --plot_scope all \
    --dpi 180
