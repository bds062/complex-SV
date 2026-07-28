#!/usr/bin/env bash
set -euo pipefail

# Build caller-supervised labels, then train the general candidate-region model.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
BASE="${BASE:-$PROJECT_DIR/../results/pipeline14}"

"$PYTHON_BIN" "$SCRIPT_DIR/build_pipeline14_caller_labels.py" \
    --output_dir "$BASE" \
    --bfb_calls "$PROJECT_DIR/../results/bfbarchitect3/bfb_calls.tsv" \
    --shatterseek_calls "$PROJECT_DIR/../results/shatterseek2/chromothripsis_calls.tsv" \
    --coral_calls "$PROJECT_DIR/../results/coral/coral_ecDNA_candidate_calls.tsv"

MODEL_DIR="$BASE/candidate_region_classifier_general"

"$PYTHON_BIN" "$PROJECT_DIR/training/cross_fold_caller_label_study.py" \
    --source_dir "$MODEL_DIR" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --output_dir "$MODEL_DIR" \
    --class_names "ecDNA,chromothripsis,BFB" \
    --fold_size 6

rm -rf "$MODEL_DIR/test_genome_plots"
"$PYTHON_BIN" "$PROJECT_DIR/discovery/plot_predicted_chromosomes.py" \
    --manifest "$BASE/complex_sv_manifest.tsv" \
    --prototype_distances "$MODEL_DIR/cross_fold_predictions.tsv" \
    --output_dir "$MODEL_DIR/test_genome_plots" \
    --plot_scope all \
    --dpi 180
