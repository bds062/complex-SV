#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-../results/pipeline8/prototype_chrom_arm_sample_norm}"
CLASSIFICATION_DIR="${CLASSIFICATION_DIR:-$PROTOTYPE_ROOT/classification_head_applied}"
MULTILABEL_DIR="${MULTILABEL_DIR:-$PROTOTYPE_ROOT/multilabel_classification_head_applied}"
FEWSHOT_DIR="${FEWSHOT_DIR:-$PROTOTYPE_ROOT/fewshot_classification_head_applied}"
# PROTOTYPE_ROOT="${PROTOTYPE_ROOT:-../results/pipeline5/prototype_chrom_arm_sample_norm}"
# CLASSIFICATION_DIR="${CLASSIFICATION_DIR:-$PROTOTYPE_ROOT/classification_head}"
# MULTILABEL_DIR="${MULTILABEL_DIR:-$PROTOTYPE_ROOT/multilabel_classification_head}"
# FEWSHOT_DIR="${FEWSHOT_DIR:-$PROTOTYPE_ROOT/fewshot_classification_head}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROTOTYPE_ROOT/model_analysis}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "../envs/env2/bin/python" ]]; then
    PYTHON_BIN="../envs/env2/bin/python"
fi

if ! "$PYTHON_BIN" -c "import pandas, matplotlib" >/dev/null 2>&1; then
    echo "[model_call_analysis] ERROR: $PYTHON_BIN cannot import pandas and matplotlib." >&2
    exit 1
fi

ANALYZE_CMD=(
    "$PYTHON_BIN" discovery/analyze_model_call_overlap.py
    --classification_dir "$CLASSIFICATION_DIR"
    --multilabel_dir "$MULTILABEL_DIR"
    --fewshot_dir "$FEWSHOT_DIR"
    --output_dir "$OUTPUT_DIR"
)

echo "[model_call_analysis] Comparing model calls:"
printf '  %q' "${ANALYZE_CMD[@]}"
printf '\n'
"${ANALYZE_CMD[@]}"

for expected_output in \
    "$OUTPUT_DIR/called_overlap_venn_all_calls.png" \
    "$OUTPUT_DIR/called_overlap_venn_unlabeled_scan.png" \
    "$OUTPUT_DIR/called_overlap_upset_all_calls.png" \
    "$OUTPUT_DIR/called_overlap_upset_unlabeled_scan.png" \
    "$OUTPUT_DIR/pairwise_jaccard_all_calls.png" \
    "$OUTPUT_DIR/pairwise_jaccard_unlabeled_scan.png" \
    "$OUTPUT_DIR/class_distribution_all_calls.png" \
    "$OUTPUT_DIR/class_distribution_unlabeled_scan.png" \
    "$OUTPUT_DIR/overlap_summary.tsv" \
    "$OUTPUT_DIR/pairwise_overlap.tsv" \
    "$OUTPUT_DIR/model_agreement_by_candidate.tsv"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[model_call_analysis] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

echo "[model_call_analysis] Done: $OUTPUT_DIR"
