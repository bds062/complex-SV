#!/usr/bin/env bash
set -euo pipefail

# Generate caller-free v3 candidates, then benchmark them offline against pipeline17 calls.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
MANIFEST="${MANIFEST:-$PROJECT_DIR/../results/pipeline17/complex_sv_manifest.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/../results/standalone_candidate_generator_v3}"
EXTERNAL_REGIONS="${EXTERNAL_REGIONS:-$PROJECT_DIR/../results/pipeline17/external_regions.tsv}"

"$PYTHON_BIN" "$SCRIPT_DIR/candidate_generator_v3.py" "$MANIFEST" \
    --output_dir "$OUTPUT_DIR" \
    --keep_going

echo "[offline evaluation] External calls are read only after candidate generation."
"$PYTHON_BIN" "$SCRIPT_DIR/evaluate_candidate_region_recall.py" \
    --candidates "$OUTPUT_DIR/merged_candidate_regions.csv" \
    --external_regions "$EXTERNAL_REGIONS" \
    --output_dir "$OUTPUT_DIR/evaluation"

echo "Candidate generator v3 benchmark complete: $OUTPUT_DIR"
