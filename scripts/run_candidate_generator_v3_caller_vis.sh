#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/../envs/env2/bin/python}"
CANDIDATES="${CANDIDATES:-$PROJECT_DIR/../results/standalone_candidate_generator_v3/merged_candidate_regions.csv}"
EXTERNAL_REGIONS="${EXTERNAL_REGIONS:-$PROJECT_DIR/../results/pipeline17/external_regions.tsv}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/../results/caller_vis/candidate_generator_v3}"

"$PYTHON_BIN" "$SCRIPT_DIR/plot_candidate_generator_caller_vis.py" \
    --candidates "$CANDIDATES" \
    --external_regions "$EXTERNAL_REGIONS" \
    --output_dir "$OUTPUT_DIR"

echo "Candidate generator v3 caller figures: $OUTPUT_DIR"
