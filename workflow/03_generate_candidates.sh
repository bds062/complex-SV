#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MANIFEST.tsv OUTPUT_DIR [additional generator arguments]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
manifest="$1"
output_dir="$2"
shift 2

exec "$python_bin" "$repo_dir/scripts/generate_standalone_candidate_regions.py" \
    "$manifest" \
    --output_dir "$output_dir" \
    --centromeres "$repo_dir/data/grch38.cen_coord.curated.bed" \
    "$@"
