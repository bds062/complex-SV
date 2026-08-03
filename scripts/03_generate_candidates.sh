#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MANIFEST.tsv OUTPUT_DIR [additional generator arguments]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$1"
output_dir="$2"
shift 2

exec "$repo_dir/candidate_generator/run.sh" \
    "$manifest" \
    "$output_dir" \
    "$@"
