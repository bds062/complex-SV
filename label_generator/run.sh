#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 CURATED_LABEL_DIR OUTPUT_DIR" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
exec "$python_bin" "$repo_dir/label_generator/generate_labels.py" \
    --label-dir "$1" \
    --output-dir "$2"
