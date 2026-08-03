#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 SEVERUS_VCF_LIST OUTPUT_DIR [additional pretrain_graph.py arguments]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
input_list="$1"
output_dir="$2"
shift 2

exec "$python_bin" "$repo_dir/pretrain/pretrain_graph.py" \
    --input_list "$input_list" \
    --output_dir "$output_dir" \
    "$@"
