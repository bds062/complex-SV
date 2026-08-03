#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 {loo SAMPLE_ID|grouped5 FOLD_NUMBER} OUTPUT_DIR" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
split="$1"

if [[ "$split" == "loo" ]]; then
    [[ $# -eq 3 ]] || { echo "Usage: $0 loo SAMPLE_ID OUTPUT_DIR" >&2; exit 2; }
    exec "$python_bin" "$repo_dir/final-results/chromosome_model/code/chromosome_model.py" \
        --split loo --test-sample "$2" --output "$3"
elif [[ "$split" == "grouped5" ]]; then
    [[ $# -eq 3 ]] || { echo "Usage: $0 grouped5 FOLD_NUMBER OUTPUT_DIR" >&2; exit 2; }
    exec "$python_bin" "$repo_dir/final-results/chromosome_model/code/chromosome_model.py" \
        --split grouped5 --fold "$2" --output "$3"
else
    echo "Unknown split '$split'; expected loo or grouped5" >&2
    exit 2
fi
