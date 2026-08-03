#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
    cat >&2 <<'EOF'
Usage: 08_train_chromosome_model.sh EMBEDDING_DIR TABULAR.tsv LABELS.tsv \
       TEST_SAMPLE OUTPUT_DIR [additional train_chromosome_loo.py arguments]

Run once per labeled TEST_SAMPLE for leave-one-genome-out evaluation.
EOF
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
embedding_dir="$1"; tabular="$2"; labels="$3"; test_sample="$4"; output="$5"
shift 5

exec "$python_bin" "$repo_dir/scripts/train_chromosome_loo.py" \
    --embedding-dir "$embedding_dir" \
    --tabular "$tabular" \
    --labels "$labels" \
    --test-sample "$test_sample" \
    --output "$output" \
    "$@"
