#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
    cat >&2 <<'EOF'
Usage: 05_train_localization_model.sh CANDIDATES.csv LABELS.tsv EMBEDDINGS.npz \
       SELECTED_EMBEDDINGS.npz TABULAR_FEATURES.npz TEST_SAMPLE OUTPUT_DIR \
       [additional train_localization_loo.py arguments]

LABELS.tsv must contain: region_id, sample_id, chrom, start, end, label.
Run once for every labeled TEST_SAMPLE to reproduce full LOO evaluation.
EOF
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
candidates="$1"; labels="$2"; embedding_bundle="$3"; selected="$4"
tabular="$5"; test_sample="$6"; output_dir="$7"
shift 7

exec "$python_bin" "$repo_dir/scripts/train_localization_loo.py" \
    --candidates "$candidates" \
    --labels "$labels" \
    --embedding-bundle "$embedding_bundle" \
    --selected-embeddings "$selected" \
    --tabular-features "$tabular" \
    --config "$repo_dir/configs/localization.json" \
    --test-sample "$test_sample" \
    --output "$output_dir" \
    "$@"
