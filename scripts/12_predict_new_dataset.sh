#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 MANIFEST.tsv OUTPUT_DIR [DEVICE]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
manifest="$1"
output_dir="$2"
device="${3:-auto}"
profile="${CANDIDATE_PROFILE:-balanced}"

mkdir -p "$output_dir"

"$repo_dir/candidate_generator/run.sh" \
    "$manifest" "$output_dir/candidates" --profile "$profile" --keep_going

"$repo_dir/scripts/04_embed_candidates.sh" \
    "$manifest" \
    "$output_dir/candidates/merged_candidate_regions.csv" \
    "$output_dir/localization_features" \
    --device "$device"

"$python_bin" "$repo_dir/scripts/10_predict_localization.py" \
    --candidates "$output_dir/candidates/merged_candidate_regions.csv" \
    --embedding-bundle "$output_dir/localization_features/embeddings.npz" \
    --selected-embeddings "$output_dir/localization_features/selected_embedding_features.npz" \
    --tabular-features "$output_dir/localization_features/tabular_features.npz" \
    --output "$output_dir/localized_calls" \
    --device "$device"

"$repo_dir/scripts/07_embed_chromosomes.sh" \
    "$manifest" "$output_dir/chromosome_features" --device "$device"

"$python_bin" "$repo_dir/scripts/11_predict_chromosomes.py" \
    --embedding-dir "$output_dir/chromosome_features/embeddings" \
    --tabular "$output_dir/chromosome_features/chromosome_tabular.tsv" \
    --output "$output_dir/chromosome_calls" \
    --device "$device"

echo "Predictions written under $output_dir"
