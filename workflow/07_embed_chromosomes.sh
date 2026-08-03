#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 MANIFEST.tsv OUTPUT_DIR [additional embed_corpus.py arguments]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
manifest="$1"
output_dir="$2"
shift 2
mkdir -p "$output_dir"

"$python_bin" "$repo_dir/discovery/embed_corpus.py" \
    --manifest "$manifest" \
    --cn_checkpoint "$repo_dir/models/pretrained_featurizer/cn_encoder.pt" \
    --graph_checkpoint "$repo_dir/models/pretrained_featurizer/sv_graph_encoder.pt" \
    --output_dir "$output_dir/embeddings" \
    --candidate_source chromosomes \
    --embedding_normalization none \
    --report_scope all \
    "$@"

"$python_bin" "$repo_dir/workflow/build_chromosome_tabular.py" \
    --manifest "$manifest" \
    --output "$output_dir/chromosome_tabular.tsv"
