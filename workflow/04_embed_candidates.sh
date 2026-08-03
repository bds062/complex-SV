#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 MANIFEST.tsv CANDIDATES.csv OUTPUT_DIR [additional classifier arguments]" >&2
    exit 2
fi

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
manifest="$1"
candidates="$2"
output_dir="$3"
shift 3

exec "$python_bin" "$repo_dir/training/train_candidate_region_classifier.py" \
    --manifest "$manifest" \
    --candidate_regions "$candidates" \
    --cn_checkpoint "$repo_dir/models/pretrained/cn_encoder.pt" \
    --graph_checkpoint "$repo_dir/models/pretrained/sv_graph_encoder.pt" \
    --output_dir "$output_dir" \
    --class_names "ecDNA,chromothripsis,BFB,seismic_amplification" \
    --embedding_normalization none \
    --embedding_features full \
    --tabular_features safe \
    --unlabeled_candidates \
    --embeddings_only \
    "$@"
