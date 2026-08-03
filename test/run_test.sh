#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-quick}"
path_config="${2:-$repo_dir/test/paths.hpc.env}"

if [[ "$mode" != "quick" && "$mode" != "full" ]]; then
    echo "Usage: $0 [quick|full] [PATH_CONFIG.env]" >&2
    exit 2
fi
if [[ ! -f "$path_config" ]]; then
    echo "Path configuration not found: $path_config" >&2
    exit 2
fi

# shellcheck source=paths.hpc.env
source "$path_config"
: "${SAMPLE_ID:?SAMPLE_ID is required}"
: "${WAKHAN_ROOT:?WAKHAN_ROOT is required}"
: "${SEVERUS_VCF:?SEVERUS_VCF is required}"
: "${LABEL_SOURCE_DIR:?LABEL_SOURCE_DIR is required}"
: "${TEST_OUTPUT_DIR:?TEST_OUTPUT_DIR is required}"

python_bin="${PYTHON_BIN:-python}"
mkdir -p "$TEST_OUTPUT_DIR"
manifest="$TEST_OUTPUT_DIR/manifest.tsv"
printf 'sample_id\twakhan_root\tseverus_vcf\n%s\t%s\t%s\n' \
    "$SAMPLE_ID" "$WAKHAN_ROOT" "$SEVERUS_VCF" > "$manifest"

echo "[1/3] Checking the installation and packaged checkpoints"
"$python_bin" "$repo_dir/workflow/00_check_install.py"

echo "[2/3] Rebuilding normalized labels"
"$python_bin" "$repo_dir/label_generator/generate_labels.py" \
    --label-dir "$LABEL_SOURCE_DIR" \
    --output-dir "$TEST_OUTPUT_DIR/labels"

echo "[3/3] Generating candidates for $SAMPLE_ID"
"$repo_dir/candidate_generator/run.sh" \
    "$manifest" \
    "$TEST_OUTPUT_DIR/candidates" \
    --profile "${TEST_PROFILE:-balanced}" \
    --max_candidates_per_sample "${TEST_MAX_CANDIDATES:-25}"

if [[ "$mode" == "quick" ]]; then
    echo "Quick test passed. Outputs: $TEST_OUTPUT_DIR"
    exit 0
fi

echo "[4/7] Embedding localized candidates"
"$repo_dir/workflow/04_embed_candidates.sh" \
    "$manifest" \
    "$TEST_OUTPUT_DIR/candidates/merged_candidate_regions.csv" \
    "$TEST_OUTPUT_DIR/localization_features" \
    --device "${TEST_DEVICE:-cpu}"

echo "[5/7] Applying the localization ensemble"
"$python_bin" "$repo_dir/workflow/06_predict_localization_ensemble.py" \
    --candidates "$TEST_OUTPUT_DIR/candidates/merged_candidate_regions.csv" \
    --embedding-bundle "$TEST_OUTPUT_DIR/localization_features/embeddings.npz" \
    --selected-embeddings "$TEST_OUTPUT_DIR/localization_features/selected_embedding_features.npz" \
    --tabular-features "$TEST_OUTPUT_DIR/localization_features/tabular_features.npz" \
    --output "$TEST_OUTPUT_DIR/localization_predictions" \
    --device "${TEST_DEVICE:-cpu}"

echo "[6/7] Embedding chromosomes"
"$repo_dir/workflow/07_embed_chromosomes.sh" \
    "$manifest" \
    "$TEST_OUTPUT_DIR/chromosome_features" \
    --device "${TEST_DEVICE:-cpu}"

echo "[7/7] Applying the chromosome ensemble"
"$python_bin" "$repo_dir/workflow/09_predict_chromosome_ensemble.py" \
    --embedding-dir "$TEST_OUTPUT_DIR/chromosome_features/embeddings" \
    --tabular "$TEST_OUTPUT_DIR/chromosome_features/chromosome_tabular.tsv" \
    --output "$TEST_OUTPUT_DIR/chromosome_predictions" \
    --device "${TEST_DEVICE:-cpu}"

echo "Full end-to-end test passed. Outputs: $TEST_OUTPUT_DIR"
