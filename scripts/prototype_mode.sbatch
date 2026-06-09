#!/usr/bin/env bash
#SBATCH --job-name=complex_sv_proto
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=../logs/complex_sv_proto-%j.out
#SBATCH --error=../logs/complex_sv_proto-%j.out

set -euo pipefail

# Usage:
#   1. Edit the values in the block below.
#   2. Submit with:
#        sbatch scripts/prototype_mode.sbatch
#
# This script runs both prototype-mode stages:
#   A. Build chromothripsis prototypes from labeled anchor regions.
#   B. Run prototype inference/ranking on anchors + proposed candidates.

# -------------------------
# User-editable parameters
# -------------------------
PROJECT_DIR="./"

MANIFEST="../results/pipeline3/complex_sv_manifest.tsv"
LABELS="../results/pipeline3/complex_sv_labels.tsv"
OUTPUT_ROOT="../results/pipeline3/prototype_mode"

# Global pretrained checkpoint parameters.
CN_CHECKPOINT="../results/pipeline3/cn_pretrain_chrom/cn_encoder.pt"
GRAPH_CHECKPOINT="../results/pipeline3/sv_pretrain_chrom/graph_encoder.pt"

PYTHON_BIN="${PYTHON_BIN:-python}"
TAU="0.5"

PROTOTYPE_DIR="${OUTPUT_ROOT%/}/anchors"
INFER_DIR="${OUTPUT_ROOT%/}/inference"
PROTOTYPES="${PROTOTYPE_DIR%/}/prototypes.pt"

# Add optional prototype-building arguments here, one argument per line.
PROTOTYPE_EXTRA_ARGS=(
    # --strict
)

# Add optional inference arguments here, one argument per line.
INFER_EXTRA_ARGS=(
    # --strict
)

cd "$PROJECT_DIR"

if [[ "$PYTHON_BIN" == "python" && -x "../envs/env2/bin/python" ]]; then
    PYTHON_BIN="../envs/env2/bin/python"
fi

if ! "$PYTHON_BIN" -c "import pandas, torch, torch_geometric" >/dev/null 2>&1; then
    echo "[prototype_mode] ERROR: $PYTHON_BIN cannot import pandas, torch, and torch_geometric. Activate env2 or set PYTHON_BIN." >&2
    exit 1
fi

mkdir -p "$PROTOTYPE_DIR" "$INFER_DIR" "../logs"

for required_file in "$MANIFEST" "$LABELS" "$CN_CHECKPOINT" "$GRAPH_CHECKPOINT"; do
    if [[ ! -f "$required_file" ]]; then
        echo "[prototype_mode] ERROR: required file not found: $required_file" >&2
        exit 1
    fi
done

echo "[prototype_mode] project_dir=$PROJECT_DIR"
echo "[prototype_mode] manifest=$MANIFEST"
echo "[prototype_mode] labels=$LABELS"
echo "[prototype_mode] cn_checkpoint=$CN_CHECKPOINT"
echo "[prototype_mode] graph_checkpoint=$GRAPH_CHECKPOINT"
echo "[prototype_mode] output_root=$OUTPUT_ROOT"
echo "[prototype_mode] prototype_dir=$PROTOTYPE_DIR"
echo "[prototype_mode] infer_dir=$INFER_DIR"
echo "[prototype_mode] prototypes=$PROTOTYPES"
echo "[prototype_mode] python_bin=$PYTHON_BIN"
echo "[prototype_mode] tau=$TAU"
echo "[prototype_mode] prototype_extra_args=${PROTOTYPE_EXTRA_ARGS[*]:-}"
echo "[prototype_mode] infer_extra_args=${INFER_EXTRA_ARGS[*]:-}"

PROTO_CMD=(
    "$PYTHON_BIN" discovery/embed_corpus.py
    --manifest "$MANIFEST"
    --labels "$LABELS"
    --cn_checkpoint "$CN_CHECKPOINT"
    --graph_checkpoint "$GRAPH_CHECKPOINT"
    --output_dir "$PROTOTYPE_DIR"
    --candidate_source labels
    --tau "$TAU"
)
PROTO_CMD+=("${PROTOTYPE_EXTRA_ARGS[@]}")

echo "[prototype_mode] Building anchor prototypes:"
printf '  %q' "${PROTO_CMD[@]}"
printf '\n'
"${PROTO_CMD[@]}"

if [[ ! -f "$PROTOTYPES" ]]; then
    echo "[prototype_mode] ERROR: prototype cache not found after anchor stage: $PROTOTYPES" >&2
    exit 1
fi

INFER_CMD=(
    "$PYTHON_BIN" infer.py
    --manifest "$MANIFEST"
    --labels "$LABELS"
    --cn_checkpoint "$CN_CHECKPOINT"
    --graph_checkpoint "$GRAPH_CHECKPOINT"
    --prototypes "$PROTOTYPES"
    --output_dir "$INFER_DIR"
    --candidate_source all
    --tau "$TAU"
)
INFER_CMD+=("${INFER_EXTRA_ARGS[@]}")

echo "[prototype_mode] Running prototype inference:"
printf '  %q' "${INFER_CMD[@]}"
printf '\n'
"${INFER_CMD[@]}"

for expected_output in \
    "$PROTOTYPE_DIR/candidate_embeddings.tsv" \
    "$PROTOTYPE_DIR/anchor_leave_one_out.tsv" \
    "$INFER_DIR/predictions.tsv" \
    "$INFER_DIR/prototype_distances.tsv" \
    "$INFER_DIR/embeddings.npz"; do
    if [[ ! -f "$expected_output" ]]; then
        echo "[prototype_mode] ERROR: expected output not found: $expected_output" >&2
        exit 1
    fi
done

echo "[prototype_mode] Done."
