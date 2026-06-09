#!/usr/bin/env bash
set -euo pipefail

# Plain bash launcher for prototype-mode complex-SV inference.
# Run from anywhere with:
#   bash complex-SV/scripts/prototype_mode.sh
# or from the repo with:
#   bash scripts/prototype_mode.sh
#
# Common overrides:
#   OUTPUT_ROOT=../results/pipeline3/prototype_chrom \
#   INFER_CANDIDATE_SOURCE=chromosomes \
#   bash scripts/prototype_mode.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

BASE="${BASE:-../results/pipeline2}"
MANIFEST="${MANIFEST:-$BASE/complex_sv_manifest.tsv}"
LABELS="${LABELS:-$BASE/complex_sv_labels.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASE/prototype_chrom}"

CN_CHECKPOINT="${CN_CHECKPOINT:-$BASE/cn_pretrain_chrom/cn_encoder.pt}"
# GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-$BASE/sv_pretrain_chrom/graph_encoder.pt}"
GRAPH_CHECKPOINT="${GRAPH_CHECKPOINT:-../results/pipeline2/sv_pretrain_chrom/graph_encoder.pt}"

PYTHON_BIN="${PYTHON_BIN:-python}"
TAU="${TAU:-0.06819}"
INFER_CANDIDATE_SOURCE="${INFER_CANDIDATE_SOURCE:-chromosomes}"

PROTOTYPE_DIR="${OUTPUT_ROOT%/}/anchors"
INFER_DIR="${OUTPUT_ROOT%/}/inference"
PLOT_DIR="${INFER_DIR%/}/predicted_chromosome_plots"
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
echo "[prototype_mode] plot_dir=$PLOT_DIR"
echo "[prototype_mode] prototypes=$PROTOTYPES"
echo "[prototype_mode] python_bin=$PYTHON_BIN"
echo "[prototype_mode] tau=$TAU"
echo "[prototype_mode] infer_candidate_source=$INFER_CANDIDATE_SOURCE"
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
    --candidate_source "$INFER_CANDIDATE_SOURCE"
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

PLOT_CMD=(
    "$PYTHON_BIN" discovery/plot_predicted_chromosomes.py
    --manifest "$MANIFEST"
    --prototype_distances "$INFER_DIR/prototype_distances.tsv"
    --output_dir "$PLOT_DIR"
)

echo "[prototype_mode] Plotting predicted unlabeled chromosomes:"
printf '  %q' "${PLOT_CMD[@]}"
printf '\n'
"${PLOT_CMD[@]}"

if [[ ! -f "$PLOT_DIR/selected_predictions.tsv" ]]; then
    echo "[prototype_mode] ERROR: expected output not found: $PLOT_DIR/selected_predictions.tsv" >&2
    exit 1
fi

echo "[prototype_mode] Done."
