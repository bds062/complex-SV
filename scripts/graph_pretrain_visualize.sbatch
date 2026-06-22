#!/usr/bin/env bash
#SBATCH --job-name=graph_pretrain_viz
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4-00:00:00
#SBATCH --output=../logs/graph_pretrain_viz-%j.out
#SBATCH --error=../logs/graph_pretrain_viz-%j.out

set -euo pipefail

# Usage:
#   1. Edit the values in the block below.
#   2. Submit with:
#        sbatch scripts/graph_pretrain_visualize.sbatch
#
# Slurm resources can still be changed by editing the #SBATCH lines above.

# -------------------------
# User-editable parameters
# -------------------------
PROJECT_DIR="./"

OUTPUT_DIR="../results/pipeline3/sv3"
INPUT_LIST="${OUTPUT_DIR}/severus_list.txt"
HIGHLIGHT_LIST="${OUTPUT_DIR}/highlight_list.txt"
CHECKPOINT="${OUTPUT_DIR%/}/graph_encoder.pt"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Add optional training arguments here, one argument per line.
TRAIN_EXTRA_ARGS=(
    --window_bp_sizes 1000000 2500000 5000000 10000000 25000000 50000000 100000000
    # --windows_per_chrom_per_size 10
    # --cluster_windows_per_chrom_per_size 10
    --min_sv_per_window 2
    --batch_size 16
    --max_windows 100000
)

# Add optional visualization arguments here, one argument per line.
VIZ_EXTRA_ARGS=(
    --window_bp_sizes 1000000 2500000 5000000 10000000 25000000 50000000 100000000
    # --windows_per_chrom_per_size 10
    # --cluster_windows_per_chrom_per_size 10
    --min_sv_per_window 1
    --max_windows 100000
)

cd "$PROJECT_DIR"

CONDA_SH="/data/KolmogorovLab/srinivasanbd/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "[graph_pretrain_visualize] ERROR: conda shell hook not found: $CONDA_SH" >&2
    exit 1
fi
source "$CONDA_SH"
conda activate env2

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

if [[ "$PYTHON_BIN" == "python" ]]; then
    PYTHON_BIN="$(which python)"
fi

if ! "$PYTHON_BIN" -c "import torch_geometric" >/dev/null 2>&1; then
    echo "[graph_pretrain_visualize] ERROR: $PYTHON_BIN cannot import torch_geometric after activating env2." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_LIST" ]]; then
    echo "[graph_pretrain_visualize] ERROR: input list not found: $INPUT_LIST" >&2
    exit 1
fi

if [[ -n "$HIGHLIGHT_LIST" && ! -f "$HIGHLIGHT_LIST" ]]; then
    echo "[graph_pretrain_visualize] WARNING: highlight list not found, continuing without it: $HIGHLIGHT_LIST" >&2
    HIGHLIGHT_LIST=""
fi

echo "[graph_pretrain_visualize] project_dir=$PROJECT_DIR"
echo "[graph_pretrain_visualize] input_list=$INPUT_LIST"
echo "[graph_pretrain_visualize] output_dir=$OUTPUT_DIR"
echo "[graph_pretrain_visualize] checkpoint=$CHECKPOINT"
echo "[graph_pretrain_visualize] python_bin=$PYTHON_BIN"
echo "[graph_pretrain_visualize] highlight_list=$HIGHLIGHT_LIST"
echo "[graph_pretrain_visualize] train_extra_args=${TRAIN_EXTRA_ARGS[*]:-}"
echo "[graph_pretrain_visualize] viz_extra_args=${VIZ_EXTRA_ARGS[*]:-}"

TRAIN_CMD=(
    "$PYTHON_BIN" pretrain/pretrain_graph.py
    --input_list "$INPUT_LIST"
    --output_dir "$OUTPUT_DIR"
)
TRAIN_CMD+=("${TRAIN_EXTRA_ARGS[@]}")

echo "[graph_pretrain_visualize] Running training:"
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[graph_pretrain_visualize] ERROR: checkpoint not found after training: $CHECKPOINT" >&2
    exit 1
fi

VIZ_CMD=(
    "$PYTHON_BIN" pretrain/visualize_graph_embeddings.py
    --input_list "$INPUT_LIST"
    --checkpoint "$CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$HIGHLIGHT_LIST" ]]; then
    VIZ_CMD+=(--highlight_list "$HIGHLIGHT_LIST")
fi
VIZ_CMD+=("${VIZ_EXTRA_ARGS[@]}")

echo "[graph_pretrain_visualize] Running visualization:"
printf '  %q' "${VIZ_CMD[@]}"
printf '\n'
"${VIZ_CMD[@]}"

echo "[graph_pretrain_visualize] Done."
