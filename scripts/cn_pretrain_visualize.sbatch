#!/usr/bin/env bash
#SBATCH --job-name=cn_pretrain_viz
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=../logs/cn_pretrain_viz-%j.out
#SBATCH --error=../logs/cn_pretrain_viz-%j.out

set -euo pipefail


# Usage:
#   1. Edit the values in the block below.
#   2. Submit with:
#        sbatch scripts/cn_pretrain_visualize.sbatch
#
# Slurm resources can still be changed by editing the #SBATCH lines above.

# -------------------------
# User-editable parameters
# -------------------------
PROJECT_DIR="./"

OUTPUT_DIR="../results/pipeline3/cn_pretrain_chrom"
INPUT_LIST="${OUTPUT_DIR}/wakhan_list.txt"
HIGHLIGHT_LIST="${OUTPUT_DIR}/highlight_list.txt"
CHECKPOINT="${OUTPUT_DIR%/}/cn_encoder.pt"

# Add optional training arguments here, one argument per line.
# Examples:
#   --epochs 100
#   --max_windows 50000
TRAIN_EXTRA_ARGS=(
    --window_bp_sizes 1000000 2500000 5000000 10000000 25000000 50000000 100000000
    --n_bins_region 256   
    --windows_per_chrom_per_size 20
    --batch_size 16
    --max_windows 100000
)

# Add optional visualization arguments here, one argument per line.
# Examples:
#   --reduction pca
#   --n_clusters 8
VIZ_EXTRA_ARGS=(
    --window_bp_sizes 1000000 2500000 5000000 10000000 25000000 50000000 100000000
    --n_bins 256
    --windows_per_chrom_per_size 20
    --max_windows 100000
)

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$INPUT_LIST" ]]; then
    echo "[cn_pretrain_visualize] ERROR: input list not found: $INPUT_LIST" >&2
    exit 1
fi

if [[ -n "$HIGHLIGHT_LIST" && ! -f "$HIGHLIGHT_LIST" ]]; then
    echo "[cn_pretrain_visualize] ERROR: highlight list not found: $HIGHLIGHT_LIST" >&2
    exit 1
fi

echo "[cn_pretrain_visualize] project_dir=$PROJECT_DIR"
echo "[cn_pretrain_visualize] input_list=$INPUT_LIST"
echo "[cn_pretrain_visualize] output_dir=$OUTPUT_DIR"
echo "[cn_pretrain_visualize] checkpoint=$CHECKPOINT"
echo "[cn_pretrain_visualize] highlight_list=$HIGHLIGHT_LIST"
echo "[cn_pretrain_visualize] train_extra_args=${TRAIN_EXTRA_ARGS[*]:-}"
echo "[cn_pretrain_visualize] viz_extra_args=${VIZ_EXTRA_ARGS[*]:-}"

# source ~/.bashrc
# conda activate env2

TRAIN_CMD=(
    python pretrain/pretrain_cn.py
    --input_list "$INPUT_LIST"
    --output_dir "$OUTPUT_DIR"
)
TRAIN_CMD+=("${TRAIN_EXTRA_ARGS[@]}")

echo "[cn_pretrain_visualize] Running training:"
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'
"${TRAIN_CMD[@]}"

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "[cn_pretrain_visualize] ERROR: checkpoint not found after training: $CHECKPOINT" >&2
    exit 1
fi

VIZ_CMD=(
    python pretrain/visualize_cn_embeddings.py
    --input_list "$INPUT_LIST"
    --checkpoint "$CHECKPOINT"
    --output_dir "$OUTPUT_DIR"
)

if [[ -n "$HIGHLIGHT_LIST" ]]; then
    VIZ_CMD+=(--highlight_list "$HIGHLIGHT_LIST")
fi
VIZ_CMD+=("${VIZ_EXTRA_ARGS[@]}")

echo "[cn_pretrain_visualize] Running visualization:"
printf '  %q' "${VIZ_CMD[@]}"
printf '\n'
"${VIZ_CMD[@]}"

echo "[cn_pretrain_visualize] Done."

