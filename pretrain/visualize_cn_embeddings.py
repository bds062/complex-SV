"""
Visualise pretrained CN encoder bp-window embeddings.

Recommended pipeline position
-----------------------------
Run after Phase 1 CN bp-window pretraining:

    python -m complex_sv.pretrain.visualize_cn_embeddings \
        --input wakhan_outputs/*.vcf \
        --checkpoint results/cn_encoder.pt \
        --output_dir results/cn_embedding_qc \
        --window_bp_sizes 50000 100000 250000 500000 1000000 \
        --windows_per_chrom_per_size 40

Outputs:
    cn_embedding_plot.png
    cn_embedding_facets.png
    cn_cluster_summary.tsv
    cn_embedding_metrics.npz
    cn_silhouette_sweep.png  optional
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset

try:
    from config import CNEncoderConfig
    from data.cn_resampler import CN_CHANNELS, build_bp_window_tensors
    from data.severus_parser import CHROM_ORDER
    from data.wakhan_parser import parse_all_wakhan
    from pretrain.cn_encoder import CNMaskedAutoencoder
    from utils import get_device, l2_normalize, setup_logging
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import CNEncoderConfig
    from data.cn_resampler import CN_CHANNELS, build_bp_window_tensors
    from data.severus_parser import CHROM_ORDER
    from data.wakhan_parser import parse_all_wakhan
    from pretrain.cn_encoder import CNMaskedAutoencoder
    from utils import get_device, l2_normalize, setup_logging


log = logging.getLogger(__name__)

_PALETTE_20 = [
    "#E63946",
    "#457B9D",
    "#2A9D8F",
    "#E9C46A",
    "#F4A261",
    "#6A4C93",
    "#1982C4",
    "#8AC926",
    "#FF595E",
    "#6A994E",
    "#3A86FF",
    "#FB5607",
    "#FFBE0B",
    "#8338EC",
    "#06D6A0",
    "#EF476F",
    "#118AB2",
    "#FFD166",
    "#073B4C",
    "#B5838D",
]


def _resolve_inputs(
    input_patterns: list[str] | None,
    input_list: str | None,
) -> list[str]:
    paths: list[str] = []

    if input_patterns:
        for pattern in input_patterns:
            expanded = glob.glob(pattern)
            paths.extend(expanded if expanded else [pattern])

    if input_list:
        list_path = Path(input_list)
        if not list_path.exists():
            raise FileNotFoundError(f"--input_list not found: {list_path}")

        with list_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                item = line.strip()
                if item and not item.startswith("#"):
                    paths.append(item)

    unique = list(dict.fromkeys(paths))
    valid = [p for p in unique if Path(p).exists()]

    missing = [p for p in unique if not Path(p).exists()]
    if missing:
        log.warning("%d input path(s) not found; examples: %s", len(missing), missing[:5])

    if not valid:
        raise FileNotFoundError("No valid Wakhan input files found")

    return valid


def _cfg_from_checkpoint(ckpt: dict) -> CNEncoderConfig:
    raw = ckpt.get("config", {})

    if "cn_encoder" in raw:
        raw = raw["cn_encoder"]

    values = {
        "d_model": raw.get("d_model", 256),
        "n_heads": raw.get("n_heads", 8),
        "n_layers": raw.get("n_layers", 6),
        "ff_dim": raw.get("ff_dim", raw.get("d_ff", 1024)),
        "dropout": raw.get("dropout", 0.1),
        "n_bins_arm": raw.get("n_bins_arm", raw.get("seq_len", 256)),
        "n_bins_region": raw.get("n_bins_region", raw.get("seq_len", 128)),
        "mask_prob": raw.get("mask_prob", 0.15),
    }

    return CNEncoderConfig(**values)


def _training_config_from_checkpoint(ckpt: dict) -> dict:
    raw = ckpt.get("config", {})
    return raw.get("training", {}) if isinstance(raw, dict) else {}


def _make_encoder_cfg(base: CNEncoderConfig) -> SimpleNamespace:
    """
    Match pretrain_cn.py compatibility wrapper.

    This keeps visualization compatible with CNMaskedAutoencoder implementations
    that expect either the project config names or the older prototype names.
    """
    return SimpleNamespace(
        d_model=base.d_model,
        n_heads=base.n_heads,
        n_layers=base.n_layers,
        ff_dim=base.ff_dim,
        d_ff=base.ff_dim,
        dropout=base.dropout,
        n_bins_arm=base.n_bins_arm,
        n_bins_region=base.n_bins_region,
        seq_len=base.n_bins_region,
        mask_prob=base.mask_prob,
        embed_dim=base.d_model,
    )


def _forward_cn(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    out = model(x, mask)

    if not isinstance(out, tuple):
        raise TypeError("CNMaskedAutoencoder.forward must return a tuple")

    if len(out) == 3:
        recon, cls_emb, bin_embs = out
    elif len(out) == 2:
        recon, cls_emb = out
        bin_embs = None
    else:
        raise ValueError(f"Unexpected CNMaskedAutoencoder output length: {len(out)}")

    return recon, cls_emb, bin_embs


class WindowTensorDataset(Dataset):
    def __init__(self, tensors: list[torch.Tensor]) -> None:
        self.tensors = [x.to(dtype=torch.float32) for x in tensors]

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tensors[idx]


def build_window_dataset(
    df: pd.DataFrame,
    window_bp_sizes: list[int],
    n_bins: int,
    windows_per_chrom_per_size: int,
    min_covered_fraction: float,
    seed: int,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    rng = np.random.default_rng(seed)

    tensors, meta = build_bp_window_tensors(
        df,
        window_bp_sizes=window_bp_sizes,
        n_bins=n_bins,
        windows_per_chrom_per_size=windows_per_chrom_per_size,
        min_covered_fraction=min_covered_fraction,
        rng=rng,
    )

    if not tensors:
        raise RuntimeError(
            "No bp-window tensors were generated from Wakhan inputs. "
            "Try smaller --window_bp_sizes or lower --min_covered_fraction."
        )

    # Add summary features computed from the actual resampled model input.
    cn_total_idx = CN_CHANNELS.index("cn_total")
    loh_idx = CN_CHANNELS.index("loh")
    ai_idx = CN_CHANNELS.index("allele_imbalance")

    mean_cn_total: list[float] = []
    mean_allele_imbalance: list[float] = []
    loh_fraction: list[float] = []

    for tensor in tensors:
        arr = tensor.detach().cpu().numpy()
        mean_cn_total.append(float(arr[:, cn_total_idx].mean()))
        mean_allele_imbalance.append(float(arr[:, ai_idx].mean()))
        loh_fraction.append(float(arr[:, loh_idx].mean()))

    meta = meta.copy()
    meta["mean_cn_total"] = mean_cn_total
    meta["mean_allele_imbalance"] = mean_allele_imbalance
    meta["loh_fraction"] = loh_fraction

    log.info(
        "Built %d bp-window tensors across %d sample(s).",
        len(tensors),
        meta["sample_id"].nunique(),
    )

    return tensors, meta


def reduce_2d(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    if method == "umap":
        try:
            import umap

            return umap.UMAP(
                n_components=2,
                n_neighbors=min(15, max(2, len(embeddings) - 1)),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            ).fit_transform(embeddings)
        except ImportError:
            log.warning("umap-learn not installed; falling back to t-SNE")
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, max(2, len(embeddings) // 10))
        return TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
        ).fit_transform(embeddings)

    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(embeddings)

    raise ValueError(f"Unknown reduction method: {method}")


def _color_map(labels: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(labels))
    return {
        label: _PALETTE_20[i % len(_PALETTE_20)]
        for i, label in enumerate(unique)
    }


def _scatter_categorical(
    ax,
    xy: np.ndarray,
    labels: list[str],
    title: str,
    legend_title: str,
    max_legend_items: int = 24,
) -> None:
    colors = _color_map(labels)

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=[colors[x] for x in labels],
        s=18,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )

    if len(colors) <= max_legend_items:
        handles = [
            mpatches.Patch(color=color, label=str(label))
            for label, color in colors.items()
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            fontsize=7,
            title_fontsize=8,
            loc="best",
            framealpha=0.85,
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _scatter_categorical_ordered(
    ax,
    xy: np.ndarray,
    labels: list[str],
    order: list[str],
    title: str,
    legend_title: str,
    max_legend_items: int = 24,
) -> None:
    cmap = {
        label: _PALETTE_20[j % len(_PALETTE_20)]
        for j, label in enumerate(order)
    }

    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=[cmap[x] for x in labels],
        s=16,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )

    if len(order) <= max_legend_items:
        handles = [
            mpatches.Patch(color=cmap[label], label=label)
            for label in order
        ]
        ax.legend(
            handles=handles,
            title=legend_title,
            fontsize=7,
            title_fontsize=8,
            loc="best",
            framealpha=0.85,
        )

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _scatter_continuous(
    ax,
    xy: np.ndarray,
    values: np.ndarray,
    title: str,
    label: str,
) -> None:
    sc = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        s=18,
        alpha=0.75,
        linewidths=0,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=label)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _plot_silhouette_sweep(
    embeddings: np.ndarray,
    k_min: int,
    k_max: int,
    out_dir: Path,
) -> None:
    ks: list[int] = []
    scores: list[float] = []

    max_k = min(k_max, len(embeddings) - 1)

    for k in range(k_min, max_k + 1):
        labels = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20,
        ).fit_predict(embeddings)

        ks.append(k)
        scores.append(silhouette_score(embeddings, labels))

    if not ks:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, scores, "-o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title("CN bp-window embedding silhouette sweep")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cn_silhouette_sweep.png", dpi=150)
    plt.close(fig)


def _load_model_state_dict(ckpt: dict) -> dict:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    raise KeyError("Checkpoint does not contain model_state_dict or state_dict")


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = _cfg_from_checkpoint(ckpt)
    training_cfg = _training_config_from_checkpoint(ckpt)

    if args.n_bins is not None:
        cfg.n_bins_region = args.n_bins

    if args.window_bp_sizes is None:
        args.window_bp_sizes = training_cfg.get(
            "window_bp_sizes",
            [50_000, 100_000, 250_000, 500_000, 1_000_000],
        )

    if args.windows_per_chrom_per_size is None:
        args.windows_per_chrom_per_size = int(
            training_cfg.get("windows_per_chrom_per_size", 40)
        )

    if args.min_covered_fraction is None:
        args.min_covered_fraction = float(
            training_cfg.get("min_covered_fraction", 0.10)
        )

    device = get_device()
    model = CNMaskedAutoencoder(_make_encoder_cfg(cfg)).to(device)

    state_dict = _load_model_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=args.strict)
    model.eval()

    inputs = _resolve_inputs(args.input, args.input_list)
    df = parse_all_wakhan(inputs)

    tensors, meta = build_window_dataset(
        df,
        window_bp_sizes=[int(x) for x in args.window_bp_sizes],
        n_bins=cfg.n_bins_region,
        windows_per_chrom_per_size=int(args.windows_per_chrom_per_size),
        min_covered_fraction=float(args.min_covered_fraction),
        seed=args.seed,
    )

    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for x in DataLoader(
            WindowTensorDataset(tensors),
            batch_size=args.batch_size,
            shuffle=False,
        ):
            x = x.to(device=device, dtype=torch.float32)
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)

            _recon, cls_emb, _bin_embs = _forward_cn(model, x, mask)
            cls_emb = l2_normalize(cls_emb, dim=-1)
            embeddings.append(cls_emb.cpu().numpy())

    emb = np.concatenate(embeddings, axis=0)

    if len(emb) < 3:
        raise RuntimeError("Need at least 3 embeddings for visualization/clustering")

    xy = reduce_2d(emb, method=args.reduction)

    if args.n_clusters >= len(emb):
        raise ValueError("--n_clusters must be smaller than the number of embeddings")

    cluster = KMeans(
        n_clusters=args.n_clusters,
        random_state=42,
        n_init=20,
    ).fit_predict(emb)

    meta["cluster"] = cluster

    sil = silhouette_score(emb, cluster) if args.n_clusters > 1 else float("nan")

    meta.to_csv(out_dir / "cn_cluster_summary.tsv", sep="\t", index=False)

    reserved_npz_keys = {"embeddings", "xy", "cluster", "silhouette"}

    meta_arrays = {
        c: meta[c].values.astype(str)
        if meta[c].dtype == object
        else meta[c].values
        for c in meta.columns
        if c not in reserved_npz_keys
    }

    np.savez(
        out_dir / "cn_embedding_metrics.npz",
        embeddings=emb,
        xy=xy,
        cluster=cluster,
        silhouette=np.array([sil]),
        **meta_arrays,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_categorical(
        ax,
        xy,
        [str(x) for x in cluster],
        f"CN bp-window embeddings ({args.reduction.upper()}); silhouette={sil:.3f}",
        "Cluster",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "cn_embedding_plot.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_categorical(
        ax1,
        xy,
        meta["cluster"].astype(str).tolist(),
        "Cluster",
        "Cluster",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_categorical(
        ax2,
        xy,
        meta["sample_id"].astype(str).tolist(),
        "Sample ID",
        "Sample",
    )

    ax3 = fig.add_subplot(gs[1, 0])
    chrom_labels = meta["chrom"].astype(str).tolist()
    chrom_order = sorted(
        set(chrom_labels),
        key=lambda c: CHROM_ORDER.get(c, CHROM_ORDER.get(c.removeprefix("chr"), 99)),
    )
    _scatter_categorical_ordered(
        ax3,
        xy,
        chrom_labels,
        chrom_order,
        "Chromosome",
        "Chrom",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    if "requested_window_bp_size" in meta.columns:
        size_col = "requested_window_bp_size"
    else:
        size_col = "window_bp_size"

    window_labels = meta[size_col].astype(int).astype(str).tolist()
    window_order = [
        str(x)
        for x in sorted(meta[size_col].astype(int).unique().tolist())
    ]
    _scatter_categorical_ordered(
        ax4,
        xy,
        window_labels,
        window_order,
        "Window bp size",
        "bp",
    )

    fig.suptitle("CN encoder bp-window embedding QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "cn_embedding_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_continuous(
        ax1,
        xy,
        meta["mean_cn_total"].values,
        "Mean total CN",
        "Mean total CN",
    )

    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_continuous(
        ax2,
        xy,
        meta["mean_allele_imbalance"].values,
        "Mean allele imbalance",
        "Mean allele imbalance",
    )

    ax3 = fig.add_subplot(gs[1, 0])
    _scatter_continuous(
        ax3,
        xy,
        meta["loh_fraction"].values,
        "LOH fraction",
        "LOH fraction",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    _scatter_continuous(
        ax4,
        xy,
        meta["covered_fraction"].values,
        "Window covered fraction",
        "Covered fraction",
    )

    fig.suptitle("CN encoder bp-window continuous QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "cn_embedding_continuous_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.sweep_clusters:
        _plot_silhouette_sweep(
            emb,
            args.sweep_k_min,
            args.sweep_k_max,
            out_dir,
        )

    log.info("Saved CN bp-window embedding visualisation outputs to %s", out_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise pretrained CN encoder bp-window embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--input", nargs="+", metavar="WAKHAN")
    parser.add_argument("--input_list", metavar="FILE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=".")

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument(
        "--n_bins",
        type=int,
        default=None,
        help="Override checkpoint n_bins_region for bp-window resampling.",
    )

    parser.add_argument(
        "--window_bp_sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Fixed genomic bp window sizes to visualize. "
            "Defaults to checkpoint training config if available."
        ),
    )
    parser.add_argument(
        "--windows_per_chrom_per_size",
        type=int,
        default=None,
        help=(
            "Random bp windows sampled per sample/chromosome/window size. "
            "Defaults to checkpoint training config if available."
        ),
    )
    parser.add_argument(
        "--min_covered_fraction",
        type=float,
        default=None,
        help=(
            "Minimum fraction of a bp window overlapped by observed CN segments. "
            "Defaults to checkpoint training config if available."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument(
        "--reduction",
        choices=["umap", "tsne", "pca"],
        default="umap",
    )
    parser.add_argument("--sweep_clusters", action="store_true")
    parser.add_argument("--sweep_k_min", type=int, default=2)
    parser.add_argument("--sweep_k_max", type=int, default=15)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict checkpoint loading.",
    )

    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())