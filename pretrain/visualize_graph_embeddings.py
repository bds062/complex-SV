"""
Visualise pretrained Severus graph encoder regional embeddings.

Recommended pipeline position
-----------------------------
Run after Phase 2 pretraining:

    python -m complex_sv.pretrain.visualize_graph_embeddings \
        --input severus_outputs/*.vcf \
        --checkpoint results/graph_encoder.pt \
        --output_dir results/graph_embedding_qc

Outputs:
    graph_embedding_plot.png
    graph_embedding_facets.png
    graph_cluster_summary.tsv
    graph_embedding_metrics.npz
    graph_silhouette_sweep.png  (optional)
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
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
from sklearn.preprocessing import RobustScaler

try:
    from config import GraphEncoderConfig
    from data.graph_builder import build_all_graphs
    from data.severus_parser import CHROM_ORDER, CONTINUOUS_COLS, N_CONT, build_node_features, parse_all_severus
    from pretrain.graph_encoder import SVGraphMAE
    from utils import get_device, l2_normalize, setup_logging
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from config import GraphEncoderConfig
    from data.graph_builder import build_all_graphs
    from data.severus_parser import CHROM_ORDER, CONTINUOUS_COLS, N_CONT, build_node_features, parse_all_severus
    from pretrain.graph_encoder import SVGraphMAE
    from utils import get_device, l2_normalize, setup_logging

log = logging.getLogger(__name__)

_PALETTE_20 = [
    "#E63946", "#457B9D", "#2A9D8F", "#E9C46A", "#F4A261",
    "#6A4C93", "#1982C4", "#8AC926", "#FF595E", "#6A994E",
    "#3A86FF", "#FB5607", "#FFBE0B", "#8338EC", "#06D6A0",
    "#EF476F", "#118AB2", "#FFD166", "#073B4C", "#B5838D",
]

SV_COLOURS = {
    "DEL": "#E63946",
    "INS": "#2A9D8F",
    "DUP": "#457B9D",
    "BND": "#E9C46A",
    "sBND": "#F4A261",
    "INV": "#6A4C93",
}


def _resolve_inputs(input_patterns: list[str] | None, input_list: str | None) -> list[str]:
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
    if not valid:
        raise FileNotFoundError("No valid Severus VCF input files found")
    return valid


def _cfg_from_checkpoint(ckpt: dict) -> GraphEncoderConfig:
    raw = ckpt.get("config", {})
    if "graph_encoder" in raw:
        raw = raw["graph_encoder"]
    return GraphEncoderConfig(
        d_model=raw.get("d_model", 128),
        n_heads=raw.get("n_heads", 8),
        n_layers=raw.get("n_layers", 4),
        embed_dim=raw.get("embed_dim", 64),
        dropout=raw.get("dropout", 0.1),
        proximity_bp=raw.get("proximity_bp", 1_000_000),
        mask_prob=raw.get("mask_prob", 0.15),
    )


def _scaler_from_checkpoint(ckpt: dict) -> RobustScaler | None:
    if "scaler_center" not in ckpt or "scaler_scale" not in ckpt:
        return None
    scaler = RobustScaler()
    scaler.center_ = np.asarray(ckpt["scaler_center"])
    scaler.scale_ = np.asarray(ckpt["scaler_scale"])
    scaler.n_features_in_ = int(ckpt.get("scaler_n_features_in", ckpt.get("n_cont", N_CONT)))
    scaler.feature_names_in_ = np.asarray(CONTINUOUS_COLS, dtype=object)
    return scaler


def build_regional_windows(
    df: pd.DataFrame,
    window_sizes: list[int],
    windows_per_chrom_per_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    windows: list[dict] = []
    for (sample_id, chrom), grp in df.groupby(["sample_id", "chrom"], sort=False):
        idx = grp.index.tolist()
        n = len(idx)
        if n < 2:
            continue
        for ws in window_sizes:
            ws_actual = min(ws, n)
            max_start = n - ws_actual
            n_samples = min(windows_per_chrom_per_size, max_start + 1)
            starts = (
                rng.choice(max_start + 1, size=n_samples, replace=False)
                if max_start >= n_samples
                else np.arange(max_start + 1)
            )
            for start in starts:
                win_idx = idx[start : start + ws_actual]
                windows.append({
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "start_bp": int(df.loc[win_idx[0], "pos"]),
                    "end_bp": int(df.loc[win_idx[-1], "end"]),
                    "window_size": int(ws_actual),
                    "global_node_indices": win_idx,
                    "dom_sv_type": str(pd.Series(df.loc[win_idx, "sv_type_str"].values).mode()[0]),
                    "mean_vaf": float(df.loc[win_idx, "vaf_mean"].mean()),
                    "n_bnd": int((df.loc[win_idx, "is_bnd"] > 0).sum()),
                    "n_phased": int((df.loc[win_idx, "has_phase"] > 0).sum()),
                })
    if not windows:
        raise RuntimeError("No graph regional windows were generated")
    return windows


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
        return TSNE(n_components=2, perplexity=perplexity, random_state=42).fit_transform(embeddings)
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=42).fit_transform(embeddings)
    raise ValueError(f"Unknown reduction method: {method}")


def _color_map(labels: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(labels))
    return {label: _PALETTE_20[i % len(_PALETTE_20)] for i, label in enumerate(unique)}


def _scatter_categorical(ax, xy: np.ndarray, labels: list[str], title: str, legend_title: str, colors: dict[str, str] | None = None) -> None:
    cmap = colors or _color_map(labels)
    ax.scatter(xy[:, 0], xy[:, 1], c=[cmap.get(x, "#999999") for x in labels], s=16, alpha=0.75, linewidths=0, rasterized=True)
    handles = [mpatches.Patch(color=color, label=str(label)) for label, color in cmap.items() if label in set(labels)]
    ax.legend(handles=handles, title=legend_title, fontsize=7, title_fontsize=8, loc="best", framealpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _scatter_continuous(ax, xy: np.ndarray, values: np.ndarray, title: str, label: str) -> None:
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=16, alpha=0.75, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=label)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _plot_silhouette_sweep(embeddings: np.ndarray, k_min: int, k_max: int, out_dir: Path) -> None:
    ks, scores = [], []
    max_k = min(k_max, len(embeddings) - 1)
    for k in range(k_min, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=20).fit_predict(embeddings)
        ks.append(k)
        scores.append(silhouette_score(embeddings, labels))
    if not ks:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, scores, "-o")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Graph embedding silhouette sweep")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_silhouette_sweep.png", dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = _cfg_from_checkpoint(ckpt)
    window_sizes = args.window_sizes or [5, 10, 20, 50]

    device = get_device()
    model = SVGraphMAE(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=args.strict)
    model.eval()

    inputs = _resolve_inputs(args.input, args.input_list)
    df = parse_all_severus(inputs)
    scaler = _scaler_from_checkpoint(ckpt)
    feat_matrix, _ = build_node_features(df, scaler=scaler)
    graphs, sample_ids = build_all_graphs(df, feat_matrix, proximity_bp=cfg.proximity_bp)

    rng = np.random.default_rng(args.seed)
    windows = build_regional_windows(df, window_sizes, args.windows_per_chrom_per_size, rng)
    sample_dfs = {sid: grp.reset_index() for sid, grp in df.groupby("sample_id", sort=False)}

    sample_node_embeds: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for graph, sample_id in zip(graphs, sample_ids):
            graph = graph.to(device)
            mask = torch.zeros(graph["sv"].x.shape[0], dtype=torch.bool, device=device)
            _recon, node_h = model(graph, mask)
            sample_node_embeds[sample_id] = node_h

    embeddings: list[np.ndarray] = []
    records: list[dict] = []
    with torch.no_grad():
        for win in windows:
            sid = win["sample_id"]
            sample_df = sample_dfs[sid]
            global_to_local = {orig: local for local, orig in enumerate(sample_df["index"].values)}
            local_idx = [global_to_local[i] for i in win["global_node_indices"] if i in global_to_local]
            if not local_idx:
                continue
            emb = model.regional_embed(sample_node_embeds[sid], local_idx)
            emb = l2_normalize(emb, dim=-1)
            embeddings.append(emb.cpu().numpy())
            records.append({k: v for k, v in win.items() if k != "global_node_indices"})

    if not embeddings:
        raise RuntimeError("No regional graph embeddings were extracted")
    emb_arr = np.stack(embeddings, axis=0)
    meta = pd.DataFrame(records)

    xy = reduce_2d(emb_arr, method=args.reduction)
    if args.n_clusters >= len(emb_arr):
        raise ValueError("--n_clusters must be smaller than the number of embeddings")
    cluster = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=20).fit_predict(emb_arr)
    meta["cluster"] = cluster
    sil = silhouette_score(emb_arr, cluster) if args.n_clusters > 1 else float("nan")

    meta.to_csv(out_dir / "graph_cluster_summary.tsv", sep="\t", index=False)
    np.savez(
        out_dir / "graph_embedding_metrics.npz",
        embeddings=emb_arr,
        xy=xy,
        cluster=cluster,
        silhouette=np.array([sil]),
        **{c: meta[c].values.astype(str) if meta[c].dtype == object else meta[c].values for c in meta.columns},
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_categorical(ax, xy, [str(x) for x in cluster], f"Graph regional embeddings ({args.reduction.upper()}); silhouette={sil:.3f}", "Cluster")
    fig.tight_layout()
    fig.savefig(out_dir / "graph_embedding_plot.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.25)
    facet_specs = [
        ("cluster", "Cluster", "Cluster", None),
        ("sample_id", "Sample ID", "Sample", None),
        ("chrom", "Chromosome", "Chrom", None),
        ("window_size", "Window size (# SVs)", "Window", None),
        ("dom_sv_type", "Dominant SV type", "SV type", SV_COLOURS),
    ]
    for i, (col, title, legend_title, colors) in enumerate(facet_specs):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        labels = meta[col].astype(str).tolist()
        if col == "chrom":
            order = sorted(set(labels), key=lambda c: CHROM_ORDER.get(c, 99))
            colors = {label: _PALETTE_20[j % len(_PALETTE_20)] for j, label in enumerate(order)}
        _scatter_categorical(ax, xy, labels, title, legend_title, colors=colors)
    ax6 = fig.add_subplot(gs[1, 2])
    _scatter_continuous(ax6, xy, meta["mean_vaf"].values, "Mean VAF", "Mean VAF")
    fig.suptitle("SV graph encoder embedding QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "graph_embedding_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.sweep_clusters:
        _plot_silhouette_sweep(emb_arr, args.sweep_k_min, args.sweep_k_max, out_dir)

    log.info("Saved graph embedding visualisation outputs to %s", out_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise pretrained SV graph encoder regional embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", metavar="VCF")
    parser.add_argument("--input_list", metavar="FILE")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--window_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--windows_per_chrom_per_size", type=int, default=30)
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--reduction", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--sweep_clusters", action="store_true")
    parser.add_argument("--sweep_k_min", type=int, default=2)
    parser.add_argument("--sweep_k_max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
