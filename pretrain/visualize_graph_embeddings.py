"""
Visualise pretrained Severus graph encoder genomic-window embeddings.

This QC script mirrors visualize_cn_embeddings.py: it samples fixed genomic bp
windows from VCF-derived SV breakpoints, embeds regional graphs with the
pretrained graph MAE, clusters the embeddings, and writes plots plus metadata.

Outputs include graph_embedding_association_scores.tsv with distance-correlation
scores for each plotted facet, plus graph_visualize.log.
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from dataclasses import dataclass
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
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from torch_geometric.data import Batch
except ImportError as exc:  # pragma: no cover - dependency message only
    raise ImportError(
        "PyTorch Geometric is required for complex_sv.pretrain.visualize_graph_embeddings. "
        "Install torch-geometric and matching PyG wheels for your torch build."
    ) from exc

try:
    from config import GraphEncoderConfig
    from data.severus_parser import (
        CHROM_ORDER,
        CONTINUOUS_COLS,
        N_CONT,
        build_node_features,
        infer_sample_id_from_vcf,
        parse_all_severus,
    )
    from data.sv_region_sampler import (
        build_region_graphs,
        build_sv_bp_windows,
        build_sv_interval_windows,
        window_metadata_frame,
    )
    from pretrain.graph_encoder import SVGraphMAE
    from utils import get_device, l2_normalize, setup_logging, torch_load_checkpoint
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import GraphEncoderConfig
    from data.severus_parser import (
        CHROM_ORDER,
        CONTINUOUS_COLS,
        N_CONT,
        build_node_features,
        infer_sample_id_from_vcf,
        parse_all_severus,
    )
    from data.sv_region_sampler import (
        build_region_graphs,
        build_sv_bp_windows,
        build_sv_interval_windows,
        window_metadata_frame,
    )
    from pretrain.graph_encoder import SVGraphMAE
    from utils import get_device, l2_normalize, setup_logging, torch_load_checkpoint


LOG_NAME = "graph_visualize.log"
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

SV_COLOURS = {
    "DEL": "#E63946",
    "INS": "#2A9D8F",
    "DUP": "#457B9D",
    "BND": "#E9C46A",
    "sBND": "#F4A261",
    "INV": "#6A4C93",
}


@dataclass(frozen=True)
class HighlightSpec:
    label: str
    vcf_path: str
    chrom: str
    start_bp: int | None = None
    end_bp: int | None = None


def _status(message: str) -> None:
    print(f"[visualize_graph_embeddings] {message}", flush=True)


def _setup_script_logging(output_dir: Path) -> logging.Logger:
    logger = setup_logging(output_dir)
    dedicated = (output_dir / LOG_NAME).resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == dedicated:
            logger.info("Writing dedicated graph visualisation log to %s", dedicated)
            return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.FileHandler(dedicated)
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Writing dedicated graph visualisation log to %s", dedicated)
    return logger


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


def _read_highlight_specs(path: str | None) -> list[HighlightSpec]:
    if path is None:
        return []
    highlight_path = Path(path)
    if not highlight_path.exists():
        raise FileNotFoundError(f"--highlight_list not found: {highlight_path}")

    specs: list[HighlightSpec] = []
    with highlight_path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [x.strip() for x in line.split("\t")]
            if len(parts) == 1:
                parts = line.split()
            if len(parts) not in {3, 5}:
                raise ValueError(
                    f"Invalid highlight row {highlight_path}:{line_no}. Expected "
                    "label<TAB>vcf_path<TAB>chrom or label<TAB>vcf_path<TAB>chrom<TAB>start_bp<TAB>end_bp."
                )
            label, vcf_path, chrom = parts[:3]
            if len(parts) == 5:
                start_bp = int(parts[3])
                end_bp = int(parts[4])
                if end_bp <= start_bp:
                    raise ValueError(
                        f"Invalid highlight interval {highlight_path}:{line_no}: "
                        f"end_bp must be greater than start_bp ({start_bp}, {end_bp})"
                    )
            else:
                start_bp = None
                end_bp = None
            specs.append(HighlightSpec(label, vcf_path, chrom, start_bp, end_bp))
    return specs


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
        edge_attr_dim=raw.get("edge_attr_dim", 3),
    )


def _training_config_from_checkpoint(ckpt: dict) -> dict:
    raw = ckpt.get("config", {})
    return raw.get("training", {}) if isinstance(raw, dict) else {}


def _load_model_state_dict(ckpt: dict) -> dict:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    raise KeyError("Checkpoint does not contain model_state_dict or state_dict")


def _scaler_from_checkpoint(ckpt: dict) -> RobustScaler | None:
    if "scaler_center" not in ckpt or "scaler_scale" not in ckpt:
        return None
    scaler = RobustScaler()
    scaler.center_ = np.asarray(ckpt["scaler_center"])
    scaler.scale_ = np.asarray(ckpt["scaler_scale"])
    scaler.n_features_in_ = int(ckpt.get("scaler_n_features_in", ckpt.get("n_cont", N_CONT)))
    return scaler


def _canonical_chrom(chrom: object) -> str:
    text = str(chrom).strip()
    lower = text.lower()
    if lower.startswith("chromosome"):
        text = text[len("chromosome") :]
    elif lower.startswith("chr"):
        text = text[3:]
    text = text.strip()
    if text.upper() in {"X", "Y", "M", "MT"}:
        return "chr" + text.upper().replace("MT", "M")
    return "chr" + text


def _resolve_observed_chrom(df, sample_id: str, chrom: str) -> str:
    sample_df = df[df["sample_id"].astype(str) == str(sample_id)]
    if sample_df.empty:
        raise ValueError(f"Sample {sample_id!r} from highlight list was not found in parsed VCF inputs")
    target = _canonical_chrom(chrom)
    observed = sample_df["chrom"].astype(str).unique().tolist()
    for obs in observed:
        if _canonical_chrom(obs) == target:
            return str(obs)
    raise ValueError(f"Chromosome {chrom!r} not found for sample {sample_id!r}; observed={observed[:30]}")


def build_highlight_windows(specs: list[HighlightSpec], df) -> list[dict]:
    intervals: list[dict] = []
    for spec in specs:
        sample_id = infer_sample_id_from_vcf(spec.vcf_path)
        chrom = _resolve_observed_chrom(df, sample_id, spec.chrom)
        intervals.append(
            {
                "label": spec.label,
                "vcf_path": spec.vcf_path,
                "sample_id": sample_id,
                "chrom": chrom,
                "start_bp": spec.start_bp,
                "end_bp": spec.end_bp,
                "window_source": "highlight",
            }
        )
    return build_sv_interval_windows(df, intervals, min_sv_per_window=1)


def embed_graphs(model: SVGraphMAE, graphs: list, device: torch.device, batch_size: int) -> np.ndarray:
    if not graphs:
        return np.zeros((0, 0), dtype=np.float32)
    embeddings: list[np.ndarray] = []
    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=Batch.from_data_list)
    with torch.no_grad():
        for batch_graph in tqdm(loader, desc="embed", leave=False):
            batch_graph = batch_graph.to(device)
            mask = torch.zeros(batch_graph["sv"].x.shape[0], dtype=torch.bool, device=device)
            _recon, node_h = model(batch_graph, mask)
            batch_vec = batch_graph["sv"].batch
            batch_embeds = []
            for graph_idx in range(int(batch_graph.num_graphs)):
                node_idx = torch.nonzero(batch_vec == graph_idx, as_tuple=False).flatten().tolist()
                if not node_idx:
                    continue
                batch_embeds.append(l2_normalize(model.regional_embed(node_h, node_idx), dim=-1))
            if batch_embeds:
                embeddings.append(torch.stack(batch_embeds, dim=0).cpu().numpy())
    if not embeddings:
        raise RuntimeError("No regional graph embeddings were extracted")
    return np.concatenate(embeddings, axis=0)


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




def _pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (x @ x.T)
    return np.sqrt(np.maximum(dist2, 0.0))


def _double_center(dist: np.ndarray) -> np.ndarray:
    dist = np.asarray(dist, dtype=np.float64)
    return dist - dist.mean(axis=0, keepdims=True) - dist.mean(axis=1, keepdims=True) + dist.mean()


def _distance_correlation_from_distances(x_dist: np.ndarray, y_dist: np.ndarray) -> float:
    x_centered = _double_center(x_dist)
    y_centered = _double_center(y_dist)
    dcov2 = float(np.mean(x_centered * y_centered))
    dvar_x2 = float(np.mean(x_centered * x_centered))
    dvar_y2 = float(np.mean(y_centered * y_centered))
    if dvar_x2 <= 0.0 or dvar_y2 <= 0.0:
        return float("nan")
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar_x2 * dvar_y2)))


def _facet_distance(values: pd.Series, kind: str) -> tuple[np.ndarray, np.ndarray, str]:
    if kind == "numeric":
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(numeric)
        if valid.sum() < 3 or np.unique(numeric[valid]).size < 2:
            return np.zeros((0, 0)), valid, "constant_or_insufficient_numeric_values"
        y = numeric[valid]
        return np.abs(y[:, None] - y[None, :]), valid, "ok"

    labels = values.astype(str).fillna("NA").to_numpy()
    valid = pd.notna(values).to_numpy()
    if valid.sum() < 3 or np.unique(labels[valid]).size < 2:
        return np.zeros((0, 0)), valid, "constant_or_insufficient_categories"
    y = labels[valid]
    return (y[:, None] != y[None, :]).astype(np.float64), valid, "ok"


def _embedding_association_scores(
    embeddings: np.ndarray,
    meta: pd.DataFrame,
    facets: list[tuple[str, str, str]],
    max_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Estimate dependence between each metadata facet and the original embeddings.

    The reported score is distance correlation (dCor): values near 0 indicate
    weak association, while larger values indicate that the facet is more tied
    to geometry in the embedding space. Categorical facets use a same/different
    label distance; numeric facets use absolute value differences.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    n_total = int(len(embeddings))
    max_samples = max(10, int(max_samples))

    for col, label, kind in facets:
        if col not in meta.columns:
            rows.append({"facet": col, "label": label, "kind": kind, "distance_correlation": np.nan, "n_used": 0, "n_total": n_total, "note": "missing_column"})
            continue

        y_dist_full, valid_mask, note = _facet_distance(meta[col], kind)
        valid_indices = np.flatnonzero(valid_mask)
        if note != "ok":
            rows.append({"facet": col, "label": label, "kind": kind, "distance_correlation": np.nan, "n_used": int(valid_indices.size), "n_total": n_total, "note": note})
            continue

        if valid_indices.size > max_samples:
            chosen = np.sort(rng.choice(valid_indices, size=max_samples, replace=False))
            values = meta[col].iloc[chosen]
            y_dist, valid_submask, note = _facet_distance(values, kind)
            sample_indices = chosen[np.flatnonzero(valid_submask)]
        else:
            sample_indices = valid_indices
            y_dist = y_dist_full

        if sample_indices.size < 3 or y_dist.shape[0] < 3:
            score = float("nan")
            note = "insufficient_after_sampling"
        else:
            x_dist = _pairwise_euclidean(embeddings[sample_indices])
            score = _distance_correlation_from_distances(x_dist, y_dist)
            note = "ok"

        rows.append(
            {
                "facet": col,
                "label": label,
                "kind": kind,
                "distance_correlation": score,
                "n_used": int(sample_indices.size),
                "n_total": n_total,
                "note": note,
            }
        )

    return pd.DataFrame(rows)


def _score_map(score_df: pd.DataFrame) -> dict[str, float]:
    if score_df.empty:
        return {}
    return dict(zip(score_df["facet"].astype(str), pd.to_numeric(score_df["distance_correlation"], errors="coerce")))


def _title_with_score(title: str, scores: dict[str, float], facet: str) -> str:
    score = scores.get(facet)
    if score is None or not np.isfinite(score):
        return f"{title} (dCor=NA)"
    return f"{title} (dCor={score:.2f})"


def _color_map(labels: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(labels))
    return {label: _PALETTE_20[i % len(_PALETTE_20)] for i, label in enumerate(unique)}


def _scatter_categorical(ax, xy: np.ndarray, labels: list[str], title: str, legend_title: str, colors: dict[str, str] | None = None, max_legend_items: int = 24) -> None:
    cmap = colors or _color_map(labels)
    ax.scatter(xy[:, 0], xy[:, 1], c=[cmap.get(x, "#999999") for x in labels], s=16, alpha=0.75, linewidths=0, rasterized=True)
    if len(cmap) <= max_legend_items:
        handles = [mpatches.Patch(color=color, label=str(label)) for label, color in cmap.items() if label in set(labels)]
        ax.legend(handles=handles, title=legend_title, fontsize=7, title_fontsize=8, loc="best", framealpha=0.85)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _scatter_ordered(ax, xy: np.ndarray, labels: list[str], order: list[str], title: str, legend_title: str) -> None:
    colors = {label: _PALETTE_20[i % len(_PALETTE_20)] for i, label in enumerate(order)}
    _scatter_categorical(ax, xy, labels, title, legend_title, colors=colors)


def _scatter_continuous(ax, xy: np.ndarray, values: np.ndarray, title: str, label: str) -> None:
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=16, alpha=0.75, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=label)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _overlay_highlights(ax, highlight_xy: np.ndarray | None, labels: list[str] | None, annotate: bool = False) -> None:
    if highlight_xy is None or len(highlight_xy) == 0:
        return
    ax.scatter(
        highlight_xy[:, 0],
        highlight_xy[:, 1],
        marker="*",
        s=220,
        c="#FFD166",
        edgecolors="#111111",
        linewidths=1.2,
        zorder=8,
    )
    if annotate and labels is not None:
        for label, xy in zip(labels, highlight_xy):
            ax.annotate(
                str(label),
                xy=(xy[0], xy[1]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="bold",
                color="#111111",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#111111", "alpha": 0.85},
                zorder=9,
            )


def _plot_silhouette_sweep(embeddings: np.ndarray, k_min: int, k_max: int, out_dir: Path) -> None:
    ks: list[int] = []
    scores: list[float] = []
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
    ax.set_title("Graph bp-window embedding silhouette sweep")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_silhouette_sweep.png", dpi=150)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = _setup_script_logging(out_dir)

    _status(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch_load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = _cfg_from_checkpoint(ckpt)
    training_cfg = _training_config_from_checkpoint(ckpt)

    if args.window_bp_sizes is None:
        args.window_bp_sizes = training_cfg.get(
            "window_bp_sizes",
            [1_000_000, 2_500_000, 5_000_000, 10_000_000, 25_000_000, 50_000_000, 100_000_000],
        )
    if args.windows_per_chrom_per_size is None:
        args.windows_per_chrom_per_size = int(training_cfg.get("windows_per_chrom_per_size", 20))
    if args.cluster_windows_per_chrom_per_size is None:
        args.cluster_windows_per_chrom_per_size = int(training_cfg.get("cluster_windows_per_chrom_per_size", 20))
    if args.min_sv_per_window is None:
        args.min_sv_per_window = 1
    if args.max_windows is None and training_cfg.get("max_windows") is not None:
        args.max_windows = int(training_cfg["max_windows"])

    device = get_device()
    _status(f"Using device: {device}")
    model = SVGraphMAE(cfg).to(device)
    model.load_state_dict(_load_model_state_dict(ckpt), strict=args.strict)
    model.eval()
    _status("Model loaded")

    inputs = _resolve_inputs(args.input, args.input_list)
    _status(f"Parsing {len(inputs):,} Severus VCF file(s)")
    df = parse_all_severus(inputs)
    _status(f"Parsed {len(df):,} SV record(s) across {df['sample_id'].nunique():,} sample(s)")

    scaler = _scaler_from_checkpoint(ckpt)
    feat_matrix, _ = build_node_features(df, scaler=scaler)

    rng = np.random.default_rng(args.seed)
    _status(
        "Sampling graph bp windows with "
        f"sizes={list(args.window_bp_sizes)}, random_per_chrom={args.windows_per_chrom_per_size}, "
        f"dense_per_chrom={args.cluster_windows_per_chrom_per_size}, "
        f"min_sv_per_window={args.min_sv_per_window}, max_windows={args.max_windows}"
    )
    windows = build_sv_bp_windows(
        df,
        window_bp_sizes=[int(x) for x in args.window_bp_sizes],
        windows_per_chrom_per_size=int(args.windows_per_chrom_per_size),
        cluster_windows_per_chrom_per_size=int(args.cluster_windows_per_chrom_per_size),
        min_sv_per_window=int(args.min_sv_per_window),
        rng=rng,
        max_windows=args.max_windows,
        progress=True,
    )
    if not windows:
        raise RuntimeError("No graph bp windows generated for visualization")
    meta = window_metadata_frame(windows, include_node_indices=True)
    _status(f"Built {len(windows):,} background graph window(s)")

    highlight_specs = _read_highlight_specs(args.highlight_list)
    if highlight_specs:
        _status(f"Building {len(highlight_specs):,} highlighted graph window(s)")
    highlight_windows = build_highlight_windows(highlight_specs, df) if highlight_specs else []
    highlight_meta = window_metadata_frame(highlight_windows, include_node_indices=True) if highlight_windows else pd.DataFrame()

    _status("Building regional graph objects")
    graphs = build_region_graphs(df, feat_matrix, windows, proximity_bp=cfg.proximity_bp, progress=True)
    highlight_graphs = build_region_graphs(df, feat_matrix, highlight_windows, proximity_bp=cfg.proximity_bp, progress=True) if highlight_windows else []

    _status(f"Embedding {len(graphs):,} background graph window(s)")
    emb = embed_graphs(model, graphs, device=device, batch_size=args.batch_size)
    highlight_emb = embed_graphs(model, highlight_graphs, device=device, batch_size=args.batch_size) if highlight_graphs else np.zeros((0, emb.shape[1]), dtype=np.float32)

    if len(emb) < 3:
        raise RuntimeError("Need at least 3 background embeddings for visualization/clustering")
    combined_emb = np.concatenate([emb, highlight_emb], axis=0) if len(highlight_emb) > 0 else emb

    _status(f"Running {args.reduction.upper()} reduction on {len(combined_emb):,} embedding(s)")
    xy_all = reduce_2d(combined_emb, method=args.reduction)
    xy = xy_all[: len(emb)]
    highlight_xy = xy_all[len(emb) :] if len(highlight_emb) > 0 else None

    if args.n_clusters >= len(emb):
        raise ValueError("--n_clusters must be smaller than the number of background embeddings")
    _status(f"Clustering {len(emb):,} background embedding(s) into {args.n_clusters} cluster(s)")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=20)
    cluster = kmeans.fit_predict(emb)
    meta["cluster"] = cluster
    sil = silhouette_score(emb, cluster) if args.n_clusters > 1 else float("nan")

    association_facets = [
        ("cluster", "Cluster", "categorical"),
        ("sample_id", "Sample ID", "categorical"),
        ("chrom", "Chromosome", "categorical"),
        ("requested_window_bp_size", "Window bp size", "numeric"),
        ("window_source", "Window source", "categorical"),
        ("dom_sv_type", "Dominant SV type", "categorical"),
        ("breakpoint_density_per_mb", "Breakpoint density", "numeric"),
        ("n_sv", "SV count", "numeric"),
        ("n_bnd", "BND count", "numeric"),
        ("n_inv_like_bnd", "INV-like BND count", "numeric"),
        ("n_foldback", "Foldback count", "numeric"),
        ("n_interchrom_mate", "Interchrom mate count", "numeric"),
        ("n_phased", "Phased SV count", "numeric"),
    ]
    association_scores = _embedding_association_scores(
        emb,
        meta,
        association_facets,
        max_samples=args.association_max_samples,
        seed=args.seed,
    )
    association_scores.to_csv(out_dir / "graph_embedding_association_scores.tsv", sep="\t", index=False)
    score_by_facet = _score_map(association_scores)
    _status(f"Saved association scores: {out_dir / 'graph_embedding_association_scores.tsv'}")

    if len(highlight_emb) > 0:
        highlight_meta = highlight_meta.copy()
        highlight_meta["cluster"] = kmeans.predict(highlight_emb)
        highlight_meta["x"] = highlight_xy[:, 0]
        highlight_meta["y"] = highlight_xy[:, 1]
        highlight_meta.to_csv(out_dir / "graph_highlighted_points.tsv", sep="\t", index=False)
        _status(f"Saved highlighted point metadata: {out_dir / 'graph_highlighted_points.tsv'}")

    meta.to_csv(out_dir / "graph_cluster_summary.tsv", sep="\t", index=False)
    dense_report = meta.sort_values("breakpoint_density_per_mb", ascending=False).head(min(len(meta), 500))
    dense_report.to_csv(out_dir / "graph_dense_window_report.tsv", sep="\t", index=False)
    _status(f"Saved cluster metadata: {out_dir / 'graph_cluster_summary.tsv'}")

    reserved_npz_keys = {"embeddings", "xy", "cluster", "silhouette", "highlight_embeddings", "highlight_xy"}
    meta_arrays = {
        c: meta[c].values.astype(str) if meta[c].dtype == object else meta[c].values
        for c in meta.columns
        if c not in reserved_npz_keys
    }
    highlight_arrays = {}
    if len(highlight_emb) > 0:
        for c in highlight_meta.columns:
            values = highlight_meta[c].values
            if values.dtype == object:
                values = values.astype(str)
            highlight_arrays[f"highlight_{c}"] = values

    np.savez(
        out_dir / "graph_embedding_metrics.npz",
        embeddings=emb,
        xy=xy,
        cluster=cluster,
        silhouette=np.array([sil]),
        highlight_embeddings=highlight_emb,
        highlight_xy=np.zeros((0, 2), dtype=np.float32) if highlight_xy is None else highlight_xy,
        **meta_arrays,
        **highlight_arrays,
    )
    _status(f"Saved embedding arrays: {out_dir / 'graph_embedding_metrics.npz'}")

    highlight_labels = highlight_meta["label"].astype(str).tolist() if len(highlight_meta) > 0 and "label" in highlight_meta else []

    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_categorical(
        ax,
        xy,
        [str(x) for x in cluster],
        (
            f"Graph bp-window embeddings ({args.reduction.upper()}); "
            f"silhouette={sil:.3f}; {_title_with_score('cluster', score_by_facet, 'cluster')}"
        ),
        "Cluster",
    )
    _overlay_highlights(ax, highlight_xy, highlight_labels, annotate=True)
    fig.tight_layout()
    fig.savefig(out_dir / "graph_embedding_plot.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.34, wspace=0.25)
    facet_specs = [
        ("cluster", _title_with_score("Cluster", score_by_facet, "cluster"), "Cluster", None),
        ("sample_id", _title_with_score("Sample ID", score_by_facet, "sample_id"), "Sample", None),
        ("chrom", _title_with_score("Chromosome", score_by_facet, "chrom"), "Chrom", None),
        ("requested_window_bp_size", _title_with_score("Window bp size", score_by_facet, "requested_window_bp_size"), "bp", None),
        ("window_source", _title_with_score("Window source", score_by_facet, "window_source"), "Source", None),
        ("dom_sv_type", _title_with_score("Dominant SV type", score_by_facet, "dom_sv_type"), "SV type", SV_COLOURS),
    ]
    for i, (col, title, legend_title, colors) in enumerate(facet_specs):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        labels = meta[col].astype(str).tolist()
        if col == "chrom":
            order = sorted(set(labels), key=lambda c: CHROM_ORDER.get(c, CHROM_ORDER.get(c.removeprefix("chr"), 99)))
            _scatter_ordered(ax, xy, labels, order, title, legend_title)
        elif col == "requested_window_bp_size":
            order = [str(x) for x in sorted(meta[col].astype(int).unique().tolist())]
            _scatter_ordered(ax, xy, labels, order, title, legend_title)
        else:
            _scatter_categorical(ax, xy, labels, title, legend_title, colors=colors)
        _overlay_highlights(ax, highlight_xy, highlight_labels)
    fig.suptitle("SV graph encoder bp-window embedding QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "graph_embedding_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.25)
    continuous_specs = [
        ("breakpoint_density_per_mb", _title_with_score("Breakpoint density", score_by_facet, "breakpoint_density_per_mb"), "SVs per Mb"),
        ("n_sv", _title_with_score("SV count", score_by_facet, "n_sv"), "SV count"),
        ("n_bnd", _title_with_score("BND count", score_by_facet, "n_bnd"), "BND count"),
        ("n_inv_like_bnd", _title_with_score("INV-like BND count", score_by_facet, "n_inv_like_bnd"), "INV-like BND count"),
        ("n_foldback", _title_with_score("Foldback count", score_by_facet, "n_foldback"), "Foldback count"),
        ("n_interchrom_mate", _title_with_score("Interchrom mate count", score_by_facet, "n_interchrom_mate"), "Interchrom mate count"),
    ]
    for i, (col, title, label) in enumerate(continuous_specs):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        _scatter_continuous(ax, xy, meta[col].values, title, label)
        _overlay_highlights(ax, highlight_xy, highlight_labels)
    fig.suptitle("SV graph encoder bp-window continuous QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "graph_embedding_continuous_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.sweep_clusters:
        _status(f"Running silhouette sweep k={args.sweep_k_min}..{args.sweep_k_max}")
        _plot_silhouette_sweep(emb, args.sweep_k_min, args.sweep_k_max, out_dir)

    logger.info("Saved graph bp-window embedding visualisation outputs to %s", out_dir)
    _status(f"Done. Dedicated log: {out_dir / LOG_NAME}")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise pretrained SV graph encoder bp-window embeddings from Severus VCFs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", nargs="+", metavar="VCF")
    parser.add_argument("--input_list", metavar="FILE", help="Text file with one Severus VCF path per line.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument(
        "--highlight_list",
        default=None,
        metavar="FILE",
        help=(
            "Optional TSV of highlighted regions: label, vcf_path, chrom, optional start_bp, end_bp. "
            "Rows with only label/vcf_path/chrom embed the whole observed chromosome span."
        ),
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--window_bp_sizes", type=int, nargs="+", default=None)
    parser.add_argument("--windows_per_chrom_per_size", type=int, default=None)
    parser.add_argument("--cluster_windows_per_chrom_per_size", type=int, default=None)
    parser.add_argument("--min_sv_per_window", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--reduction", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--sweep_clusters", action="store_true")
    parser.add_argument("--sweep_k_min", type=int, default=2)
    parser.add_argument("--sweep_k_max", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--association_max_samples",
        type=int,
        default=2000,
        help="Maximum background embeddings sampled per facet when estimating distance-correlation scores.",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
