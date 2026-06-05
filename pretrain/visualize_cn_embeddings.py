"""
Visualise pretrained CN encoder bp-window embeddings.

Recommended pipeline position
-----------------------------
Run after Phase 1 CN bp-window pretraining:

    python pretrain/visualize_cn_embeddings.py \
        --input_list results/pipeline2/cn_pretrain/wakhan_list.txt \
        --checkpoint results/pipeline2/cn_pretrain/cn_encoder.pt \
        --output_dir results/pipeline2/cn_pretrain/embedding_qc \
        --max_windows 50000

The input list contains one Wakhan BED root per line. Each root resolves to the
paired haplotype files:

    ROOT_copynumbers_segments_HP_1.bed
    ROOT_copynumbers_segments_HP_2.bed

Optional highlighted points can be supplied with --highlight_list. Each row is:

    label<TAB>bed_root<TAB>chrom
    label<TAB>bed_root<TAB>chrom<TAB>start_bp<TAB>end_bp

If start/end are omitted, the whole observed chromosome span is embedded.

Outputs:
    cn_embedding_plot.png
    cn_embedding_facets.png
    cn_embedding_continuous_facets.png
    cn_cluster_summary.tsv
    cn_embedding_metrics.npz
    cn_highlighted_points.tsv      optional
    cn_silhouette_sweep.png        optional
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
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
    from data.cn_resampler import CN_CHANNELS, build_bp_window_tensors, region_to_tensor
    from data.severus_parser import CHROM_ORDER
    from data.wakhan_parser import parse_all_wakhan, parse_wakhan
    from pretrain.cn_encoder import CNMaskedAutoencoder
    from utils import get_device, l2_normalize, setup_logging
except ImportError:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    from config import CNEncoderConfig
    from data.cn_resampler import CN_CHANNELS, build_bp_window_tensors, region_to_tensor
    from data.severus_parser import CHROM_ORDER
    from data.wakhan_parser import parse_all_wakhan, parse_wakhan
    from pretrain.cn_encoder import CNMaskedAutoencoder
    from utils import get_device, l2_normalize, setup_logging


log = logging.getLogger(__name__)


def _status(message: str) -> None:
    print(f"[visualize_cn_embeddings] {message}", flush=True)

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


@dataclass(frozen=True)
class HighlightSpec:
    label: str
    bed_root: str
    chrom: str
    start_bp: int | None = None
    end_bp: int | None = None


def _resolve_input_roots(input_list: str) -> list[str]:
    """Collect Wakhan BED roots from --input_list without requiring root files to exist."""
    list_path = Path(input_list)
    if not list_path.exists():
        raise FileNotFoundError(f"--input_list not found: {list_path}")

    roots: list[str] = []
    with list_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            item = line.strip()
            if item and not item.startswith("#"):
                roots.append(item)

    unique = list(dict.fromkeys(roots))
    if not unique:
        raise FileNotFoundError("--input_list did not contain any Wakhan BED roots")

    return unique


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
                    f"Invalid highlight row {highlight_path}:{line_no}. "
                    "Expected label<TAB>bed_root<TAB>chrom or "
                    "label<TAB>bed_root<TAB>chrom<TAB>start_bp<TAB>end_bp."
                )

            label, bed_root, chrom = parts[:3]
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

            specs.append(
                HighlightSpec(
                    label=label,
                    bed_root=bed_root,
                    chrom=chrom,
                    start_bp=start_bp,
                    end_bp=end_bp,
                )
            )

    return specs


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


def _checkpoint_channels(ckpt: dict) -> list[str] | None:
    raw = ckpt.get("config", {})
    channels = raw.get("cn_channels") if isinstance(raw, dict) else None
    return list(channels) if channels is not None else None


def _make_encoder_cfg(base: CNEncoderConfig) -> SimpleNamespace:
    """Match pretrain_cn.py compatibility wrapper."""
    max_bins = max(base.n_bins_arm, base.n_bins_region)
    return SimpleNamespace(
        d_model=base.d_model,
        n_heads=base.n_heads,
        n_layers=base.n_layers,
        ff_dim=base.ff_dim,
        d_ff=base.ff_dim,
        dropout=base.dropout,
        n_bins_arm=max_bins,
        n_bins_region=base.n_bins_region,
        seq_len=max_bins,
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


def _add_tensor_summaries(tensors: list[torch.Tensor], meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()

    cn_total_idx = CN_CHANNELS.index("cn_total")
    loh_idx = CN_CHANNELS.index("loh")
    ai_idx = CN_CHANNELS.index("allele_imbalance")
    bp_idx = CN_CHANNELS.index("breakpoint_count") if "breakpoint_count" in CN_CHANNELS else None

    mean_cn_total: list[float] = []
    mean_allele_imbalance: list[float] = []
    loh_fraction: list[float] = []
    mean_breakpoint_count: list[float] = []

    for tensor in tensors:
        arr = tensor.detach().cpu().numpy()
        mean_cn_total.append(float(arr[:, cn_total_idx].mean()))
        mean_allele_imbalance.append(float(arr[:, ai_idx].mean()))
        loh_fraction.append(float(arr[:, loh_idx].mean()))
        if bp_idx is not None:
            mean_breakpoint_count.append(float(arr[:, bp_idx].mean()))

    meta["mean_cn_total"] = mean_cn_total
    meta["mean_allele_imbalance"] = mean_allele_imbalance
    meta["loh_fraction"] = loh_fraction
    if bp_idx is not None:
        meta["mean_breakpoint_count"] = mean_breakpoint_count

    return meta


def build_window_dataset(
    df: pd.DataFrame,
    window_bp_sizes: list[int],
    n_bins: int,
    windows_per_chrom_per_size: int,
    min_covered_fraction: float,
    seed: int,
    max_windows: int | None = None,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    rng = np.random.default_rng(seed)

    tensors, meta = build_bp_window_tensors(
        df,
        window_bp_sizes=window_bp_sizes,
        n_bins=n_bins,
        windows_per_chrom_per_size=windows_per_chrom_per_size,
        min_covered_fraction=min_covered_fraction,
        rng=rng,
        max_windows=max_windows,
        progress=True,
    )

    if not tensors:
        raise RuntimeError(
            "No bp-window tensors were generated from Wakhan inputs. "
            "Try smaller --window_bp_sizes or lower --min_covered_fraction."
        )

    meta = _add_tensor_summaries(tensors, meta)

    log.info(
        "Built %d bp-window tensors across %d sample(s).",
        len(tensors),
        meta["sample_id"].nunique(),
    )

    return tensors, meta


def _canonical_chrom(chrom: object) -> str:
    text = str(chrom).strip()
    lower = text.lower()
    if lower.startswith("chromosome"):
        text = text[len("chromosome") :]
    elif lower.startswith("chr"):
        text = text[3:]

    text = text.strip()
    if text.upper() in {"X", "Y", "M", "MT"}:
        return "chr" + text.upper()
    return "chr" + text


def _chrom_subset(df: pd.DataFrame, chrom: str) -> pd.DataFrame:
    target = _canonical_chrom(chrom)
    mask = df["chrom"].map(_canonical_chrom) == target
    out = df.loc[mask].copy()
    if out.empty:
        observed = sorted(df["chrom"].astype(str).unique().tolist())
        raise ValueError(f"Chromosome {chrom!r} not found. Observed chromosomes: {observed[:30]}")
    return out


def build_highlight_dataset(
    specs: list[HighlightSpec],
    n_bins: int,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    if not specs:
        return [], pd.DataFrame()

    parsed_cache: dict[str, pd.DataFrame] = {}
    tensors: list[torch.Tensor] = []
    rows: list[dict[str, object]] = []

    for spec in specs:
        if spec.bed_root not in parsed_cache:
            parsed_cache[spec.bed_root] = parse_wakhan(spec.bed_root)

        sample_df = parsed_cache[spec.bed_root]
        chrom_df = _chrom_subset(sample_df, spec.chrom)

        if spec.start_bp is None or spec.end_bp is None:
            start_bp = int(chrom_df["start"].min())
            end_bp = int(chrom_df["end"].max())
        else:
            start_bp = int(spec.start_bp)
            end_bp = int(spec.end_bp)

        if end_bp <= start_bp:
            raise ValueError(f"Invalid highlighted interval for {spec.label}: {start_bp}-{end_bp}")

        tensor = region_to_tensor(chrom_df, start_bp=start_bp, end_bp=end_bp, n_bins=n_bins)
        tensors.append(tensor)

        overlap = chrom_df[(chrom_df["end"] > start_bp) & (chrom_df["start"] < end_bp)].copy()
        clipped_start = overlap["start"].clip(lower=start_bp) if not overlap.empty else pd.Series(dtype=int)
        clipped_end = overlap["end"].clip(upper=end_bp) if not overlap.empty else pd.Series(dtype=int)
        covered_bp = int((clipped_end - clipped_start).clip(lower=0).sum()) if not overlap.empty else 0

        rows.append(
            {
                "label": spec.label,
                "sample_id": str(chrom_df["sample_id"].iloc[0]),
                "bed_root": spec.bed_root,
                "chrom": str(chrom_df["chrom"].iloc[0]),
                "start_bp": start_bp,
                "end_bp": end_bp,
                "window_bp_size": end_bp - start_bp,
                "covered_bp": covered_bp,
                "covered_fraction": covered_bp / max(end_bp - start_bp, 1),
                "n_segments": int(len(overlap)),
            }
        )

    meta = _add_tensor_summaries(tensors, pd.DataFrame(rows))
    log.info("Built %d highlighted tensor(s).", len(tensors))
    return tensors, meta


def embed_tensors(
    model: torch.nn.Module,
    tensors: list[torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if not tensors:
        return np.zeros((0, 0), dtype=np.float32)

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for x in DataLoader(WindowTensorDataset(tensors), batch_size=batch_size, shuffle=False):
            x = x.to(device=device, dtype=torch.float32)
            mask = torch.zeros(x.shape[:2], dtype=torch.bool, device=device)
            _recon, cls_emb, _bin_embs = _forward_cn(model, x, mask)
            cls_emb = l2_normalize(cls_emb, dim=-1)
            embeddings.append(cls_emb.cpu().numpy())

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


def _color_map(labels: list[str]) -> dict[str, str]:
    unique = list(dict.fromkeys(labels))
    return {label: _PALETTE_20[i % len(_PALETTE_20)] for i, label in enumerate(unique)}


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
        handles = [mpatches.Patch(color=color, label=str(label)) for label, color in colors.items()]
        ax.legend(handles=handles, title=legend_title, fontsize=7, title_fontsize=8, loc="best", framealpha=0.85)

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
    cmap = {label: _PALETTE_20[j % len(_PALETTE_20)] for j, label in enumerate(order)}
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
        handles = [mpatches.Patch(color=cmap[label], label=label) for label in order]
        ax.legend(handles=handles, title=legend_title, fontsize=7, title_fontsize=8, loc="best", framealpha=0.85)

    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _scatter_continuous(ax, xy: np.ndarray, values: np.ndarray, title: str, label: str) -> None:
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=18, alpha=0.75, linewidths=0, rasterized=True)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, label=label)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")


def _overlay_highlights(
    ax,
    highlight_xy: np.ndarray | None,
    labels: list[str] | None,
    annotate: bool = False,
) -> None:
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

    _status(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = _cfg_from_checkpoint(ckpt)
    training_cfg = _training_config_from_checkpoint(ckpt)

    ckpt_channels = _checkpoint_channels(ckpt)
    if ckpt_channels is not None and ckpt_channels != list(CN_CHANNELS):
        log.warning(
            "Checkpoint CN channels differ from current CN_CHANNELS. checkpoint=%s current=%s",
            ckpt_channels,
            list(CN_CHANNELS),
        )

    if args.n_bins is not None:
        cfg.n_bins_region = args.n_bins

    if args.window_bp_sizes is None:
        args.window_bp_sizes = training_cfg.get("window_bp_sizes", [50_000, 100_000, 250_000, 500_000, 1_000_000])

    _status(
        "Using n_bins_region="
        f"{cfg.n_bins_region}, window_bp_sizes={list(args.window_bp_sizes)}"
    )

    if args.windows_per_chrom_per_size is None:
        args.windows_per_chrom_per_size = int(training_cfg.get("windows_per_chrom_per_size", 40))

    if args.min_covered_fraction is None:
        args.min_covered_fraction = float(training_cfg.get("min_covered_fraction", 0.10))

    if args.max_windows is None and training_cfg.get("max_windows") is not None:
        args.max_windows = int(training_cfg["max_windows"])

    device = get_device()
    _status(f"Using device: {device}")
    model = CNMaskedAutoencoder(_make_encoder_cfg(cfg)).to(device)
    state_dict = _load_model_state_dict(ckpt)
    model.load_state_dict(state_dict, strict=args.strict)
    model.eval()
    _status("Model loaded")

    input_roots = _resolve_input_roots(args.input_list)
    _status(f"Parsing {len(input_roots)} Wakhan BED root(s) from {args.input_list}")
    df = parse_all_wakhan(input_roots)
    _status(
        f"Parsed {len(df):,} paired segment(s) across "
        f"{df['sample_id'].nunique():,} sample(s)"
    )

    n_chrom_groups = df.groupby(["sample_id", "chrom"]).ngroups
    upper_bound = n_chrom_groups * len(args.window_bp_sizes) * int(args.windows_per_chrom_per_size)
    _status(
        "Sampling background windows with "
        f"windows_per_chrom_per_size={args.windows_per_chrom_per_size}, "
        f"min_covered_fraction={args.min_covered_fraction}, "
        f"upper_bound~{upper_bound:,}, max_windows={args.max_windows}"
    )
    tensors, meta = build_window_dataset(
        df,
        window_bp_sizes=[int(x) for x in args.window_bp_sizes],
        n_bins=cfg.n_bins_region,
        windows_per_chrom_per_size=int(args.windows_per_chrom_per_size),
        min_covered_fraction=float(args.min_covered_fraction),
        seed=args.seed,
        max_windows=args.max_windows,
    )
    _status(f"Built {len(tensors):,} background window tensor(s)")

    highlight_specs = _read_highlight_specs(args.highlight_list)
    if highlight_specs:
        _status(f"Building {len(highlight_specs):,} highlighted point tensor(s)")
    highlight_tensors, highlight_meta = build_highlight_dataset(highlight_specs, n_bins=cfg.n_bins_region)

    _status(f"Embedding {len(tensors):,} background tensor(s)")
    emb = embed_tensors(model, tensors, device=device, batch_size=args.batch_size)
    if highlight_tensors:
        _status(f"Embedding {len(highlight_tensors):,} highlighted tensor(s)")
    highlight_emb = embed_tensors(model, highlight_tensors, device=device, batch_size=args.batch_size)

    if len(emb) < 3:
        raise RuntimeError("Need at least 3 background embeddings for visualization/clustering")

    if len(highlight_emb) > 0:
        combined_emb = np.concatenate([emb, highlight_emb], axis=0)
    else:
        combined_emb = emb

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

    if len(highlight_emb) > 0:
        highlight_meta = highlight_meta.copy()
        highlight_meta["cluster"] = kmeans.predict(highlight_emb)
        highlight_meta["x"] = highlight_xy[:, 0]
        highlight_meta["y"] = highlight_xy[:, 1]
        highlight_meta.to_csv(out_dir / "cn_highlighted_points.tsv", sep="\t", index=False)
        _status(f"Saved highlighted point metadata: {out_dir / 'cn_highlighted_points.tsv'}")

    meta.to_csv(out_dir / "cn_cluster_summary.tsv", sep="\t", index=False)
    _status(f"Saved cluster metadata: {out_dir / 'cn_cluster_summary.tsv'}")

    reserved_npz_keys = {
        "embeddings",
        "xy",
        "cluster",
        "silhouette",
        "highlight_embeddings",
        "highlight_xy",
    }
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

    _status(f"Saving embedding arrays: {out_dir / 'cn_embedding_metrics.npz'}")
    np.savez(
        out_dir / "cn_embedding_metrics.npz",
        embeddings=emb,
        xy=xy,
        cluster=cluster,
        silhouette=np.array([sil]),
        highlight_embeddings=highlight_emb,
        highlight_xy=np.zeros((0, 2), dtype=np.float32) if highlight_xy is None else highlight_xy,
        **meta_arrays,
        **highlight_arrays,
    )

    highlight_labels = highlight_meta["label"].astype(str).tolist() if len(highlight_meta) > 0 else []

    _status("Writing embedding plots")
    fig, ax = plt.subplots(figsize=(9, 7))
    _scatter_categorical(
        ax,
        xy,
        [str(x) for x in cluster],
        f"CN bp-window embeddings ({args.reduction.upper()}); silhouette={sil:.3f}",
        "Cluster",
    )
    _overlay_highlights(ax, highlight_xy, highlight_labels, annotate=True)
    fig.tight_layout()
    fig.savefig(out_dir / "cn_embedding_plot.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_categorical(ax1, xy, meta["cluster"].astype(str).tolist(), "Cluster", "Cluster")
    _overlay_highlights(ax1, highlight_xy, highlight_labels)

    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_categorical(ax2, xy, meta["sample_id"].astype(str).tolist(), "Sample ID", "Sample")
    _overlay_highlights(ax2, highlight_xy, highlight_labels)

    ax3 = fig.add_subplot(gs[1, 0])
    chrom_labels = meta["chrom"].astype(str).tolist()
    chrom_order = sorted(set(chrom_labels), key=lambda c: CHROM_ORDER.get(c, CHROM_ORDER.get(c.removeprefix("chr"), 99)))
    _scatter_categorical_ordered(ax3, xy, chrom_labels, chrom_order, "Chromosome", "Chrom")
    _overlay_highlights(ax3, highlight_xy, highlight_labels)

    ax4 = fig.add_subplot(gs[1, 1])
    size_col = "requested_window_bp_size" if "requested_window_bp_size" in meta.columns else "window_bp_size"
    window_labels = meta[size_col].astype(int).astype(str).tolist()
    window_order = [str(x) for x in sorted(meta[size_col].astype(int).unique().tolist())]
    _scatter_categorical_ordered(ax4, xy, window_labels, window_order, "Window bp size", "bp")
    _overlay_highlights(ax4, highlight_xy, highlight_labels)

    fig.suptitle("CN encoder bp-window embedding QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "cn_embedding_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    _scatter_continuous(ax1, xy, meta["mean_cn_total"].values, "Mean total CN", "Mean total CN")
    _overlay_highlights(ax1, highlight_xy, highlight_labels)

    ax2 = fig.add_subplot(gs[0, 1])
    _scatter_continuous(
        ax2,
        xy,
        meta["mean_allele_imbalance"].values,
        "Mean allele imbalance",
        "Mean allele imbalance",
    )
    _overlay_highlights(ax2, highlight_xy, highlight_labels)

    ax3 = fig.add_subplot(gs[1, 0])
    _scatter_continuous(ax3, xy, meta["loh_fraction"].values, "LOH fraction", "LOH fraction")
    _overlay_highlights(ax3, highlight_xy, highlight_labels)

    ax4 = fig.add_subplot(gs[1, 1])
    if "mean_breakpoint_count" in meta.columns:
        _scatter_continuous(
            ax4,
            xy,
            meta["mean_breakpoint_count"].values,
            "Mean breakpoint count",
            "Mean breakpoint count",
        )
    else:
        _scatter_continuous(ax4, xy, meta["covered_fraction"].values, "Window covered fraction", "Covered fraction")
    _overlay_highlights(ax4, highlight_xy, highlight_labels)

    fig.suptitle("CN encoder bp-window continuous QC", fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "cn_embedding_continuous_facets.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    if args.sweep_clusters:
        _status(f"Running silhouette sweep k={args.sweep_k_min}..{args.sweep_k_max}")
        _plot_silhouette_sweep(emb, args.sweep_k_min, args.sweep_k_max, out_dir)

    _status(f"Done. Outputs written to {out_dir}")
    log.info("Saved CN bp-window embedding visualisation outputs to %s", out_dir)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise pretrained CN encoder bp-window embeddings from paired Wakhan BED roots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_list",
        required=True,
        metavar="FILE",
        help="Text file with one Wakhan BED root per line.",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument(
        "--highlight_list",
        default=None,
        metavar="FILE",
        help=(
            "Optional TSV of highlighted points: label, bed_root, chrom, optional start_bp, end_bp. "
            "Rows with only label/bed_root/chrom embed the whole observed chromosome span."
        ),
    )

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
        help="Fixed genomic bp window sizes to visualize. Defaults to checkpoint training config if available.",
    )
    parser.add_argument(
        "--windows_per_chrom_per_size",
        type=int,
        default=None,
        help="Random bp windows sampled per sample/chromosome/window size. Defaults to checkpoint training config if available.",
    )
    parser.add_argument(
        "--min_covered_fraction",
        type=float,
        default=None,
        help="Minimum fraction of a bp window overlapped by observed CN segments. Defaults to checkpoint training config if available.",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help=(
            "Maximum number of random background windows to embed. "
            "Defaults to checkpoint training config if available; highlighted points are always embedded."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters", type=int, default=6)
    parser.add_argument("--reduction", choices=["umap", "tsne", "pca"], default="umap")
    parser.add_argument("--sweep_clusters", action="store_true")
    parser.add_argument("--sweep_k_min", type=int, default=2)
    parser.add_argument("--sweep_k_max", type=int, default=15)
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
