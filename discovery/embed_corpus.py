"""Prototype-mode embedding extraction for complex-SV candidates."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import RobustScaler

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from torch_geometric.data import Batch
except ImportError:  # pragma: no cover
    Batch = None  # type: ignore

from config import CNEncoderConfig, GraphEncoderConfig
from data.anchor_manifest import canonical_sample_id
from data.cn_resampler import CN_CHANNELS, region_to_tensor
from data.graph_builder import build_sample_graph
from data.region_proposal import (
    candidates_to_frame,
    label_rows_to_candidates,
    merge_candidates,
    propose_cn_candidates,
)
from data.severus_parser import N_CONT, build_node_features, parse_severus
from data.wakhan_parser import parse_wakhan
from model.prototypes import PrototypeCache
from pretrain.cn_encoder import CNMaskedAutoencoder
from pretrain.graph_encoder import SVGraphMAE
from utils import get_device, l2_normalize, torch_load_checkpoint

log = logging.getLogger(__name__)
MAX_REGION_SV_NODES = 500


@dataclass
class SampleBundle:
    sample_id: str
    wakhan_root: str
    severus_vcf: str
    wakhan_df: pd.DataFrame
    severus_df: pd.DataFrame
    graph: Any | None
    feat_matrix: np.ndarray | None = None
    node_h: torch.Tensor | None = None


def _load_model_state_dict(ckpt: dict) -> dict:
    if "model_state_dict" in ckpt:
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt:
        return ckpt["state_dict"]
    raise KeyError("Checkpoint does not contain model_state_dict or state_dict")


def _cn_cfg_from_checkpoint(ckpt: dict) -> CNEncoderConfig:
    raw = ckpt.get("config", {})
    if "cn_encoder" in raw:
        raw = raw["cn_encoder"]
    return CNEncoderConfig(
        d_model=raw.get("d_model", 256),
        n_heads=raw.get("n_heads", 8),
        n_layers=raw.get("n_layers", 6),
        ff_dim=raw.get("ff_dim", raw.get("d_ff", 1024)),
        dropout=raw.get("dropout", 0.1),
        n_bins_arm=raw.get("n_bins_arm", raw.get("seq_len", 256)),
        n_bins_region=raw.get("n_bins_region", raw.get("seq_len", 128)),
        mask_prob=raw.get("mask_prob", 0.15),
    )


def _make_cn_encoder_cfg(cfg: CNEncoderConfig) -> SimpleNamespace:
    max_bins = max(int(cfg.n_bins_arm), int(cfg.n_bins_region))
    return SimpleNamespace(
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        ff_dim=cfg.ff_dim,
        d_ff=cfg.ff_dim,
        dropout=cfg.dropout,
        n_bins_arm=max_bins,
        n_bins_region=cfg.n_bins_region,
        seq_len=max_bins,
        mask_prob=cfg.mask_prob,
        embed_dim=cfg.d_model,
    )


def _graph_cfg_from_checkpoint(ckpt: dict) -> GraphEncoderConfig:
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
    return scaler


def load_cn_encoder(path: str | Path, device: torch.device, strict: bool = True) -> tuple[CNMaskedAutoencoder, CNEncoderConfig]:
    ckpt = torch_load_checkpoint(path, map_location="cpu")
    cfg = _cn_cfg_from_checkpoint(ckpt)
    model = CNMaskedAutoencoder(_make_cn_encoder_cfg(cfg)).to(device)
    model.load_state_dict(_load_model_state_dict(ckpt), strict=strict)
    model.eval()
    return model, cfg


def load_graph_encoder(
    path: str | Path | None,
    device: torch.device,
    strict: bool = True,
) -> tuple[SVGraphMAE | None, GraphEncoderConfig | None, RobustScaler | None]:
    if path is None:
        return None, None, None
    ckpt = torch_load_checkpoint(path, map_location="cpu")
    cfg = _graph_cfg_from_checkpoint(ckpt)
    model = SVGraphMAE(cfg).to(device)
    model.load_state_dict(_load_model_state_dict(ckpt), strict=strict)
    model.eval()
    return model, cfg, _scaler_from_checkpoint(ckpt)


def _chrom_equal(a: object, b: object) -> bool:
    return str(a).removeprefix("chr") == str(b).removeprefix("chr")


def _candidate_segments(bundle: SampleBundle, candidate: dict[str, Any]) -> pd.DataFrame:
    if "df_segments" in candidate and isinstance(candidate["df_segments"], pd.DataFrame) and not candidate["df_segments"].empty:
        return candidate["df_segments"].copy()
    start_bp = int(candidate["start_bp"])
    end_bp = int(candidate["end_bp"])
    chrom = str(candidate["chrom"])
    df = bundle.wakhan_df
    mask = (
        df["chrom"].astype(str).map(lambda c: _chrom_equal(c, chrom))
        & (pd.to_numeric(df["end"], errors="coerce") > start_bp)
        & (pd.to_numeric(df["start"], errors="coerce") < end_bp)
    )
    return df.loc[mask].copy()


def _sv_indices_in_candidate(bundle: SampleBundle, candidate: dict[str, Any]) -> list[int]:
    explicit = candidate.get("sv_node_indices")
    if explicit:
        return [int(x) for x in explicit]
    if bundle.severus_df.empty:
        return []
    start_bp = int(candidate["start_bp"])
    end_bp = int(candidate["end_bp"])
    chrom = str(candidate["chrom"])
    df = bundle.severus_df
    mask = (
        df["chrom"].astype(str).map(lambda c: _chrom_equal(c, chrom))
        & (pd.to_numeric(df["pos"], errors="coerce") >= start_bp)
        & (pd.to_numeric(df["pos"], errors="coerce") < end_bp)
    )
    return df.index[mask].astype(int).tolist()


def extract_segment_stats(segs: pd.DataFrame, start_bp: int, end_bp: int) -> torch.Tensor:
    """Return the 18-dimensional bounded segment-statistics vector."""
    if segs.empty or end_bp <= start_bp:
        return torch.zeros(18, dtype=torch.float32)

    cn_total = pd.to_numeric(segs["cn_total"], errors="coerce").fillna(2.0)
    hp1 = pd.to_numeric(segs["cn_hp1"], errors="coerce").fillna(1.0)
    hp2 = pd.to_numeric(segs["cn_hp2"], errors="coerce").fillna(1.0)
    loh = pd.to_numeric(segs["loh"], errors="coerce").fillna(0.0)
    imbalance = pd.to_numeric(segs["allele_imbalance"], errors="coerce").fillna(0.0)
    bp = pd.to_numeric(segs["breakpoint_count"], errors="coerce").fillna(0.0)
    span = max(int(end_bp) - int(start_bp), 1)
    span_mb = span / 1_000_000.0
    cn_values = cn_total.to_numpy(dtype=float)
    diffs = np.diff(cn_values)
    signs = np.sign(diffs[np.abs(diffs) > 1e-6])
    oscillation = float(np.sum(signs[1:] != signs[:-1]) / max(len(signs) - 1, 1)) if len(signs) > 1 else 0.0
    cn_state_count = float(np.unique(np.round(cn_values / 0.3)).size)
    covered = float((pd.to_numeric(segs["end"], errors="coerce") - pd.to_numeric(segs["start"], errors="coerce")).clip(lower=0).sum())
    values = np.array(
        [
            min(np.log1p(span_mb) / 5.0, 1.0),
            min(len(segs) / 200.0, 1.0),
            min(float(cn_total.mean()) / 10.0, 1.0),
            min(float(cn_total.std(ddof=0)) / 5.0, 1.0),
            min(float(cn_total.min()) / 10.0, 1.0),
            min(float(cn_total.max()) / 10.0, 1.0),
            min(cn_state_count / 20.0, 1.0),
            oscillation,
            float(loh.mean()),
            float(imbalance.mean()),
            float(imbalance.max()),
            min(float(bp.mean()) / 10.0, 1.0),
            min(float(bp.sum()) / 200.0, 1.0),
            min(float(bp.sum()) / max(span_mb, 1e-6) / 50.0, 1.0),
            min(float(hp1.mean()) / 10.0, 1.0),
            min(float(hp2.mean()) / 10.0, 1.0),
            min(float((hp1 - hp2).abs().mean()) / 5.0, 1.0),
            min(covered / span, 1.0),
        ],
        dtype=np.float32,
    )
    return torch.as_tensor(values, dtype=torch.float32)


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t").fillna("")
    required = {"sample_id", "wakhan_root", "severus_vcf"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return df


def read_labels(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t").fillna("")
    required = {"label_id", "sample_id", "chrom", "start_bp", "end_bp", "sv_class"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Labels file missing columns: {missing}")
    return df


def load_sample_bundles(
    manifest: pd.DataFrame,
    graph_cfg: GraphEncoderConfig | None,
    graph_scaler: RobustScaler | None,
) -> dict[str, SampleBundle]:
    """Parse sample inputs without constructing expensive full-sample graphs."""
    bundles: dict[str, SampleBundle] = {}
    rows = manifest.to_dict("records")
    for row_idx, row in enumerate(rows, start=1):
        sample_id = canonical_sample_id(str(row["sample_id"]))
        wakhan_root = str(row["wakhan_root"])
        severus_vcf = str(row.get("severus_vcf", ""))
        log.info("Loading sample %d/%d: %s", row_idx, len(rows), sample_id)
        wakhan_df = parse_wakhan(wakhan_root)
        wakhan_df["sample_id"] = sample_id

        severus_df = pd.DataFrame()
        feat_matrix = None
        if severus_vcf:
            severus_df = parse_severus(severus_vcf, sample_id=sample_id).reset_index(drop=True)
            if graph_cfg is not None and not severus_df.empty:
                feat_matrix, _scaler = build_node_features(severus_df, scaler=graph_scaler)
                log.info(
                    "  %s: prepared Severus feature matrix %s; regional graphs will be built per candidate",
                    sample_id,
                    tuple(feat_matrix.shape),
                )

        bundles[sample_id] = SampleBundle(
            sample_id=sample_id,
            wakhan_root=wakhan_root,
            severus_vcf=severus_vcf,
            wakhan_df=wakhan_df,
            severus_df=severus_df,
            graph=None,
            feat_matrix=feat_matrix,
        )
    return bundles


def encode_graph_bundles(
    bundles: dict[str, SampleBundle],
    graph_model: SVGraphMAE | None,
    device: torch.device,
) -> None:
    """Deprecated full-graph hook kept for compatibility.

    Prototype mode now encodes regional candidate graphs on demand to avoid
    full-sample dense mate/phase edge explosions on large Severus samples.
    """
    return


def _light_graph_candidates(bundle: SampleBundle, min_junctions: int = 3, min_span_bp: int = 1_000_000) -> list[dict[str, Any]]:
    """Cheap Severus graph-style candidates from cluster and phase groups."""
    df = bundle.severus_df
    if df.empty:
        return []
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, tuple[int, ...]]] = set()

    group_specs: list[tuple[str, pd.core.groupby.DataFrameGroupBy]] = []
    if "cluster_id" in df.columns:
        valid = df[df["cluster_id"].astype(str).isin(["", "."]) == False]
        if not valid.empty:
            group_specs.append(("cluster", valid.groupby("cluster_id", sort=False)))
    if "phase_set" in df.columns:
        phase = pd.to_numeric(df["phase_set"], errors="coerce").fillna(0).astype(int)
        valid = df[phase != 0]
        if not valid.empty:
            group_specs.append(("phase", valid.groupby("phase_set", sort=False)))

    for source, grouped in group_specs:
        for _group_id, grp in grouped:
            for chrom, chrom_grp in grp.groupby("chrom", sort=False):
                if len(chrom_grp) < int(min_junctions):
                    continue
                idx = sorted(int(i) for i in chrom_grp.index.tolist())
                start_bp = int(pd.to_numeric(chrom_grp["pos"], errors="coerce").min())
                end_bp = int(pd.to_numeric(chrom_grp["end"], errors="coerce").max())
                if end_bp - start_bp < int(min_span_bp):
                    continue
                key = (str(chrom), start_bp, end_bp, tuple(idx))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "sample_id": bundle.sample_id,
                        "chrom": str(chrom),
                        "start_bp": start_bp,
                        "end_bp": end_bp,
                        "evidence": "graph_only",
                        "sv_node_indices": idx,
                        "df_segments": pd.DataFrame(),
                        "n_sv": int(len(idx)),
                        "graph_source": source,
                    }
                )
    return candidates


def embed_candidate(
    candidate: dict[str, Any],
    bundle: SampleBundle,
    cn_model: CNMaskedAutoencoder,
    cn_cfg: CNEncoderConfig,
    graph_model: SVGraphMAE | None,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Any]]:
    start_bp = int(candidate["start_bp"])
    end_bp = int(candidate["end_bp"])
    chrom = str(candidate["chrom"])
    segs = _candidate_segments(bundle, candidate)
    cn_tensor = region_to_tensor(segs, start_bp, end_bp, n_bins=cn_cfg.n_bins_region).unsqueeze(0).to(device)
    cn_mask = torch.zeros(cn_tensor.shape[:2], dtype=torch.bool, device=device)

    with torch.no_grad():
        _cn_recon, cn_cls, _cn_bins = cn_model(cn_tensor, cn_mask)
        parts = [l2_normalize(cn_cls.squeeze(0), dim=0)]

        sv_indices = _sv_indices_in_candidate(bundle, candidate)
        original_n_sv = len(sv_indices)
        if len(sv_indices) > MAX_REGION_SV_NODES:
            positions = np.linspace(0, len(sv_indices) - 1, MAX_REGION_SV_NODES).round().astype(int)
            sv_indices = [sv_indices[int(pos)] for pos in positions]
            log.warning(
                "Candidate %s:%s-%s has %d SV nodes; embedding an even subset of %d nodes",
                bundle.sample_id,
                start_bp,
                end_bp,
                original_n_sv,
                len(sv_indices),
            )
        if graph_model is not None and bundle.feat_matrix is not None and sv_indices:
            idx = [i for i in sv_indices if 0 <= int(i) < len(bundle.severus_df)]
            sub_df = bundle.severus_df.loc[idx].copy().reset_index(drop=True)
            sub_feat = np.asarray(bundle.feat_matrix, dtype=np.float32)[idx]
            region_graph = build_sample_graph(
                sub_df,
                sub_feat,
                proximity_bp=int(getattr(getattr(graph_model, "cfg", None), "proximity_bp", 1_000_000)),
            ).to(device)
            mask = torch.zeros(region_graph["sv"].x.shape[0], dtype=torch.bool, device=device)
            _graph_recon, node_h = graph_model(region_graph, mask)
            local_indices = list(range(region_graph["sv"].x.shape[0]))
            graph_regional = graph_model.regional_embed(node_h, local_indices)
            graph_global = graph_model.global_embed(node_h)
            parts.extend([graph_regional, graph_global])
        else:
            embed_dim = int(getattr(getattr(graph_model, "cfg", None), "embed_dim", 64)) if graph_model is not None else 64
            parts.extend(
                [
                    torch.zeros(embed_dim, dtype=torch.float32, device=device),
                    torch.zeros(embed_dim, dtype=torch.float32, device=device),
                ]
            )

        stats = extract_segment_stats(segs, start_bp, end_bp).to(device)
        parts.append(F.normalize(stats, p=2, dim=0) if torch.linalg.norm(stats) > 0 else stats)
        embedding = l2_normalize(torch.cat(parts, dim=0), dim=0)

    metadata = {
        "candidate_id": candidate.get("candidate_id", candidate.get("label_id", "")),
        "label_id": candidate.get("label_id", ""),
        "sample_id": bundle.sample_id,
        "chrom": chrom,
        "start_bp": start_bp,
        "end_bp": end_bp,
        "evidence": candidate.get("evidence", ""),
        "sv_class": candidate.get("sv_class", ""),
        "label_scope": candidate.get("label_scope", ""),
        "n_segments": int(len(segs)),
        "n_sv_nodes": int(original_n_sv),
        "encoded_sv_nodes": int(len(sv_indices)),
        "embedding_mode": "encoder_concat",
    }
    return embedding.detach().cpu().numpy().astype(np.float32), metadata


def build_candidates(
    source: str,
    labels: pd.DataFrame,
    bundles: dict[str, SampleBundle],
) -> list[dict[str, Any]]:
    wakhan_by_sample = {sample_id: bundle.wakhan_df for sample_id, bundle in bundles.items()}
    label_candidates = label_rows_to_candidates(labels, wakhan_by_sample) if not labels.empty else []
    if source == "labels":
        return label_candidates

    proposal_candidates: list[dict[str, Any]] = []
    bundle_items = list(bundles.values())
    for bundle_idx, bundle in enumerate(bundle_items, start=1):
        log.info("Proposing candidates for sample %d/%d: %s", bundle_idx, len(bundle_items), bundle.sample_id)
        cn = propose_cn_candidates(bundle.wakhan_df)
        graph = _light_graph_candidates(bundle)
        merged = merge_candidates(
            cn,
            graph,
            overlap_threshold=0.5,
            df_severus_sample=bundle.severus_df,
        )
        log.info(
            "  %s: %d CN candidates, %d graph candidates, %d merged candidates",
            bundle.sample_id,
            len(cn),
            len(graph),
            len(merged),
        )
        proposal_candidates.extend(merged)
    log.info("Built %d proposal candidate(s) without full-sample graph encoding", len(proposal_candidates))

    if source == "proposals":
        return proposal_candidates
    if source == "all":
        return label_candidates + proposal_candidates
    raise ValueError("candidate_source must be one of labels, proposals, all")


def write_embedding_outputs(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv_path = out_dir / "candidate_embeddings.tsv"
    npz_path = out_dir / "embeddings.npz"
    metadata.to_csv(tsv_path, sep="\t", index=False)
    arrays = {}
    for col in metadata.columns:
        if metadata[col].dtype == object:
            arrays[col] = metadata[col].astype(str).to_numpy()
        else:
            arrays[col] = metadata[col].to_numpy()
    np.savez(npz_path, embeddings=embeddings, **arrays)
    return tsv_path, npz_path


def build_prototypes(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    tau: float,
    output_path: str | Path,
) -> PrototypeCache:
    labels = metadata["sv_class"].astype(str)
    known = labels != ""
    cache = PrototypeCache(embed_dim=int(embeddings.shape[1]), tau=tau)
    for class_name in sorted(labels[known].unique()):
        idx = np.where((labels == class_name).to_numpy())[0]
        cache.add_class(class_name, torch.as_tensor(embeddings[idx], dtype=torch.float32))
    cache.save(output_path)
    return cache


def write_distance_table(
    cache: PrototypeCache,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, result in enumerate(cache.classify_batch(torch.as_tensor(embeddings, dtype=torch.float32))):
        pred, confidence, distances = result
        row = metadata.iloc[i].to_dict()
        row["predicted_class"] = pred
        row["prototype_confidence"] = confidence
        for name, dist in distances.items():
            row[f"d_{name}"] = dist
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    return df


def write_leave_one_out(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    tau: float,
    output_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = metadata["sv_class"].astype(str).to_numpy()
    for i, class_name in enumerate(labels):
        if not class_name:
            continue
        support_idx = np.where((labels == class_name) & (np.arange(len(labels)) != i))[0]
        if support_idx.size == 0:
            continue
        cache = PrototypeCache(embed_dim=int(embeddings.shape[1]), tau=tau)
        cache.add_class(class_name, torch.as_tensor(embeddings[support_idx], dtype=torch.float32))
        pred, confidence, distances = cache.classify(torch.as_tensor(embeddings[i], dtype=torch.float32))
        row = metadata.iloc[i].to_dict()
        row["held_out_class"] = class_name
        row["predicted_class"] = pred
        row["prototype_confidence"] = confidence
        row["leave_one_out_distance"] = distances.get(class_name, np.nan)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    return df


def _reduce_embeddings_2d(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    """Return a 2D projection for plots, using UMAP when reasonable."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), "empty"
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32), "single point"

    if n >= 5:
        try:
            import umap

            xy = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            ).fit_transform(embeddings)
            return np.asarray(xy, dtype=np.float32), "UMAP"
        except Exception as exc:  # pragma: no cover - plotting fallback
            log.warning("UMAP projection failed, falling back to PCA: %s", exc)

    from sklearn.decomposition import PCA

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    xy_small = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
    xy = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    xy[:, :n_components] = xy_small
    return xy, "PCA"


def _label_colors(values: pd.Series) -> tuple[list[str], dict[str, object]]:
    labels = values.astype(str).replace("", "unlabeled").tolist()
    unique = list(dict.fromkeys(labels))
    cmap = plt.get_cmap("tab20")
    colors = {label: cmap(i % 20) for i, label in enumerate(unique)}
    return [colors[label] for label in labels], colors


def _short_label(row: pd.Series) -> str:
    value = str(row.get("candidate_id", "")) or str(row.get("label_id", ""))
    return value[:32]


def write_visualizations(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    distances: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    output_dir: str | Path,
    tau: float,
) -> None:
    """Write PNG summaries for prototype-mode outputs."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    xy, method = _reduce_embeddings_2d(embeddings)
    if "predicted_class" in distances:
        color_source = distances["predicted_class"]
    elif "sv_class" in metadata:
        color_source = metadata["sv_class"]
    else:
        color_source = pd.Series([""] * len(metadata))
    point_colors, legend_colors = _label_colors(color_source)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(xy[:, 0], xy[:, 1], c=point_colors, s=56, alpha=0.85, linewidths=0.4, edgecolors="black")
    if len(metadata) <= 40:
        for i, row in metadata.reset_index(drop=True).iterrows():
            ax.annotate(_short_label(row), (xy[i, 0], xy[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    handles = [plt.Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=color, markersize=8) for label, color in legend_colors.items()]
    if handles:
        ax.legend(handles=handles, title="Class", fontsize=8, title_fontsize=9)
    ax.set_title(f"Candidate Embedding Projection ({method})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "embedding_projection.png", dpi=180)
    plt.close(fig)

    distance_cols = [c for c in distances.columns if c.startswith("d_")]
    if distance_cols:
        primary_col = distance_cols[0]
        plot_df = distances.sort_values(primary_col, ascending=True).reset_index(drop=True)
        fallback = plot_df["label_id"].astype(str) if "label_id" in plot_df else pd.Series([""] * len(plot_df))
        labels = plot_df["candidate_id"].astype(str).mask(plot_df["candidate_id"].astype(str) == "", fallback)
        fig_h = max(4.0, 0.32 * len(plot_df) + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.barh(np.arange(len(plot_df)), plot_df[primary_col].astype(float), color="#4C78A8")
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(x=tau, color="#E45756", linestyle="--", linewidth=1.2, label=f"tau={tau:g}")
        ax.set_xlabel("Cosine distance to prototype")
        ax.set_title(f"Prototype Distance ({primary_col.removeprefix('d_')})")
        ax.legend()
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "prototype_distances.png", dpi=180)
        plt.close(fig)

    if not leave_one_out.empty and "leave_one_out_distance" in leave_one_out:
        plot_df = leave_one_out.sort_values("leave_one_out_distance", ascending=True).reset_index(drop=True)
        fallback = plot_df["label_id"].astype(str) if "label_id" in plot_df else pd.Series([""] * len(plot_df))
        labels = plot_df["candidate_id"].astype(str).mask(plot_df["candidate_id"].astype(str) == "", fallback)
        fig_h = max(4.0, 0.32 * len(plot_df) + 1.5)
        fig, ax = plt.subplots(figsize=(8, fig_h))
        ax.barh(np.arange(len(plot_df)), plot_df["leave_one_out_distance"].astype(float), color="#59A14F")
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(x=tau, color="#E45756", linestyle="--", linewidth=1.2, label=f"tau={tau:g}")
        ax.set_xlabel("Held-out distance")
        ax.set_title("Leave-One-Out Prototype Distances")
        ax.legend()
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(out_dir / "anchor_leave_one_out.png", dpi=180)
        plt.close(fig)

    if "sv_class" in metadata:
        known_mask = metadata["sv_class"].astype(str) != ""
    else:
        known_mask = pd.Series([False] * len(metadata))
    known_idx = np.where(known_mask.to_numpy())[0]
    if known_idx.size >= 2 and known_idx.size <= 80:
        known_emb = embeddings[known_idx]
        known_emb = known_emb / np.clip(np.linalg.norm(known_emb, axis=1, keepdims=True), 1e-12, None)
        sim = known_emb @ known_emb.T
        labels = metadata.iloc[known_idx]["candidate_id"].astype(str).tolist()
        fig_size = max(5.5, min(12.0, 0.45 * len(labels) + 3.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Known-Anchor Cosine Similarity")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "anchor_similarity_heatmap.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    log.info("Using device: %s", device)

    cn_model, cn_cfg = load_cn_encoder(args.cn_checkpoint, device=device, strict=args.strict)
    graph_model, graph_cfg, graph_scaler = load_graph_encoder(args.graph_checkpoint, device=device, strict=args.strict)

    manifest = read_manifest(args.manifest)
    labels = read_labels(args.labels)
    bundles = load_sample_bundles(manifest, graph_cfg=graph_cfg, graph_scaler=graph_scaler)
    encode_graph_bundles(bundles, graph_model, device)

    candidates = build_candidates(args.candidate_source, labels, bundles)
    log.info("Embedding %d candidate(s) from source=%s", len(candidates), args.candidate_source)
    if not candidates:
        raise RuntimeError("No candidates were available to embed")
    candidates_to_frame(candidates).to_csv(output_dir / "candidate_regions.tsv", sep="\t", index=False)

    embeddings: list[np.ndarray] = []
    meta_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        if i == 0 or (i + 1) % 25 == 0 or (i + 1) == len(candidates):
            log.info("Embedding candidate %d/%d", i + 1, len(candidates))
        sample_id = canonical_sample_id(str(cand["sample_id"]))
        if sample_id not in bundles:
            log.warning("Skipping candidate for unknown sample_id=%s", sample_id)
            continue
        emb, meta = embed_candidate(cand, bundles[sample_id], cn_model, cn_cfg, graph_model, device)
        if not meta["candidate_id"]:
            meta["candidate_id"] = f"cand_{i:06d}"
        embeddings.append(emb)
        meta_rows.append(meta)

    if not embeddings:
        raise RuntimeError("No candidate embeddings were produced")
    emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)
    log.info("Writing embedding tables and prototype outputs")
    write_embedding_outputs(emb_array, meta_df, output_dir)

    proto_path = output_dir / args.prototypes_name
    cache = build_prototypes(emb_array, meta_df, tau=args.tau, output_path=proto_path)
    distances = write_distance_table(cache, emb_array, meta_df, output_dir / "prototype_distances.tsv")
    leave_one_out = write_leave_one_out(emb_array, meta_df, tau=args.tau, output_path=output_dir / "anchor_leave_one_out.tsv")
    log.info("Writing visualization PNGs")
    write_visualizations(emb_array, meta_df, distances, leave_one_out, output_dir, tau=args.tau)
    log.info("Wrote %d embedding(s), %d prototype class(es)", len(meta_df), cache.n_classes())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Sample manifest TSV")
    parser.add_argument("--labels", required=False, help="Label TSV with anchor intervals")
    parser.add_argument("--cn_checkpoint", required=True, help="Pretrained CN checkpoint")
    parser.add_argument("--graph_checkpoint", required=False, help="Pretrained graph checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--candidate_source", choices=["labels", "proposals", "all"], default="labels")
    parser.add_argument("--prototypes_name", default="prototypes.pt")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
