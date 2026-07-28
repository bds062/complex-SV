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
from data.cn_resampler import CN_CHANNELS, get_arm_bounds, region_to_tensor
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
DEFAULT_TAU = 0.1
EMBEDDING_NORMALIZATION_CHOICES = ("none", "sample_residual")
REPORT_SCOPE_CHOICES = ("all", "anchors")
CANDIDATE_SOURCE_CHOICES = ("labels", "chromosomes", "chromosome-arms", "proposals", "all")
SCAN_EVIDENCE_VALUES = {"chromosome_scan", "chromosome_arm_scan", "candidate_region_empty"}


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
        edge_attr_dim=raw.get("edge_attr_dim", 3),
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


def _clean_arm(value: object) -> str:
    text = str(value).strip().lower()
    return text if text in {"p", "q"} else ""


def _chrom_key(chrom: object) -> str:
    return str(chrom).removeprefix("chr")


def _scan_evidence_mask(evidence: pd.Series) -> np.ndarray:
    return evidence.astype(str).isin(SCAN_EVIDENCE_VALUES).to_numpy()


def _chromosome_arm_bounds(grp: pd.DataFrame) -> list[tuple[str, int, int]]:
    bounds: list[tuple[str, int, int]] = []
    for arm_name, start_bp, end_bp in get_arm_bounds(grp):
        arm_name_text = str(arm_name)
        arm = _clean_arm(arm_name_text[-1:])
        if not arm:
            arm = ""
        bounds.append((arm, int(start_bp), int(end_bp)))
    return bounds


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


def _predict_haplotype(segs: pd.DataFrame, sub_df: pd.DataFrame | None) -> dict[str, Any]:
    """Rule-based haplotype prediction from CN segments and SV haplotype support.

    Combines three signals: (1) SV read-support vote (supp_hp1 vs supp_hp2),
    (2) length-weighted CN asymmetry, and (3) mixed_HP detection when both
    haplotypes show elevated copy number.

    Returns keys: haplotype (HP1/HP2/mixed_HP), haplotype_score [-1,1],
    haplotype_confidence [0,1], sv_hp1_support, sv_hp2_support,
    cn_hp1_mean, cn_hp2_mean.
    """
    sv_hp1, sv_hp2, sv_score = 0.0, 0.0, 0.0
    if sub_df is not None and not sub_df.empty and "supp_hp1" in sub_df and "supp_hp2" in sub_df:
        sv_hp1 = float(pd.to_numeric(sub_df["supp_hp1"], errors="coerce").fillna(0).sum())
        sv_hp2 = float(pd.to_numeric(sub_df["supp_hp2"], errors="coerce").fillna(0).sum())
        total = sv_hp1 + sv_hp2
        if total > 0:
            sv_score = (sv_hp1 - sv_hp2) / total

    cn_hp1_mean, cn_hp2_mean, cn_score = 1.0, 1.0, 0.0
    if not segs.empty and "cn_hp1" in segs and "cn_hp2" in segs:
        hp1 = pd.to_numeric(segs["cn_hp1"], errors="coerce").fillna(1.0)
        hp2 = pd.to_numeric(segs["cn_hp2"], errors="coerce").fillna(1.0)
        lengths = (
            pd.to_numeric(segs.get("end", pd.Series(dtype=float)), errors="coerce")
            - pd.to_numeric(segs.get("start", pd.Series(dtype=float)), errors="coerce")
        ).clip(lower=0).fillna(0)
        total_len = float(lengths.sum())
        if total_len > 0:
            cn_hp1_mean = float((hp1 * lengths).sum() / total_len)
            cn_hp2_mean = float((hp2 * lengths).sum() / total_len)
        else:
            cn_hp1_mean, cn_hp2_mean = float(hp1.mean()), float(hp2.mean())
        cn_total = cn_hp1_mean + cn_hp2_mean
        if cn_total > 0:
            cn_score = (cn_hp1_mean - cn_hp2_mean) / cn_total

    sv_weight = 2.0 if (sv_hp1 + sv_hp2) > 0 else 0.0
    haplotype_score = (sv_weight * sv_score + cn_score) / (sv_weight + 1.0)

    sv_frac = sv_hp1 / (sv_hp1 + sv_hp2 + 1e-9)
    both_cn_elevated = cn_hp1_mean > 2.5 and cn_hp2_mean > 2.5
    sv_mixed = (sv_hp1 + sv_hp2) > 0 and 0.3 <= sv_frac <= 0.7
    mixed_hp = both_cn_elevated or sv_mixed

    THRESHOLD = 0.25
    if mixed_hp:
        haplotype = "mixed_HP"
        if both_cn_elevated:
            confidence = float(min(1.0, (min(cn_hp1_mean, cn_hp2_mean) - 2.0) / 3.0))
        else:
            confidence = float(max(0.3, 1.0 - abs(sv_frac - 0.5) / 0.2))
    elif haplotype_score > THRESHOLD:
        haplotype = "HP1"
        confidence = float(min(1.0, (haplotype_score - THRESHOLD) / (1.0 - THRESHOLD)))
    elif haplotype_score < -THRESHOLD:
        haplotype = "HP2"
        confidence = float(min(1.0, (-haplotype_score - THRESHOLD) / (1.0 - THRESHOLD)))
    else:
        haplotype = "mixed_HP"
        confidence = 0.2

    return {
        "haplotype": haplotype,
        "haplotype_score": round(float(haplotype_score), 4),
        "haplotype_confidence": round(confidence, 4),
        "sv_hp1_support": round(sv_hp1, 2),
        "sv_hp2_support": round(sv_hp2, 2),
        "cn_hp1_mean": round(cn_hp1_mean, 4),
        "cn_hp2_mean": round(cn_hp2_mean, 4),
    }



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

    sub_df: pd.DataFrame | None = None

    with torch.no_grad():
        _cn_recon, cn_cls, _cn_bin_embs = cn_model(cn_tensor, cn_mask)
        cn_cls_vec = cn_cls.squeeze(0)

        def normalize_block(x: torch.Tensor) -> torch.Tensor:
            return l2_normalize(x, dim=0)

        parts = [normalize_block(cn_cls_vec)]

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
            parts.extend([normalize_block(graph_regional), normalize_block(graph_global)])
        else:
            embed_dim = int(getattr(getattr(graph_model, "cfg", None), "embed_dim", 64)) if graph_model is not None else 64
            parts.extend(
                [
                    normalize_block(torch.zeros(embed_dim, dtype=torch.float32, device=device)),
                    normalize_block(torch.zeros(embed_dim, dtype=torch.float32, device=device)),
                ]
            )

        stats = extract_segment_stats(segs, start_bp, end_bp).to(device)
        parts.append(normalize_block(stats))
        embedding = l2_normalize(torch.cat(parts, dim=0), dim=0)

    hap = _predict_haplotype(segs, sub_df)

    metadata: dict[str, Any] = {
        "candidate_id": candidate.get("candidate_id", candidate.get("label_id", "")),
        "label_id": candidate.get("label_id", ""),
        "sample_id": bundle.sample_id,
        "chrom": chrom,
        "arm": candidate.get("arm", ""),
        "start_bp": start_bp,
        "end_bp": end_bp,
        "evidence": candidate.get("evidence", ""),
        "sv_class": candidate.get("sv_class", ""),
        "label_scope": candidate.get("label_scope", ""),
        "candidate_scope": candidate.get("candidate_scope", candidate.get("label_scope", "")),
        "n_segments": int(len(segs)),
        "n_sv_nodes": int(original_n_sv),
        "encoded_sv_nodes": int(len(sv_indices)),
        "embedding_mode": "encoder_concat",
        **hap,
    }
    return embedding.detach().cpu().numpy().astype(np.float32), metadata


def build_candidates(
    source: str,
    labels: pd.DataFrame,
    bundles: dict[str, SampleBundle],
) -> list[dict[str, Any]]:
    source = str(source).strip().replace("_", "-")
    wakhan_by_sample = {sample_id: bundle.wakhan_df for sample_id, bundle in bundles.items()}
    label_candidates = label_rows_to_candidates(labels, wakhan_by_sample) if not labels.empty else []
    if source == "labels":
        return label_candidates
    if source == "chromosomes":
        labeled_keys = {
            (str(cand["sample_id"]), _chrom_key(cand["chrom"]))
            for cand in label_candidates
        }
        chromosome_candidates: list[dict[str, Any]] = []
        for bundle in bundles.values():
            df = bundle.wakhan_df
            if df.empty:
                continue
            for chrom, grp_raw in df.groupby("chrom", sort=False):
                chrom_text = str(chrom)
                key = (bundle.sample_id, _chrom_key(chrom_text))
                if key in labeled_keys:
                    continue
                grp = grp_raw.copy()
                chromosome_candidates.append(
                    {
                        "candidate_id": f"{bundle.sample_id}_{chrom_text}",
                        "label_id": "",
                        "sample_id": bundle.sample_id,
                        "chrom": chrom_text,
                        "arm": "",
                        "start_bp": int(pd.to_numeric(grp["start"], errors="coerce").min()),
                        "end_bp": int(pd.to_numeric(grp["end"], errors="coerce").max()),
                        "evidence": "chromosome_scan",
                        "sv_node_indices": [],
                        "df_segments": grp,
                        "sv_class": "",
                        "label_scope": "chromosome",
                        "candidate_scope": "chromosome",
                    }
                )
        return label_candidates + chromosome_candidates

    if source == "chromosome-arms":
        labeled_whole_keys: set[tuple[str, str]] = set()
        labeled_arm_keys: set[tuple[str, str, str]] = set()
        for cand in label_candidates:
            sample_id = str(cand["sample_id"])
            chrom_key = _chrom_key(cand["chrom"])
            arm = _clean_arm(cand.get("arm", ""))
            if arm:
                labeled_arm_keys.add((sample_id, chrom_key, arm))
            else:
                labeled_whole_keys.add((sample_id, chrom_key))

        arm_candidates: list[dict[str, Any]] = []
        for bundle in bundles.values():
            df = bundle.wakhan_df
            if df.empty:
                continue
            for chrom, grp_raw in df.groupby("chrom", sort=False):
                chrom_text = str(chrom)
                chrom_key = (bundle.sample_id, _chrom_key(chrom_text))
                if chrom_key in labeled_whole_keys:
                    continue
                grp = grp_raw.copy()
                for arm, start_bp, end_bp in _chromosome_arm_bounds(grp):
                    if arm and (bundle.sample_id, _chrom_key(chrom_text), arm) in labeled_arm_keys:
                        continue
                    segs = grp[
                        (pd.to_numeric(grp["end"], errors="coerce") > start_bp)
                        & (pd.to_numeric(grp["start"], errors="coerce") < end_bp)
                    ].copy()
                    if segs.empty:
                        continue
                    suffix = f"_{arm}" if arm else ""
                    scope = "chromosome_arm" if arm else "chromosome"
                    evidence = "chromosome_arm_scan" if arm else "chromosome_scan"
                    arm_candidates.append(
                        {
                            "candidate_id": f"{bundle.sample_id}_{chrom_text}{suffix}",
                            "label_id": "",
                            "sample_id": bundle.sample_id,
                            "chrom": chrom_text,
                            "arm": arm,
                            "start_bp": int(start_bp),
                            "end_bp": int(end_bp),
                            "evidence": evidence,
                            "sv_node_indices": [],
                            "df_segments": segs,
                            "sv_class": "",
                            "label_scope": scope,
                            "candidate_scope": scope,
                        }
                    )
        return label_candidates + arm_candidates

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
    raise ValueError(f"candidate_source must be one of {', '.join(CANDIDATE_SOURCE_CHOICES)}")


def validate_candidate_resolution(candidates: list[dict[str, Any]], source: str) -> None:
    """Fail fast if arm-mode candidates fell back to whole-chromosome rows."""
    source = str(source or "").strip().replace("_", "-")
    if source != "chromosome-arms":
        return

    bad_rows: list[str] = []
    for cand in candidates:
        arm = _clean_arm(cand.get("arm", ""))
        evidence = str(cand.get("evidence", ""))
        sv_class = str(cand.get("sv_class", ""))
        scope = str(cand.get("candidate_scope", cand.get("label_scope", "")))
        label_scope = str(cand.get("label_scope", ""))
        is_labeled = bool(sv_class) or evidence == "label_anchor"
        is_scan = evidence in SCAN_EVIDENCE_VALUES and not sv_class

        if is_labeled and (scope == "chromosome_arm" or label_scope == "chromosome_arm") and arm not in {"p", "q"}:
            bad_rows.append(str(cand.get("candidate_id", cand.get("label_id", "label_anchor"))))
        if is_scan and (evidence != "chromosome_arm_scan" or scope != "chromosome_arm" or arm not in {"p", "q"}):
            bad_rows.append(str(cand.get("candidate_id", "scan_candidate")))

    if bad_rows:
        preview = ", ".join(bad_rows[:8])
        more = "" if len(bad_rows) <= 8 else f" (+{len(bad_rows) - 8} more)"
        raise ValueError(
            "candidate_source=chromosome-arms produced non-arm candidate rows. "
            f"Examples: {preview}{more}. Check label arm values and Wakhan chromosome spans."
        )


def _metadata_npz_arrays(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for col in metadata.columns:
        if metadata[col].dtype == object:
            arrays[col] = metadata[col].astype(str).to_numpy()
        else:
            arrays[col] = metadata[col].to_numpy()
    return arrays


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
    np.savez(npz_path, embeddings=embeddings, **_metadata_npz_arrays(metadata))
    return tsv_path, npz_path


def write_raw_embedding_outputs(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "embeddings_raw.npz"
    np.savez(npz_path, embeddings=embeddings, **_metadata_npz_arrays(metadata))
    return npz_path


def _l2_normalize_array(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def sample_residualize_embeddings(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    min_background: int = 3,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    if metadata.empty:
        return np.asarray(embeddings, dtype=np.float32), pd.DataFrame(), np.zeros((0, embeddings.shape[1]), dtype=np.float32)
    required = {"sample_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"Cannot sample-normalize embeddings; metadata missing columns: {missing}")

    raw = np.asarray(embeddings, dtype=np.float32)
    residual = raw.copy()
    labels = metadata["sv_class"].astype(str) if "sv_class" in metadata else pd.Series([""] * len(metadata))
    evidence = metadata["evidence"].astype(str) if "evidence" in metadata else pd.Series([""] * len(metadata))
    scan_mask = _scan_evidence_mask(evidence)
    samples = metadata["sample_id"].astype(str)
    rows: list[dict[str, Any]] = []
    baselines: list[np.ndarray] = []

    for sample_id in sorted(samples.unique()):
        sample_mask = (samples == sample_id).to_numpy()
        background_mask = (
            sample_mask
            & scan_mask
            & (labels.to_numpy() == "")
        )
        source = "unlabeled_scan"
        if int(background_mask.sum()) < int(min_background):
            background_mask = sample_mask & scan_mask
            source = "all_scan"
        if int(background_mask.sum()) < int(min_background):
            background_mask = sample_mask
            source = "all_sample_candidates"

        baseline = np.median(raw[background_mask], axis=0).astype(np.float32)
        sample_residual = raw[sample_mask] - baseline
        zero = np.linalg.norm(sample_residual, axis=1) < 1e-12
        if np.any(zero):
            sample_residual[zero] = raw[sample_mask][zero]
        residual[sample_mask] = sample_residual
        baselines.append(baseline)
        rows.append(
            {
                "sample_id": sample_id,
                "baseline_source": source,
                "n_baseline_candidates": int(background_mask.sum()),
                "n_sample_candidates": int(sample_mask.sum()),
                "baseline_norm": float(np.linalg.norm(baseline)),
            }
        )

    residual = _l2_normalize_array(residual)
    baseline_df = pd.DataFrame(rows)
    baseline_array = np.stack(baselines, axis=0).astype(np.float32) if baselines else np.zeros((0, raw.shape[1]), dtype=np.float32)
    return residual, baseline_df, baseline_array


def apply_embedding_normalization(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    mode: str,
    output_dir: str | Path,
    min_background: int = 3,
) -> np.ndarray:
    mode = str(mode or "none")
    if mode not in EMBEDDING_NORMALIZATION_CHOICES:
        choices = ", ".join(EMBEDDING_NORMALIZATION_CHOICES)
        raise ValueError(f"embedding_normalization must be one of: {choices}")
    if mode == "none":
        return np.asarray(embeddings, dtype=np.float32)

    if mode == "sample_residual":
        residual, baseline_df, baseline_array = sample_residualize_embeddings(
            embeddings,
            metadata,
            min_background=min_background,
        )
        out_dir = Path(output_dir)
        baseline_df.to_csv(out_dir / "sample_embedding_baselines.tsv", sep="\t", index=False)
        np.savez(
            out_dir / "sample_embedding_baselines.npz",
            sample_ids=baseline_df["sample_id"].astype(str).to_numpy(),
            baselines=baseline_array,
        )
        log.info(
            "Applied sample_residual embedding normalization using per-sample baselines: %s",
            out_dir / "sample_embedding_baselines.tsv",
        )
        return residual

    raise AssertionError(f"Unhandled embedding normalization mode: {mode}")


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


def _prediction_with_none(
    pred: str,
    confidence: float,
    distances: dict[str, float],
    tau: float,
) -> tuple[str, float, str, float]:
    if distances:
        nearest_class, nearest_distance = min(distances.items(), key=lambda item: item[1])
    else:
        nearest_class, nearest_distance = "", float("nan")
    if pred == "unknown" or not np.isfinite(nearest_distance) or nearest_distance >= float(tau):
        return "none", 0.0, nearest_class, nearest_distance
    return pred, confidence, nearest_class, nearest_distance


def write_distance_table(
    cache: PrototypeCache,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    output_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for i, result in enumerate(cache.classify_batch(torch.as_tensor(embeddings, dtype=torch.float32))):
        pred, confidence, distances = result
        pred, confidence, nearest_class, nearest_distance = _prediction_with_none(
            pred,
            confidence,
            distances,
            tau=cache.tau,
        )
        row = metadata.iloc[i].to_dict()
        row["predicted_class"] = pred
        row["prototype_confidence"] = confidence
        row["nearest_prototype_class"] = nearest_class
        row["nearest_prototype_distance"] = nearest_distance
        for name, dist in distances.items():
            row[f"d_{name}"] = dist
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, sep="\t", index=False)
    return df




def filter_report_scope(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    distances: pd.DataFrame,
    scope: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Filter report-only outputs without changing prototype construction."""
    scope = str(scope or "all")
    if scope not in REPORT_SCOPE_CHOICES:
        choices = ", ".join(REPORT_SCOPE_CHOICES)
        raise ValueError(f"report_scope must be one of: {choices}")
    if scope == "all":
        return embeddings, metadata.reset_index(drop=True), distances.reset_index(drop=True)
    if "sv_class" not in metadata:
        return embeddings[:0], metadata.iloc[:0].copy(), distances.iloc[:0].copy()
    mask = metadata["sv_class"].astype(str).str.strip().to_numpy() != ""
    return embeddings[mask], metadata.loc[mask].reset_index(drop=True), distances.loc[mask].reset_index(drop=True)

def write_leave_one_out(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    tau: float,
    output_path: str | Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels = metadata["sv_class"].astype(str).to_numpy()
    known_classes = sorted({label for label in labels if label})
    row_indices = np.arange(len(labels))

    for i, class_name in enumerate(labels):
        if not class_name:
            continue
        cache = PrototypeCache(embed_dim=int(embeddings.shape[1]), tau=tau)
        for support_class in known_classes:
            support_idx = np.where((labels == support_class) & (row_indices != i))[0]
            if support_idx.size == 0:
                continue
            cache.add_class(support_class, torch.as_tensor(embeddings[support_idx], dtype=torch.float32))
        if class_name not in cache.prototypes:
            continue
        pred, confidence, distances = cache.classify(torch.as_tensor(embeddings[i], dtype=torch.float32))
        pred, confidence, nearest_class, nearest_distance = _prediction_with_none(
            pred,
            confidence,
            distances,
            tau=cache.tau,
        )
        row = metadata.iloc[i].to_dict()
        row["held_out_class"] = class_name
        row["predicted_class"] = pred
        row["prototype_confidence"] = confidence
        row["nearest_prototype_class"] = nearest_class
        row["nearest_prototype_distance"] = nearest_distance
        row["leave_one_out_distance"] = distances.get(class_name, np.nan)
        row["leave_one_out_correct"] = pred == class_name
        for name, dist in distances.items():
            row[f"loo_d_{name}"] = dist
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


CLASS_COLOR_OVERRIDES = {
    "BFB": "#E15759",
    "chromothripsis": "#4E79A7",
    "seismic_amplification": "#F28E2B",
    "none": "#6C6C6C",
    "unknown": "#9C755F",
    "unlabeled": "#BAB0AC",
}


def _class_key(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null"}:
        return "unlabeled"
    return text


def _display_class(value: object) -> str:
    return _class_key(value).replace("_", " ")


def _split_class_set(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "null", "none"}:
        return []
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def _row_predicted_classes(row: pd.Series) -> list[str]:
    for column in ["predicted_raw_classes", "predicted_classes", "predicted_raw_class", "predicted_class"]:
        if column in row and str(row.get(column, "")).strip():
            values = _split_class_set(row.get(column, ""))
            if values:
                return values
    return []


def _prediction_label(row: pd.Series) -> str:
    classes = _row_predicted_classes(row)
    return ";".join(classes) if classes else "none"


def _label_colors(values: pd.Series) -> tuple[list[str], dict[str, object]]:
    labels = [_class_key(value) for value in values.tolist()]
    unique = list(dict.fromkeys(labels))
    fallback = plt.get_cmap("tab20")
    colors: dict[str, object] = {}
    fallback_i = 0
    for label in unique:
        if label in CLASS_COLOR_OVERRIDES:
            colors[label] = CLASS_COLOR_OVERRIDES[label]
        else:
            colors[label] = fallback(fallback_i % 20)
            fallback_i += 1
    return [colors[label] for label in labels], colors


def _candidate_label(row: pd.Series, include_class: bool = False, include_prediction: bool = False) -> str:
    value = str(row.get("candidate_id", "")) or str(row.get("label_id", ""))
    value = value[:36]
    details: list[str] = []
    if include_class:
        cls = str(row.get("sv_class", ""))
        if cls:
            details.append(f"gt={_display_class(cls)}")
    if include_prediction:
        pred = _prediction_label(row)
        if pred != "none":
            details.append(f"pred={_display_class(pred)}")
    if details:
        value = f"{value} [{'; '.join(details)}]"
    return value


def _short_label(row: pd.Series) -> str:
    return _candidate_label(row, include_class=False)[:32]


def _prototype_distance_label(row: pd.Series) -> str:
    return _candidate_label(row, include_class=bool(str(row.get("sv_class", "")).strip()))


def _dense_tick_fontsize(n_labels: int, max_size: float = 7.0, min_size: float = 3.0) -> float:
    if n_labels <= 0:
        return max_size
    if n_labels <= 90:
        return max_size
    return max(min_size, max_size * float(np.sqrt(90.0 / n_labels)))


def _known_anchor_mask(metadata: pd.DataFrame) -> np.ndarray:
    if "sv_class" not in metadata:
        return np.zeros(len(metadata), dtype=bool)
    return (metadata["sv_class"].astype(str) != "").to_numpy()


def _plot_embedding_projection(
    xy: np.ndarray,
    metadata: pd.DataFrame,
    color_source: pd.Series,
    title: str,
    output_path: Path,
    method: str,
) -> None:
    point_colors, legend_colors = _label_colors(color_source)
    known_mask = _known_anchor_mask(metadata)
    sizes = np.where(known_mask, 76, 40)
    edgecolors = np.where(known_mask, "black", "#777777")
    linewidths = np.where(known_mask, 0.8, 0.25)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=point_colors,
        s=sizes,
        alpha=0.88,
        linewidths=linewidths,
        edgecolors=edgecolors,
    )
    if len(metadata) <= 60:
        for i, row in metadata.reset_index(drop=True).iterrows():
            ax.annotate(_short_label(row), (xy[i, 0], xy[i, 1]), fontsize=7, xytext=(3, 3), textcoords="offset points")

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=_display_class(label),
            markerfacecolor=color,
            markeredgecolor="black" if label != "unlabeled" else "#777777",
            markersize=8,
        )
        for label, color in legend_colors.items()
    ]
    if handles:
        ax.legend(handles=handles, title="Class", fontsize=8, title_fontsize=9, loc="best")
    ax.set_title(f"{title} ({method})")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _distance_columns(distances: pd.DataFrame) -> list[str]:
    return sorted(
        [col for col in distances.columns if col.startswith("d_")],
        key=lambda col: _display_class(col.removeprefix("d_")).lower(),
    )


def _plot_prototype_distances(
    distances: pd.DataFrame,
    output_path: Path,
    tau: float,
) -> None:
    distance_cols = _distance_columns(distances)
    if not distance_cols:
        return

    plot_df = distances.copy()
    fallback = plot_df["label_id"].astype(str) if "label_id" in plot_df else pd.Series([""] * len(plot_df))
    plot_df["_plot_label"] = plot_df["candidate_id"].astype(str).mask(plot_df["candidate_id"].astype(str) == "", fallback)
    plot_df["_min_distance"] = plot_df[distance_cols].astype(float).min(axis=1)
    if "sv_class" not in plot_df:
        plot_df["sv_class"] = ""
    if "predicted_class" not in plot_df:
        plot_df["predicted_class"] = ""
    if "predicted_classes" not in plot_df:
        plot_df["predicted_classes"] = plot_df["predicted_class"].astype(str)
    plot_df["_is_unlabeled"] = plot_df["sv_class"].astype(str) == ""
    plot_df["_true_sort"] = plot_df["sv_class"].astype(str).replace("", "unlabeled")
    plot_df["_pred_sort"] = plot_df.apply(_prediction_label, axis=1)
    plot_df = plot_df.sort_values(["_is_unlabeled", "_true_sort", "_pred_sort", "_min_distance", "_plot_label"]).reset_index(drop=True)

    if len(distance_cols) == 1:
        primary_col = distance_cols[0]
        fig_h = max(4.0, min(18.0, 0.30 * len(plot_df) + 1.5))
        fig, ax = plt.subplots(figsize=(8, fig_h))
        bar_colors, _legend = _label_colors(plot_df["sv_class"].astype(str))
        ax.barh(np.arange(len(plot_df)), plot_df[primary_col].astype(float), color=bar_colors)
        ax.set_yticks(np.arange(len(plot_df)))
        ax.set_yticklabels(
            [_candidate_label(row, include_class=True) for _, row in plot_df.iterrows()],
            fontsize=_dense_tick_fontsize(len(plot_df), max_size=8.0),
        )
        ax.invert_yaxis()
        ax.axvline(x=tau, color="#E45756", linestyle="--", linewidth=1.2, label=f"tau={tau:g}")
        ax.set_xlabel("Cosine distance to prototype")
        ax.set_title(f"Prototype Distance ({_display_class(primary_col.removeprefix('d_'))})")
        ax.legend()
        ax.grid(axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_path, dpi=180)
        plt.close(fig)
        return

    matrix = plot_df[distance_cols].astype(float).to_numpy()
    matrix_masked = np.ma.masked_invalid(matrix)
    finite_vals = matrix[np.isfinite(matrix)]
    vmax = float(finite_vals.max()) if finite_vals.size else 1.0
    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="#d0d0d0")
    fig_h = max(5.0, min(22.0, 0.24 * len(plot_df) + 2.0))
    fig_w = max(6.5, 1.2 * len(distance_cols) + 4.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(matrix_masked, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    pred_to_col = {col.removeprefix("d_"): i for i, col in enumerate(distance_cols)}
    star_x: list[int] = []
    star_y: list[int] = []
    for row_i, row in plot_df.iterrows():
        for pred in _row_predicted_classes(row):
            if pred in pred_to_col:
                star_x.append(pred_to_col[pred])
                star_y.append(int(row_i))
    if star_x:
        ax.scatter(star_x, star_y, marker="*", s=45, color="white", edgecolors="black", linewidths=0.4)
    ax.set_xticks(np.arange(len(distance_cols)))
    ax.set_xticklabels([_display_class(col.removeprefix("d_")) for col in distance_cols], rotation=30, ha="right")

    y_ticks = np.arange(len(plot_df))
    y_labels = [_prototype_distance_label(row) for _, row in plot_df.iterrows()]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=_dense_tick_fontsize(len(plot_df)))
    _draw_gt_class_boundaries(ax, plot_df)
    ax.set_xlabel("Prototype class")
    ax.set_title(f"Prototype Distances by Class (tau={tau:g})\nstars = predicted class; no star = none")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Distance (gray = other family, not computed)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _draw_gt_class_boundaries(ax, plot_df: pd.DataFrame) -> None:
    if "sv_class" not in plot_df or plot_df.empty:
        return
    classes = plot_df["sv_class"].astype(str).tolist()
    labeled_positions = [i for i, cls in enumerate(classes) if cls]
    if not labeled_positions:
        return

    group_start = labeled_positions[0]
    prev_cls = classes[group_start]
    for pos in labeled_positions[1:] + [labeled_positions[-1] + 1]:
        cls = classes[pos] if pos < len(classes) else None
        if cls != prev_cls:
            group_end = pos - 1
            ax.axhline(group_end + 0.5, color="white", linewidth=2.2)
            group_start = pos
            prev_cls = cls


def _plot_leave_one_out(
    leave_one_out: pd.DataFrame,
    output_path: Path,
    tau: float,
) -> None:
    if leave_one_out.empty or "leave_one_out_distance" not in leave_one_out:
        return
    plot_df = leave_one_out.sort_values(["held_out_class", "leave_one_out_distance"], ascending=[True, True]).reset_index(drop=True)
    labels = [_candidate_label(row, include_class=True) for _, row in plot_df.iterrows()]
    bar_colors, legend_colors = _label_colors(plot_df["held_out_class"].astype(str))
    fig_h = max(4.0, 0.32 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8.5, fig_h))
    ax.barh(np.arange(len(plot_df)), plot_df["leave_one_out_distance"].astype(float), color=bar_colors)
    ax.set_yticks(np.arange(len(plot_df)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(x=tau, color="#E45756", linestyle="--", linewidth=1.2, label=f"tau={tau:g}")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=_display_class(label), markerfacecolor=color, markersize=8)
        for label, color in legend_colors.items()
    ]
    if handles:
        ax.legend(handles=handles, title="Held-out class", fontsize=8, title_fontsize=9)
    ax.set_xlabel("Held-out distance to true-class prototype")
    ax.set_title("Leave-One-Out Prototype Distances by Class")
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _bool_flag(value: object) -> bool | None:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y"}:
        return True
    if text in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _summary_class_order(values: pd.Series) -> list[str]:
    unique = list(dict.fromkeys([str(value).strip() for value in values.tolist() if str(value).strip()]))
    none_values = [value for value in unique if value.lower() == "none"]
    class_values = [value for value in unique if value.lower() != "none"]
    class_values = sorted(class_values, key=lambda value: _display_class(value).lower())
    return class_values + none_values


def _summary_gt_class(row: pd.Series) -> str:
    for column in ["raw_sv_classes", "raw_true_classes", "sv_class", "true_classes"]:
        classes = _split_class_set(row.get(column, ""))
        if classes:
            return ";".join(classes)

    evidence = str(row.get("evidence", "")).strip()
    label_scope = str(row.get("label_scope", "")).strip()
    is_background = _bool_flag(row.get("is_background_chromosome", ""))
    if is_background is True or evidence == "candidate_region_empty" or label_scope == "empty_candidate_region":
        return "none"
    return ""


def _summary_class_set(label: object) -> set[str]:
    text = str(label).strip()
    if text.lower() == "none":
        return set()
    return set(_split_class_set(text))


def _summary_base_class_set(label: object) -> set[str]:
    return {part.split(":", 1)[0].strip() for part in _summary_class_set(label) if part.split(":", 1)[0].strip()}


def _summary_base_label(label: object) -> str:
    text = str(label).strip()
    if not text or text.lower() == "none":
        return "none"
    bases: list[str] = []
    for part in _split_class_set(text):
        base = part.split(":", 1)[0].strip()
        if base and base not in bases:
            bases.append(base)
    return ";".join(bases) if bases else "none"


def _summary_correctness_category(true_label: object, pred_label: object) -> str:
    true_raw = _summary_class_set(true_label)
    pred_raw = _summary_class_set(pred_label)
    true_base = _summary_base_class_set(true_label)
    pred_base = _summary_base_class_set(pred_label)
    if true_raw == pred_raw:
        return "exact"
    if true_base == pred_base and true_base:
        return "wrong_level"
    if bool(true_base & pred_base):
        return "partial_class"
    return "wrong"


def _plot_anchor_prediction_summary(
    distances: pd.DataFrame,
    output_path: Path,
    tau: float | None,
    title_suffix: str = "",
) -> None:
    if distances.empty or "predicted_class" not in distances:
        return
    gt = distances.copy()
    gt["_gt_raw_class"] = gt.apply(_summary_gt_class, axis=1)
    gt = gt[gt["_gt_raw_class"].astype(str) != ""].copy()
    if gt.empty:
        return

    gt["_gt_class"] = gt["_gt_raw_class"].map(_summary_base_label)
    gt["_predicted_raw_class"] = gt.apply(_prediction_label, axis=1)
    gt["_predicted_class"] = gt["_predicted_raw_class"].map(_summary_base_label)
    gt["correctness_category"] = [
        _summary_correctness_category(true_value, pred_value)
        for true_value, pred_value in zip(gt["_gt_raw_class"].tolist(), gt["_predicted_raw_class"].tolist())
    ]
    class_order = _summary_class_order(gt["_gt_class"].astype(str))
    category_order = [
        ("exact", "Exact", "#59A14F"),
        ("wrong_level", "Wrong subtype", "#F28E2B"),
        ("partial_class", "Partial class", "#9467BD"),
        ("wrong", "Wrong", "#E15759"),
    ]
    category_counts = {
        category: [int(gt[(gt["_gt_class"] == cls) & (gt["correctness_category"] == category)].shape[0]) for cls in class_order]
        for category, _label, _color in category_order
    }
    category_order = [
        (category, label, color)
        for category, label, color in category_order
        if sum(category_counts.get(category, [])) > 0
    ]

    pred_order = list(class_order)
    for pred in _summary_class_order(gt["_predicted_class"].astype(str)):
        if pred not in pred_order:
            pred_order.append(pred)
    confusion = pd.crosstab(gt["_gt_class"].astype(str), gt["_predicted_class"].astype(str))
    confusion = confusion.reindex(index=class_order, columns=pred_order, fill_value=0)
    display_class_order = [_display_class(cls).replace(";", "\n") for cls in class_order]
    display_pred_order = [_display_class(cls).replace(";", "\n") for cls in pred_order]

    fig_w = max(10.5, 1.2 * max(len(class_order), len(pred_order)) + 6.0)
    fig, axes = plt.subplots(1, 2, figsize=(fig_w, 5.4), gridspec_kw={"width_ratios": [1.0, 1.25]})

    suffix = f"\n{title_suffix}" if title_suffix else ""
    x = np.arange(len(class_order))
    bottom = np.zeros(len(class_order), dtype=float)
    for category, label, color in category_order:
        counts = np.asarray(category_counts[category], dtype=float)
        axes[0].bar(x, counts, bottom=bottom, color=color, label=label)
        bottom += counts
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [f"{label}\n(n={int(total)})" for label, total in zip(display_class_order, bottom, strict=False)],
        rotation=20,
        ha="right",
        fontsize=9,
    )
    axes[0].set_ylabel("GT region count")
    tau_suffix = f" (tau={tau:g})" if tau is not None else ""
    axes[0].set_title(f"Prediction outcome by GT class{tau_suffix}{suffix}", fontsize=12, pad=10)
    axes[0].tick_params(axis="y", labelsize=9)
    axes[0].set_axisbelow(True)
    axes[0].grid(axis="y", alpha=0.18)
    axes[0].spines[["top", "right"]].set_visible(False)

    matrix = confusion.to_numpy(dtype=int)
    im = axes[1].imshow(matrix, cmap="Blues", vmin=0)
    axes[1].set_xticks(np.arange(len(pred_order)))
    axes[1].set_yticks(np.arange(len(class_order)))
    axes[1].set_xticklabels(display_pred_order, rotation=20, ha="right", fontsize=8.5)
    axes[1].set_yticklabels(display_class_order, fontsize=8.5)
    axes[1].set_xlabel("Predicted class")
    axes[1].set_ylabel("GT class")
    axes[1].set_title(f"GT vs predicted class{suffix}", fontsize=12, pad=10)
    for y in range(matrix.shape[0]):
        for x_i in range(matrix.shape[1]):
            value = int(matrix[y, x_i])
            if value:
                axes[1].text(x_i, y, str(value), ha="center", va="center", color="black", fontsize=9)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=min(4, len(labels)),
        loc="lower center",
        bbox_to_anchor=(0.31, 0.015),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _positive_negative_tau_frame(
    distances: pd.DataFrame,
    leave_one_out: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not leave_one_out.empty and "nearest_prototype_distance" in leave_one_out:
        for _, row in leave_one_out.iterrows():
            dist = pd.to_numeric(pd.Series([row.get("nearest_prototype_distance")]), errors="coerce").iloc[0]
            if np.isfinite(dist):
                rows.append(
                    {
                        "candidate_id": row.get("candidate_id", ""),
                        "label": 1,
                        "set": "gt_label_leave_one_out",
                        "distance": float(dist),
                    }
                )

    if not distances.empty and "nearest_prototype_distance" in distances:
        if "sv_class" in distances:
            neg = distances[distances["sv_class"].astype(str) == ""].copy()
        else:
            neg = distances.copy()
        if "evidence" in neg:
            scan = neg["evidence"].astype(str).isin(SCAN_EVIDENCE_VALUES)
            if scan.any():
                neg = neg[scan]
        for _, row in neg.iterrows():
            dist = pd.to_numeric(pd.Series([row.get("nearest_prototype_distance")]), errors="coerce").iloc[0]
            if np.isfinite(dist):
                rows.append(
                    {
                        "candidate_id": row.get("candidate_id", ""),
                        "label": 0,
                        "set": "unlabeled_scan_candidate",
                        "distance": float(dist),
                    }
                )

    return pd.DataFrame(rows)


def _tau_precision_recall_table(
    score_df: pd.DataFrame,
    extra_thresholds: list[float] | None = None,
) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    labels = score_df["label"].astype(int).to_numpy()
    distances = score_df["distance"].astype(float).to_numpy()
    n_pos = int(labels.sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return pd.DataFrame()

    finite_distances = np.unique(distances[np.isfinite(distances)])
    if finite_distances.size == 0:
        return pd.DataFrame()

    thresholds = [0.0, float(finite_distances.max() + 1e-6)]
    if finite_distances.size == 1:
        thresholds.append(float(finite_distances[0] + 1e-6))
    else:
        mids = (finite_distances[:-1] + finite_distances[1:]) / 2.0
        thresholds.extend(float(value) for value in mids)
    if extra_thresholds:
        thresholds.extend(float(value) for value in extra_thresholds if np.isfinite(value))
    thresholds = np.asarray(sorted(set(thresholds)), dtype=float)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        predicted = distances < threshold
        tp = int(((labels == 1) & predicted).sum())
        fp = int(((labels == 0) & predicted).sum())
        fn = int(((labels == 1) & ~predicted).sum())
        tn = int(((labels == 0) & ~predicted).sum())
        precision = float(tp / (tp + fp)) if (tp + fp) else 1.0
        recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        f1 = float((2 * precision * recall) / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "tau": float(threshold),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )
    return pd.DataFrame(rows)


def _plot_tau_precision_recall(
    distances: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    output_path: Path,
    tau: float,
) -> None:
    score_df = _positive_negative_tau_frame(distances, leave_one_out)
    pr_df = _tau_precision_recall_table(score_df, extra_thresholds=[tau])
    if pr_df.empty:
        return

    table_path = output_path.with_suffix(".tsv")
    pr_df.to_csv(table_path, sep="\t", index=False)

    best_idx = int(pr_df["f1"].astype(float).idxmax())
    best = pr_df.loc[best_idx]
    current_idx = int((pr_df["tau"].astype(float) - float(tau)).abs().idxmin())
    current = pr_df.loc[current_idx]
    n_pos = int(score_df["label"].astype(int).sum())
    n_neg = int((score_df["label"].astype(int) == 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))

    scatter = axes[0].scatter(
        pr_df["recall"],
        pr_df["precision"],
        c=pr_df["tau"],
        cmap="viridis",
        s=32,
        edgecolors="none",
    )
    axes[0].scatter(
        [best["recall"]],
        [best["precision"]],
        marker="*",
        s=150,
        color="#E15759",
        edgecolors="black",
        linewidths=0.5,
        label=f"best F1 tau={best['tau']:.4g}",
    )
    axes[0].scatter(
        [current["recall"]],
        [current["precision"]],
        marker="o",
        s=70,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label=f"current tau={tau:g}",
    )
    axes[0].set_xlabel("Recall on GT labels")
    axes[0].set_ylabel("Precision vs unlabeled scan candidates")
    axes[0].set_xlim(-0.03, 1.03)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title("Tau Precision-Recall")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.04, label="tau")

    axes[1].plot(pr_df["tau"], pr_df["precision"], label="Precision", color="#4E79A7")
    axes[1].plot(pr_df["tau"], pr_df["recall"], label="Recall", color="#59A14F")
    axes[1].plot(pr_df["tau"], pr_df["f1"], label="F1", color="#E15759")
    axes[1].axvline(float(best["tau"]), color="#E15759", linestyle="--", linewidth=1.0)
    axes[1].axvline(float(tau), color="black", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("tau distance threshold")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title(
        f"Best F1={best['f1']:.2f} (P={best['precision']:.2f}, R={best['recall']:.2f})\n"
        f"{n_pos} GT positives, {n_neg} other scan candidates"
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8, loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    true_classes = metadata["sv_class"] if "sv_class" in metadata else pd.Series([""] * len(metadata))
    _plot_embedding_projection(
        xy,
        metadata,
        true_classes,
        "Candidate Embedding Projection - True Labels",
        out_dir / "embedding_projection.png",
        method,
    )
    if "predicted_class" in distances:
        predicted_color_source = distances["predicted_classes"] if "predicted_classes" in distances else distances["predicted_class"]
        _plot_embedding_projection(
            xy,
            metadata,
            predicted_color_source,
            "Candidate Embedding Projection - Predicted Classes",
            out_dir / "embedding_projection_predicted.png",
            method,
        )

    _plot_prototype_distances(distances, out_dir / "prototype_distances.png", tau=tau)
    _plot_leave_one_out(leave_one_out, out_dir / "anchor_leave_one_out.png", tau=tau)
    _plot_anchor_prediction_summary(distances, out_dir / "anchor_prediction_summary.png", tau=tau)
    if "split" in distances:
        split_values = distances["split"].astype(str).str.lower()
        for split_name in ["train", "test"]:
            split_df = distances.loc[split_values == split_name].copy()
            _plot_anchor_prediction_summary(
                split_df,
                out_dir / f"anchor_prediction_summary_{split_name}.png",
                tau=tau,
                title_suffix=split_name,
            )
    _plot_tau_precision_recall(distances, leave_one_out, out_dir / "tau_precision_recall.png", tau=tau)

    known_mask = metadata["sv_class"].astype(str) != "" if "sv_class" in metadata else pd.Series([False] * len(metadata))
    known_meta = metadata.loc[known_mask].copy()
    if len(known_meta) >= 2 and len(known_meta) <= 80:
        known_meta["_orig_idx"] = np.where(known_mask.to_numpy())[0]
        known_meta = known_meta.sort_values(["sv_class", "candidate_id"]).reset_index(drop=True)
        known_idx = known_meta["_orig_idx"].astype(int).to_numpy()
        known_emb = embeddings[known_idx]
        known_emb = known_emb / np.clip(np.linalg.norm(known_emb, axis=1, keepdims=True), 1e-12, None)
        sim = known_emb @ known_emb.T
        labels = [_candidate_label(row, include_class=True) for _, row in known_meta.iterrows()]
        fig_size = max(5.5, min(12.0, 0.45 * len(labels) + 3.0))
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(sim, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_title("Known-Anchor Cosine Similarity by Class")
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
    validate_candidate_resolution(candidates, args.candidate_source)
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
    raw_emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    meta_df = pd.DataFrame(meta_rows)
    normalization_mode = getattr(args, "embedding_normalization", "none")
    meta_df["embedding_normalization"] = normalization_mode
    log.info("Writing raw embedding table and applying embedding_normalization=%s", normalization_mode)
    write_raw_embedding_outputs(raw_emb_array, meta_df, output_dir)
    emb_array = apply_embedding_normalization(
        raw_emb_array,
        meta_df,
        mode=normalization_mode,
        output_dir=output_dir,
        min_background=int(getattr(args, "sample_baseline_min_candidates", 3)),
    )
    log.info("Writing embedding tables and prototype outputs")
    write_embedding_outputs(emb_array, meta_df, output_dir)

    proto_path = output_dir / args.prototypes_name
    cache = build_prototypes(emb_array, meta_df, tau=args.tau, output_path=proto_path)
    tmp_distances_path = output_dir / ".prototype_distances_all.tmp.tsv"
    distances_all = write_distance_table(cache, emb_array, meta_df, tmp_distances_path)
    report_emb, report_meta, report_distances = filter_report_scope(
        emb_array,
        meta_df,
        distances_all,
        scope=getattr(args, "report_scope", "all"),
    )
    report_distances.to_csv(output_dir / "prototype_distances.tsv", sep="\t", index=False)
    try:
        tmp_distances_path.unlink()
    except FileNotFoundError:
        pass
    leave_one_out = write_leave_one_out(emb_array, meta_df, tau=args.tau, output_path=output_dir / "anchor_leave_one_out.tsv")
    log.info(
        "Writing visualization PNGs with report_scope=%s (%d/%d rows)",
        getattr(args, "report_scope", "all"),
        len(report_meta),
        len(meta_df),
    )
    write_visualizations(report_emb, report_meta, report_distances, leave_one_out, output_dir, tau=args.tau)
    log.info("Wrote %d embedding(s), %d report row(s), %d prototype class(es)", len(meta_df), len(report_meta), cache.n_classes())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Sample manifest TSV")
    parser.add_argument("--labels", required=False, help="Label TSV with anchor intervals")
    parser.add_argument("--cn_checkpoint", required=True, help="Pretrained CN checkpoint")
    parser.add_argument("--graph_checkpoint", required=False, help="Pretrained graph checkpoint")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--candidate_source",
        choices=CANDIDATE_SOURCE_CHOICES,
        default="labels",
        help=(
            "Region source: labels uses only labeled intervals; chromosomes uses labeled "
            "intervals plus one unlabeled whole-chromosome candidate for each unlabeled "
            "manifest sample/chromosome; chromosome-arms uses labeled intervals plus "
            "unlabeled p/q arm candidates; proposals uses heuristic CN/SV windows; all "
            "uses labels plus heuristic proposals."
        ),
    )
    parser.add_argument("--prototypes_name", default="prototypes.pt")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument(
        "--embedding_normalization",
        choices=EMBEDDING_NORMALIZATION_CHOICES,
        default="none",
        help="Optional post-encoder normalization before prototype building/scoring.",
    )
    parser.add_argument(
        "--report_scope",
        choices=REPORT_SCOPE_CHOICES,
        default="all",
        help=(
            "Rows to include in report tables/plots. Use anchors when candidate_source=chromosomes "
            "is only needed for sample-residual baselines but anchor-stage reports should stay label-only."
        ),
    )
    parser.add_argument(
        "--sample_baseline_min_candidates",
        type=int,
        default=3,
        help="Minimum same-sample background candidates for sample_residual baselines before falling back.",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict checkpoint loading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
