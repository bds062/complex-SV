"""Stage 3 candidate-region proposal from Wakhan CN and Severus graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None  # type: ignore

try:
    import community as community_louvain
except ImportError:  # pragma: no cover
    community_louvain = None  # type: ignore

try:
    from config import RegionProposalConfig
    from utils import reciprocal_overlap
except ImportError:  # pragma: no cover
    from ..config import RegionProposalConfig  # type: ignore
    from ..utils import reciprocal_overlap  # type: ignore

from .graph_builder import EDGE_MATE, EDGE_PHASE, EDGE_PROXIMITY


def _chrom_equal(a: object, b: object) -> bool:
    aa = str(a).removeprefix("chr")
    bb = str(b).removeprefix("chr")
    return aa == bb


def _cn_state_count(values: pd.Series, tolerance: float) -> int:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return 0
    bins = np.round(vals / max(float(tolerance), 1e-6)).astype(int)
    return int(np.unique(bins).size)


def _oscillation_score(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size < 3:
        return 0.0
    diffs = np.diff(vals)
    signs = np.sign(diffs[np.abs(diffs) > 1e-6])
    if signs.size < 2:
        return 0.0
    flips = np.sum(signs[1:] != signs[:-1])
    return float(flips / max(signs.size - 1, 1))


def _candidate_from_segments(sample_id: str, chrom: str, segs: pd.DataFrame, evidence: str) -> dict[str, Any]:
    start_bp = int(pd.to_numeric(segs["start"], errors="coerce").min())
    end_bp = int(pd.to_numeric(segs["end"], errors="coerce").max())
    span_mb = max((end_bp - start_bp) / 1_000_000.0, 1e-9)
    return {
        "sample_id": str(sample_id),
        "chrom": str(chrom),
        "start_bp": start_bp,
        "end_bp": end_bp,
        "evidence": evidence,
        "sv_node_indices": [],
        "df_segments": segs.copy(),
        "n_segments": int(len(segs)),
        "cn_states": _cn_state_count(segs["cn_total"], 0.3),
        "cn_oscillation_score": _oscillation_score(segs["cn_total"]),
        "loh_fraction": float(pd.to_numeric(segs["loh"], errors="coerce").fillna(0).mean()),
        "mean_imbalance": float(pd.to_numeric(segs["allele_imbalance"], errors="coerce").fillna(0).mean()),
        "breakpoint_density_per_10mb": float(
            pd.to_numeric(segs["breakpoint_count"], errors="coerce").fillna(0).sum() / span_mb * 10.0
        ),
    }


def _merge_overlapping(records: list[dict[str, Any]], overlap_threshold: float = 0.5) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        by_key[(str(rec["sample_id"]), str(rec["chrom"]))].append(rec)

    for (_sample_id, _chrom), group in by_key.items():
        group = sorted(group, key=lambda r: (int(r["start_bp"]), int(r["end_bp"])))
        current: dict[str, Any] | None = None
        for rec in group:
            if current is None:
                current = rec.copy()
                continue
            ov = reciprocal_overlap(
                int(current["start_bp"]),
                int(current["end_bp"]),
                int(rec["start_bp"]),
                int(rec["end_bp"]),
            )
            if ov >= overlap_threshold:
                current["start_bp"] = min(int(current["start_bp"]), int(rec["start_bp"]))
                current["end_bp"] = max(int(current["end_bp"]), int(rec["end_bp"]))
                current["df_segments"] = pd.concat([current["df_segments"], rec["df_segments"]]).drop_duplicates()
            else:
                merged.append(current)
                current = rec.copy()
        if current is not None:
            merged.append(current)
    return merged


def propose_cn_candidates(
    df_wakhan_sample: pd.DataFrame,
    min_segments: int = 6,
    max_cn_states: int = 3,
    cn_tolerance: float = 0.3,
    min_breakpoints_per_10mb: float = 5.0,
    min_loh_or_imbalance: float = 0.4,
) -> list[dict[str, Any]]:
    """Scan each chromosome for broad CN oscillation candidate intervals."""
    if df_wakhan_sample.empty:
        return []
    required = {"sample_id", "chrom", "start", "end", "cn_total", "loh", "allele_imbalance", "breakpoint_count"}
    missing = sorted(required.difference(df_wakhan_sample.columns))
    if missing:
        raise ValueError(f"Wakhan candidate proposal missing columns: {missing}")

    records: list[dict[str, Any]] = []
    sample_id = str(df_wakhan_sample["sample_id"].iloc[0])
    for chrom, grp_raw in df_wakhan_sample.groupby("chrom", sort=False):
        grp = grp_raw.sort_values(["start", "end"]).reset_index(drop=True)
        if len(grp) < int(min_segments):
            continue

        max_window = min(max(int(min_segments) * 4, int(min_segments)), len(grp))
        for left in range(0, len(grp) - int(min_segments) + 1):
            best: dict[str, Any] | None = None
            for right in range(left + int(min_segments), min(len(grp), left + max_window) + 1):
                win = grp.iloc[left:right]
                start_bp = int(win["start"].min())
                end_bp = int(win["end"].max())
                span_mb = max((end_bp - start_bp) / 1_000_000.0, 1e-9)
                cn_states = _cn_state_count(win["cn_total"], cn_tolerance)
                bp_density = float(pd.to_numeric(win["breakpoint_count"], errors="coerce").fillna(0).sum() / span_mb * 10.0)
                loh_change = pd.to_numeric(win["loh"], errors="coerce").fillna(0).nunique() > 1
                high_imbalance = float(pd.to_numeric(win["allele_imbalance"], errors="coerce").fillna(0).max()) > min_loh_or_imbalance
                oscillating = _oscillation_score(win["cn_total"]) > 0.25
                if (
                    cn_states <= int(max_cn_states)
                    and bp_density >= float(min_breakpoints_per_10mb)
                    and (loh_change or high_imbalance)
                    and oscillating
                ):
                    best = _candidate_from_segments(sample_id, str(chrom), win, "cn_only")
            if best is not None:
                records.append(best)

    return _merge_overlapping(records)


def _edge_pairs(graph, edge_type: tuple[str, str, str]) -> list[tuple[int, int]]:
    if not hasattr(graph[edge_type], "edge_index"):
        return []
    edge_index = graph[edge_type].edge_index
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.detach().cpu()
    return [(int(edge_index[0, i]), int(edge_index[1, i])) for i in range(edge_index.shape[1])]


def propose_graph_candidates(
    df_severus_sample: pd.DataFrame,
    graph,
    proximity_bp: int = 500_000,
    min_junctions: int = 3,
    min_span_bp: int = 1_000_000,
) -> list[dict[str, Any]]:
    """Find densely connected same-chromosome SV communities."""
    if graph is None or df_severus_sample.empty or nx is None:
        return []

    df = df_severus_sample.reset_index(drop=True).copy()
    sample_id = str(df["sample_id"].iloc[0]) if "sample_id" in df else str(getattr(graph, "sample_id", ""))
    candidates: list[dict[str, Any]] = []

    for chrom, chrom_df in df.groupby("chrom", sort=False):
        node_ids = chrom_df.index.astype(int).tolist()
        node_set = set(node_ids)
        g = nx.Graph()
        g.add_nodes_from(node_ids)

        for edge_type in (EDGE_PROXIMITY, EDGE_MATE, EDGE_PHASE):
            for src, dst in _edge_pairs(graph, edge_type):
                if src == dst or src not in node_set or dst not in node_set:
                    continue
                if edge_type == EDGE_PROXIMITY:
                    dist = abs(int(df.loc[src, "pos"]) - int(df.loc[dst, "pos"]))
                    if dist > int(proximity_bp):
                        continue
                g.add_edge(src, dst)

        if g.number_of_edges() == 0:
            continue

        if community_louvain is not None:
            partition = community_louvain.best_partition(g)
            communities: dict[int, list[int]] = defaultdict(list)
            for node_id, comm_id in partition.items():
                communities[int(comm_id)].append(int(node_id))
            groups = list(communities.values())
        else:
            groups = [list(comp) for comp in nx.connected_components(g)]

        for nodes in groups:
            if len(nodes) < int(min_junctions):
                continue
            starts = pd.to_numeric(df.loc[nodes, "pos"], errors="coerce")
            ends = pd.to_numeric(df.loc[nodes, "end"], errors="coerce")
            start_bp = int(starts.min())
            end_bp = int(ends.max())
            if end_bp - start_bp < int(min_span_bp):
                continue
            candidates.append(
                {
                    "sample_id": sample_id,
                    "chrom": str(chrom),
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "evidence": "graph_only",
                    "sv_node_indices": sorted(int(n) for n in nodes),
                    "df_segments": pd.DataFrame(),
                    "n_sv": int(len(nodes)),
                }
            )

    return candidates


def _sv_indices_in_interval(df_severus_sample: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> list[int]:
    if df_severus_sample.empty:
        return []
    observed = df_severus_sample["chrom"].astype(str)
    chrom_mask = observed.map(lambda c: _chrom_equal(c, chrom))
    pos = pd.to_numeric(df_severus_sample["pos"], errors="coerce")
    mask = chrom_mask & (pos >= int(start_bp)) & (pos < int(end_bp))
    return df_severus_sample.index[mask].astype(int).tolist()


def merge_candidates(
    cn_candidates: list[dict[str, Any]],
    graph_candidates: list[dict[str, Any]],
    overlap_threshold: float = 0.5,
    df_severus_sample: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Merge CN and graph candidates by reciprocal overlap."""
    used_graph: set[int] = set()
    merged: list[dict[str, Any]] = []
    for cn in cn_candidates:
        out = cn.copy()
        for i, graph in enumerate(graph_candidates):
            if i in used_graph:
                continue
            if str(cn["sample_id"]) != str(graph["sample_id"]) or not _chrom_equal(cn["chrom"], graph["chrom"]):
                continue
            ov = reciprocal_overlap(
                int(cn["start_bp"]),
                int(cn["end_bp"]),
                int(graph["start_bp"]),
                int(graph["end_bp"]),
            )
            if ov < overlap_threshold:
                continue
            used_graph.add(i)
            out["start_bp"] = min(int(out["start_bp"]), int(graph["start_bp"]))
            out["end_bp"] = max(int(out["end_bp"]), int(graph["end_bp"]))
            out["evidence"] = "both"
            out["sv_node_indices"] = sorted(set(out.get("sv_node_indices", [])) | set(graph.get("sv_node_indices", [])))
        if df_severus_sample is not None and not out.get("sv_node_indices"):
            out["sv_node_indices"] = _sv_indices_in_interval(
                df_severus_sample,
                str(out["chrom"]),
                int(out["start_bp"]),
                int(out["end_bp"]),
            )
        merged.append(out)

    for i, graph in enumerate(graph_candidates):
        if i not in used_graph:
            merged.append(graph)

    return sorted(merged, key=lambda r: (str(r["sample_id"]), str(r["chrom"]), int(r["start_bp"])))


def label_rows_to_candidates(labels: pd.DataFrame, wakhan_by_sample: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    """Turn validated label rows into forced candidate intervals."""
    candidates: list[dict[str, Any]] = []
    for row in labels.to_dict("records"):
        sample_id = str(row["sample_id"])
        chrom = str(row["chrom"])
        start_bp = int(row["start_bp"])
        end_bp = int(row["end_bp"])
        df_w = wakhan_by_sample.get(sample_id, pd.DataFrame())
        if not df_w.empty:
            mask = (
                df_w["chrom"].astype(str).map(lambda c: _chrom_equal(c, chrom))
                & (pd.to_numeric(df_w["end"], errors="coerce") > start_bp)
                & (pd.to_numeric(df_w["start"], errors="coerce") < end_bp)
            )
            segs = df_w.loc[mask].copy()
        else:
            segs = pd.DataFrame()
        candidates.append(
            {
                "candidate_id": str(row.get("label_id", "")),
                "label_id": str(row.get("label_id", "")),
                "sample_id": sample_id,
                "chrom": chrom,
                "start_bp": start_bp,
                "end_bp": end_bp,
                "evidence": "label_anchor",
                "sv_node_indices": [],
                "df_segments": segs,
                "sv_class": str(row.get("sv_class", "")),
                "label_scope": str(row.get("label_scope", "")),
            }
        )
    return candidates


def propose_all(
    df_wakhan: pd.DataFrame,
    df_severus: pd.DataFrame,
    graphs: list,
    sample_ids: list[str],
    **kwargs,
) -> list[dict[str, Any]]:
    """Run CN and graph candidate proposal across all samples."""
    cfg = RegionProposalConfig()
    params = {**asdict(cfg), **kwargs}
    graph_by_sample = {str(sample_id): graph for sample_id, graph in zip(sample_ids, graphs)}
    all_candidates: list[dict[str, Any]] = []

    sample_order = sorted(set(df_wakhan.get("sample_id", pd.Series(dtype=str)).astype(str)))
    for sample_id in sample_order:
        df_w = df_wakhan[df_wakhan["sample_id"].astype(str) == sample_id].copy()
        df_s = df_severus[df_severus["sample_id"].astype(str) == sample_id].copy().reset_index(drop=True)
        cn = propose_cn_candidates(
            df_w,
            min_segments=params["min_segments"],
            max_cn_states=params["max_cn_states"],
            cn_tolerance=params["cn_tolerance"],
            min_breakpoints_per_10mb=params["min_breakpoints_per_10mb"],
            min_loh_or_imbalance=params["min_loh_or_imbalance"],
        )
        graph = propose_graph_candidates(
            df_s,
            graph_by_sample.get(sample_id),
            proximity_bp=params["graph_proximity_bp"],
            min_junctions=params["min_junctions"],
            min_span_bp=params["min_span_bp"],
        )
        all_candidates.extend(
            merge_candidates(
                cn,
                graph,
                overlap_threshold=params["overlap_threshold"],
                df_severus_sample=df_s,
            )
        )
    return all_candidates


def candidates_to_frame(candidates: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert candidate dicts to a TSV-friendly summary."""
    rows: list[dict[str, Any]] = []
    for i, cand in enumerate(candidates):
        rows.append(
            {
                "candidate_id": cand.get("candidate_id", f"cand_{i:06d}"),
                "label_id": cand.get("label_id", ""),
                "sample_id": cand["sample_id"],
                "chrom": cand["chrom"],
                "start_bp": int(cand["start_bp"]),
                "end_bp": int(cand["end_bp"]),
                "evidence": cand.get("evidence", ""),
                "n_sv_nodes": len(cand.get("sv_node_indices", [])),
                "n_segments": len(cand.get("df_segments", [])),
                "sv_class": cand.get("sv_class", ""),
                "label_scope": cand.get("label_scope", ""),
            }
        )
    return pd.DataFrame(rows)


def write_candidates(candidates: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidates_to_frame(candidates).to_csv(path, sep="\t", index=False)
