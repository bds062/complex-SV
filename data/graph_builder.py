"""
Build PyTorch Geometric heterogeneous SV graphs from parsed Severus records.

This module contains no model code.  It converts the canonical DataFrame and
node-feature matrix produced by data.severus_parser into the
HeteroData objects consumed by the graph encoder and downstream region proposal.

Every graph always contains the required edge types, even when an edge
category has zero edges.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd
import torch

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover - dependency message only
    raise ImportError(
        "torch-geometric is required for complex_sv.data.graph_builder. "
        "Install a PyG build matching your PyTorch version."
    ) from exc

from .severus_parser import N_FEAT

log = logging.getLogger(__name__)

EDGE_PROXIMITY = ("sv", "proximal_to", "sv")
EDGE_MATE = ("sv", "mate_of", "sv")
EDGE_INTERCHROM = ("sv", "interchrom_mate_of", "sv")
EDGE_CLUSTER = ("sv", "cluster_linked", "sv")
EDGE_PHASE = ("sv", "phase_linked", "sv")
EDGE_TYPES = (EDGE_PROXIMITY, EDGE_MATE, EDGE_INTERCHROM, EDGE_CLUSTER, EDGE_PHASE)
EDGE_ATTR_DIM = 3


def _empty_edge_index() -> torch.Tensor:
    return torch.zeros((2, 0), dtype=torch.long)


def _empty_edge_attr() -> torch.Tensor:
    return torch.zeros((0, EDGE_ATTR_DIM), dtype=torch.float32)


def _coerce_edge_attr(value: object) -> list[float]:
    if isinstance(value, (int, float, np.floating)):
        out = [float(value), 0.0, 0.0]
    else:
        out = [float(x) for x in value]  # type: ignore[arg-type]
    if len(out) < EDGE_ATTR_DIM:
        out.extend([0.0] * (EDGE_ATTR_DIM - len(out)))
    return out[:EDGE_ATTR_DIM]


def _as_edge_tensors(src: list[int], dst: list[int], weights: list[object] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if not src:
        return _empty_edge_index(), _empty_edge_attr()
    if weights is None:
        weights = [[1.0, 0.0, 0.0] for _ in src]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor([_coerce_edge_attr(w) for w in weights], dtype=torch.float32)
    return edge_index, edge_attr


def _chrom_key(chrom: object) -> str:
    return str(chrom).strip().removeprefix("chr")


def _chrom_equal(a: object, b: object) -> bool:
    return _chrom_key(a) == _chrom_key(b)


def _distance_feature(df: pd.DataFrame, i: int, j: int, proximity_bp: int) -> float:
    if not _chrom_equal(df.loc[i, "chrom"], df.loc[j, "chrom"]):
        return 0.0
    dist = abs(int(df.loc[i, "pos"]) - int(df.loc[j, "pos"]))
    return float(min(dist / max(float(proximity_bp), 1.0), 1.0))


def _relation_attr(df: pd.DataFrame, i: int, j: int, proximity_bp: int, strength: float = 1.0) -> list[float]:
    interchrom = 0.0 if _chrom_equal(df.loc[i, "chrom"], df.loc[j, "chrom"]) else 1.0
    return [float(strength), _distance_feature(df, i, j, proximity_bp), interchrom]


def _ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"sample_id", "sv_id", "mate_id", "cluster_id", "phase_set", "chrom", "pos"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Cannot build graph; missing Severus columns: {missing}")


def build_sample_graph(
    df: pd.DataFrame,
    feat_matrix: np.ndarray,
    proximity_bp: int = 1_000_000,
) -> HeteroData:
    """
    Build one HeteroData graph for a single sample.

    Parameters
    ----------
    df:
        Single-sample Severus DataFrame.  The caller should pass a reset-index
        DataFrame so node ids correspond to row positions.
    feat_matrix:
        Float array with shape [N, N_FEAT] containing node features for df.
    proximity_bp:
        Maximum same-chromosome distance for a proximity edge.

    Returns
    -------
    HeteroData
        A graph with one node type ("sv") and the three required edge types.
    """
    _ensure_required_columns(df)
    df = df.reset_index(drop=True).copy()
    if proximity_bp <= 0:
        raise ValueError("proximity_bp must be positive")

    feat_matrix = np.asarray(feat_matrix, dtype=np.float32)
    if not feat_matrix.flags.writeable:
        feat_matrix = feat_matrix.copy()
    n = len(df)
    if feat_matrix.shape != (n, N_FEAT):
        raise ValueError(f"feat_matrix must have shape ({n}, {N_FEAT}); observed {feat_matrix.shape}")

    data = HeteroData()
    data["sv"].x = torch.from_numpy(feat_matrix)
    data["sv"].pos = torch.as_tensor(df["pos"].to_numpy(dtype=np.int64, copy=True), dtype=torch.long)
    data["sv"].chrom = [str(c) for c in df["chrom"].tolist()]

    if "end" in df.columns:
        data["sv"].end = torch.as_tensor(df["end"].to_numpy(dtype=np.int64, copy=True), dtype=torch.long)
    if "sv_id" in df.columns:
        data["sv"].sv_id = [str(x) for x in df["sv_id"].tolist()]
    for col in ("is_foldback", "is_inv_like_bnd", "has_interchrom_mate", "has_samechrom_mate"):
        if col in df.columns:
            data["sv"][col] = torch.as_tensor(
                pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
                dtype=torch.float32,
            )
    if "sample_id" in df.columns and n:
        data.sample_id = str(df["sample_id"].iloc[0])

    # Proximity edges: same chromosome, sorted by pos, early exit by distance.
    prox_src: list[int] = []
    prox_dst: list[int] = []
    prox_w: list[list[float]] = []
    for _chrom, grp in df.groupby("chrom", sort=False):
        grp_sorted = grp.sort_values("pos")
        idx = grp_sorted.index.to_list()
        positions = grp_sorted["pos"].to_numpy(dtype=np.int64)
        for li, gi in enumerate(idx):
            pos_i = int(positions[li])
            for lj in range(li + 1, len(idx)):
                dist = int(positions[lj]) - pos_i
                if dist > proximity_bp:
                    break
                gj = idx[lj]
                weight = 1.0 - (float(dist) / float(proximity_bp))
                dist_feature = 1.0 - weight
                prox_src.extend([int(gi), int(gj)])
                prox_dst.extend([int(gj), int(gi)])
                prox_w.extend([[weight, dist_feature, 0.0], [weight, dist_feature, 0.0]])

    data[EDGE_PROXIMITY].edge_index, data[EDGE_PROXIMITY].edge_attr = _as_edge_tensors(
        prox_src, prox_dst, prox_w
    )

    # Explicit mate edges are split by chromosome so the encoder can
    # distinguish local rearrangements from interchromosomal junctions.
    id_to_idx = {str(sv_id): i for i, sv_id in enumerate(df["sv_id"].tolist()) if str(sv_id)}
    same_mate_edges: dict[tuple[int, int], list[float]] = {}
    inter_mate_edges: dict[tuple[int, int], list[float]] = {}

    for i, row in df.reset_index(drop=True).iterrows():
        mate_id = str(row.get("mate_id", "") or "")
        if mate_id and mate_id != "." and mate_id in id_to_idx:
            j = int(id_to_idx[mate_id])
            if i == j:
                continue
            attr_ij = _relation_attr(df, int(i), j, proximity_bp)
            attr_ji = _relation_attr(df, j, int(i), proximity_bp)
            target = inter_mate_edges if attr_ij[2] > 0 else same_mate_edges
            target[(int(i), j)] = attr_ij
            target[(j, int(i))] = attr_ji

    for edge_type, edge_map in ((EDGE_MATE, same_mate_edges), (EDGE_INTERCHROM, inter_mate_edges)):
        if edge_map:
            src, dst, attrs = zip(*[(i, j, attr) for (i, j), attr in sorted(edge_map.items())])
            data[edge_type].edge_index, data[edge_type].edge_attr = _as_edge_tensors(
                list(src), list(dst), list(attrs)
            )
        else:
            data[edge_type].edge_index = _empty_edge_index()
            data[edge_type].edge_attr = _empty_edge_attr()

    # Cluster edges: fully connect CLUSTERID groups as their own relation.
    cluster_to_idx: Dict[str, list[int]] = {}
    for i, cluster_id in enumerate(df["cluster_id"].tolist()):
        cid = str(cluster_id or "")
        if cid and cid != ".":
            cluster_to_idx.setdefault(cid, []).append(i)

    cluster_edges: dict[tuple[int, int], list[float]] = {}
    for members in cluster_to_idx.values():
        if len(members) < 2:
            continue
        strength = 1.0 / float(max(len(members) - 1, 1))
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = int(members[a]), int(members[b])
                cluster_edges[(i, j)] = _relation_attr(df, i, j, proximity_bp, strength=strength)
                cluster_edges[(j, i)] = _relation_attr(df, j, i, proximity_bp, strength=strength)

    if cluster_edges:
        cluster_src, cluster_dst, cluster_attr = zip(
            *[(i, j, attr) for (i, j), attr in sorted(cluster_edges.items())]
        )
        data[EDGE_CLUSTER].edge_index, data[EDGE_CLUSTER].edge_attr = _as_edge_tensors(
            list(cluster_src), list(cluster_dst), list(cluster_attr)
        )
    else:
        data[EDGE_CLUSTER].edge_index = _empty_edge_index()
        data[EDGE_CLUSTER].edge_attr = _empty_edge_attr()

    # Phase edges: fully connect all members of each non-zero PHASESETID.
    phase_to_idx: Dict[int, list[int]] = {}
    for i, phase_set in enumerate(df["phase_set"].tolist()):
        try:
            phase = int(phase_set)
        except (TypeError, ValueError):
            phase = 0
        if phase != 0:
            phase_to_idx.setdefault(phase, []).append(i)

    phase_src: list[int] = []
    phase_dst: list[int] = []
    phase_attr: list[list[float]] = []
    for members in phase_to_idx.values():
        if len(members) < 2:
            continue
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = int(members[a]), int(members[b])
                phase_src.extend([i, j])
                phase_dst.extend([j, i])
                phase_attr.extend([
                    _relation_attr(df, i, j, proximity_bp),
                    _relation_attr(df, j, i, proximity_bp),
                ])

    data[EDGE_PHASE].edge_index, data[EDGE_PHASE].edge_attr = _as_edge_tensors(phase_src, phase_dst, phase_attr)

    # Defensive final check: HeteroConv expects all edge types to be present.
    for edge_type in EDGE_TYPES:
        if not hasattr(data[edge_type], "edge_index"):
            data[edge_type].edge_index = _empty_edge_index()
        if not hasattr(data[edge_type], "edge_attr"):
            data[edge_type].edge_attr = _empty_edge_attr()

    return data


def build_all_graphs(
    df: pd.DataFrame,
    feat_matrix: np.ndarray,
    proximity_bp: int = 1_000_000,
) -> tuple[list[HeteroData], list[str]]:
    """Build one HeteroData graph per sample_id group."""
    if "sample_id" not in df.columns:
        raise ValueError("Severus DataFrame must contain a sample_id column")

    feat_matrix = np.asarray(feat_matrix, dtype=np.float32)
    if len(df) != feat_matrix.shape[0]:
        raise ValueError(f"df length ({len(df)}) and feature rows ({feat_matrix.shape[0]}) do not match")

    graphs: list[HeteroData] = []
    sample_ids: list[str] = []
    for sample_id, grp in df.groupby("sample_id", sort=False):
        idx = grp.index.to_list()
        graph = build_sample_graph(grp.reset_index(drop=True), feat_matrix[idx], proximity_bp=proximity_bp)
        graphs.append(graph)
        sample_ids.append(str(sample_id))
        graph.sample_id = str(sample_id)

        log.info(
            "  %s: %d nodes, prox=%d mate=%d interchrom=%d cluster=%d phase=%d undirected edge pairs",
            sample_id,
            graph["sv"].x.shape[0],
            graph[EDGE_PROXIMITY].edge_index.shape[1] // 2,
            graph[EDGE_MATE].edge_index.shape[1] // 2,
            graph[EDGE_INTERCHROM].edge_index.shape[1] // 2,
            graph[EDGE_CLUSTER].edge_index.shape[1] // 2,
            graph[EDGE_PHASE].edge_index.shape[1] // 2,
        )

    return graphs, sample_ids


def _safe_cache_name(sample_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._")
    return safe or "sample"


def cache_graphs(graphs: list[HeteroData], sample_ids: list[str], cache_dir: str | Path) -> None:
    """Save each graph as a .pt file in cache_dir."""
    if len(graphs) != len(sample_ids):
        raise ValueError("graphs and sample_ids must have the same length")

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    for graph, sample_id in zip(graphs, sample_ids):
        base = _safe_cache_name(str(sample_id))
        name = base
        counter = 1
        while name in used_names:
            counter += 1
            name = f"{base}_{counter}"
        used_names.add(name)
        graph.sample_id = str(sample_id)
        torch.save({"sample_id": str(sample_id), "graph": graph}, cache_path / f"{name}.pt")

    log.info("Cached %d graph(s) in %s", len(graphs), cache_path)


def load_cached_graphs(cache_dir: str | Path) -> tuple[list[HeteroData], list[str]]:
    """Load cached HeteroData graphs sorted by sample_id for reproducibility."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        raise FileNotFoundError(f"Graph cache directory does not exist: {cache_path}")

    records: list[tuple[str, HeteroData]] = []
    for path in sorted(cache_path.glob("*.pt")):
        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # Older PyTorch versions do not accept weights_only.
            obj = torch.load(path, map_location="cpu")

        if isinstance(obj, dict) and "graph" in obj:
            graph = obj["graph"]
            sample_id = str(obj.get("sample_id", getattr(graph, "sample_id", path.stem)))
        else:
            graph = obj
            sample_id = str(getattr(graph, "sample_id", path.stem))
        graph.sample_id = sample_id
        records.append((sample_id, graph))

    if not records:
        raise FileNotFoundError(f"No cached .pt graphs found in {cache_path}")

    records.sort(key=lambda item: item[0])
    sample_ids = [sample_id for sample_id, _graph in records]
    graphs = [graph for _sample_id, graph in records]
    log.info("Loaded %d cached graph(s) from %s", len(graphs), cache_path)
    return graphs, sample_ids
