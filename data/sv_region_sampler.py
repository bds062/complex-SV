"""
Sample fixed-bp Severus SV regions and build regional graph objects.

The graph pretraining and visualisation CLIs use this module to mirror the
CN encoder's genomic-window interface while keeping Severus VCFs as the raw
SV input.  Windows are sampled in 0-based half-open coordinates and retain the
global DataFrame indices needed to slice the aligned node feature matrix.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

try:
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover - dependency message only
    raise ImportError(
        "torch-geometric is required for complex_sv.data.sv_region_sampler. "
        "Install a PyG build matching your PyTorch version."
    ) from exc

from .graph_builder import build_sample_graph
from .severus_parser import N_FEAT


def _validate_window_args(
    window_bp_sizes: Sequence[int],
    windows_per_chrom_per_size: int,
    cluster_windows_per_chrom_per_size: int,
    min_sv_per_window: int,
    max_windows: int | None,
) -> None:
    if not window_bp_sizes:
        raise ValueError("window_bp_sizes must contain at least one bp size")

    bad_sizes = [int(x) for x in window_bp_sizes if int(x) <= 0]
    if bad_sizes:
        raise ValueError(f"window_bp_sizes must be positive; got {bad_sizes}")
    if int(windows_per_chrom_per_size) < 0:
        raise ValueError("windows_per_chrom_per_size must be non-negative")
    if int(cluster_windows_per_chrom_per_size) < 0:
        raise ValueError("cluster_windows_per_chrom_per_size must be non-negative")
    if int(min_sv_per_window) <= 0:
        raise ValueError("min_sv_per_window must be positive")
    if max_windows is not None and int(max_windows) <= 0:
        raise ValueError("max_windows must be positive when provided")


def _mode_or_empty(values: pd.Series) -> str:
    if values.empty:
        return ""
    mode = values.astype(str).mode()
    return str(mode.iloc[0]) if not mode.empty else str(values.iloc[0])


def _window_record(
    df: pd.DataFrame,
    group: pd.DataFrame,
    sample_id: str,
    chrom: str,
    start_bp: int,
    end_bp: int,
    requested_window_bp: int,
    window_source: str,
    min_sv_per_window: int,
) -> dict[str, Any] | None:
    if end_bp <= start_bp:
        return None

    overlap = group[(group["pos"] >= start_bp) & (group["pos"] < end_bp)]
    if len(overlap) < int(min_sv_per_window):
        return None

    idx = overlap.index.astype(int).tolist()
    span_bp = max(int(end_bp) - int(start_bp), 1)
    density = float(len(idx) / (span_bp / 1_000_000.0))

    return {
        "sample_id": str(sample_id),
        "chrom": str(chrom),
        "start_bp": int(start_bp),
        "end_bp": int(end_bp),
        "window_bp_size": int(span_bp),
        "requested_window_bp_size": int(requested_window_bp),
        "window_source": str(window_source),
        "n_sv": int(len(idx)),
        "breakpoint_density_per_mb": density,
        "n_bnd": int((df.loc[idx, "is_bnd"] > 0).sum()),
        "n_phased": int((df.loc[idx, "has_phase"] > 0).sum()),
        "dom_sv_type": _mode_or_empty(df.loc[idx, "sv_type_str"]),
        "mean_vaf": float(pd.to_numeric(df.loc[idx, "vaf_mean"], errors="coerce").fillna(0.0).mean()),
        "global_node_indices": idx,
    }


def _effective_span(chrom_start: int, chrom_end: int, window_bp: int) -> tuple[int, int, int]:
    chrom_start = int(chrom_start)
    chrom_end = int(chrom_end)
    window_bp = int(window_bp)
    chrom_span = chrom_end - chrom_start
    if chrom_span <= 0:
        return chrom_start, chrom_end, 0
    if window_bp >= chrom_span:
        return chrom_start, chrom_end, chrom_span
    return chrom_start, chrom_start + window_bp, window_bp


def _clip_start(center_start: int, chrom_start: int, chrom_end: int, window_bp: int) -> int:
    if window_bp >= chrom_end - chrom_start:
        return int(chrom_start)
    max_start = int(chrom_end) - int(window_bp)
    return int(min(max(int(center_start), int(chrom_start)), max_start))


def build_sv_bp_windows(
    df: pd.DataFrame,
    window_bp_sizes: Sequence[int],
    windows_per_chrom_per_size: int = 20,
    cluster_windows_per_chrom_per_size: int = 20,
    min_sv_per_window: int = 2,
    rng: np.random.Generator | None = None,
    max_windows: int | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """
    Build mixed random and dense breakpoint fixed-bp graph windows.

    Random windows mirror the CN encoder's bp-window sampling. Dense windows are
    anchored around observed breakpoints and ranked by SV count within the same
    requested bp span, which enriches the window set for chromothripsis-like
    breakpoint clusters without discarding background regions.
    """
    _validate_window_args(
        window_bp_sizes,
        windows_per_chrom_per_size,
        cluster_windows_per_chrom_per_size,
        min_sv_per_window,
        max_windows,
    )
    if rng is None:
        rng = np.random.default_rng()

    required = {"sample_id", "chrom", "pos", "end", "is_bnd", "has_phase", "sv_type_str", "vaf_mean"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Cannot sample SV bp windows; missing columns: {missing}")

    max_windows = int(max_windows) if max_windows is not None else None
    groups = list(df.groupby(["sample_id", "chrom"], sort=False))
    group_iter = tqdm(groups, desc="build SV windows", unit="chrom", leave=False) if progress else groups
    windows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int, int]] = set()

    def add_record(record: dict[str, Any] | None) -> None:
        if record is None:
            return
        key = (
            str(record["sample_id"]),
            str(record["chrom"]),
            int(record["start_bp"]),
            int(record["end_bp"]),
        )
        if key in seen:
            return
        seen.add(key)
        windows.append(record)

    for (sample_id, chrom), grp_raw in group_iter:
        if max_windows is not None and len(windows) >= max_windows:
            break

        grp = grp_raw.copy()
        grp["pos"] = pd.to_numeric(grp["pos"], errors="coerce")
        grp["end"] = pd.to_numeric(grp["end"], errors="coerce")
        grp = grp.dropna(subset=["pos", "end"]).sort_values(["pos", "end"])
        if grp.empty:
            continue

        chrom_start = int(grp["pos"].min())
        chrom_end = int(grp["end"].max())
        if chrom_end <= chrom_start:
            continue

        positions = grp["pos"].to_numpy(dtype=np.int64)
        for requested_bp in window_bp_sizes:
            if max_windows is not None and len(windows) >= max_windows:
                break

            requested_bp = int(requested_bp)
            span_start, span_end, effective_bp = _effective_span(chrom_start, chrom_end, requested_bp)
            if effective_bp <= 0:
                continue

            if requested_bp >= chrom_end - chrom_start:
                random_starts = np.array([chrom_start], dtype=np.int64)
            else:
                max_start = chrom_end - requested_bp
                possible = max_start - chrom_start + 1
                n_random = min(int(windows_per_chrom_per_size), int(possible))
                random_starts = (
                    rng.integers(
                        low=chrom_start,
                        high=max_start + 1,
                        size=n_random,
                        endpoint=False,
                        dtype=np.int64,
                    )
                    if n_random > 0
                    else np.array([], dtype=np.int64)
                )

            for start_bp in random_starts:
                if max_windows is not None and len(windows) >= max_windows:
                    break
                start_bp = int(start_bp)
                end_bp = int(start_bp + effective_bp)
                add_record(
                    _window_record(
                        df,
                        grp,
                        str(sample_id),
                        str(chrom),
                        start_bp,
                        end_bp,
                        requested_bp,
                        "random",
                        min_sv_per_window,
                    )
                )

            if int(cluster_windows_per_chrom_per_size) <= 0:
                continue

            dense_candidates: list[tuple[int, int, int]] = []
            for pos in positions:
                start_bp = _clip_start(
                    int(pos) - effective_bp // 2,
                    chrom_start,
                    chrom_end,
                    effective_bp,
                )
                end_bp = int(start_bp + effective_bp)
                count = int(((positions >= start_bp) & (positions < end_bp)).sum())
                dense_candidates.append((-count, start_bp, end_bp))

            dense_candidates = sorted(set(dense_candidates))
            for _neg_count, start_bp, end_bp in dense_candidates[: int(cluster_windows_per_chrom_per_size)]:
                if max_windows is not None and len(windows) >= max_windows:
                    break
                add_record(
                    _window_record(
                        df,
                        grp,
                        str(sample_id),
                        str(chrom),
                        int(start_bp),
                        int(end_bp),
                        requested_bp,
                        "dense",
                        min_sv_per_window,
                    )
                )

    return windows



def build_sv_interval_windows(
    df: pd.DataFrame,
    intervals: Iterable[dict[str, Any]],
    min_sv_per_window: int = 1,
) -> list[dict[str, Any]]:
    """Build graph windows from explicit sample/chrom/start/end intervals."""
    if int(min_sv_per_window) <= 0:
        raise ValueError("min_sv_per_window must be positive")

    required = {"sample_id", "chrom", "pos", "end", "is_bnd", "has_phase", "sv_type_str", "vaf_mean"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Cannot build interval SV windows; missing columns: {missing}")

    windows: list[dict[str, Any]] = []
    for interval in intervals:
        sample_id = str(interval["sample_id"])
        chrom = str(interval["chrom"])
        sample_chrom = df[(df["sample_id"].astype(str) == sample_id) & (df["chrom"].astype(str) == chrom)]
        if sample_chrom.empty:
            raise ValueError(f"No parsed SV records for highlighted interval {sample_id}:{chrom}")

        if interval.get("start_bp") is None or interval.get("end_bp") is None:
            start_bp = int(pd.to_numeric(sample_chrom["pos"], errors="coerce").min())
            end_bp = int(pd.to_numeric(sample_chrom["end"], errors="coerce").max())
        else:
            start_bp = int(interval["start_bp"])
            end_bp = int(interval["end_bp"])
        if end_bp <= start_bp:
            raise ValueError(f"Invalid interval for {sample_id}:{chrom}: {start_bp}-{end_bp}")

        record = _window_record(
            df,
            sample_chrom,
            sample_id,
            chrom,
            start_bp,
            end_bp,
            end_bp - start_bp,
            str(interval.get("window_source", "highlight")),
            min_sv_per_window,
        )
        if record is None:
            raise ValueError(
                f"Interval {sample_id}:{chrom}:{start_bp}-{end_bp} contains fewer than "
                f"{min_sv_per_window} SV node(s)"
            )
        for key, value in interval.items():
            if key not in record:
                record[key] = value
        windows.append(record)

    return windows

def window_metadata_frame(
    windows: Iterable[dict[str, Any]],
    include_node_indices: bool = False,
) -> pd.DataFrame:
    """Return a TSV/NPZ-friendly metadata DataFrame for sampled windows."""
    records: list[dict[str, Any]] = []
    for win in windows:
        row = {k: v for k, v in win.items() if k != "global_node_indices"}
        if include_node_indices:
            row["global_node_indices"] = ",".join(str(int(i)) for i in win.get("global_node_indices", []))
        records.append(row)
    return pd.DataFrame(records)


def build_region_graph(
    df: pd.DataFrame,
    feat_matrix: np.ndarray,
    window: dict[str, Any],
    proximity_bp: int = 1_000_000,
) -> HeteroData:
    """Build one regional HeteroData graph for a sampled window."""
    feat_matrix = np.asarray(feat_matrix, dtype=np.float32)
    if feat_matrix.ndim != 2 or feat_matrix.shape[1] != N_FEAT:
        raise ValueError(f"feat_matrix must have shape [N, {N_FEAT}], got {feat_matrix.shape}")
    if len(df) != feat_matrix.shape[0]:
        raise ValueError(f"df length ({len(df)}) and feature rows ({feat_matrix.shape[0]}) do not match")

    idx = [int(i) for i in window.get("global_node_indices", [])]
    if not idx:
        raise ValueError("Window has no global_node_indices")

    sub_df = df.loc[idx].copy().reset_index(drop=True)
    graph = build_sample_graph(sub_df, feat_matrix[idx], proximity_bp=proximity_bp)
    graph.sample_id = str(window["sample_id"])
    graph.chrom = str(window["chrom"])
    graph.region_start_bp = int(window["start_bp"])
    graph.region_end_bp = int(window["end_bp"])
    graph.requested_window_bp_size = int(window["requested_window_bp_size"])
    graph.window_source = str(window["window_source"])
    graph["sv"].global_index = torch.as_tensor(idx, dtype=torch.long)
    return graph


def build_region_graphs(
    df: pd.DataFrame,
    feat_matrix: np.ndarray,
    windows: Sequence[dict[str, Any]],
    proximity_bp: int = 1_000_000,
    progress: bool = False,
) -> list[HeteroData]:
    """Build regional graphs for all sampled windows."""
    iterable = tqdm(windows, desc="build region graphs", unit="window", leave=False) if progress else windows
    return [build_region_graph(df, feat_matrix, win, proximity_bp=proximity_bp) for win in iterable]
