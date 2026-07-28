#!/usr/bin/env python3
"""Utilities for converting CNA VCF files into pandas DataFrames."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


BASE_COLUMNS = ["CHROM", "POS", "ID"]
SAMPLE_COLUMNS = ["TCN", "CN1", "CN2"]

__all__ = [
    "assign_linked_cluster_ids",
    "calculate_ploidy",
    "classify_linked_clusters",
    "count_cna_segments_in_windows",
    "count_breakpoints_in_candidate_windows",
    "find_small_segment_candidate_intervals",
    "generate_candidate_plot",
    "get_bps",
    "read_centromere_bed",
    "merge_candidate_windows",
    "merge_candidate_intervals_by_sv_or_distance",
    "merge_sv_cna_candidate_segments",
    "merge_sv_cna_candidate_windows",
    "read_cna_vcf_to_dataframe",
]


SV_TYPE_COLORS = {
    "DEL": "#d62728",
    "DUP": "#1f77b4",
    "FB": "#9467bd",
    "INTER_CHR": "#ff7f0e",
    "INV_LIKE": "#2ca02c",
    "sBND": "#8c564b",
}
FIXED_SV_TYPES = tuple(SV_TYPE_COLORS.keys())

REGION_COLORS = [
    "#4e79a7",
    "#f28e2b",
    "#59a14f",
    "#e15759",
    "#76b7b2",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
]

PLOT_FLANK_SIZE = 10_000_000
PLOT_FLANK_COLOR = "#d9d9d9"
CLASSIFICATION_MIN_SEGMENT_SIZE = 100_000


def _extract_info_value(info: str, key: str) -> str | None:
    """Return one INFO value from a semicolon-delimited VCF INFO field."""
    prefix = f"{key}="
    for item in str(info).split(";"):
        if item.startswith(prefix):
            return item[len(prefix) :]
    return None


def _parse_info(info: str) -> dict[str, str | bool]:
    parsed = {}
    for item in info.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            parsed[key] = value
        else:
            parsed[item] = True
    return parsed


def _parse_sample(format_field: str, sample_field: str) -> dict[str, str]:
    keys = format_field.split(":")
    values = sample_field.split(":")
    return dict(zip(keys, values))


def _parse_bnd_alt(alt: str) -> tuple[str, int]:
    match = re.search(r"[\[\]]([^:\[\]]+):(\d+)[\[\]]", alt)
    if not match:
        raise ValueError(f"Could not parse BND ALT field: {alt}")
    return match.group(1), int(match.group(2))


def _as_int(value: str | bool | None) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    return int(value)


def _split_read_counts(value: str | bool | None) -> list[int]:
    if value is None or isinstance(value, bool):
        return []
    return [int(count) for count in value.split(":")]


def _get_sv_type(info: dict[str, str | bool]) -> str | bool | None:
    svt = info.get("SVTYPE")
    if svt == "sBND":
        return "sBND"
    if info.get("DETAILED_TYPE") == "foldback":
        return "FB"
    if svt == "BND":
        bnd_type = info.get("BND_TYPE", "BND")
        if bnd_type == "DEL_LIKE":
            return "DEL"
        if bnd_type == "DUP_LIKE":
            return "DUP"
        return bnd_type
    return svt


def _split_sample_values(row: pd.Series, sample_column: str) -> pd.Series:
    format_keys = str(row["FORMAT"]).split(":")
    sample_values = str(row[sample_column]).split(":")
    sample_map = dict(zip(format_keys, sample_values))
    return pd.Series({column: sample_map.get(column) for column in SAMPLE_COLUMNS})


def _parse_contig_header(line: str) -> tuple[str, int] | None:
    match = re.match(r"##contig=<ID=([^,>]+),length=(\d+)", line)
    if not match:
        return None
    return match.group(1), int(match.group(2))


def _chrom_sort_key(chrom: str) -> tuple[int, int | str]:
    label = str(chrom)
    if label.startswith("chr"):
        label = label[3:]
    if label.isdigit():
        return 0, int(label)
    if label == "X":
        return 1, 23
    if label == "Y":
        return 1, 24
    if label in {"M", "MT"}:
        return 1, 25
    return 2, label


def _natural_sort_dataframe(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    if df.empty or "chrom" not in df.columns:
        return df

    sort_df = df.copy()
    sort_df["_chrom_sort_key"] = sort_df["chrom"].map(_chrom_sort_key)
    sort_columns = ["_chrom_sort_key", *[column for column in columns if column in sort_df.columns]]
    sort_df = sort_df.sort_values(sort_columns).drop(columns="_chrom_sort_key")
    return sort_df.reset_index(drop=True)


def read_cna_vcf_to_dataframe(vcf_path: str | Path) -> pd.DataFrame:
    """Read a CNA VCF file and return selected fields as a pandas DataFrame.

    The returned columns are:
    chrom, start, end, cna_id, TCN, CN1, CN2.
    """
    vcf_path = Path(vcf_path)

    df = pd.read_csv(vcf_path, sep="\t", comment="#", header=None)
    contig_lengths = {}
    with vcf_path.open() as handle:
        for line in handle:
            contig = _parse_contig_header(line)
            if contig is not None:
                chrom, length = contig
                contig_lengths[chrom] = length
            if line.startswith("#CHROM"):
                df.columns = line.lstrip("#").rstrip("\n").split("\t")
                break
        else:
            raise ValueError(f"No #CHROM header found in {vcf_path}")

    sample_column = df.columns[-1]
    sample_df = df.apply(_split_sample_values, axis=1, sample_column=sample_column)

    result = df.loc[:, BASE_COLUMNS].rename(
        columns={"CHROM": "chrom", "POS": "start", "ID": "cna_id"}
    )
    result["end"] = df["INFO"].map(lambda info: _extract_info_value(info, "END"))
    result = result.loc[:, ["chrom", "start", "end", "cna_id"]]
    result = pd.concat([result, sample_df], axis=1)

    numeric_columns = ["start", "end", "TCN", "CN1", "CN2"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result = _fill_reference_gaps(result, contig_lengths=contig_lengths)
    result = _merge_consecutive_same_tcn(result)

    return _natural_sort_dataframe(result, ["start", "end"])


def _neutral_cna_row(chrom: str, start: int, end: int) -> dict:
    return {
        "chrom": chrom,
        "start": start,
        "end": end,
        "cna_id": f"wakhan:NEUTRAL:{chrom}:{start}-{end}",
        "TCN": 2.0,
        "CN1": 1.0,
        "CN2": 1.0,
    }


def _fill_reference_gaps(
    df: pd.DataFrame,
    contig_lengths: dict[str, int] | None = None,
) -> pd.DataFrame:
    filled_rows = []
    contig_lengths = contig_lengths or {}

    for _, chrom_df in df.groupby("chrom", sort=False):
        rows = chrom_df.sort_values("start").to_dict("records")
        for index, row in enumerate(rows):
            if index == 0 and int(row["start"]) > 0:
                gap_start = 0
                gap_end = int(row["start"]) - 1
                filled_rows.append(_neutral_cna_row(row["chrom"], gap_start, gap_end))

            filled_rows.append(row)
            if index == len(rows) - 1:
                continue

            next_row = rows[index + 1]
            gap_start = int(row["end"]) + 1
            gap_end = int(next_row["start"]) - 1
            if gap_start <= gap_end:
                filled_rows.append(_neutral_cna_row(row["chrom"], gap_start, gap_end))

        chrom = rows[-1]["chrom"]
        chrom_length = contig_lengths.get(chrom)
        if chrom_length is not None:
            gap_start = int(rows[-1]["end"]) + 1
            gap_end = chrom_length
            if gap_start <= gap_end:
                filled_rows.append(_neutral_cna_row(chrom, gap_start, gap_end))

    return pd.DataFrame(filled_rows, columns=df.columns)


def _merge_consecutive_same_tcn(df: pd.DataFrame) -> pd.DataFrame:
    merged_rows = []

    for _, chrom_df in df.groupby("chrom", sort=False):
        rows = chrom_df.sort_values("start").to_dict("records")
        active = None

        for row in rows:
            if active is None:
                active = row.copy()
                continue

            is_adjacent = int(row["start"]) <= int(active["end"]) + 1
            same_tcn = row["TCN"] == active["TCN"]
            if is_adjacent and same_tcn:
                active["end"] = max(int(active["end"]), int(row["end"]))
                active["cna_id"] = (
                    f"merged:TCN{_format_tcn_label(active['TCN'])}:"
                    f"{active['chrom']}:{active['start']}-{active['end']}"
                )
                if not _same_value(row["CN1"], active["CN1"]):
                    active["CN1"] = pd.NA
                if not _same_value(row["CN2"], active["CN2"]):
                    active["CN2"] = pd.NA
            else:
                merged_rows.append(active)
                active = row.copy()

        if active is not None:
            merged_rows.append(active)

    return pd.DataFrame(merged_rows, columns=df.columns)


def _same_value(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return left == right


def _format_tcn_label(tcn: float) -> str:
    if pd.isna(tcn):
        return "NA"
    if float(tcn).is_integer():
        return str(int(tcn))
    return str(tcn).replace(".", "_")


TCN_BINS = [
    ("0", 0, 0),
    ("1", 1, 1),
    ("2", 2, 2),
    ("3", 3, 3),
    ("4", 4, 4),
    ("5", 5, 5),
    ("6", 6, 6),
    ("7", 7, 7),
    ("7_10", 8, 10),
    ("11_20", 11, 20),
    ("20_40", 21, 40),
    ("gt_40", 41, None),
]


def _tcn_bin_columns() -> list[str]:
    return [f"n_TCN_{label}" for label, _, _ in TCN_BINS]


def _count_tcn_bins(tcn_series: pd.Series) -> dict[str, int]:
    counts = {}
    for label, lower, upper in TCN_BINS:
        if upper is None:
            count = (tcn_series >= lower).sum()
        else:
            count = ((tcn_series >= lower) & (tcn_series <= upper)).sum()
        counts[f"n_TCN_{label}"] = int(count)
    return counts


def calculate_ploidy(
    cna_df: pd.DataFrame,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
) -> float:
    """Calculate length-weighted ploidy from CNA segments and TCN."""
    required_columns = {"chrom", "start", "end", "TCN"}
    missing_columns = required_columns - set(cna_df.columns)
    if missing_columns:
        raise ValueError(f"cna_df is missing columns: {', '.join(sorted(missing_columns))}")

    df = cna_df.copy()
    if chrom is not None:
        df = df[df["chrom"] == chrom]
    if start is not None:
        df = df[df["end"] >= start]
    if end is not None:
        df = df[df["start"] <= end]
    if df.empty:
        return float("nan")

    overlap_start = df["start"]
    overlap_end = df["end"]
    if start is not None:
        overlap_start = overlap_start.clip(lower=start)
    if end is not None:
        overlap_end = overlap_end.clip(upper=end)

    lengths = (overlap_end - overlap_start + 1).clip(lower=0)
    total_length = lengths.sum()
    if total_length == 0:
        return float("nan")
    return float((lengths * df["TCN"]).sum() / total_length)


def _component_intervals_from_row(row: dict | pd.Series) -> list[tuple[int, int]]:
    value = row.get("component_intervals")
    if value is None or pd.isna(value):
        return [(int(row["start"]), int(row["end"]))]

    components = []
    for item in str(value).split(";"):
        if not item:
            continue
        start, end = item.split("-", 1)
        components.append((int(start), int(end)))
    return components or [(int(row["start"]), int(row["end"]))]


def _component_intervals_to_string(components: list[tuple[int, int]]) -> str:
    return ";".join(f"{start}-{end}" for start, end in components)


def _intervals_length(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


def _merge_component_intervals(
    components: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    if not components:
        return []

    merged = []
    for start, end in sorted(components):
        if not merged or start > merged[-1][1] + 1:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _filter_cna_by_components(
    cna_df: pd.DataFrame,
    chrom: str,
    components: list[tuple[int, int]],
) -> pd.DataFrame:
    pieces = []
    for start, end in components:
        overlap = cna_df[
            (cna_df["chrom"] == chrom) & (cna_df["start"] < end) & (cna_df["end"] > start)
        ]
        pieces.append(overlap)
    if not pieces:
        return cna_df.iloc[0:0].copy()
    return pd.concat(pieces).drop_duplicates().sort_values("start")


def _filter_breakpoints_by_components(
    breakpoints_df: pd.DataFrame,
    chrom: str,
    components: list[tuple[int, int]],
    padding: int = 0,
) -> pd.DataFrame:
    pieces = []
    for start, end in components:
        padded_start = max(0, start - padding)
        padded_end = end + padding
        overlap = breakpoints_df[
            (breakpoints_df["chrom"] == chrom)
            & (breakpoints_df["pos"] >= padded_start)
            & (breakpoints_df["pos"] <= padded_end)
        ]
        pieces.append(overlap)
    if not pieces:
        return breakpoints_df.iloc[0:0].copy()
    return (
        pd.concat(pieces)
        .drop_duplicates(subset=["chrom", "pos", "sv_id", "SV_TYPE", "st"])
        .sort_values(["pos", "sv_id"])
    )


def _classification_cna_segments(
    cna_df: pd.DataFrame,
    min_segment_size: int = CLASSIFICATION_MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    if cna_df.empty:
        return cna_df.copy()
    segment_lengths = cna_df["end"].astype(int) - cna_df["start"].astype(int) + 1
    return cna_df[segment_lengths > min_segment_size].copy()


def _breakpoints_matching_cna_boundaries(
    breakpoints_df: pd.DataFrame,
    cna_df: pd.DataFrame,
    wiggle: int = 100,
) -> pd.DataFrame:
    if breakpoints_df.empty or cna_df.empty:
        return breakpoints_df.iloc[0:0].copy()

    boundary_rows = []
    for chrom, chrom_breakpoints in breakpoints_df.groupby("chrom", sort=False):
        chrom_cna = cna_df[cna_df["chrom"] == chrom]
        if chrom_cna.empty:
            continue
        starts = chrom_cna["start"].astype(int)
        ends = chrom_cna["end"].astype(int)
        positions = chrom_breakpoints["pos"].astype(int)
        is_near_boundary = positions.map(
            lambda pos: ((starts - pos).abs().le(wiggle).any() or (ends - pos).abs().le(wiggle).any())
        )
        boundary_rows.append(chrom_breakpoints[is_near_boundary])

    if not boundary_rows:
        return breakpoints_df.iloc[0:0].copy()
    return pd.concat(boundary_rows).sort_values(["chrom", "pos", "sv_id"])


def _calculate_ploidy_for_components(
    cna_df: pd.DataFrame,
    chrom: str,
    components: list[tuple[int, int]],
) -> float:
    weighted_sum = 0
    total_length = 0
    for start, end in components:
        overlap = cna_df[
            (cna_df["chrom"] == chrom) & (cna_df["start"] <= end) & (cna_df["end"] >= start)
        ]
        if overlap.empty:
            continue
        overlap_start = overlap["start"].clip(lower=start)
        overlap_end = overlap["end"].clip(upper=end)
        lengths = (overlap_end - overlap_start + 1).clip(lower=0)
        weighted_sum += (lengths * overlap["TCN"]).sum()
        total_length += lengths.sum()
    if total_length == 0:
        return float("nan")
    return float(weighted_sum / total_length)


def read_centromere_bed(centromere_bed: str | Path) -> pd.DataFrame:
    """Read a BED file with chrom, centromere start, and centromere end."""
    return pd.read_csv(
        centromere_bed,
        sep="\t",
        header=None,
        usecols=[0, 1, 2],
        names=["chrom", "cen_start", "cen_end"],
    )


def _chrom_arm(chrom: str, start: int, end: int, centromeres_df: pd.DataFrame | None) -> str:
    if centromeres_df is None:
        return "whole"

    centromere = centromeres_df.loc[centromeres_df["chrom"] == chrom]
    if centromere.empty:
        return "whole"

    cen_start = int(centromere.iloc[0]["cen_start"])
    cen_end = int(centromere.iloc[0]["cen_end"])
    if end < cen_start:
        return "p"
    if start > cen_end:
        return "q"
    return "cen"


def _normalize_centromeres(
    centromeres: pd.DataFrame | str | Path | None,
) -> pd.DataFrame | None:
    if centromeres is not None and not isinstance(centromeres, pd.DataFrame):
        return read_centromere_bed(centromeres)
    return centromeres


def _split_interval_by_arm(
    chrom: str,
    start: int,
    end: int,
    centromeres_df: pd.DataFrame | None,
) -> list[tuple[str, int, int]]:
    if centromeres_df is None:
        return [("whole", start, end)]

    centromere = centromeres_df.loc[centromeres_df["chrom"] == chrom]
    if centromere.empty:
        return [("whole", start, end)]

    cen_start = int(centromere.iloc[0]["cen_start"])
    cen_end = int(centromere.iloc[0]["cen_end"])
    pieces = []
    if cen_start > 1 and start < cen_start:
        pieces.append(("p", start, min(end, cen_start - 1)))
    if end > cen_end:
        pieces.append(("q", max(start, cen_end + 1), end))

    return [piece for piece in pieces if piece[1] <= piece[2]]


def count_cna_segments_in_windows(
    cna_df: pd.DataFrame,
    window_size: int = 50_000_000,
    step_size: int = 10_000_000,
    frequency_threshold: float = 1 / 2_000_000,
    centromeres: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Count CNA segments in sliding windows.

    A segment is counted for a window when the segment overlaps the window.
    State-specific columns are grouped by TCN value.
    Windows are tagged as candidates when n_segments / window_length is below
    frequency_threshold.
    """
    required_columns = {"chrom", "start", "end", "cna_id", "TCN"}
    missing_columns = required_columns - set(cna_df.columns)
    if missing_columns:
        raise ValueError(f"cna_df is missing columns: {', '.join(sorted(missing_columns))}")

    centromeres = _normalize_centromeres(centromeres)

    cna_df = cna_df.copy()
    window_rows = []

    for chrom, chrom_df in cna_df.groupby("chrom", sort=False):
        chrom_df = chrom_df.sort_values("start")
        chrom_start = int(chrom_df["start"].min())
        chrom_end = int(chrom_df["end"].max())

        for window_start in range(chrom_start, chrom_end + 1, step_size):
            window_end = window_start + window_size - 1
            overlapping = chrom_df[
                (chrom_df["start"] <= window_end) & (chrom_df["end"] >= window_start)
            ]

            row = {
                "chrom": chrom,
                "window_start": window_start,
                "window_end": min(window_end, chrom_end),
                "n_segments": len(overlapping),
            }
            window_length = row["window_end"] - row["window_start"] + 1
            row["segment_frequency"] = row["n_segments"] / window_length
            row["cand"] = row["segment_frequency"] < frequency_threshold
            row.update(_count_tcn_bins(overlapping["TCN"]))
            window_rows.append(row)

            if window_end >= chrom_end:
                break

    return pd.DataFrame(window_rows)


def _split_window_by_arm(
    row: dict,
    centromeres_df: pd.DataFrame | None,
) -> list[dict]:
    chrom = row["chrom"]
    start = int(row["window_start"])
    end = int(row["window_end"])

    if centromeres_df is None:
        return [{**row, "arm": "whole", "arm_start": start, "arm_end": end}]

    centromere = centromeres_df.loc[centromeres_df["chrom"] == chrom]
    if centromere.empty:
        return [{**row, "arm": "whole", "arm_start": start, "arm_end": end}]

    cen_start = int(centromere.iloc[0]["cen_start"])
    cen_end = int(centromere.iloc[0]["cen_end"])
    pieces = []

    if cen_start > 1 and start < cen_start:
        pieces.append(
            {
                **row,
                "arm": "p",
                "arm_start": start,
                "arm_end": min(end, cen_start - 1),
            }
        )
    if end > cen_end:
        pieces.append(
            {
                **row,
                "arm": "q",
                "arm_start": max(start, cen_end + 1),
                "arm_end": end,
            }
        )

    return [piece for piece in pieces if piece["arm_start"] <= piece["arm_end"]]


def merge_candidate_windows(
    windows_df: pd.DataFrame,
    centromeres: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Merge consecutive candidate windows on the same chromosome."""
    required_columns = {"chrom", "window_start", "window_end", "cand"}
    missing_columns = required_columns - set(windows_df.columns)
    if missing_columns:
        raise ValueError(f"windows_df is missing columns: {', '.join(sorted(missing_columns))}")

    centromeres = _normalize_centromeres(centromeres)

    merged_rows = []
    split_rows = []
    for row in windows_df.sort_values(["chrom", "window_start"]).to_dict("records"):
        split_rows.extend(_split_window_by_arm(row, centromeres))

    split_df = pd.DataFrame(split_rows)
    if split_df.empty:
        return pd.DataFrame(columns=["chrom", "arm", "start", "end", "n_windows"])

    for (chrom, arm), chrom_df in split_df.groupby(["chrom", "arm"], sort=False):
        active = None
        chrom_df = chrom_df.sort_values("arm_start")

        for row in chrom_df.to_dict("records"):
            if not row["cand"]:
                if active is not None:
                    merged_rows.append(active)
                    active = None
                continue

            if active is None:
                active = {
                    "chrom": chrom,
                    "arm": arm,
                    "start": int(row["arm_start"]),
                    "end": int(row["arm_end"]),
                    "n_windows": 1,
                }
            else:
                active["end"] = max(active["end"], int(row["arm_end"]))
                active["n_windows"] += 1

        if active is not None:
            merged_rows.append(active)

    result = pd.DataFrame(merged_rows, columns=["chrom", "arm", "start", "end", "n_windows"])
    return _natural_sort_dataframe(result, ["start", "end"])


def find_small_segment_candidate_intervals(
    cna_df: pd.DataFrame,
    max_segment_size: int = 5_000_000,
    centromeres: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Find candidate intervals from consecutive CNA segments under max_segment_size."""
    required_columns = {"chrom", "start", "end"}
    missing_columns = required_columns - set(cna_df.columns)
    if missing_columns:
        raise ValueError(f"cna_df is missing columns: {', '.join(sorted(missing_columns))}")

    centromeres = _normalize_centromeres(centromeres)
    pieces = []
    for row in cna_df.sort_values(["chrom", "start"]).to_dict("records"):
        chrom = row["chrom"]
        start = int(row["start"])
        end = int(row["end"])
        segment_length = end - start + 1
        for arm, arm_start, arm_end in _split_interval_by_arm(chrom, start, end, centromeres):
            pieces.append(
                {
                    "chrom": chrom,
                    "arm": arm,
                    "start": arm_start,
                    "end": arm_end,
                    "segment_length": segment_length,
                    "is_small": segment_length < max_segment_size,
                }
            )

    pieces_df = pd.DataFrame(pieces)
    if pieces_df.empty:
        return pd.DataFrame(columns=["chrom", "arm", "start", "end", "n_windows"])

    intervals = []
    for (chrom, arm), arm_df in pieces_df.groupby(["chrom", "arm"], sort=False):
        active = None
        arm_df = arm_df.sort_values("start")

        for row in arm_df.to_dict("records"):
            if not row["is_small"]:
                if active is not None:
                    intervals.append(active)
                    active = None
                continue

            if active is None:
                active = {
                    "chrom": chrom,
                    "arm": arm,
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "n_windows": 1,
                }
            else:
                active["end"] = max(active["end"], int(row["end"]))
                active["n_windows"] += 1

        if active is not None:
            intervals.append(active)

    result = pd.DataFrame(intervals, columns=["chrom", "arm", "start", "end", "n_windows"])
    return _natural_sort_dataframe(result, ["start", "end"])


def _summarize_candidate_intervals(
    intervals_df: pd.DataFrame,
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    sample_ploidy: float | None = None,
    apply_candidate_filter: bool = True,
) -> pd.DataFrame:
    if intervals_df.empty:
        return intervals_df

    cna_df = cna_df.copy()
    breakpoints_df = breakpoints_df.copy()
    if sample_ploidy is None:
        sample_ploidy = calculate_ploidy(cna_df)
    sv_types = sorted(set(FIXED_SV_TYPES).union(breakpoints_df["SV_TYPE"].dropna().unique()))
    rows = []

    for interval in intervals_df.to_dict("records"):
        chrom = interval["chrom"]
        start = interval["start"]
        end = interval["end"]
        components = _component_intervals_from_row(interval)
        component_intervals = _component_intervals_to_string(components)

        cna_overlap = _filter_cna_by_components(cna_df, chrom, components)
        bp_overlap = _filter_breakpoints_by_components(
            breakpoints_df,
            chrom,
            components,
            padding=100,
        )
        bp_overlap = _breakpoints_matching_cna_boundaries(bp_overlap, cna_overlap)

        row = {
            "chrom": chrom,
            "arm": interval.get("arm", "whole"),
            "start": start,
            "end": end,
            "component_intervals": component_intervals,
            "n_windows": interval["n_windows"],
            "n_segments": len(cna_overlap),
            "n_segments_ge_100kb": len(_classification_cna_segments(cna_overlap)),
        }
        interval_length = sum(component_end - component_start + 1 for component_start, component_end in components)
        row["component_length"] = interval_length
        row["segment_frequency"] = row["n_segments"] / interval_length
        row["cand"] = True
        row["ploidy"] = _calculate_ploidy_for_components(cna_df, chrom, components)
        row["sample_ploidy"] = sample_ploidy
        segment_lengths = cna_overlap["end"] - cna_overlap["start"] + 1
        row["segment_len_q25"] = segment_lengths.quantile(0.25) if len(segment_lengths) else pd.NA
        row["segment_len_q50"] = segment_lengths.quantile(0.50) if len(segment_lengths) else pd.NA
        row["segment_len_q75"] = segment_lengths.quantile(0.75) if len(segment_lengths) else pd.NA
        row.update(_two_state_oscillation(cna_overlap))

        row.update(_count_tcn_bins(cna_overlap["TCN"]))

        row["n_breakpoints"] = len(bp_overlap)
        ordered_bp = bp_overlap.sort_values(["pos", "sv_id"]).reset_index(drop=True)
        fb_positions = ordered_bp.index[ordered_bp["SV_TYPE"] == "FB"] + 1
        row["FB_first_index"] = int(fb_positions.min()) if len(fb_positions) else pd.NA
        row["FB_last_index"] = int(fb_positions.max()) if len(fb_positions) else pd.NA
        fb_low_cn_positions = _fb_low_cn_indices(
            ordered_bp,
            cna_df,
            min_region_size=2_000_000,
        )
        row["n_FB_lowCN_2Mb"] = len(fb_low_cn_positions)
        row["FB_lowCN_first_index"] = (
            int(min(fb_low_cn_positions)) if fb_low_cn_positions else pd.NA
        )
        row["FB_lowCN_last_index"] = (
            int(max(fb_low_cn_positions)) if fb_low_cn_positions else pd.NA
        )
        sv_type_counts = bp_overlap["SV_TYPE"].value_counts()
        for sv_type in sv_types:
            row[f"n_{sv_type}"] = int(sv_type_counts.get(sv_type, 0))
        row["n_interchromosomal_SV"] = _count_interchromosomal_svs(bp_overlap, breakpoints_df)
        row.update(_classify_candidate_event(cna_overlap, bp_overlap, breakpoints_df, row))

        rows.append(row)

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    for sv_type in FIXED_SV_TYPES:
        column = f"n_{sv_type}"
        if column not in result.columns:
            result[column] = 0
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    if apply_candidate_filter:
        min_region_length = 2_000_000
        region_length = result["component_length"]
        result = result[region_length >= min_region_length]
        contains_fb = result["n_FB"].fillna(0) > 0
        has_high_tcn = result["ecDNA"].fillna(False).astype(bool)
        enough_segments = contains_fb | has_high_tcn | (result["n_segments_ge_100kb"] >= 4)
        result = result[enough_segments]
    return _natural_sort_dataframe(result, ["start", "end"])


def _classify_candidate_event(
    cna_overlap: pd.DataFrame,
    bp_overlap: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    row: dict,
) -> dict:
    connected_chromosomes = {row["chrom"]}
    sv_ids = bp_overlap["sv_id"].dropna().unique() if not bp_overlap.empty else []
    if len(sv_ids):
        connected = breakpoints_df[breakpoints_df["sv_id"].isin(sv_ids)]
        connected_chromosomes.update(connected["chrom"].dropna().unique())

    is_single_chromosome = len(connected_chromosomes) == 1

    classification_cna = _classification_cna_segments(cna_overlap)
    chromothripsis_oscillation = _two_state_oscillation(classification_cna)
    chromothripsis_multiple_cn_states = classification_cna["TCN"].nunique(dropna=True) > 1
    has_min_chromothripsis_segments = len(classification_cna) >= 4
    low_cn_multistate = chromothripsis_multiple_cn_states and bool(
        (classification_cna["TCN"] < 7).all()
    )
    multiple_cn_states = cna_overlap["TCN"].nunique(dropna=True) > 1
    sample_ploidy = row.get("sample_ploidy", pd.NA)
    ploidy_for_threshold = row.get("ploidy") if pd.isna(sample_ploidy) else sample_ploidy
    seismic_tcn_threshold = _seismic_tcn_threshold(ploidy_for_threshold)
    ecdna_tcn_threshold = _ecdna_tcn_threshold(sample_ploidy)
    multiple_high_cn_segments = int((cna_overlap["TCN"] >= seismic_tcn_threshold).sum()) > 1
    seismic_segment_spacing = _seismic_segment_spacing_passes(row)
    seismic = multiple_cn_states and multiple_high_cn_segments and seismic_segment_spacing
    n_segments = len(classification_cna)
    interchromosomal_sv_limit = 3 if n_segments > 7 else 2
    canonical_interchromosomal_sv_count = (
        row.get("n_interchromosomal_SV", 0) < interchromosomal_sv_limit
    )
    has_canonical_chromothripsis_sv_types = (
        row.get("n_DEL", 0) > 0
        and row.get("n_DUP", 0) > 0
        and row.get("n_INV_LIKE", 0) > 0
    )
    if (
        has_canonical_chromothripsis_sv_types
        and chromothripsis_oscillation["oscillating_two_state"]
        and has_min_chromothripsis_segments
    ):
        if (
            canonical_interchromosomal_sv_count
        ):
            chromothripsis = "canonical"
        else:
            chromothripsis = "noncanonical"
    elif (
        has_canonical_chromothripsis_sv_types
        and low_cn_multistate
        and has_min_chromothripsis_segments
    ):
        chromothripsis = "noncanonical"
    elif (
        chromothripsis_oscillation["oscillating_two_state"]
        and has_min_chromothripsis_segments
    ):
        chromothripsis = "noncanonicalB"
    else:
        chromothripsis = pd.NA

    bfb_at_edge = (
        row["n_FB_lowCN_2Mb"] > 0
        and (
            _same_value(row["FB_lowCN_first_index"], 1)
            or _same_value(row["FB_lowCN_last_index"], row["n_breakpoints"])
        )
    )
    only_foldbacks = not bp_overlap.empty and bp_overlap["SV_TYPE"].eq("FB").all()
    has_foldback = row.get("n_FB", 0) > 0
    if bfb_at_edge:
        bfb = _canonical_label(only_foldbacks)
    elif has_foldback:
        bfb = "noncanonicalB"
    else:
        bfb = pd.NA

    return {
        "ecDNA": bool((cna_overlap["TCN"] > ecdna_tcn_threshold).any()),
        "Seismic_Amplification": (
            _canonical_label(is_single_chromosome) if seismic else pd.NA
        ),
        "chromothripsis": chromothripsis,
        "BFB": bfb,
    }


def _aggregate_cluster_for_classification(
    cluster_rows: pd.DataFrame,
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    sample_ploidy: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    cna_pieces = []
    bp_pieces = []
    for _, region in cluster_rows.iterrows():
        chrom = region["chrom"]
        components = _component_intervals_from_row(region)
        cna_overlap = _filter_cna_by_components(cna_df, chrom, components)
        cna_pieces.append(cna_overlap)

        bp_overlap = _filter_breakpoints_by_components(
            breakpoints_df,
            chrom,
            components,
            padding=100,
        )
        bp_overlap = _breakpoints_matching_cna_boundaries(bp_overlap, cna_overlap)
        bp_pieces.append(bp_overlap)

    cna_overlap = (
        pd.concat(cna_pieces)
        .drop_duplicates(subset=["chrom", "start", "end", "TCN", "CN1", "CN2"])
        .pipe(_natural_sort_dataframe, ["start", "end"])
        if cna_pieces
        else cna_df.iloc[0:0].copy()
    )
    bp_overlap = (
        pd.concat(bp_pieces)
        .drop_duplicates(subset=["chrom", "pos", "sv_id", "SV_TYPE", "st"])
        .pipe(_natural_sort_dataframe, ["pos", "sv_id"])
        if bp_pieces
        else breakpoints_df.iloc[0:0].copy()
    )

    segment_lengths = cna_overlap["end"] - cna_overlap["start"] + 1
    component_length = int(cluster_rows["component_length"].sum()) if "component_length" in cluster_rows else int(segment_lengths.sum())
    n_segments = len(cna_overlap)
    row = {
        "chrom": cluster_rows.iloc[0]["chrom"],
        "ploidy": _calculate_ploidy_for_cluster(cna_overlap),
        "sample_ploidy": sample_ploidy,
        "n_segments": n_segments,
        "n_segments_ge_100kb": len(_classification_cna_segments(cna_overlap)),
        "component_length": component_length,
        "segment_frequency": n_segments / component_length if component_length else pd.NA,
        "segment_len_q50": segment_lengths.quantile(0.50) if len(segment_lengths) else pd.NA,
        "n_breakpoints": len(bp_overlap),
        "n_interchromosomal_SV": _count_interchromosomal_svs(bp_overlap, breakpoints_df),
    }
    row.update(_two_state_oscillation(cna_overlap))

    sv_type_counts = bp_overlap["SV_TYPE"].value_counts()
    for sv_type in sorted(set(FIXED_SV_TYPES).union(breakpoints_df["SV_TYPE"].dropna().unique())):
        row[f"n_{sv_type}"] = int(sv_type_counts.get(sv_type, 0))

    ordered_bp = _natural_sort_dataframe(bp_overlap, ["pos", "sv_id"]).reset_index(drop=True)
    fb_positions = ordered_bp.index[ordered_bp["SV_TYPE"] == "FB"] + 1
    fb_low_cn_positions = _fb_low_cn_indices(
        ordered_bp,
        cna_df,
        min_region_size=2_000_000,
    )
    row["n_FB"] = int(sv_type_counts.get("FB", 0))
    row["n_FB_lowCN_2Mb"] = len(fb_low_cn_positions)
    row["FB_lowCN_first_index"] = (
        int(min(fb_low_cn_positions)) if fb_low_cn_positions else pd.NA
    )
    row["FB_lowCN_last_index"] = (
        int(max(fb_low_cn_positions)) if fb_low_cn_positions else pd.NA
    )
    row["FB_first_index"] = int(fb_positions.min()) if len(fb_positions) else pd.NA
    row["FB_last_index"] = int(fb_positions.max()) if len(fb_positions) else pd.NA
    return cna_overlap, bp_overlap, row


def _calculate_ploidy_for_cluster(cna_overlap: pd.DataFrame) -> float:
    if cna_overlap.empty:
        return float("nan")
    lengths = cna_overlap["end"] - cna_overlap["start"] + 1
    total_length = lengths.sum()
    if total_length == 0:
        return float("nan")
    return float((lengths * cna_overlap["TCN"]).sum() / total_length)


def classify_linked_clusters(
    candidate_df: pd.DataFrame,
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    sample_ploidy: float | None = None,
) -> pd.DataFrame:
    """Classify linked candidate rows together and propagate shared annotations."""
    if candidate_df.empty or "cluster_id" not in candidate_df.columns:
        return candidate_df

    if sample_ploidy is None:
        sample_ploidy = calculate_ploidy(cna_df)

    result = candidate_df.copy()
    annotation_columns = ["ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"]
    for cluster_id, cluster_rows in result.groupby("cluster_id", sort=False):
        cna_overlap, bp_overlap, cluster_row = _aggregate_cluster_for_classification(
            cluster_rows,
            cna_df,
            breakpoints_df,
            sample_ploidy,
        )
        annotations = _classify_candidate_event(
            cna_overlap,
            bp_overlap,
            breakpoints_df,
            cluster_row,
        )
        cluster_mask = result["cluster_id"] == cluster_id
        for column in annotation_columns:
            result.loc[cluster_mask, column] = annotations[column]
    return result


def _canonical_label(is_canonical: bool) -> str:
    return "canonical" if is_canonical else "noncanonical"


def _seismic_tcn_threshold(ploidy: float | None) -> int:
    if ploidy is None or pd.isna(ploidy):
        return 7
    return 7 if float(ploidy) < 3.5 else 8


def _seismic_segment_spacing_passes(row: dict | pd.Series) -> bool:
    component_length = row.get("component_length", pd.NA)
    n_segments = row.get("n_segments", pd.NA)
    if pd.isna(component_length) or pd.isna(n_segments) or n_segments == 0:
        return False
    return (float(component_length) / float(n_segments)) < 1_500_000


def _ecdna_tcn_threshold(sample_ploidy: float | None) -> float:
    if sample_ploidy is None or pd.isna(sample_ploidy):
        return 20
    return 10 * float(sample_ploidy)


def _count_interchromosomal_svs(bp_overlap: pd.DataFrame, breakpoints_df: pd.DataFrame) -> int:
    if bp_overlap.empty:
        return 0

    count = 0
    for sv_id in bp_overlap["sv_id"].dropna().unique():
        sv_rows = breakpoints_df[breakpoints_df["sv_id"] == sv_id]
        if sv_rows["chrom"].nunique(dropna=True) > 1:
            count += 1
    return count


def _two_state_oscillation(cna_overlap: pd.DataFrame) -> dict:
    result = {
        "oscillating_two_state": False,
        "oscillating_tcn_states": pd.NA,
        "oscillating_segment_fraction": 0.0,
        "oscillating_transition_fraction": 0.0,
    }
    if len(cna_overlap) < 3:
        return result

    ordered = cna_overlap.sort_values("start")
    tcn_values = ordered["TCN"].tolist()
    counts = ordered["TCN"].value_counts()
    if len(counts) < 2:
        return result

    states = list(counts.index[:2])
    state_set = set(states)
    segment_fraction = sum(tcn in state_set for tcn in tcn_values) / len(tcn_values)

    transitions = list(zip(tcn_values[:-1], tcn_values[1:]))
    alternating_transitions = sum(
        left in state_set and right in state_set and left != right for left, right in transitions
    )
    transition_fraction = alternating_transitions / len(transitions) if transitions else 0.0

    result["oscillating_two_state"] = segment_fraction > 0.5 and transition_fraction > 0.5
    result["oscillating_tcn_states"] = "|".join(_format_tcn_label(state) for state in states)
    result["oscillating_segment_fraction"] = segment_fraction
    result["oscillating_transition_fraction"] = transition_fraction
    return result


def _fb_segment_tcn(cna_df: pd.DataFrame, chrom: str, pos: int) -> float | None:
    containing = cna_df[
        (cna_df["chrom"] == chrom) & (cna_df["start"] <= pos) & (cna_df["end"] >= pos)
    ]
    if containing.empty:
        return None
    return float(containing.iloc[0]["TCN"])


def _tcn_difference_coverage(
    cna_df: pd.DataFrame,
    chrom: str,
    start: int,
    end: int,
    fb_tcn: float,
    min_tcn_difference: float,
) -> int:
    if start > end:
        return 0

    overlapping = cna_df[
        (cna_df["chrom"] == chrom)
        & ((cna_df["TCN"] - fb_tcn).abs() >= min_tcn_difference)
        & (cna_df["start"] <= end)
        & (cna_df["end"] >= start)
    ]
    if overlapping.empty:
        return 0

    overlap_start = overlapping["start"].clip(lower=start)
    overlap_end = overlapping["end"].clip(upper=end)
    return int((overlap_end - overlap_start + 1).clip(lower=0).sum())


def _fb_has_directional_low_cn(
    fb_row: pd.Series,
    cna_df: pd.DataFrame,
    min_region_size: int = 2_000_000,
    min_tcn_difference: float = 2,
) -> bool:
    chrom = fb_row["chrom"]
    pos = int(fb_row["pos"])
    strand = fb_row.get("st")
    fb_tcn = _fb_segment_tcn(cna_df, chrom, pos)
    if fb_tcn is None:
        return False

    if strand == "+":
        start = pos + 1
        end = pos + min_region_size
    elif strand == "-":
        start = max(0, pos - min_region_size)
        end = pos - 1
    else:
        return False

    return (
        _tcn_difference_coverage(
            cna_df,
            chrom,
            start,
            end,
            fb_tcn=fb_tcn,
            min_tcn_difference=min_tcn_difference,
        )
        >= min_region_size
    )


def _fb_low_cn_indices(
    ordered_bp: pd.DataFrame,
    cna_df: pd.DataFrame,
    min_region_size: int = 2_000_000,
    min_tcn_difference: float = 2,
) -> list[int]:
    indices = []
    for index, row in ordered_bp.iterrows():
        if row["SV_TYPE"] != "FB":
            continue
        if _fb_has_directional_low_cn(
            row,
            cna_df,
            min_region_size=min_region_size,
            min_tcn_difference=min_tcn_difference,
        ):
            indices.append(int(index) + 1)
    return indices


def count_breakpoints_in_candidate_windows(
    windows_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add breakpoint counts to candidate CNA windows.

    Breakpoints are counted when their chrom/pos falls inside a candidate
    window. Non-candidate windows are kept and assigned zero breakpoints.
    """
    required_window_columns = {"chrom", "window_start", "window_end", "cand"}
    missing_window_columns = required_window_columns - set(windows_df.columns)
    if missing_window_columns:
        raise ValueError(
            f"windows_df is missing columns: {', '.join(sorted(missing_window_columns))}"
        )

    required_breakpoint_columns = {"chrom", "pos", "SV_TYPE"}
    missing_breakpoint_columns = required_breakpoint_columns - set(breakpoints_df.columns)
    if missing_breakpoint_columns:
        raise ValueError(
            f"breakpoints_df is missing columns: {', '.join(sorted(missing_breakpoint_columns))}"
        )

    windows_df = windows_df.copy()
    windows_df["n_breakpoints"] = 0
    windows_df["FB_first_index"] = pd.NA
    windows_df["FB_last_index"] = pd.NA
    sv_types = sorted(breakpoints_df["SV_TYPE"].dropna().unique())
    for sv_type in sv_types:
        windows_df[f"n_{sv_type}"] = 0

    for chrom, chrom_windows in windows_df[windows_df["cand"]].groupby("chrom", sort=False):
        chrom_breakpoints = breakpoints_df.loc[breakpoints_df["chrom"] == chrom]
        if chrom_breakpoints.empty:
            continue

        for index, window in chrom_windows.iterrows():
            in_window = (
                (chrom_breakpoints["pos"] >= window["window_start"])
                & (chrom_breakpoints["pos"] <= window["window_end"])
            )
            overlapping = chrom_breakpoints[in_window]
            windows_df.at[index, "n_breakpoints"] = len(overlapping)
            ordered_bp = overlapping.sort_values(["pos", "sv_id"]).reset_index(drop=True)
            fb_positions = ordered_bp.index[ordered_bp["SV_TYPE"] == "FB"] + 1
            if len(fb_positions):
                windows_df.at[index, "FB_first_index"] = int(fb_positions.min())
                windows_df.at[index, "FB_last_index"] = int(fb_positions.max())

            sv_type_counts = overlapping["SV_TYPE"].value_counts()
            for sv_type in sv_types:
                windows_df.at[index, f"n_{sv_type}"] = int(sv_type_counts.get(sv_type, 0))

    return windows_df


def merge_candidate_intervals_by_sv_or_distance(
    intervals_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    max_distance: int = 10_000_000,
) -> pd.DataFrame:
    """Merge same-chromosome candidate intervals only when they are close by distance."""
    if intervals_df.empty:
        return intervals_df

    merged_rows = []
    intervals_df = _natural_sort_dataframe(intervals_df, ["start", "end"])
    for chrom, chrom_df in intervals_df.groupby("chrom", sort=False):
        rows = chrom_df.sort_values("start").to_dict("records")
        active = rows[0].copy()
        active["component_intervals"] = _component_intervals_to_string(
            _component_intervals_from_row(active)
        )

        for row in rows[1:]:
            row["component_intervals"] = _component_intervals_to_string(
                _component_intervals_from_row(row)
            )
            if _candidate_intervals_within_distance(active, row, max_distance):
                merged_start = min(int(active["start"]), int(row["start"]))
                merged_end = max(int(active["end"]), int(row["end"]))
                active["end"] = max(int(active["end"]), int(row["end"]))
                active["start"] = merged_start
                active["n_windows"] = int(active.get("n_windows", 1)) + int(row.get("n_windows", 1))
                active["arm"] = (
                    active.get("arm", "whole")
                    if active.get("arm", "whole") == row.get("arm", "whole")
                    else "mixed"
                )
                active_components = _component_intervals_from_row(active)
                row_components = _component_intervals_from_row(row)
                components = active_components[:-1] + [
                    (active_components[-1][0], row_components[-1][1])
                ]
                active["component_intervals"] = _component_intervals_to_string(
                    _merge_component_intervals(components)
                )
            else:
                merged_rows.append(active)
                active = row.copy()
                active["component_intervals"] = _component_intervals_to_string(
                    _component_intervals_from_row(active)
                )

        merged_rows.append(active)

    result_columns = list(intervals_df.columns)
    if "component_intervals" not in result_columns:
        result_columns.append("component_intervals")
    result = pd.DataFrame(merged_rows, columns=result_columns)
    if "component_intervals" not in result.columns:
        result["component_intervals"] = result.apply(
            lambda row: _component_intervals_to_string(_component_intervals_from_row(row)),
            axis=1,
        )
    return _natural_sort_dataframe(result, ["start", "end"])


def _candidate_intervals_within_distance(
    left: dict,
    right: dict,
    max_distance: int,
) -> bool:
    distance = int(right["start"]) - int(left["end"])
    return distance < max_distance


def _candidate_intervals_have_connecting_sv(
    left: dict,
    right: dict,
    breakpoints_df: pd.DataFrame,
) -> bool:
    chrom = left["chrom"]
    if chrom != right["chrom"]:
        return False

    chrom_breakpoints = breakpoints_df[breakpoints_df["chrom"] == chrom]
    left_bp = _filter_breakpoints_by_components(
        chrom_breakpoints,
        chrom,
        _component_intervals_from_row(left),
    )
    left_sv_ids = set(left_bp["sv_id"])
    if not left_sv_ids:
        return False

    right_bp = _filter_breakpoints_by_components(
        chrom_breakpoints,
        chrom,
        _component_intervals_from_row(right),
    )
    right_sv_ids = set(right_bp["sv_id"])
    return bool(left_sv_ids & right_sv_ids)


def assign_linked_cluster_ids(
    candidate_df: pd.DataFrame,
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    min_interchrom_connecting_svs: int = 3,
    min_intrachrom_connecting_svs: int = 1,
) -> pd.DataFrame:
    """Assign shared cluster IDs to candidate regions linked by SVs."""
    if candidate_df.empty:
        result = candidate_df.copy()
        result["cluster_id"] = pd.Series(dtype="object")
        return result

    result = candidate_df.reset_index(drop=True).copy()
    region_sv_ids = []
    for _, region in result.iterrows():
        chrom = region["chrom"]
        components = _component_intervals_from_row(region)
        cna_overlap = _filter_cna_by_components(cna_df, chrom, components)
        bp_overlap = _filter_breakpoints_by_components(
            breakpoints_df,
            chrom,
            components,
            padding=100,
        )
        bp_overlap = _breakpoints_matching_cna_boundaries(bp_overlap, cna_overlap)
        region_sv_ids.append(set(bp_overlap["sv_id"].dropna()))

    parent = list(range(len(result)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(result)):
        for right in range(left + 1, len(result)):
            connecting_svs = region_sv_ids[left] & region_sv_ids[right]
            min_connecting_svs = (
                min_intrachrom_connecting_svs
                if result.at[left, "chrom"] == result.at[right, "chrom"]
                else min_interchrom_connecting_svs
            )
            if len(connecting_svs) >= min_connecting_svs:
                union(left, right)

    cluster_names = {}
    cluster_ids = []
    for index in range(len(result)):
        root = find(index)
        if root not in cluster_names:
            cluster_names[root] = f"C{len(cluster_names) + 1}"
        cluster_ids.append(cluster_names[root])
    if "cluster_id" in result.columns:
        result["cluster_id"] = cluster_ids
    else:
        result.insert(0, "cluster_id", cluster_ids)
    return result


def merge_sv_cna_candidate_windows(
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    window_size: int = 50_000_000,
    step_size: int = 10_000_000,
    frequency_threshold: float = 1 / 2_000_000,
    centromeres: pd.DataFrame | str | Path | None = None,
) -> pd.DataFrame:
    """Return candidate windows with both CNA and SV summary columns."""
    windows_df = count_cna_segments_in_windows(
        cna_df,
        window_size=window_size,
        step_size=step_size,
        frequency_threshold=frequency_threshold,
        centromeres=centromeres,
    )
    merged_df = count_breakpoints_in_candidate_windows(windows_df, breakpoints_df)
    return merged_df.loc[merged_df["cand"]].reset_index(drop=True)


def merge_sv_cna_candidate_segments(
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    window_size: int = 50_000_000,
    step_size: int = 10_000_000,
    frequency_threshold: float = 1 / 2_000_000,
    centromeres: pd.DataFrame | str | Path | None = None,
    method: str = "sliding",
    max_segment_size: int = 5_000_000,
    region_merge_distance: int = 10_000_000,
) -> pd.DataFrame:
    """Return merged candidate intervals with CNA and SV summary columns."""
    sample_ploidy = calculate_ploidy(cna_df)
    if method == "sliding":
        windows_df = count_cna_segments_in_windows(
            cna_df,
            window_size=window_size,
            step_size=step_size,
            frequency_threshold=frequency_threshold,
            centromeres=centromeres,
        )
        intervals_df = merge_candidate_windows(windows_df, centromeres=centromeres)
    elif method == "small_segments":
        intervals_df = find_small_segment_candidate_intervals(
            cna_df,
            max_segment_size=max_segment_size,
            centromeres=centromeres,
        )
    else:
        raise ValueError("method must be 'sliding' or 'small_segments'")

    candidate_df = _summarize_candidate_intervals(
        intervals_df,
        cna_df,
        breakpoints_df,
        sample_ploidy=sample_ploidy,
    )
    if candidate_df.empty:
        return candidate_df

    merged_intervals = merge_candidate_intervals_by_sv_or_distance(
        candidate_df[["chrom", "arm", "start", "end", "n_windows"]],
        breakpoints_df,
        max_distance=region_merge_distance,
    )
    final_df = _summarize_candidate_intervals(
        merged_intervals,
        cna_df,
        breakpoints_df,
        sample_ploidy=sample_ploidy,
    )
    final_df = assign_linked_cluster_ids(final_df, cna_df, breakpoints_df)
    final_df = classify_linked_clusters(
        final_df,
        cna_df,
        breakpoints_df,
        sample_ploidy=sample_ploidy,
    )
    return final_df


def get_bps(vcf_file):
    MIN_DEL_DUP_SIZE = 10_000
    PHASE_THR = 2_000_000

    refs, pos1, hp, sv_supp, dir1, sv_id, sv_type = [], [], [], [], [], [], []
    ref_reads, supp_reads = [], []

    hp1 = hp2 = 0

    with Path(vcf_file).open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue

            (
                chrom,
                pos,
                record_id,
                _,
                alt,
                _,
                _,
                info_field,
                format_field,
                sample_field,
            ) = fields[:10]
            pos = int(pos)
            info = _parse_info(info_field)
            sample = _parse_sample(format_field, sample_field)

            svt = info.get("SVTYPE")
            if svt in ("INS", "INV"):
                continue
            if info.get("INSIDE_VNTR"):
                continue
            if info.get("DETAILED_TYPE") == "reciprocal_inversion":
                continue

            if info.get("HP"):
                hp1, hp2 = [int(h) for h in str(info.get("HP")).split("|")]

            dir_ls = str(info.get("STRANDS", ""))

            if svt == "BND":
                if "_2" in record_id:
                    continue
                chr2, pos2 = _parse_bnd_alt(alt)
            elif svt in ("DEL", "DUP", "INV"):
                svlen = _as_int(info.get("SVLEN"))
                if svt in ("DEL", "DUP") and svlen is not None and abs(svlen) < MIN_DEL_DUP_SIZE:
                    continue
                pos2 = _as_int(info.get("END"))
                if pos2 is None:
                    if svlen is None:
                        continue
                    pos2 = pos + svlen
                chr2 = chrom
            elif svt == "sBND":
                chr2, pos2 = chrom, pos
            else:
                continue

            # tighten phase if nearby on same chrom
            if chr2 == chrom and abs(int(pos2) - int(pos)) < PHASE_THR and hp1 != hp2:
                nonzero = [h for h in (hp1, hp2) if h != 0]
                if nonzero:
                    hp1 = hp2 = nonzero[0]

            dv = int(sample["DV"]) if sample.get("DV") not in (None, ".") else 0

            refs += [chrom, chr2]
            pos1 += [pos, int(pos2)]
            hp += [int(hp1), int(hp2)]
            sv_supp += [dv, dv]
            dir1 += [dir_ls[0] if dir_ls else "", dir_ls[-1] if dir_ls else ""]
            sv_id += [record_id, record_id]
            sv_type += [_get_sv_type(info), _get_sv_type(info)]

            dv_reads = _split_read_counts(info.get("SUPP_READS"))
            dr_reads = _split_read_counts(info.get("REF_READS"))
            ref_reads += [dr_reads[:3], dr_reads[3:]]
            supp_reads += [dv_reads[:3], dv_reads[3:]]

    result = pd.DataFrame(
        {
            "chrom": refs,
            "pos": pos1,
            "hp": hp,
            "supp": sv_supp,
            "st": dir1,
            "sv_id": sv_id,
            "SV_TYPE": sv_type,
            "ref_reads": ref_reads,
            "supp_reads": supp_reads,
        }
    )
    return _natural_sort_dataframe(result, ["pos", "sv_id"])


def _arc_points(x1: float, x2: float, base_y: float = 0, height: float = 1, n: int = 40):
    if x1 == x2:
        x2 = x1 + 1
    xs = [x1 + (x2 - x1) * i / (n - 1) for i in range(n)]
    mid = (x1 + x2) / 2
    half_width = abs(x2 - x1) / 2
    ys = [
        base_y + height * max(0, 1 - ((x - mid) / half_width) ** 2)
        for x in xs
    ]
    return xs, ys


def _chrom_plot_length(cna_df: pd.DataFrame, chrom: str) -> int | None:
    chrom_cna = cna_df[cna_df["chrom"] == chrom]
    if chrom_cna.empty:
        return None
    return int(chrom_cna["end"].max()) + 1


def _plot_flank_intervals(
    region: pd.Series,
    cna_df: pd.DataFrame,
    flank_size: int = PLOT_FLANK_SIZE,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    chrom_length = _chrom_plot_length(cna_df, region["chrom"])
    start = int(region["start"])
    end = int(region["end"])

    left_start = max(0, start - flank_size)
    left_end = start - 1
    left = [(left_start, left_end)] if left_start <= left_end else []

    right_start = end + 1
    right_end = end + flank_size
    if chrom_length is not None:
        right_end = min(chrom_length - 1, right_end)
    right = [(right_start, right_end)] if right_start <= right_end else []
    return left, right


def _region_x_offsets(
    candidate_df: pd.DataFrame,
    cna_df: pd.DataFrame | None = None,
    flank_size: int = 0,
    gap: int = 5_000_000,
) -> dict[int, int]:
    offsets = {}
    current = 0
    for index, region in candidate_df.reset_index(drop=True).iterrows():
        offsets[index] = current
        components = _component_intervals_from_row(region)
        region_length = _intervals_length(components)
        if cna_df is not None and flank_size > 0:
            left_flank, right_flank = _plot_flank_intervals(region, cna_df, flank_size)
            region_length += _intervals_length(left_flank) + _intervals_length(right_flank)
        current += region_length + gap
    return offsets


def _component_x_position(
    pos: int,
    components: list[tuple[int, int]],
    offset: int,
    wiggle: int = 0,
) -> int | None:
    current = offset
    for start, end in components:
        if start <= pos <= end:
            return current + pos - start
        if wiggle and start - wiggle <= pos < start:
            return current
        if wiggle and end < pos <= end + wiggle:
            return current + end - start
        current += end - start + 1
    return None


def _breakpoint_in_components(
    row: pd.Series,
    chrom: str,
    components: list[tuple[int, int]],
    wiggle: int = 0,
) -> bool:
    if row["chrom"] != chrom:
        return False
    pos = int(row["pos"])
    return any(start - wiggle <= pos <= end + wiggle for start, end in components)


def _single_breakpoint_mate_label(
    original_sv_rows: pd.DataFrame,
    chrom: str,
    components: list[tuple[int, int]],
    wiggle: int = 100,
) -> str:
    mate_rows = original_sv_rows[
        ~original_sv_rows.apply(
            lambda row: _breakpoint_in_components(row, chrom, components, wiggle=wiggle),
            axis=1,
        )
    ]
    if mate_rows.empty:
        return "unpaired"

    interchrom_mates = mate_rows[mate_rows["chrom"] != chrom]
    mate = interchrom_mates.iloc[0] if not interchrom_mates.empty else mate_rows.iloc[0]
    return f"to {mate['chrom']}:{int(mate['pos'])}"


def _component_x_ranges(
    start: int,
    end: int,
    components: list[tuple[int, int]],
    offset: int,
) -> list[tuple[int, int]]:
    ranges = []
    current = offset
    for component_start, component_end in components:
        overlap_start = max(start, component_start)
        overlap_end = min(end, component_end)
        if overlap_start <= overlap_end:
            ranges.append(
                (
                    current + overlap_start - component_start,
                    current + overlap_end - component_start,
                )
            )
        current += component_end - component_start + 1
    return ranges


EVENT_TYPE_COLORS = {
    "ecDNA": "#e15759",
    "Seismic": "#f28e2b",
    "Chromothripsis": "#4e79a7",
    "BFB": "#59a14f",
}

EVENT_LABEL_SYMBOLS = {
    "canonical": "circle",
    "noncanonical": "diamond",
    "noncanonicalB": "x",
    "noncanonical_B": "x",
    "ecDNA": "square",
}


def _event_marker_symbol(event_label: str) -> str:
    return EVENT_LABEL_SYMBOLS.get(event_label, "circle")


def _event_labels_for_plot(region: pd.Series) -> list[tuple[str, int, str]]:
    labels = []
    if bool(region.get("ecDNA", False)):
        labels.append(("ecDNA", 1, "ecDNA"))

    seismic = region.get("Seismic_Amplification", pd.NA)
    if not pd.isna(seismic):
        labels.append((str(seismic), 2, "Seismic"))

    chromothripsis = region.get("chromothripsis", pd.NA)
    if not pd.isna(chromothripsis):
        labels.append((str(chromothripsis), 3, "Chromothripsis"))

    bfb = region.get("BFB", pd.NA)
    if not pd.isna(bfb):
        labels.append((str(bfb), 4, "BFB"))

    return labels


def generate_candidate_plot(
    candidate_df: pd.DataFrame,
    cna_df: pd.DataFrame,
    breakpoints_df: pd.DataFrame,
    output_html: str | Path,
    max_regions: int | None = None,
) -> None:
    """Write an interactive Plotly HTML plot for candidate regions."""
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    candidate_df = candidate_df.reset_index(drop=True)
    if max_regions is not None:
        candidate_df = candidate_df.head(max_regions).reset_index(drop=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.48, 0.34, 0.18],
        subplot_titles=("SV arcs", "Total copy number", "Event classes"),
    )
    sample_ploidy_values = (
        candidate_df["sample_ploidy"].dropna().unique()
        if "sample_ploidy" in candidate_df.columns
        else []
    )
    sample_ploidy_label = (
        f"{float(sample_ploidy_values[0]):.3f}" if len(sample_ploidy_values) else "NA"
    )

    offsets = _region_x_offsets(candidate_df, cna_df=cna_df, flank_size=PLOT_FLANK_SIZE)
    legend_seen = set()
    region_tick_positions = []
    region_tick_labels = []
    cluster_order = (
        list(dict.fromkeys(candidate_df["cluster_id"].astype(str)))
        if "cluster_id" in candidate_df.columns
        else []
    )
    cluster_colors = {
        cluster_id: REGION_COLORS[index % len(REGION_COLORS)]
        for index, cluster_id in enumerate(cluster_order)
    }
    for event_type, event_color in EVENT_TYPE_COLORS.items():
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": event_color, "symbol": "circle"},
                name=f"{event_type} event",
                legendgroup="event-type",
                showlegend=True,
            ),
            row=3,
            col=1,
        )
    for event_label, event_symbol in [
        ("canonical", "circle"),
        ("noncanonical", "diamond"),
        ("noncanonicalB", "x"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": "#555555", "symbol": event_symbol},
                name=event_label,
                legendgroup="event-shape",
                showlegend=True,
            ),
            row=3,
            col=1,
        )

    for region_index, region in candidate_df.iterrows():
        chrom = region["chrom"]
        start = int(region["start"])
        end = int(region["end"])
        components = _component_intervals_from_row(region)
        component_length = _intervals_length(components)
        offset = offsets[region_index]
        left_flank, right_flank = _plot_flank_intervals(region, cna_df, PLOT_FLANK_SIZE)
        left_flank_length = _intervals_length(left_flank)
        event_offset = offset + left_flank_length
        right_flank_offset = event_offset + component_length
        cluster_id = str(region.get("cluster_id", region_index))
        region_color = cluster_colors.get(cluster_id, REGION_COLORS[region_index % len(REGION_COLORS)])
        region_label = f"{chrom}:{start}-{end}"
        region_ploidy = region.get("ploidy", pd.NA)
        region_ploidy_label = "NA" if pd.isna(region_ploidy) else f"{float(region_ploidy):.3f}"
        row_sample_ploidy = region.get("sample_ploidy", pd.NA)
        row_sample_ploidy_label = (
            "NA" if pd.isna(row_sample_ploidy) else f"{float(row_sample_ploidy):.3f}"
        )
        region_tick_positions.append(event_offset + component_length / 2)
        region_tick_labels.append(f"{region_label}<br>ploidy={region_ploidy_label}")

        for flank_intervals, flank_base_offset, flank_label in [
            (left_flank, offset, "5' flank"),
            (right_flank, right_flank_offset, "3' flank"),
        ]:
            flank_length = _intervals_length(flank_intervals)
            if flank_length == 0:
                continue
            for plot_row, opacity in [(1, 0.14), (2, 0.20), (3, 0.16)]:
                fig.add_vrect(
                    x0=flank_base_offset,
                    x1=flank_base_offset + flank_length - 1,
                    fillcolor=PLOT_FLANK_COLOR,
                    opacity=opacity,
                    line_width=0,
                    row=plot_row,
                    col=1,
                )

            flank_cna = _filter_cna_by_components(cna_df, chrom, flank_intervals)
            for _, segment in flank_cna.iterrows():
                y = float(segment["TCN"])
                segment_length = int(segment["end"]) - int(segment["start"]) + 1
                for x0, x1 in _component_x_ranges(
                    int(segment["start"]),
                    int(segment["end"]),
                    flank_intervals,
                    flank_base_offset,
                ):
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1],
                            y=[y, y],
                            mode="lines",
                            line={"color": "#8f8f8f", "width": 4},
                            hovertemplate=(
                                f"{region_label}<br>{flank_label}<br>"
                                f"{segment['chrom']}:{int(segment['start'])}-{int(segment['end'])}<br>"
                                f"length={segment_length:,} bp<br>"
                                f"region ploidy={region_ploidy_label}<br>"
                                f"sample ploidy={row_sample_ploidy_label}<br>"
                                f"TCN={segment['TCN']}<extra></extra>"
                            ),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )

        component_offset = event_offset
        for component_start, component_end in components:
            x0 = component_offset
            x1 = component_offset + component_end - component_start
            for plot_row, opacity in [(1, 0.08), (2, 0.12), (3, 0.10)]:
                fig.add_vrect(
                    x0=x0,
                    x1=x1,
                    fillcolor=region_color,
                    opacity=opacity,
                    line_width=0,
                    row=plot_row,
                    col=1,
                )
            component_offset += component_end - component_start + 1

        event_calls = _event_labels_for_plot(region)
        if event_calls:
            for event_label, event_y, event_type in event_calls:
                event_color = EVENT_TYPE_COLORS[event_type]
                fig.add_trace(
                    go.Scatter(
                        x=[event_offset + component_length / 2],
                        y=[event_y],
                        mode="markers+text",
                        marker={
                            "size": 14,
                            "color": event_color,
                            "symbol": _event_marker_symbol(event_label),
                            "line": {"color": "#333333", "width": 1},
                        },
                        text=[event_label],
                        textposition="middle right",
                        textfont={"size": 11},
                        hovertemplate=(
                            f"{region_label}<br>"
                            f"region ploidy={region_ploidy_label}<br>"
                            f"sample ploidy={row_sample_ploidy_label}<br>"
                            f"{event_type}: {event_label}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=3,
                    col=1,
                )

        cna_overlap = _filter_cna_by_components(cna_df, chrom, components)
        for _, segment in cna_overlap.iterrows():
            y = float(segment["TCN"])
            segment_length = int(segment["end"]) - int(segment["start"]) + 1
            for x0, x1 in _component_x_ranges(
                int(segment["start"]),
                int(segment["end"]),
                components,
                event_offset,
            ):
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y, y],
                        mode="lines",
                        line={"color": region_color, "width": 5},
                        hovertemplate=(
                            f"{region_label}<br>"
                            f"{segment['chrom']}:{int(segment['start'])}-{int(segment['end'])}<br>"
                            f"length={segment_length:,} bp<br>"
                            f"region ploidy={region_ploidy_label}<br>"
                            f"sample ploidy={row_sample_ploidy_label}<br>"
                            f"TCN={segment['TCN']}<extra></extra>"
                        ),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

        bp_in_region = _filter_breakpoints_by_components(
            breakpoints_df,
            chrom,
            components,
            padding=100,
        )
        bp_in_region = _breakpoints_matching_cna_boundaries(bp_in_region, cna_overlap)
        for sv_id, sv_rows in bp_in_region.groupby("sv_id", sort=False):
            original_sv_rows = breakpoints_df[breakpoints_df["sv_id"] == sv_id].sort_values(
                ["chrom", "pos"]
            )
            padded_sv_rows = _filter_breakpoints_by_components(
                original_sv_rows,
                chrom,
                components,
                padding=100,
            )
            boundary_sv_rows = _breakpoints_matching_cna_boundaries(
                padded_sv_rows,
                cna_overlap,
            ).sort_values("pos")
            sv_type = str(sv_rows.iloc[0]["SV_TYPE"])
            color = SV_TYPE_COLORS.get(sv_type, "#444444")
            showlegend = sv_type not in legend_seen
            legend_seen.add(sv_type)

            local_positions = [
                _component_x_position(int(row["pos"]), components, event_offset, wiggle=100)
                for _, row in boundary_sv_rows.iterrows()
                if row["chrom"] == chrom
            ]
            local_positions = [position for position in local_positions if position is not None]
            local_positions = sorted(set(local_positions))
            if not local_positions:
                continue

            if len(local_positions) >= 2:
                x1, x2 = local_positions[0], local_positions[-1]
                span = max(1, x2 - x1)
                height = min(6, max(1, span / 8_000_000))
                xs, ys = _arc_points(x1, x2, height=height)
                hover = f"{sv_id}<br>{sv_type}<br>within {region_label}<extra></extra>"
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line={"color": color, "width": 2},
                        name=sv_type,
                        legendgroup=sv_type,
                        showlegend=showlegend,
                        hovertemplate=hover,
                    ),
                    row=1,
                    col=1,
                )
            else:
                local_x = local_positions[0]
                mate_label = _single_breakpoint_mate_label(original_sv_rows, chrom, components)
                xs, ys = _arc_points(local_x, local_x, height=1.2)
                hover = f"{sv_id}<br>{sv_type}<br>{mate_label}<extra></extra>"
                fig.add_trace(
                    go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line={"color": color, "width": 2, "dash": "dot"},
                        name=sv_type,
                        legendgroup=sv_type,
                        showlegend=showlegend,
                        hovertemplate=hover,
                    ),
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=local_x,
                    y=1.4,
                    text=mate_label,
                    showarrow=False,
                    textangle=-35,
                    font={"size": 9, "color": color},
                    row=1,
                    col=1,
                )

    fig.update_xaxes(
        tickmode="array",
        tickvals=region_tick_positions,
        ticktext=region_tick_labels,
        tickangle=35,
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="SV arc height", showticklabels=False, row=1, col=1)
    fig.update_yaxes(title_text="TCN", row=2, col=1)
    fig.update_yaxes(
        title_text="Events",
        range=[0, 5],
        tickmode="array",
        tickvals=[1, 2, 3, 4],
        ticktext=["ecDNA", "Seismic", "Chromothripsis", "BFB"],
        row=3,
        col=1,
    )
    fig.update_layout(
        title=(
            "Candidate regions: SV arcs, total copy number, and event classes"
            f" (sample ploidy={sample_ploidy_label})"
        ),
        template="plotly_white",
        height=max(850, 120 + 32 * len(candidate_df)),
        hovermode="closest",
        legend_title_text="SV type",
    )
    fig.write_html(output_html)


def _output_paths(output: Path) -> tuple[Path, Path]:
    if output.suffix.lower() in {".csv", ".tsv", ".html"}:
        return output.with_suffix(".csv"), output.with_suffix(".html")
    return output.with_suffix(".csv"), output.with_suffix(".html")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create final candidate CNA/SV table.")
    parser.add_argument("cna_vcf", type=Path, help="Path to Wakhan CNA VCF file")
    parser.add_argument("sv_vcf", type=Path, help="Path to Severus SV VCF file")
    parser.add_argument("-o", "--output", type=Path, help="Output path or prefix for CSV and HTML files")
    parser.add_argument("--plot-max-regions", type=int, default=None, help="Plot only first N regions")
    parser.add_argument(
        "--centromeres",
        type=Path,
        default=Path("/Users/keskusa2/complex_sv_annotation/example_data/grch38.cen_coord.curated.bed"),
        help="Centromere BED used to keep chromosome arms separate",
    )
    parser.add_argument("--window-size", type=int, default=50_000_000, help="Sliding window size")
    parser.add_argument("--step-size", type=int, default=10_000_000, help="Sliding window step size")
    parser.add_argument(
        "--method",
        choices=["sliding", "small_segments"],
        default="sliding",
        help="Candidate interval method",
    )
    parser.add_argument(
        "--max-segment-size",
        type=int,
        default=5_000_000,
        help="Maximum CNA segment size for --method small_segments",
    )
    parser.add_argument(
        "--region-merge-distance",
        type=int,
        default=10_000_000,
        help="Merge same-chromosome candidate regions separated by less than this distance",
    )
    parser.add_argument(
        "--frequency-threshold",
        type=float,
        default=1 / 2_000_000,
        help="Candidate threshold for n_segments / window_length",
    )
    parser.add_argument("--rows", type=int, default=None, help="Print only the first N rows")
    args = parser.parse_args()

    cna_df = read_cna_vcf_to_dataframe(args.cna_vcf)
    breakpoints_df = get_bps(args.sv_vcf)
    final_df = merge_sv_cna_candidate_segments(
        cna_df,
        breakpoints_df,
        window_size=args.window_size,
        step_size=args.step_size,
        frequency_threshold=args.frequency_threshold,
        centromeres=args.centromeres,
        method=args.method,
        max_segment_size=args.max_segment_size,
        region_merge_distance=args.region_merge_distance,
    )

    if args.output:
        csv_path, html_path = _output_paths(args.output)
        final_df.to_csv(csv_path, index=False)
        generate_candidate_plot(
            final_df,
            cna_df,
            breakpoints_df,
            html_path,
            max_regions=args.plot_max_regions,
        )
        print(f"Wrote {len(final_df)} rows to {csv_path}")
        print(f"Wrote candidate plot to {html_path}")
        return

    output_df = final_df.head(args.rows) if args.rows is not None else final_df
    print(output_df.to_csv(sep="\t", index=False), end="")


if __name__ == "__main__":
    main()