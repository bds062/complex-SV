
"""
Resample paired Wakhan haplotype copy-number segments to fixed tensors.

The CN encoder consumes a fixed bin count with these channels:

    cn_total, cn_hp1, cn_hp2, log_coverage_total,
    coverage_hp1_fraction, coverage_hp2_fraction,
    confidence_hp1, confidence_hp2, loh, allele_imbalance,
    breakpoint_count

The bp-window interface samples fixed genomic spans and resamples each span to
[n_bins, len(CN_CHANNELS)]. Long CN segments naturally occupy more bins than
short segments because bins are evenly spaced in bp coordinates.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

CN_CHANNELS = [
    "cn_total",
    "cn_hp1",
    "cn_hp2",
    "log_coverage_total",
    "coverage_hp1_fraction",
    "coverage_hp2_fraction",
    "confidence_hp1",
    "confidence_hp2",
    "loh",
    "allele_imbalance",
    "breakpoint_count",
]

DISCRETE_CHANNELS = {"loh", "breakpoint_count"}
UNIT_INTERVAL_CHANNELS = {
    "coverage_hp1_fraction",
    "coverage_hp2_fraction",
    "confidence_hp1",
    "confidence_hp2",
    "loh",
    "allele_imbalance",
}
NONNEGATIVE_CHANNELS = {
    "cn_total",
    "cn_hp1",
    "cn_hp2",
    "log_coverage_total",
    "breakpoint_count",
}

HG38_CENTROMERE: dict[str, int] = {
    "chr1": 123_459_000,
    "chr2": 93_989_000,
    "chr3": 90_951_000,
    "chr4": 50_494_000,
    "chr5": 48_464_000,
    "chr6": 60_589_000,
    "chr7": 59_904_000,
    "chr8": 45_644_000,
    "chr9": 49_224_000,
    "chr10": 40_246_000,
    "chr11": 53_714_000,
    "chr12": 35_807_000,
    "chr13": 17_900_000,
    "chr14": 17_600_000,
    "chr15": 19_050_000,
    "chr16": 36_800_000,
    "chr17": 25_150_000,
    "chr18": 18_540_000,
    "chr19": 26_260_000,
    "chr20": 28_120_000,
    "chr21": 12_910_000,
    "chr22": 15_050_000,
    "chrX": 61_090_000,
    "chrY": 10_450_000,
}


def _normalise_chrom_for_centromere(chrom: object) -> str:
    text = str(chrom)
    if text in {"X", "Y", "M", "MT"}:
        return "chr" + text if text in {"X", "Y"} else text
    if text.startswith("chr"):
        return text
    return f"chr{text}"


def _validate_segment_columns(df: pd.DataFrame) -> None:
    required = {"start", "end", *CN_CHANNELS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"CN segment DataFrame is missing required columns: {missing}")


def _prepare_segments(
    df: pd.DataFrame,
    start_bp: int | None = None,
    end_bp: int | None = None,
) -> pd.DataFrame:
    _validate_segment_columns(df)

    segs = df.copy()
    segs["start"] = pd.to_numeric(segs["start"], errors="coerce")
    segs["end"] = pd.to_numeric(segs["end"], errors="coerce")

    for col in CN_CHANNELS:
        segs[col] = pd.to_numeric(segs[col], errors="coerce")

    segs = segs.dropna(subset=["start", "end", *CN_CHANNELS])

    if start_bp is not None and end_bp is not None:
        start_bp = int(start_bp)
        end_bp = int(end_bp)
        if end_bp <= start_bp:
            raise ValueError("end_bp must be greater than start_bp")

        segs = segs[(segs["end"] > start_bp) & (segs["start"] < end_bp)].copy()
        if not segs.empty:
            segs["start"] = segs["start"].clip(lower=start_bp)
            segs["end"] = segs["end"].clip(upper=end_bp)

    segs = segs[segs["end"] > segs["start"]]
    segs = segs.sort_values(["start", "end"]).reset_index(drop=True)
    return segs


def _nearest_values(xp: np.ndarray, fp: np.ndarray, x: np.ndarray) -> np.ndarray:
    if xp.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    insert = np.searchsorted(xp, x, side="left")
    left = np.clip(insert - 1, 0, xp.size - 1)
    right = np.clip(insert, 0, xp.size - 1)
    choose_right = np.abs(xp[right] - x) < np.abs(x - xp[left])
    nearest = np.where(choose_right, right, left)
    return fp[nearest].astype(np.float32)


def _postprocess_channel(values: np.ndarray, col: str) -> np.ndarray:
    out = values.astype(np.float32)
    if col in DISCRETE_CHANNELS:
        out = np.rint(out)
    if col in UNIT_INTERVAL_CHANNELS:
        out = np.clip(out, 0.0, 1.0)
    if col in NONNEGATIVE_CHANNELS:
        out = np.clip(out, 0.0, None)
    return out.astype(np.float32)


def _resample_segments(
    segs: pd.DataFrame,
    n_bins: int,
    span_start: int | None = None,
    span_end: int | None = None,
) -> np.ndarray:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")

    if span_start is not None and span_end is not None:
        start = int(span_start)
        end = int(span_end)
    elif not segs.empty:
        start = int(segs["start"].min())
        end = int(segs["end"].max())
    else:
        start = 0
        end = 1

    if end <= start or segs.empty:
        return np.zeros((n_bins, len(CN_CHANNELS)), dtype=np.float32)

    mids = (segs["start"].to_numpy(dtype=np.float64) + segs["end"].to_numpy(dtype=np.float64)) / 2.0
    order = np.argsort(mids)
    mids = mids[order]
    segs = segs.iloc[order].reset_index(drop=True)

    if len(np.unique(mids)) != len(mids):
        temp = segs.copy()
        temp["_mid"] = mids
        agg = {col: "mean" for col in CN_CHANNELS}
        collapsed = temp.groupby("_mid", sort=True, as_index=False).agg(agg)
        mids = collapsed["_mid"].to_numpy(dtype=np.float64)
        segs = collapsed

    bin_edges = np.linspace(float(start), float(end), n_bins + 1, dtype=np.float64)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    out = np.zeros((n_bins, len(CN_CHANNELS)), dtype=np.float32)

    for j, col in enumerate(CN_CHANNELS):
        values = segs[col].to_numpy(dtype=np.float64)
        if col in DISCRETE_CHANNELS:
            sampled = _nearest_values(mids, values, bin_mids)
        else:
            sampled = np.interp(bin_mids, mids, values, left=values[0], right=values[-1]).astype(np.float32)
        out[:, j] = _postprocess_channel(sampled, col)

    return out.astype(np.float32)


def resample_arm(df_arm: pd.DataFrame, n_bins: int = 256) -> np.ndarray:
    segs = _prepare_segments(df_arm)
    return _resample_segments(segs, n_bins=n_bins)


def resample_region(
    df_segments: pd.DataFrame,
    start_bp: int,
    end_bp: int,
    n_bins: int = 128,
) -> np.ndarray:
    segs = _prepare_segments(df_segments, start_bp=int(start_bp), end_bp=int(end_bp))
    return _resample_segments(segs, n_bins=n_bins, span_start=int(start_bp), span_end=int(end_bp))


def arm_to_tensor(df_arm: pd.DataFrame, n_bins: int = 256) -> torch.Tensor:
    return torch.as_tensor(resample_arm(df_arm, n_bins=n_bins), dtype=torch.float32)


def region_to_tensor(
    df_segments: pd.DataFrame,
    start_bp: int,
    end_bp: int,
    n_bins: int = 128,
) -> torch.Tensor:
    return torch.as_tensor(
        resample_region(df_segments, start_bp=start_bp, end_bp=end_bp, n_bins=n_bins),
        dtype=torch.float32,
    )


def get_arm_bounds(df_chrom: pd.DataFrame) -> list[tuple[str, int, int]]:
    if df_chrom.empty:
        return []
    if not {"chrom", "start", "end"}.issubset(df_chrom.columns):
        raise ValueError("df_chrom must contain chrom, start, and end columns")

    chrom_values = df_chrom["chrom"].dropna().astype(str).unique()
    if len(chrom_values) != 1:
        raise ValueError("get_arm_bounds expects segments from exactly one chromosome")

    chrom = chrom_values[0]
    chrom_key = _normalise_chrom_for_centromere(chrom)
    span_start = int(pd.to_numeric(df_chrom["start"], errors="coerce").min())
    span_end = int(pd.to_numeric(df_chrom["end"], errors="coerce").max())
    if span_end <= span_start:
        return []

    split = HG38_CENTROMERE.get(chrom_key)
    if split is None or split <= span_start or split >= span_end:
        return [(f"{chrom}", span_start, span_end)]
    return [(f"{chrom}p", span_start, int(split)), (f"{chrom}q", int(split), span_end)]


def build_bp_window_tensors(
    df: pd.DataFrame,
    window_bp_sizes: Sequence[int],
    n_bins: int = 128,
    windows_per_chrom_per_size: int = 40,
    min_covered_fraction: float = 0.10,
    rng: np.random.Generator | None = None,
    max_windows: int | None = None,
    progress: bool = False,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    """
    Build fixed-size genomic-window training examples.

    For each sample/chromosome and requested bp size, this samples intervals,
    clips paired CN segments to the interval, and resamples to
    [n_bins, len(CN_CHANNELS)].

    max_windows optionally caps the number of accepted tensors. This is useful
    for large cohorts where samples * chromosomes * sizes * random windows can
    easily produce hundreds of thousands of examples before training starts.
    """
    if rng is None:
        rng = np.random.default_rng()
    if not window_bp_sizes:
        raise ValueError("window_bp_sizes must contain at least one bp size")

    bad_sizes = [int(x) for x in window_bp_sizes if int(x) <= 0]
    if bad_sizes:
        raise ValueError(f"window_bp_sizes must be positive; got {bad_sizes}")
    if max_windows is not None and int(max_windows) <= 0:
        raise ValueError("max_windows must be positive when provided")

    max_windows = int(max_windows) if max_windows is not None else None
    tensors: list[torch.Tensor] = []
    meta: list[dict[str, object]] = []

    groups = list(df.groupby(["sample_id", "chrom"], sort=False))
    group_iter = tqdm(groups, desc="build windows", unit="chrom", leave=False) if progress else groups

    for (sample_id, chrom), grp_raw in group_iter:
        if max_windows is not None and len(tensors) >= max_windows:
            break

        # Prepare numeric, sorted segments once per sample/chromosome. Each
        # sampled window then only clips the already-prepared overlap.
        grp = _prepare_segments(grp_raw)
        if grp.empty:
            continue

        chrom_start = int(grp["start"].min())
        chrom_end = int(grp["end"].max())
        if chrom_end <= chrom_start:
            continue

        for window_bp in window_bp_sizes:
            if max_windows is not None and len(tensors) >= max_windows:
                break

            window_bp = int(window_bp)
            if window_bp > chrom_end - chrom_start:
                candidate_starts = np.array([chrom_start], dtype=np.int64)
                effective_window_bp = chrom_end - chrom_start
            else:
                max_start = chrom_end - window_bp
                possible = max_start - chrom_start + 1
                n_samples = min(int(windows_per_chrom_per_size), int(possible))
                if n_samples <= 0:
                    continue
                candidate_starts = rng.integers(
                    low=chrom_start,
                    high=max_start + 1,
                    size=n_samples,
                    endpoint=False,
                    dtype=np.int64,
                )
                effective_window_bp = window_bp

            for start_bp in candidate_starts:
                if max_windows is not None and len(tensors) >= max_windows:
                    break

                start_bp = int(start_bp)
                end_bp = int(start_bp + effective_window_bp)
                overlap = grp[(grp["end"] > start_bp) & (grp["start"] < end_bp)]
                if overlap.empty:
                    continue

                clipped_start = overlap["start"].clip(lower=start_bp)
                clipped_end = overlap["end"].clip(upper=end_bp)
                covered_bp = int((clipped_end - clipped_start).clip(lower=0).sum())
                covered_fraction = covered_bp / max(end_bp - start_bp, 1)
                if covered_fraction < min_covered_fraction:
                    continue

                clipped = overlap.copy()
                clipped["start"] = clipped_start
                clipped["end"] = clipped_end
                tensor = torch.as_tensor(
                    _resample_segments(
                        clipped,
                        n_bins=n_bins,
                        span_start=start_bp,
                        span_end=end_bp,
                    ),
                    dtype=torch.float32,
                )

                tensors.append(tensor)
                meta.append(
                    {
                        "sample_id": sample_id,
                        "chrom": chrom,
                        "start_bp": start_bp,
                        "end_bp": end_bp,
                        "window_bp_size": int(end_bp - start_bp),
                        "requested_window_bp_size": int(window_bp),
                        "covered_bp": covered_bp,
                        "covered_fraction": float(covered_fraction),
                        "n_segments": int(len(overlap)),
                    }
                )

    return tensors, pd.DataFrame(meta)
