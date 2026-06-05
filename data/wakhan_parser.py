
"""
Parse paired Wakhan haplotype BED copy-number output.

Input lists contain one BED root per line. For a root such as

    /path/H1395_3.06_0.94_0.79

this parser reads:

    /path/H1395_3.06_0.94_0.79_copynumbers_segments_HP_1.bed
    /path/H1395_3.06_0.94_0.79_copynumbers_segments_HP_2.bed

Rows may also point directly at either HP BED file; the sibling HP file is then
resolved automatically. VCF input is intentionally unsupported.

The public output schema is consumed by complex_sv.data.cn_resampler:

    sample_id, chrom, start, end, cn_total, cn_hp1, cn_hp2,
    log_coverage_total, coverage_hp1_fraction, coverage_hp2_fraction,
    confidence_hp1, confidence_hp2, loh, allele_imbalance, breakpoint_count

Canonical coordinates are 0-based half-open intervals.
"""

from __future__ import annotations

import ast
import logging
import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "sample_id",
    "chrom",
    "start",
    "end",
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
FEATURE_COLUMNS = REQUIRED_COLUMNS[4:]

CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
CHROM_ORDER.update({str(i): i for i in range(1, 23)})
CHROM_ORDER.update({"chrX": 23, "X": 23, "chrY": 24, "Y": 24, "chrM": 25, "MT": 25, "M": 25})

HP_BED_RE = re.compile(
    r"^(?P<sample>.+?)(?:_copynumbers_segments)?_HP[_-]?(?P<hap>[12])\.bed$",
    re.IGNORECASE,
)

ALIASES: dict[str, tuple[str, ...]] = {
    "chrom": ("chrom", "chr", "#chrom", "#chr", "chromosome", "contig", "seqnames"),
    "start": ("start", "start_bp", "chromstart", "chrom_start", "begin"),
    "end": ("end", "end_bp", "chromend", "chrom_end", "stop"),
    "copynumber": ("copynumber_state", "copy_number", "copy_number_state", "cn_state", "cn"),
    "coverage": ("coverage", "cov", "depth"),
    "confidence": ("confidence", "conf", "cn_confidence"),
    "breakpoints": ("svs_breakpoints_ids", "sv_breakpoints_ids", "breakpoints", "breakpoint_ids", "bps"),
}


def _normalise_col(name: object) -> str:
    text = str(name).strip().lower()
    for ch in (" ", "-", ".", "/", ":"):
        text = text.replace(ch, "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def _sort_chrom_key(chrom: object) -> int:
    text = str(chrom)
    no_chr = text.removeprefix("chr")
    return CHROM_ORDER.get(text, CHROM_ORDER.get(no_chr, 99))


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if text in {"", ".", "nan", "NaN", "None"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _parse_breakpoint_ids(value: object) -> set[str]:
    if value is None or pd.isna(value):
        return set()

    text = str(value).strip()
    if text in {"", ".", "[]", "nan", "NaN", "None"}:
        return set()

    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        parsed = None

    if isinstance(parsed, (list, tuple, set)):
        return {str(x).strip() for x in parsed if str(x).strip()}
    if isinstance(parsed, str):
        return {parsed.strip()} if parsed.strip() else set()

    cleaned = text.strip("[](){}")
    out: set[str] = set()
    for item in re.split(r"[,;]", cleaned):
        item = item.strip().strip("'\"")
        if item:
            out.add(item)
    return out


def _scale_to_unit_interval(values: pd.Series | float | int) -> pd.Series | float:
    if isinstance(values, (float, int)):
        return float(np.clip(values, 0.0, 1.0))

    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    finite = arr[np.isfinite(arr)]
    if not finite.empty and finite.max() > 1.0 and finite.max() <= 100.0:
        arr = arr / 100.0
    return arr.clip(0.0, 1.0)


def _find_column(df: pd.DataFrame, canonical: str) -> str | None:
    norm_to_original = {_normalise_col(c): c for c in df.columns}
    for alias in ALIASES[canonical]:
        norm_alias = _normalise_col(alias)
        if norm_alias in norm_to_original:
            return norm_to_original[norm_alias]
    return None


def _require_column(df: pd.DataFrame, canonical: str, path: Path) -> str:
    col = _find_column(df, canonical)
    if col is None:
        aliases = ", ".join(ALIASES[canonical])
        raise ValueError(
            f"Missing required column for '{canonical}' in {path}. "
            f"Accepted aliases: {aliases}. Observed columns: {list(df.columns)}"
        )
    return col


def _to_numeric(series: pd.Series, name: str, path: Path, default: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if default is not None:
        return values.fillna(default)
    if values.isna().any():
        n_bad = int(values.isna().sum())
        raise ValueError(f"Column '{name}' in {path} has {n_bad} non-numeric value(s).")
    return values


def resolve_wakhan_bed_root(root: str | Path) -> tuple[str, Path, Path]:
    """
    Resolve an input-list root to (sample_id, hp1_path, hp2_path).

    Accepted root forms:
        sample_root
        sample_root_copynumbers_segments
        sample_root_copynumbers_segments_HP_1.bed
        sample_root_copynumbers_segments_HP_2.bed
    """
    root_path = Path(root)
    root_text = str(root_path)

    match = HP_BED_RE.match(root_path.name)
    if match:
        sample_id = match.group("sample")
        hap = int(match.group("hap"))
        other_hap = 2 if hap == 1 else 1
        other_name = re.sub(
            rf"HP[_-]?{hap}\.bed$",
            f"HP_{other_hap}.bed",
            root_path.name,
            flags=re.IGNORECASE,
        )
        hp1 = root_path if hap == 1 else root_path.with_name(other_name)
        hp2 = root_path if hap == 2 else root_path.with_name(other_name)
        return sample_id, hp1, hp2

    if root_text.endswith("_copynumbers_segments"):
        bed_prefix = root_text
        sample_id = Path(root_text.removesuffix("_copynumbers_segments")).name
    else:
        bed_prefix = root_text + "_copynumbers_segments"
        sample_id = root_path.name
    return sample_id, Path(bed_prefix + "_HP_1.bed"), Path(bed_prefix + "_HP_2.bed")


def _read_haplotype_bed(path: Path, hap: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing HP{hap} BED file: {path}")

    header: list[str] | None = None
    data_lines: list[str] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.rstrip("\n")
            if not stripped:
                continue

            if stripped.startswith("#"):
                candidate = stripped[1:]
                fields = [x.strip() for x in candidate.split("\t")]
                normalized = {_normalise_col(field) for field in fields}
                if (
                    len(fields) >= 4
                    and _normalise_col(fields[0]) in {"chr", "chrom"}
                    and {"start", "end"}.issubset(normalized)
                ):
                    header = fields
                continue

            if header is not None:
                data_lines.append(line)

    if header is None:
        raise ValueError(
            f"Could not find tabular BED header in {path}; expected a line like "
            "'#chr\tstart\tend\tcoverage\tcopynumber_state...'"
        )

    if not data_lines:
        raise ValueError(f"No data rows found after BED header in {path}")

    raw = pd.read_csv(
        StringIO("".join(data_lines)),
        sep="\t",
        names=header,
        dtype=str,
        engine="python",
        on_bad_lines="skip",
    )

    chrom_col = _require_column(raw, "chrom", path)
    start_col = _require_column(raw, "start", path)
    end_col = _require_column(raw, "end", path)
    cn_col = _require_column(raw, "copynumber", path)
    coverage_col = _find_column(raw, "coverage")
    confidence_col = _find_column(raw, "confidence")
    breakpoints_col = _find_column(raw, "breakpoints")

    out = pd.DataFrame(
        {
            "chrom": raw[chrom_col].astype(str),
            "start": _to_numeric(raw[start_col], start_col, path).astype(np.int64),
            "end": _to_numeric(raw[end_col], end_col, path).astype(np.int64),
            "cn": _to_numeric(raw[cn_col], cn_col, path, default=1.0).astype(float),
            "coverage": (
                _to_numeric(raw[coverage_col], coverage_col, path, default=0.0).astype(float)
                if coverage_col is not None
                else 0.0
            ),
            "confidence": (
                _scale_to_unit_interval(raw[confidence_col])
                if confidence_col is not None
                else 0.0
            ),
            "breakpoint_ids": (
                raw[breakpoints_col].map(_parse_breakpoint_ids)
                if breakpoints_col is not None
                else [set() for _ in range(len(raw))]
            ),
        }
    )

    out["start"] = out["start"].clip(lower=0)
    out = out[out["end"] > out["start"]].copy()
    out = out.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    log.info("  %s HP%d: %d segments.", path.name, hap, len(out))
    return out


def _hap_values_at_midpoints(
    df_hap: pd.DataFrame,
    mids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[set[str]], np.ndarray]:
    n = len(mids)
    cn = np.ones(n, dtype=np.float64)
    coverage = np.zeros(n, dtype=np.float64)
    confidence = np.zeros(n, dtype=np.float64)
    breakpoint_ids = [set() for _ in range(n)]
    covered = np.zeros(n, dtype=bool)

    if df_hap.empty or n == 0:
        return cn, coverage, confidence, breakpoint_ids, covered

    starts = df_hap["start"].to_numpy(dtype=np.int64)
    ends = df_hap["end"].to_numpy(dtype=np.int64)
    idx = np.searchsorted(starts, mids, side="right") - 1
    valid = (idx >= 0) & (idx < len(df_hap))
    valid[valid] = mids[valid] < ends[idx[valid]]
    if not valid.any():
        return cn, coverage, confidence, breakpoint_ids, covered

    valid_idx = idx[valid]
    cn[valid] = df_hap["cn"].to_numpy(dtype=np.float64)[valid_idx]
    coverage[valid] = df_hap["coverage"].to_numpy(dtype=np.float64)[valid_idx]
    confidence[valid] = df_hap["confidence"].to_numpy(dtype=np.float64)[valid_idx]

    bp_values = df_hap["breakpoint_ids"].tolist()
    for out_i, source_i in zip(np.flatnonzero(valid), valid_idx):
        breakpoint_ids[int(out_i)] = set(bp_values[int(source_i)])

    covered[valid] = True
    return cn, coverage, confidence, breakpoint_ids, covered


def _coverage_features(cov_hp1: float, cov_hp2: float) -> tuple[float, float, float]:
    cov1 = max(float(cov_hp1), 0.0)
    cov2 = max(float(cov_hp2), 0.0)
    total = cov1 + cov2
    if total > 0.0:
        return float(np.log1p(total)), float(cov1 / total), float(cov2 / total)
    return 0.0, 0.5, 0.5


def _derived_cn_features(cn_hp1: float, cn_hp2: float) -> tuple[float, float, float]:
    cn1 = float(cn_hp1)
    cn2 = float(cn_hp2)
    total = cn1 + cn2
    allele_imbalance = float(np.clip(abs(cn1 - cn2) / (total + 1e-6), 0.0, 1.0))
    loh = 1.0 if min(cn1, cn2) <= 0.0 else 0.0
    return float(total), loh, allele_imbalance


def _align_haplotypes(hp1: pd.DataFrame, hp2: pd.DataFrame, sample_id: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    chroms = sorted(
        set(hp1["chrom"].astype(str)).union(set(hp2["chrom"].astype(str))),
        key=_sort_chrom_key,
    )

    for chrom in chroms:
        h1 = hp1[hp1["chrom"].astype(str) == chrom].sort_values(["start", "end"]).reset_index(drop=True)
        h2 = hp2[hp2["chrom"].astype(str) == chrom].sort_values(["start", "end"]).reset_index(drop=True)

        boundary_arrays = []
        if not h1.empty:
            boundary_arrays.append(h1[["start", "end"]].to_numpy(dtype=np.int64).ravel())
        if not h2.empty:
            boundary_arrays.append(h2[["start", "end"]].to_numpy(dtype=np.int64).ravel())
        if not boundary_arrays:
            continue

        boundaries = np.sort(np.unique(np.concatenate(boundary_arrays)))
        starts = boundaries[:-1]
        ends = boundaries[1:]
        valid_intervals = ends > starts
        starts = starts[valid_intervals]
        ends = ends[valid_intervals]
        if len(starts) == 0:
            continue

        mids = (starts.astype(np.float64) + ends.astype(np.float64)) / 2.0
        cn1, cov1, conf1, bp1, covered1 = _hap_values_at_midpoints(h1, mids)
        cn2, cov2, conf2, bp2, covered2 = _hap_values_at_midpoints(h2, mids)

        for i in np.flatnonzero(covered1 | covered2):
            cn_total, loh, allele_imbalance = _derived_cn_features(cn1[i], cn2[i])
            log_cov_total, cov_frac1, cov_frac2 = _coverage_features(cov1[i], cov2[i])
            records.append(
                {
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "start": int(starts[i]),
                    "end": int(ends[i]),
                    "cn_total": cn_total,
                    "cn_hp1": float(cn1[i]),
                    "cn_hp2": float(cn2[i]),
                    "log_coverage_total": log_cov_total,
                    "coverage_hp1_fraction": cov_frac1,
                    "coverage_hp2_fraction": cov_frac2,
                    "confidence_hp1": float(conf1[i]),
                    "confidence_hp2": float(conf2[i]),
                    "loh": loh,
                    "allele_imbalance": allele_imbalance,
                    "breakpoint_count": len(bp1[i].union(bp2[i])),
                }
            )

    return pd.DataFrame(records, columns=REQUIRED_COLUMNS)


def _finalize(df: pd.DataFrame, root: str | Path, sample_id: str) -> pd.DataFrame:
    if df.empty:
        log.warning("No Wakhan BED records parsed from %s", root)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out["sample_id"] = out["sample_id"].fillna(str(sample_id)).astype(str)
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype(np.int64)
    out["end"] = pd.to_numeric(out["end"], errors="coerce").fillna(0).astype(np.int64)

    for col, default in [
        ("cn_hp1", 1.0),
        ("cn_hp2", 1.0),
        ("log_coverage_total", 0.0),
        ("coverage_hp1_fraction", 0.5),
        ("coverage_hp2_fraction", 0.5),
        ("confidence_hp1", 0.0),
        ("confidence_hp2", 0.0),
        ("allele_imbalance", 0.0),
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)

    out["cn_total"] = pd.to_numeric(out["cn_total"], errors="coerce")
    out["cn_total"] = out["cn_total"].fillna(out["cn_hp1"] + out["cn_hp2"]).astype(float)
    out["loh"] = pd.to_numeric(out["loh"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    out["breakpoint_count"] = pd.to_numeric(out["breakpoint_count"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    out["start"] = out["start"].clip(lower=0)
    out["log_coverage_total"] = out["log_coverage_total"].clip(lower=0.0)
    for col in [
        "coverage_hp1_fraction",
        "coverage_hp2_fraction",
        "confidence_hp1",
        "confidence_hp2",
        "allele_imbalance",
    ]:
        out[col] = out[col].clip(0.0, 1.0)

    invalid = out["end"] <= out["start"]
    if invalid.any():
        log.warning("Dropping %d invalid segment(s) from %s", int(invalid.sum()), root)
        out = out.loc[~invalid].copy()

    if out.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out["_chrom_order"] = out["chrom"].map(_sort_chrom_key)
    out = out.sort_values(["_chrom_order", "start", "end"]).drop(columns="_chrom_order")
    out = out.reset_index(drop=True)
    log.info("  %s: %d paired haplotype segments, %d chromosomes.", sample_id, len(out), out["chrom"].nunique())
    return out[REQUIRED_COLUMNS]


def parse_wakhan(root: str | Path) -> pd.DataFrame:
    """Parse one Wakhan BED root or HP BED path into canonical segments."""
    sample_id, hp1_path, hp2_path = resolve_wakhan_bed_root(root)
    hp1 = _read_haplotype_bed(hp1_path, hap=1)
    hp2 = _read_haplotype_bed(hp2_path, hap=2)
    paired = _align_haplotypes(hp1, hp2, sample_id)
    return _finalize(paired, root, sample_id)


def parse_all_wakhan(roots: list[str | Path]) -> pd.DataFrame:
    """Parse every Wakhan BED root and concatenate canonical segment tables."""
    log.info("Parsing %d Wakhan BED root(s) ...", len(roots))
    frames: list[pd.DataFrame] = []

    seen_samples: set[str] = set()
    for root in tqdm(roots, desc="parse Wakhan BED roots", unit="root", leave=False):
        try:
            sample_id, _hp1, _hp2 = resolve_wakhan_bed_root(root)
            if sample_id in seen_samples:
                log.warning("Skipping duplicate Wakhan BED root for sample %s: %s", sample_id, root)
                continue
            seen_samples.add(sample_id)
            df = parse_wakhan(root)
        except Exception as exc:
            log.warning("Skipping unreadable Wakhan BED root %s: %s", root, exc)
            continue

        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No valid paired Wakhan BED records found in any input root.")

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Total: %d paired Wakhan segments across %d sample(s).",
        len(combined),
        combined["sample_id"].nunique(),
    )
    return combined[REQUIRED_COLUMNS]
