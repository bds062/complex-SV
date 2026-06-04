"""
Parse Wakhan copy-number output.

Supports two Wakhan-like input styles:

1. VCF-style Wakhan calls, with CHROM POS ID REF ALT QUAL FILTER INFO FORMAT SAMPLE
2. TSV/BED-like segment tables with recognizable column aliases

The public output schema is the canonical segment table consumed by
complex_sv.data.cn_resampler:

    sample_id, chrom, start, end, cn_total, cn_hp1, cn_hp2,
    loh, allele_imbalance, baf

Canonical coordinates are 0-based half-open intervals.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "sample_id",
    "chrom",
    "start",
    "end",
    "cn_total",
    "cn_hp1",
    "cn_hp2",
    "loh",
    "allele_imbalance",
    "baf",
]

CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
CHROM_ORDER.update({str(i): i for i in range(1, 23)})
CHROM_ORDER.update(
    {
        "chrX": 23,
        "X": 23,
        "chrY": 24,
        "Y": 24,
        "chrM": 25,
        "MT": 25,
        "M": 25,
    }
)

ALIASES: dict[str, tuple[str, ...]] = {
    "chrom": (
        "chrom",
        "chr",
        "chromosome",
        "contig",
        "seqnames",
        "#chrom",
    ),
    "start": (
        "start",
        "start_bp",
        "chromstart",
        "chrom_start",
        "begin",
    ),
    "pos": (
        "pos",
        "position",
    ),
    "end": (
        "end",
        "end_bp",
        "chromend",
        "chrom_end",
        "stop",
    ),
    "cn_total": (
        "cn_total",
        "total_cn",
        "tcn",
        "total_copy_number",
        "copy_number",
        "copy_number_total",
        "cn",
    ),
    "cn_hp1": (
        "cn_hp1",
        "hp1_cn",
        "hap1_cn",
        "haplotype1_cn",
        "haplotype_1_cn",
        "cn1",
        "copy_number_1",
        "major_cn",
    ),
    "cn_hp2": (
        "cn_hp2",
        "hp2_cn",
        "hap2_cn",
        "haplotype2_cn",
        "haplotype_2_cn",
        "cn2",
        "copy_number_2",
        "minor_cn",
    ),
    "loh": (
        "loh",
        "is_loh",
        "cnloh",
        "loss_of_heterozygosity",
        "loh_state",
        "cnv_type",
        "type",
        "svtype",
    ),
    "allele_imbalance": (
        "allele_imbalance",
        "allelic_imbalance",
        "allelic_imb",
        "dosage_imbalance",
        "imbalance",
        "ai",
    ),
    "baf": (
        "baf",
        "phased_baf",
        "b_allele_frequency",
        "ballelefreq",
    ),
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


def _safe_int(value: object, default: int = 0) -> int:
    return int(round(_safe_float(value, float(default))))


def _parse_info(info_str: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for item in info_str.split(";"):
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            info[key] = value
        else:
            info[item] = "1"
    return info


def _parse_sample(fmt_str: str, sample_str: str) -> dict[str, str]:
    keys = fmt_str.split(":")
    vals = sample_str.split(":")
    return dict(zip(keys, vals))


def _infer_cnv_type(record_id: str, info: dict[str, str]) -> str:
    joined = f"{record_id};" + ";".join(f"{k}={v}" for k, v in info.items())
    joined = joined.upper()

    if "CNLOH" in joined:
        return "CNLOH"
    if "LOSS" in joined or "DEL" in joined:
        return "LOSS"
    if "GAIN" in joined or "DUP" in joined:
        return "GAIN"

    svtype = info.get("SVTYPE", "").upper()
    if svtype in {"CNLOH", "LOSS", "GAIN", "DEL", "DUP"}:
        if svtype == "DEL":
            return "LOSS"
        if svtype == "DUP":
            return "GAIN"
        return svtype

    return "CNV"


def _vcf_like(path: Path) -> bool:
    """
    Detect VCF-style input without depending on pandas delimiter inference.
    """
    saw_header = False
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                saw_header = True
                continue
            if line.startswith("#") or not line.strip():
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 10:
                return True

            # If the file had a VCF header, treat even unusual first rows as VCF.
            return saw_header

    return saw_header


def _parse_vcf_wakhan(path: Path, sample_id: str) -> pd.DataFrame:
    """
    Parse Wakhan VCF records manually.

    This mirrors the working prototype: skip comment lines, split only by tabs,
    read INFO and FORMAT/SAMPLE fields explicitly, and derive canonical CN
    segment features.
    """
    records: list[dict[str, object]] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip() or line.startswith("#"):
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                log.debug("Skipping short VCF row in %s:%d", path, line_no)
                continue

            chrom, pos, rec_id, _ref, _alt, _qual, _filt, info_str, fmt_str, sample_str = parts[:10]

            info = _parse_info(info_str)
            sample = _parse_sample(fmt_str, sample_str)

            pos_1based = _safe_int(pos, default=1)
            start = max(pos_1based - 1, 0)

            end = _safe_int(info.get("END"), default=0)
            if end <= 0:
                svlen = abs(_safe_int(info.get("SVLEN"), default=1))
                end = pos_1based + max(svlen, 1) - 1

            # Canonical output uses 0-based half-open intervals.
            # VCF END is conventionally 1-based inclusive, so using END as the
            # half-open stop is the standard POS-1, END conversion.
            if end <= start:
                end = start + 1

            cn_total = _safe_float(
                sample.get("TCN", sample.get("CN", sample.get("CN_TOTAL"))),
                default=2.0,
            )
            cn_hp1 = _safe_float(
                sample.get("CN1", sample.get("HP1_CN", sample.get("CN_HP1"))),
                default=1.0,
            )
            cn_hp2 = _safe_float(
                sample.get("CN2", sample.get("HP2_CN", sample.get("CN_HP2"))),
                default=1.0,
            )

            cnv_type = _infer_cnv_type(rec_id, info)
            loh = 1 if cnv_type == "CNLOH" or min(cn_hp1, cn_hp2) == 0 else 0

            allele_imbalance = abs(cn_hp1 - cn_hp2) / (cn_total + 1e-6)
            allele_imbalance = float(np.clip(allele_imbalance, 0.0, 1.0))

            # If Wakhan provides BAF-like fields, use them; otherwise derive a
            # stable allele-fraction proxy from haplotype copy numbers.
            baf_raw = (
                sample.get("BAF")
                or sample.get("PHASED_BAF")
                or sample.get("B_ALLELE_FREQUENCY")
                or info.get("BAF")
            )
            if baf_raw is not None:
                baf = _safe_float(baf_raw, default=0.5)
                if baf > 1.0 and baf <= 100.0:
                    baf /= 100.0
            else:
                denom = cn_hp1 + cn_hp2
                baf = cn_hp2 / (denom + 1e-6) if denom > 0 else 0.5
            baf = float(np.clip(baf, 0.0, 1.0))

            records.append(
                {
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "start": int(start),
                    "end": int(end),
                    "cn_total": float(cn_total),
                    "cn_hp1": float(cn_hp1),
                    "cn_hp2": float(cn_hp2),
                    "loh": int(loh),
                    "allele_imbalance": allele_imbalance,
                    "baf": baf,
                }
            )

    out = pd.DataFrame(records, columns=REQUIRED_COLUMNS)
    return _finalize(out, path, sample_id)


def _read_table(path: Path) -> pd.DataFrame:
    """
    Read non-VCF Wakhan tables robustly.

    Avoid sep=None because pandas delimiter sniffing caused:
        ParserError: Expected 4 fields in line N, saw 6
    """
    attempts = [
        ("tab", "\t"),
        ("whitespace", r"\s+"),
        ("comma", ","),
    ]

    for name, sep in attempts:
        try:
            df = pd.read_csv(
                path,
                sep=sep,
                comment="#",
                dtype=str,
                engine="python",
                on_bad_lines="skip",
            )
            if not df.empty and df.shape[1] >= 3:
                log.debug("Read %s as %s-delimited table", path, name)
                return df
        except pd.errors.ParserError:
            continue

    raise ValueError(f"Could not parse Wakhan table: {path}")


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
            f"Missing required Wakhan column for '{canonical}' in {path}. "
            f"Accepted aliases: {aliases}. Observed columns: {list(df.columns)}"
        )
    return col


def _to_numeric(
    series: pd.Series,
    name: str,
    path: Path,
    default: float | None = None,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if default is not None:
        return values.fillna(default)

    if values.isna().any():
        n_bad = int(values.isna().sum())
        raise ValueError(f"Column '{name}' in {path} has {n_bad} non-numeric value(s).")

    return values


def _parse_loh_value(value: object) -> int:
    if pd.isna(value):
        return 0

    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "loh", "cnloh", "loss"}:
        return 1
    if text in {"0", "false", "f", "no", "n", "normal", "none", ".", ""}:
        return 0

    if any(tok in text for tok in ("cnloh", "loh")):
        return 1

    try:
        return 1 if float(text) > 0 else 0
    except ValueError:
        return 0


def _scale_to_unit_interval(values: pd.Series | float | int) -> pd.Series | float:
    if isinstance(values, (float, int)):
        return float(np.clip(values, 0.0, 1.0))

    arr = pd.to_numeric(values, errors="coerce").fillna(0.0).astype(float)
    finite = arr[np.isfinite(arr)]
    if not finite.empty and finite.max() > 1.0 and finite.max() <= 100.0:
        arr = arr / 100.0
    return arr.clip(0.0, 1.0)


def _parse_table_wakhan(path: Path, sample_id: str) -> pd.DataFrame:
    raw = _read_table(path)
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    chrom_col = _require_column(raw, "chrom", path)
    end_col = _require_column(raw, "end", path)

    start_col = _find_column(raw, "start")
    pos_col = _find_column(raw, "pos")
    if start_col is None and pos_col is None:
        raise ValueError(
            f"Missing required coordinate column in {path}. "
            "Expected start/start_bp or pos."
        )

    cn_total_col = _require_column(raw, "cn_total", path)
    cn_hp1_col = _require_column(raw, "cn_hp1", path)
    cn_hp2_col = _require_column(raw, "cn_hp2", path)

    loh_col = _find_column(raw, "loh")
    ai_col = _find_column(raw, "allele_imbalance")
    baf_col = _find_column(raw, "baf")

    chrom = raw[chrom_col].astype(str)

    if start_col is not None:
        start = _to_numeric(raw[start_col], start_col, path).astype(np.int64)
    else:
        # POS is VCF-like 1-based.
        start = (_to_numeric(raw[pos_col], pos_col or "pos", path) - 1).astype(np.int64)

    end = _to_numeric(raw[end_col], end_col, path).astype(np.int64)
    cn_total = _to_numeric(raw[cn_total_col], cn_total_col, path, default=2.0).astype(float)
    cn_hp1 = _to_numeric(raw[cn_hp1_col], cn_hp1_col, path, default=1.0).astype(float)
    cn_hp2 = _to_numeric(raw[cn_hp2_col], cn_hp2_col, path, default=1.0).astype(float)

    if loh_col is not None:
        loh = raw[loh_col].map(_parse_loh_value).astype(int)
    else:
        loh = (np.minimum(cn_hp1, cn_hp2) == 0).astype(int)

    if ai_col is not None:
        allele_imbalance = _scale_to_unit_interval(raw[ai_col])
    else:
        allele_imbalance = (abs(cn_hp1 - cn_hp2) / (cn_total + 1e-6)).clip(0.0, 1.0)

    if baf_col is not None:
        baf = _scale_to_unit_interval(raw[baf_col])
    else:
        baf = (cn_hp2 / (cn_hp1 + cn_hp2 + 1e-6)).fillna(0.5).clip(0.0, 1.0)

    out = pd.DataFrame(
        {
            "sample_id": str(sample_id),
            "chrom": chrom,
            "start": start,
            "end": end,
            "cn_total": cn_total,
            "cn_hp1": cn_hp1,
            "cn_hp2": cn_hp2,
            "loh": loh,
            "allele_imbalance": allele_imbalance,
            "baf": baf,
        }
    )

    return _finalize(out, path, sample_id)


def _finalize(df: pd.DataFrame, path: Path, sample_id: str) -> pd.DataFrame:
    if df.empty:
        log.warning("No Wakhan records parsed from %s", path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out = df.copy()

    out["sample_id"] = out["sample_id"].astype(str)
    out["chrom"] = out["chrom"].astype(str)
    out["start"] = pd.to_numeric(out["start"], errors="coerce").fillna(0).astype(np.int64)
    out["end"] = pd.to_numeric(out["end"], errors="coerce").fillna(0).astype(np.int64)

    for col, default in [
        ("cn_total", 2.0),
        ("cn_hp1", 1.0),
        ("cn_hp2", 1.0),
        ("allele_imbalance", 0.0),
        ("baf", 0.5),
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(default).astype(float)

    out["loh"] = pd.to_numeric(out["loh"], errors="coerce").fillna(0).astype(int)

    out["start"] = out["start"].clip(lower=0)
    out["allele_imbalance"] = out["allele_imbalance"].clip(0.0, 1.0)
    out["baf"] = out["baf"].clip(0.0, 1.0)

    invalid = out["end"] <= out["start"]
    if invalid.any():
        n_bad = int(invalid.sum())
        log.warning("Dropping %d invalid segment(s) with end <= start from %s", n_bad, path)
        out = out.loc[~invalid].copy()

    if out.empty:
        log.warning("No valid Wakhan segments remain after filtering %s", path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    out["_chrom_order"] = out["chrom"].map(_sort_chrom_key)
    out = out.sort_values(["_chrom_order", "start", "end"]).drop(columns="_chrom_order")
    out = out.reset_index(drop=True)

    log.info(
        "  %s: %d Wakhan segments, %d chromosomes.",
        sample_id,
        len(out),
        out["chrom"].nunique(),
    )

    return out[REQUIRED_COLUMNS]


def parse_wakhan(path: str | Path, sample_id: str | None = None) -> pd.DataFrame:
    """
    Parse a single Wakhan VCF or TSV/BED-like file into canonical CN segments.
    """
    path = Path(path)

    if sample_id is None:
        sample_id = path.stem

    if not path.exists():
        raise FileNotFoundError(f"Wakhan file not found: {path}")

    if _vcf_like(path):
        return _parse_vcf_wakhan(path, str(sample_id))

    return _parse_table_wakhan(path, str(sample_id))


def parse_all_wakhan(paths: list[str | Path]) -> pd.DataFrame:
    """
    Parse every Wakhan file and concatenate into one canonical DataFrame.
    """
    log.info("Parsing %d Wakhan file(s) ...", len(paths))

    frames: list[pd.DataFrame] = []

    for p in paths:
        try:
            df = parse_wakhan(p)
        except Exception as exc:
            log.warning("Skipping unreadable Wakhan file %s: %s", p, exc)
            continue

        if df.empty:
            log.warning("Wakhan file returned no records: %s", p)
            continue

        frames.append(df)

    if not frames:
        raise RuntimeError("No valid Wakhan records found in any input file.")

    combined = pd.concat(frames, ignore_index=True)

    log.info(
        "Total: %d Wakhan segments across %d sample(s).",
        len(combined),
        combined["sample_id"].nunique(),
    )

    return combined[REQUIRED_COLUMNS]