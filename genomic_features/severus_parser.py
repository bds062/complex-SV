"""
Parse Severus VCF output and construct graph-node feature matrices.

This module is the only project component that reads Severus VCF text.  It
produces the canonical per-SV DataFrame and the canonical node feature
matrix expected by the heterogeneous graph encoder.

Canonical coordinates are 0-based half-open intervals.  VCF POS is converted
from 1-based to 0-based start; INFO/END is used as the half-open end.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

log = logging.getLogger(__name__)

CONTINUOUS_COLS = [
    "qual",
    "log_svlen",
    "mapq",
    "vaf_mean",
    "vaf_std",
    "hvaf_hp1",
    "hvaf_hp2",
    "hvaf_unph",
    "dr_mean",
    "dv_mean",
    "supp_total",
    "supp_hp1",
    "supp_hp2",
    "ref_total",
    "phase_balance",
    "n_samples_gt",
    "log_bnd_span",
    "local_sv_density_100kb",
    "local_sv_density_1mb",
    "cluster_size",
    "phase_set_size",
]
BINARY_COLS = [
    "is_precise",
    "is_vntr",
    "is_bnd",
    "has_phase",
    "hp_concordant",
    "strand_1_plus",
    "strand_1_minus",
    "strand_2_plus",
    "strand_2_minus",
    "strand_same_orientation",
    "is_inv_like_bnd",
    "is_foldback",
    "is_foldback_like",
    "has_interchrom_mate",
    "has_samechrom_mate",
]
SV_TYPE_MAP = {"DEL": 0, "INS": 1, "DUP": 2, "BND": 3, "sBND": 4, "INV": 5}
GT_MAP = {"0/0": 0, "0/1": 1, "1/1": 2, "./.": 3}

N_CONT = len(CONTINUOUS_COLS)
N_BIN = len(BINARY_COLS)
N_SVTYPE = len(SV_TYPE_MAP)
N_GT = len(GT_MAP)
N_FEAT = N_CONT + N_BIN + N_SVTYPE + N_GT

CHROM_ORDER = {f"chr{i}": i for i in range(1, 23)}
CHROM_ORDER.update({str(i): i for i in range(1, 23)})
CHROM_ORDER.update({"chrX": 23, "X": 23, "chrY": 24, "Y": 24, "chrM": 25, "MT": 25, "M": 25})

REQUIRED_COLUMNS = [
    "sample_id",
    "sv_id",
    "mate_id",
    "mate_chrom",
    "mate_pos",
    "cluster_id",
    "phase_set",
    "chrom",
    "pos",
    "end",
    "bnd_type",
    "detailed_type",
    "bnd_span",
    "sv_type_str",
    "sv_type",
    "maj_gt",
    *CONTINUOUS_COLS,
    *BINARY_COLS,
]

BND_ALT_RE = re.compile(r"[\[\]]([^:\[\]]+):([0-9]+)[\[\]]")


def _parse_info(info_str: str) -> dict[str, str | bool]:
    info: dict[str, str | bool] = {}
    if not info_str or info_str == ".":
        return info
    for field in info_str.split(";"):
        if not field:
            continue
        if "=" in field:
            key, value = field.split("=", 1)
            info[key] = value
        else:
            info[field] = True
    return info


def _clean_info_text(value: object) -> str:
    if value is None or value is True:
        return ""
    text = str(value).strip()
    return "" if text in {"", "."} else text


def _chrom_key(chrom: object) -> str:
    return str(chrom).strip().removeprefix("chr")


def _parse_bnd_alt(alt: object) -> tuple[str, int]:
    """Return mate chrom and 0-based mate position from a VCF BND ALT string."""
    text = "" if alt is None else str(alt)
    match = BND_ALT_RE.search(text)
    if not match:
        return "", -1
    chrom = match.group(1)
    pos = max(_as_int(match.group(2), default=0) - 1, 0)
    return chrom, pos


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value is True or value == "." or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value is True or value == "." or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_hvaf(value: object) -> tuple[float, float, float]:
    try:
        parts = [float(x) for x in str(value).replace("|", ",").split(",")]
        if len(parts) >= 3:
            return parts[0], parts[1], parts[2]
    except Exception:
        pass
    return 0.0, 0.0, 0.0


def _parse_supp(value: object) -> tuple[float, float, float]:
    """Parse SUPP_READS as total/hp1/hp2 triplets, summing all samples/groups."""
    try:
        vals = [float(x) for x in str(value).replace(",", ":").split(":") if x not in {"", "."}]
    except Exception:
        return 0.0, 0.0, 0.0

    total = hp1 = hp2 = 0.0
    if len(vals) >= 3:
        for i in range(0, len(vals) - 2, 3):
            total += vals[i]
            hp1 += vals[i + 1]
            hp2 += vals[i + 2]
    elif len(vals) == 1:
        total = vals[0]
    return total, hp1, hp2


def _parse_ref_reads(value: object) -> float:
    try:
        return float(sum(float(x) for x in str(value).replace(",", ":").split(":") if x not in {"", "."}))
    except Exception:
        return 0.0


def _normalise_gt(gt: str) -> str:
    text = (gt or "./.").strip()
    if text in {".", "./.", ".|."}:
        return "./."
    text = text.replace("|", "/")
    # Severus can emit phased genotypes; sort diploid alleles so 1/0 maps to 0/1.
    if "/" in text:
        alleles = text.split("/")
        if len(alleles) == 2 and all(a in {"0", "1"} for a in alleles):
            return "/".join(sorted(alleles))
    return text if text in GT_MAP else "./."


def _majority_gt(gts: list[str]) -> int:
    counts: dict[str, int] = {}
    for gt in gts:
        key = _normalise_gt(gt)
        if key != "./.":
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return GT_MAP["./."]
    return GT_MAP.get(max(counts, key=counts.get), GT_MAP["./."])


def _parse_hp(value: object) -> tuple[bool, bool]:
    try:
        parts = str(value).replace("/", "|").split("|")
        if len(parts) < 2:
            return False, False
        a, b = int(parts[0]), int(parts[1])
        return (a > 0 or b > 0), (a > 0 and b > 0 and a == b)
    except Exception:
        return False, False


def _parse_phase_set(value: object) -> int:
    try:
        parts = [int(float(x)) for x in str(value).replace(",", "|").split("|") if x not in {"", "."}]
        nonzero = [p for p in parts if p != 0]
        return nonzero[0] if nonzero else 0
    except Exception:
        return 0


def _parse_strands(value: object) -> tuple[float, float, float, float]:
    """
    Encode Severus INFO/STRANDS as ordered breakpoint-side orientation flags.

    Severus emits two-character orientations such as '+-', '-+', '++', and
    '--'. Single breakends can carry one side only. Unknown/missing sides are
    all-zero.
    """
    text = "" if value is None or value is True else str(value).strip()
    signs = [ch for ch in text if ch in {"+", "-"}]
    first = signs[0] if len(signs) >= 1 else ""
    second = signs[1] if len(signs) >= 2 else ""
    return (
        1.0 if first == "+" else 0.0,
        1.0 if first == "-" else 0.0,
        1.0 if second == "+" else 0.0,
        1.0 if second == "-" else 0.0,
    )


def _same_orientation(strands: object) -> bool:
    signs = [ch for ch in str(strands or "") if ch in {"+", "-"}]
    return len(signs) >= 2 and signs[0] == signs[1]


def _add_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cohort-local density and group-size features after sorting."""
    df = df.copy()

    for span_bp, col in [(100_000, "local_sv_density_100kb"), (1_000_000, "local_sv_density_1mb")]:
        df[col] = 0.0
        half_span = int(span_bp) // 2
        for (_sample_id, _chrom), grp in df.groupby(["sample_id", "chrom"], sort=False):
            positions = pd.to_numeric(grp["pos"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
            order = np.argsort(positions)
            sorted_pos = positions[order]
            left = np.searchsorted(sorted_pos, sorted_pos - half_span, side="left")
            right = np.searchsorted(sorted_pos, sorted_pos + half_span, side="right")
            density = (right - left).astype(np.float32) / (float(span_bp) / 1_000_000.0)
            df.loc[grp.index.to_numpy()[order], col] = density

    df["cluster_size"] = 0.0
    cluster_text = df["cluster_id"].astype(str)
    valid_cluster = ~cluster_text.isin(["", ".", "nan", "None"])
    if valid_cluster.any():
        df.loc[valid_cluster, "cluster_size"] = (
            df.loc[valid_cluster]
            .groupby(["sample_id", "cluster_id"], sort=False)["sv_id"]
            .transform("size")
            .astype(float)
        )

    df["phase_set_size"] = 0.0
    phase_numeric = pd.to_numeric(df["phase_set"], errors="coerce").fillna(0).astype(int)
    valid_phase = phase_numeric != 0
    if valid_phase.any():
        tmp = df.loc[valid_phase, ["sample_id", "sv_id"]].copy()
        tmp["phase_set"] = phase_numeric.loc[valid_phase].to_numpy(dtype=np.int64)
        df.loc[valid_phase, "phase_set_size"] = (
            tmp.groupby(["sample_id", "phase_set"], sort=False)["sv_id"]
            .transform("size")
            .astype(float)
            .to_numpy()
        )

    return df


def _chrom_sort_key(chrom: object) -> int:
    text = str(chrom)
    return CHROM_ORDER.get(text, CHROM_ORDER.get(text.removeprefix("chr"), 99))


def _clean_sample_id_part(part: str) -> str:
    if part.startswith("wakhan_") and len(part) > len("wakhan_"):
        return part.replace("wakhan_", "", 1)
    return part


def _looks_like_analysis_dir(part: str) -> bool:
    return part.startswith("sv_cna_") or part in {"lumos_out", "mishas_analysis", "sniffles2"}


def infer_sample_id_from_vcf(path: str | Path) -> str:
    """
    Infer a stable sample id from a Severus VCF path.

    Many cohort layouts store every VCF under the same basename, for example:
        sample_a/severus/somatic_SVs/severus_somatic.vcf
        sample_a/sv_cna_v2/severus/somatic_SVs/severus_somatic.vcf

    Falling back to only path.stem would collapse those inputs into one sample.
    Prefer the enclosing sample directory when a standard Severus path shape is
    present, then use informative parent directory names before the stem.
    """
    path = Path(path)

    if path.parent.name in {"somatic_SVs", "all_SVs"} and path.parent.parent.name == "severus":
        before_severus = path.parent.parent.parent
        if _looks_like_analysis_dir(before_severus.name) and len(path.parents) > 3:
            return _clean_sample_id_part(path.parents[3].name)
        return _clean_sample_id_part(before_severus.name)

    if path.stem in {"severus_somatic", "severus_all", "severus", "somatic_SVs", "all_SVs"}:
        for parent in path.parents:
            if parent.name == "severus" and parent.parent.name:
                return _clean_sample_id_part(parent.parent.name)
        if path.parent.name:
            return _clean_sample_id_part(path.parent.name)

    for part in reversed(path.parts):
        if part.startswith("wakhan_") and part not in {"wakhan_paper"}:
            return _clean_sample_id_part(part)

    return path.stem


def parse_severus(path: str | Path, sample_id: Optional[str] = None) -> pd.DataFrame:
    """
    Parse one Severus VCF into a canonical per-SV DataFrame.

    Multi-sample VCF records are aggregated across all sample columns to match
    the graph pretraining feature schema: VAF/DR/DV/hVAF are averaged over
    non-missing genotypes, and the majority genotype is encoded as maj_gt.
    """
    path = Path(path)
    if sample_id is None:
        sample_id = infer_sample_id_from_vcf(path)
    if not path.exists():
        raise FileNotFoundError(f"Severus VCF not found: {path}")

    records: list[dict] = []
    with path.open() as fh:
        for line in fh:
            if not line or line.startswith("##"):
                continue
            if line.startswith("#CHROM"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue

            chrom = parts[0]
            pos_1based = _as_int(parts[1], default=1)
            pos = max(pos_1based - 1, 0)
            sv_id = parts[2] if parts[2] != "." else f"{chrom}:{pos_1based}:{len(records)}"
            qual = _as_float(parts[5], default=0.0)
            info = _parse_info(parts[7])
            fmt_keys = parts[8].split(":") if parts[8] and parts[8] != "." else []
            sample_cols = parts[9:]

            sv_type_raw = str(info.get("SVTYPE", "DEL"))
            alt = parts[4] if len(parts) > 4 else ""
            bnd_type = _clean_info_text(info.get("BND_TYPE", ""))
            detailed_type = _clean_info_text(info.get("DETAILED_TYPE", ""))
            svlen = abs(_as_float(info.get("SVLEN", 0.0), default=0.0))
            end = _as_int(info.get("END"), default=pos + max(1, int(svlen) if svlen else 1))
            # INFO/END is 1-based inclusive in VCF; as a half-open BED end it is
            # numerically the same endpoint after POS -> POS-1 conversion.
            end = max(end, pos + 1)
            mapq = _as_float(info.get("MAPQ", 0.0), default=0.0)
            is_bnd = 1.0 if sv_type_raw in {"BND", "sBND"} else 0.0
            mate_chrom, mate_pos = _parse_bnd_alt(alt) if is_bnd else ("", -1)
            has_samechrom_mate = bool(mate_chrom) and _chrom_key(mate_chrom) == _chrom_key(chrom)
            has_interchrom_mate = bool(mate_chrom) and _chrom_key(mate_chrom) != _chrom_key(chrom)
            if has_samechrom_mate and mate_pos >= 0:
                bnd_span = abs(int(mate_pos) - int(pos))
            elif is_bnd and svlen > 0:
                bnd_span = int(svlen)
            else:
                bnd_span = 0

            mate_id = str(info.get("MATE_ID", "") or "")
            cluster_id = str(info.get("CLUSTERID", "") or "")
            phase_set = _parse_phase_set(info.get("PHASESETID", "0|0"))

            supp_total, supp_hp1, supp_hp2 = _parse_supp(info.get("SUPP_READS", "0"))
            ref_total = _parse_ref_reads(info.get("REF_READS", "0"))
            has_phase_hp, hp_concordant = _parse_hp(info.get("HP", "0|0"))
            strands_raw = info.get("STRANDS", "")
            strand_1_plus, strand_1_minus, strand_2_plus, strand_2_minus = _parse_strands(
                strands_raw
            )
            strand_same_orientation = _same_orientation(strands_raw)
            is_inv_like_bnd = bool(is_bnd) and (
                bnd_type == "INV_LIKE" or (strand_same_orientation and has_samechrom_mate)
            )
            detailed_lower = detailed_type.lower()
            is_foldback_exact = detailed_lower == "foldback"
            is_foldback_like = bool(is_inv_like_bnd and has_samechrom_mate and 0 < bnd_span <= 50_000)
            is_foldback = bool(is_foldback_exact or is_foldback_like)

            vafs: list[float] = []
            drs: list[float] = []
            dvs: list[float] = []
            hp1s: list[float] = []
            hp2s: list[float] = []
            unphs: list[float] = []
            gts: list[str] = []
            n_valid = 0

            for sample_text in sample_cols:
                sample_values = sample_text.split(":")
                sample_dict = dict(zip(fmt_keys, sample_values))
                gt = sample_dict.get("GT", "./.")
                gts.append(gt)
                if _normalise_gt(gt) == "./.":
                    continue
                n_valid += 1

                if "VAF" in sample_dict:
                    vafs.append(_as_float(sample_dict.get("VAF"), 0.0))
                if "DR" in sample_dict:
                    drs.append(_as_float(sample_dict.get("DR"), 0.0))
                if "DV" in sample_dict:
                    dvs.append(_as_float(sample_dict.get("DV"), 0.0))
                h1, h2, hu = _parse_hvaf(sample_dict.get("hVAF", "0,0,0"))
                hp1s.append(h1)
                hp2s.append(h2)
                unphs.append(hu)

            mean = lambda values: float(np.mean(values)) if values else 0.0
            std = lambda values: float(np.std(values)) if len(values) > 1 else 0.0

            records.append(
                {
                    "sample_id": str(sample_id),
                    "sv_id": sv_id,
                    "mate_id": mate_id if mate_id != "." else "",
                    "mate_chrom": mate_chrom,
                    "mate_pos": int(mate_pos),
                    "cluster_id": cluster_id if cluster_id != "." else "",
                    "phase_set": int(phase_set),
                    "chrom": chrom,
                    "pos": int(pos),
                    "end": int(end),
                    "bnd_type": bnd_type,
                    "detailed_type": detailed_type,
                    "bnd_span": int(bnd_span),
                    "sv_type_str": sv_type_raw,
                    "sv_type": SV_TYPE_MAP.get(sv_type_raw, 0),
                    "maj_gt": _majority_gt(gts),
                    "qual": float(qual),
                    "log_svlen": math.log1p(svlen),
                    "mapq": float(mapq),
                    "vaf_mean": mean(vafs),
                    "vaf_std": std(vafs),
                    "hvaf_hp1": mean(hp1s),
                    "hvaf_hp2": mean(hp2s),
                    "hvaf_unph": mean(unphs),
                    "dr_mean": mean(drs),
                    "dv_mean": mean(dvs),
                    "supp_total": float(supp_total),
                    "supp_hp1": float(supp_hp1),
                    "supp_hp2": float(supp_hp2),
                    "ref_total": float(ref_total),
                    "phase_balance": float(abs(supp_hp1 - supp_hp2) / (supp_hp1 + supp_hp2 + 1.0)),
                    "n_samples_gt": float(n_valid),
                    "log_bnd_span": math.log1p(float(bnd_span)) if is_bnd and bnd_span > 0 else 0.0,
                    "is_precise": 1.0 if "PRECISE" in info else 0.0,
                    "is_vntr": 1.0 if str(info.get("INSIDE_VNTR", "")).upper() == "TRUE" else 0.0,
                    "is_bnd": float(is_bnd),
                    "has_phase": 1.0 if (phase_set != 0 or has_phase_hp) else 0.0,
                    "hp_concordant": 1.0 if hp_concordant else 0.0,
                    "strand_1_plus": strand_1_plus,
                    "strand_1_minus": strand_1_minus,
                    "strand_2_plus": strand_2_plus,
                    "strand_2_minus": strand_2_minus,
                    "strand_same_orientation": 1.0 if strand_same_orientation else 0.0,
                    "is_inv_like_bnd": 1.0 if is_inv_like_bnd else 0.0,
                    "is_foldback": 1.0 if is_foldback else 0.0,
                    "is_foldback_like": 1.0 if is_foldback_like else 0.0,
                    "has_interchrom_mate": 1.0 if has_interchrom_mate else 0.0,
                    "has_samechrom_mate": 1.0 if has_samechrom_mate else 0.0,
                }
            )

    df = pd.DataFrame(records)
    if df.empty:
        log.warning("No Severus records parsed from %s", path)
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df["_chrom_order"] = df["chrom"].map(_chrom_sort_key)
    df = df.sort_values(["_chrom_order", "pos", "end"]).drop(columns="_chrom_order")
    df = df.reset_index(drop=True)
    df = _add_context_features(df)

    log.info(
        "  %s: %d SVs, %d chroms, %d BND mates, %d foldback, %d interchrom, %d phased",
        sample_id,
        len(df),
        df["chrom"].nunique(),
        int((df["mate_id"] != "").sum()),
        int((df["is_foldback"] > 0).sum()),
        int((df["has_interchrom_mate"] > 0).sum()),
        int((df["phase_set"] != 0).sum()),
    )
    return df[REQUIRED_COLUMNS]


def parse_all_severus(paths: list[str | Path]) -> pd.DataFrame:
    """Parse and concatenate one or more Severus VCF files."""
    log.info("Parsing %d Severus VCF file(s) ...", len(paths))
    frames: list[pd.DataFrame] = []
    for p in paths:
        df = parse_severus(p)
        if df.empty:
            log.warning("Severus file returned no records: %s", p)
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("No valid Severus records found in any input file.")

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Total: %d SVs across %d sample(s).",
        len(combined),
        combined["sample_id"].nunique(),
    )
    return combined


def build_node_features(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
) -> tuple[np.ndarray, RobustScaler]:
    """
    Build the graph-node feature matrix.

    Column order:
      continuous features after percentile clipping and RobustScaler,
      binary features, including ordered STRANDS orientation flags,
      6 SVTYPE one-hot features,
      4 genotype one-hot features.
    """
    missing = [c for c in [*CONTINUOUS_COLS, *BINARY_COLS, "sv_type", "maj_gt"] if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot build Severus node features; missing columns: {missing}")

    cont = df[CONTINUOUS_COLS].to_numpy(dtype=np.float32, copy=True)
    if cont.size == 0:
        raise ValueError("Cannot build node features from an empty Severus DataFrame.")

    p1 = np.percentile(cont, 1, axis=0)
    p99 = np.percentile(cont, 99, axis=0)
    cont = np.clip(cont, p1, p99)

    if scaler is None:
        scaler = RobustScaler()
        cont = scaler.fit_transform(cont)
    else:
        cont = scaler.transform(cont)

    binary = df[BINARY_COLS].to_numpy(dtype=np.float32, copy=True)
    sv_oh = np.eye(N_SVTYPE, dtype=np.float32)[
        np.clip(df["sv_type"].to_numpy(dtype=int), 0, N_SVTYPE - 1)
    ]
    gt_oh = np.eye(N_GT, dtype=np.float32)[
        np.clip(df["maj_gt"].to_numpy(dtype=int), 0, N_GT - 1)
    ]

    features = np.concatenate([cont.astype(np.float32), binary, sv_oh, gt_oh], axis=1)
    if features.shape[1] != N_FEAT:
        raise RuntimeError(f"Expected {N_FEAT} features, observed {features.shape[1]}.")
    return features.astype(np.float32), scaler


# Backward-compatible alias matching the prototype scripts.
def build_feature_matrix(
    df: pd.DataFrame,
    scaler: RobustScaler | None = None,
) -> tuple[np.ndarray, RobustScaler]:
    return build_node_features(df, scaler=scaler)
