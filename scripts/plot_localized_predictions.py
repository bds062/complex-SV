"""Plot whole-arm or whole-chromosome CN and SV context for localized predictions."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from genomic_features.anchor_manifest import canonical_sample_id
from genomic_features.severus_parser import parse_severus
from genomic_features.wakhan_parser import parse_wakhan

log = logging.getLogger(__name__)
SCAN_EVIDENCE_VALUES = {"chromosome_scan", "chromosome_arm_scan"}

MISSING_VALUES = {"", "none", "nan", "null", "unknown", "unlabeled"}
CLASS_COLORS = {
    "BFB": "#E15759",
    "chromothripsis": "#4E79A7",
    "seismic_amplification": "#F28E2B",
    "TIC": "#59A14F",
}
SV_TYPE_COLORS = {
    "DEL": "#CF0759",
    "INV": "#2830DE",
    "INS": "#D4B000",
    "BND": "#737373",
    "sBND": "#9C755F",
    "DUP": "#178117",
}
STRAND_ORIENTATION_COLORS = {
    "+-": "#0072B2",
    "-+": "#009E73",
    "++": "#D55E00",
    "--": "#CC79A7",
    "+?": "#56B4E9",
    "-?": "#E69F00",
    "?+": "#8A63D2",
    "?-": "#A6761D",
    "unknown": "#6C6C6C",
}
FOLDBACK_HIGHLIGHT_COLOR = "#C51B29"
HAPLOTYPE_COLORS = {"HP1": "#D62728", "HP2": "#1F77B4", "mixed_HP": "#737373"}
DEFAULT_CENTROMERE_BED = PROJECT_ROOT / "genomic_features" / "grch38.cen_coord.curated.bed"
CENTROMERE_COLOR = "#9A9A9A"


def _clean_text(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"nan", "null"}:
        return ""
    return text


def _is_empty(value: object) -> bool:
    return _clean_text(value).lower() in MISSING_VALUES


def _split_class_set(value: object) -> list[str]:
    text = _clean_text(value)
    if not text or text.lower() in MISSING_VALUES:
        return []
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def _row_predicted_classes(row: pd.Series) -> list[str]:
    if "predicted_classes" in row and _clean_text(row.get("predicted_classes", "")):
        return _split_class_set(row.get("predicted_classes", ""))
    return _split_class_set(row.get("predicted_class", ""))


def _prediction_label(row: pd.Series) -> str:
    classes = _row_predicted_classes(row)
    return ";".join(classes) if classes else "none"


def _bool_value(value: object) -> bool | None:
    text = _clean_text(value).lower()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f"}:
        return False
    return None


def _display_haplotype_tag(value: object) -> str:
    tag = _clean_text(value)
    if not tag:
        return ""
    normalized = tag.lower().replace("-", "_")
    if normalized in {"bilateral", "mixed_hp"}:
        return "mixed_HP"
    if normalized == "hp1":
        return "HP1"
    if normalized == "hp2":
        return "HP2"
    return tag


def _plot_correctness_bucket(row: pd.Series) -> str:
    is_labeled = _bool_value(row.get("is_labeled", ""))
    if is_labeled is True:
        exact = _bool_value(row.get("class_exact_match", ""))
        if exact is not None:
            return "correct_preds" if exact else "incorrect_preds"
        any_match = _bool_value(row.get("class_any_match", ""))
        return "correct_preds" if any_match is True else "incorrect_preds"

    is_background = _bool_value(row.get("is_background_chromosome", ""))
    if is_background is True:
        called = _bool_value(row.get("called_complex_sv", ""))
        return "incorrect_preds" if called is True else "correct_preds"

    objectness_correct = _bool_value(row.get("objectness_correct", ""))
    if objectness_correct is not None:
        return "correct_preds" if objectness_correct else "incorrect_preds"

    exact = _bool_value(row.get("class_exact_match", ""))
    if exact is not None:
        return "correct_preds" if exact else "incorrect_preds"
    return ""


def _region_label(chrom: object, arm: object) -> str:
    chrom_text = _clean_text(chrom)
    arm_text = _clean_text(arm).lower()
    return f"{chrom_text}{arm_text}" if arm_text in {"p", "q"} else chrom_text


def _type_probability_classes(row: pd.Series) -> list[str]:
    available = [str(col).removeprefix("type_probability_") for col in row.index if str(col).startswith("type_probability_")]
    preferred = ["BFB", "chromothripsis", "seismic_amplification", "TIC"]
    ordered = [name for name in preferred if name in available]
    ordered.extend(sorted(name for name in available if name not in set(ordered)))
    return ordered


def _format_score_text(row: pd.Series, pred_classes: list[str]) -> str:
    explicit = _clean_text(row.get("score_text", ""))
    if explicit:
        return explicit

    objectness_prob = _numeric(row, "objectness_prob", np.nan)
    if np.isfinite(objectness_prob):
        parts = [f"objectness(sigmoid)={objectness_prob:.3g}"]
        type_parts: list[str] = []
        for class_name in _type_probability_classes(row):
            prob = _numeric(row, f"type_probability_{class_name}", np.nan)
            if np.isfinite(prob):
                type_parts.append(f"{class_name}={prob:.3g}")
        if type_parts:
            parts.append("type_probs(sigmoid): " + "; ".join(type_parts))
        return " | ".join(parts)

    distance = _numeric(row, "nearest_prototype_distance", np.nan)
    confidence = _numeric(row, "prototype_confidence", np.nan)
    legacy_parts: list[str] = []
    if np.isfinite(distance):
        legacy_parts.append(f"distance={distance:.4g}")
    if np.isfinite(confidence):
        legacy_parts.append(f"confidence={confidence:.3g}")
    return " | ".join(legacy_parts) if legacy_parts else "scores unavailable"


def _safe_name(value: object) -> str:
    text = _clean_text(value) or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unknown"


def _chrom_key(chrom: object) -> str:
    text = _clean_text(chrom)
    return text.removeprefix("chr").removeprefix("CHR")


def _chrom_equal(a: object, b: object) -> bool:
    return _chrom_key(a).lower() == _chrom_key(b).lower()


def _numeric(row: pd.Series, key: str, default: float = 0.0) -> float:
    value = pd.to_numeric(pd.Series([row.get(key, default)]), errors="coerce").iloc[0]
    return float(default if pd.isna(value) else value)


def read_centromeres(path: str | Path | None) -> pd.DataFrame:
    if path is None or not str(path).strip():
        return pd.DataFrame(columns=["chrom", "start", "end"])
    bed_path = Path(path)
    if not bed_path.exists():
        log.warning("Centromere BED not found: %s", bed_path)
        return pd.DataFrame(columns=["chrom", "start", "end"])
    df = pd.read_csv(
        bed_path,
        sep="\t",
        comment="#",
        header=None,
        names=["chrom", "start", "end"],
        usecols=[0, 1, 2],
    )
    df["start"] = pd.to_numeric(df["start"], errors="coerce")
    df["end"] = pd.to_numeric(df["end"], errors="coerce")
    df = df.dropna(subset=["chrom", "start", "end"]).copy()
    df = df[df["end"] > df["start"]].copy()
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df.reset_index(drop=True)


def _subset_centromeres(centromeres: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> pd.DataFrame:
    if centromeres.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "plot_start", "plot_end"])
    mask = (
        centromeres["chrom"].map(lambda value: _chrom_equal(value, chrom))
        & (pd.to_numeric(centromeres["end"], errors="coerce") > start_bp)
        & (pd.to_numeric(centromeres["start"], errors="coerce") < end_bp)
    )
    out = centromeres.loc[mask].copy()
    if out.empty:
        return pd.DataFrame(columns=["chrom", "start", "end", "plot_start", "plot_end"])
    out["plot_start"] = pd.to_numeric(out["start"], errors="coerce").clip(lower=start_bp, upper=end_bp).astype(int)
    out["plot_end"] = pd.to_numeric(out["end"], errors="coerce").clip(lower=start_bp, upper=end_bp).astype(int)
    out = out[out["plot_end"] > out["plot_start"]].copy()
    return out.reset_index(drop=True)


def _format_centromere_intervals(centromeres: pd.DataFrame) -> str:
    if centromeres.empty:
        return ""
    return ";".join(f"{int(row.plot_start)}-{int(row.plot_end)}" for row in centromeres.itertuples())


def _draw_centromere_annotations(axes: np.ndarray, centromeres: pd.DataFrame) -> None:
    if centromeres.empty:
        return
    for _, row in centromeres.iterrows():
        cen_start = int(row["plot_start"]) / 1e6
        cen_end = int(row["plot_end"]) / 1e6
        cen_mid = (cen_start + cen_end) / 2.0
        for ax in axes:
            ax.axvspan(cen_start, cen_end, color=CENTROMERE_COLOR, alpha=0.16, linewidth=0, zorder=0)
            ax.axvline(cen_start, color=CENTROMERE_COLOR, linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
            ax.axvline(cen_end, color=CENTROMERE_COLOR, linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
        axes[0].text(
            cen_mid,
            0.96,
            "centromere",
            transform=axes[0].get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
            color="#555555",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.75),
            zorder=22,
        )


def _format_bp(value: int | float) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return str(value)


def _plot_window_interval(row: pd.Series, candidate_start_bp: int, candidate_end_bp: int) -> tuple[int, int]:
    context_start = _numeric(row, "context_start_bp", np.nan)
    context_end = _numeric(row, "context_end_bp", np.nan)
    if np.isfinite(context_start) and np.isfinite(context_end) and int(context_end) > int(context_start):
        return int(context_start), int(context_end)
    return candidate_start_bp, candidate_end_bp


def _highlight_interval(
    row: pd.Series,
    plot_start_bp: int,
    plot_end_bp: int,
    candidate_start_bp: int,
    candidate_end_bp: int,
) -> tuple[int, int]:
    # Candidate-region plots show the full arm/chromosome context, with the merged
    # candidate interval highlighted. Only explicit external highlights override it.
    highlight_start = int(_numeric(row, "highlight_start_bp", candidate_start_bp))
    highlight_end = int(_numeric(row, "highlight_end_bp", candidate_end_bp))
    if highlight_end < highlight_start:
        highlight_start, highlight_end = highlight_end, highlight_start
    highlight_start = max(plot_start_bp, min(highlight_start, plot_end_bp))
    highlight_end = max(plot_start_bp, min(highlight_end, plot_end_bp))
    if highlight_end <= highlight_start:
        highlight_start, highlight_end = candidate_start_bp, candidate_end_bp
        highlight_start = max(plot_start_bp, min(highlight_start, plot_end_bp))
        highlight_end = max(plot_start_bp, min(highlight_end, plot_end_bp))
    return highlight_start, highlight_end


def _bool_numeric(row: pd.Series, key: str) -> bool:
    return _numeric(row, key, 0.0) > 0


def _strand_orientation(row: pd.Series) -> str:
    if _bool_numeric(row, "strand_1_plus"):
        first = "+"
    elif _bool_numeric(row, "strand_1_minus"):
        first = "-"
    else:
        first = "?"

    if _bool_numeric(row, "strand_2_plus"):
        second = "+"
    elif _bool_numeric(row, "strand_2_minus"):
        second = "-"
    else:
        second = "?"

    orientation = f"{first}{second}"
    return "unknown" if orientation == "??" else orientation


def _orientation_color(orientation: object) -> str:
    return STRAND_ORIENTATION_COLORS.get(str(orientation), STRAND_ORIENTATION_COLORS["unknown"])


def _orientation_sort_key(orientation: object) -> tuple[int, str]:
    text = str(orientation)
    ordered = list(STRAND_ORIENTATION_COLORS)
    rank = ordered.index(text) if text in STRAND_ORIENTATION_COLORS else len(ordered)
    return rank, text


def _is_foldback_sv(row: pd.Series) -> bool:
    return _bool_numeric(row, "is_foldback") or _bool_numeric(row, "is_foldback_like")


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t").fillna("")
    required = {"sample_id", "wakhan_root", "severus_vcf"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    df["_sample_key"] = df["sample_id"].map(lambda value: canonical_sample_id(str(value)))
    return df


def select_predicted_chromosomes(distances: pd.DataFrame, plot_scope: str = "unlabeled-called") -> pd.DataFrame:
    if distances.empty:
        return pd.DataFrame()
    required = {"sample_id", "chrom"}
    missing = sorted(required.difference(distances.columns))
    if missing:
        raise ValueError(f"Prediction table missing columns: {missing}")

    df = distances.copy().fillna("")
    scope = str(plot_scope or "unlabeled-called").strip().lower().replace("_", "-")
    if scope == "test":
        if "split" not in df.columns:
            raise ValueError("--plot_scope test requires a split column in the prediction table")
        df = df[df["split"].astype(str) == "test"].copy()
    elif scope == "unlabeled-called":
        if "evidence" in df.columns:
            df = df[df["evidence"].astype(str).isin(SCAN_EVIDENCE_VALUES)].copy()
        if "sv_class" in df.columns:
            df = df[df["sv_class"].map(_is_empty)].copy()
        df = df[df.apply(lambda row: bool(_row_predicted_classes(row)), axis=1)].copy()
    elif scope == "called":
        df = df[df.apply(lambda row: bool(_row_predicted_classes(row)), axis=1)].copy()
    elif scope == "all":
        pass
    else:
        raise ValueError("plot_scope must be one of: unlabeled-called, called, all, test")

    if df.empty:
        return df

    for col in ["start_bp", "end_bp", "nearest_prototype_distance", "prototype_confidence", "objectness_prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_pred_sort"] = df.apply(_prediction_label, axis=1)
    sort_cols = [col for col in ["split", "sample_id", "chrom", "arm", "start_bp", "nearest_prototype_distance", "_pred_sort"] if col in df.columns]
    return df.sort_values(sort_cols).drop(columns=["_pred_sort"], errors="ignore").reset_index(drop=True)


def _subset_segments(wakhan_df: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> pd.DataFrame:
    if wakhan_df.empty:
        return pd.DataFrame()
    mask = (
        wakhan_df["chrom"].map(lambda value: _chrom_equal(value, chrom))
        & (pd.to_numeric(wakhan_df["end"], errors="coerce") > start_bp)
        & (pd.to_numeric(wakhan_df["start"], errors="coerce") < end_bp)
    )
    return wakhan_df.loc[mask].copy().sort_values(["start", "end"])


def _subset_sv_on_chrom(severus_df: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> pd.DataFrame:
    if severus_df.empty:
        return pd.DataFrame()
    pos = pd.to_numeric(severus_df["pos"], errors="coerce")
    end = pd.to_numeric(severus_df["end"], errors="coerce")
    mask = (
        severus_df["chrom"].map(lambda value: _chrom_equal(value, chrom))
        & (((pos >= start_bp) & (pos <= end_bp)) | ((end >= start_bp) & (end <= end_bp)))
    )
    return severus_df.loc[mask].copy().sort_values(["pos", "end"])


def _line_collection_for_segments(
    segs: pd.DataFrame,
    column: str,
    start_bp: int,
    end_bp: int,
    y_scale: float = 1.0,
) -> tuple[LineCollection | None, np.ndarray]:
    if segs.empty or column not in segs:
        return None, np.array([], dtype=float)
    starts = pd.to_numeric(segs["start"], errors="coerce").fillna(start_bp).clip(lower=start_bp, upper=end_bp).to_numpy(float) / 1e6
    ends = pd.to_numeric(segs["end"], errors="coerce").fillna(end_bp).clip(lower=start_bp, upper=end_bp).to_numpy(float) / 1e6
    raw_values = pd.to_numeric(segs[column], errors="coerce").fillna(0.0).to_numpy(float)
    plot_values = raw_values * float(y_scale)
    valid = ends > starts
    if not np.any(valid):
        return None, np.array([], dtype=float)
    lines = [[(starts[i], plot_values[i]), (ends[i], plot_values[i])] for i in np.flatnonzero(valid)]
    return LineCollection(lines, linewidths=2.2), raw_values[valid]


def _plot_cn_panel(ax, segs: pd.DataFrame, start_bp: int, end_bp: int) -> None:
    ax.set_ylabel("Haplotype CN")
    ax.axhline(0, color="#2F2F2F", linewidth=0.9)
    if segs.empty:
        ax.text(0.5, 0.5, "No WAKHAN CN segments", transform=ax.transAxes, ha="center", va="center", fontsize=10)
        ax.grid(axis="y", alpha=0.2)
        return

    for _, row in segs.iterrows():
        if int(_numeric(row, "loh", 0)) > 0:
            x0 = max(float(row["start"]), start_bp) / 1e6
            x1 = min(float(row["end"]), end_bp) / 1e6
            ax.axvspan(x0, x1, color="#E15759", alpha=0.07, linewidth=0)

    tracks = [
        ("cn_hp1", "HP1", "#E15759", 2.2, 1.0),
        ("cn_hp2", "HP2", "#4E79A7", 2.2, -1.0),
    ]
    y_values: list[np.ndarray] = []
    handles: list[Line2D] = []
    for column, label, color, width, y_scale in tracks:
        collection, values = _line_collection_for_segments(segs, column, start_bp, end_bp, y_scale=y_scale)
        if collection is None:
            continue
        collection.set_color(color)
        collection.set_linewidth(width)
        ax.add_collection(collection)
        y_values.append(values)
        handles.append(Line2D([0], [0], color=color, lw=width, label=label))

    if y_values:
        max_y = float(np.nanmax(np.concatenate(y_values)))
        limit = max(2.5, max_y + 0.8)
        ax.set_ylim(-limit, limit)
        tick_max = int(np.ceil(limit))
        tick_step = max(1, int(np.ceil(tick_max / 5)))
        ticks = np.arange(-tick_max, tick_max + 1, tick_step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(abs(int(tick))) if tick != 0 else "0" for tick in ticks])
    ax.text(0.01, 0.90, "HP1", transform=ax.transAxes, color="#E15759", fontsize=9, fontweight="bold")
    ax.text(0.01, 0.08, "HP2", transform=ax.transAxes, color="#4E79A7", fontsize=9, fontweight="bold")
    ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.2)


def _plot_breakpoint_panel(
    ax,
    segs: pd.DataFrame,
    severus_df: pd.DataFrame,
    sv_chr: pd.DataFrame,
    chrom: str,
    start_bp: int,
    end_bp: int,
) -> None:
    ax.set_ylabel("Breakpoints")
    max_bp = 1.0
    if not segs.empty and "breakpoint_count" in segs:
        bp = pd.to_numeric(segs["breakpoint_count"], errors="coerce").fillna(0.0).to_numpy(float)
        starts = pd.to_numeric(segs["start"], errors="coerce").fillna(start_bp).clip(lower=start_bp, upper=end_bp).to_numpy(float)
        ends = pd.to_numeric(segs["end"], errors="coerce").fillna(end_bp).clip(lower=start_bp, upper=end_bp).to_numpy(float)
        valid = (ends > starts) & (bp > 0)
        if np.any(valid):
            centers = (starts[valid] + ends[valid]) / 2e6
            widths = np.maximum((ends[valid] - starts[valid]) / 1e6, 0.05)
            ax.bar(centers, bp[valid], width=widths, color="#9C755F", alpha=0.32, align="center", label="WAKHAN breakpoint count")
            max_bp = max(max_bp, float(np.max(bp[valid])))

    if not sv_chr.empty:
        y_top = max_bp * 1.12
        for _, sv in sv_chr.iterrows():
            x_bp = _numeric(sv, "pos", np.nan)
            if not np.isfinite(x_bp) or x_bp < start_bp or x_bp > end_bp:
                continue
            color = _orientation_color(_strand_orientation(sv))
            x_mb = x_bp / 1e6
            if _is_foldback_sv(sv):
                ax.vlines([x_mb], 0, y_top, colors=FOLDBACK_HIGHLIGHT_COLOR, linewidth=1.9, alpha=0.82)
                ax.vlines([x_mb], 0, y_top, colors=color, linewidth=0.85, alpha=0.96)
            else:
                ax.vlines([x_mb], 0, y_top, colors=color, linewidth=0.65, alpha=0.52)
        max_bp = max(max_bp, y_top)

    arcs, _markers = _same_chrom_arcs(severus_df, sv_chr, chrom, start_bp, end_bp)
    if arcs:
        region_mb = max((end_bp - start_bp) / 1e6, 1e-4)
        arc_base = max_bp * 1.10
        arc_height = max(max_bp * 0.55, 0.8)
        for arc in arcs:
            x0 = float(arc["x0"]) / 1e6
            x1 = float(arc["x1"]) / 1e6
            color = _orientation_color(arc.get("orientation", "unknown"))
            xs, unit_y = _arc_points(x0, x1, region_mb)
            is_offscreen = str(arc.get("kind", "")).startswith("offscreen") or str(arc.get("kind", "")) == "interchrom_mate"
            is_foldback = bool(arc.get("is_foldback", False))
            y = arc_base + unit_y * arc_height
            if is_foldback:
                ax.plot(xs, y, color=FOLDBACK_HIGHLIGHT_COLOR, linewidth=2.0, alpha=0.72)
            ax.plot(
                xs,
                y,
                color=color,
                linewidth=1.05 if is_foldback else (0.85 if is_offscreen else 0.7),
                alpha=0.90 if is_foldback else (0.55 if is_offscreen else 0.42),
            )
        max_bp = max(max_bp, arc_base + arc_height * 1.12)

    ax.set_ylim(0, max_bp * 1.08 + 0.05)
    ax.grid(axis="y", alpha=0.2)


def _arc_points(x0: float, x1: float, region_mb: float) -> tuple[np.ndarray, np.ndarray]:
    if x1 < x0:
        x0, x1 = x1, x0
    xs = np.linspace(x0, x1, 80)
    span = max(x1 - x0, 1e-4)
    height = 0.12 + 0.86 * min(1.0, np.sqrt(span / max(region_mb, 1e-4)))
    ys = 0.05 + height * np.sin(np.pi * (xs - x0) / span)
    return xs, ys


def _offscreen_endpoint(anchor_bp: int, target_bp: int | None, start_bp: int, end_bp: int) -> int:
    """Return an endpoint just outside the plotted interval for clipped arcs."""
    span = max(int(end_bp) - int(start_bp), 1)
    margin = max(int(span * 0.08), 1)
    if target_bp is not None:
        if int(target_bp) < int(start_bp):
            return int(start_bp) - margin
        if int(target_bp) > int(end_bp):
            return int(end_bp) + margin
        return int(target_bp)
    midpoint = (int(start_bp) + int(end_bp)) / 2.0
    return int(end_bp) + margin if int(anchor_bp) <= midpoint else int(start_bp) - margin


def _visible_anchor_bp(x0: int, x1: int, start_bp: int, end_bp: int) -> int:
    if start_bp <= int(x0) <= end_bp:
        return int(x0)
    if start_bp <= int(x1) <= end_bp:
        return int(x1)
    return int(np.clip(int(x0), start_bp, end_bp))


def _same_chrom_arcs(severus_df: pd.DataFrame, sv_chr: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arcs: list[dict[str, Any]] = []
    markers: list[dict[str, Any]] = []
    if sv_chr.empty:
        return arcs, markers

    by_id = {str(row["sv_id"]): row for _, row in severus_df.iterrows()} if "sv_id" in severus_df else {}
    seen_pairs: set[tuple[str, str]] = set()
    for _, row in sv_chr.iterrows():
        sv_id = str(row.get("sv_id", ""))
        mate_id = str(row.get("mate_id", ""))
        sv_type = str(row.get("sv_type_str", "SV"))
        orientation = _strand_orientation(row)
        is_foldback = _is_foldback_sv(row)
        pos = int(_numeric(row, "pos", start_bp))
        end = int(_numeric(row, "end", pos + 1))
        if mate_id and mate_id in by_id:
            mate = by_id[mate_id]
            pair = tuple(sorted([sv_id, mate_id]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            mate_chrom = str(mate.get("chrom", ""))
            mate_pos = int(_numeric(mate, "pos", pos))
            if _chrom_equal(mate_chrom, chrom):
                x1 = mate_pos if start_bp <= mate_pos <= end_bp else _offscreen_endpoint(pos, mate_pos, start_bp, end_bp)
                kind = "mate" if start_bp <= mate_pos <= end_bp else "offscreen_mate"
                label = "" if kind == "mate" else f"to {mate_chrom}"
            else:
                x1 = _offscreen_endpoint(pos, None, start_bp, end_bp)
                kind = "interchrom_mate"
                label = f"to {mate_chrom}" if mate_chrom else "BND"
            arcs.append({
                "x0": pos,
                "x1": x1,
                "type": sv_type,
                "orientation": orientation,
                "is_foldback": is_foldback,
                "kind": kind,
                "label": label,
                "anchor": pos,
            })
            continue

        if sv_type in {"DEL", "DUP", "INV", "BND"} and end > pos + 1:
            x0 = pos
            x1 = end if start_bp <= end <= end_bp else _offscreen_endpoint(pos, end, start_bp, end_bp)
            kind = "interval" if start_bp <= end <= end_bp and start_bp <= pos <= end_bp else "offscreen_interval"
            arcs.append({
                "x0": x0,
                "x1": x1,
                "type": sv_type,
                "orientation": orientation,
                "is_foldback": is_foldback,
                "kind": kind,
                "label": sv_type if kind != "interval" else "",
                "anchor": _visible_anchor_bp(x0, x1, start_bp, end_bp),
            })
        else:
            markers.append({"x": pos, "type": sv_type, "orientation": orientation, "is_foldback": is_foldback, "label": sv_type})
    return arcs, markers


def _plot_sv_panel(ax, severus_df: pd.DataFrame, sv_chr: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> None:
    region_mb = max((end_bp - start_bp) / 1e6, 1e-4)
    arcs, markers = _same_chrom_arcs(severus_df, sv_chr, chrom, start_bp, end_bp)

    offscreen_arcs: list[dict[str, Any]] = []
    for arc in arcs:
        x0 = float(arc["x0"]) / 1e6
        x1 = float(arc["x1"]) / 1e6
        color = _orientation_color(arc.get("orientation", "unknown"))
        xs, ys = _arc_points(x0, x1, region_mb)
        is_offscreen = str(arc.get("kind", "")).startswith("offscreen") or str(arc.get("kind", "")) == "interchrom_mate"
        is_foldback = bool(arc.get("is_foldback", False))
        if is_foldback:
            ax.plot(xs, ys, color=FOLDBACK_HIGHLIGHT_COLOR, linewidth=3.0, alpha=0.78, zorder=3)
        ax.plot(
            xs,
            ys,
            color=color,
            linewidth=1.55 if is_foldback else (1.05 if is_offscreen else 0.9),
            alpha=0.96 if is_foldback else (0.72 if is_offscreen else 0.62),
            zorder=4 if is_foldback else 2,
        )
        if is_foldback:
            ax.scatter(
                [x0, x1],
                [0.035, 0.035],
                marker="*",
                s=56,
                color=FOLDBACK_HIGHLIGHT_COLOR,
                edgecolors="white",
                linewidths=0.35,
                alpha=0.98,
                zorder=5,
            )
        else:
            ax.scatter([x0, x1], [0.035, 0.035], s=9 if is_offscreen else 8, color=color, alpha=0.78 if is_offscreen else 0.75)
        if is_offscreen and str(arc.get("label", "")):
            offscreen_arcs.append(arc)

    if offscreen_arcs and len(offscreen_arcs) <= 18:
        for arc in offscreen_arcs:
            anchor_x = float(arc.get("anchor", arc["x0"])) / 1e6
            ax.annotate(
                str(arc.get("label", ""))[:14],
                (anchor_x, 0.075),
                fontsize=6,
                rotation=90,
                ha="center",
                va="bottom",
                color=_orientation_color(arc.get("orientation", "unknown")),
            )

    if markers:
        for marker in markers:
            x = float(marker["x"]) / 1e6
            color = _orientation_color(marker.get("orientation", "unknown"))
            if bool(marker.get("is_foldback", False)):
                ax.scatter([x], [0.03], marker="*", s=54, color=FOLDBACK_HIGHLIGHT_COLOR, edgecolors="white", linewidths=0.35, alpha=0.98, zorder=5)
            else:
                ax.scatter([x], [0.03], marker="v", s=18, color=color, alpha=0.78)
        if len(markers) <= 18:
            for marker in markers:
                ax.annotate(
                    str(marker.get("label", ""))[:14],
                    (float(marker["x"]) / 1e6, 0.055),
                    fontsize=6,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    color=_orientation_color(marker.get("orientation", "unknown")),
                )

    if not arcs and not markers:
        ax.text(0.5, 0.5, "No Severus SVs on chromosome", transform=ax.transAxes, ha="center", va="center", fontsize=10)

    present_orientations = sorted(
        {_strand_orientation(row) for _, row in sv_chr.iterrows()} if not sv_chr.empty else [],
        key=_orientation_sort_key,
    )
    handles = [
        Line2D([0], [0], color=_orientation_color(orientation), lw=2, label=f"STRANDS {orientation}")
        for orientation in present_orientations
    ]
    if not sv_chr.empty and any(_is_foldback_sv(row) for _, row in sv_chr.iterrows()):
        handles.append(
            Line2D(
                [0],
                [0],
                color=FOLDBACK_HIGHLIGHT_COLOR,
                marker="*",
                markersize=8,
                lw=2.4,
                label="foldback / foldback-like",
            )
        )
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=min(5, len(handles)), frameon=False)
    ax.set_ylabel("SV arcs")
    ax.set_yticks([])
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", alpha=0.12)


def plot_chromosome_prediction(
    row: pd.Series,
    wakhan_df: pd.DataFrame,
    severus_df: pd.DataFrame,
    centromeres: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    sample_id = _clean_text(row.get("sample_id", "sample"))
    chrom = _clean_text(row.get("chrom", "chrom"))
    arm = _clean_text(row.get("arm", ""))
    region = _region_label(chrom, arm)
    pred_classes = _row_predicted_classes(row)
    pred = ";".join(pred_classes) if pred_classes else _clean_text(row.get("predicted_class", "predicted"))
    top_pred = pred_classes[0] if pred_classes else pred
    candidate_start_bp = int(_numeric(row, "start_bp", 0))
    candidate_end_bp = int(_numeric(row, "end_bp", candidate_start_bp + 1))

    if candidate_end_bp <= candidate_start_bp:
        chrom_segs = wakhan_df[wakhan_df["chrom"].map(lambda value: _chrom_equal(value, chrom))].copy() if not wakhan_df.empty else pd.DataFrame()
        if not chrom_segs.empty:
            candidate_start_bp = int(pd.to_numeric(chrom_segs["start"], errors="coerce").min())
            candidate_end_bp = int(pd.to_numeric(chrom_segs["end"], errors="coerce").max())
        else:
            candidate_end_bp = candidate_start_bp + 1

    start_bp, end_bp = _plot_window_interval(row, candidate_start_bp, candidate_end_bp)
    if end_bp <= start_bp:
        start_bp, end_bp = candidate_start_bp, candidate_end_bp
    highlight_start_bp, highlight_end_bp = _highlight_interval(row, start_bp, end_bp, candidate_start_bp, candidate_end_bp)
    confidence = _clean_text(row.get("confidence", ""))
    highlight_label = _clean_text(row.get("highlight_label", ""))
    if not highlight_label and ("highlight_start_bp" in row or "highlight_end_bp" in row):
        highlight_label = f"{chrom}:{_format_bp(highlight_start_bp)}-{_format_bp(highlight_end_bp)}"
    haplotype_tag = _display_haplotype_tag(row.get("haplotype", ""))
    haplotype_conf = _numeric(row, "haplotype_confidence", np.nan)

    segs = _subset_segments(wakhan_df, chrom, start_bp, end_bp)
    sv_chr = _subset_sv_on_chrom(severus_df, chrom, start_bp, end_bp)
    cen_chr = _subset_centromeres(centromeres, chrom, start_bp, end_bp)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 2.05, 1.15], "hspace": 0.08},
    )
    highlight_color = "#2F6BFF"
    score_text = _format_score_text(row, pred_classes)
    count_text = f"{len(segs)} CN segments, {len(sv_chr)} SV records"
    true_label = _clean_text(row.get("true_classes", row.get("sv_classes", row.get("sv_class", ""))))
    split_label = _clean_text(row.get("split", ""))
    is_unlabeled_inference = split_label.lower() == "inference" and not true_label
    cluster_title = _clean_text(row.get("plot_title", "")) or _clean_text(row.get("outgroup_name", ""))
    if cluster_title:
        title_prefix = f"{cluster_title}: {sample_id} {region} | predicted {pred}"
    else:
        title_prefix = f"{sample_id} {region}: predicted {pred}"
    if not is_unlabeled_inference:
        title_prefix = f"{title_prefix} | true {true_label or 'empty'}"
    if split_label and not is_unlabeled_inference:
        title_prefix = f"{title_prefix} | split {split_label}"
    title_lines = [title_prefix]
    if highlight_label:
        if is_unlabeled_inference:
            title_lines.append(f"Localized interval: {highlight_label}")
        else:
            shatterseek_label = "ShatterSeek" + (f" {confidence}" if confidence else "")
            title_lines.append(f"{shatterseek_label}: {highlight_label}")
    if " | type_probs" in score_text:
        score_main, score_types = score_text.split(" | type_probs", 1)
        title_lines.append(f"{score_main} | {count_text}")
        title_lines.append(f"type_probs{score_types}")
    else:
        title_lines.append(f"{score_text} | {count_text}")
    title = "\n".join(title_lines)
    fig.suptitle(title, x=0.5, y=0.985, fontsize=12, color="black")
    fig.patch.set_facecolor("white")
    _draw_centromere_annotations(axes, cen_chr)
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        interval_box = Rectangle(
            (highlight_start_bp / 1e6, 0),
            (highlight_end_bp - highlight_start_bp) / 1e6,
            1,
            transform=ax.get_xaxis_transform(),
            fill=False,
            edgecolor=highlight_color,
            linewidth=1.5,
            alpha=0.95,
            zorder=20,
            clip_on=False,
        )
        ax.add_patch(interval_box)

    _plot_sv_panel(axes[0], severus_df, sv_chr, chrom, start_bp, end_bp)
    _plot_cn_panel(axes[1], segs, start_bp, end_bp)
    _plot_breakpoint_panel(axes[2], segs, severus_df, sv_chr, chrom, start_bp, end_bp)

    if haplotype_tag:
        hap_color = HAPLOTYPE_COLORS.get(haplotype_tag, "#737373")
        hap_text = haplotype_tag
        if np.isfinite(haplotype_conf):
            hap_text = f"{haplotype_tag} ({haplotype_conf:.2f})"
        fig.text(
            0.985,
            0.985,
            hap_text,
            ha="right",
            va="top",
            fontsize=9,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="round,pad=0.32", facecolor=hap_color, edgecolor="none", alpha=0.92),
            zorder=25,
        )

    axes[-1].set_xlim(start_bp / 1e6, end_bp / 1e6)
    axes[-1].set_xlabel(f"{chrom} position (Mb)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.92, bottom=0.08, hspace=0.10)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    manifest = read_manifest(args.manifest)
    distances = pd.read_csv(args.prototype_distances, sep="\t").fillna("")
    centromeres = read_centromeres(args.centromeres)
    if not centromeres.empty:
        log.info("Loaded %d centromere interval(s) from %s", len(centromeres), args.centromeres)
    selected = select_predicted_chromosomes(distances, plot_scope=args.plot_scope)
    if args.max_plots is not None:
        selected = selected.head(int(args.max_plots)).copy()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.prototype_distances).resolve().parent / "predicted_chromosome_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    if selected.empty:
        log.info("No unlabeled chromosome-scan predictions with non-none class were found.")
        pd.DataFrame().to_csv(output_dir / "selected_predictions.tsv", sep="\t", index=False)
        return

    manifest_by_sample = {str(row["_sample_key"]): row for _, row in manifest.iterrows()}
    data_cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    log.info("Plotting %d predicted chromosome(s) into %s", len(selected), output_dir)

    for i, row in selected.iterrows():
        sample_id = canonical_sample_id(str(row["sample_id"]))
        if sample_id not in manifest_by_sample:
            out_row = row.to_dict()
            if "haplotype" in out_row:
                out_row["haplotype"] = _display_haplotype_tag(out_row.get("haplotype", ""))
            out_row["plot_correctness"] = _plot_correctness_bucket(row)
            out_row["plot_status"] = "missing_manifest_sample"
            out_row["plot_path"] = ""
            summary_rows.append(out_row)
            log.warning("Skipping %s: sample not found in manifest", sample_id)
            continue

        if sample_id not in data_cache:
            manifest_row = manifest_by_sample[sample_id]
            log.info("Loading WAKHAN/Severus for %s", sample_id)
            wakhan_df = parse_wakhan(manifest_row["wakhan_root"])
            severus_vcf = _clean_text(manifest_row.get("severus_vcf", ""))
            if severus_vcf and Path(severus_vcf).exists():
                severus_df = parse_severus(severus_vcf, sample_id=sample_id)
            else:
                log.warning("No Severus VCF found for %s: %s", sample_id, severus_vcf)
                severus_df = pd.DataFrame()
            data_cache[sample_id] = (wakhan_df, severus_df)

        pred = _prediction_label(row)
        correctness_bucket = _plot_correctness_bucket(row)
        group_value = _clean_text(row.get(args.group_by_column, "")) if args.group_by_column else ""
        if group_value:
            class_dir = output_dir / _safe_name(group_value)
        elif args.no_correctness_dirs:
            class_dir = output_dir / _safe_name(pred)
        else:
            class_dir = output_dir / correctness_bucket / _safe_name(pred) if correctness_bucket else output_dir / _safe_name(pred)
        chrom = _clean_text(row["chrom"])
        arm = _clean_text(row.get("arm", ""))
        region = _region_label(chrom, arm)
        objectness_prob = _numeric(row, "objectness_prob", np.nan)
        distance = _numeric(row, "nearest_prototype_distance", np.nan)
        suffix = f"obj{objectness_prob:.4g}" if np.isfinite(objectness_prob) else (f"d{distance:.4g}" if np.isfinite(distance) else "scoreNA")
        highlight_start = _numeric(row, "highlight_start_bp", np.nan)
        highlight_end = _numeric(row, "highlight_end_bp", np.nan)
        name_start = highlight_start if np.isfinite(highlight_start) else _numeric(row, "start_bp", np.nan)
        name_end = highlight_end if np.isfinite(highlight_end) else _numeric(row, "end_bp", np.nan)
        highlight_tag = ""
        if np.isfinite(name_start) and np.isfinite(name_end):
            highlight_tag = f"_{int(round(name_start))}_{int(round(name_end))}"
        plot_name = f"{_safe_name(sample_id)}_{_safe_name(region)}_{_safe_name(pred)}_{suffix}{highlight_tag}.png"
        plot_path = class_dir / plot_name

        out_row = row.to_dict()
        if "haplotype" in out_row:
            out_row["haplotype"] = _display_haplotype_tag(out_row.get("haplotype", ""))
        candidate_start_bp = int(_numeric(row, "start_bp", 0))
        candidate_end_bp = int(_numeric(row, "end_bp", candidate_start_bp + 1))
        plot_start_bp, plot_end_bp = _plot_window_interval(row, candidate_start_bp, candidate_end_bp)
        highlight_start_bp, highlight_end_bp = _highlight_interval(row, plot_start_bp, plot_end_bp, candidate_start_bp, candidate_end_bp)
        out_row["plot_start_bp"] = plot_start_bp
        out_row["plot_end_bp"] = plot_end_bp
        out_row["highlight_start_bp"] = highlight_start_bp
        out_row["highlight_end_bp"] = highlight_end_bp
        plot_cens = _subset_centromeres(centromeres, chrom, plot_start_bp, plot_end_bp)
        out_row["centromere_intervals"] = _format_centromere_intervals(plot_cens)
        out_row["plot_correctness"] = correctness_bucket
        try:
            wakhan_df, severus_df = data_cache[sample_id]
            plot_chromosome_prediction(row, wakhan_df, severus_df, centromeres, plot_path, dpi=args.dpi)
            out_row["plot_status"] = "ok"
            out_row["plot_path"] = str(plot_path)
            log.info("Wrote %s (%d/%d)", plot_path, i + 1, len(selected))
        except Exception as exc:  # pragma: no cover - keeps batch plotting moving
            out_row["plot_status"] = f"error: {exc}"
            out_row["plot_path"] = str(plot_path)
            log.exception("Failed plotting %s %s", sample_id, region)
        summary_rows.append(out_row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "selected_predictions.tsv", sep="\t", index=False)
    log.info("Wrote prediction plot index: %s", output_dir / "selected_predictions.tsv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Complex-SV manifest with wakhan_root and severus_vcf columns.")
    parser.add_argument("--predictions", "--prototype_distances", dest="prototype_distances", required=True, help="Localized prediction rows prepared for plotting.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to <prototype_distances parent>/predicted_chromosome_plots.")
    parser.add_argument("--max_plots", type=int, default=None, help="Optional cap for quick previews/tests.")
    parser.add_argument(
        "--plot_scope",
        choices=("unlabeled-called", "called", "all", "test"),
        default="unlabeled-called",
        help="Rows to plot from the prediction table. Use all for held-out candidate-region tables.",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--centromeres", default=str(DEFAULT_CENTROMERE_BED), help="BED file with centromere intervals to annotate; set empty to disable.")
    parser.add_argument("--group_by_column", default="", help="Optional prediction-table column used as the output subdirectory instead of correctness/predicted-class buckets.")
    parser.add_argument("--no_correctness_dirs", action="store_true", help="Do not create correct_preds/incorrect_preds output directories when group_by_column is not set.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
