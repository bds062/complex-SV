"""Plot CN and SV patterns for unlabeled chromosomes predicted as complex SVs."""

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

from data.anchor_manifest import canonical_sample_id
from data.severus_parser import parse_severus
from data.wakhan_parser import parse_wakhan

log = logging.getLogger(__name__)

MISSING_VALUES = {"", "none", "nan", "null", "unknown", "unlabeled"}
CLASS_COLORS = {
    "BFB": "#E15759",
    "chromothripsis": "#4E79A7",
    "seismic_amplification": "#F28E2B",
}
SV_TYPE_COLORS = {
    "DEL": "#CF0759",
    "INV": "#2830DE",
    "INS": "#D4B000",
    "BND": "#737373",
    "sBND": "#9C755F",
    "DUP": "#178117",
}


def _clean_text(value: object) -> str:
    text = str(value).strip()
    if text.lower() in {"nan", "null"}:
        return ""
    return text


def _is_empty(value: object) -> bool:
    return _clean_text(value).lower() in MISSING_VALUES


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


def read_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t").fillna("")
    required = {"sample_id", "wakhan_root", "severus_vcf"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    df["_sample_key"] = df["sample_id"].map(lambda value: canonical_sample_id(str(value)))
    return df


def select_predicted_chromosomes(distances: pd.DataFrame) -> pd.DataFrame:
    if distances.empty:
        return pd.DataFrame()
    required = {"sample_id", "chrom", "predicted_class", "sv_class"}
    missing = sorted(required.difference(distances.columns))
    if missing:
        raise ValueError(f"Prototype distance table missing columns: {missing}")

    df = distances.copy().fillna("")
    if "evidence" in df.columns:
        df = df[df["evidence"].astype(str) == "chromosome_scan"].copy()
    df = df[df["sv_class"].map(_is_empty)].copy()
    df = df[~df["predicted_class"].map(_is_empty)].copy()
    if df.empty:
        return df

    for col in ["start_bp", "end_bp", "nearest_prototype_distance", "prototype_confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_cols = [col for col in ["predicted_class", "sample_id", "chrom", "nearest_prototype_distance"] if col in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


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


def _plot_breakpoint_panel(ax, segs: pd.DataFrame, sv_chr: pd.DataFrame, start_bp: int, end_bp: int) -> None:
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
        for sv_type, grp in sv_chr.groupby(sv_chr["sv_type_str"].astype(str)):
            color = SV_TYPE_COLORS.get(sv_type, "#6C6C6C")
            xs = pd.to_numeric(grp["pos"], errors="coerce").dropna().to_numpy(float) / 1e6
            xs = xs[(xs >= start_bp / 1e6) & (xs <= end_bp / 1e6)]
            if xs.size:
                ax.vlines(xs, 0, y_top, colors=color, linewidth=0.55, alpha=0.45)
        max_bp = max(max_bp, y_top)

    ax.set_ylim(0, max_bp * 1.25 + 0.05)
    ax.grid(axis="y", alpha=0.2)


def _arc_points(x0: float, x1: float, region_mb: float) -> tuple[np.ndarray, np.ndarray]:
    if x1 < x0:
        x0, x1 = x1, x0
    xs = np.linspace(x0, x1, 80)
    span = max(x1 - x0, 1e-4)
    height = 0.12 + 0.86 * min(1.0, np.sqrt(span / max(region_mb, 1e-4)))
    ys = 0.05 + height * np.sin(np.pi * (xs - x0) / span)
    return xs, ys


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
            if _chrom_equal(mate_chrom, chrom) and start_bp <= mate_pos <= end_bp:
                arcs.append({"x0": pos, "x1": mate_pos, "type": sv_type, "kind": "mate"})
            else:
                markers.append({"x": pos, "type": sv_type, "label": f"to {mate_chrom}" if mate_chrom else "BND"})
            continue

        if sv_type in {"DEL", "DUP", "INV", "BND"} and end > pos + 1 and start_bp <= end <= end_bp:
            arcs.append({"x0": pos, "x1": end, "type": sv_type, "kind": "interval"})
        else:
            markers.append({"x": pos, "type": sv_type, "label": sv_type})
    return arcs, markers


def _plot_sv_panel(ax, severus_df: pd.DataFrame, sv_chr: pd.DataFrame, chrom: str, start_bp: int, end_bp: int) -> None:
    region_mb = max((end_bp - start_bp) / 1e6, 1e-4)
    arcs, markers = _same_chrom_arcs(severus_df, sv_chr, chrom, start_bp, end_bp)

    for arc in arcs:
        x0 = float(arc["x0"]) / 1e6
        x1 = float(arc["x1"]) / 1e6
        color = SV_TYPE_COLORS.get(str(arc["type"]), "#6C6C6C")
        xs, ys = _arc_points(x0, x1, region_mb)
        ax.plot(xs, ys, color=color, linewidth=0.9, alpha=0.62)
        ax.scatter([x0, x1], [0.035, 0.035], s=8, color=color, alpha=0.75)

    if markers:
        by_type: dict[str, list[dict[str, Any]]] = {}
        for marker in markers:
            by_type.setdefault(str(marker["type"]), []).append(marker)
        for sv_type, group in by_type.items():
            color = SV_TYPE_COLORS.get(sv_type, "#6C6C6C")
            xs = [float(marker["x"]) / 1e6 for marker in group]
            ax.scatter(xs, [0.03] * len(xs), marker="v", s=18, color=color, alpha=0.78)
        if len(markers) <= 18:
            for marker in markers:
                ax.annotate(str(marker.get("label", ""))[:14], (float(marker["x"]) / 1e6, 0.055), fontsize=6, rotation=90, ha="center", va="bottom")

    if not arcs and not markers:
        ax.text(0.5, 0.5, "No Severus SVs on chromosome", transform=ax.transAxes, ha="center", va="center", fontsize=10)

    present_types = sorted(set(sv_chr["sv_type_str"].astype(str))) if not sv_chr.empty and "sv_type_str" in sv_chr else []
    handles = [
        Line2D([0], [0], color=SV_TYPE_COLORS.get(sv_type, "#6C6C6C"), lw=2, label=sv_type)
        for sv_type in present_types
    ]
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=8, ncol=min(6, len(handles)), frameon=False)
    ax.set_ylabel("SV arcs")
    ax.set_yticks([])
    ax.set_ylim(0, 1.12)
    ax.grid(axis="x", alpha=0.12)


def plot_chromosome_prediction(
    row: pd.Series,
    wakhan_df: pd.DataFrame,
    severus_df: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    sample_id = _clean_text(row.get("sample_id", "sample"))
    chrom = _clean_text(row.get("chrom", "chrom"))
    pred = _clean_text(row.get("predicted_class", "predicted"))
    start_bp = int(_numeric(row, "start_bp", 0))
    end_bp = int(_numeric(row, "end_bp", start_bp + 1))

    if end_bp <= start_bp:
        chrom_segs = wakhan_df[wakhan_df["chrom"].map(lambda value: _chrom_equal(value, chrom))].copy() if not wakhan_df.empty else pd.DataFrame()
        if not chrom_segs.empty:
            start_bp = int(pd.to_numeric(chrom_segs["start"], errors="coerce").min())
            end_bp = int(pd.to_numeric(chrom_segs["end"], errors="coerce").max())
        else:
            end_bp = start_bp + 1

    segs = _subset_segments(wakhan_df, chrom, start_bp, end_bp)
    sv_chr = _subset_sv_on_chrom(severus_df, chrom, start_bp, end_bp)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.55, 2.05, 1.15], "hspace": 0.08},
    )
    class_color = CLASS_COLORS.get(pred, "#4E79A7")
    distance = _numeric(row, "nearest_prototype_distance", np.nan)
    confidence = _numeric(row, "prototype_confidence", np.nan)
    title = (
        f"{sample_id} {chrom}: predicted {pred}"
        f" | distance={distance:.4g} confidence={confidence:.3g}"
        f" | {len(segs)} CN segments, {len(sv_chr)} SV records"
    )
    fig.suptitle(title, x=0.5, y=0.985, fontsize=12, color="black")
    fig.patch.set_facecolor("white")
    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axvspan(start_bp / 1e6, end_bp / 1e6, color=class_color, alpha=0.025, linewidth=0)

    _plot_sv_panel(axes[0], severus_df, sv_chr, chrom, start_bp, end_bp)
    _plot_cn_panel(axes[1], segs, start_bp, end_bp)
    _plot_breakpoint_panel(axes[2], segs, sv_chr, start_bp, end_bp)

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
    selected = select_predicted_chromosomes(distances)
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

        pred = _clean_text(row["predicted_class"])
        class_dir = output_dir / _safe_name(pred)
        chrom = _clean_text(row["chrom"])
        distance = _numeric(row, "nearest_prototype_distance", np.nan)
        suffix = f"d{distance:.4g}" if np.isfinite(distance) else "dNA"
        plot_name = f"{_safe_name(sample_id)}_{_safe_name(chrom)}_{_safe_name(pred)}_{suffix}.png"
        plot_path = class_dir / plot_name

        out_row = row.to_dict()
        try:
            wakhan_df, severus_df = data_cache[sample_id]
            plot_chromosome_prediction(row, wakhan_df, severus_df, plot_path, dpi=args.dpi)
            out_row["plot_status"] = "ok"
            out_row["plot_path"] = str(plot_path)
            log.info("Wrote %s (%d/%d)", plot_path, i + 1, len(selected))
        except Exception as exc:  # pragma: no cover - keeps batch plotting moving
            out_row["plot_status"] = f"error: {exc}"
            out_row["plot_path"] = str(plot_path)
            log.exception("Failed plotting %s %s", sample_id, chrom)
        summary_rows.append(out_row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "selected_predictions.tsv", sep="\t", index=False)
    log.info("Wrote prediction plot index: %s", output_dir / "selected_predictions.tsv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Complex-SV manifest with wakhan_root and severus_vcf columns.")
    parser.add_argument("--prototype_distances", required=True, help="prototype_distances.tsv from chromosome-scale inference.")
    parser.add_argument("--output_dir", default=None, help="Output directory. Defaults to <prototype_distances parent>/predicted_chromosome_plots.")
    parser.add_argument("--max_plots", type=int, default=None, help="Optional cap for quick previews/tests.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
