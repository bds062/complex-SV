#!/usr/bin/env python3
"""Run candidate-region calling for every sample in a complex-SV manifest."""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = REPO_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from process_vcfs import (  # noqa: E402
    generate_candidate_plot,
    get_bps,
    merge_sv_cna_candidate_segments,
    read_cna_vcf_to_dataframe,
)


REQUIRED_MANIFEST_COLUMNS = {"sample_id", "wakhan_root", "severus_vcf"}
EVENT_CLASS_COLUMNS = ["ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"]
LABEL_COLUMNS = [
    "label_id",
    "sample_id",
    "wakhan_sample_id",
    "wakhan_root",
    "severus_vcf",
    "chrom",
    "arm",
    "start_bp",
    "end_bp",
    "sv_class",
    "validated",
    "label_scope",
    "source",
    "notes",
]
EVENT_CLASS_LABELS = {
    "ecDNA": "ecDNA",
    "Seismic_Amplification": "Seismic Amplification",
    "chromothripsis": "Chromothripsis",
    "BFB": "BFB",
}
SUBTYPE_LABELS = {
    "canonical": "Canonical",
    "noncanonical": "Noncanonical",
    "noncanonicalb": "NoncanonicalB",
}
SUBTYPE_ORDER = ["canonical", "noncanonical", "noncanonicalb"]
DEFAULT_CENTROMERES = Path(
    "/data/KolmogorovLab/srinivasanbd/results/grch38.cen_coord.curated.bed"
)
EMPTY_BREAKPOINT_COLUMNS = [
    "chrom",
    "pos",
    "hp",
    "supp",
    "st",
    "sv_id",
    "SV_TYPE",
    "ref_reads",
    "supp_reads",
]


def _require_plotly() -> None:
    if importlib.util.find_spec("plotly") is None:
        raise ModuleNotFoundError(
            "Plotly is required to write candidate HTML plots. Install plotly or rerun with --skip-plots."
        )


def _clean_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: object) -> bool:
    text = _clean_text(value).lower()
    if text in {"", "0", "false", "f", "no", "n", "none", "nan"}:
        return False
    return True


def _has_class_call(value: object) -> bool:
    text = _clean_text(value).lower()
    return text not in {"", "0", "false", "f", "no", "n", "none", "nan", "<na>"}


def _subtype_counts(series: pd.Series) -> list[tuple[str, int]]:
    cleaned = series.map(_clean_text)
    counts_by_key = cleaned.map(str.lower).value_counts(sort=False).to_dict()
    rows = []
    for key in SUBTYPE_ORDER:
        count = int(counts_by_key.pop(key, 0))
        if count:
            rows.append((SUBTYPE_LABELS[key], count))
    for key, count in counts_by_key.items():
        label = SUBTYPE_LABELS.get(key, key[:1].upper() + key[1:])
        rows.append((label, int(count)))
    return rows


def _print_class_summary(candidate_df: pd.DataFrame) -> None:
    class_masks = {
        column: (
            candidate_df[column].map(_has_class_call)
            if column in candidate_df.columns
            else pd.Series(False, index=candidate_df.index)
        )
        for column in EVENT_CLASS_COLUMNS
    }
    mask_df = pd.DataFrame(class_masks, index=candidate_df.index)
    empty_candidates = int((~mask_df.any(axis=1)).sum())

    print("Class summary:")
    for column in EVENT_CLASS_COLUMNS:
        print(f"  {EVENT_CLASS_LABELS[column]}: {int(mask_df[column].sum())}")
        if column not in candidate_df.columns:
            continue
        subtype_series = candidate_df.loc[mask_df[column], column]
        subtype_series = subtype_series[
            ~subtype_series.map(lambda value: _clean_text(value).lower()).isin(
                {"true", "t", "1", "yes", "y"}
            )
        ]
        for subtype, count in _subtype_counts(subtype_series):
            print(f"    {subtype}: {count}")
    print(f"  empty_candidate_regions: {empty_candidates}")


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("._") or "sample"


def _safe_label_id(*parts: object) -> str:
    text = "_".join(_clean_text(part) for part in parts if _clean_text(part))
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return cleaned.strip("._") or "label"


def _candidate_sv_class(
    event_column: str,
    value: object,
    base_labels: bool = False,
) -> str | None:
    if not _has_class_call(value):
        return None

    if base_labels:
        return event_column

    if event_column == "ecDNA":
        return "ecDNA"

    subtype = _clean_text(value)
    return f"{event_column}_{subtype}" if subtype else event_column


def candidate_calls_to_labels(
    candidate_df: pd.DataFrame,
    base_labels: bool = False,
) -> pd.DataFrame:
    rows = []
    for _, candidate in candidate_df.iterrows():
        sample_id = _clean_text(candidate.get("sample_id", ""))
        candidate_id = _clean_text(candidate.get("candidate_id", ""))
        chrom = _clean_text(candidate.get("chrom", ""))
        arm = _clean_text(candidate.get("arm", ""))
        start_bp = int(candidate.get("start", candidate.get("start_bp", 0)))
        end_bp = int(candidate.get("end", candidate.get("end_bp", 0)))

        for event_column in EVENT_CLASS_COLUMNS:
            if event_column not in candidate.index:
                continue
            raw_value = candidate[event_column]
            sv_class = _candidate_sv_class(event_column, raw_value, base_labels=base_labels)
            if sv_class is None:
                continue

            rows.append(
                {
                    "label_id": _safe_label_id(candidate_id, sv_class),
                    "sample_id": sample_id,
                    "wakhan_sample_id": _clean_text(candidate.get("wakhan_sample_id", "")),
                    "wakhan_root": _clean_text(candidate.get("wakhan_root", "")),
                    "severus_vcf": _clean_text(candidate.get("severus_vcf", "")),
                    "chrom": chrom,
                    "arm": arm,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "sv_class": sv_class,
                    "validated": False,
                    "label_scope": "region",
                    "source": "gen_candidates",
                    "notes": (
                        f"candidate_id={candidate_id};"
                        f"event={event_column};"
                        f"subtype={_clean_text(raw_value)}"
                    ),
                }
            )

    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def read_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path, sep="\t").fillna("")
    missing = sorted(REQUIRED_MANIFEST_COLUMNS.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return manifest


def _strip_wakhan_bed_suffix(path: Path) -> Path:
    name = path.name
    name = re.sub(
        r"_copynumbers_segments_HP[_-]?[12]\.bed$",
        "",
        name,
        flags=re.IGNORECASE,
    )
    name = re.sub(r"_copynumbers_segments$", "", name, flags=re.IGNORECASE)
    return path.with_name(name)


def _replace_path_part(path: Path, old: str, new: str) -> Path:
    parts = list(path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == old:
            parts[index] = new
            return Path(*parts)
    raise ValueError(f"Expected {old!r} in Wakhan root path: {path}")


def resolve_cna_vcf(wakhan_root: str | Path) -> Path:
    """Resolve a manifest Wakhan BED root to the matching CNA integer VCF."""
    bed_root = _strip_wakhan_bed_suffix(Path(wakhan_root))
    vcf_root = _replace_path_part(bed_root, "bed_output", "vcf_output")
    return vcf_root.with_name(f"{vcf_root.name}_wakhan_cna_integers.vcf")


def _empty_breakpoints() -> pd.DataFrame:
    return pd.DataFrame(columns=EMPTY_BREAKPOINT_COLUMNS)


def _add_sample_metadata(
    df: pd.DataFrame,
    row: pd.Series,
    sample_id: str,
    cna_vcf: Path,
    severus_vcf: Path | None,
) -> pd.DataFrame:
    result = df.copy()
    metadata = {
        "sample_id": sample_id,
        "wakhan_sample_id": _clean_text(row.get("wakhan_sample_id", "")),
        "wakhan_root": _clean_text(row.get("wakhan_root", "")),
        "cna_vcf": str(cna_vcf),
        "severus_vcf": str(severus_vcf) if severus_vcf is not None else "",
    }
    for index, (column, value) in enumerate(metadata.items()):
        if column in result.columns:
            result[column] = value
        else:
            result.insert(index, column, value)

    if not result.empty and "candidate_id" not in result.columns:
        candidate_ids = [
            f"{sample_id}:{record.chrom}:{int(record.start)}-{int(record.end)}:{idx + 1}"
            for idx, record in enumerate(result.itertuples(index=False))
        ]
        result.insert(1, "candidate_id", candidate_ids)
    return result


def _write_empty_sample_outputs(
    sample_dir: Path,
    sample_id: str,
    row: pd.Series,
    cna_vcf: Path,
    severus_vcf: Path | None,
) -> pd.DataFrame:
    sample_dir.mkdir(parents=True, exist_ok=True)
    empty = _add_sample_metadata(pd.DataFrame(), row, sample_id, cna_vcf, severus_vcf)
    empty.to_csv(sample_dir / f"{sample_id}_candidate_regions.csv", index=False)
    return empty


def process_sample(
    row: pd.Series,
    args: argparse.Namespace,
    sample_index: int,
    n_samples: int,
) -> pd.DataFrame:
    sample_id = _safe_name(_clean_text(row.get("sample_id", "")) or f"sample_{sample_index}")
    sample_dir = Path(args.out_dir) / sample_id
    cna_vcf = resolve_cna_vcf(row["wakhan_root"])
    severus_text = _clean_text(row.get("severus_vcf", ""))
    has_severus = _truthy(row.get("has_severus", True)) and bool(severus_text)
    severus_vcf = Path(severus_text) if has_severus else None

    print(f"[{sample_index}/{n_samples}] {sample_id}")
    print(f"  CNA VCF: {cna_vcf}")
    if severus_vcf is not None:
        print(f"  SV VCF:  {severus_vcf}")
    else:
        print("  SV VCF:  none; using CNA-only candidate summaries")

    if not cna_vcf.exists():
        raise FileNotFoundError(f"Missing CNA VCF for {sample_id}: {cna_vcf}")
    if severus_vcf is not None and not severus_vcf.exists():
        raise FileNotFoundError(f"Missing Severus VCF for {sample_id}: {severus_vcf}")

    if args.dry_run:
        return pd.DataFrame()

    sample_dir.mkdir(parents=True, exist_ok=True)
    cna_df = read_cna_vcf_to_dataframe(cna_vcf)
    breakpoints_df = get_bps(severus_vcf) if severus_vcf is not None else _empty_breakpoints()
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
    final_df = _add_sample_metadata(final_df, row, sample_id, cna_vcf, severus_vcf)

    csv_path = sample_dir / f"{sample_id}_candidate_regions.csv"
    html_path = sample_dir / f"{sample_id}_candidate_regions.html"
    final_df.to_csv(csv_path, index=False)
    if not args.skip_plots:
        generate_candidate_plot(
            final_df,
            cna_df,
            breakpoints_df,
            html_path,
            max_regions=args.plot_max_regions,
        )

    print(f"  Wrote {len(final_df)} rows to {csv_path}")
    if not args.skip_plots:
        print(f"  Wrote plot to {html_path}")
    return final_df


def run(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.skip_plots:
        _require_plotly()

    manifest = read_manifest(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)

    out_dir = Path(args.out_dir)
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    sample_frames = []
    failures = []
    rows = list(manifest.iterrows())
    for sample_index, (_, row) in enumerate(rows, start=1):
        try:
            sample_df = process_sample(row, args, sample_index, len(rows))
        except Exception as exc:
            sample_id = _safe_name(_clean_text(row.get("sample_id", "")) or f"sample_{sample_index}")
            if not args.keep_going:
                raise
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"  ERROR: {exc}")
            continue
        if not args.dry_run:
            sample_frames.append(sample_df)

    if args.dry_run:
        print(f"Dry run complete: checked {len(rows)} manifest row(s).")
        return

    merged = pd.concat(sample_frames, ignore_index=True, sort=False) if sample_frames else pd.DataFrame()
    merged_path = out_dir / args.merged_name
    merged.to_csv(merged_path, index=False)
    print(f"Wrote merged CSV with {len(merged)} rows to {merged_path}")

    labels = candidate_calls_to_labels(merged, base_labels=args.base_labels)
    labels_path = out_dir / args.labels_name
    labels.to_csv(labels_path, sep="\t", index=False)
    print(f"Wrote labels TSV with {len(labels)} rows to {labels_path}")

    if failures:
        failures_path = out_dir / "failed_samples.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        print(f"Wrote {len(failures)} failure(s) to {failures_path}")

    _print_class_summary(merged)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Complex-SV manifest TSV")
    parser.add_argument("--out_dir", "--output_dir", dest="out_dir", type=Path, required=True)
    parser.add_argument("--merged-name", default="merged_candidate_regions.csv")
    parser.add_argument("--labels-name", default="complex_sv_labels.tsv")
    parser.add_argument(
        "--base_labels",
        "--base-labels",
        action="store_true",
        help="Write base event classes in the labels TSV instead of subtype-specific labels.",
    )
    parser.add_argument("--centromeres", type=Path, default=DEFAULT_CENTROMERES)
    parser.add_argument("--window-size", type=int, default=50_000_000)
    parser.add_argument("--step-size", type=int, default=10_000_000)
    parser.add_argument(
        "--method",
        choices=["sliding", "small_segments"],
        default="sliding",
        help="Candidate interval method",
    )
    parser.add_argument("--max-segment-size", type=int, default=5_000_000)
    parser.add_argument("--region-merge-distance", type=int, default=10_000_000)
    parser.add_argument("--frequency-threshold", type=float, default=1 / 2_000_000)
    parser.add_argument("--plot-max-regions", type=int, default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N manifest rows")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and check input paths only")
    parser.add_argument("--keep-going", action="store_true", help="Continue after per-sample failures")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
