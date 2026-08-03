"""
Build multimodal input manifests from weak chromosome-level label highlights.

The current chromothripsis anchors are stored in the same lightweight format
used by the CN visualizer:

    label<TAB or spaces>wakhan_root<TAB or spaces>chrom

This module resolves those anchors into two TSVs:

* a sample-level manifest with one row per sample and optional Severus VCF path
* a label-level table with chromosome-span coordinates resolved from Wakhan

Coordinates are 0-based half-open intervals.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from .severus_parser import infer_sample_id_from_vcf
    from .wakhan_parser import parse_wakhan, resolve_wakhan_bed_root
except ImportError:  # Support direct execution as python genomic_features/anchor_manifest.py.
    from severus_parser import infer_sample_id_from_vcf  # type: ignore
    from wakhan_parser import parse_wakhan, resolve_wakhan_bed_root  # type: ignore

log = logging.getLogger(__name__)

NUMERIC_WAKHAN_SUFFIX_RE = re.compile(r"^(?P<sample>.+?)(?:_\d+(?:\.\d+)?){3}$")

MANIFEST_COLUMNS = [
    "sample_id",
    "wakhan_sample_id",
    "wakhan_root",
    "severus_vcf",
    "has_severus",
    "n_labels",
    "label_ids",
    "cohort",
    "split",
    "notes",
]

LABEL_COLUMNS = [
    "label_id",
    "sample_id",
    "wakhan_sample_id",
    "wakhan_root",
    "severus_vcf",
    "chrom",
    "start_bp",
    "end_bp",
    "sv_class",
    "validated",
    "label_scope",
    "source",
    "notes",
]


@dataclass(frozen=True)
class HighlightAnchor:
    """One row from a visualizer-style highlight list."""

    label_id: str
    wakhan_root: str
    chrom: str
    line_no: int


def canonical_sample_id(sample_id: str) -> str:
    """
    Normalize IDs enough to match Wakhan roots to Severus VCF paths.

    Wakhan bed roots commonly include suffixes such as _3.06_0.94_0.79,
    while Severus paths usually carry only the cell line/sample name.
    """
    text = str(sample_id).strip()
    if text.startswith("wakhan_") and len(text) > len("wakhan_"):
        text = text.replace("wakhan_", "", 1)
    match = NUMERIC_WAKHAN_SUFFIX_RE.match(text)
    if match:
        return match.group("sample")
    return text


def read_highlight_anchors(path: str | Path) -> list[HighlightAnchor]:
    """Read label/wakhan_root/chrom rows, accepting arbitrary whitespace."""
    path = Path(path)
    anchors: list[HighlightAnchor] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"Invalid highlight row {path}:{line_no}; expected "
                    "label wakhan_root chrom"
                )
            anchors.append(
                HighlightAnchor(
                    label_id=parts[0],
                    wakhan_root=parts[1],
                    chrom=parts[2],
                    line_no=line_no,
                )
            )
    if not anchors:
        raise ValueError(f"No highlight anchors found in {path}")
    return anchors


def read_severus_vcfs(path: str | Path) -> dict[str, str]:
    """Return canonical_sample_id -> VCF path from a one-path-per-line file."""
    path = Path(path)
    vcf_by_sample: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw_line in enumerate(fh, start=1):
            text = raw_line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            vcf_path = fields[0]
            sample_id = canonical_sample_id(infer_sample_id_from_vcf(vcf_path))
            if sample_id in vcf_by_sample and vcf_by_sample[sample_id] != vcf_path:
                log.warning(
                    "Duplicate Severus sample %s at %s:%d; keeping first path %s",
                    sample_id,
                    path,
                    line_no,
                    vcf_by_sample[sample_id],
                )
                continue
            vcf_by_sample[sample_id] = vcf_path
    if not vcf_by_sample:
        raise ValueError(f"No Severus VCF paths found in {path}")
    return vcf_by_sample


def _chrom_subset(df: pd.DataFrame, chrom: str) -> pd.DataFrame:
    wanted = str(chrom)
    wanted_no_chr = wanted.removeprefix("chr")
    observed = df["chrom"].astype(str)
    mask = (observed == wanted) | (observed.str.removeprefix("chr") == wanted_no_chr)
    return df.loc[mask].copy()


def _resolve_chrom_span(df: pd.DataFrame, chrom: str, wakhan_root: str) -> tuple[int, int]:
    subset = _chrom_subset(df, chrom)
    if subset.empty:
        observed = ", ".join(sorted(df["chrom"].astype(str).unique())[:20])
        raise ValueError(
            f"No Wakhan segments for chromosome {chrom!r} in {wakhan_root}. "
            f"Observed chromosomes include: {observed}"
        )
    starts = pd.to_numeric(subset["start"], errors="coerce")
    ends = pd.to_numeric(subset["end"], errors="coerce")
    start_bp = int(starts.min())
    end_bp = int(ends.max())
    if end_bp <= start_bp:
        raise ValueError(f"Invalid chromosome span for {wakhan_root} {chrom}: {start_bp}-{end_bp}")
    return start_bp, end_bp


def build_anchor_tables(
    anchors: Iterable[HighlightAnchor],
    severus_by_sample: dict[str, str],
    sv_class: str = "chromothripsis",
    source: str = "highlight_list",
    label_scope: str = "chromosome",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build manifest and label DataFrames from anchors plus Severus paths."""
    labels: list[dict[str, object]] = []
    root_cache: dict[str, tuple[str, pd.DataFrame]] = {}

    for anchor in anchors:
        if anchor.wakhan_root not in root_cache:
            wakhan_sample_id, _hp1, _hp2 = resolve_wakhan_bed_root(anchor.wakhan_root)
            root_cache[anchor.wakhan_root] = (wakhan_sample_id, parse_wakhan(anchor.wakhan_root))

        wakhan_sample_id, df_wakhan = root_cache[anchor.wakhan_root]
        sample_id = canonical_sample_id(wakhan_sample_id)
        severus_vcf = severus_by_sample.get(sample_id, "")
        start_bp, end_bp = _resolve_chrom_span(df_wakhan, anchor.chrom, anchor.wakhan_root)

        labels.append(
            {
                "label_id": anchor.label_id,
                "sample_id": sample_id,
                "wakhan_sample_id": wakhan_sample_id,
                "wakhan_root": anchor.wakhan_root,
                "severus_vcf": severus_vcf,
                "chrom": anchor.chrom,
                "start_bp": start_bp,
                "end_bp": end_bp,
                "sv_class": sv_class,
                "validated": True,
                "label_scope": label_scope,
                "source": source,
                "notes": "" if severus_vcf else "missing_severus_vcf",
            }
        )

    labels_df = pd.DataFrame(labels, columns=LABEL_COLUMNS)

    manifest_rows: list[dict[str, object]] = []
    for (sample_id, wakhan_root), grp in labels_df.groupby(["sample_id", "wakhan_root"], sort=True):
        wakhan_sample_id = str(grp["wakhan_sample_id"].iloc[0])
        severus_vcf = str(grp["severus_vcf"].iloc[0])
        manifest_rows.append(
            {
                "sample_id": str(sample_id),
                "wakhan_sample_id": wakhan_sample_id,
                "wakhan_root": str(wakhan_root),
                "severus_vcf": severus_vcf,
                "has_severus": bool(severus_vcf),
                "n_labels": int(len(grp)),
                "label_ids": ",".join(grp["label_id"].astype(str).tolist()),
                "cohort": "",
                "split": "anchor",
                "notes": "" if severus_vcf else "missing_severus_vcf",
            }
        )

    manifest_df = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    return manifest_df, labels_df


def write_anchor_tables(
    highlight_list: str | Path,
    severus_list: str | Path,
    output_dir: str | Path,
    manifest_name: str = "complex_sv_manifest.tsv",
    labels_name: str = "complex_sv_labels.tsv",
) -> tuple[Path, Path]:
    """Build and write the multimodal input TSVs."""
    anchors = read_highlight_anchors(highlight_list)
    severus_by_sample = read_severus_vcfs(severus_list)
    manifest_df, labels_df = build_anchor_tables(anchors, severus_by_sample)

    missing = labels_df.loc[labels_df["severus_vcf"].astype(str) == "", "sample_id"].unique().tolist()
    if missing:
        log.warning("Labels without a matching Severus VCF: %s", ", ".join(map(str, missing)))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    labels_path = output_dir / labels_name
    manifest_df.to_csv(manifest_path, sep="\t", index=False)
    labels_df.to_csv(labels_path, sep="\t", index=False)
    return manifest_path, labels_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--highlight_list", required=True, help="TSV/whitespace file: label wakhan_root chrom")
    parser.add_argument("--severus_list", required=True, help="One Severus VCF path per line")
    parser.add_argument("--output_dir", required=True, help="Directory for output TSVs")
    parser.add_argument("--manifest_name", default="complex_sv_manifest.tsv")
    parser.add_argument("--labels_name", default="complex_sv_labels.tsv")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)
    manifest_path, labels_path = write_anchor_tables(
        highlight_list=args.highlight_list,
        severus_list=args.severus_list,
        output_dir=args.output_dir,
        manifest_name=args.manifest_name,
        labels_name=args.labels_name,
    )
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote labels:   {labels_path}")


if __name__ == "__main__":
    main()
