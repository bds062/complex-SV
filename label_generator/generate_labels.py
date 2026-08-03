#!/usr/bin/env python3
"""Build publication label tables from curated complex-SV caller TSV files."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


CLASSES = ("BFB", "chromothripsis", "ecDNA", "seismic_amplification")
DEFAULT_FILES = {
    "BFB": ("bfbarchitect_calls.tsv", "bfbarchitect_noncanonical_calls.tsv"),
    "chromothripsis": ("chromothripsis_calls.tsv", "chromothripsis_noncanonical_calls.tsv"),
    "ecDNA": ("coral_ecDNA_calls.tsv",),
    "seismic_amplification": (
        "seismic_amplification_calls.tsv",
        "seismic_amplification_noncanonical_calls.tsv",
    ),
}
CHROM_LENGTHS_GRCH38 = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559,
    "chr4": 190_214_555, "chr5": 181_538_259, "chr6": 170_805_979,
    "chr7": 159_345_973, "chr8": 145_138_636, "chr9": 138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16": 90_338_345, "chr17": 83_257_441, "chr18": 80_373_285,
    "chr19": 58_617_616, "chr20": 64_444_167, "chr21": 46_709_983,
    "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}
REGION_PATTERN = re.compile(r"(chr[^:;\s]+):(\d+)-(\d+)")


def normalize_chrom(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise ValueError("empty chromosome")
    text = re.sub(r"^chr", "", text, flags=re.IGNORECASE)
    aliases = {"23": "X", "24": "Y"}
    return f"chr{aliases.get(text, text)}"


def annotation_type(filename: str) -> str:
    return "noncanonical" if "noncanonical" in filename.lower() else "canonical"


def parse_intervals(row: pd.Series, broad_class: str) -> list[tuple[str, int, int]]:
    if broad_class == "ecDNA" and "region" in row.index and pd.notna(row["region"]):
        intervals = [
            (normalize_chrom(match.group(1)), int(match.group(2)), int(match.group(3)))
            for match in REGION_PATTERN.finditer(str(row["region"]))
        ]
        if intervals:
            return intervals

    required = {"chrom", "start", "end"}
    missing = sorted(required - set(row.index))
    if missing:
        raise ValueError(f"missing coordinate columns: {', '.join(missing)}")
    chromosomes = [normalize_chrom(value) for value in str(row["chrom"]).split(";")]
    if len(chromosomes) != 1:
        raise ValueError("multi-chromosome rows require a parseable region column")
    return [(chromosomes[0], int(row["start"]), int(row["end"]))]


def read_source(label_dir: Path, broad_class: str, filename: str) -> list[dict[str, object]]:
    path = label_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"required label file not found: {path}")
    table = pd.read_csv(path, sep="\t")
    if "sample_id" not in table.columns:
        raise ValueError(f"{path} lacks required sample_id column")

    records: list[dict[str, object]] = []
    for row_index, row in table.iterrows():
        source_id = str(row.get("call_id", row.get("region", f"row_{row_index + 2}")))
        intervals = parse_intervals(row, broad_class)
        for interval_index, (chrom, start, end) in enumerate(intervals, start=1):
            if chrom not in CHROM_LENGTHS_GRCH38:
                raise ValueError(f"{path}:{row_index + 2}: unsupported GRCh38 chromosome {chrom}")
            if start < 0 or end <= start:
                raise ValueError(f"{path}:{row_index + 2}: invalid interval {chrom}:{start}-{end}")
            if end > CHROM_LENGTHS_GRCH38[chrom]:
                raise ValueError(f"{path}:{row_index + 2}: interval exceeds {chrom} length")
            records.append({
                "sample_id": str(row["sample_id"]).strip(),
                "chrom": chrom,
                "start": start,
                "end": end,
                "label": broad_class,
                "source_file": filename,
                "source_record_id": (
                    f"{source_id}:interval{interval_index}" if len(intervals) > 1 else source_id
                ),
                "source_row": row_index + 2,
                "annotation_type": annotation_type(filename),
            })
    return records


def build_labels(label_dir: Path) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for broad_class in CLASSES:
        for filename in DEFAULT_FILES[broad_class]:
            records.extend(read_source(label_dir, broad_class, filename))
    labels = pd.DataFrame(records)
    if labels.empty:
        raise ValueError("no labels were read")
    keys = ["sample_id", "chrom", "start", "end", "label"]
    duplicated = labels.duplicated(keys, keep=False)
    if duplicated.any():
        examples = labels.loc[duplicated, keys + ["source_file"]].head(10).to_dict("records")
        raise ValueError(f"duplicate class intervals across source files: {examples}")
    labels = labels.sort_values(["sample_id", "chrom", "start", "end", "label"]).reset_index(drop=True)
    labels.insert(0, "region_id", [f"CSVL{i:06d}" for i in range(1, len(labels) + 1)])
    return labels


def write_outputs(labels: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_dir / "interval_labels.tsv", sep="\t", index=False)

    chromosome_sources = (
        labels.groupby(["sample_id", "chrom", "label"], sort=True)
        .agg(
            n_source_intervals=("region_id", "size"),
            source_region_ids=("region_id", lambda x: ";".join(map(str, x))),
            source_files=("source_file", lambda x: ";".join(sorted(set(map(str, x))))),
            annotation_types=("annotation_type", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    chromosome_sources.insert(
        0, "chromosome_label_id", [f"CSVC{i:06d}" for i in range(1, len(chromosome_sources) + 1)]
    )
    chromosome_sources["start_bp"] = 0
    chromosome_sources["end_bp"] = chromosome_sources["chrom"].map(CHROM_LENGTHS_GRCH38).astype(int)
    chromosome_sources.to_csv(output_dir / "chromosome_labels.tsv", sep="\t", index=False)

    embedding = chromosome_sources.rename(
        columns={"chromosome_label_id": "label_id", "label": "sv_class"}
    )
    embedding["label_scope"] = "chromosome"
    embedding[["label_id", "sample_id", "chrom", "start_bp", "end_bp", "sv_class", "label_scope"]].to_csv(
        output_dir / "embedding_labels.tsv", sep="\t", index=False
    )

    summary = (
        labels.groupby("label", sort=False)
        .agg(interval_labels=("region_id", "size"), labeled_genomes=("sample_id", "nunique"))
        .reindex(CLASSES)
        .reset_index()
    )
    chromosome_counts = chromosome_sources.groupby("label").size().rename("chromosome_labels")
    summary = summary.join(chromosome_counts, on="label")
    summary.to_csv(output_dir / "label_summary.tsv", sep="\t", index=False)

    manifest = {
        "coordinate_convention": "GRCh38; interval start/end retained from input; end treated as inclusive downstream",
        "classes": list(CLASSES),
        "source_files": {key: list(value) for key, value in DEFAULT_FILES.items()},
        "interval_labels": int(len(labels)),
        "chromosome_class_labels": int(len(chromosome_sources)),
        "labeled_genomes": int(labels["sample_id"].nunique()),
        "canonicality_is_prediction_target": False,
    }
    (output_dir / "label_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True, help="Directory containing the seven curated TSV files.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    write_outputs(build_labels(args.label_dir), args.output_dir)
