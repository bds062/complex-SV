#!/usr/bin/env python3
"""Normalize the cervical-panel BFB labels into split final-label TSVs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REGION_RE = re.compile(r"^(chr[^:\s]+):(\d+)-(\d+)$", re.IGNORECASE)


def normalize_column(name: object) -> str:
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower()
    return text or "unnamed"


def load_centromeres(path: Path) -> dict[str, tuple[int, int]]:
    centromeres: dict[str, tuple[int, int]] = {}
    if not path.exists():
        return centromeres
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 3:
            continue
        try:
            chrom, start, end = fields[0], int(fields[1]), int(fields[2])
        except ValueError:
            continue
        centromeres[chrom] = (start, end)
    return centromeres


def infer_arm(chrom: str, start: int, end: int, centromeres: dict[str, tuple[int, int]]) -> str:
    centromere = centromeres.get(chrom)
    if centromere is None:
        return ""
    cent_start, cent_end = centromere
    midpoint = (start + end) // 2
    if midpoint < cent_start:
        return "p"
    if midpoint > cent_end:
        return "q"
    return "cen"


def convert(input_csv: Path, output_dir: Path, centromere_bed: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    required = {"Sample Name", "Genomic Region (hg38)", "BFBArchitect Prediction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    predictions = frame["BFBArchitect Prediction"].str.strip()
    mapping = {"BFB(+)": "canonical", "BFB(-)": "noncanonical"}
    unexpected = sorted(set(predictions) - set(mapping))
    if unexpected:
        raise ValueError(f"Unexpected BFBArchitect Prediction values: {unexpected}")

    centromeres = load_centromeres(centromere_bed)
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()
    for zero_based_index, source_row in frame.iterrows():
        sample = source_row["Sample Name"].strip()
        region_text = source_row["Genomic Region (hg38)"].strip()
        match = REGION_RE.fullmatch(region_text)
        if not sample or match is None:
            raise ValueError(f"Invalid sample or interval at CSV row {zero_based_index + 2}: {source_row.to_dict()}")
        chrom = match.group(1)
        start, end = sorted((int(match.group(2)), int(match.group(3))))
        key = (sample, chrom, start, end)
        if key in seen:
            raise ValueError(f"Duplicate sample and interval at CSV row {zero_based_index + 2}: {key}")
        seen.add(key)

        row: dict[str, object] = {
            "sample_id": sample,
            "chrom": chrom,
            "arm": infer_arm(chrom, start, end, centromeres),
            "start": start,
            "end": end,
            "region": f"{chrom}:{start}-{end}",
            "call_id": f"CervicalPanelBFB:{zero_based_index + 2}",
            "source": "BFBArchitect",
            "bfb_class": mapping[predictions.iloc[zero_based_index]],
            "bfbarchitect_prediction": predictions.iloc[zero_based_index],
            "source_csv_row": zero_based_index + 2,
        }
        for column in frame.columns:
            normalized = normalize_column(column)
            if normalized in {"sample_name", "genomic_region_hg38", "bfbarchitect_prediction"}:
                continue
            if normalized == "bfb_score":
                row["bfb_score"] = source_row[column]
            else:
                row[normalized] = source_row[column]
        rows.append(row)

    result = pd.DataFrame(rows)
    canonical = result[result["bfb_class"] == "canonical"].reset_index(drop=True)
    noncanonical = result[result["bfb_class"] == "noncanonical"].reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    canonical.to_csv(output_dir / "bfbarchitect_calls.tsv", sep="\t", index=False)
    noncanonical.to_csv(output_dir / "bfbarchitect_noncanonical_calls.tsv", sep="\t", index=False)
    return canonical, noncanonical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--centromere-bed",
        type=Path,
        default=Path("/data/KolmogorovLab/srinivasanbd/results/grch38.cen_coord.curated.bed"),
    )
    args = parser.parse_args()
    canonical, noncanonical = convert(args.input_csv, args.output_dir, args.centromere_bed)
    print(f"canonical={len(canonical)} samples={canonical['sample_id'].nunique()}")
    print(f"noncanonical={len(noncanonical)} samples={noncanonical['sample_id'].nunique()}")


if __name__ == "__main__":
    main()
