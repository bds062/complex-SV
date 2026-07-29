#!/usr/bin/env python3
"""Collapse the selected Pipeline24 interval labels to chromosome-level targets."""

from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "summer_results/localization_model/labels/all_labels.tsv"
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]

# GRCh38 primary chromosome lengths. Wakhan rows are clipped to the available span.
CHROM_LENGTHS = {
    "chr1": 248_956_422, "chr2": 242_193_529, "chr3": 198_295_559,
    "chr4": 190_214_555, "chr5": 181_538_259, "chr6": 170_805_979,
    "chr7": 159_345_973, "chr8": 145_138_636, "chr9": 138_394_717,
    "chr10": 133_797_422, "chr11": 135_086_622, "chr12": 133_275_309,
    "chr13": 114_364_328, "chr14": 107_043_718, "chr15": 101_991_189,
    "chr16": 90_338_345, "chr17": 83_257_441, "chr18": 80_373_285,
    "chr19": 58_617_616, "chr20": 64_444_167, "chr21": 46_709_983,
    "chr22": 50_818_468, "chrX": 156_040_895, "chrY": 57_227_415,
}


def normalize_chrom(value: object) -> str:
    text = str(value).strip()
    if not text.startswith("chr"):
        text = f"chr{text}"
    return text


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(SOURCE, sep="\t")
    source["chrom"] = source["chrom"].map(normalize_chrom)
    source = source[source["label"].isin(CLASSES)].copy()

    collapsed = (
        source.groupby(["sample_id", "chrom", "label"], sort=True)
        .agg(
            n_interval_labels=("region_id", "size"),
            interval_label_ids=("region_id", lambda x: ";".join(map(str, x))),
        )
        .reset_index()
    )
    collapsed.insert(
        0, "chromosome_label_id",
        [f"P27C{i:06d}" for i in range(1, len(collapsed) + 1)],
    )
    collapsed["start_bp"] = 0
    collapsed["end_bp"] = collapsed["chrom"].map(CHROM_LENGTHS)
    if collapsed["end_bp"].isna().any():
        missing = sorted(collapsed.loc[collapsed.end_bp.isna(), "chrom"].unique())
        raise ValueError(f"missing GRCh38 lengths for {missing}")
    collapsed["end_bp"] = collapsed["end_bp"].astype(int)
    collapsed.to_csv(HERE / "chromosome_labels.tsv", sep="\t", index=False)

    embedding_labels = collapsed.rename(
        columns={
            "chromosome_label_id": "label_id",
            "label": "sv_class",
        }
    )
    embedding_labels["label_scope"] = "chromosome"
    embedding_labels[
        ["label_id", "sample_id", "chrom", "start_bp", "end_bp", "sv_class", "label_scope"]
    ].to_csv(HERE / "embedding_labels.tsv", sep="\t", index=False)

    samples = (
        collapsed.groupby("sample_id")["label"]
        .agg(lambda x: ";".join(sorted(set(map(str, x)))))
        .rename("classes")
        .reset_index()
    )
    samples.to_csv(HERE / "labeled_samples.tsv", sep="\t", index=False)

    summary = (
        collapsed.groupby("label")
        .agg(
            chromosome_class_positives=("chromosome_label_id", "size"),
            positive_chromosomes=("chrom", "size"),
            positive_samples=("sample_id", "nunique"),
            source_interval_labels=("n_interval_labels", "sum"),
        )
        .reindex(CLASSES)
        .reset_index()
    )
    summary.to_csv(HERE / "label_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(
        f"\n{len(source)} interval labels -> {len(collapsed)} chromosome-class positives, "
        f"{collapsed[['sample_id', 'chrom']].drop_duplicates().shape[0]} positive chromosomes, "
        f"{collapsed.sample_id.nunique()} labeled genomes"
    )


if __name__ == "__main__":
    main()
