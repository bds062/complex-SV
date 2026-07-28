#!/usr/bin/env python3
"""Build an exact external-caller interval table with regional CNA/SV metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gen_candidates import resolve_cna_vcf  # noqa: E402
from process_vcfs import (  # noqa: E402
    _chrom_arm,
    _summarize_candidate_intervals,
    calculate_ploidy,
    get_bps,
    read_centromere_bed,
    read_cna_vcf_to_dataframe,
)


CLASS_NAMES = ("ecDNA", "chromothripsis", "BFB")


def norm_chrom(value: object) -> str:
    text = str(value).strip()
    return text if text.lower().startswith("chr") else f"chr{text}"


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, sep="\t").fillna("")
    calls = pd.read_csv(args.external_regions, sep="\t").fillna("")
    centromeres = read_centromere_bed(args.centromeres)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calls["chrom"] = calls["chrom"].map(norm_chrom)
    calls["start"] = pd.to_numeric(calls["start"], errors="raise").astype(int)
    calls["end"] = pd.to_numeric(calls["end"], errors="raise").astype(int)
    calls = calls[calls["label"].isin(CLASS_NAMES)].copy()
    manifest_by_sample = manifest.set_index(manifest["sample_id"].astype(str), drop=False)

    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    for sample_index, (sample_id, sample_calls) in enumerate(calls.groupby("sample_id", sort=True), start=1):
        print(f"[{sample_index}/{calls['sample_id'].nunique()}] {sample_id}: {len(sample_calls)} method intervals")
        try:
            if str(sample_id) not in manifest_by_sample.index:
                raise KeyError(f"sample absent from manifest: {sample_id}")
            manifest_row = manifest_by_sample.loc[str(sample_id)]
            if isinstance(manifest_row, pd.DataFrame):
                manifest_row = manifest_row.iloc[0]
            cna_vcf = resolve_cna_vcf(manifest_row["wakhan_root"])
            severus_vcf = Path(str(manifest_row["severus_vcf"]))
            cna = read_cna_vcf_to_dataframe(cna_vcf)
            breakpoints = get_bps(severus_vcf)
            sample_ploidy = calculate_ploidy(cna)

            coordinates = sample_calls[["chrom", "start", "end"]].drop_duplicates().copy()
            coordinates["arm"] = [
                _chrom_arm(str(row.chrom), int(row.start), int(row.end), centromeres)
                for row in coordinates.itertuples(index=False)
            ]
            coordinates["n_windows"] = 1
            metrics = _summarize_candidate_intervals(
                coordinates,
                cna,
                breakpoints,
                sample_ploidy=sample_ploidy,
                apply_candidate_filter=False,
            )
            for class_name in (*CLASS_NAMES, "Seismic_Amplification"):
                if class_name in metrics.columns:
                    metrics = metrics.drop(columns=class_name)
            merged = sample_calls.merge(metrics, on=["chrom", "start", "end"], how="left", validate="many_to_one")
            if merged["arm"].isna().any():
                raise RuntimeError("regional metric extraction omitted one or more exact caller intervals")
            merged.insert(0, "candidate_id", merged["region_id"].astype(str))
            merged.insert(1, "cluster_id", merged["region_id"].astype(str))
            merged["discovery_source"] = "exact_external_caller_interval"
            merged["cna_vcf"] = str(cna_vcf)
            merged["severus_vcf"] = str(severus_vcf)
            for class_name in CLASS_NAMES:
                merged[class_name] = merged["label"].eq(class_name)
            merged["Seismic_Amplification"] = False
            frames.append(merged)
        except Exception as exc:
            failures.append({"sample_id": str(sample_id), "error": str(exc)})
            if not args.keep_going:
                raise
            print(f"  ERROR: {exc}")

    regions = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if regions.empty:
        raise RuntimeError("No exact method regions were prepared")
    regions.to_csv(output_dir / "method_regions.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failed_samples.tsv", sep="\t", index=False)

    coordinate_groups = regions.groupby(["sample_id", "chrom", "start", "end"])["label"].agg(lambda x: ";".join(sorted(set(x))))
    summary = {
        "input_external_regions": str(args.external_regions),
        "manifest": str(args.manifest),
        "region_definition": "exact deduplicated intervals emitted by the external methods",
        "n_regions": int(len(regions)),
        "n_samples": int(regions["sample_id"].nunique()),
        "class_counts": {name: int(regions["label"].eq(name).sum()) for name in CLASS_NAMES},
        "source_counts": {str(k): int(v) for k, v in regions["source"].value_counts().items()},
        "n_unique_coordinates": int(len(coordinate_groups)),
        "n_cross_label_coordinate_conflicts": int(coordinate_groups.str.contains(";").sum()),
        "failed_samples": int(len(failures)),
    }
    (output_dir / "method_region_corpus_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--external_regions", type=Path, required=True)
    parser.add_argument("--centromeres", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
