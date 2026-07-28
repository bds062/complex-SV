#!/usr/bin/env python3
"""Identify unlabeled candidate regions from structural signals and caller seeds.

The output deliberately contains no complex-SV class columns. External callers
contribute coordinates only; their identities and labels are retained in a
separate membership table for a later supervision step.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
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


CLASS_COLUMNS = ["ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"]


@dataclass(frozen=True)
class ExternalRegion:
    region_id: str
    sample_id: str
    chrom: str
    start: int
    end: int
    label: str
    source: str
    source_call_id: str
    source_region: str


def norm_chrom(value: object) -> str:
    text = str(value).strip()
    return text if text.lower().startswith("chr") else f"chr{text}"


def parse_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_region(value: object) -> tuple[str, int, int] | None:
    match = re.match(r"([^:;\s]+):(\d+(?:\.0+)?)-(\d+(?:\.0+)?)", str(value).strip())
    if not match:
        return None
    start, end = parse_int(match.group(2)), parse_int(match.group(3))
    if start is None or end is None:
        return None
    return norm_chrom(match.group(1)), min(start, end), max(start, end)


def split_regions(value: object) -> list[tuple[str, int, int]]:
    return [parsed for token in str(value or "").replace(",", ";").split(";") if (parsed := parse_region(token))]


def load_external_regions(
    bfb_path: Path,
    shatterseek_path: Path,
    coral_path: Path,
    manifest_samples: set[str],
) -> list[ExternalRegion]:
    rows: list[tuple[str, str, int, int, str, str, str, str]] = []

    bfb = pd.read_csv(bfb_path, sep="\t").fillna("")
    for _, row in bfb.iterrows():
        sample = str(row.get("sample_id", "")).strip()
        if sample not in manifest_samples:
            continue
        parsed = parse_region(row.get("amplified_region", ""))
        if parsed is None:
            start, end = parse_int(row.get("start")), parse_int(row.get("end"))
            if start is None or end is None:
                continue
            parsed = (norm_chrom(row.get("chrom", "")), min(start, end), max(start, end))
        chrom, start, end = parsed
        call_id = str(row.get("call_id", "")).strip() or f"{sample}:{chrom}:{start}-{end}"
        source = str(row.get("source", "")).strip() or "BFBArchitect"
        rows.append((sample, chrom, start, end, "BFB", source, call_id, f"{chrom}:{start}-{end}"))

    shatter = pd.read_csv(shatterseek_path, sep="\t").fillna("")
    for _, row in shatter.iterrows():
        sample = str(row.get("sample_id", "")).strip()
        call_class = str(row.get("chromothripsis_class", "")).strip().lower()
        if sample not in manifest_samples or call_class not in {"canonical", "noncanonical"}:
            continue
        chrom = norm_chrom(row.get("chrom", ""))
        start = parse_int(row.get("shatterseek_start", row.get("start")))
        end = parse_int(row.get("shatterseek_end", row.get("end")))
        if start is None or end is None:
            continue
        start, end = min(start, end), max(start, end)
        call_id = f"{sample}:{chrom}:{start}-{end}:{call_class}"
        rows.append((sample, chrom, start, end, "chromothripsis", "ShatterSeek", call_id, f"{chrom}:{start}-{end}"))

    coral = pd.read_csv(coral_path, sep="\t").fillna("")
    coral = coral.drop_duplicates([c for c in ["sample_id", "amplicon_id", "intervals"] if c in coral.columns])
    for row_index, row in coral.iterrows():
        sample = str(row.get("sample_id", "")).strip()
        if sample not in manifest_samples:
            continue
        call_id = str(row.get("call_id", "")).strip() or f"{sample}:{row.get('amplicon_id', '')}:{row_index}"
        intervals = split_regions(row.get("intervals", ""))
        if not intervals:
            parsed = parse_region(row.get("region", ""))
            intervals = [parsed] if parsed else []
        for interval_index, (chrom, start, end) in enumerate(intervals, start=1):
            interval_id = f"{call_id}:interval{interval_index}"
            rows.append((sample, chrom, start, end, "ecDNA", "CORAL", interval_id, f"{chrom}:{start}-{end}"))

    seen: set[tuple[str, str, int, int, str, str]] = set()
    regions: list[ExternalRegion] = []
    for sample, chrom, start, end, label, source, source_id, source_region in rows:
        key = (sample, chrom, start, end, label, source)
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            ExternalRegion(
                region_id=f"ER{len(regions) + 1:06d}",
                sample_id=sample,
                chrom=chrom,
                start=start,
                end=end,
                label=label,
                source=source,
                source_call_id=source_id,
                source_region=source_region,
            )
        )
    return regions


def overlap_fraction(left: dict, right: dict) -> float:
    overlap = max(0, min(left["end"], right["end"]) - max(left["start"], right["start"]) + 1)
    if not overlap:
        return 0.0
    left_len = max(1, left["end"] - left["start"] + 1)
    right_len = max(1, right["end"] - right["start"] + 1)
    return min(overlap / left_len, overlap / right_len)


def merge_seeds(base: pd.DataFrame, external: list[ExternalRegion], threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds: list[dict] = []
    for row_index, row in base.iterrows():
        seeds.append(
            {
                "sample_id": str(row["sample_id"]),
                "chrom": norm_chrom(row["chrom"]),
                "start": int(row.get("start", row.get("start_bp"))),
                "end": int(row.get("end", row.get("end_bp"))),
                "member_type": "structural",
                "member_id": str(row.get("candidate_id", f"structural_{row_index + 1}")),
                "external_region_id": "",
            }
        )
    for region in external:
        seeds.append(
            {
                "sample_id": region.sample_id,
                "chrom": region.chrom,
                "start": region.start,
                "end": region.end,
                "member_type": "external",
                "member_id": region.region_id,
                "external_region_id": region.region_id,
            }
        )

    parent = list(range(len(seeds)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, seed in enumerate(seeds):
        by_key[(seed["sample_id"], seed["chrom"])].append(index)
    for indices in by_key.values():
        for offset, left_index in enumerate(indices):
            for right_index in indices[offset + 1 :]:
                if overlap_fraction(seeds[left_index], seeds[right_index]) > threshold:
                    union(left_index, right_index)

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(seeds)):
        components[find(index)].append(index)

    ordered = sorted(
        components.values(),
        key=lambda members: (
            seeds[members[0]]["sample_id"],
            seeds[members[0]]["chrom"],
            min(seeds[i]["start"] for i in members),
            max(seeds[i]["end"] for i in members),
        ),
    )
    interval_rows: list[dict] = []
    member_rows: list[dict] = []
    for component_index, members in enumerate(ordered, start=1):
        candidate_id = f"P15C{component_index:06d}"
        member_seeds = [seeds[i] for i in members]
        has_structural = any(seed["member_type"] == "structural" for seed in member_seeds)
        has_external = any(seed["member_type"] == "external" for seed in member_seeds)
        discovery_source = "merged" if has_structural and has_external else ("external_seed" if has_external else "structural")
        interval_rows.append(
            {
                "candidate_id": candidate_id,
                "sample_id": member_seeds[0]["sample_id"],
                "chrom": member_seeds[0]["chrom"],
                "start": min(seed["start"] for seed in member_seeds),
                "end": max(seed["end"] for seed in member_seeds),
                "n_windows": 1,
                "discovery_source": discovery_source,
            }
        )
        for seed in member_seeds:
            member_rows.append({"candidate_id": candidate_id, **seed})
    return pd.DataFrame(interval_rows), pd.DataFrame(member_rows)


def external_frame(regions: list[ExternalRegion]) -> pd.DataFrame:
    return pd.DataFrame([region.__dict__ for region in regions])


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest, sep="\t").fillna("")
    manifest_samples = set(manifest["sample_id"].astype(str))
    base = pd.read_csv(args.base_candidates).fillna("")
    base = base[base["sample_id"].astype(str).isin(manifest_samples)].copy()
    external = load_external_regions(
        Path(args.bfb_calls), Path(args.shatterseek_calls), Path(args.coral_calls), manifest_samples
    )
    intervals, members = merge_seeds(base, external, float(args.reciprocal_overlap))
    centromeres = read_centromere_bed(args.centromeres)
    manifest_by_sample = manifest.set_index(manifest["sample_id"].astype(str), drop=False)

    feature_frames: list[pd.DataFrame] = []
    for sample_index, (sample_id, sample_intervals) in enumerate(intervals.groupby("sample_id", sort=True), start=1):
        print(f"[{sample_index}/{intervals['sample_id'].nunique()}] {sample_id}: {len(sample_intervals)} candidate regions")
        manifest_row = manifest_by_sample.loc[sample_id]
        cna_vcf = resolve_cna_vcf(manifest_row["wakhan_root"])
        severus_vcf = Path(str(manifest_row["severus_vcf"]))
        cna = read_cna_vcf_to_dataframe(cna_vcf)
        breakpoints = get_bps(severus_vcf)
        valid_rows: list[dict] = []
        for record in sample_intervals.to_dict("records"):
            chrom_cna = cna[cna["chrom"].astype(str).eq(record["chrom"])]
            if not chrom_cna.empty:
                chrom_start = int(chrom_cna["start"].min())
                chrom_end = int(chrom_cna["end"].max())
                record["start"] = max(chrom_start, int(record["start"]))
                record["end"] = min(chrom_end, int(record["end"]))
            if record["start"] > record["end"]:
                continue
            record["arm"] = _chrom_arm(record["chrom"], record["start"], record["end"], centromeres)
            record["component_intervals"] = f"{record['start']}-{record['end']}"
            valid_rows.append(record)
        interval_df = pd.DataFrame(valid_rows)
        summarized = _summarize_candidate_intervals(
            interval_df,
            cna,
            breakpoints,
            sample_ploidy=calculate_ploidy(cna),
            apply_candidate_filter=False,
        )
        key_meta = interval_df.set_index(["chrom", "start", "end"])[["candidate_id", "discovery_source"]]
        summarized = summarized.join(key_meta, on=["chrom", "start", "end"])
        summarized.insert(0, "sample_id", sample_id)
        summarized.insert(2, "wakhan_sample_id", str(manifest_row.get("wakhan_sample_id", "")))
        summarized.insert(3, "wakhan_root", str(manifest_row["wakhan_root"]))
        summarized.insert(4, "cna_vcf", str(cna_vcf))
        summarized.insert(5, "severus_vcf", str(severus_vcf))
        feature_frames.append(summarized)

    candidates = pd.concat(feature_frames, ignore_index=True, sort=False)
    for column in CLASS_COLUMNS:
        if column in candidates.columns:
            candidates = candidates.drop(columns=column)
    candidates["cluster_id"] = [f"C{index + 1}" for index in range(len(candidates))]
    preferred = [
        "sample_id", "candidate_id", "wakhan_sample_id", "wakhan_root", "cna_vcf", "severus_vcf",
        "cluster_id", "chrom", "arm", "start", "end", "discovery_source",
    ]
    candidates = candidates[preferred + [column for column in candidates.columns if column not in preferred]]
    retained_ids = set(candidates["candidate_id"].astype(str))
    members = members[members["candidate_id"].astype(str).isin(retained_ids)].copy()
    represented_external = set(members.loc[members["member_type"].eq("external"), "external_region_id"].astype(str))
    all_external = {region.region_id for region in external}
    if represented_external != all_external:
        missing = sorted(all_external - represented_external)
        raise RuntimeError(f"External caller regions lost during candidate construction: {missing[:10]}")

    candidates.to_csv(output_dir / "candidate_regions_unlabeled.csv", index=False)
    members.to_csv(output_dir / "candidate_region_members.tsv", sep="\t", index=False)
    external_frame(external).to_csv(output_dir / "external_regions.tsv", sep="\t", index=False)
    summary = {
        "candidate_identifier": "structural_plus_external_caller_seeds",
        "base_structural_candidates": int(len(base)),
        "external_regions": int(len(external)),
        "output_candidates": int(len(candidates)),
        "output_samples": int(candidates["sample_id"].nunique()),
        "external_regions_represented": int(len(represented_external)),
        "external_region_recall": float(len(represented_external) / len(external)) if external else 1.0,
        "candidates_by_discovery_source": {
            str(key): int(value) for key, value in candidates["discovery_source"].value_counts().items()
        },
        "external_regions_by_source": {
            str(key): int(value) for key, value in external_frame(external)["source"].value_counts().items()
        },
        "reciprocal_overlap_threshold": float(args.reciprocal_overlap),
        "bfb_coordinate_rule": "amplified_region from the supplied BFB call table; chrom/start/end only as fallback",
        "shatterseek_coordinate_rule": "deduplicated shatterseek_start-shatterseek_end event interval",
        "coral_coordinate_rule": "each native interval in the passing ecDNA call",
        "contains_event_labels": False,
    }
    (output_dir / "candidate_identifier_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--base_candidates", required=True)
    parser.add_argument("--bfb_calls", required=True)
    parser.add_argument("--shatterseek_calls", required=True)
    parser.add_argument("--coral_calls", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--centromeres", default="/data/KolmogorovLab/srinivasanbd/results/grch38.cen_coord.curated.bed")
    parser.add_argument("--reciprocal_overlap", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
