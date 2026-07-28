#!/usr/bin/env python
"""Build pipeline14 labels from native external caller intervals.

Pipeline14 treats BFBArchitect, CoRaL, and ShatterSeek as the label sources.
Unlike pipeline13, no caller intervals are union-expanded before candidate
labeling.  This is important for CoRaL amplicons: a multi-interval amplicon is
kept as its native genomic pieces instead of being replaced by its enclosing
chromosomal span. A caller interval is matched to a candidate when either one
contains more than half of the other. Calls are linked into caller events for
auditability, but the links never alter the intervals used to assign labels.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CLASS_NAMES = ["ecDNA", "chromothripsis", "BFB"]
POSITIVE_EVIDENCE = "candidate_region_label"
EMPTY_EVIDENCE = "candidate_region_empty"


@dataclass
class NativeCall:
    native_call_id: str
    sample_id: str
    chrom: str
    start: int
    end: int
    label: str
    source: str
    source_call_id: str
    source_region: str
    event_ids: set[str] = field(default_factory=set)

    @property
    def length(self) -> int:
        return max(1, int(self.end) - int(self.start) + 1)


def norm_chrom(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    return text if text.lower().startswith("chr") else f"chr{text}"


def clean_sample(value: object) -> str:
    return str(value).strip()


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
    chrom = norm_chrom(match.group(1))
    start = parse_int(match.group(2))
    end = parse_int(match.group(3))
    if not chrom or start is None or end is None:
        return None
    return (chrom, min(start, end), max(start, end))


def split_interval_string(value: object) -> list[tuple[str, int, int]]:
    return [parsed for token in str(value or "").replace(",", ";").split(";") if (parsed := parse_region(token))]


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)) + 1)


def reciprocal_overlap(a: NativeCall, b: NativeCall) -> float:
    overlap = overlap_len(a.start, a.end, b.start, b.end)
    return min(overlap / a.length, overlap / b.length) if overlap else 0.0


def add_call(
    calls: list[NativeCall],
    sample_id: object,
    chrom: object,
    start: object,
    end: object,
    label: str,
    source: str,
    source_call_id: object,
    source_region: object = "",
) -> None:
    start_i, end_i = parse_int(start), parse_int(end)
    sample, chrom_s = clean_sample(sample_id), norm_chrom(chrom)
    if not sample or not chrom_s or start_i is None or end_i is None:
        return
    start_i, end_i = min(start_i, end_i), max(start_i, end_i)
    native_id = f"NC{len(calls) + 1:06d}"
    calls.append(
        NativeCall(
            native_call_id=native_id,
            sample_id=sample,
            chrom=chrom_s,
            start=start_i,
            end=end_i,
            label=label,
            source=source,
            source_call_id=str(source_call_id),
            source_region=str(source_region).strip() or f"{chrom_s}:{start_i}-{end_i}",
        )
    )


def load_bfb(path: Path) -> list[NativeCall]:
    calls: list[NativeCall] = []
    for _, row in pd.read_csv(path, sep="\t").fillna("").iterrows():
        add_call(calls, row.get("sample_id", ""), row.get("chrom", ""), row.get("start", ""), row.get("end", ""), "BFB", "BFBArchitect", row.get("call_id", ""), row.get("region", ""))
    return calls


def load_shatterseek(path: Path) -> list[NativeCall]:
    calls: list[NativeCall] = []
    for _, row in pd.read_csv(path, sep="\t").fillna("").iterrows():
        call_class = str(row.get("chromothripsis_class", "")).strip().lower()
        if call_class not in {"canonical", "noncanonical"}:
            continue
        # Pipeline14 deliberately uses one ShatterSeek class: chromothripsis.
        source_id = f"{row.get('sample_id', '')}:{row.get('chrom', '')}:{row.get('start', '')}-{row.get('end', '')}:{call_class}"
        add_call(calls, row.get("sample_id", ""), row.get("chrom", ""), row.get("start", ""), row.get("end", ""), "chromothripsis", "ShatterSeek", source_id, f"{row.get('chrom', '')}:{row.get('start', '')}-{row.get('end', '')}")
    return calls


def load_coral(path: Path) -> list[NativeCall]:
    df = pd.read_csv(path, sep="\t").fillna("")
    if {"sample_id", "amplicon_id", "intervals"}.issubset(df.columns):
        df = df.drop_duplicates(["sample_id", "amplicon_id", "intervals"])
    calls: list[NativeCall] = []
    for row_index, row in df.iterrows():
        source_call_id = str(row.get("call_id", "")).strip() or f"{row.get('sample_id', '')}:{row.get('amplicon_id', '')}:{row_index}"
        intervals = split_interval_string(row.get("intervals", ""))
        if intervals:
            for interval_index, (chrom, start, end) in enumerate(intervals, start=1):
                add_call(calls, row.get("sample_id", ""), chrom, start, end, "ecDNA", "CORAL", f"{source_call_id}:interval{interval_index}", f"{chrom}:{start}-{end}")
        elif ";" not in str(row.get("chrom", "")):
            add_call(calls, row.get("sample_id", ""), row.get("chrom", ""), row.get("start", ""), row.get("end", ""), "ecDNA", "CORAL", source_call_id, row.get("region", ""))
    return calls


def deduplicate_native_calls(calls: list[NativeCall]) -> list[NativeCall]:
    seen: set[tuple[str, str, int, int, str, str]] = set()
    deduplicated: list[NativeCall] = []
    for call in calls:
        key = (call.sample_id, call.chrom, call.start, call.end, call.label, call.source)
        if key not in seen:
            seen.add(key)
            deduplicated.append(call)
    for i, call in enumerate(deduplicated, start=1):
        call.native_call_id = f"NC{i:06d}"
    return deduplicated


def link_caller_events(calls: list[NativeCall], reciprocal_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Link, but do not union-expand, overlapping calls from different callers."""
    parent = list(range(len(calls)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        left, right = find(i), find(j)
        if left != right:
            parent[right] = left

    by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, call in enumerate(calls):
        by_key[(call.sample_id, call.chrom)].append(index)
    link_rows: list[dict[str, Any]] = []
    for key_indices in by_key.values():
        for offset, i in enumerate(key_indices):
            for j in key_indices[offset + 1 :]:
                left, right = calls[i], calls[j]
                if left.source == right.source:
                    continue
                rec = reciprocal_overlap(left, right)
                if rec > reciprocal_threshold:
                    union(i, j)
                    link_rows.append({
                        "left_native_call_id": left.native_call_id,
                        "right_native_call_id": right.native_call_id,
                        "sample_id": left.sample_id,
                        "chrom": left.chrom,
                        "reciprocal_overlap_fraction": rec,
                        "left_label": left.label,
                        "right_label": right.label,
                        "left_source": left.source,
                        "right_source": right.source,
                    })
    components: dict[int, list[int]] = defaultdict(list)
    for i in range(len(calls)):
        components[find(i)].append(i)
    event_rows: list[dict[str, Any]] = []
    for event_index, members in enumerate(sorted(components.values(), key=lambda group: (calls[group[0]].sample_id, calls[group[0]].chrom, min(calls[i].start for i in group))), start=1):
        event_id = f"CE{event_index:05d}"
        for i in members:
            calls[i].event_ids.add(event_id)
        event_rows.append({
            "caller_event_id": event_id,
            "sample_id": calls[members[0]].sample_id,
            "chrom": calls[members[0]].chrom,
            "event_start": min(calls[i].start for i in members),
            "event_end": max(calls[i].end for i in members),
            "labels": ";".join(sorted({calls[i].label for i in members})),
            "sources": ";".join(sorted({calls[i].source for i in members})),
            "native_call_ids": ";".join(calls[i].native_call_id for i in members),
            "n_native_calls": len(members),
            "uses_envelope_for_labels": False,
        })
    return pd.DataFrame(event_rows), pd.DataFrame(link_rows)


def calls_frame(calls: list[NativeCall]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "native_call_id": call.native_call_id,
            "sample_id": call.sample_id,
            "chrom": call.chrom,
            "start": call.start,
            "end": call.end,
            "length": call.length,
            "label": call.label,
            "source": call.source,
            "source_call_id": call.source_call_id,
            "source_region": call.source_region,
            "caller_event_ids": ";".join(sorted(call.event_ids)),
        }
        for call in calls
    ])


def has_label(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "f", "no", "n", "none", "nan", "<na>"}


def label_strings(row: pd.Series) -> tuple[str, str]:
    labels = [name for name in CLASS_NAMES if has_label(row.get(name, ""))]
    return ";".join(labels), ";".join(labels)


def assign_labels(candidates: pd.DataFrame, calls: list[NativeCall], candidate_overlap_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = candidates.copy().fillna("")
    for class_name in CLASS_NAMES:
        out[class_name] = ""
    by_key: dict[tuple[str, str], list[NativeCall]] = defaultdict(list)
    for call in calls:
        by_key[(call.sample_id, call.chrom)].append(call)
    detail_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for index, row in out.iterrows():
        sample_id, chrom = clean_sample(row.get("sample_id", "")), norm_chrom(row.get("chrom", ""))
        start, end = parse_int(row.get("start", "")), parse_int(row.get("end", ""))
        matched: list[NativeCall] = []
        max_candidate_overlap = 0.0
        max_reciprocal_overlap = 0.0
        if start is not None and end is not None:
            start, end = min(start, end), max(start, end)
            candidate_length = max(1, end - start + 1)
            for call in by_key[(sample_id, chrom)]:
                overlap = overlap_len(start, end, call.start, call.end)
                if not overlap:
                    continue
                candidate_overlap = overlap / candidate_length
                call_overlap = overlap / call.length
                max_candidate_overlap = max(max_candidate_overlap, candidate_overlap)
                max_reciprocal_overlap = max(max_reciprocal_overlap, min(candidate_overlap, call_overlap))
                matched_by_candidate_coverage = candidate_overlap > candidate_overlap_threshold
                matched_by_call_coverage = call_overlap > candidate_overlap_threshold
                is_match = matched_by_candidate_coverage or matched_by_call_coverage
                overlap_rows.append({
                    "candidate_id": row.get("candidate_id", ""),
                    "native_call_id": call.native_call_id,
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "class_name": call.label,
                    "source": call.source,
                    "candidate_overlap_fraction": candidate_overlap,
                    "call_overlap_fraction": call_overlap,
                    "reciprocal_overlap_fraction": min(candidate_overlap, call_overlap),
                    "matched_by_candidate_coverage": matched_by_candidate_coverage,
                    "matched_by_call_coverage": matched_by_call_coverage,
                    "is_label_match": is_match,
                })
                if is_match:
                    matched.append(call)
        for class_name in {call.label for call in matched}:
            out.at[index, class_name] = "True"
        sv_classes, raw_sv_classes = label_strings(out.loc[index])
        detail_rows.append({
            "candidate_id": row.get("candidate_id", ""),
            "sample_id": sample_id,
            "chrom": chrom,
            "arm": row.get("arm", ""),
            "start": start,
            "end": end,
            "assigned_classes": sv_classes,
            "matched_native_call_ids": ";".join(sorted({call.native_call_id for call in matched})),
            "matched_caller_event_ids": ";".join(sorted({event_id for call in matched for event_id in call.event_ids})),
            "matched_sources": ";".join(sorted({call.source for call in matched})),
            "max_candidate_overlap_fraction": max_candidate_overlap,
            "max_reciprocal_overlap_fraction": max_reciprocal_overlap,
        })
    return out, pd.DataFrame(detail_rows), pd.DataFrame(overlap_rows)


def unmatched_calls_frame(
    candidates: pd.DataFrame,
    calls: list[NativeCall],
    overlaps: pd.DataFrame,
) -> pd.DataFrame:
    matched_ids = set(
        overlaps.loc[overlaps["is_label_match"].astype(bool), "native_call_id"].astype(str)
    ) if not overlaps.empty else set()
    overlapping_ids = set(overlaps["native_call_id"].astype(str)) if not overlaps.empty else set()
    candidate_samples = set(candidates["sample_id"].astype(str))
    candidate_keys = set(zip(candidates["sample_id"].astype(str), candidates["chrom"].map(norm_chrom)))
    rows: list[dict[str, Any]] = []
    for call in calls:
        if call.native_call_id in matched_ids:
            continue
        if call.sample_id not in candidate_samples:
            reason = "sample_has_no_candidate_rows"
        elif (call.sample_id, call.chrom) not in candidate_keys:
            reason = "chromosome_has_no_candidate_rows"
        elif call.native_call_id not in overlapping_ids:
            reason = "no_genomic_overlap_with_candidate"
        else:
            reason = "overlap_below_containment_threshold"
        rows.append(
            {
                "native_call_id": call.native_call_id,
                "sample_id": call.sample_id,
                "chrom": call.chrom,
                "start": call.start,
                "end": call.end,
                "length": call.length,
                "label": call.label,
                "source": call.source,
                "source_call_id": call.source_call_id,
                "source_region": call.source_region,
                "unmatched_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def write_embedding_metadata(source_metadata: Path, output_metadata: Path, candidates: pd.DataFrame) -> None:
    metadata = pd.read_csv(source_metadata, sep="\t").fillna("")
    if len(metadata) != len(candidates) or not (metadata["candidate_id"].astype(str).to_numpy() == candidates["candidate_id"].astype(str).to_numpy()).all():
        raise ValueError("Source embedding metadata is not aligned to candidate rows")
    for class_name in CLASS_NAMES:
        metadata[class_name] = candidates[class_name].astype(str).to_numpy()
    metadata["sv_class"] = [label_strings(row)[0] for _, row in metadata.iterrows()]
    metadata["sv_classes"] = metadata["sv_class"]
    metadata["raw_sv_classes"] = [label_strings(row)[1] for _, row in metadata.iterrows()]
    positive = metadata["sv_class"].astype(str).ne("")
    metadata["evidence"] = np.where(positive, POSITIVE_EVIDENCE, EMPTY_EVIDENCE)
    metadata["label_scope"] = np.where(positive, "region", "empty_candidate_region")
    metadata["candidate_scope"] = "candidate_region"
    metadata["label_id"] = np.where(positive, metadata["candidate_id"].astype(str), "")
    for column in ["raw_true_classes", "true_classes"]:
        if column in metadata.columns:
            metadata[column] = metadata["raw_sv_classes"] if column == "raw_true_classes" else metadata["sv_classes"]
    metadata.to_csv(output_metadata, sep="\t", index=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline12_candidates", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/merged_candidate_regions.csv")
    parser.add_argument("--pipeline12_manifest", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/complex_sv_manifest.tsv")
    parser.add_argument("--source_embedding_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/candidate_region_classifier_general")
    parser.add_argument("--bfb_calls", default="/data/KolmogorovLab/srinivasanbd/results/bfbarchitect3/bfb_calls.tsv")
    parser.add_argument("--shatterseek_calls", default="/data/KolmogorovLab/srinivasanbd/results/shatterseek2/chromothripsis_calls.tsv")
    parser.add_argument("--coral_calls", default="/data/KolmogorovLab/srinivasanbd/results/coral/coral_ecDNA_candidate_calls.tsv")
    parser.add_argument("--output_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline14")
    parser.add_argument("--event_reciprocal_overlap", type=float, default=0.50)
    parser.add_argument("--candidate_overlap", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "candidate_region_classifier_general"
    model_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.pipeline12_candidates).fillna("")
    shutil.copy2(args.pipeline12_manifest, output_dir / "complex_sv_manifest.tsv")
    calls = deduplicate_native_calls(load_bfb(Path(args.bfb_calls)) + load_shatterseek(Path(args.shatterseek_calls)) + load_coral(Path(args.coral_calls)))
    events, links = link_caller_events(calls, float(args.event_reciprocal_overlap))
    calls_frame(calls).to_csv(output_dir / "external_calls_native.tsv", sep="\t", index=False)
    events.to_csv(output_dir / "caller_events.tsv", sep="\t", index=False)
    links.to_csv(output_dir / "caller_event_links.tsv", sep="\t", index=False)
    labeled, assignments, overlaps = assign_labels(candidates, calls, float(args.candidate_overlap))
    labeled.to_csv(output_dir / "merged_candidate_regions.csv", index=False)
    assignments.to_csv(output_dir / "external_label_assignments.tsv", sep="\t", index=False)
    overlaps.to_csv(output_dir / "candidate_caller_overlaps.tsv", sep="\t", index=False)
    unmatched = unmatched_calls_frame(candidates, calls, overlaps)
    unmatched.to_csv(output_dir / "unmatched_caller_calls.tsv", sep="\t", index=False)
    source_dir = Path(args.source_embedding_dir)
    for name in ["embeddings.npz", "selected_embedding_features.npz", "embedding_features.txt"]:
        if (source_dir / name).exists():
            shutil.copy2(source_dir / name, model_dir / name)
    write_embedding_metadata(source_dir / "candidate_embeddings.tsv", model_dir / "candidate_embeddings.tsv", labeled)
    labeled.to_csv(model_dir / "candidate_regions_from_csv.tsv", sep="\t", index=False)
    assigned = assignments["assigned_classes"].astype(str).ne("")
    summary = {
        "pipeline": "pipeline14_caller_labels",
        "classes": CLASS_NAMES,
        "candidate_rows": int(len(labeled)),
        "labeled_candidate_rows": int(assigned.sum()),
        "empty_candidate_rows": int((~assigned).sum()),
        "class_counts": {name: int(labeled[name].map(has_label).sum()) for name in CLASS_NAMES},
        "native_external_call_rows": int(len(calls)),
        "caller_event_rows": int(len(events)),
        "cross_caller_event_links": int(len(links)),
        "unmatched_native_call_rows": int(len(unmatched)),
        "unmatched_calls_by_source": {
            str(source): int(count)
            for source, count in unmatched["source"].value_counts().sort_index().items()
        } if not unmatched.empty else {},
        "unmatched_calls_by_reason": {
            str(reason): int(count)
            for reason, count in unmatched["unmatched_reason"].value_counts().sort_index().items()
        } if not unmatched.empty else {},
        "candidate_overlap_threshold": float(args.candidate_overlap),
        "event_reciprocal_overlap_threshold": float(args.event_reciprocal_overlap),
        "caller_label_rule": "A candidate receives a source label when native source intervals cover >50% of the candidate or the candidate covers >50% of a native source interval. Native intervals are never envelope-merged for labeling.",
        "caller_event_rule": "Calls from different callers with >50% reciprocal overlap are linked for auditing and multilabel event interpretation; links do not modify source boundaries.",
        "shatterseek_rule": "Both canonical and noncanonical ShatterSeek calls are labeled chromothripsis; Seismic Amplification is omitted.",
        "inputs": {
            "candidate_regions": str(args.pipeline12_candidates),
            "manifest": str(args.pipeline12_manifest),
            "bfb_calls": str(args.bfb_calls),
            "shatterseek_calls": str(args.shatterseek_calls),
            "coral_calls": str(args.coral_calls),
            "source_embedding_dir": str(args.source_embedding_dir),
        },
    }
    (output_dir / "external_label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{"class_name": name, "n_candidates": count} for name, count in summary["class_counts"].items()]).to_csv(output_dir / "external_label_class_counts.tsv", sep="\t", index=False)
    print(f"Wrote pipeline14 caller labels to {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
