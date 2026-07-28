#!/usr/bin/env python
"""Build pipeline13 candidate-region labels from external callsets.

The candidate universe is inherited from pipeline12. Old labels are cleared,
external calls are merged by >50% reciprocal overlap within sample/chromosome,
and candidate rows are labeled when at least 50% of the candidate interval is
covered by a merged external call. Candidate rows that receive no external label
remain empty_candidate_region negatives.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CLASS_NAMES = ["ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"]
POSITIVE_EVIDENCE = "candidate_region_label"
EMPTY_EVIDENCE = "candidate_region_empty"


@dataclass
class CallInterval:
    sample_id: str
    chrom: str
    start: int
    end: int
    labels: set[str]
    sources: set[str] = field(default_factory=set)
    call_ids: set[str] = field(default_factory=set)
    source_regions: set[str] = field(default_factory=set)
    n_source_calls: int = 1

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
    text = str(value).strip()
    match = re.match(r"([^:;\s]+):(\d+(?:\.0+)?)-(\d+(?:\.0+)?)", text)
    if not match:
        return None
    chrom = norm_chrom(match.group(1))
    start = parse_int(match.group(2))
    end = parse_int(match.group(3))
    if not chrom or start is None or end is None:
        return None
    if end < start:
        start, end = end, start
    return chrom, start, end


def split_interval_string(value: object) -> list[tuple[str, int, int]]:
    intervals: list[tuple[str, int, int]] = []
    for token in str(value or "").replace(",", ";").split(";"):
        parsed = parse_region(token)
        if parsed is not None:
            intervals.append(parsed)
    return intervals


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(int(a_end), int(b_end)) - max(int(a_start), int(b_start)) + 1)


def reciprocal_overlap(a: CallInterval, b: CallInterval) -> float:
    ov = overlap_len(a.start, a.end, b.start, b.end)
    if ov <= 0:
        return 0.0
    return min(ov / a.length(), ov / b.length())


def merge_two(a: CallInterval, b: CallInterval) -> CallInterval:
    return CallInterval(
        sample_id=a.sample_id,
        chrom=a.chrom,
        start=min(a.start, b.start),
        end=max(a.end, b.end),
        labels=set(a.labels) | set(b.labels),
        sources=set(a.sources) | set(b.sources),
        call_ids=set(a.call_ids) | set(b.call_ids),
        source_regions=set(a.source_regions) | set(b.source_regions),
        n_source_calls=int(a.n_source_calls) + int(b.n_source_calls),
    )


def merge_calls(calls: list[CallInterval], reciprocal_threshold: float) -> list[CallInterval]:
    merged = list(calls)
    changed = True
    while changed:
        changed = False
        next_records: list[CallInterval] = []
        used = [False] * len(merged)
        for i, call_i in enumerate(merged):
            if used[i]:
                continue
            current = call_i
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j]:
                    continue
                call_j = merged[j]
                if current.sample_id != call_j.sample_id or current.chrom != call_j.chrom:
                    continue
                if reciprocal_overlap(current, call_j) > float(reciprocal_threshold):
                    current = merge_two(current, call_j)
                    used[j] = True
                    changed = True
            next_records.append(current)
        merged = next_records
    return sorted(merged, key=lambda c: (c.sample_id, c.chrom, c.start, c.end, sorted(c.labels)))


def add_call(calls: list[CallInterval], sample_id: object, chrom: object, start: object, end: object, label: str, source: str, call_id: object, region: object = "") -> None:
    start_i = parse_int(start)
    end_i = parse_int(end)
    chrom_s = norm_chrom(chrom)
    sample_s = clean_sample(sample_id)
    if not sample_s or not chrom_s or start_i is None or end_i is None:
        return
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    calls.append(
        CallInterval(
            sample_id=sample_s,
            chrom=chrom_s,
            start=start_i,
            end=end_i,
            labels={label},
            sources={source},
            call_ids={str(call_id)},
            source_regions={str(region) if str(region).strip() else f"{chrom_s}:{start_i}-{end_i}"},
        )
    )


def load_bfb(path: Path) -> list[CallInterval]:
    df = pd.read_csv(path, sep="\t").fillna("")
    calls: list[CallInterval] = []
    for _, row in df.iterrows():
        add_call(
            calls,
            row.get("sample_id", ""),
            row.get("chrom", ""),
            row.get("start", ""),
            row.get("end", ""),
            "BFB",
            "BFBArchitect",
            row.get("call_id", ""),
            row.get("region", ""),
        )
    return calls


def load_shatterseek(path: Path) -> list[CallInterval]:
    df = pd.read_csv(path, sep="\t").fillna("")
    calls: list[CallInterval] = []
    for _, row in df.iterrows():
        cls = str(row.get("chromothripsis_class", "")).strip().lower()
        if cls == "canonical":
            label = "chromothripsis"
        elif cls == "noncanonical":
            label = "Seismic_Amplification"
        else:
            continue
        add_call(
            calls,
            row.get("sample_id", ""),
            row.get("chrom", ""),
            row.get("start", ""),
            row.get("end", ""),
            label,
            "ShatterSeek",
            f"{row.get('sample_id','')}:{row.get('chrom','')}:{row.get('start','')}-{row.get('end','')}:{cls}",
            f"{row.get('chrom','')}:{row.get('start','')}-{row.get('end','')}",
        )
    return calls


def load_coral(path: Path) -> list[CallInterval]:
    df = pd.read_csv(path, sep="\t").fillna("")
    if {"sample_id", "amplicon_id", "intervals"}.issubset(df.columns):
        df = df.drop_duplicates(["sample_id", "amplicon_id", "intervals"])
    calls: list[CallInterval] = []
    for _, row in df.iterrows():
        intervals = split_interval_string(row.get("intervals", ""))
        if not intervals:
            chrom = row.get("chrom", "")
            if ";" not in str(chrom):
                add_call(
                    calls,
                    row.get("sample_id", ""),
                    chrom,
                    row.get("start", ""),
                    row.get("end", ""),
                    "ecDNA",
                    "CORAL",
                    row.get("call_id", ""),
                    row.get("region", ""),
                )
            continue
        by_chrom: dict[str, list[tuple[int, int]]] = {}
        for chrom, start, end in intervals:
            by_chrom.setdefault(chrom, []).append((start, end))
        for chrom, chrom_intervals in by_chrom.items():
            start = min(s for s, _e in chrom_intervals)
            end = max(e for _s, e in chrom_intervals)
            add_call(
                calls,
                row.get("sample_id", ""),
                chrom,
                start,
                end,
                "ecDNA",
                "CORAL",
                row.get("call_id", ""),
                ";".join(f"{chrom}:{s}-{e}" for s, e in chrom_intervals),
            )
    return calls


def has_label(value: object) -> bool:
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "f", "no", "n", "none", "nan", "<na>"}


def label_value(class_name: str) -> str:
    if class_name == "chromothripsis":
        return "canonical"
    if class_name == "Seismic_Amplification":
        return "noncanonical"
    return "True"


def build_label_strings(row: pd.Series) -> tuple[str, str]:
    classes: list[str] = []
    raw: list[str] = []
    for class_name in CLASS_NAMES:
        value = row.get(class_name, "")
        if not has_label(value):
            continue
        classes.append(class_name)
        value_s = str(value).strip()
        raw.append(class_name if value_s in {"", "True", "true", "1"} else f"{class_name}:{value_s}")
    return ";".join(classes), ";".join(raw)


def calls_to_frame(calls: list[CallInterval]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "merged_call_id": f"MC{i + 1:05d}",
                "sample_id": c.sample_id,
                "chrom": c.chrom,
                "start": int(c.start),
                "end": int(c.end),
                "length": int(c.length()),
                "labels": ";".join(sorted(c.labels)),
                "sources": ";".join(sorted(c.sources)),
                "source_call_ids": ";".join(sorted(c.call_ids)),
                "source_regions": ";".join(sorted(c.source_regions)),
                "n_source_calls": int(c.n_source_calls),
            }
            for i, c in enumerate(calls)
        ]
    )


def assign_labels(candidates: pd.DataFrame, merged_calls: list[CallInterval], candidate_overlap_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = candidates.copy().fillna("")
    for class_name in CLASS_NAMES:
        out[class_name] = ""
    by_key: dict[tuple[str, str], list[tuple[str, CallInterval]]] = {}
    for i, call in enumerate(merged_calls):
        by_key.setdefault((call.sample_id, call.chrom), []).append((f"MC{i + 1:05d}", call))

    details: list[dict[str, Any]] = []
    for idx, row in out.iterrows():
        sample_id = clean_sample(row.get("sample_id", ""))
        chrom = norm_chrom(row.get("chrom", ""))
        start = parse_int(row.get("start", ""))
        end = parse_int(row.get("end", ""))
        labels: set[str] = set()
        matched_call_ids: list[str] = []
        matched_sources: set[str] = set()
        max_candidate_overlap = 0.0
        max_reciprocal_overlap = 0.0
        if start is not None and end is not None:
            if end < start:
                start, end = end, start
            candidate_len = max(1, end - start + 1)
            for merged_call_id, call in by_key.get((sample_id, chrom), []):
                ov = overlap_len(start, end, call.start, call.end)
                if ov <= 0:
                    continue
                candidate_overlap = ov / candidate_len
                call_overlap = ov / call.length()
                rec = min(candidate_overlap, call_overlap)
                max_candidate_overlap = max(max_candidate_overlap, candidate_overlap)
                max_reciprocal_overlap = max(max_reciprocal_overlap, rec)
                if candidate_overlap > float(candidate_overlap_threshold):
                    labels |= set(call.labels)
                    matched_call_ids.append(merged_call_id)
                    matched_sources |= set(call.sources)
        for class_name in sorted(labels):
            out.at[idx, class_name] = label_value(class_name)
        sv_classes, raw_sv_classes = build_label_strings(out.loc[idx])
        details.append(
            {
                "candidate_id": row.get("candidate_id", ""),
                "sample_id": sample_id,
                "chrom": chrom,
                "arm": row.get("arm", ""),
                "start": start,
                "end": end,
                "assigned_classes": sv_classes,
                "matched_merged_call_ids": ";".join(sorted(set(matched_call_ids))),
                "matched_sources": ";".join(sorted(matched_sources)),
                "max_candidate_overlap_fraction": max_candidate_overlap,
                "max_reciprocal_overlap_fraction": max_reciprocal_overlap,
            }
        )
    return out, pd.DataFrame(details)


def rewrite_candidate_embeddings(source_metadata: Path, output_metadata: Path, labeled_candidates: pd.DataFrame) -> None:
    meta = pd.read_csv(source_metadata, sep="\t").fillna("")
    if len(meta) != len(labeled_candidates):
        raise ValueError(f"metadata rows {len(meta)} != candidate rows {len(labeled_candidates)}")
    if "candidate_id" in meta.columns and not (meta["candidate_id"].astype(str).to_numpy() == labeled_candidates["candidate_id"].astype(str).to_numpy()).all():
        raise ValueError("candidate_id order mismatch between source metadata and labeled candidates")
    out = meta.copy()
    for class_name in CLASS_NAMES:
        out[class_name] = labeled_candidates[class_name].fillna("").astype(str).to_numpy()
    sv_classes: list[str] = []
    raw_sv_classes: list[str] = []
    evidence: list[str] = []
    label_scope: list[str] = []
    label_ids: list[str] = []
    for _, row in out.iterrows():
        sv, raw = build_label_strings(row)
        sv_classes.append(sv)
        raw_sv_classes.append(raw)
        positive = bool(sv)
        evidence.append(POSITIVE_EVIDENCE if positive else EMPTY_EVIDENCE)
        label_scope.append("region" if positive else "empty_candidate_region")
        label_ids.append(str(row.get("candidate_id", "")) if positive else "")
    out["sv_class"] = sv_classes
    out["sv_classes"] = sv_classes
    out["raw_sv_classes"] = raw_sv_classes
    out["evidence"] = evidence
    out["label_scope"] = label_scope
    out["candidate_scope"] = "candidate_region"
    out["label_id"] = label_ids
    if "raw_true_classes" in out.columns:
        out["raw_true_classes"] = raw_sv_classes
    if "true_classes" in out.columns:
        out["true_classes"] = sv_classes
    out.to_csv(output_metadata, sep="\t", index=False)


def write_summary(output_dir: Path, candidates: pd.DataFrame, merged_calls: pd.DataFrame, details: pd.DataFrame, args: argparse.Namespace) -> None:
    labeled = details["assigned_classes"].astype(str).ne("")
    class_counts: dict[str, int] = {}
    for class_name in CLASS_NAMES:
        class_counts[class_name] = int(candidates[class_name].map(has_label).sum())
    summary = {
        "pipeline": "pipeline13_external_labels",
        "candidate_rows": int(len(candidates)),
        "labeled_candidate_rows": int(labeled.sum()),
        "empty_candidate_rows": int((~labeled).sum()),
        "class_counts": class_counts,
        "merged_external_call_rows": int(len(merged_calls)),
        "merge_reciprocal_overlap_threshold": float(args.merge_reciprocal_overlap),
        "candidate_overlap_threshold": float(args.candidate_overlap),
        "candidate_label_rule": "candidate is assigned all labels from merged external calls covering > candidate_overlap fraction of the candidate interval",
        "external_merge_rule": "external calls within sample/chrom are union-merged when reciprocal overlap > merge_reciprocal_overlap; labels are unioned for multilabel calls",
        "inputs": {
            "pipeline12_candidates": str(args.pipeline12_candidates),
            "pipeline12_manifest": str(args.pipeline12_manifest),
            "bfb_calls": str(args.bfb_calls),
            "shatterseek_calls": str(args.shatterseek_calls),
            "coral_calls": str(args.coral_calls),
            "source_embedding_dir": str(args.source_embedding_dir),
        },
    }
    (output_dir / "external_label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{"class_name": k, "n_candidates": v} for k, v in class_counts.items()]).to_csv(
        output_dir / "external_label_class_counts.tsv", sep="\t", index=False
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline12_candidates", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/merged_candidate_regions.csv")
    parser.add_argument("--pipeline12_manifest", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/complex_sv_manifest.tsv")
    parser.add_argument("--source_embedding_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/candidate_region_classifier_general")
    parser.add_argument("--bfb_calls", default="/data/KolmogorovLab/srinivasanbd/results/bfbarchitect3/bfb_calls.tsv")
    parser.add_argument("--shatterseek_calls", default="/data/KolmogorovLab/srinivasanbd/results/shatterseek2/chromothripsis_calls.tsv")
    parser.add_argument("--coral_calls", default="/data/KolmogorovLab/srinivasanbd/results/coral/coral_calls.tsv")
    parser.add_argument("--output_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline13")
    parser.add_argument("--merge_reciprocal_overlap", type=float, default=0.50)
    parser.add_argument("--candidate_overlap", type=float, default=0.50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = output_dir / "candidate_region_classifier_general"
    model_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.pipeline12_candidates).fillna("")
    manifest_src = Path(args.pipeline12_manifest)
    shutil.copy2(manifest_src, output_dir / "complex_sv_manifest.tsv")

    calls = []
    calls.extend(load_bfb(Path(args.bfb_calls)))
    calls.extend(load_shatterseek(Path(args.shatterseek_calls)))
    calls.extend(load_coral(Path(args.coral_calls)))
    raw_calls_df = calls_to_frame(calls)
    raw_calls_df.rename(columns={"merged_call_id": "raw_call_id"}).to_csv(output_dir / "external_calls_raw.tsv", sep="\t", index=False)

    merged_calls = merge_calls(calls, float(args.merge_reciprocal_overlap))
    merged_calls_df = calls_to_frame(merged_calls)
    merged_calls_df.to_csv(output_dir / "external_calls_merged.tsv", sep="\t", index=False)

    labeled_candidates, assignment_details = assign_labels(candidates, merged_calls, float(args.candidate_overlap))
    labeled_candidates.to_csv(output_dir / "merged_candidate_regions.csv", index=False)
    assignment_details.to_csv(output_dir / "external_label_assignments.tsv", sep="\t", index=False)

    source_dir = Path(args.source_embedding_dir)
    for name in ["embeddings.npz", "selected_embedding_features.npz", "embedding_features.txt"]:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, model_dir / name)
    rewrite_candidate_embeddings(source_dir / "candidate_embeddings.tsv", model_dir / "candidate_embeddings.tsv", labeled_candidates)
    labeled_candidates.to_csv(model_dir / "candidate_regions_from_csv.tsv", sep="\t", index=False)
    write_summary(output_dir, labeled_candidates, merged_calls_df, assignment_details, args)
    print(f"Wrote pipeline13 labels to {output_dir}")
    print(json.dumps(json.loads((output_dir / "external_label_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
