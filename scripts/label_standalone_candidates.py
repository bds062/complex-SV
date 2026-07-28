#!/usr/bin/env python3
"""Assign offline external-caller labels to standalone CNA/SV candidates."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


CLASS_NAMES = ["ecDNA", "chromothripsis", "BFB"]


def norm_chrom(value: object) -> str:
    text = str(value).strip()
    return text if text.lower().startswith("chr") else f"chr{text}"


def overlap_length(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def cluster_key(row: pd.Series) -> str:
    cluster_id = str(row.get("cluster_id", "")).strip()
    if not cluster_id or cluster_id.lower() in {"nan", "none"}:
        cluster_id = str(row["candidate_id"])
    return f"{row['sample_id']}:{cluster_id}"


def run(args: argparse.Namespace) -> None:
    candidates = pd.read_csv(args.candidates).fillna("")
    external = pd.read_csv(args.external_regions, sep="\t").fillna("")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calls_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for call in external.to_dict("records"):
        calls_by_key[(str(call["sample_id"]), norm_chrom(call["chrom"]))].append(call)

    direct_labels: dict[str, set[str]] = defaultdict(set)
    direct_assignments: list[dict] = []
    overlapping_call_ids: set[str] = set()
    matched_call_ids: set[str] = set()
    for candidate in candidates.to_dict("records"):
        candidate_id = str(candidate["candidate_id"])
        sample_id = str(candidate["sample_id"])
        chrom = norm_chrom(candidate["chrom"])
        start, end = int(candidate["start"]), int(candidate["end"])
        candidate_length = max(1, end - start + 1)
        for call in calls_by_key[(sample_id, chrom)]:
            call_start, call_end = int(call["start"]), int(call["end"])
            overlap = overlap_length(start, end, call_start, call_end)
            if overlap <= 0:
                continue
            call_id = str(call["region_id"])
            overlapping_call_ids.add(call_id)
            call_length = max(1, call_end - call_start + 1)
            candidate_fraction = overlap / candidate_length
            call_fraction = overlap / call_length
            reciprocal = min(candidate_fraction, call_fraction)
            is_match = candidate_fraction > args.overlap_threshold or call_fraction > args.overlap_threshold
            if is_match:
                direct_labels[candidate_id].add(str(call["label"]))
                matched_call_ids.add(call_id)
            direct_assignments.append(
                {
                    "candidate_id": candidate_id,
                    "external_region_id": call_id,
                    "sample_id": sample_id,
                    "chrom": chrom,
                    "class_name": call["label"],
                    "source": call["source"],
                    "candidate_overlap_fraction": candidate_fraction,
                    "call_overlap_fraction": call_fraction,
                    "reciprocal_overlap_fraction": reciprocal,
                    "is_direct_label_match": is_match,
                }
            )

    labeled = candidates.copy()
    labeled["_cluster_key"] = labeled.apply(cluster_key, axis=1)
    labels_by_cluster: dict[str, set[str]] = defaultdict(set)
    for row in labeled.to_dict("records"):
        labels_by_cluster[str(row["_cluster_key"])].update(direct_labels.get(str(row["candidate_id"]), set()))

    final_labels: dict[str, set[str]] = {}
    propagation_rows: list[dict] = []
    for row in labeled.to_dict("records"):
        candidate_id = str(row["candidate_id"])
        key = str(row["_cluster_key"])
        direct = direct_labels.get(candidate_id, set())
        propagated = labels_by_cluster.get(key, set()) if args.propagate_cluster_labels else direct
        final_labels[candidate_id] = set(propagated)
        propagation_rows.append(
            {
                "candidate_id": candidate_id,
                "sample_id": row["sample_id"],
                "cluster_key": key,
                "direct_classes": ";".join(sorted(direct)),
                "final_classes": ";".join(sorted(propagated)),
                "received_cluster_label": bool(propagated and propagated != direct),
            }
        )

    for class_name in CLASS_NAMES:
        labeled[class_name] = [
            "True" if class_name in final_labels[str(candidate_id)] else ""
            for candidate_id in labeled["candidate_id"]
        ]
    labeled["Seismic_Amplification"] = ""
    labeled = labeled.drop(columns=["_cluster_key"])
    labeled.to_csv(output_dir / "merged_candidate_regions.csv", index=False)
    pd.DataFrame(direct_assignments).to_csv(output_dir / "candidate_caller_overlaps.tsv", sep="\t", index=False)
    propagation = pd.DataFrame(propagation_rows)
    propagation.to_csv(output_dir / "cluster_label_propagation.tsv", sep="\t", index=False)

    direct_positive = {candidate_id for candidate_id, labels in direct_labels.items() if labels}
    final_positive = {candidate_id for candidate_id, labels in final_labels.items() if labels}
    summary = {
        "candidate_rows": int(len(labeled)),
        "direct_labeled_candidate_rows": int(len(direct_positive)),
        "final_labeled_candidate_rows": int(len(final_positive)),
        "cluster_propagated_candidate_rows": int(len(final_positive - direct_positive)),
        "empty_candidate_rows": int(len(labeled) - len(final_positive)),
        "multilabel_candidate_rows": int(sum(len(labels) > 1 for labels in final_labels.values())),
        "class_counts": {
            name: int(sum(name in labels for labels in final_labels.values())) for name in CLASS_NAMES
        },
        "external_call_rows": int(len(external)),
        "matched_external_call_rows": int(len(matched_call_ids)),
        "overlapping_external_call_rows": int(len(overlapping_call_ids)),
        "unmatched_external_call_rows": int(len(external) - len(matched_call_ids)),
        "overlap_threshold": float(args.overlap_threshold),
        "direct_label_rule": "candidate coverage > threshold OR caller-region coverage > threshold",
        "cluster_label_propagation": bool(args.propagate_cluster_labels),
        "uses_callers_as_model_features": False,
    }
    (output_dir / "label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--external_regions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--overlap_threshold", type=float, default=0.50)
    parser.add_argument("--propagate_cluster_labels", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
