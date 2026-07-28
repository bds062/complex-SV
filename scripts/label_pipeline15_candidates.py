#!/usr/bin/env python3
"""Apply external caller labels to caller-aware unlabeled candidate regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLASS_NAMES = ["ecDNA", "chromothripsis", "BFB"]


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(args.candidates).fillna("")
    members = pd.read_csv(args.members, sep="\t").fillna("")
    external = pd.read_csv(args.external_regions, sep="\t").fillna("")
    external_by_id = external.set_index(external["region_id"].astype(str), drop=False)

    labels_by_candidate: dict[str, set[str]] = {}
    assignment_rows: list[dict] = []
    external_members = members[members["member_type"].astype(str).eq("external")]
    for _, member in external_members.iterrows():
        candidate_id = str(member["candidate_id"])
        external_id = str(member["external_region_id"])
        if external_id not in external_by_id.index:
            raise KeyError(f"Unknown external region in candidate membership: {external_id}")
        region = external_by_id.loc[external_id]
        label = str(region["label"])
        labels_by_candidate.setdefault(candidate_id, set()).add(label)
        assignment_rows.append(
            {
                "candidate_id": candidate_id,
                "external_region_id": external_id,
                "sample_id": region["sample_id"],
                "chrom": region["chrom"],
                "start": region["start"],
                "end": region["end"],
                "class_name": label,
                "source": region["source"],
                "source_call_id": region["source_call_id"],
            }
        )

    labeled = candidates.copy()
    for class_name in CLASS_NAMES:
        labeled[class_name] = [
            "True" if class_name in labels_by_candidate.get(str(candidate_id), set()) else ""
            for candidate_id in labeled["candidate_id"]
        ]
    labeled["Seismic_Amplification"] = ""
    labeled.to_csv(output_dir / "merged_candidate_regions.csv", index=False)
    pd.DataFrame(assignment_rows).to_csv(output_dir / "external_label_assignments.tsv", sep="\t", index=False)

    class_sets = [labels_by_candidate.get(str(candidate_id), set()) for candidate_id in labeled["candidate_id"]]
    positive = [bool(labels) for labels in class_sets]
    summary = {
        "label_source": "external caller seed membership only",
        "candidate_rows": int(len(labeled)),
        "labeled_candidate_rows": int(sum(positive)),
        "empty_candidate_rows": int(len(labeled) - sum(positive)),
        "multilabel_candidate_rows": int(sum(len(labels) > 1 for labels in class_sets)),
        "class_counts": {
            class_name: int(sum(class_name in labels for labels in class_sets)) for class_name in CLASS_NAMES
        },
        "label_rule": "A candidate inherits labels only from external caller regions in its >50% reciprocal-overlap seed component.",
        "own_candidate_heuristic_labels_used": False,
    }
    (output_dir / "external_label_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--members", required=True)
    parser.add_argument("--external_regions", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
