#!/usr/bin/env python3
"""Measure label-free candidate proposal coverage against external call regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def overlap_length(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def run(args: argparse.Namespace) -> None:
    candidates = pd.read_csv(args.candidates).fillna("")
    calls = pd.read_csv(args.external_regions, sep="\t").fillna("")
    by_key = {
        key: frame for key, frame in candidates.groupby([candidates["sample_id"].astype(str), candidates["chrom"].astype(str)])
    }
    rows: list[dict] = []
    for _, call in calls.iterrows():
        key = (str(call["sample_id"]), str(call["chrom"]))
        call_start, call_end = int(call["start"]), int(call["end"])
        call_length = max(1, call_end - call_start + 1)
        best: dict | None = None
        max_call_fraction = 0.0
        max_candidate_fraction = 0.0
        max_either_fraction = 0.0
        max_reciprocal = 0.0
        for _, candidate in by_key.get(key, pd.DataFrame()).iterrows():
            candidate_start, candidate_end = int(candidate["start"]), int(candidate["end"])
            candidate_length = max(1, candidate_end - candidate_start + 1)
            overlap = overlap_length(call_start, call_end, candidate_start, candidate_end)
            call_fraction = overlap / call_length
            candidate_fraction = overlap / candidate_length
            reciprocal = min(call_fraction, candidate_fraction)
            max_call_fraction = max(max_call_fraction, call_fraction)
            max_candidate_fraction = max(max_candidate_fraction, candidate_fraction)
            max_either_fraction = max(max_either_fraction, call_fraction, candidate_fraction)
            max_reciprocal = max(max_reciprocal, reciprocal)
            score = (reciprocal, call_fraction, candidate_fraction, overlap)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "candidate_id": candidate.get("candidate_id", ""),
                    "candidate_start": candidate_start,
                    "candidate_end": candidate_end,
                    "proposal_reasons": candidate.get("proposal_reasons", ""),
                    "overlap_bp": overlap,
                    "call_overlap_fraction": call_fraction,
                    "candidate_overlap_fraction": candidate_fraction,
                    "reciprocal_overlap_fraction": reciprocal,
                }
        best = best or {
            "candidate_id": "", "candidate_start": "", "candidate_end": "", "proposal_reasons": "",
            "overlap_bp": 0, "call_overlap_fraction": 0.0, "candidate_overlap_fraction": 0.0,
            "reciprocal_overlap_fraction": 0.0,
        }
        best.pop("score", None)
        rows.append(
            {
                "external_region_id": call["region_id"],
                "sample_id": call["sample_id"],
                "chrom": call["chrom"],
                "call_start": call_start,
                "call_end": call_end,
                "class_name": call["label"],
                "source": call["source"],
                **best,
                "max_call_overlap_fraction": max_call_fraction,
                "max_candidate_overlap_fraction": max_candidate_fraction,
                "max_either_containment_fraction": max_either_fraction,
                "max_reciprocal_overlap_fraction": max_reciprocal,
                "any_overlap": bool(best["overlap_bp"] > 0),
                "call_coverage_gt_50": bool(max_call_fraction > args.threshold),
                "either_containment_gt_50": bool(max_either_fraction > args.threshold),
                "reciprocal_overlap_gt_50": bool(max_reciprocal > args.threshold),
            }
        )
    details = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "candidate_recall_details.tsv", sep="\t", index=False)
    metrics = ["any_overlap", "call_coverage_gt_50", "either_containment_gt_50", "reciprocal_overlap_gt_50"]
    summary_rows: list[dict] = []
    for source, frame in [("ALL", details), *list(details.groupby("source", sort=True))]:
        row = {"source": source, "n_calls": int(len(frame))}
        for metric in metrics:
            row[metric] = float(frame[metric].astype(bool).mean()) if len(frame) else float("nan")
            row[f"n_{metric}"] = int(frame[metric].astype(bool).sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "candidate_recall_summary.tsv", sep="\t", index=False)
    payload = {
        "candidate_rows": int(len(candidates)),
        "candidate_samples": int(candidates["sample_id"].nunique()),
        "external_call_rows": int(len(calls)),
        "threshold": float(args.threshold),
        "overall": summary.iloc[0].to_dict(),
    }
    (output_dir / "candidate_recall_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--external_regions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
