#!/usr/bin/env python3
"""Audit whether BFBArchitect fold-back coordinates exist in manifest Severus VCFs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from process_vcfs import get_bps  # noqa: E402


COORDINATE_RE = re.compile(r"(chr[^:;]+):(\d+)")


def parse_coordinates(value: object) -> list[tuple[str, int]]:
    return [(chrom, int(position)) for chrom, position in COORDINATE_RE.findall(str(value))]


def chrom_key(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, sep="\t").fillna("")
    calls = pd.read_csv(args.bfb_calls, sep="\t").fillna("")
    severus_by_sample = dict(zip(manifest["sample_id"].astype(str), manifest["severus_vcf"].astype(str)))
    breakpoint_cache: dict[str, pd.DataFrame] = {}
    detail_rows: list[dict] = []

    for call in calls.to_dict("records"):
        sample_id = str(call["sample_id"])
        severus_path = severus_by_sample.get(sample_id, "")
        if sample_id not in breakpoint_cache:
            breakpoint_cache[sample_id] = get_bps(severus_path) if severus_path and Path(severus_path).exists() else pd.DataFrame()
        breakpoints = breakpoint_cache[sample_id]
        coordinates = parse_coordinates(call.get("fold_back_coords", ""))
        coordinate_distances: list[float] = []
        coordinate_fb_distances: list[float] = []
        for chrom, position in coordinates:
            chrom_rows = breakpoints[breakpoints["chrom"].astype(str).map(chrom_key).eq(chrom_key(chrom))] if not breakpoints.empty else pd.DataFrame()
            all_positions = pd.to_numeric(chrom_rows.get("pos", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy()
            fb_positions = pd.to_numeric(
                chrom_rows.loc[chrom_rows.get("SV_TYPE", pd.Series(dtype=str)).astype(str).eq("FB"), "pos"]
                if not chrom_rows.empty and "SV_TYPE" in chrom_rows else pd.Series(dtype=float),
                errors="coerce",
            ).dropna().to_numpy()
            coordinate_distances.append(float(np.min(np.abs(all_positions - position))) if all_positions.size else float("inf"))
            coordinate_fb_distances.append(float(np.min(np.abs(fb_positions - position))) if fb_positions.size else float("inf"))

        row = {
            "sample_id": sample_id,
            "call_id": call.get("call_id", ""),
            "chrom": call.get("chrom", ""),
            "amplified_region": call.get("amplified_region", ""),
            "fold_back_coords": call.get("fold_back_coords", ""),
            "severus_vcf": severus_path,
            "n_fold_back_coords": len(coordinates),
            "n_manifest_severus_breakpoints": int(len(breakpoints)),
            "n_manifest_severus_foldbacks": int(breakpoints["SV_TYPE"].astype(str).eq("FB").sum()) if not breakpoints.empty else 0,
            "nearest_breakpoint_bp": min(coordinate_distances, default=float("inf")),
            "nearest_foldback_bp": min(coordinate_fb_distances, default=float("inf")),
        }
        for distance in args.distances:
            row[f"any_coord_within_{distance}bp"] = any(value <= distance for value in coordinate_distances)
            row[f"all_coords_within_{distance}bp"] = bool(coordinate_distances) and all(value <= distance for value in coordinate_distances)
            row[f"any_foldback_within_{distance}bp"] = any(value <= distance for value in coordinate_fb_distances)
        detail_rows.append(row)

    details = pd.DataFrame(detail_rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(output_dir / "bfbarchitect_severus_concordance.tsv", sep="\t", index=False)

    summary_rows: list[dict] = []
    for sample_id, frame in [("ALL", details), *list(details.groupby("sample_id", sort=True))]:
        row = {"sample_id": sample_id, "n_bfb_calls": int(len(frame))}
        for distance in args.distances:
            for prefix in ["any_coord", "all_coords", "any_foldback"]:
                column = f"{prefix}_within_{distance}bp"
                row[column] = float(frame[column].mean()) if len(frame) else float("nan")
                row[f"n_{column}"] = int(frame[column].sum())
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "bfbarchitect_severus_concordance_summary.tsv", sep="\t", index=False)
    payload = {"n_calls": int(len(details)), "n_samples": int(details["sample_id"].nunique()), "overall": summary.iloc[0].to_dict()}
    (output_dir / "bfbarchitect_severus_concordance_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(summary.head(1).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--bfb_calls", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--distances", default="0,10000,100000,1000000")
    args = parser.parse_args()
    args.distances = [int(value) for value in str(args.distances).split(",") if value.strip()]
    return args


if __name__ == "__main__":
    run(parse_args())
