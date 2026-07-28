#!/usr/bin/env python3
"""Generate unlabeled complex-SV candidates from CNA and SV inputs.

V3 keeps the small-segment/nearby-region workflow used by process_vcfs.py and
adds compact caller-free proposals for focal amplification, local SV clusters,
foldbacks, and chromosomes whose SVs are too sparse to form a focal proposal.
External caller files are never read.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gen_candidates import resolve_cna_vcf  # noqa: E402
from process_vcfs import (  # noqa: E402
    _chrom_arm,
    _split_interval_by_arm,
    _summarize_candidate_intervals,
    assign_linked_cluster_ids,
    calculate_ploidy,
    find_small_segment_candidate_intervals,
    get_bps,
    read_centromere_bed,
    read_cna_vcf_to_dataframe,
)


CLASS_COLUMNS = ["ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"]


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._") or "sample"


def add_interval(
    proposals: list[dict],
    chrom: str,
    start: int,
    end: int,
    reason: str,
    score: float,
    centromeres: pd.DataFrame,
    split_by_arm: bool = True,
) -> None:
    start, end = max(0, int(min(start, end))), int(max(start, end))
    if split_by_arm:
        pieces = _split_interval_by_arm(chrom, start, end, centromeres)
    else:
        split = _split_interval_by_arm(chrom, start, end, centromeres)
        arm = split[0][0] if len(split) == 1 else "mixed"
        pieces = [(arm, start, end)]
    for arm, piece_start, piece_end in pieces:
        if piece_start <= piece_end:
            proposals.append(
                {
                    "chrom": str(chrom),
                    "arm": str(arm),
                    "start": int(piece_start),
                    "end": int(piece_end),
                    "n_windows": 1,
                    "proposal_reasons": reason,
                    "proposal_score": float(score),
                }
            )


def small_segment_proposals(
    cna: pd.DataFrame,
    centromeres: pd.DataFrame,
    max_segment_size: int,
    min_run_segments: int,
) -> list[dict]:
    runs = find_small_segment_candidate_intervals(
        cna,
        max_segment_size=max_segment_size,
        centromeres=centromeres,
    )
    proposals: list[dict] = []
    for row in runs.to_dict("records"):
        if int(row.get("n_windows", 0)) < min_run_segments:
            continue
        proposals.append(
            {
                "chrom": str(row["chrom"]),
                "arm": str(row["arm"]),
                "start": int(row["start"]),
                "end": int(row["end"]),
                "n_windows": int(row.get("n_windows", 1)),
                "proposal_reasons": "small_cna_run",
                "proposal_score": float(row.get("n_windows", 1)),
            }
        )
    return proposals


def high_copy_proposals(
    cna: pd.DataFrame,
    centromeres: pd.DataFrame,
    sample_ploidy: float,
    copy_ratio: float,
    copy_floor: float,
    merge_gap: int,
) -> list[dict]:
    threshold = max(float(copy_floor), float(sample_ploidy) * float(copy_ratio))
    proposals: list[dict] = []
    for chrom, chrom_df in cna.groupby("chrom", sort=False):
        selected = chrom_df[pd.to_numeric(chrom_df["TCN"], errors="coerce").ge(threshold)].sort_values("start")
        active: dict | None = None
        for row in selected.to_dict("records"):
            if active is None or int(row["start"]) - int(active["end"]) > merge_gap:
                if active is not None:
                    add_interval(
                        proposals,
                        str(chrom),
                        active["start"],
                        active["end"],
                        "high_copy_run",
                        active["score"],
                        centromeres,
                    )
                active = {
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "score": float(row["TCN"]) / max(float(sample_ploidy), 1e-3),
                }
            else:
                active["end"] = max(int(active["end"]), int(row["end"]))
                active["score"] = max(float(active["score"]), float(row["TCN"]) / max(float(sample_ploidy), 1e-3))
        if active is not None:
            add_interval(
                proposals,
                str(chrom),
                active["start"],
                active["end"],
                "high_copy_run",
                active["score"],
                centromeres,
            )
    return proposals


def expand_to_cna_segments(cna: pd.DataFrame, chrom: str, start: int, end: int) -> tuple[int, int]:
    overlap = cna[
        cna["chrom"].astype(str).eq(chrom)
        & cna["start"].le(end)
        & cna["end"].ge(start)
    ]
    if overlap.empty:
        return int(start), int(end)
    return int(overlap["start"].min()), int(overlap["end"].max())


def breakpoint_cluster_proposals(
    cna: pd.DataFrame,
    breakpoints: pd.DataFrame,
    centromeres: pd.DataFrame,
    cluster_gap: int,
    min_svs: int,
) -> list[dict]:
    proposals: list[dict] = []
    for chrom, chrom_df in breakpoints.groupby("chrom", sort=False):
        ordered = chrom_df.sort_values("pos").to_dict("records")
        clusters: list[list[dict]] = []
        active: list[dict] = []
        for row in ordered:
            if active and int(row["pos"]) - int(active[-1]["pos"]) > cluster_gap:
                clusters.append(active)
                active = []
            active.append(row)
        if active:
            clusters.append(active)
        for cluster in clusters:
            sv_ids = {str(row.get("sv_id", "")) for row in cluster if str(row.get("sv_id", ""))}
            if len(sv_ids) < min_svs:
                continue
            start = min(int(row["pos"]) for row in cluster)
            end = max(int(row["pos"]) for row in cluster)
            start, end = expand_to_cna_segments(cna, str(chrom), start, end)
            add_interval(
                proposals,
                str(chrom),
                start,
                end,
                "local_sv_cluster",
                float(len(sv_ids)),
                centromeres,
            )
    return proposals


def foldback_proposals(
    cna: pd.DataFrame,
    breakpoints: pd.DataFrame,
    centromeres: pd.DataFrame,
    cluster_gap: int,
    single_padding: int,
) -> list[dict]:
    proposals: list[dict] = []
    foldbacks = breakpoints[breakpoints["SV_TYPE"].astype(str).eq("FB")]
    for _, sv_rows in foldbacks.groupby("sv_id", sort=False):
        for chrom, chrom_rows in sv_rows.groupby("chrom", sort=False):
            positions = sorted(pd.to_numeric(chrom_rows["pos"], errors="coerce").dropna().astype(int).unique())
            if not positions:
                continue
            start = positions[0] - single_padding
            end = positions[-1] + single_padding
            add_interval(
                proposals,
                str(chrom),
                start,
                end,
                "foldback_interval",
                10.0 + len(positions),
                centromeres,
                split_by_arm=False,
            )

    for chrom, chrom_rows in foldbacks.groupby("chrom", sort=False):
        positions = sorted(pd.to_numeric(chrom_rows["pos"], errors="coerce").dropna().astype(int).unique())
        if not positions:
            continue
        active = [positions[0]]
        groups: list[list[int]] = []
        for position in positions[1:]:
            if position - active[-1] <= cluster_gap:
                active.append(position)
            else:
                groups.append(active)
                active = [position]
        groups.append(active)
        for group in groups:
            if len(group) < 2:
                continue
            add_interval(
                proposals,
                str(chrom),
                group[0] - single_padding,
                group[-1] + single_padding,
                "foldback_cluster",
                8.0 + len(group),
                centromeres,
                split_by_arm=False,
            )
    return proposals


def merge_nearby_local_proposals(proposals: list[dict], max_distance: int) -> pd.DataFrame:
    columns = ["chrom", "arm", "start", "end", "component_intervals", "n_windows", "proposal_reasons", "proposal_score"]
    if not proposals:
        return pd.DataFrame(columns=columns)
    rows: list[dict] = []
    frame = pd.DataFrame(proposals).sort_values(["chrom", "arm", "start", "end"])
    for (_, _), group in frame.groupby(["chrom", "arm"], sort=False):
        active: dict | None = None
        active_reasons: set[str] = set()
        for row in group.to_dict("records"):
            if active is None or int(row["start"]) - int(active["end"]) >= max_distance:
                if active is not None:
                    active["proposal_reasons"] = ";".join(sorted(active_reasons))
                    active["component_intervals"] = f"{active['start']}-{active['end']}"
                    rows.append(active)
                active = row.copy()
                active_reasons = set(str(row["proposal_reasons"]).split(";"))
            else:
                active["start"] = min(int(active["start"]), int(row["start"]))
                active["end"] = max(int(active["end"]), int(row["end"]))
                active["n_windows"] = int(active.get("n_windows", 1)) + int(row.get("n_windows", 1))
                active["proposal_score"] = max(float(active["proposal_score"]), float(row["proposal_score"]))
                active_reasons.update(str(row["proposal_reasons"]).split(";"))
        if active is not None:
            active["proposal_reasons"] = ";".join(sorted(active_reasons))
            active["component_intervals"] = f"{active['start']}-{active['end']}"
            rows.append(active)
    return pd.DataFrame(rows, columns=columns)


def chromosome_sv_span_fallbacks(
    cna: pd.DataFrame,
    breakpoints: pd.DataFrame,
    centromeres: pd.DataFrame,
    local: pd.DataFrame,
    min_svs: int,
    mode: str,
    max_sparse_segments: int,
) -> list[dict]:
    proposals: list[dict] = []
    local_chroms = set(local["chrom"].astype(str)) if not local.empty else set()
    for chrom, chrom_df in breakpoints.groupby("chrom", sort=False):
        sv_ids = chrom_df["sv_id"].astype(str)
        n_svs = int(sv_ids[sv_ids.ne("")].nunique())
        chrom_cna = cna[cna["chrom"].astype(str).eq(str(chrom))]
        has_local = str(chrom) in local_chroms
        if n_svs < min_svs:
            continue
        if mode == "empty_chromosome" and has_local:
            continue
        if mode == "empty_or_sparse" and has_local and len(chrom_cna) > max_sparse_segments:
            continue
        start = int(pd.to_numeric(chrom_df["pos"], errors="coerce").min())
        end = int(pd.to_numeric(chrom_df["pos"], errors="coerce").max())
        start, end = expand_to_cna_segments(cna, str(chrom), start, end)
        add_interval(
            proposals,
            str(chrom),
            start,
            end,
            "chromosome_sv_span",
            float(n_svs),
            centromeres,
            split_by_arm=False,
        )
    return proposals


def add_priority(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    numeric = lambda name: pd.to_numeric(out.get(name, 0), errors="coerce").fillna(0.0)
    high_cn = sum((numeric(name) for name in ["n_TCN_7_10", "n_TCN_11_20", "n_TCN_20_40", "n_TCN_gt_40"]), start=pd.Series(0.0, index=out.index))
    out["candidate_evidence_score"] = (
        1.5 * np.log1p(numeric("n_segments"))
        + 2.0 * np.log1p(numeric("n_breakpoints"))
        + 2.5 * np.log1p(numeric("n_FB"))
        + 1.5 * np.log1p(high_cn)
        + 0.5 * np.log1p(numeric("proposal_score"))
    )
    out["candidate_priority_rank"] = out["candidate_evidence_score"].rank(method="first", ascending=False).astype(int)
    return out


def generate_sample(row: pd.Series, args: argparse.Namespace, centromeres: pd.DataFrame) -> pd.DataFrame:
    sample_id = str(row["sample_id"]).strip()
    cna_vcf = resolve_cna_vcf(row["wakhan_root"])
    severus_vcf = Path(str(row["severus_vcf"]))
    cna = read_cna_vcf_to_dataframe(cna_vcf)
    breakpoints = get_bps(severus_vcf)
    sample_ploidy = calculate_ploidy(cna)

    proposals: list[dict] = []
    proposals.extend(
        small_segment_proposals(
            cna,
            centromeres,
            int(args.max_small_segment_size),
            int(args.min_small_run_segments),
        )
    )
    proposals.extend(
        high_copy_proposals(
            cna,
            centromeres,
            sample_ploidy,
            float(args.high_copy_ratio),
            float(args.high_copy_floor),
            int(args.high_copy_merge_gap),
        )
    )
    proposals.extend(
        breakpoint_cluster_proposals(
            cna,
            breakpoints,
            centromeres,
            int(args.local_sv_cluster_gap),
            int(args.local_sv_cluster_min_svs),
        )
    )
    proposals.extend(
        foldback_proposals(
            cna,
            breakpoints,
            centromeres,
            int(args.foldback_cluster_gap),
            int(args.single_foldback_padding),
        )
    )
    local = merge_nearby_local_proposals(proposals, int(args.local_merge_distance))
    focal_reasons = {"high_copy_run", "foldback_interval", "foldback_cluster"}
    focal = pd.DataFrame(
        [proposal for proposal in proposals if str(proposal["proposal_reasons"]) in focal_reasons]
    )
    fallback = chromosome_sv_span_fallbacks(
        cna,
        breakpoints,
        centromeres,
        local,
        int(args.fallback_min_svs),
        str(args.fallback_mode),
        int(args.fallback_max_segments),
    )
    combined = pd.concat([local, focal, pd.DataFrame(fallback)], ignore_index=True, sort=False)
    if combined.empty:
        return combined

    exact_rows: list[dict] = []
    for (_, _, _), group in combined.groupby(["chrom", "start", "end"], sort=False):
        first = group.iloc[0].to_dict()
        first["n_windows"] = int(pd.to_numeric(group["n_windows"], errors="coerce").fillna(1).sum())
        first["proposal_score"] = float(pd.to_numeric(group["proposal_score"], errors="coerce").max())
        first["proposal_reasons"] = ";".join(
            sorted({reason for value in group["proposal_reasons"].astype(str) for reason in value.split(";") if reason})
        )
        first["component_intervals"] = f"{int(first['start'])}-{int(first['end'])}"
        exact_rows.append(first)
    combined = pd.DataFrame(exact_rows)
    metadata = combined.set_index(["chrom", "start", "end"])[["proposal_reasons", "proposal_score"]]
    summarized = _summarize_candidate_intervals(
        combined,
        cna,
        breakpoints,
        sample_ploidy=sample_ploidy,
        apply_candidate_filter=False,
    )
    summarized = summarized.join(metadata, on=["chrom", "start", "end"])
    summarized = summarized[pd.to_numeric(summarized["component_length"], errors="coerce").ge(int(args.min_candidate_size))].copy()
    for column in CLASS_COLUMNS:
        if column in summarized:
            summarized = summarized.drop(columns=column)
    summarized = assign_linked_cluster_ids(summarized, cna, breakpoints)
    summarized = add_priority(summarized)
    summarized = summarized.sort_values(["chrom", "start", "end"]).reset_index(drop=True)
    summarized.insert(0, "sample_id", sample_id)
    summarized.insert(1, "candidate_id", [f"{safe_name(sample_id)}:v3:{index + 1:04d}" for index in range(len(summarized))])
    summarized.insert(2, "wakhan_sample_id", str(row.get("wakhan_sample_id", "")))
    summarized.insert(3, "wakhan_root", str(row["wakhan_root"]))
    summarized.insert(4, "cna_vcf", str(cna_vcf))
    summarized.insert(5, "severus_vcf", str(severus_vcf))
    summarized.insert(6, "discovery_source", "candidate_generator_v3")
    return summarized


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, sep="\t").fillna("")
    output_dir = Path(args.output_dir)
    sample_dir = output_dir / "candidate_regions"
    sample_dir.mkdir(parents=True, exist_ok=True)
    centromeres = read_centromere_bed(args.centromeres)
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    for sample_index, (_, row) in enumerate(manifest.iterrows(), start=1):
        sample_id = str(row["sample_id"]).strip()
        print(f"[{sample_index}/{len(manifest)}] {sample_id}")
        try:
            candidates = generate_sample(row, args, centromeres)
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"  ERROR: {exc}")
            continue
        candidates.to_csv(sample_dir / f"{safe_name(sample_id)}_candidate_regions.csv", index=False)
        print(f"  {len(candidates)} candidates")
        frames.append(candidates)

    merged = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    merged.to_csv(output_dir / "merged_candidate_regions.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failed_samples.csv", index=False)
    reasons = Counter()
    for value in merged.get("proposal_reasons", pd.Series(dtype=str)).astype(str):
        reasons.update(reason for reason in value.split(";") if reason)
    summary = {
        "generator": "candidate_generator_v3_ayse_style",
        "uses_external_callers": False,
        "produces_labels": False,
        "manifest_samples": int(len(manifest)),
        "candidate_samples": int(merged["sample_id"].nunique()) if not merged.empty else 0,
        "candidate_rows": int(len(merged)),
        "failed_samples": int(len(failures)),
        "candidate_rows_by_reason": dict(sorted(reasons.items())),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (output_dir / "candidate_generator_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--centromeres", type=Path, default=Path("/data/KolmogorovLab/srinivasanbd/results/grch38.cen_coord.curated.bed"))
    parser.add_argument("--max_small_segment_size", type=int, default=5_000_000)
    parser.add_argument("--min_small_run_segments", type=int, default=2)
    parser.add_argument("--high_copy_ratio", type=float, default=2.0)
    parser.add_argument("--high_copy_floor", type=float, default=6.0)
    parser.add_argument("--high_copy_merge_gap", type=int, default=2_000_000)
    parser.add_argument("--local_sv_cluster_gap", type=int, default=2_000_000)
    parser.add_argument("--local_sv_cluster_min_svs", type=int, default=2)
    parser.add_argument("--foldback_cluster_gap", type=int, default=10_000_000)
    parser.add_argument("--single_foldback_padding", type=int, default=1_000_000)
    parser.add_argument("--local_merge_distance", type=int, default=10_000_000)
    parser.add_argument("--fallback_min_svs", type=int, default=2)
    parser.add_argument("--fallback_mode", choices=("empty_chromosome", "empty_or_sparse", "all"), default="empty_or_sparse")
    parser.add_argument("--fallback_max_segments", type=int, default=8)
    parser.add_argument("--min_candidate_size", type=int, default=10_000)
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
