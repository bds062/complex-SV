#!/usr/bin/env python3
"""Generate unlabeled complex-SV candidate regions from CNA and SV data only.

The generator combines complementary, class-agnostic proposal families:
small-CNA-segment runs, focal high-copy runs, foldback intervals, adaptive SV
clusters, multi-scale CNA/SV density windows, and unusually complex chromosome
arms. External caller results are neither accepted nor read by this script.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO))

from data.candidate_features import (  # noqa: E402
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


def _strip_wakhan_bed_suffix(path: Path) -> Path:
    name = re.sub(r"_copynumbers_segments_HP[_-]?[12]\.bed$", "", path.name, flags=re.IGNORECASE)
    name = re.sub(r"_copynumbers_segments$", "", name, flags=re.IGNORECASE)
    return path.with_name(name)

def resolve_cna_vcf(wakhan_root: str | Path) -> Path:
    """Resolve a Wakhan BED root to the matching integer-CNA VCF."""
    bed_root = _strip_wakhan_bed_suffix(Path(wakhan_root))
    parts = list(bed_root.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "bed_output":
            parts[index] = "vcf_output"
            vcf_root = Path(*parts)
            return vcf_root.with_name(f"{vcf_root.name}_wakhan_cna_integers.vcf")
    raise ValueError("Wakhan root must contain a bed_output directory")


def interval_length(row: dict | pd.Series) -> int:
    return max(1, int(row["end"]) - int(row["start"]) + 1)


def reciprocal_overlap(left: dict, right: dict) -> float:
    overlap = max(0, min(int(left["end"]), int(right["end"])) - max(int(left["start"]), int(right["start"])) + 1)
    if not overlap:
        return 0.0
    return min(overlap / interval_length(left), overlap / interval_length(right))


def add_proposal(
    proposals: list[dict],
    chrom: str,
    start: int,
    end: int,
    reason: str,
    score: float,
    centromeres: pd.DataFrame,
    split_by_arm: bool = False,
) -> None:
    start, end = max(0, int(min(start, end))), int(max(start, end))
    pieces = _split_interval_by_arm(chrom, start, end, centromeres) if split_by_arm else [(_chrom_arm(chrom, start, end, centromeres), start, end)]
    for arm, piece_start, piece_end in pieces:
        if piece_start > piece_end:
            continue
        proposals.append(
            {
                "chrom": chrom,
                "arm": arm,
                "start": int(piece_start),
                "end": int(piece_end),
                "n_windows": 1,
                "proposal_reasons": reason,
                "proposal_score": float(score),
            }
        )


def cna_segment_runs(
    cna: pd.DataFrame,
    centromeres: pd.DataFrame,
    max_segment_size: int,
    min_run_segments: int,
) -> list[dict]:
    runs = find_small_segment_candidate_intervals(cna, max_segment_size=max_segment_size, centromeres=centromeres)
    proposals: list[dict] = []
    for row in runs.to_dict("records"):
        if int(row.get("n_windows", 0)) < min_run_segments:
            continue
        add_proposal(
            proposals,
            str(row["chrom"]),
            int(row["start"]),
            int(row["end"]),
            "small_cna_run",
            float(row.get("n_windows", 1)),
            centromeres,
        )
    return proposals


def high_copy_runs(
    cna: pd.DataFrame,
    centromeres: pd.DataFrame,
    sample_ploidy: float,
    high_copy_ratio: float,
    high_copy_floor: float,
    merge_gap: int,
) -> list[dict]:
    threshold = max(float(high_copy_floor), float(sample_ploidy) * float(high_copy_ratio))
    proposals: list[dict] = []
    for chrom, chrom_df in cna.groupby("chrom", sort=False):
        high = chrom_df[pd.to_numeric(chrom_df["TCN"], errors="coerce").ge(threshold)].sort_values("start")
        active: dict | None = None
        for row in high.to_dict("records"):
            if active is None or int(row["start"]) - int(active["end"]) > merge_gap:
                if active is not None:
                    add_proposal(proposals, chrom, active["start"], active["end"], "high_copy_run", active["score"], centromeres)
                active = {"start": int(row["start"]), "end": int(row["end"]), "score": float(row["TCN"]) / max(sample_ploidy, 1e-3)}
            else:
                active["end"] = max(int(active["end"]), int(row["end"]))
                active["score"] = max(float(active["score"]), float(row["TCN"]) / max(sample_ploidy, 1e-3))
        if active is not None:
            add_proposal(proposals, chrom, active["start"], active["end"], "high_copy_run", active["score"], centromeres)
    return proposals


def foldback_proposals(
    breakpoints: pd.DataFrame,
    cna: pd.DataFrame,
    centromeres: pd.DataFrame,
    single_breakend_padding: int,
) -> list[dict]:
    proposals: list[dict] = []
    foldbacks = breakpoints[breakpoints["SV_TYPE"].astype(str).eq("FB")]
    for sv_id, sv_rows in foldbacks.groupby("sv_id", sort=False):
        for chrom, chrom_rows in sv_rows.groupby("chrom", sort=False):
            positions = sorted(pd.to_numeric(chrom_rows["pos"], errors="coerce").dropna().astype(int).unique().tolist())
            if not positions:
                continue
            if len(positions) >= 2:
                start, end = positions[0], positions[-1]
            else:
                start, end = positions[0] - single_breakend_padding, positions[0] + single_breakend_padding
            add_proposal(proposals, chrom, start, end, "foldback_interval", 10.0 + len(positions), centromeres)

    # Foldbacks close on the chromosome can describe one larger amplified path.
    for chrom, chrom_rows in foldbacks.groupby("chrom", sort=False):
        positions = sorted(pd.to_numeric(chrom_rows["pos"], errors="coerce").dropna().astype(int).unique().tolist())
        if len(positions) < 2:
            continue
        active = [positions[0]]
        for position in positions[1:]:
            if position - active[-1] <= 10_000_000:
                active.append(position)
            else:
                if len(active) >= 2:
                    add_proposal(proposals, chrom, active[0], active[-1], "foldback_cluster", 8.0 + len(active), centromeres)
                active = [position]
        if len(active) >= 2:
            add_proposal(proposals, chrom, active[0], active[-1], "foldback_cluster", 8.0 + len(active), centromeres)
    return proposals


def expand_to_cna_boundaries(cna: pd.DataFrame, chrom: str, start: int, end: int) -> tuple[int, int]:
    overlap = cna[(cna["chrom"].astype(str).eq(chrom)) & (cna["start"] <= end) & (cna["end"] >= start)]
    if overlap.empty:
        return start, end
    return min(start, int(overlap["start"].min())), max(end, int(overlap["end"].max()))


def sv_cluster_proposals(
    breakpoints: pd.DataFrame,
    cna: pd.DataFrame,
    centromeres: pd.DataFrame,
    gap_sizes: list[int],
    minimum_svs: list[int],
) -> list[dict]:
    proposals: list[dict] = []
    for gap_size, min_svs in zip(gap_sizes, minimum_svs):
        for chrom, chrom_df in breakpoints.groupby("chrom", sort=False):
            ordered = chrom_df.sort_values("pos")
            clusters: list[list[dict]] = []
            active: list[dict] = []
            for row in ordered.to_dict("records"):
                if active and int(row["pos"]) - int(active[-1]["pos"]) > gap_size:
                    clusters.append(active)
                    active = []
                active.append(row)
            if active:
                clusters.append(active)
            for cluster in clusters:
                unique_svs = {str(row.get("sv_id", "")) for row in cluster if str(row.get("sv_id", ""))}
                if len(unique_svs) < min_svs:
                    continue
                start, end = min(int(row["pos"]) for row in cluster), max(int(row["pos"]) for row in cluster)
                start, end = expand_to_cna_boundaries(cna, chrom, start, end)
                score = len(unique_svs) + np.log10(max(10, end - start + 1))
                add_proposal(proposals, chrom, start, end, f"sv_cluster_{gap_size // 1_000_000}mb", score, centromeres)
    return proposals


def multiscale_density_proposals(
    cna: pd.DataFrame,
    breakpoints: pd.DataFrame,
    centromeres: pd.DataFrame,
    window_sizes: list[int],
) -> list[dict]:
    proposals: list[dict] = []
    for chrom, chrom_cna in cna.groupby("chrom", sort=False):
        chrom_cna = chrom_cna.sort_values("start")
        chrom_bp = breakpoints[breakpoints["chrom"].astype(str).eq(chrom)]
        chrom_start, chrom_end = int(chrom_cna["start"].min()), int(chrom_cna["end"].max())
        for window_size in window_sizes:
            step = max(1, window_size // 2)
            positive_windows: list[tuple[int, int, float]] = []
            for start in range(chrom_start, chrom_end + 1, step):
                end = min(chrom_end, start + window_size - 1)
                cn_overlap = chrom_cna[(chrom_cna["start"] <= end) & (chrom_cna["end"] >= start)]
                bp_overlap = chrom_bp[(chrom_bp["pos"] >= start) & (chrom_bp["pos"] <= end)]
                n_segments = int(len(cn_overlap))
                n_svs = int(bp_overlap["sv_id"].astype(str).nunique())
                n_foldbacks = int(bp_overlap["SV_TYPE"].astype(str).eq("FB").sum())
                length_mb = max(0.1, (end - start + 1) / 1_000_000)
                cn_density = n_segments / length_mb
                sv_density = n_svs / length_mb
                is_candidate = (
                    n_foldbacks >= 1
                    or (window_size <= 5_000_000 and (n_segments >= 4 or n_svs >= 4))
                    or (window_size <= 20_000_000 and (n_segments >= 7 or n_svs >= 8))
                    or (window_size > 20_000_000 and (n_segments >= 12 or n_svs >= 15))
                )
                if is_candidate:
                    positive_windows.append((start, end, cn_density + sv_density + 2 * n_foldbacks))
                if end >= chrom_end:
                    break
            if not positive_windows:
                continue
            active_start, active_end, active_score = positive_windows[0]
            for start, end, score in positive_windows[1:]:
                if start <= active_end:
                    active_end = max(active_end, end)
                    active_score = max(active_score, score)
                else:
                    add_proposal(proposals, chrom, active_start, active_end, f"density_window_{window_size // 1_000_000}mb", active_score, centromeres, split_by_arm=True)
                    active_start, active_end, active_score = start, end, score
            add_proposal(proposals, chrom, active_start, active_end, f"density_window_{window_size // 1_000_000}mb", active_score, centromeres, split_by_arm=True)
    return proposals


def complex_arm_proposals(
    cna: pd.DataFrame,
    breakpoints: pd.DataFrame,
    centromeres: pd.DataFrame,
) -> list[dict]:
    arm_rows: list[dict] = []
    for chrom, chrom_df in cna.groupby("chrom", sort=False):
        chrom_start, chrom_end = int(chrom_df["start"].min()), int(chrom_df["end"].max())
        for arm, start, end in _split_interval_by_arm(chrom, chrom_start, chrom_end, centromeres):
            cn_overlap = chrom_df[(chrom_df["start"] <= end) & (chrom_df["end"] >= start)]
            bp_overlap = breakpoints[(breakpoints["chrom"].astype(str).eq(chrom)) & (breakpoints["pos"] >= start) & (breakpoints["pos"] <= end)]
            length_mb = max(1.0, (end - start + 1) / 1_000_000)
            arm_rows.append(
                {
                    "chrom": chrom,
                    "arm": arm,
                    "start": start,
                    "end": end,
                    "n_segments": len(cn_overlap),
                    "n_svs": bp_overlap["sv_id"].astype(str).nunique(),
                    "n_foldbacks": bp_overlap["SV_TYPE"].astype(str).eq("FB").sum(),
                    "score": len(cn_overlap) / length_mb + bp_overlap["sv_id"].astype(str).nunique() / length_mb,
                }
            )
    if not arm_rows:
        return []
    arms = pd.DataFrame(arm_rows)
    score_cutoff = float(arms["score"].quantile(0.80))
    segment_cutoff = max(8.0, float(arms["n_segments"].quantile(0.75)))
    sv_cutoff = max(8.0, float(arms["n_svs"].quantile(0.75)))
    selected = arms[
        (arms["score"] >= score_cutoff)
        & ((arms["n_segments"] >= segment_cutoff) | (arms["n_svs"] >= sv_cutoff) | (arms["n_foldbacks"] >= 2))
    ]
    proposals: list[dict] = []
    for row in selected.to_dict("records"):
        add_proposal(proposals, row["chrom"], row["start"], row["end"], "complex_arm", row["score"], centromeres)
    return proposals


def merge_redundant_proposals(proposals: list[dict], threshold: float) -> pd.DataFrame:
    if not proposals:
        return pd.DataFrame(columns=["chrom", "arm", "start", "end", "n_windows", "proposal_reasons", "proposal_score"])
    parent = list(range(len(proposals)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_chrom: dict[str, list[int]] = defaultdict(list)
    for index, proposal in enumerate(proposals):
        by_chrom[str(proposal["chrom"])].append(index)
    for indices in by_chrom.values():
        for offset, left in enumerate(indices):
            for right in indices[offset + 1 :]:
                if reciprocal_overlap(proposals[left], proposals[right]) >= threshold:
                    union(left, right)
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(proposals)):
        components[find(index)].append(index)

    rows: list[dict] = []
    for members in components.values():
        group = [proposals[index] for index in members]
        start, end = min(row["start"] for row in group), max(row["end"] for row in group)
        arms = {str(row["arm"]) for row in group}
        rows.append(
            {
                "chrom": group[0]["chrom"],
                "arm": next(iter(arms)) if len(arms) == 1 else "mixed",
                "start": start,
                "end": end,
                "component_intervals": f"{start}-{end}",
                "n_windows": len(group),
                "proposal_reasons": ";".join(sorted({reason for row in group for reason in str(row["proposal_reasons"]).split(";")})),
                "proposal_score": max(float(row["proposal_score"]) for row in group),
            }
        )
    return pd.DataFrame(rows).sort_values(["chrom", "start", "end"]).reset_index(drop=True)


def add_candidate_priority(candidates: pd.DataFrame) -> pd.DataFrame:
    """Rank proposals using only CNA/SV evidence available at deployment."""
    out = candidates.copy()

    def numeric(name: str) -> pd.Series:
        values = out[name] if name in out.columns else pd.Series(0.0, index=out.index)
        return pd.to_numeric(values, errors="coerce").fillna(0.0).clip(lower=0.0)

    high_cn = sum(
        (numeric(name) for name in ["n_TCN_7_10", "n_TCN_11_20", "n_TCN_20_40", "n_TCN_gt_40"]),
        start=pd.Series(0.0, index=out.index),
    )
    n_segments = numeric("n_segments")
    n_breakpoints = numeric("n_breakpoints")
    n_foldbacks = numeric("n_FB")
    n_interchromosomal = numeric("n_interchromosomal_SV")
    oscillation = numeric("oscillating_segment_fraction")
    proposal_score = numeric("proposal_score")
    reasons = out.get("proposal_reasons", pd.Series([""] * len(out), index=out.index)).astype(str)
    high_copy_reason = reasons.str.contains(r"(?:^|;)high_copy_run(?:;|$)", regex=True).astype(float)

    out["candidate_evidence_score"] = (
        1.5 * np.log1p(n_segments)
        + 2.0 * np.log1p(n_breakpoints)
        + 2.5 * np.log1p(n_foldbacks)
        + 1.5 * np.log1p(n_interchromosomal)
        + 1.5 * np.log1p(high_cn)
        + oscillation
        + 0.5 * np.log1p(proposal_score)
        + high_copy_reason
    )
    order = out["candidate_evidence_score"].rank(method="first", ascending=False).astype(int)
    out["candidate_priority_rank"] = order
    denominator = max(1, len(out) - 1)
    out["candidate_priority_percentile"] = 1.0 - (order - 1) / denominator

    balanced = (
        n_foldbacks.ge(1)
        | n_breakpoints.ge(5)
        | n_segments.ge(5)
        | high_cn.ge(1)
        | high_copy_reason.astype(bool)
    )
    out["candidate_tier"] = np.where(balanced, "balanced", "sensitive_only")
    return out.sort_values(["candidate_priority_rank", "chrom", "start", "end"]).reset_index(drop=True)


def generate_sample(row: pd.Series, args: argparse.Namespace, centromeres: pd.DataFrame) -> pd.DataFrame:
    sample_id = safe_name(row["sample_id"])
    cna_vcf = resolve_cna_vcf(row["wakhan_root"])
    severus_vcf = Path(str(row["severus_vcf"]))
    cna = read_cna_vcf_to_dataframe(cna_vcf)
    breakpoints = get_bps(severus_vcf)
    sample_ploidy = calculate_ploidy(cna)

    proposals: list[dict] = []
    proposals.extend(cna_segment_runs(cna, centromeres, int(args.max_small_segment_size), int(args.min_small_run_segments)))
    proposals.extend(high_copy_runs(cna, centromeres, sample_ploidy, float(args.high_copy_ratio), float(args.high_copy_floor), int(args.high_copy_merge_gap)))
    proposals.extend(foldback_proposals(breakpoints, cna, centromeres, int(args.single_breakend_padding)))
    proposals.extend(sv_cluster_proposals(breakpoints, cna, centromeres, [int(x) for x in args.sv_cluster_gaps.split(",")], [int(x) for x in args.sv_cluster_min_svs.split(",")]))
    proposals.extend(multiscale_density_proposals(cna, breakpoints, centromeres, [int(x) for x in args.window_sizes.split(",")]))
    proposals.extend(complex_arm_proposals(cna, breakpoints, centromeres))
    merged = merge_redundant_proposals(proposals, float(args.proposal_reciprocal_overlap))
    if merged.empty:
        return merged

    proposal_meta = merged.set_index(["chrom", "start", "end"])[["proposal_reasons", "proposal_score"]]
    summarized = _summarize_candidate_intervals(
        merged,
        cna,
        breakpoints,
        sample_ploidy=sample_ploidy,
        apply_candidate_filter=False,
    )
    summarized = summarized.join(proposal_meta, on=["chrom", "start", "end"], rsuffix="_proposal")
    for column in CLASS_COLUMNS:
        if column in summarized.columns:
            summarized = summarized.drop(columns=column)
    summarized = assign_linked_cluster_ids(summarized, cna, breakpoints)
    summarized = add_candidate_priority(summarized)
    summarized.insert(0, "sample_id", sample_id)
    summarized.insert(1, "candidate_id", [f"{sample_id}:standalone:{index + 1:04d}" for index in range(len(summarized))])
    summarized.insert(2, "wakhan_sample_id", str(row.get("wakhan_sample_id", "")))
    summarized.insert(3, "wakhan_root", str(row["wakhan_root"]))
    summarized.insert(4, "cna_vcf", str(cna_vcf))
    summarized.insert(5, "severus_vcf", str(severus_vcf))
    summarized.insert(6, "discovery_source", "standalone_cna_sv")
    return summarized


def run(args: argparse.Namespace) -> None:
    manifest = pd.read_csv(args.manifest, sep="\t").fillna("")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = output_dir / "candidate_regions"
    sample_dir.mkdir(exist_ok=True)
    centromeres = read_centromere_bed(args.centromeres)
    sensitive_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    for sample_index, (_, row) in enumerate(manifest.iterrows(), start=1):
        sample_id = safe_name(row["sample_id"])
        print(f"[{sample_index}/{len(manifest)}] {sample_id}")
        try:
            candidates = generate_sample(row, args, centromeres)
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"  ERROR: {exc}")
            continue
        sensitive = candidates.copy()
        selected = (
            sensitive
            if args.profile == "sensitive" or sensitive.empty
            else sensitive[sensitive["candidate_tier"].eq("balanced")].copy()
        )
        if int(args.max_candidates_per_sample) > 0:
            selected = selected.nsmallest(int(args.max_candidates_per_sample), "candidate_priority_rank").copy()
        selected.to_csv(sample_dir / f"{sample_id}_candidate_regions.csv", index=False)
        sensitive.to_csv(sample_dir / f"{sample_id}_candidate_regions_sensitive.csv", index=False)
        print(f"  {len(selected)} selected / {len(sensitive)} sensitive candidates")
        sensitive_frames.append(sensitive)
        selected_frames.append(selected)

    sensitive_merged = pd.concat(sensitive_frames, ignore_index=True, sort=False) if sensitive_frames else pd.DataFrame()
    if sensitive_merged.empty:
        balanced_merged = pd.DataFrame()
    else:
        balanced_merged = sensitive_merged[sensitive_merged["candidate_tier"].eq("balanced")].copy()
    merged = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    merged.to_csv(output_dir / "merged_candidate_regions.csv", index=False)
    sensitive_merged.to_csv(output_dir / "merged_candidate_regions_sensitive.csv", index=False)
    balanced_merged.to_csv(output_dir / "merged_candidate_regions_balanced.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "failed_samples.csv", index=False)
    reason_counts: dict[str, int] = defaultdict(int)
    for value in merged.get("proposal_reasons", pd.Series(dtype=str)).astype(str):
        for reason in value.split(";"):
            if reason:
                reason_counts[reason] += 1
    summary = {
        "generator": "standalone_multiscale_cna_sv",
        "uses_external_callers": False,
        "manifest_samples": int(len(manifest)),
        "candidate_samples": int(merged["sample_id"].nunique()) if not merged.empty else 0,
        "candidate_rows": int(len(merged)),
        "failed_samples": int(len(failures)),
        "sensitive_candidate_rows": int(len(sensitive_merged)),
        "balanced_candidate_rows": int(len(balanced_merged)),
        "candidate_rows_by_reason": dict(sorted(reason_counts.items())),
        "config": vars(args),
    }
    summary["config"] = {key: str(value) if isinstance(value, Path) else value for key, value in summary["config"].items()}
    (output_dir / "candidate_generator_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--centromeres", type=Path, default=REPO / "data/grch38.cen_coord.curated.bed")
    parser.add_argument("--max_small_segment_size", type=int, default=5_000_000)
    parser.add_argument("--min_small_run_segments", type=int, default=2)
    parser.add_argument("--high_copy_ratio", type=float, default=2.0)
    parser.add_argument("--high_copy_floor", type=float, default=6.0)
    parser.add_argument("--high_copy_merge_gap", type=int, default=2_000_000)
    parser.add_argument("--single_breakend_padding", type=int, default=1_000_000)
    parser.add_argument("--sv_cluster_gaps", default="2000000,10000000")
    parser.add_argument("--sv_cluster_min_svs", default="3,6")
    parser.add_argument("--window_sizes", default="5000000,20000000,50000000")
    parser.add_argument("--proposal_reciprocal_overlap", type=float, default=0.80)
    parser.add_argument(
        "--profile",
        choices=("sensitive", "balanced"),
        default="sensitive",
        help="Select every proposal or the evidence-ranked balanced subset. Both tables are always written.",
    )
    parser.add_argument(
        "--max_candidates_per_sample",
        type=int,
        default=0,
        help="Optional cap after profile selection; 0 keeps all selected candidates.",
    )
    parser.add_argument("--keep_going", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
