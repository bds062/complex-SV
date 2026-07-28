#!/usr/bin/env python3
"""Visualize how label-free candidate regions relate to external caller regions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_ORDER = ["CORAL", "CervicalPanelBFB", "ShatterSeek"]
SOURCE_NAMES = {
    "CORAL": "CoRaL ecDNA",
    "CervicalPanelBFB": "Curated BFB",
    "ShatterSeek": "ShatterSeek\nchromothripsis",
}
SOURCE_COLORS = {
    "CORAL": "#2A9D8F",
    "CervicalPanelBFB": "#D1495B",
    "ShatterSeek": "#3C78B5",
}
REASON_NAMES = {
    "chromosome_sv_span": "Chromosome SV span",
    "foldback_cluster": "Foldback cluster",
    "foldback_interval": "Foldback interval",
    "high_copy_run": "High-copy run",
    "local_sv_cluster": "Local SV cluster",
    "small_cna_run": "Small-CNA run",
}


def norm_chrom(value: object) -> str:
    chrom = str(value).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    return f"chr{chrom}"


def overlap_length(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def load_inputs(candidate_path: Path, caller_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(candidate_path).fillna("")
    callers = pd.read_csv(caller_path, sep="\t").fillna("")
    candidate_required = {"sample_id", "candidate_id", "chrom", "start", "end", "proposal_reasons"}
    caller_required = {"region_id", "sample_id", "chrom", "start", "end", "label", "source"}
    for name, frame, required in [
        ("candidates", candidates, candidate_required),
        ("external regions", callers, caller_required),
    ]:
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")

    for frame in (candidates, callers):
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["chrom"] = frame["chrom"].map(norm_chrom)
        frame["start"] = pd.to_numeric(frame["start"], errors="raise").astype(np.int64)
        frame["end"] = pd.to_numeric(frame["end"], errors="raise").astype(np.int64)
        frame["length_bp"] = (frame["end"] - frame["start"] + 1).clip(lower=1)
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    callers["region_id"] = callers["region_id"].astype(str)
    return candidates, callers


def build_overlap_tables(
    candidates: pd.DataFrame,
    callers: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_groups = {
        key: frame for key, frame in candidates.groupby(["sample_id", "chrom"], sort=False)
    }
    pairs: list[dict] = []
    for _, call in callers.iterrows():
        key = (call["sample_id"], call["chrom"])
        for _, candidate in candidate_groups.get(key, pd.DataFrame()).iterrows():
            overlap = overlap_length(int(call.start), int(call.end), int(candidate.start), int(candidate.end))
            if overlap <= 0:
                continue
            call_fraction = overlap / int(call.length_bp)
            candidate_fraction = overlap / int(candidate.length_bp)
            pairs.append(
                {
                    "region_id": call.region_id,
                    "source": call.source,
                    "label": call.label,
                    "sample_id": call.sample_id,
                    "chrom": call.chrom,
                    "call_start": int(call.start),
                    "call_end": int(call.end),
                    "call_length_bp": int(call.length_bp),
                    "candidate_id": candidate.candidate_id,
                    "candidate_start": int(candidate.start),
                    "candidate_end": int(candidate.end),
                    "candidate_length_bp": int(candidate.length_bp),
                    "proposal_reasons": candidate.proposal_reasons,
                    "overlap_bp": overlap,
                    "call_overlap_fraction": call_fraction,
                    "candidate_overlap_fraction": candidate_fraction,
                    "either_containment_fraction": max(call_fraction, candidate_fraction),
                    "reciprocal_overlap_fraction": min(call_fraction, candidate_fraction),
                    "call_coverage_gt_threshold": call_fraction > threshold,
                    "candidate_coverage_gt_threshold": candidate_fraction > threshold,
                    "either_containment_gt_threshold": max(call_fraction, candidate_fraction) > threshold,
                    "reciprocal_overlap_gt_threshold": min(call_fraction, candidate_fraction) > threshold,
                }
            )
    pairs_df = pd.DataFrame(pairs)

    caller_rows: list[dict] = []
    for _, call in callers.iterrows():
        match = pairs_df[pairs_df.region_id == call.region_id] if len(pairs_df) else pd.DataFrame()
        caller_rows.append(
            {
                **call.to_dict(),
                "n_any_overlap_candidates": int(len(match)),
                "n_call_coverage_candidates": int(match.call_coverage_gt_threshold.sum()) if len(match) else 0,
                "n_candidate_coverage_candidates": int(match.candidate_coverage_gt_threshold.sum()) if len(match) else 0,
                "n_either_containment_candidates": int(match.either_containment_gt_threshold.sum()) if len(match) else 0,
                "n_reciprocal_candidates": int(match.reciprocal_overlap_gt_threshold.sum()) if len(match) else 0,
                "max_call_overlap_fraction": float(match.call_overlap_fraction.max()) if len(match) else 0.0,
                "max_candidate_overlap_fraction": float(match.candidate_overlap_fraction.max()) if len(match) else 0.0,
                "max_either_containment_fraction": float(match.either_containment_fraction.max()) if len(match) else 0.0,
                "max_reciprocal_overlap_fraction": float(match.reciprocal_overlap_fraction.max()) if len(match) else 0.0,
            }
        )
    caller_summary = pd.DataFrame(caller_rows)
    for column in [
        "any_overlap",
        "call_coverage_gt_threshold",
        "candidate_coverage_gt_threshold",
        "either_containment_gt_threshold",
        "reciprocal_overlap_gt_threshold",
    ]:
        count_column = {
            "any_overlap": "n_any_overlap_candidates",
            "call_coverage_gt_threshold": "n_call_coverage_candidates",
            "candidate_coverage_gt_threshold": "n_candidate_coverage_candidates",
            "either_containment_gt_threshold": "n_either_containment_candidates",
            "reciprocal_overlap_gt_threshold": "n_reciprocal_candidates",
        }[column]
        caller_summary[column] = caller_summary[count_column] > 0

    candidate_rows: list[dict] = []
    for _, candidate in candidates.iterrows():
        match = pairs_df[pairs_df.candidate_id == candidate.candidate_id] if len(pairs_df) else pd.DataFrame()
        production = match[match.either_containment_gt_threshold] if len(match) else pd.DataFrame()
        candidate_rows.append(
            {
                **candidate.to_dict(),
                "n_overlapping_calls": int(len(match)),
                "n_production_matches": int(len(production)),
                "any_caller_overlap": bool(len(match)),
                "production_rule_match": bool(len(production)),
                "matched_labels": ";".join(sorted(set(production.label.astype(str)))) if len(production) else "",
                "matched_sources": ";".join(sorted(set(production.source.astype(str)))) if len(production) else "",
                "max_reciprocal_overlap_fraction": float(match.reciprocal_overlap_fraction.max()) if len(match) else 0.0,
            }
        )
    candidate_summary = pd.DataFrame(candidate_rows)
    return pairs_df, caller_summary, candidate_summary


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color="#D8DEE5", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / name, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_recovery(caller_summary: pd.DataFrame, output: Path, threshold: float) -> pd.DataFrame:
    metric_info = [
        ("any_overlap", "Any overlap"),
        ("call_coverage_gt_threshold", f">{threshold:.0%} caller\ncoverage"),
        ("candidate_coverage_gt_threshold", f">{threshold:.0%} candidate\ncoverage"),
        ("either_containment_gt_threshold", f">{threshold:.0%} either\n(production rule)"),
        ("reciprocal_overlap_gt_threshold", f">{threshold:.0%} reciprocal"),
    ]
    rows = []
    for source in SOURCE_ORDER:
        frame = caller_summary[caller_summary.source == source]
        for metric, display in metric_info:
            rows.append(
                {
                    "source": source,
                    "metric": metric,
                    "metric_display": display,
                    "n_calls": len(frame),
                    "n_recovered": int(frame[metric].sum()),
                    "recovery_fraction": float(frame[metric].mean()) if len(frame) else np.nan,
                }
            )
    summary = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    x = np.arange(len(metric_info))
    width = 0.24
    for index, source in enumerate(SOURCE_ORDER):
        data = summary[summary.source == source]
        positions = x + (index - 1) * width
        bars = ax.bar(
            positions,
            data.recovery_fraction,
            width,
            label=SOURCE_NAMES[source].replace("\n", " "),
            color=SOURCE_COLORS[source],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar, (_, row) in zip(bars, data.iterrows()):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.055, bar.get_height() + 0.025),
                f"{row.n_recovered}/{row.n_calls}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=0,
            )
    ax.set_xticks(x, [display for _, display in metric_info])
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("Fraction of caller events recovered")
    ax.set_title("Candidate generator v3 recovers external caller events", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    style_axes(ax)
    save_figure(fig, output, "caller_event_recovery.png")
    return summary


def reason_summary(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, candidate in candidate_summary.iterrows():
        for reason in filter(None, str(candidate.proposal_reasons).split(";")):
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": reason,
                    "any_caller_overlap": candidate.any_caller_overlap,
                    "production_rule_match": candidate.production_rule_match,
                }
            )
    exploded = pd.DataFrame(rows)
    summary = (
        exploded.groupby("reason", as_index=False)
        .agg(
            n_candidates=("candidate_id", "nunique"),
            n_any_overlap=("any_caller_overlap", "sum"),
            n_production_match=("production_rule_match", "sum"),
        )
    )
    summary["any_overlap_fraction"] = summary.n_any_overlap / summary.n_candidates
    summary["production_match_fraction"] = summary.n_production_match / summary.n_candidates
    return summary.sort_values("production_match_fraction", ascending=True)


def plot_reason_yield(summary: pd.DataFrame, output: Path) -> None:
    labels = [REASON_NAMES.get(reason, reason) for reason in summary.reason]
    y = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.barh(y, summary.any_overlap_fraction, color="#A9C2D8", height=0.7, label="Any caller overlap")
    ax.barh(y, summary.production_match_fraction, color="#355C7D", height=0.42, label=">50% either containment")
    for i, row in enumerate(summary.itertuples()):
        ax.text(
            min(1.02, row.production_match_fraction + 0.012),
            i,
            f"{row.n_production_match}/{row.n_candidates}",
            va="center",
            fontsize=10,
        )
    ax.set_yticks(y, labels)
    ax.set_xlim(0, max(0.55, summary.any_overlap_fraction.max() + 0.12))
    ax.set_xlabel("Fraction of v3 candidates confirmed by a caller")
    ax.set_title("Caller confirmation varies by proposal heuristic", loc="left", fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    style_axes(ax, "x")
    save_figure(fig, output, "candidate_confirmation_by_heuristic.png")


def plot_multiplicity(caller_summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.3))
    rng = np.random.default_rng(17)
    values = []
    positions = []
    labels = []
    for i, source in enumerate(SOURCE_ORDER, 1):
        data = caller_summary.loc[caller_summary.source == source, "n_either_containment_candidates"].to_numpy()
        values.append(data)
        positions.append(i)
        labels.append(SOURCE_NAMES[source])
        jitter = rng.uniform(-0.12, 0.12, size=len(data))
        ax.scatter(
            np.full(len(data), i) + jitter,
            data,
            color=SOURCE_COLORS[source],
            alpha=0.65,
            s=27,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
    bp = ax.boxplot(values, positions=positions, widths=0.46, patch_artist=True, showfliers=False)
    for patch, source in zip(bp["boxes"], SOURCE_ORDER):
        patch.set_facecolor(SOURCE_COLORS[source])
        patch.set_alpha(0.18)
        patch.set_edgecolor(SOURCE_COLORS[source])
    for item in bp["medians"]:
        item.set_color("#20252B")
        item.set_linewidth(1.8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Matching v3 candidates per caller event")
    ax.set_title("A caller event often maps to multiple v3 proposals", loc="left", fontweight="bold")
    style_axes(ax)
    save_figure(fig, output, "candidates_per_caller_event.png")


def plot_sizes(candidates: pd.DataFrame, callers: pd.DataFrame, output: Path) -> None:
    groups = [candidates.length_bp.to_numpy() / 1e6]
    labels = ["All v3\ncandidates"]
    colors = ["#6C757D"]
    for source in SOURCE_ORDER:
        groups.append(callers.loc[callers.source == source, "length_bp"].to_numpy() / 1e6)
        labels.append(SOURCE_NAMES[source])
        colors.append(SOURCE_COLORS[source])
    fig, ax = plt.subplots(figsize=(10, 6.2))
    bp = ax.boxplot(groups, patch_artist=True, showfliers=False, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.68)
        patch.set_edgecolor(color)
    for median in bp["medians"]:
        median.set_color("white")
        median.set_linewidth(2)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set_yscale("log")
    ax.set_ylabel("Interval length (Mb, log scale)")
    ax.set_title("Candidate and caller interval sizes", loc="left", fontweight="bold")
    style_axes(ax)
    save_figure(fig, output, "interval_size_comparison.png")


def plot_overlap_distributions(caller_summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("max_call_overlap_fraction", "Caller interval covered"),
        ("max_candidate_overlap_fraction", "Candidate interval covered"),
        ("max_reciprocal_overlap_fraction", "Reciprocal overlap"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.1), sharey=True)
    rng = np.random.default_rng(23)
    for ax, (metric, title) in zip(axes, metrics):
        data = []
        for i, source in enumerate(SOURCE_ORDER, 1):
            values = caller_summary.loc[caller_summary.source == source, metric].to_numpy()
            data.append(values)
            ax.scatter(
                np.full(len(values), i) + rng.uniform(-0.10, 0.10, size=len(values)),
                values,
                s=16,
                color=SOURCE_COLORS[source],
                alpha=0.5,
                edgecolor="none",
            )
        bp = ax.boxplot(data, widths=0.48, showfliers=False, patch_artist=True)
        for patch, source in zip(bp["boxes"], SOURCE_ORDER):
            patch.set_facecolor(SOURCE_COLORS[source])
            patch.set_alpha(0.14)
            patch.set_edgecolor(SOURCE_COLORS[source])
        ax.axhline(0.5, color="#444444", linestyle="--", linewidth=1)
        ax.set_xticks(range(1, 4), ["ecDNA", "BFB", "Chromo"], rotation=20)
        ax.set_title(title, fontsize=12, fontweight="bold")
        style_axes(ax)
    axes[0].set_ylabel("Best overlap fraction per caller event")
    axes[0].set_ylim(-0.04, 1.04)
    fig.suptitle("Best-matching v3 candidate for each caller event", x=0.04, ha="left", fontweight="bold")
    save_figure(fig, output, "best_overlap_distributions.png")


def per_sample_summary(candidates: pd.DataFrame, callers: pd.DataFrame) -> pd.DataFrame:
    samples = sorted(set(candidates.sample_id) | set(callers.sample_id))
    result = pd.DataFrame({"sample_id": samples})
    result["v3_candidates"] = result.sample_id.map(candidates.groupby("sample_id").size()).fillna(0).astype(int)
    for source in SOURCE_ORDER:
        column = f"caller_{source}"
        counts = callers[callers.source == source].groupby("sample_id").size()
        result[column] = result.sample_id.map(counts).fillna(0).astype(int)
    result["all_caller_events"] = result[[f"caller_{source}" for source in SOURCE_ORDER]].sum(axis=1)
    return result


def plot_per_sample(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values(["all_caller_events", "v3_candidates", "sample_id"], ascending=[False, False, True])
    x = np.arange(len(ordered))
    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.2], "hspace": 0.08},
    )
    bottom = np.zeros(len(ordered))
    for source in SOURCE_ORDER:
        values = ordered[f"caller_{source}"].to_numpy()
        ax1.bar(x, values, bottom=bottom, color=SOURCE_COLORS[source], width=0.82, label=SOURCE_NAMES[source].replace("\n", " "))
        bottom += values
    ax2.bar(x, ordered.v3_candidates, color="#59636E", width=0.82)
    ax1.set_ylabel("Caller events")
    ax2.set_ylabel("v3 candidates")
    ax2.set_xticks(x, ordered.sample_id, rotation=55, ha="right", fontsize=8)
    ax1.legend(frameon=False, ncol=3, loc="upper right")
    ax1.set_title("Caller events and label-free candidates by genome", loc="left", fontweight="bold")
    style_axes(ax1)
    style_axes(ax2)
    save_figure(fig, output, "per_sample_call_and_candidate_counts.png")


def plot_summary_dashboard(
    candidates: pd.DataFrame,
    callers: pd.DataFrame,
    caller_summary: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ax = axes[0, 0]
    source_counts = callers.source.value_counts().reindex(SOURCE_ORDER)
    bars = ax.bar(range(3), source_counts, color=[SOURCE_COLORS[s] for s in SOURCE_ORDER])
    ax.set_xticks(range(3), [SOURCE_NAMES[s] for s in SOURCE_ORDER])
    ax.set_ylabel("Caller events")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{int(bar.get_height())}", ha="center")
    ax.set_title("External truth set", loc="left", fontweight="bold")
    style_axes(ax)

    ax = axes[0, 1]
    recovered = caller_summary.groupby("source").either_containment_gt_threshold.mean().reindex(SOURCE_ORDER)
    bars = ax.bar(range(3), recovered, color=[SOURCE_COLORS[s] for s in SOURCE_ORDER])
    ax.set_xticks(range(3), [SOURCE_NAMES[s] for s in SOURCE_ORDER])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Event recovery")
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{bar.get_height():.0%}", ha="center")
    ax.set_title(">50% either-containment recovery", loc="left", fontweight="bold")
    style_axes(ax)

    ax = axes[1, 0]
    candidate_counts = [
        int(candidate_summary.production_rule_match.sum()),
        int((~candidate_summary.production_rule_match).sum()),
    ]
    ax.pie(
        candidate_counts,
        labels=["Caller-confirmed", "Unmatched"],
        colors=["#4C956C", "#D9DEE3"],
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    ax.set_title("Candidate-level confirmation", loc="left", fontweight="bold")

    ax = axes[1, 1]
    text_lines = [
        f"{len(candidates):,} v3 candidate regions",
        f"{candidates.sample_id.nunique()} genomes proposed",
        f"{len(callers):,} external caller events",
        f"{callers.sample_id.nunique()} genomes with caller labels",
        "",
        f"{int(candidate_summary.production_rule_match.sum()):,} candidates match >=1 caller",
        f"{candidate_summary.production_rule_match.mean():.1%} candidate confirmation rate",
        f"{caller_summary.n_either_containment_candidates.median():.0f} median candidates per caller event",
    ]
    ax.axis("off")
    ax.text(0.02, 0.95, "Analysis snapshot", transform=ax.transAxes, va="top", fontsize=15, fontweight="bold")
    ax.text(0.02, 0.83, "\n".join(text_lines), transform=ax.transAxes, va="top", fontsize=12, linespacing=1.45)
    fig.suptitle("Candidate generator v3: relationship to caller-derived regions", x=0.03, ha="left", fontsize=18, fontweight="bold")
    save_figure(fig, output, "caller_vis_summary.png")


def run(args: argparse.Namespace) -> None:
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
        }
    )
    candidates, callers = load_inputs(Path(args.candidates), Path(args.external_regions))
    pairs, caller_summary, candidate_summary = build_overlap_tables(candidates, callers, args.threshold)

    pairs.to_csv(output / "candidate_caller_overlap_pairs.tsv", sep="\t", index=False)
    caller_summary.to_csv(output / "caller_event_matches.tsv", sep="\t", index=False)
    candidate_summary.to_csv(output / "candidate_match_status.tsv", sep="\t", index=False)

    recovery = plot_recovery(caller_summary, output, args.threshold)
    recovery.to_csv(output / "caller_event_recovery_summary.tsv", sep="\t", index=False)
    reasons = reason_summary(candidate_summary)
    reasons.to_csv(output / "candidate_confirmation_by_heuristic.tsv", sep="\t", index=False)
    plot_reason_yield(reasons, output)
    plot_multiplicity(caller_summary, output)
    plot_sizes(candidates, callers, output)
    plot_overlap_distributions(caller_summary, output)
    samples = per_sample_summary(candidates, callers)
    samples.to_csv(output / "per_sample_call_and_candidate_counts.tsv", sep="\t", index=False)
    plot_per_sample(samples, output)
    plot_summary_dashboard(candidates, callers, caller_summary, candidate_summary, output)

    summary = {
        "candidate_file": str(Path(args.candidates).resolve()),
        "external_regions_file": str(Path(args.external_regions).resolve()),
        "threshold": args.threshold,
        "candidate_rows": int(len(candidates)),
        "candidate_samples": int(candidates.sample_id.nunique()),
        "caller_events": int(len(callers)),
        "caller_samples": int(callers.sample_id.nunique()),
        "caller_events_recovered_by_production_rule": int(caller_summary.either_containment_gt_threshold.sum()),
        "caller_event_recall_by_production_rule": float(caller_summary.either_containment_gt_threshold.mean()),
        "caller_events_recovered_by_reciprocal_rule": int(caller_summary.reciprocal_overlap_gt_threshold.sum()),
        "caller_event_recall_by_reciprocal_rule": float(caller_summary.reciprocal_overlap_gt_threshold.mean()),
        "caller_confirmed_candidates": int(candidate_summary.production_rule_match.sum()),
        "candidate_confirmation_fraction": float(candidate_summary.production_rule_match.mean()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.txt").write_text(
        "Candidate generator v3 caller visualization\n\n"
        "Candidate regions are generated without caller labels. External calls are used only for this offline analysis.\n"
        f"The production matching rule is same sample/chromosome and >{args.threshold:.0%} coverage of either interval. "
        "Reciprocal overlap requires the threshold for both intervals and is reported separately.\n\n"
        "The event-level plots ask whether each external caller event was recovered. Candidate-level plots ask how many "
        "v3 proposals are externally confirmed; several v3 proposals may correspond to the same caller event. Proposal "
        "heuristics are nonexclusive because merged candidates can retain multiple reasons.\n"
    )
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--external_regions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.50)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
