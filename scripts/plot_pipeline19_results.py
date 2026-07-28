#!/usr/bin/env python3
"""Create publication-ready summary figures for pipeline19 ASM-Loc results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


COLORS = {
    "navy": "#173F5F",
    "blue": "#20639B",
    "teal": "#3CAEA3",
    "yellow": "#F6D55C",
    "red": "#ED553B",
    "gray": "#7A8793",
    "light": "#EAF0F4",
    "ecDNA": "#D95F02",
    "chromothripsis": "#1B9E77",
    "BFB": "#7570B3",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220)
    fig.savefig(output_dir / f"{stem}.pdf")
    plt.close(fig)


def pipeline_overview(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = [
        (0.25, 1.65, 2.0, 1.65, "New genome", "Wakhan CNA segments\n+ Severus SV calls", COLORS["navy"]),
        (2.75, 1.65, 2.15, 1.65, "1 Mb binning", "18 cheap CNA/SV\nfeatures per bin", COLORS["blue"]),
        (5.4, 1.65, 2.25, 1.65, "ASM-Loc", "Local convolution\n+ chromosome attention", COLORS["teal"]),
        (8.15, 1.65, 2.15, 1.65, "Localization", "Foreground runs\n+ boundary refinement", COLORS["yellow"]),
        (10.8, 1.65, 2.2, 1.65, "Pipeline18 MIL", "Expensive CN/graph embedding\n+ event typing", COLORS["red"]),
    ]
    for x, y, width, height, title, body, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.1",
            facecolor=color,
            edgecolor="none",
            alpha=0.95,
        )
        ax.add_patch(patch)
        text_color = "#1B1B1B" if color == COLORS["yellow"] else "white"
        ax.text(x + width / 2, y + 1.16, title, ha="center", va="center", weight="bold", color=text_color, fontsize=11)
        ax.text(x + width / 2, y + 0.62, body, ha="center", va="center", color=text_color, fontsize=9)
    for left, right in zip(boxes[:-1], boxes[1:]):
        x1 = left[0] + left[2] + 0.08
        x2 = right[0] - 0.08
        ax.add_patch(FancyArrowPatch((x1, 2.47), (x2, 2.47), arrowstyle="-|>", mutation_scale=16, lw=1.6, color="#425466"))
    ax.text(
        6.5,
        4.35,
        "Genome-wide localization first; expensive event classification second",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color=COLORS["navy"],
    )
    ax.text(6.5, 3.88, "ASM-Loc answers WHERE. Pipeline18 answers WHETHER and WHAT.", ha="center", color=COLORS["gray"], fontsize=11)
    ax.annotate(
        "Runs on every chromosome bin",
        xy=(6.5, 1.52),
        xytext=(6.5, 0.85),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": COLORS["teal"]},
        color=COLORS["teal"],
        weight="bold",
    )
    ax.annotate(
        "Runs only on 818 proposals\n(2,970 in v3)",
        xy=(11.9, 1.52),
        xytext=(11.9, 0.65),
        ha="center",
        arrowprops={"arrowstyle": "-|>", "color": COLORS["red"]},
        color=COLORS["red"],
        weight="bold",
    )
    save(fig, output_dir, "figure1_pipeline_overview")


def overlap_metrics(proposals: pd.DataFrame, calls: pd.DataFrame) -> pd.DataFrame:
    by_key = {
        key: frame
        for key, frame in proposals.groupby(
            [proposals["sample_id"].astype(str), proposals["chrom"].astype(str)], sort=False
        )
    }
    rows = []
    for call in calls.itertuples():
        best = (0.0, 0.0, 0.0)
        for proposal in by_key.get((str(call.sample_id), str(call.chrom)), pd.DataFrame()).itertuples():
            overlap = max(0, min(int(call.end) + 1, int(proposal.end)) - max(int(call.start), int(proposal.start)))
            call_fraction = overlap / max(1, int(call.end) + 1 - int(call.start))
            proposal_fraction = overlap / max(1, int(proposal.end) - int(proposal.start))
            union = max(int(call.end) + 1, int(proposal.end)) - min(int(call.start), int(proposal.start))
            iou = overlap / max(1, union)
            score = (min(call_fraction, proposal_fraction), iou, call_fraction)
            if score > best:
                best = score
        rows.append(
            {
                "sample_id": call.sample_id,
                "label": call.label,
                "source": call.source,
                "reciprocal": best[0],
                "iou": best[1],
                "call_coverage": best[2],
            }
        )
    return pd.DataFrame(rows)


def results_dashboard(base: Path, pipeline18: Path, output_dir: Path) -> None:
    training = pd.read_csv(base / "asm_loc_model.training.tsv", sep="\t")
    split = json.loads((base / "asm_loc_model.split.json").read_text())
    proposals = pd.read_csv(base / "localized_proposals.tsv", sep="\t")
    calls = pd.read_csv(pipeline18 / "external_regions.tsv", sep="\t")
    metrics = overlap_metrics(proposals, calls)
    held_out = metrics[metrics["sample_id"].isin(split["validation_samples"])]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(training["epoch"], training["train_loss"], marker="o", ms=3, color=COLORS["blue"], label="Train")
    ax.plot(training["epoch"], training["validation_loss"], marker="o", ms=3, color=COLORS["red"], label="Held-out genomes")
    best_idx = training["validation_loss"].idxmin()
    ax.scatter(training.loc[best_idx, "epoch"], training.loc[best_idx, "validation_loss"], s=90, color=COLORS["yellow"], edgecolor="black", zorder=4)
    ax.annotate(
        f"selected epoch {int(training.loc[best_idx, 'epoch'])}\nloss {training.loc[best_idx, 'validation_loss']:.3f}",
        (training.loc[best_idx, "epoch"], training.loc[best_idx, "validation_loss"]),
        xytext=(12, 18),
        textcoords="offset points",
    )
    ax.set(title="A. Localization training", xlabel="Epoch", ylabel="Total loss")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    categories = ["Proposal count", "Call coverage ≥0.5", "Reciprocal overlap ≥0.5"]
    v3 = [2970, 0.7924528302, 0.6666666667]
    asm = [len(proposals), (metrics["call_coverage"] >= 0.5).mean(), (metrics["reciprocal"] >= 0.5).mean()]
    x = np.arange(len(categories))
    # Count is converted to percentage of v3 so the panel can share an axis.
    v3_plot = [100, v3[1] * 100, v3[2] * 100]
    asm_plot = [asm[0] / v3[0] * 100, asm[1] * 100, asm[2] * 100]
    width = 0.36
    ax.bar(x - width / 2, v3_plot, width, color=COLORS["gray"], label="v3 generator")
    ax.bar(x + width / 2, asm_plot, width, color=COLORS["teal"], label="ASM-Loc")
    ax.set_xticks(x, ["Proposal burden\n(v3 = 100%)", "Call coverage\n≥0.5", "Reciprocal overlap\n≥0.5"])
    ax.set_ylabel("Percent")
    ax.set_ylim(0, 110)
    ax.set_title("B. Coverage–boundary tradeoff")
    ax.legend(frameon=False)
    for i, values in enumerate(zip(v3_plot, asm_plot)):
        for offset, value in zip((-width / 2, width / 2), values):
            ax.text(i + offset, value + 2, f"{value:.1f}", ha="center", fontsize=9)
    ax.text(0 - width / 2, 93, "2,970", ha="center", fontsize=8, color="white", weight="bold")
    ax.text(0 + width / 2, asm_plot[0] / 2, "818", ha="center", fontsize=8, color="white", weight="bold")

    ax = axes[1, 0]
    labels = list(CLASS_ORDER)
    class_rows = []
    for label in labels:
        frame = metrics[metrics["label"] == label]
        class_rows.append(
            (
                (frame["call_coverage"] >= 0.5).mean() * 100,
                (frame["reciprocal"] >= 0.5).mean() * 100,
            )
        )
    coverage = [row[0] for row in class_rows]
    reciprocal = [row[1] for row in class_rows]
    x = np.arange(len(labels))
    ax.bar(x - width / 2, coverage, width, color=COLORS["blue"], label="Call coverage ≥0.5")
    ax.bar(x + width / 2, reciprocal, width, color=COLORS["yellow"], edgecolor="#B59520", label="Reciprocal overlap ≥0.5")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Calls (%)")
    ax.set_title("C. Localization by event class")
    ax.legend(frameon=False, loc="upper right")
    for i, (left, right) in enumerate(zip(coverage, reciprocal)):
        ax.text(i - width / 2, left + 2, f"{left:.0f}", ha="center", fontsize=9)
        ax.text(i + width / 2, right + 2, f"{right:.0f}", ha="center", fontsize=9)

    ax = axes[1, 1]
    overall = [
        (metrics["call_coverage"] >= 0.5).mean() * 100,
        (metrics["reciprocal"] >= 0.5).mean() * 100,
        (metrics["iou"] >= 0.5).mean() * 100,
    ]
    hold = [
        (held_out["call_coverage"] >= 0.5).mean() * 100,
        (held_out["reciprocal"] >= 0.5).mean() * 100,
        (held_out["iou"] >= 0.5).mean() * 100,
    ]
    x = np.arange(3)
    ax.bar(x - width / 2, overall, width, color=COLORS["navy"], label=f"All calls (n={len(metrics)})")
    ax.bar(x + width / 2, hold, width, color=COLORS["red"], label=f"Held-out genomes (n={len(held_out)})")
    ax.set_xticks(x, ["Call coverage\n≥0.5", "Reciprocal\n≥0.5", "IoU\n≥0.5"])
    ax.set_ylim(0, 108)
    ax.set_ylabel("Calls (%)")
    ax.set_title("D. Generalization to held-out genomes")
    ax.legend(frameon=False)
    for i, (left, right) in enumerate(zip(overall, hold)):
        ax.text(i - width / 2, left + 2, f"{left:.1f}", ha="center", fontsize=9)
        ax.text(i + width / 2, right + 2, f"{right:.1f}", ha="center", fontsize=9)
    fig.suptitle("Pipeline19 ASM-Loc: high proposal recall, focal boundaries remain the bottleneck", fontsize=15, weight="bold", y=1.01)
    fig.tight_layout()
    save(fig, output_dir, "figure2_training_and_localization")


CLASS_ORDER = ("ecDNA", "chromothripsis", "BFB")


def tracks(base: Path, pipeline18: Path, output_dir: Path) -> None:
    bins = pd.read_csv(base / "bin_predictions.tsv.gz", sep="\t")
    proposals = pd.read_csv(base / "localized_proposals.tsv", sep="\t")
    calls = pd.read_csv(pipeline18 / "external_regions.tsv", sep="\t")
    examples = [
        ("OMC1", "chr19", "chromothripsis", "well-localized chromosome-scale event"),
        ("HT3", "chr17", "BFB", "covered but over-expanded focal event"),
        ("CERV196", "chr6", "ecDNA", "covered but over-expanded focal event"),
    ]
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, 9), sharey=True)
    for ax, (sample, chrom, event_class, note) in zip(axes, examples):
        frame = bins[(bins["sample_id"] == sample) & (bins["chrom"] == chrom)].sort_values("start")
        x = (frame["start"] + frame["end"]) / 2 / 1e6
        ax.plot(x, frame["foreground_probability"], color=COLORS["navy"], lw=1.8, label="Foreground")
        for label in CLASS_ORDER:
            ax.plot(x, frame[f"probability_{label}"], color=COLORS[label], lw=1.0, alpha=0.8, label=label)
        local_calls = calls[(calls["sample_id"] == sample) & (calls["chrom"] == chrom)]
        local_props = proposals[(proposals["sample_id"] == sample) & (proposals["chrom"] == chrom)]
        for row in local_props.itertuples():
            ax.axvspan(row.start / 1e6, row.end / 1e6, color=COLORS["yellow"], alpha=0.13)
        for row in local_calls.itertuples():
            ax.axvspan(row.start / 1e6, row.end / 1e6, color=COLORS.get(row.label, COLORS["red"]), alpha=0.25)
            ax.text((row.start + row.end) / 2 / 1e6, 1.02, row.label, color=COLORS.get(row.label, COLORS["red"]), ha="center", va="bottom", fontsize=8, weight="bold")
        ax.axhline(0.45, color=COLORS["gray"], ls="--", lw=0.9)
        ax.set_ylim(0, 1.13)
        ax.set_ylabel("Probability")
        ax.set_title(f"{sample} {chrom}: {event_class} — {note}", loc="left", weight="bold")
        ax.grid(axis="y", alpha=0.15)
    axes[-1].set_xlabel("Chromosome position (Mb)")
    handles = [
        plt.Line2D([0], [0], color=COLORS["navy"], lw=2, label="ASM-Loc foreground"),
        *[plt.Line2D([0], [0], color=COLORS[label], lw=2, label=f"{label} score") for label in CLASS_ORDER],
        Patch(facecolor=COLORS["yellow"], alpha=0.3, label="Localized proposal"),
        Patch(facecolor=COLORS["red"], alpha=0.25, label="Known region"),
    ]
    fig.legend(handles=handles, ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.01), frameon=False)
    fig.suptitle("Held-out chromosome examples", fontsize=15, weight="bold", y=1.06)
    fig.tight_layout()
    save(fig, output_dir, "figure3_heldout_chromosome_tracks")


def typed_calls(base: Path, output_dir: Path) -> None:
    calls = pd.read_csv(base / "final_predictions" / "predicted_complex_sv.tsv", sep="\t")
    counts = calls["predicted_class"].value_counts()
    order = counts.index.tolist()
    labels = [value.replace(";", "\n+") for value in order]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    bars = ax.bar(np.arange(len(order)), counts.values, color=[COLORS.get(value, COLORS["gray"]) for value in order])
    ax.set_xticks(np.arange(len(order)), labels, rotation=25, ha="right")
    ax.set_ylabel("Called proposals")
    ax.set_title("A. Pipeline18 event typing (572 calls)")
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 5, str(value), ha="center", fontsize=9)

    ax = axes[1]
    class_groups = []
    data = []
    for label in CLASS_ORDER:
        mask = calls["predicted_classes"].astype(str).str.split(";").map(lambda values: label in values)
        if mask.any():
            class_groups.append(label)
            data.append(calls.loc[mask, "objectness_prob"].astype(float).to_numpy())
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)
    for body, label in zip(parts["bodies"], class_groups):
        body.set_facecolor(COLORS[label])
        body.set_edgecolor("none")
        body.set_alpha(0.75)
    ax.set_xticks(np.arange(1, len(class_groups) + 1), class_groups)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Pipeline18 objectness probability")
    ax.set_title("B. Classifier confidence by called class")
    fig.suptitle("Localized proposals after pipeline18 MIL classification", fontsize=15, weight="bold", y=1.02)
    fig.tight_layout()
    save(fig, output_dir, "figure4_final_typed_calls")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="../results/pipeline19_asm_loc")
    parser.add_argument("--pipeline18", default="../results/pipeline18")
    parser.add_argument("--output_dir", default="../results/pipeline19_asm_loc/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    style()
    base = Path(args.results).resolve()
    pipeline18 = Path(args.pipeline18).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_overview(output_dir)
    results_dashboard(base, pipeline18, output_dir)
    tracks(base, pipeline18, output_dir)
    typed_calls(base, output_dir)
    print(f"Wrote result figures to {output_dir}")


if __name__ == "__main__":
    main()
