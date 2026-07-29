#!/usr/bin/env python3
"""Generate model-only summary figures for the selected localization model."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
METRICS = ROOT / "metrics"
FIGURES.mkdir(parents=True, exist_ok=True)


def annotate(ax) -> None:
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)


def overall_plot(summary: dict) -> None:
    labels = ["Localization\nrecall", "Correct-class\nrecall", "Precision", "F1", "F2"]
    values = [
        summary["localization_recall"], summary["classified_recall"],
        summary["classified_precision"], summary["classified_f1"],
        summary["classified_f2"],
    ]
    colors = ["#76B7B2", "#4E79A7", "#F28E2B", "#59A14F", "#B07AA1"]
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    bars = ax.bar(labels, values, color=colors, width=0.68)
    ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=4, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 0.70)
    ax.set_ylabel("LOO held-out metric")
    ax.set_title("Selected localization model", loc="left", fontsize=15, fontweight="bold")
    ax.text(
        0, 1.02,
        f"Frozen event decoder · {summary['true_predictions']}/108 one-to-one matches · "
        f"{summary['n_predictions']} predictions",
        transform=ax.transAxes, fontsize=10, color="#52616B",
    )
    ax.grid(axis="y", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "overall_metrics.png", dpi=220)
    plt.close(fig)


def per_class_plot(frame: pd.DataFrame) -> None:
    order = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
    display = {
        "BFB": "BFB", "chromothripsis": "Chromothripsis",
        "ecDNA": "ecDNA", "seismic_amplification": "Seismic amplification",
    }
    frame = frame.set_index("label").loc[order].reset_index()
    x = np.arange(len(frame))
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    for offset, column, color in [
        (-width, "precision", "#F28E2B"),
        (0, "recall", "#4E79A7"),
        (width, "f1", "#59A14F"),
    ]:
        bars = ax.bar(x + offset, frame[column], width, label=column.capitalize(), color=color)
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in frame[column]], padding=3, fontsize=8)
    ax.set_xticks(x, [display[value] for value in frame.label])
    ax.set_ylim(0, 0.84)
    ax.set_ylabel("LOO held-out metric")
    ax.set_title("Performance by event class", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.16)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_metrics.png", dpi=220)
    plt.close(fig)


def outcome_plot(losses: pd.DataFrame) -> None:
    stage_order = [
        "proposal_miss", "event_geometry", "score_threshold", "nms",
        "output_cap", "one_to_one_collision", "recovered",
    ]
    colors = {
        "proposal_miss": "#9C755F", "event_geometry": "#BAB0AC",
        "score_threshold": "#E15759", "nms": "#F28E2B",
        "output_cap": "#EDC948", "one_to_one_collision": "#B07AA1",
        "recovered": "#59A14F",
    }
    classes = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
    display = ["BFB", "Chromothripsis", "ecDNA", "Seismic amplification"]
    pivot = losses.pivot_table(index="truth_label", columns="loss_stage", values="labels", fill_value=0)
    y = np.arange(len(classes))
    left = np.zeros(len(classes))
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for stage in stage_order:
        values = np.array([pivot.loc[label, stage] if stage in pivot.columns else 0 for label in classes])
        bars = ax.barh(y, values, left=left, color=colors[stage], edgecolor="white", label=stage.replace("_", " "))
        for bar, value in zip(bars, values, strict=True):
            if value:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + bar.get_height() / 2, str(int(value)), ha="center", va="center", fontsize=9, fontweight="bold", color="white" if stage not in {"event_geometry", "output_cap"} else "#263238")
        left += values
    ax.set_yticks(y, display)
    ax.invert_yaxis()
    ax.set_xlabel("Caller labels")
    ax.set_title("Where labels are lost", loc="left", fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.14)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "label_outcomes_by_class.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = json.loads((METRICS / "run_summary.json").read_text())
    per_class = pd.read_csv(METRICS / "per_class_metrics.tsv", sep="\t")
    losses = pd.read_csv(METRICS / "loss_by_class.tsv", sep="\t")
    overall_plot(summary)
    per_class_plot(per_class)
    outcome_plot(losses)
    print(f"Wrote summary figures to {FIGURES}")


if __name__ == "__main__":
    main()
