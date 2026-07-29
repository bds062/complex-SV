#!/usr/bin/env python3
"""Aggregate Pipeline27 held-out predictions, metrics, plots, and insights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]


def metrics(frame: pd.DataFrame) -> dict:
    truth = frame.truth.to_numpy(dtype=bool)
    pred = frame.predicted.to_numpy(dtype=bool)
    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    return {
        "tp": tp, "fp": fp, "fn": fn, "predictions": tp + fp,
        "positives": tp + fn, "precision": precision, "recall": recall,
        "f1": f1, "f2": f2,
    }


def aggregate(split: str) -> None:
    run_dir = HERE / split / "runs"
    files = sorted(run_dir.glob("*/predictions.tsv"))
    if not files:
        raise RuntimeError(f"no predictions found under {run_dir}")
    predictions = pd.concat([pd.read_csv(path, sep="\t") for path in files], ignore_index=True)
    out = HERE / split
    predictions.to_csv(out / "oof_predictions.tsv", sep="\t", index=False)

    overall = metrics(predictions)
    per_class = pd.DataFrame([
        {"class": class_name, **metrics(predictions[predictions["class"] == class_name])}
        for class_name in CLASSES
    ])
    per_class.to_csv(out / "per_class_metrics.tsv", sep="\t", index=False)
    (out / "run_summary.json").write_text(json.dumps(overall, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    axes[0].bar(["Precision", "Recall", "F1", "F2"],
                [overall["precision"], overall["recall"], overall["f1"], overall["f2"]],
                color=["#4E79A7", "#59A14F", "#F28E2B", "#E15759"])
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Chromosome-class score")
    axes[0].set_title(f"Pipeline27 {split}: overall held-out metrics")
    for i, value in enumerate([
        overall["precision"], overall["recall"], overall["f1"], overall["f2"]
    ]):
        axes[0].text(i, value + 0.025, f"{value:.3f}", ha="center")

    x = np.arange(len(CLASSES))
    width = 0.24
    axes[1].bar(x - width, per_class.precision, width, label="Precision")
    axes[1].bar(x, per_class.recall, width, label="Recall")
    axes[1].bar(x + width, per_class.f1, width, label="F1")
    axes[1].set_xticks(x, ["BFB", "Chromothripsis", "ecDNA", "Seismic"], rotation=18)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Per-class chromosome detection")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "held_out_metrics.png", dpi=180)
    plt.close(fig)
    print(split, json.dumps(overall))


def final_insights() -> None:
    summaries = {}
    for split in ["loo", "fivefold"]:
        path = HERE / split / "run_summary.json"
        if path.exists():
            summaries[split] = json.loads(path.read_text())
    if len(summaries) < 2:
        return
    label_summary = pd.read_csv(HERE / "label_summary.tsv", sep="\t")
    lines = [
        "# Pipeline27 whole-chromosome classifier",
        "",
        "## Method",
        "",
        "- Prediction unit: one entire chromosome from one genome.",
        "- Inputs: the same frozen 1,214-dimensional CN/SV encoder representation used by Pipeline24, 37 safe chromosome-level CN/SV summaries, and three neutral zero slots replacing proposal priors.",
        "- Neural head: `1254 → 96 → 48 → 4` with LayerNorm, GELU, dropout 0.35, and independent sigmoid outputs.",
        "- No proposal generator, interval localization, overlap gate, boundary scaling, NMS, or output cap is used.",
        "- Interval labels are collapsed to a multi-hot `(sample, chromosome)` target. Multiple event classes can be positive on the same chromosome.",
        "- Unlabeled chromosomes receive 0.10 negative-loss weight. Missing classes on an annotated chromosome receive 0.25 weight; positive targets receive full weight plus inverse-square-root class balancing.",
        "- All splits are grouped by genome. Validation selects per-class thresholds by chromosome-level F2.",
        "",
        "## Label collapse",
        "",
        "| Class | Chromosome-class positives | Positive chromosomes | Positive samples | Source interval labels |",
        "|---|---:|---:|---:|---:|",
        *[
            f"| {row['label']} | {int(row['chromosome_class_positives'])} | "
            f"{int(row['positive_chromosomes'])} | {int(row['positive_samples'])} | "
            f"{int(row['source_interval_labels'])} |"
            for _, row in label_summary.iterrows()
        ],
        "",
        "## Held-out results",
        "",
        "| Evaluation | Precision | Recall | F1 | F2 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [("loo", "Leave one genome out"), ("fivefold", "Grouped five-fold")]:
        m = summaries[key]
        lines.append(
            f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {m['f2']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |"
        )
    lines += ["", "## Per-class held-out results", ""]
    for key, label in [("loo", "Leave one genome out"), ("fivefold", "Grouped five-fold")]:
        table = pd.read_csv(HERE / key / "per_class_metrics.tsv", sep="\t")
        lines += [
            f"### {label}", "",
            "| Class | Precision | Recall | F1 | F2 | TP | FP | FN |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for _, row in table.iterrows():
            lines.append(
                f"| {row['class']} | {row['precision']:.3f} | {row['recall']:.3f} | "
                f"{row['f1']:.3f} | {row['f2']:.3f} | {int(row['tp'])} | "
                f"{int(row['fp'])} | {int(row['fn'])} |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "Pipeline27 evaluates chromosome triage rather than localization. Its scores therefore must not be compared directly with Pipeline24 interval-level F1. A positive prediction means that a class is present somewhere on the chromosome; it does not recover event boundaries or distinguish multiple same-class events on that chromosome.",
        "",
        "Because the caller tables are incomplete, reported false positives include chromosomes that may contain unannotated complex events. Weak-negative weighting reduces this supervision error but does not eliminate it.",
    ]
    (HERE / "insights.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["loo", "fivefold", "all"], default="all")
    args = parser.parse_args()
    if args.split in {"loo", "all"}:
        aggregate("loo")
    if args.split in {"fivefold", "all"}:
        aggregate("fivefold")
    final_insights()
