#!/usr/bin/env python3
"""Plot every localization interval belonging to an FP key shared by both models."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
RESULTS = WORKSPACE / "results"
LOCALIZATION = WORKSPACE / "summer_results/localization_model"
CHROMOSOME = WORKSPACE / "summer_results/chrom_model"
OUT = HERE / "shared_false_positives"
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
HG38 = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}


def chrom_key(value: object) -> str:
    return str(value).strip().removeprefix("chr").removeprefix("CHR")


def overlap_coefficient(a0: int, a1: int, b0: int, b1: int) -> float:
    return max(0, min(a1, b1) - max(a0, b0)) / max(1, min(a1 - a0, b1 - b0))


def maximum_match(pred: pd.DataFrame, truth: pd.DataFrame) -> set[int]:
    """Return prediction IDs in a maximum one-to-one OC>=0.5 truth assignment."""
    matched: set[int] = set()
    for keys, calls in truth.groupby(["sample_id", "chrom", "label"], sort=False):
        possible = pred
        for col, value in zip(["sample_id", "chrom", "label"], keys):
            possible = possible[possible[col].astype(str) == str(value)]
        if possible.empty:
            continue
        possible = possible.sort_values("score", ascending=False).reset_index(drop=True)
        call_rows = list(calls.itertuples())
        edges = []
        for p in possible.itertuples():
            edges.append([
                j for j, call in enumerate(call_rows)
                if overlap_coefficient(
                    int(p.start), int(p.end) + 1, int(call.start), int(call.end) + 1
                ) >= 0.5
            ])
        owner = [-1] * len(call_rows)

        def augment(i: int, seen: list[bool]) -> bool:
            for j in edges[i]:
                if seen[j]:
                    continue
                seen[j] = True
                if owner[j] < 0 or augment(owner[j], seen):
                    owner[j] = i
                    return True
            return False

        for i in range(len(possible)):
            augment(i, [False] * len(call_rows))
        matched.update(int(possible.iloc[i]["_prediction_id"]) for i in owner if i >= 0)
    return matched


def build_plot_table() -> pd.DataFrame:
    pred = pd.read_csv(LOCALIZATION / "predictions/loo_predictions.tsv", sep="\t")
    pred["_prediction_id"] = np.arange(len(pred), dtype=int)
    truth = pd.read_csv(LOCALIZATION / "labels/all_labels.tsv", sep="\t")
    pred["is_true_positive"] = pred["_prediction_id"].isin(maximum_match(pred, truth))
    fp = pred[~pred["is_true_positive"]].copy()

    shared = pd.read_csv(HERE / "shared_false_positive_chromosome_classes.tsv", sep="\t")
    shared["_shared"] = True
    selected = fp.merge(
        shared,
        left_on=["sample_id", "chrom", "label"],
        right_on=["sample_id", "chrom", "class"],
        how="inner",
        validate="many_to_one",
    )

    chromosome_scores = pd.read_csv(HERE / "shared_false_positive_scores.tsv", sep="\t")
    selected = selected.merge(
        chromosome_scores[["sample_id", "chrom", "class", "chromosome_probability"]],
        on=["sample_id", "chrom", "class"],
        how="left",
        validate="many_to_one",
    )
    selected = selected.sort_values(["label", "score"], ascending=[True, False]).reset_index(drop=True)
    selected["class_fp_rank"] = selected.groupby("label").cumcount() + 1
    selected["estimate_number_for_key"] = (
        selected.groupby(["sample_id", "chrom", "label"]).cumcount() + 1
    )
    selected["estimates_for_key"] = selected.groupby(
        ["sample_id", "chrom", "label"]
    )["event_id"].transform("size")

    true_classes = (
        truth.groupby(["sample_id", "chrom"])["label"]
        .agg(lambda values: ";".join(sorted(set(map(str, values)))))
        .rename("true_classes")
        .reset_index()
    )
    selected = selected.merge(true_classes, on=["sample_id", "chrom"], how="left")
    selected["true_classes"] = selected["true_classes"].fillna("")

    rows = []
    for r in selected.itertuples(index=False):
        chrom_end = HG38[chrom_key(r.chrom)]
        rows.append({
            "candidate_id": r.event_id,
            "sample_id": r.sample_id,
            "chrom": r.chrom,
            "arm": "",
            "start_bp": 0,
            "end_bp": chrom_end,
            "context_start_bp": 0,
            "context_end_bp": chrom_end,
            "highlight_start_bp": max(0, int(r.start)),
            "highlight_end_bp": min(chrom_end, int(r.end)),
            "highlight_label": "Localization-model estimated interval",
            "predicted_class": r.label,
            "predicted_classes": r.label,
            "called_complex_sv": True,
            "is_labeled": bool(r.true_classes),
            "sv_class": r.true_classes,
            "true_classes": r.true_classes,
            "evidence": "shared_false_positive",
            "split": "LOO",
            "score": r.score,
            "objectness_prob": r.score,
            "score_text": (
                f"Localization score={r.score:.3f}; chromosome P={r.chromosome_probability:.3f}; "
                f"shared-FP class rank={r.class_fp_rank}; localized estimate "
                f"{r.estimate_number_for_key}/{r.estimates_for_key}"
            ),
            "fp_rank": r.class_fp_rank,
            "estimate_number_for_key": r.estimate_number_for_key,
            "estimates_for_key": r.estimates_for_key,
            "chromosome_probability": r.chromosome_probability,
        })
    return pd.DataFrame(rows)


def prefix_ranked_names(plot: pd.DataFrame) -> None:
    for r in plot.itertuples(index=False):
        class_dir = OUT / str(r.predicted_class)
        pattern = (
            f"{r.sample_id}_{r.chrom}_{r.predicted_class}_obj*"
            f"_{int(r.highlight_start_bp)}_{int(r.highlight_end_bp)}.png"
        )
        matches = [
            path for path in class_dir.glob(pattern)
            if not path.name.split("_", 1)[0].isdigit()
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one unranked plot for {pattern}; found {matches}")
        source = matches[0]
        source.rename(source.with_name(f"{int(r.fp_rank)}_{source.name}"))


def write_readme(plot: pd.DataFrame) -> None:
    counts = plot.groupby("predicted_class").size().reindex(CLASSES, fill_value=0)
    key_counts = (
        plot.drop_duplicates(["sample_id", "chrom", "predicted_class"])
        .groupby("predicted_class").size().reindex(CLASSES, fill_value=0)
    )
    multi = (
        plot[plot["estimates_for_key"] > 1]
        .drop_duplicates(["sample_id", "chrom", "predicted_class"])
    )
    lines = [
        "# Shared false-positive localization gallery",
        "",
        "Each image spans the complete chromosome and highlights only one localization-model "
        "interval. A shared false positive means that both models called the same "
        "sample/chromosome/class key and that key is false positive under the observed labels.",
        "",
        "Images are ranked within class by the held-out localization score. When one shared key "
        "has multiple localized FP intervals, each estimate is rendered as a separate image.",
        "",
        "| Class | Shared keys | Localized estimates plotted |",
        "|---|---:|---:|",
    ]
    for cls in CLASSES:
        lines.append(f"| {cls} | {int(key_counts[cls])} | {int(counts[cls])} |")
    lines.extend(["", f"Shared keys with multiple localized estimates: {len(multi)}.", ""])
    (OUT / "README.md").write_text("\n".join(lines))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    plot = build_plot_table()
    plot_path = OUT / "plot_input.tsv"
    plot.to_csv(plot_path, sep="\t", index=False)
    subprocess.run([
        sys.executable,
        str(WORKSPACE / "complex-SV/discovery/plot_predicted_chromosomes.py"),
        "--manifest", str(RESULTS / "pipeline18/complex_sv_manifest.tsv"),
        "--prototype_distances", str(plot_path),
        "--output_dir", str(OUT),
        "--plot_scope", "all",
        "--group_by_column", "predicted_class",
        "--dpi", "160",
        "--centromeres", str(RESULTS / "grch38.cen_coord.curated.bed"),
    ], check=True)
    prefix_ranked_names(plot)
    write_readme(plot)
    print(
        plot.groupby("predicted_class")
        .agg(shared_keys=("candidate_id", "count"), unique_keys=("sample_id", "count"))
        .to_string()
    )


if __name__ == "__main__":
    main()
