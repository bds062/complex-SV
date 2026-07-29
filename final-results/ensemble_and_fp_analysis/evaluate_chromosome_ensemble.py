#!/usr/bin/env python3
"""Leakage-safe chromosome-level ensemble evaluation for the two final models."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import OneHotEncoder


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
CHROM = WORKSPACE / "summer_results/chrom_model"
LOC = WORKSPACE / "summer_results/localization_model"
OUT = HERE / "chromosome_level_ensemble"
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
DISPLAY = {
    "BFB": "BFB",
    "chromothripsis": "Chromothripsis",
    "ecDNA": "ecDNA",
    "seismic_amplification": "Seismic",
}


def metrics(y: pd.Series, pred: pd.Series) -> dict[str, float | int]:
    yv = y.astype(int).to_numpy()
    pv = pred.astype(int).to_numpy()
    tp = int(((yv == 1) & (pv == 1)).sum())
    fp = int(((yv == 0) & (pv == 1)).sum())
    fn = int(((yv == 1) & (pv == 0)).sum())
    tn = int(((yv == 0) & (pv == 0)).sum())
    precision, recall, f1, _ = precision_recall_fscore_support(
        yv, pv, average="binary", zero_division=0
    )
    f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "f2": f2,
        "accuracy": (tp + tn) / max(tp + fp + fn + tn, 1),
    }


def best_threshold(y: np.ndarray, score: np.ndarray) -> float:
    """Select the F1-optimal observed threshold in O(n log n)."""
    order = np.argsort(-score, kind="stable")
    sorted_score = score[order]
    sorted_y = y[order].astype(int)
    tp = np.cumsum(sorted_y)
    fp = np.cumsum(1 - sorted_y)
    positives = max(int(sorted_y.sum()), 1)
    recall = tp / positives
    precision = tp / np.maximum(tp + fp, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    endpoints = np.r_[sorted_score[:-1] != sorted_score[1:], True]
    candidates = np.flatnonzero(endpoints)
    best = max(candidates, key=lambda i: (f1[i], recall[i], sorted_score[i]))
    return float(sorted_score[best])


def make_table() -> pd.DataFrame:
    base = pd.read_csv(CHROM / "predictions/loo_oof_predictions.tsv", sep="\t")
    loc = pd.read_csv(LOC / "predictions/loo_predictions.tsv", sep="\t")
    loc = (
        loc.groupby(["sample_id", "chrom", "label"], as_index=False)["score"]
        .max().rename(columns={"label": "class", "score": "localization_score"})
    )
    base = base.merge(
        loc, on=["sample_id", "chrom", "class"], how="left", validate="one_to_one"
    )
    base["localization_score"] = base["localization_score"].fillna(0.0)
    base["localization_call"] = (base["localization_score"] > 0).astype(int)
    base["chromosome_call"] = base["predicted"].astype(int)
    base["chromosome_margin"] = base["probability"] - base["threshold"]
    base["or_call"] = base[["chromosome_call", "localization_call"]].max(axis=1)
    base["and_call"] = base[["chromosome_call", "localization_call"]].min(axis=1)
    return base


def cross_fitted_stack(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Train a meta-model on other genomes and calibrate its threshold there."""
    class_encoder = OneHotEncoder(
        categories=[CLASSES], sparse_output=False, handle_unknown="ignore"
    )
    class_encoder.fit(np.asarray(CLASSES).reshape(-1, 1))
    stacked_score = np.zeros(len(table), dtype=float)
    stacked_call = np.zeros(len(table), dtype=int)
    for sample in table["sample_id"].unique():
        test_mask = table["sample_id"].eq(sample).to_numpy()
        train = table.loc[~test_mask]
        test = table.loc[test_mask]

        def features(frame: pd.DataFrame) -> np.ndarray:
            class_features = class_encoder.transform(frame[["class"]].to_numpy())
            numeric = frame[[
                "probability", "chromosome_margin",
                "localization_score", "localization_call",
            ]].to_numpy()
            return np.c_[numeric, class_features]

        model = LogisticRegression(
            C=0.25, class_weight="balanced", max_iter=2000, solver="liblinear"
        )
        model.fit(features(train), train["truth"].to_numpy())
        train_score = model.predict_proba(features(train))[:, 1]
        threshold = best_threshold(train["truth"].to_numpy(), train_score)
        test_score = model.predict_proba(features(test))[:, 1]
        stacked_score[test_mask] = test_score
        stacked_call[test_mask] = (test_score >= threshold).astype(int)
    return stacked_score, stacked_call


def evaluate(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table["stacked_score"], table["stacked_call"] = cross_fitted_stack(table)
    methods = {
        "Chromosome model": "chromosome_call",
        "Localization→chromosome": "localization_call",
        "OR ensemble": "or_call",
        "AND ensemble": "and_call",
        "LOO stacked ensemble": "stacked_call",
    }
    overall = []
    per_class = []
    for method, column in methods.items():
        overall.append({"method": method, **metrics(table["truth"], table[column])})
        for cls, frame in table.groupby("class", sort=False):
            per_class.append({
                "method": method, "class": cls,
                **metrics(frame["truth"], frame[column]),
            })
    return pd.DataFrame(overall), pd.DataFrame(per_class)


def plot_metrics(overall: pd.DataFrame) -> None:
    order = [
        "Chromosome model", "Localization→chromosome",
        "OR ensemble", "AND ensemble", "LOO stacked ensemble",
    ]
    frame = overall.set_index("method").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    x = np.arange(len(frame))
    width = 0.24
    for offset, metric, color in [
        (-width, "precision", "#4E79A7"),
        (0, "recall", "#F28E2B"),
        (width, "f1", "#59A14F"),
    ]:
        axes[0].bar(x + offset, frame[metric], width, label=metric.title(), color=color)
    axes[0].set_xticks(x, frame.index, rotation=18, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Chromosome-level held-out performance", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.2)

    fp = frame["fp"].to_numpy()
    fn = frame["fn"].to_numpy()
    axes[1].bar(x, fp, label="False positives", color="#E15759")
    axes[1].bar(x, fn, bottom=fp, label="False negatives", color="#B07AA1")
    axes[1].set_xticks(x, frame.index, rotation=18, ha="right")
    axes[1].set_ylabel("Errors")
    axes[1].set_title("Error tradeoff", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_comparison.png", dpi=210, bbox_inches="tight")
    plt.close(fig)


def write_summary(overall: pd.DataFrame, per_class: pd.DataFrame) -> None:
    best = overall.sort_values("f1", ascending=False).iloc[0]
    chromosome = overall[overall["method"] == "Chromosome model"].iloc[0]
    lines = [
        "# Chromosome-level ensemble experiment",
        "",
        "Both final models were reduced to the same sample/chromosome/class unit. The "
        "localization model contributes a positive chromosome call when at least one "
        "localized event of that class survives its own held-out decoder.",
        "",
        "The OR and AND rules use the models' original held-out calls. The stacked "
        "ensemble is leakage-safe at the meta-model level: for every held-out genome, "
        "a regularized logistic combiner and its F1 threshold are fitted using only "
        "the other genomes' out-of-fold predictions.",
        "",
        "| Method | Precision | Recall | F1 | F2 | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in overall.itertuples(index=False):
        lines.append(
            f"| {r.method} | {r.precision:.3f} | {r.recall:.3f} | {r.f1:.3f} | "
            f"{r.f2:.3f} | {r.tp} | {r.fp} | {r.fn} |"
        )
    delta = best.f1 - chromosome.f1
    lines.extend([
        "",
        f"Best observed method: **{best.method}** (F1={best.f1:.3f}; "
        f"change versus chromosome model={delta:+.3f}).",
        "",
        "Accuracy is retained in the TSV but is not emphasized because the large number "
        "of negative chromosome/class combinations makes it misleadingly high.",
    ])
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    table = make_table()
    overall, per_class = evaluate(table)
    table.to_csv(OUT / "ensemble_predictions.tsv", sep="\t", index=False)
    overall.to_csv(OUT / "overall_metrics.tsv", sep="\t", index=False)
    per_class.to_csv(OUT / "per_class_metrics.tsv", sep="\t", index=False)
    plot_metrics(overall)
    write_summary(overall, per_class)
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
