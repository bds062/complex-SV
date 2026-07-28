"""Grouped cross-fold evaluation for the two-centroid few-shot candidate-region model.

This reuses the caller-supervised embedding corpus and exact grouped folds from
the neural-head study. Each class uses one only-class centroid and one
contains-class centroid; threshold calibration uses training rows only.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.sweep_candidate_region_general_splits import (  # noqa: E402
    calibrate_and_annotate,
    load_source,
    make_model_args,
    summarize_run,
    write_split_table,
)
from training.train_candidate_region_classifier import (  # noqa: E402
    _label_background_masks,
    _multi_hot_targets,
    _split_csv,
    apply_cluster_aggregation,
    metric_tables,
    predictions_to_distance_table,
)
from training.train_candidate_region_fewshot_classifier import (  # noqa: E402
    build_prototypes,
    make_prototype_features,
    predict_with_prototypes,
    prototype_arrays,
    prototype_table,
)
from discovery import embed_corpus  # noqa: E402

log = logging.getLogger(__name__)

CALLER_LABELS = {
    "ecDNA": "CORAL ecDNA",
    "Seismic_Amplification": "Shatterseek\nSeismic Amplification",
    "chromothripsis": "Shatterseek\nChromothripsis",
    "BFB": "BFBArchitect BFB",
}
COLORS = {
    "ecDNA": "#59A14F",
    "Seismic_Amplification": "#F28E2B",
    "chromothripsis": "#4E79A7",
    "BFB": "#E15759",
    "precision": "#4E79A7",
    "recall": "#59A14F",
    "f1": "#E15759",
    "tp": "#59A14F",
    "fp": "#E15759",
    "fn": "#B07AA1",
}


POSTER_FONT = {
    "suptitle": 22,
    "title": 18,
    "axis": 15,
    "tick": 12,
    "legend": 12,
    "annotation": 11,
    "heatmap_annotation": 10,
}

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def style_poster_axis(ax: plt.Axes) -> None:
    ax.tick_params(axis="both", labelsize=POSTER_FONT["tick"], width=1.2, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)


def save_poster_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "fold"


def load_manifest_samples(path: str | Path) -> list[str]:
    df = pd.read_csv(path, sep="\t").fillna("")
    if "sample_id" not in df.columns:
        raise ValueError(f"Manifest missing sample_id column: {path}")
    return sorted(df["sample_id"].astype(str).unique().tolist())


def split_classes(value: object) -> set[str]:
    text = str(value or "").strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return set()
    out: set[str] = set()
    for part in text.replace(",", ";").split(";"):
        token = part.strip()
        if not token:
            continue
        base = token.split(":", 1)[0].strip()
        if base:
            out.add(base)
    return out


def sample_support_table(
    manifest_samples: list[str],
    metadata: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    labels_by_row = [split_classes(v) for v in metadata.get("sv_classes", pd.Series([""] * len(metadata))).tolist()]
    labels_series = pd.Series(labels_by_row, index=metadata.index)
    for sample in manifest_samples:
        mask = metadata["sample_id"].astype(str).eq(sample) if "sample_id" in metadata else pd.Series([False] * len(metadata))
        row: dict[str, Any] = {
            "sample_id": sample,
            "n_candidates": int(mask.sum()),
            "n_labeled": int(sum(bool(x) for x in labels_series.loc[mask].tolist())) if bool(mask.any()) else 0,
            "n_empty": int(mask.sum()) - int(sum(bool(x) for x in labels_series.loc[mask].tolist())) if bool(mask.any()) else 0,
        }
        for class_name in class_names:
            row[f"support_{class_name}"] = int(sum(class_name in x for x in labels_series.loc[mask].tolist())) if bool(mask.any()) else 0
        rows.append(row)
    return pd.DataFrame(rows)


def make_balanced_folds(sample_support: pd.DataFrame, class_names: list[str], fold_size: int, seed: int) -> pd.DataFrame:
    samples = sample_support.copy().reset_index(drop=True)
    n_samples = len(samples)
    n_folds = int(math.ceil(n_samples / int(fold_size)))
    full_folds = n_samples - (n_folds - 1) * int(fold_size)
    if full_folds <= 0:
        capacities = [int(fold_size)] * n_folds
        capacities[-1] = n_samples - int(fold_size) * (n_folds - 1)
    else:
        # Prefer most folds of fold_size, with a smaller final fold when needed.
        capacities = [int(fold_size)] * (n_folds - 1) + [full_folds]
    capacities = [cap for cap in capacities if cap > 0]
    n_folds = len(capacities)

    support_cols = [f"support_{c}" for c in class_names]
    class_totals = samples[support_cols].sum(axis=0).replace(0, np.nan)
    weights = 1.0 / class_totals
    weights = weights.fillna(0.0)
    samples["label_weight"] = samples[support_cols].mul(weights, axis=1).sum(axis=1)
    samples["has_candidate"] = samples["n_candidates"].astype(int).gt(0).astype(int)
    rng = np.random.default_rng(int(seed))
    samples["jitter"] = rng.random(len(samples)) * 1e-6
    samples = samples.sort_values(
        ["label_weight", "n_labeled", "has_candidate", "n_candidates", "jitter"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    fold_rows: list[list[dict[str, Any]]] = [[] for _ in range(n_folds)]
    fold_support = np.zeros((n_folds, len(class_names)), dtype=float)
    fold_labeled = np.zeros(n_folds, dtype=float)
    fold_candidates = np.zeros(n_folds, dtype=float)
    for _, row in samples.iterrows():
        available = [i for i in range(n_folds) if len(fold_rows[i]) < capacities[i]]
        if not available:
            raise RuntimeError("No available fold capacity left")
        row_support = row[support_cols].astype(float).to_numpy()
        scores: list[tuple[float, float, float, int]] = []
        for fold_i in available:
            # Minimize rare-label burden first, then total labeled/candidates.
            projected = fold_support[fold_i] + row_support
            rare_score = float(np.sum(projected * weights.to_numpy()))
            scores.append((rare_score, float(fold_labeled[fold_i]), float(fold_candidates[fold_i]), fold_i))
        fold_i = min(scores)[-1]
        fold_rows[fold_i].append(row.to_dict())
        fold_support[fold_i] += row_support
        fold_labeled[fold_i] += float(row["n_labeled"])
        fold_candidates[fold_i] += float(row["n_candidates"])

    assignments: list[dict[str, Any]] = []
    for fold_i, rows in enumerate(fold_rows, start=1):
        for row in rows:
            out = dict(row)
            out["fold_id"] = fold_i
            out["fold_name"] = f"fold_{fold_i:02d}"
            assignments.append(out)
    out_df = pd.DataFrame(assignments).sort_values(["fold_id", "sample_id"]).reset_index(drop=True)
    return out_df.drop(columns=[c for c in ["label_weight", "has_candidate", "jitter"] if c in out_df.columns])


def class_metrics_from_predictions(predictions: pd.DataFrame, class_names: list[str], split_name: str = "cross_fold") -> pd.DataFrame:
    true_sets = [split_classes(v) for v in predictions.get("true_classes", pd.Series([""] * len(predictions))).tolist()]
    pred_sets = [split_classes(v) for v in predictions.get("predicted_classes", pd.Series([""] * len(predictions))).tolist()]
    rows: list[dict[str, Any]] = []
    for class_name in class_names:
        y = np.asarray([class_name in s for s in true_sets], dtype=bool)
        p = np.asarray([class_name in s for s in pred_sets], dtype=bool)
        tp = int((y & p).sum())
        fp = int((~y & p).sum())
        fn = int((y & ~p).sum())
        tn = int((~y & ~p).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "split": split_name,
                "class_name": class_name,
                "caller_label": CALLER_LABELS.get(class_name, class_name).replace("\n", " "),
                "support": int(y.sum()),
                "predicted": int(p.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    return pd.DataFrame(rows)


def add_fold_metric_aggregates(aggregate: pd.DataFrame, fold_per_class: pd.DataFrame) -> pd.DataFrame:
    test_pc = fold_per_class[(fold_per_class["split"].astype(str) == "test") & (fold_per_class["status"].astype(str) == "ok")].copy()
    if test_pc.empty:
        return aggregate
    grouped = test_pc.groupby("class_name", as_index=False).agg(
        fold_support_mean=("support", "mean"),
        folds_with_support=("support", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        fold_precision_mean=("precision", "mean"),
        fold_precision_std=("precision", "std"),
        fold_recall_mean=("recall", "mean"),
        fold_recall_std=("recall", "std"),
        fold_f1_mean=("f1", "mean"),
        fold_f1_std=("f1", "std"),
    )
    return aggregate.merge(grouped, on="class_name", how="left")


def plot_cross_fold_per_class_metrics(fold_per_class: pd.DataFrame, output_path: Path) -> None:
    test = fold_per_class[
        (fold_per_class["split"].astype(str) == "test")
        & (fold_per_class["status"].astype(str) == "ok")
    ].copy()
    if test.empty:
        return
    classes = list(dict.fromkeys(test["class_name"].astype(str).tolist()))
    x = np.arange(len(classes))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8.5, 2.25 * len(classes)), 5.2))
    for offset, metric, label, color in [
        (-width, "precision", "Precision", COLORS["precision"]),
        (0.0, "recall", "Recall", COLORS["recall"]),
        (width, "f1", "F1", COLORS["f1"]),
    ]:
        grouped = test.groupby("class_name")[metric]
        means = np.asarray([float(grouped.mean().get(name, 0.0)) for name in classes])
        stds = np.asarray([float(grouped.std().get(name, 0.0)) for name in classes])
        minimums = np.asarray([float(grouped.min().get(name, 0.0)) for name in classes])
        maximums = np.asarray([float(grouped.max().get(name, 0.0)) for name in classes])
        ranges = np.vstack((means - minimums, maximums - means))
        ranges = np.nan_to_num(ranges, nan=0.0)
        stds = np.nan_to_num(stds, nan=0.0)
        ax.bar(
            x + offset,
            means,
            width=width,
            yerr=ranges,
            capsize=4,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            error_kw={"ecolor": "#777777", "elinewidth": 0.9, "capthick": 0.9},
            label=label,
        )
        ax.errorbar(
            x + offset,
            means,
            yerr=stds,
            fmt="none",
            ecolor="#202020",
            elinewidth=1.8,
            capsize=3,
            capthick=1.8,
            zorder=4,
        )
    supports = test.groupby("class_name")["support"].sum()
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{CALLER_LABELS.get(name, name).replace(chr(10), ' ')}\n"
            f"(n={int(supports.get(name, 0))})"
            for name in classes
        ],
        fontsize=10,
    )
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Score", fontsize=11)
    ax.tick_params(axis="y", labelsize=9)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.18, linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Per-class held-out performance", fontsize=15, y=0.97)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "Bars show the fold mean; outer whiskers span fold minimum to maximum; inner whiskers show +/- 1 SD.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.10, right=0.98, top=0.82, bottom=0.23)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def plot_cross_fold_split_metrics(fold_summary: pd.DataFrame, output_path: Path) -> None:
    ok = fold_summary[fold_summary["status"].astype(str).eq("ok")].copy()
    if ok.empty:
        return
    panels = [
        (
            "Objectness",
            [
                ("test_objectness_precision", "Precision", COLORS["precision"]),
                ("test_objectness_recall", "Recall", COLORS["recall"]),
                ("test_objectness_f1", "F1", COLORS["f1"]),
            ],
        ),
        (
            "Labeled candidate types",
            [
                ("test_exact_match_accuracy", "Exact match", "#4E79A7"),
                ("test_any_match_accuracy", "Any match", "#F28E2B"),
                ("test_macro_f1", "Macro F1", "#59A14F"),
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    for ax, (title, metrics) in zip(axes, panels, strict=False):
        x = np.arange(len(metrics))
        means = [float(pd.to_numeric(ok[column], errors="coerce").mean()) for column, _label, _color in metrics]
        stds = [float(pd.to_numeric(ok[column], errors="coerce").std()) for column, _label, _color in metrics]
        minimums = [float(pd.to_numeric(ok[column], errors="coerce").min()) for column, _label, _color in metrics]
        maximums = [float(pd.to_numeric(ok[column], errors="coerce").max()) for column, _label, _color in metrics]
        colors = [color for _column, _label, color in metrics]
        ax.bar(
            x,
            means,
            yerr=np.nan_to_num(np.vstack((np.asarray(means) - np.asarray(minimums), np.asarray(maximums) - np.asarray(means))), nan=0.0),
            capsize=4,
            color=colors,
            edgecolor="white",
            linewidth=0.7,
            error_kw={"ecolor": "#777777", "elinewidth": 0.9, "capthick": 0.9},
        )
        ax.errorbar(
            x,
            means,
            yerr=np.nan_to_num(stds, nan=0.0),
            fmt="none",
            ecolor="#202020",
            elinewidth=1.8,
            capsize=3,
            capthick=1.8,
            zorder=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _column, label, _color in metrics], fontsize=10)
        ax.set_ylim(0, 1.02)
        ax.set_title(title, fontsize=13, pad=10)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.18, linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Score", fontsize=11)
    fig.suptitle("Held-out performance across folds", fontsize=15, y=0.97)
    fig.text(
        0.5,
        0.025,
        "Bars show the fold mean; outer whiskers span fold minimum to maximum; inner whiskers show +/- 1 SD.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.20, wspace=0.16)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_primary_cross_fold_outputs(
    output_dir: Path,
    test_predictions: pd.DataFrame,
    fold_summary: pd.DataFrame,
    fold_per_class: pd.DataFrame,
    aggregate_overall: pd.DataFrame,
    aggregate_per_class: pd.DataFrame,
    class_names: list[str],
) -> None:
    aggregate_overall.to_csv(output_dir / "metrics_summary.tsv", sep="\t", index=False)
    aggregate_per_class.to_csv(output_dir / "per_class_metrics.tsv", sep="\t", index=False)
    test_predictions.to_csv(output_dir / "classification_predictions.tsv", sep="\t", index=False)
    plot_cross_fold_per_class_metrics(fold_per_class, output_dir / "per_class_metrics.png")
    plot_cross_fold_split_metrics(fold_summary, output_dir / "split_metrics.png")
    compatibility = predictions_to_distance_table(test_predictions, class_names)
    compatibility.to_csv(output_dir / "prototype_distances.tsv", sep="\t", index=False)
    embed_corpus._plot_anchor_prediction_summary(
        compatibility,
        output_dir / "anchor_prediction_summary.png",
        tau=None,
        title_suffix="Cross-fold held-out predictions",
    )


def clear_fixed_holdout_artifacts(output_dir: Path, source_dir: Path) -> None:
    if output_dir.resolve() != source_dir.resolve():
        return
    stale_files = [
        "candidate_region_classifier.pt",
        "train_predictions.tsv",
        "test_predictions.tsv",
        "predicted_complex_sv.tsv",
        "cluster_predictions.tsv",
        "row_raw_predictions.tsv",
        "cluster_aggregated_raw_predictions.tsv",
        "anchor_prediction_summary_train.png",
        "anchor_prediction_summary_test.png",
        "embedding_projection_predicted.png",
        "prototype_distances.png",
        "training_curves.png",
        "training_metrics.tsv",
        "type_thresholds.tsv",
        "subtype_thresholds.tsv",
        "objectness_tau_sweep.png",
        "type_thresholds.png",
        "logo_calibration_predictions.tsv",
        "logo_calibration_raw.tsv",
        "logo_training_metrics.tsv",
        "logo_metrics_summary.tsv",
        "logo_per_class_metrics.tsv",
    ]
    for name in stale_files:
        path = output_dir / name
        if path.exists():
            path.unlink()


def plot_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> None:
    df = metrics.copy()
    df["label"] = [
        f"{CALLER_LABELS.get(class_name, class_name)}\n(n={int(support)})"
        for class_name, support in zip(df["class_name"].astype(str), df["support"], strict=False)
    ]
    x = np.arange(len(df))
    width = 0.24
    fig, ax = plt.subplots(figsize=(15.5, 8.8))
    style_poster_axis(ax)
    ax.bar(x - width, df["precision"], width=width, color=COLORS["precision"], label="Precision")
    ax.bar(x, df["recall"], width=width, color=COLORS["recall"], label="Recall")
    ax.bar(x + width, df["f1"], width=width, color=COLORS["f1"], label="F1")
    if "fold_f1_std" in df.columns:
        yerr = pd.to_numeric(df["fold_f1_std"], errors="coerce").fillna(0.0).to_numpy()
        ax.errorbar(x + width, df["f1"], yerr=yerr, fmt="none", ecolor="#333333", elinewidth=1.2, capsize=4, alpha=0.85)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Held-out score", fontsize=POSTER_FONT["axis"])
    ax.set_title("Caller-label performance", fontsize=POSTER_FONT["title"], pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=0, ha="center", fontsize=POSTER_FONT["tick"])
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=POSTER_FONT["legend"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_poster_figure(fig, output_dir, "caller_label_precision_recall_f1")
    plt.close(fig)


def plot_error_counts(metrics: pd.DataFrame, output_dir: Path) -> None:
    df = metrics.copy()
    labels = [
        f"{CALLER_LABELS.get(class_name, class_name)}\nF1={float(f1):.2f}"
        for class_name, f1 in zip(df["class_name"].astype(str), df["f1"], strict=False)
    ]
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    style_poster_axis(ax)
    ax.bar(x, df["tp"], color=COLORS["tp"], label="TP")
    ax.bar(x, df["fp"], bottom=df["tp"], color=COLORS["fp"], label="FP")
    ax.bar(x, df["fn"], bottom=df["tp"] + df["fp"], color=COLORS["fn"], label="FN")
    ax.set_ylabel("Held-out candidate count", fontsize=POSTER_FONT["axis"])
    ax.set_title("Caller-label error counts", fontsize=POSTER_FONT["title"], pad=18)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=POSTER_FONT["tick"])
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=POSTER_FONT["legend"])
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save_poster_figure(fig, output_dir, "caller_label_tp_fp_fn")
    plt.close(fig)


def plot_fold_heatmap(fold_per_class: pd.DataFrame, output_dir: Path, class_names: list[str]) -> None:
    pc = fold_per_class[(fold_per_class["split"].astype(str) == "test") & (fold_per_class["status"].astype(str) == "ok")].copy()
    if pc.empty:
        return
    pc["fold_name"] = pc["run_id"].astype(str)
    pivot = pc.pivot_table(index="class_name", columns="fold_name", values="f1", aggfunc="mean").reindex(class_names)
    support = pc.pivot_table(index="class_name", columns="fold_name", values="support", aggfunc="sum").reindex(class_names)
    fig, ax = plt.subplots(figsize=(17.5, 7.2))
    style_poster_axis(ax)
    im = ax.imshow(pivot.fillna(0.0).to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            sup = support.iloc[i, j]
            text = "-" if pd.isna(val) else f"{val:.2f}\nn={int(sup)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=POSTER_FONT["heatmap_annotation"], color="white" if (not pd.isna(val) and val < 0.55) else "black")
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels([CALLER_LABELS.get(c, c).replace("\n", " ") for c in class_names])
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([c.replace("fold_", "") for c in pivot.columns], rotation=0)
    ax.set_xlabel("Held-out fold", fontsize=POSTER_FONT["axis"])
    ax.set_title("Per-Fold F1 by Caller Label", fontsize=POSTER_FONT["title"], pad=20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("F1", fontsize=POSTER_FONT["axis"])
    cbar.ax.tick_params(labelsize=POSTER_FONT["tick"])
    fig.tight_layout()
    save_poster_figure(fig, output_dir, "fold_by_label_f1_heatmap")
    plt.close(fig)


def plot_poster_summary(metrics: pd.DataFrame, fold_summary: pd.DataFrame, output_dir: Path) -> None:
    df = metrics.copy()
    labels = [
        f"{CALLER_LABELS.get(class_name, class_name)}\n(n={int(support)})"
        for class_name, support in zip(df["class_name"].astype(str), df["support"], strict=False)
    ]
    x = np.arange(len(df))
    fig = plt.figure(figsize=(23.0, 13.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1.0], height_ratios=[1.0, 1.0])

    ax1 = fig.add_subplot(gs[:, 0])
    style_poster_axis(ax1)
    width = 0.23
    ax1.bar(x - width, df["precision"], width=width, color=COLORS["precision"], label="Precision")
    ax1.bar(x, df["recall"], width=width, color=COLORS["recall"], label="Recall")
    ax1.bar(x + width, df["f1"], width=width, color=COLORS["f1"], label="F1")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Score", fontsize=POSTER_FONT["axis"])
    ax1.set_title("Held-Out Caller-Label Performance", fontsize=POSTER_FONT["title"], pad=24)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=POSTER_FONT["annotation"])
    ax1.grid(axis="y", alpha=0.18)
    ax1.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=POSTER_FONT["legend"])

    ax2 = fig.add_subplot(gs[0, 1])
    style_poster_axis(ax2)
    ax2.bar(x, df["tp"], color=COLORS["tp"], label="TP")
    ax2.bar(x, df["fp"], bottom=df["tp"], color=COLORS["fp"], label="FP")
    ax2.bar(x, df["fn"], bottom=df["tp"] + df["fp"], color=COLORS["fn"], label="FN")
    ax2.set_title("TP / FP / FN Counts", fontsize=POSTER_FONT["title"], pad=18)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.split("\n")[0] for c in labels], rotation=20, ha="right", fontsize=POSTER_FONT["tick"])
    ax2.grid(axis="y", alpha=0.18)
    ax2.legend(frameon=False, ncol=3, fontsize=POSTER_FONT["annotation"])

    ax3 = fig.add_subplot(gs[1, 1])
    style_poster_axis(ax3)
    ok = fold_summary[fold_summary["status"].astype(str).eq("ok")].copy()
    if not ok.empty:
        ax3.plot(ok["fold_id"], ok["test_macro_f1"], marker="o", color="#4E79A7", label="Macro F1")
        ax3.plot(ok["fold_id"], ok["test_exact_match_accuracy"], marker="s", color="#F28E2B", label="Exact match")
        ax3.plot(ok["fold_id"], ok["test_objectness_f1"], marker="^", color="#59A14F", label="Objectness F1")
    ax3.set_ylim(0, 1.05)
    ax3.set_xlabel("Held-out fold")
    ax3.set_ylabel("Fold score", fontsize=POSTER_FONT["axis"])
    ax3.set_title("Fold-Level Stability", fontsize=POSTER_FONT["title"], pad=18)
    ax3.grid(alpha=0.18)
    ax3.legend(frameon=False, fontsize=POSTER_FONT["annotation"])

    fig.suptitle("Cross-Fold Study: Model vs Caller-Derived Labels", fontsize=POSTER_FONT["suptitle"], y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save_poster_figure(fig, output_dir, "cross_fold_poster_summary")
    plt.close(fig)


def write_sample_fold_plot(assignments: pd.DataFrame, output_dir: Path, class_names: list[str]) -> None:
    fold = assignments.groupby("fold_id", as_index=False).agg(
        n_samples=("sample_id", "count"),
        n_candidates=("n_candidates", "sum"),
        n_labeled=("n_labeled", "sum"),
    )
    for class_name in class_names:
        fold[f"support_{class_name}"] = assignments.groupby("fold_id")[f"support_{class_name}"].sum().to_numpy()
    x = np.arange(len(fold))
    fig, ax = plt.subplots(figsize=(17.0, 7.0))
    style_poster_axis(ax)
    bottom = np.zeros(len(fold))
    for class_name in class_names:
        vals = fold[f"support_{class_name}"].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=COLORS.get(class_name, "#999999"), label=CALLER_LABELS.get(class_name, class_name).replace("\n", " "))
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"F{int(i)}\nn={int(n)}" for i, n in zip(fold["fold_id"], fold["n_samples"], strict=False)])
    ax.set_ylabel("Held-out label support", fontsize=POSTER_FONT["axis"])
    ax.set_title("Caller-Label Support by Held-Out Fold", fontsize=POSTER_FONT["title"], pad=20)
    ax.grid(axis="y", alpha=0.18)
    ax.legend(frameon=False, fontsize=POSTER_FONT["legend"], ncol=2)
    fig.tight_layout()
    save_poster_figure(fig, output_dir, "fold_label_support")
    plt.close(fig)


def regenerate_figures_from_existing(output_dir: Path, class_names: list[str]) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "aggregate_per_class_metrics.tsv"
    fold_summary_path = output_dir / "fold_summary.tsv"
    fold_per_class_path = output_dir / "fold_per_class_metrics.tsv"
    assignments_path = output_dir / "fold_assignments.tsv"
    required = [metrics_path, fold_summary_path, fold_per_class_path]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot regenerate plots; missing required files: {missing}")

    aggregate_class = pd.read_csv(metrics_path, sep="\t")
    fold_summary = pd.read_csv(fold_summary_path, sep="\t")
    fold_per_class = pd.read_csv(fold_per_class_path, sep="\t")
    predictions = pd.read_csv(output_dir / "cross_fold_predictions.tsv", sep="\t")
    overall = pd.read_csv(output_dir / "aggregate_overall_metrics.tsv", sep="\t")

    plot_metric_bars(aggregate_class, figures_dir)
    plot_error_counts(aggregate_class, figures_dir)
    plot_fold_heatmap(fold_per_class, figures_dir, class_names)
    plot_poster_summary(aggregate_class, fold_summary, figures_dir)
    if assignments_path.exists():
        assignments = pd.read_csv(assignments_path, sep="\t")
        write_sample_fold_plot(assignments, figures_dir, class_names)
    write_primary_cross_fold_outputs(
        output_dir,
        predictions,
        fold_summary,
        fold_per_class,
        overall,
        aggregate_class,
        class_names,
    )
    log.info("Regenerated poster-scale figures in %s", figures_dir)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    CALLER_LABELS["BFB"] = str(args.bfb_label)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    runs_dir = output_dir / "runs"
    class_names = _split_csv(args.class_names)
    if getattr(args, "plot_only", False):
        regenerate_figures_from_existing(output_dir, class_names)
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(exist_ok=True)

    class_names = _split_csv(args.class_names)
    source_dir = Path(args.source_dir)
    embeddings, metadata, tabular_features, tabular_names, base_cfg = load_source(source_dir, class_names)
    features = make_prototype_features(
        embeddings,
        tabular_features,
        tabular_weight=float(args.tabular_weight),
        final_l2_normalize=bool(args.final_l2_normalize),
    )
    metadata.to_csv(output_dir / "candidate_embeddings.tsv", sep="\t", index=False)
    np.savez(output_dir / "embeddings.npz", embeddings=np.asarray(embeddings, dtype=np.float32))
    np.savez(
        output_dir / "fewshot_feature_matrix.npz",
        features=np.asarray(features, dtype=np.float32),
        embedding_dim=int(embeddings.shape[1]),
        tabular_dim=int(tabular_features.shape[1]),
    )
    np.savez(
        output_dir / "post_model_embeddings.npz",
        embeddings=np.asarray(features, dtype=np.float32),
        model_space="fewshot_prototype_feature_space",
    )
    clear_fixed_holdout_artifacts(output_dir, source_dir)
    manifest_samples = load_manifest_samples(args.manifest)
    if args.fold_assignments:
        assignment_path = Path(args.fold_assignments)
        if not assignment_path.exists():
            raise FileNotFoundError(f"--fold_assignments not found: {assignment_path}")
        fold_assignments = pd.read_csv(assignment_path, sep="\t").fillna("")
        required_columns = {"sample_id", "fold_id", "fold_name"}
        missing_columns = required_columns.difference(fold_assignments.columns)
        if missing_columns:
            raise ValueError(f"Fold assignments missing {sorted(missing_columns)}: {assignment_path}")
        fold_assignments["sample_id"] = fold_assignments["sample_id"].astype(str)
        if fold_assignments["sample_id"].duplicated().any():
            raise ValueError(f"Fold assignments contain duplicate samples: {assignment_path}")
        missing_samples = sorted(set(manifest_samples).difference(fold_assignments["sample_id"]))
        if missing_samples:
            raise ValueError(f"Fold assignments omit manifest samples: {missing_samples}")
        fold_assignments = fold_assignments[fold_assignments["sample_id"].isin(manifest_samples)].copy()
        fold_assignments["fold_id"] = fold_assignments["fold_id"].astype(int)
    else:
        sample_support = sample_support_table(manifest_samples, metadata, class_names)
        fold_assignments = make_balanced_folds(sample_support, class_names, int(args.fold_size), int(args.seed))
    fold_assignments = fold_assignments.sort_values(["fold_id", "sample_id"]).reset_index(drop=True)
    fold_assignments.to_csv(output_dir / "fold_assignments.tsv", sep="\t", index=False)
    write_sample_fold_plot(fold_assignments, figures_dir, class_names)

    labeled, background = _label_background_masks(metadata)
    samples = metadata["sample_id"].astype(str).to_numpy()

    config = {
        "source_dir": str(source_dir),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "fold_size_requested": int(args.fold_size),
        "n_manifest_samples": int(len(manifest_samples)),
        "n_candidate_samples": int(metadata["sample_id"].nunique()),
        "n_candidate_rows": int(len(metadata)),
        "class_names": class_names,
        "caller_label_map": CALLER_LABELS,
        "threshold_calibration": "train split only per fold",
        "tabular_feature_names": tabular_names,
        "model_type": "candidate_region_fewshot_prototype",
        "prototype_scheme": "one only-class centroid plus one contains-class centroid per class",
        "containing_prototypes": int(args.containing_prototypes),
        "prototype_temperature": float(args.prototype_temperature),
        "score_transform": str(args.score_transform),
        "tabular_weight": float(args.tabular_weight),
        "final_l2_normalize": bool(args.final_l2_normalize),
        "fold_assignments_source": str(args.fold_assignments),
    }
    with (output_dir / "cross_fold_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[pd.DataFrame] = []
    test_prediction_rows: list[pd.DataFrame] = []
    all_prediction_rows: list[pd.DataFrame] = []
    unique_folds = sorted(fold_assignments["fold_id"].astype(int).unique().tolist())
    for fold_index, fold_id in enumerate(unique_folds, start=1):
        fold_name = f"fold_{fold_id:02d}"
        run_dir = runs_dir / fold_name
        run_dir.mkdir(exist_ok=True)
        test_samples = sorted(fold_assignments.loc[fold_assignments["fold_id"].astype(int).eq(fold_id), "sample_id"].astype(str).tolist())
        test_mask = np.isin(samples, np.asarray(test_samples, dtype=object))
        train_mask = ~test_mask
        log.info("Fold %d/%d %s test_samples=%s test_candidates=%d", fold_index, len(unique_folds), fold_name, ",".join(test_samples), int(test_mask.sum()))
        t0 = time.time()
        try:
            write_split_table(run_dir, metadata, samples, test_samples, labeled, background)
            helper_args = make_model_args(base_cfg, args, int(args.model_seed_base) + int(fold_id))
            prototypes, class_summary = build_prototypes(
                features,
                metadata,
                class_names,
                train_mask,
                subtype_targets="general",
                containing_prototypes=int(args.containing_prototypes),
                min_prototype_members=int(args.min_prototype_members),
                min_cluster_members=int(args.min_cluster_members),
                subtype_weighting="off",
                seed=int(args.model_seed_base) + int(fold_id),
            )
            prototype_table(prototypes).to_csv(run_dir / "fewshot_prototypes.tsv", sep="\t", index=False)
            class_summary.to_csv(run_dir / "fewshot_class_summary.tsv", sep="\t", index=False)
            vectors, proto_classes, proto_names, proto_kinds = prototype_arrays(prototypes)
            np.savez(
                run_dir / "fewshot_prototypes.npz",
                prototypes=vectors,
                class_names=proto_classes,
                prototype_names=proto_names,
                prototype_kinds=proto_kinds,
            )
            row_raw = predict_with_prototypes(
                prototypes,
                features,
                metadata,
                class_names,
                subtype_targets="general",
                temperature=float(args.prototype_temperature),
                score_transform=str(args.score_transform),
            )
            row_raw.to_csv(run_dir / "row_raw_predictions.tsv", sep="\t", index=False)
            raw = apply_cluster_aggregation(row_raw, class_names, mode=str(args.cluster_aggregation))
            raw.to_csv(run_dir / "cluster_aggregated_raw_predictions.tsv", sep="\t", index=False)
            predictions, threshold_info = calibrate_and_annotate(raw, train_mask, class_names, helper_args, run_dir)
            predictions["split"] = np.where(test_mask, "test", "train")
            predictions["fold_id"] = int(fold_id)
            predictions["fold_name"] = fold_name
            predictions.to_csv(run_dir / "classification_predictions.tsv", sep="\t", index=False)
            predictions.loc[train_mask].to_csv(run_dir / "train_predictions.tsv", sep="\t", index=False)
            predictions.loc[test_mask].to_csv(run_dir / "test_predictions.tsv", sep="\t", index=False)
            torch.save(
                {
                    "model_type": "candidate_region_fewshot_prototype",
                    "class_names": class_names,
                    "fold_id": int(fold_id),
                    "test_samples": test_samples,
                    "prototypes": torch.as_tensor(vectors, dtype=torch.float32),
                    "prototype_class_names": [str(value) for value in proto_classes.tolist()],
                    "prototype_names": [str(value) for value in proto_names.tolist()],
                    "prototype_kinds": [str(value) for value in proto_kinds.tolist()],
                    "feature_dim": int(features.shape[1]),
                    **threshold_info,
                },
                run_dir / "candidate_region_classifier_fewshot.pt",
            )
            training_metrics = pd.DataFrame()
            row, overall_metrics, per_class_metrics = summarize_run(
                fold_name,
                len(test_samples),
                int(fold_id),
                test_samples,
                train_mask,
                test_mask,
                predictions,
                class_names,
                threshold_info,
                training_metrics,
                time.time() - t0,
            )
            row.update(
                {
                    "fold_id": int(fold_id),
                    "fold_name": fold_name,
                    "status": "ok",
                    "test_manifest_samples": ";".join(test_samples),
                    "test_manifest_n_samples": int(len(test_samples)),
                    "test_present_n_samples": int(pd.Series(samples[test_mask]).nunique()) if int(test_mask.sum()) else 0,
                    "test_absent_n_samples": int(len(test_samples) - (pd.Series(samples[test_mask]).nunique() if int(test_mask.sum()) else 0)),
                }
            )
            overall_metrics.insert(0, "fold_id", int(fold_id))
            overall_metrics.insert(1, "fold_name", fold_name)
            per_class_metrics.insert(0, "fold_id", int(fold_id))
            per_class_metrics.insert(1, "fold_name", fold_name)
            overall_metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
            per_class_metrics.to_csv(run_dir / "per_class_metrics.tsv", sep="\t", index=False)
            summary_rows.append(row)
            pc = per_class_metrics.copy()
            pc.insert(0, "run_id", fold_name)
            pc.insert(3, "status", "ok")
            per_class_rows.append(pc)
            test_prediction_rows.append(predictions.loc[test_mask].copy())
            all_prediction_rows.append(predictions.copy())
        except Exception as exc:
            log.exception("Fold failed: %s", fold_name)
            summary_rows.append(
                {
                    "fold_id": int(fold_id),
                    "fold_name": fold_name,
                    "run_id": fold_name,
                    "status": "failed",
                    "error": str(exc),
                    "test_manifest_samples": ";".join(test_samples),
                    "test_manifest_n_samples": int(len(test_samples)),
                }
            )
        pd.DataFrame(summary_rows).to_csv(output_dir / "fold_summary.tsv", sep="\t", index=False)
        if per_class_rows:
            pd.concat(per_class_rows, ignore_index=True).to_csv(output_dir / "fold_per_class_metrics.tsv", sep="\t", index=False)
        if test_prediction_rows:
            pd.concat(test_prediction_rows, ignore_index=True).to_csv(output_dir / "cross_fold_predictions.tsv", sep="\t", index=False)

    fold_summary = pd.DataFrame(summary_rows)
    fold_per_class = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    test_predictions = pd.concat(test_prediction_rows, ignore_index=True) if test_prediction_rows else pd.DataFrame()
    all_predictions = pd.concat(all_prediction_rows, ignore_index=True) if all_prediction_rows else pd.DataFrame()

    fold_summary.to_csv(output_dir / "fold_summary.tsv", sep="\t", index=False)
    fold_per_class.to_csv(output_dir / "fold_per_class_metrics.tsv", sep="\t", index=False)
    test_predictions.to_csv(output_dir / "cross_fold_predictions.tsv", sep="\t", index=False)
    all_predictions.to_csv(output_dir / "cross_fold_all_predictions_with_train_rows.tsv", sep="\t", index=False)

    aggregate_class = class_metrics_from_predictions(test_predictions, class_names, split_name="cross_fold_test")
    aggregate_class = add_fold_metric_aggregates(aggregate_class, fold_per_class)
    aggregate_class.to_csv(output_dir / "aggregate_per_class_metrics.tsv", sep="\t", index=False)

    overall, per_class_check = metric_tables(test_predictions.copy(), class_names, "cross_fold_test")
    overall.to_csv(output_dir / "aggregate_overall_metrics.tsv", sep="\t", index=False)
    per_class_check.to_csv(output_dir / "aggregate_per_class_metrics_from_metric_tables.tsv", sep="\t", index=False)
    write_primary_cross_fold_outputs(
        output_dir,
        test_predictions,
        fold_summary,
        fold_per_class,
        overall,
        aggregate_class,
        class_names,
    )

    overview = {
        "evaluation_mode": "grouped_cross_fold_fewshot",
        "n_folds": int(len(unique_folds)),
        "n_ok_folds": int(fold_summary["status"].astype(str).eq("ok").sum()) if not fold_summary.empty else 0,
        "n_manifest_samples": int(len(manifest_samples)),
        "n_candidate_samples": int(metadata["sample_id"].nunique()),
        "n_test_candidate_rows": int(len(test_predictions)),
        "n_test_labeled_rows": int(test_predictions.get("is_labeled", pd.Series(dtype=int)).astype(bool).sum()) if not test_predictions.empty else 0,
        "macro_f1_aggregate": float(aggregate_class["f1"].mean()) if not aggregate_class.empty else None,
        "objectness_f1_aggregate": float(overall["objectness_f1"].iloc[0]) if not overall.empty else None,
        "exact_match_aggregate": float(overall["class_exact_match_labeled"].iloc[0]) if not overall.empty else None,
        "any_match_aggregate": float(overall["class_any_match_labeled"].iloc[0]) if not overall.empty else None,
        "fold_size_manifest_genomes": int(fold_assignments.groupby("fold_id")["sample_id"].size().iloc[0]),
        "error_bars": "outer minimum-to-maximum held-out fold ranges with inner mean +/- 1 SD whiskers",
        "prototype_scheme": "one only-class centroid plus one contains-class centroid per class",
    }
    with (output_dir / "cross_fold_overview.json").open("w", encoding="utf-8") as fh:
        json.dump(overview, fh, indent=2)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(overview, fh, indent=2)

    plot_metric_bars(aggregate_class, figures_dir)
    plot_error_counts(aggregate_class, figures_dir)
    plot_fold_heatmap(fold_per_class, figures_dir, class_names)
    plot_poster_summary(aggregate_class, fold_summary, figures_dir)
    log.info("Done. Wrote grouped cross-fold few-shot caller-label study to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline14/candidate_region_classifier_general")
    parser.add_argument("--manifest", default="/data/KolmogorovLab/srinivasanbd/results/pipeline14/complex_sv_manifest.tsv")
    parser.add_argument("--output_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline14/candidate_region_classifier_general")
    parser.add_argument("--class_names", default="ecDNA,chromothripsis,BFB")
    parser.add_argument("--bfb_label", default="BFBArchitect BFB")
    parser.add_argument("--fold_assignments", default="/data/KolmogorovLab/srinivasanbd/results/pipeline14/candidate_region_classifier_general/fold_assignments.tsv")
    parser.add_argument("--fold_size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000000)
    parser.add_argument("--fast_thresholds", action="store_true")
    parser.add_argument("--model_seed_base", type=int, default=13000)
    parser.add_argument("--containing_prototypes", type=int, default=1)
    parser.add_argument("--min_prototype_members", type=int, default=1)
    parser.add_argument("--min_cluster_members", type=int, default=2)
    parser.add_argument("--prototype_temperature", type=float, default=0.25)
    parser.add_argument("--score_transform", choices=("exp", "cosine_shift"), default="exp")
    parser.add_argument("--tabular_weight", type=float, default=1.0)
    parser.add_argument("--final_l2_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster_aggregation", choices=("off", "max"), default="max")
    parser.add_argument("--plot_only", action="store_true", help="Regenerate figures from existing cross-fold TSV outputs without retraining.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
