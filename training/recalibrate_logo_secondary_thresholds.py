"""Recalibrate secondary multi-label calling for existing LOGO raw predictions."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_candidate_region_classifier import (
    annotate_predictions_with_secondary,
    metric_tables,
)


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def split_classes(value: object) -> set[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "empty", "unlabeled"}:
        return set()
    return {part.strip() for part in text.replace(",", ";").split(";") if part.strip()}


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def threshold_grid(min_value: float, max_value: float, steps: int) -> np.ndarray:
    if steps <= 1:
        return np.asarray([float(min_value)], dtype=float)
    return np.linspace(float(min_value), float(max_value), int(steps), dtype=float)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def load_original_fold_thresholds(fold_dir: Path, class_names: list[str]) -> dict[str, Any]:
    annotated = pd.read_csv(fold_dir / "classification_predictions.tsv", sep="\t")
    first = annotated.iloc[0]
    threshold_table = pd.read_csv(fold_dir / "type_thresholds.tsv", sep="\t")
    type_thresholds = {
        str(row["class_name"]): float(row["type_threshold"])
        for _, row in threshold_table.iterrows()
        if str(row["class_name"]) in class_names
    }
    return {
        "objectness_tau": float(first.get("objectness_tau", 0.5)),
        "type_thresholds": type_thresholds,
        "rescue_type_tau": float(first.get("rescue_type_tau", np.nan)),
        "rescue_objectness_floor": float(first.get("rescue_objectness_floor", np.nan)),
        "rescue_margin": float(first.get("rescue_margin", np.nan)),
    }


def clean_optional_float(value: float) -> float | None:
    return None if pd.isna(value) else float(value)


def label_metrics(predictions: pd.DataFrame, class_names: list[str]) -> dict[str, float]:
    labeled = predictions[boolish(predictions["is_labeled"])].copy()
    if labeled.empty:
        return {
            "label_precision_labeled": 1.0,
            "label_recall_labeled": 0.0,
            "label_f1_labeled": 0.0,
            "mean_predicted_labels_labeled": 0.0,
            "multicall_rate_labeled": 0.0,
            "mean_extra_labels_labeled": 0.0,
            "mean_missing_labels_labeled": 0.0,
            "exact_multilabel_labeled": 0.0,
            "any_match_labeled": 0.0,
        }
    tp = fp = fn = 0
    pred_counts: list[int] = []
    extra_counts: list[int] = []
    missing_counts: list[int] = []
    exact_flags: list[bool] = []
    any_flags: list[bool] = []
    for _, row in labeled.iterrows():
        true = split_classes(row.get("true_classes", ""))
        pred = split_classes(row.get("predicted_classes", row.get("predicted_class", "")))
        pred_counts.append(len(pred))
        extra_counts.append(len(pred - true))
        missing_counts.append(len(true - pred))
        exact_flags.append(true == pred)
        any_flags.append(bool(true & pred))
        for class_name in class_names:
            in_true = class_name in true
            in_pred = class_name in pred
            tp += int(in_true and in_pred)
            fp += int((not in_true) and in_pred)
            fn += int(in_true and (not in_pred))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "label_precision_labeled": float(precision),
        "label_recall_labeled": float(recall),
        "label_f1_labeled": float(f1),
        "mean_predicted_labels_labeled": float(np.mean(pred_counts)),
        "multicall_rate_labeled": float(np.mean([count > 1 for count in pred_counts])),
        "mean_extra_labels_labeled": float(np.mean(extra_counts)),
        "mean_missing_labels_labeled": float(np.mean(missing_counts)),
        "exact_multilabel_labeled": float(np.mean(exact_flags)),
        "any_match_labeled": float(np.mean(any_flags)),
    }


def score_secondary_candidate(predictions: pd.DataFrame, class_names: list[str], secondary_min: float, secondary_delta: float) -> dict[str, Any]:
    overall, per_class = metric_tables(predictions, class_names, "calibration")
    row = overall.iloc[0].to_dict() if not overall.empty else {}
    row.update(label_metrics(predictions, class_names))
    row["secondary_min"] = float(secondary_min)
    row["secondary_delta"] = float(secondary_delta)
    row["macro_f1"] = float(per_class["f1"].astype(float).mean()) if not per_class.empty else 0.0
    row["secondary_objective_recall"] = (
        1.20 * row["label_f1_labeled"]
        + 0.80 * row["label_recall_labeled"]
        + 0.40 * row["exact_multilabel_labeled"]
        + 0.20 * row["any_match_labeled"]
        - 0.25 * row["mean_extra_labels_labeled"]
        - 0.05 * row["multicall_rate_labeled"]
    )
    return row


def annotate_with_policy(
    raw: pd.DataFrame,
    thresholds: dict[str, Any],
    class_names: list[str],
    secondary_min: float | None,
    secondary_delta: float | None,
) -> pd.DataFrame:
    return annotate_predictions_with_secondary(
        raw,
        float(thresholds["objectness_tau"]),
        thresholds["type_thresholds"],
        class_names,
        secondary_min=secondary_min,
        secondary_delta=secondary_delta,
        rescue_type_tau=clean_optional_float(thresholds["rescue_type_tau"]),
        rescue_objectness_floor=clean_optional_float(thresholds["rescue_objectness_floor"]),
        rescue_margin=clean_optional_float(thresholds["rescue_margin"]),
        subtype_thresholds={},
        subtype_targets="general",
    )


def choose_recall_secondary(
    train_raw: pd.DataFrame,
    thresholds: dict[str, Any],
    class_names: list[str],
    min_grid: np.ndarray,
    delta_grid: np.ndarray,
) -> tuple[float, float, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for secondary_min in min_grid:
        for secondary_delta in delta_grid:
            annotated = annotate_with_policy(train_raw, thresholds, class_names, float(secondary_min), float(secondary_delta))
            rows.append(score_secondary_candidate(annotated, class_names, float(secondary_min), float(secondary_delta)))
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(
        [
            "secondary_objective_recall",
            "label_recall_labeled",
            "label_f1_labeled",
            "exact_multilabel_labeled",
            "mean_extra_labels_labeled",
            "multicall_rate_labeled",
            "secondary_min",
            "secondary_delta",
        ],
        ascending=[False, False, False, False, True, True, True, True],
    ).iloc[0]
    return float(best["secondary_min"]), float(best["secondary_delta"]), sweep


def summarize_predictions(predictions: pd.DataFrame, class_names: list[str], split_name: str) -> dict[str, Any]:
    overall, per_class = metric_tables(predictions, class_names, split_name)
    row: dict[str, Any] = {}
    if not overall.empty:
        values = overall.iloc[0].to_dict()
        row.update({f"{split_name}_{key}": value for key, value in values.items() if key != "split"})
    if not per_class.empty:
        row[f"{split_name}_macro_f1"] = float(per_class["f1"].astype(float).mean())
        present = per_class[per_class["support"].astype(float) > 0]
        row[f"{split_name}_macro_f1_present"] = float(present["f1"].astype(float).mean()) if not present.empty else np.nan
    lm = label_metrics(predictions, class_names)
    row.update({f"{split_name}_{key}": value for key, value in lm.items()})
    return row


def write_fold_outputs(
    variant_dir: Path,
    fold_name: str,
    raw: pd.DataFrame,
    split: pd.Series,
    predictions: pd.DataFrame,
    thresholds: dict[str, Any],
    secondary_min: float | None,
    secondary_delta: float | None,
    class_names: list[str],
    sweep: pd.DataFrame | None = None,
) -> dict[str, Any]:
    run_dir = variant_dir / "runs" / fold_name
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions = predictions.copy()
    predictions["split"] = split.to_numpy()
    predictions.to_csv(run_dir / "classification_predictions.tsv", sep="\t", index=False)
    predictions[predictions["split"].astype(str).eq("train")].to_csv(run_dir / "train_predictions.tsv", sep="\t", index=False)
    predictions[predictions["split"].astype(str).eq("test")].to_csv(run_dir / "test_predictions.tsv", sep="\t", index=False)
    raw.to_csv(run_dir / "raw_predictions.tsv", sep="\t", index=False)
    if sweep is not None:
        sweep.to_csv(run_dir / "secondary_threshold_sweep_recall_train.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "class_name": class_name,
                "type_threshold": float(thresholds["type_thresholds"].get(class_name, 0.5)),
                "secondary_threshold": (
                    max(float(thresholds["type_thresholds"].get(class_name, 0.5)) + float(secondary_delta), float(secondary_min))
                    if secondary_min is not None and secondary_delta is not None
                    else float(thresholds["type_thresholds"].get(class_name, 0.5))
                ),
            }
            for class_name in class_names
        ]
    ).to_csv(run_dir / "type_thresholds.tsv", sep="\t", index=False)

    train_pred = predictions[predictions["split"].astype(str).eq("train")].copy()
    test_pred = predictions[predictions["split"].astype(str).eq("test")].copy()
    overall_rows: list[pd.DataFrame] = []
    per_class_rows: list[pd.DataFrame] = []
    for split_name, split_df in [("train", train_pred), ("test", test_pred)]:
        overall, per_class = metric_tables(split_df, class_names, split_name)
        if not overall.empty:
            overall_rows.append(overall)
        if not per_class.empty:
            per_class_rows.append(per_class)
    overall_metrics = pd.concat(overall_rows, ignore_index=True) if overall_rows else pd.DataFrame()
    per_class_metrics = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    overall_metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
    per_class_metrics.to_csv(run_dir / "per_class_metrics.tsv", sep="\t", index=False)

    test_samples = sorted(test_pred["sample_id"].astype(str).unique().tolist())
    row = {
        "fold": fold_name,
        "test_samples": ";".join(test_samples),
        "secondary_min": np.nan if secondary_min is None else float(secondary_min),
        "secondary_delta": np.nan if secondary_delta is None else float(secondary_delta),
        "objectness_tau": float(thresholds["objectness_tau"]),
        "rescue_type_tau": thresholds["rescue_type_tau"],
        "rescue_objectness_floor": thresholds["rescue_objectness_floor"],
        "rescue_margin": thresholds["rescue_margin"],
    }
    row.update(summarize_predictions(train_pred, class_names, "train"))
    row.update(summarize_predictions(test_pred, class_names, "test"))
    return row


def plot_variant_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    if summary.empty:
        return
    metrics = [
        ("test_class_exact_match_labeled", "Exact Match"),
        ("test_class_any_match_labeled", "Any Match"),
        ("test_label_recall_labeled", "Label Recall"),
        ("test_label_f1_labeled", "Label F1"),
        ("test_macro_f1", "Macro F1"),
        ("test_mean_extra_labels_labeled", "Extra Labels"),
        ("test_mean_missing_labels_labeled", "Missing Labels"),
    ]
    aggregate = summary.groupby("variant", as_index=False).agg(
        **{metric: (metric, "mean") for metric, _ in metrics if metric in summary.columns}
    )
    x = np.arange(len(aggregate))
    fig, axes = plt.subplots(len(metrics), 1, figsize=(12.5, 2.3 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, (metric, label) in zip(axes, metrics):
        if metric not in aggregate:
            continue
        ax.bar(x, aggregate[metric].astype(float), color="#4E79A7")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(aggregate["variant"], rotation=25, ha="right")
    fig.suptitle("LOGO Secondary-Threshold Recalibration Variants", y=0.995)
    fig.tight_layout()
    fig.savefig(output_dir / "secondary_recalibration_variant_summary.png", dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = split_csv(args.class_names)
    min_grid = threshold_grid(args.secondary_min_min, args.secondary_min_max, args.secondary_min_steps)
    delta_grid = threshold_grid(args.secondary_delta_min, args.secondary_delta_max, args.secondary_delta_steps)
    variant_rows: list[dict[str, Any]] = []

    fold_dirs = sorted(path for path in runs_dir.glob("logo_*") if path.is_dir())
    if not fold_dirs:
        raise FileNotFoundError(f"No logo_* fold directories found under {runs_dir}")

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        raw = pd.read_csv(fold_dir / "raw_predictions.tsv", sep="\t")
        old_annotated = pd.read_csv(fold_dir / "classification_predictions.tsv", sep="\t")
        split = old_annotated["split"].astype(str)
        train_mask = split.eq("train").to_numpy()
        thresholds = load_original_fold_thresholds(fold_dir, class_names)

        off_pred = annotate_with_policy(raw, thresholds, class_names, None, None)
        row = write_fold_outputs(
            output_dir / "secondary_off",
            fold_name,
            raw,
            split,
            off_pred,
            thresholds,
            None,
            None,
            class_names,
        )
        row["variant"] = "secondary_off"
        variant_rows.append(row)

        selected_min, selected_delta, sweep = choose_recall_secondary(
            raw.loc[train_mask].copy(),
            thresholds,
            class_names,
            min_grid,
            delta_grid,
        )
        recall_pred = annotate_with_policy(raw, thresholds, class_names, selected_min, selected_delta)
        row = write_fold_outputs(
            output_dir / "recall_optimized",
            fold_name,
            raw,
            split,
            recall_pred,
            thresholds,
            selected_min,
            selected_delta,
            class_names,
            sweep=sweep,
        )
        row["variant"] = "recall_optimized"
        variant_rows.append(row)

    summary = pd.DataFrame(variant_rows)
    summary.to_csv(output_dir / "secondary_recalibration_fold_summary.tsv", sep="\t", index=False)
    aggregate = summary.groupby("variant", as_index=False).agg(
        n_folds=("fold", "count"),
        secondary_min_mean=("secondary_min", "mean"),
        secondary_delta_mean=("secondary_delta", "mean"),
        test_objectness_f1_mean=("test_objectness_f1", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_exact_match_mean=("test_class_exact_match_labeled", "mean"),
        test_any_match_mean=("test_class_any_match_labeled", "mean"),
        test_label_precision_mean=("test_label_precision_labeled", "mean"),
        test_label_recall_mean=("test_label_recall_labeled", "mean"),
        test_label_f1_mean=("test_label_f1_labeled", "mean"),
        test_multicall_rate_mean=("test_multicall_rate_labeled", "mean"),
        test_mean_extra_labels_mean=("test_mean_extra_labels_labeled", "mean"),
        test_mean_missing_labels_mean=("test_mean_missing_labels_labeled", "mean"),
    )
    aggregate.to_csv(output_dir / "secondary_recalibration_variant_summary.tsv", sep="\t", index=False)
    plot_variant_summary(summary, output_dir)
    with (output_dir / "secondary_recalibration_config.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "runs_dir": str(runs_dir),
                "output_dir": str(output_dir),
                "class_names": class_names,
                "secondary_min_grid": [float(x) for x in min_grid],
                "secondary_delta_grid": [float(x) for x in delta_grid],
                "note": "Uses saved LOGO raw predictions; NN weights are unchanged.",
            },
            fh,
            indent=2,
        )
    print(f"Wrote secondary recalibration outputs to {output_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_names", default="ecDNA,Seismic_Amplification,chromothripsis,BFB")
    parser.add_argument("--secondary_min_min", type=float, default=0.30)
    parser.add_argument("--secondary_min_max", type=float, default=0.90)
    parser.add_argument("--secondary_min_steps", type=int, default=13)
    parser.add_argument("--secondary_delta_min", type=float, default=0.0)
    parser.add_argument("--secondary_delta_max", type=float, default=0.50)
    parser.add_argument("--secondary_delta_steps", type=int, default=11)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
