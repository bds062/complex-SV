"""Sweep candidate-region general classifier performance across sample-level splits.

This side-analysis reuses an existing candidate_region_classifier_general output
folder. It trains only the frozen-embedding classifier head for many randomized
cell-line splits and summarizes test macro F1 and exact-match accuracy.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
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

from training.train_candidate_region_classifier import (  # noqa: E402
    _class_counts,
    _label_background_masks,
    _multi_hot_targets,
    _split_csv,
    _threshold_grid,
    annotate_predictions_with_secondary,
    apply_cluster_aggregation,
    choose_rescue_thresholding,
    choose_secondary_thresholding,
    choose_tau_from_sweep,
    choose_type_thresholds_from_predictions,
    metric_tables,
    predict_model,
    rescue_sweep_table_for_selected,
    secondary_sweep_table_for_selected,
    selected_tabular_feature_names,
    make_tabular_features,
    _train_model,
)
from training.train_multilabel_classifier_head import sweep_objectness_tau  # noqa: E402
from utils import set_seed  # noqa: E402

log = logging.getLogger(__name__)


DEFAULT_TEST_SIZES = "3,5,8,10,15,20"
DEFAULT_SEEDS = "0,1,2,3,4,5,6,7,8,9"


def _csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw or "").replace(";", ",").split(",") if part.strip()]


def _safe_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def load_source(source_dir: Path, class_names: list[str]) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str], dict[str, Any]]:
    embeddings_path = source_dir / "embeddings.npz"
    metadata_path = source_dir / "candidate_embeddings.tsv"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing source embeddings: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing source metadata: {metadata_path}")
    embeddings = np.asarray(np.load(embeddings_path, allow_pickle=True)["embeddings"], dtype=np.float32)
    metadata = pd.read_csv(metadata_path, sep="\t").fillna("")
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"metadata rows={len(metadata)} do not match embeddings shape={embeddings.shape}")

    summary_path = source_dir / "training_summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as fh:
            summary = json.load(fh)
    cfg = dict(summary.get("config", {}))

    tab_npz = source_dir / "tabular_features.npz"
    if tab_npz.exists():
        tab_data = np.load(tab_npz, allow_pickle=True)
        tabular = np.asarray(tab_data["features"], dtype=np.float32)
        feature_names = [str(x) for x in tab_data.get("feature_names", np.asarray([], dtype=object)).tolist()]
    else:
        feature_names = selected_tabular_feature_names(str(cfg.get("tabular_features", "safe")))
        tabular = make_tabular_features(metadata, feature_names)
    if tabular.shape[0] != embeddings.shape[0]:
        raise ValueError(f"tabular rows={tabular.shape[0]} do not match embeddings rows={embeddings.shape[0]}")

    targets = _multi_hot_targets(metadata, class_names, subtype_targets="general")
    counts = _class_counts(targets, class_names)
    log.info("Loaded source_dir=%s rows=%d samples=%d class_counts=%s", source_dir, len(metadata), metadata["sample_id"].nunique(), counts)
    return embeddings, metadata, tabular, feature_names, cfg


def make_model_args(base_cfg: dict[str, Any], cli: argparse.Namespace, run_seed: int) -> argparse.Namespace:
    cfg = dict(base_cfg)
    cfg.update(
        {
            "seed": int(run_seed),
            "subtype_targets": "general",
            "subtype_thresholding": "off",
            "threshold_calibration": "train",
            "epochs": int(cli.epochs if cli.epochs is not None else cfg.get("epochs", 300)),
            "patience": int(cli.patience if cli.patience is not None else cfg.get("patience", 60)),
            "device": str(cli.device),
            "log_every": int(cli.log_every),
        }
    )
    defaults = {
        "hidden_dims": "128",
        "tabular_hidden_dim": 32,
        "activation": "relu",
        "dropout": 0.2,
        "lr": 1e-3,
        "weight_decay": 1e-3,
        "background_weight": 1.0,
        "type_loss_weight": 1.0,
        "label_smoothing": 0.02,
        "class_weighting": "inverse_sqrt",
        "grad_clip": 5.0,
        "batch_size": 512,
        "tau_min": 0.05,
        "tau_max": 0.95,
        "tau_steps": 91,
        "tau": None,
        "tau_selection_metric": "f1",
        "threshold_tie_break": "low",
        "type_tau_min": 0.05,
        "type_tau_max": 0.95,
        "type_tau_steps": 91,
        "type_tau": None,
        "cluster_aggregation": "max",
        "rescue_thresholding": "optimize",
        "rescue_type_tau": 0.85,
        "rescue_objectness_floor": 0.0,
        "rescue_margin": 0.0,
        "rescue_type_tau_min": 0.60,
        "rescue_type_tau_max": 0.98,
        "rescue_type_tau_steps": 20,
        "rescue_objectness_floor_grid": "0,0.001,0.005,0.01,0.02,0.05,0.10",
        "rescue_margin_min": 0.0,
        "rescue_margin_max": 0.30,
        "rescue_margin_steps": 7,
        "rescue_min_recall": 0.85,
        "rescue_min_precision": 0.60,
        "rescue_max_empty_fp_rate": 0.75,
        "secondary_thresholding": "optimize",
        "secondary_objective": "recall",
        "secondary_min": 0.55,
        "secondary_delta": 0.15,
        "secondary_min_min": 0.30,
        "secondary_min_max": 0.90,
        "secondary_min_steps": 13,
        "secondary_delta_min": 0.0,
        "secondary_delta_max": 0.50,
        "secondary_delta_steps": 11,
        "secondary_min_recall": 0.85,
        "secondary_min_precision": 0.60,
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    if cli.fast_thresholds:
        cfg.update(
            {
                "tau_steps": min(int(cfg.get("tau_steps", 91)), 31),
                "type_tau_steps": min(int(cfg.get("type_tau_steps", 91)), 31),
                "rescue_type_tau_steps": min(int(cfg.get("rescue_type_tau_steps", 20)), 8),
                "rescue_margin_steps": min(int(cfg.get("rescue_margin_steps", 7)), 5),
                "secondary_min_steps": min(int(cfg.get("secondary_min_steps", 13)), 7),
                "secondary_delta_steps": min(int(cfg.get("secondary_delta_steps", 11)), 7),
            }
        )
    return argparse.Namespace(**cfg)


def sample_split(
    samples: np.ndarray,
    targets: np.ndarray,
    labeled: np.ndarray,
    background: np.ndarray,
    test_n: int,
    seed: int,
    class_names: list[str],
    max_attempts: int,
) -> tuple[list[str], str]:
    unique_samples = np.asarray(sorted(pd.unique(samples).tolist()), dtype=object)
    if test_n <= 0 or test_n >= len(unique_samples):
        raise ValueError(f"Invalid test_n={test_n}; total samples={len(unique_samples)}")
    rng = np.random.default_rng(int(seed))
    last_reason = ""
    for _attempt in range(int(max_attempts)):
        test_samples = sorted(rng.choice(unique_samples, size=int(test_n), replace=False).astype(str).tolist())
        test_mask = np.isin(samples, np.asarray(test_samples, dtype=object))
        train_mask = ~test_mask
        if int((train_mask & labeled).sum()) < 4:
            last_reason = "train_pos_lt_4"
            continue
        if int((train_mask & background).sum()) == 0:
            last_reason = "train_no_background"
            continue
        train_targets = targets[train_mask]
        missing = [class_names[i] for i in range(len(class_names)) if int((train_targets[:, i] > 0).sum()) == 0]
        if missing:
            last_reason = "train_missing_" + ";".join(missing)
            continue
        return test_samples, "ok"
    raise RuntimeError(f"Could not sample valid split for test_n={test_n}, seed={seed}; last_reason={last_reason}")


def write_split_table(run_dir: Path, metadata: pd.DataFrame, samples: np.ndarray, test_samples: list[str], labeled: np.ndarray, background: np.ndarray) -> None:
    rows: list[dict[str, Any]] = []
    for sample_id in sorted(pd.unique(samples)):
        sample_mask = samples == sample_id
        rows.append(
            {
                "sample_id": sample_id,
                "split": "test" if sample_id in set(test_samples) else "train",
                "n_candidates": int(sample_mask.sum()),
                "n_positive": int((sample_mask & labeled).sum()),
                "n_empty": int((sample_mask & background).sum()),
            }
        )
    pd.DataFrame(rows).to_csv(run_dir / "sample_splits.tsv", sep="\t", index=False)


def calibrate_and_annotate(
    raw_predictions: pd.DataFrame,
    train_mask: np.ndarray,
    class_names: list[str],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_raw = raw_predictions.loc[train_mask].copy()
    objectness_grid = _threshold_grid(args.tau_min, args.tau_max, args.tau_steps)
    type_grid = _threshold_grid(args.type_tau_min, args.type_tau_max, args.type_tau_steps)
    objectness_tau_df = sweep_objectness_tau(train_raw, objectness_grid)
    selected_objectness_tau = float(args.tau) if args.tau is not None else choose_tau_from_sweep(
        objectness_tau_df,
        metric=str(args.tau_selection_metric),
        tie_break=str(args.threshold_tie_break),
    )
    if args.type_tau is not None:
        type_thresholds = {name: float(args.type_tau) for name in class_names}
        type_threshold_sweep = pd.DataFrame()
    else:
        type_thresholds, type_threshold_sweep = choose_type_thresholds_from_predictions(
            train_raw,
            class_names,
            type_grid,
            tie_break=str(args.threshold_tie_break),
        )
    subtype_thresholds: dict[str, dict[str, float]] = {}
    selected_rescue_type_tau, selected_rescue_objectness_floor, selected_rescue_margin, rescue_threshold_sweep = choose_rescue_thresholding(
        train_raw,
        selected_objectness_tau,
        type_thresholds,
        class_names,
        args,
        subtype_thresholds=subtype_thresholds,
    )
    selected_secondary_min, selected_secondary_delta, secondary_threshold_sweep = choose_secondary_thresholding(
        train_raw,
        selected_objectness_tau,
        type_thresholds,
        class_names,
        args,
        subtype_thresholds=subtype_thresholds,
        rescue_type_tau=selected_rescue_type_tau,
        rescue_objectness_floor=selected_rescue_objectness_floor,
        rescue_margin=selected_rescue_margin,
    )
    annotated = annotate_predictions_with_secondary(
        raw_predictions,
        selected_objectness_tau,
        type_thresholds,
        class_names,
        secondary_min=selected_secondary_min,
        secondary_delta=selected_secondary_delta,
        rescue_type_tau=selected_rescue_type_tau,
        rescue_objectness_floor=selected_rescue_objectness_floor,
        rescue_margin=selected_rescue_margin,
        subtype_thresholds=subtype_thresholds,
        subtype_targets="general",
    )
    if rescue_threshold_sweep.empty:
        rescue_threshold_sweep = rescue_sweep_table_for_selected(
            annotated.loc[train_mask].copy(),
            class_names,
            selected_rescue_type_tau,
            selected_rescue_objectness_floor,
            selected_rescue_margin,
            str(args.rescue_thresholding),
        )
    if secondary_threshold_sweep.empty:
        secondary_threshold_sweep = secondary_sweep_table_for_selected(
            annotated.loc[train_mask].copy(),
            class_names,
            selected_secondary_min,
            selected_secondary_delta,
            str(args.secondary_thresholding),
        )
    objectness_tau_df.to_csv(run_dir / "objectness_tau_sweep_train.tsv", sep="\t", index=False)
    type_threshold_sweep.to_csv(run_dir / "type_threshold_sweep_train.tsv", sep="\t", index=False)
    rescue_threshold_sweep.to_csv(run_dir / "rescue_threshold_sweep_train.tsv", sep="\t", index=False)
    secondary_threshold_sweep.to_csv(run_dir / "secondary_threshold_sweep_train.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "class_name": name,
                "type_threshold": float(type_thresholds.get(name, 0.5)),
                "secondary_threshold": (
                    max(float(type_thresholds.get(name, 0.5)) + float(selected_secondary_delta), float(selected_secondary_min))
                    if selected_secondary_min is not None and selected_secondary_delta is not None
                    else float(type_thresholds.get(name, 0.5))
                ),
            }
            for name in class_names
        ]
    ).to_csv(run_dir / "type_thresholds.tsv", sep="\t", index=False)
    info = {
        "selected_objectness_tau": float(selected_objectness_tau),
        "type_thresholds": {name: float(value) for name, value in type_thresholds.items()},
        "selected_rescue_type_tau": None if selected_rescue_type_tau is None else float(selected_rescue_type_tau),
        "selected_rescue_objectness_floor": None if selected_rescue_objectness_floor is None else float(selected_rescue_objectness_floor),
        "selected_rescue_margin": None if selected_rescue_margin is None else float(selected_rescue_margin),
        "selected_secondary_min": None if selected_secondary_min is None else float(selected_secondary_min),
        "selected_secondary_delta": None if selected_secondary_delta is None else float(selected_secondary_delta),
    }
    return annotated, info


def summarize_run(
    run_id: str,
    test_n: int,
    split_seed: int,
    test_samples: list[str],
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    predictions: pd.DataFrame,
    class_names: list[str],
    threshold_info: dict[str, Any],
    training_metrics: pd.DataFrame,
    runtime_sec: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    predictions = predictions.copy()
    predictions["split"] = np.where(test_mask, "test", "train")
    overall_tables: list[pd.DataFrame] = []
    per_class_tables: list[pd.DataFrame] = []
    for split_name, split_mask in [("train", train_mask), ("test", test_mask), ("all", np.ones(len(predictions), dtype=bool))]:
        overall, per_class = metric_tables(predictions.loc[split_mask].copy(), class_names, split_name)
        if not overall.empty:
            overall_tables.append(overall)
        if not per_class.empty:
            per_class_tables.append(per_class)
    overall_metrics = pd.concat(overall_tables, ignore_index=True) if overall_tables else pd.DataFrame()
    per_class_metrics = pd.concat(per_class_tables, ignore_index=True) if per_class_tables else pd.DataFrame()
    test_overall = overall_metrics[overall_metrics["split"] == "test"].iloc[0].to_dict()
    test_per_class = per_class_metrics[per_class_metrics["split"] == "test"].copy()
    train_overall = overall_metrics[overall_metrics["split"] == "train"].iloc[0].to_dict()
    test_f1_values = test_per_class["f1"].astype(float).to_numpy()
    present = test_per_class[test_per_class["support"].astype(float) > 0]
    test_macro_f1_present = float(present["f1"].astype(float).mean()) if not present.empty else float("nan")
    row: dict[str, Any] = {
        "run_id": run_id,
        "status": "ok",
        "test_n_samples": int(test_n),
        "train_n_samples": int(len(set(predictions.loc[train_mask, "sample_id"].astype(str)))) if "sample_id" in predictions else int(train_mask.sum()),
        "split_seed": int(split_seed),
        "test_samples": ";".join(test_samples),
        "train_candidates": int(train_mask.sum()),
        "test_candidates": int(test_mask.sum()),
        "train_labeled": int(train_overall.get("n_labeled", 0)),
        "test_labeled": int(test_overall.get("n_labeled", 0)),
        "test_empty": int(test_overall.get("n_empty", 0)),
        "test_macro_f1": float(np.mean(test_f1_values)) if test_f1_values.size else float("nan"),
        "test_macro_f1_present_classes": test_macro_f1_present,
        "test_exact_match_accuracy": _safe_float(test_overall.get("class_exact_match_labeled")),
        "test_any_match_accuracy": _safe_float(test_overall.get("class_any_match_labeled")),
        "test_objectness_f1": _safe_float(test_overall.get("objectness_f1")),
        "test_objectness_precision": _safe_float(test_overall.get("objectness_precision")),
        "test_objectness_recall": _safe_float(test_overall.get("objectness_recall")),
        "train_objectness_f1": _safe_float(train_overall.get("objectness_f1")),
        "train_exact_match_accuracy": _safe_float(train_overall.get("class_exact_match_labeled")),
        "n_epochs_run": int(len(training_metrics)),
        "best_train_loss": float(training_metrics["loss"].min()) if not training_metrics.empty and "loss" in training_metrics else float("nan"),
        "runtime_sec": float(runtime_sec),
        **threshold_info,
    }
    for _, pc in test_per_class.iterrows():
        class_name = str(pc["class_name"])
        row[f"test_support_{class_name}"] = int(pc["support"])
        row[f"test_f1_{class_name}"] = float(pc["f1"])
        row[f"test_precision_{class_name}"] = float(pc["precision"])
        row[f"test_recall_{class_name}"] = float(pc["recall"])
    return row, overall_metrics, per_class_metrics


def plot_summary(summary: pd.DataFrame, per_class: pd.DataFrame, output_dir: Path) -> None:
    ok = summary[summary["status"] == "ok"].copy()
    if ok.empty:
        return
    for metric, filename, ylabel in [
        ("test_macro_f1", "macro_f1_by_test_size.png", "Test Macro F1"),
        ("test_exact_match_accuracy", "exact_match_by_test_size.png", "Test Exact-Match Accuracy"),
        ("test_objectness_f1", "objectness_f1_by_test_size.png", "Test Objectness F1"),
    ]:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        groups = [grp[metric].dropna().astype(float).to_numpy() for _n, grp in ok.groupby("test_n_samples", sort=True)]
        labels = [str(n) for n in sorted(ok["test_n_samples"].unique())]
        if groups:
            ax.boxplot(groups, tick_labels=labels, showmeans=True)
            for x_i, n in enumerate(sorted(ok["test_n_samples"].unique()), start=1):
                vals = ok.loc[ok["test_n_samples"] == n, metric].astype(float).to_numpy()
                jitter = np.linspace(-0.08, 0.08, num=len(vals)) if len(vals) else []
                ax.scatter(np.asarray(jitter) + x_i, vals, s=18, alpha=0.55, color="#4E79A7")
        ax.set_xlabel("Held-out test cell lines")
        ax.set_ylabel(ylabel)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    scatter = ax.scatter(ok["test_exact_match_accuracy"], ok["test_macro_f1"], c=ok["test_n_samples"], cmap="viridis", s=42, alpha=0.85)
    ax.set_xlabel("Test Exact-Match Accuracy")
    ax.set_ylabel("Test Macro F1")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    fig.colorbar(scatter, ax=ax, label="Held-out test cell lines")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "macro_f1_vs_exact_match.png", dpi=180)
    plt.close(fig)

    test_pc = per_class[(per_class["split"] == "test") & (per_class["status"] == "ok")].copy()
    if not test_pc.empty:
        agg = test_pc.groupby(["test_n_samples", "class_name"], as_index=False).agg(mean_f1=("f1", "mean"), std_f1=("f1", "std"))
        classes = list(dict.fromkeys(agg["class_name"].astype(str).tolist()))
        fig, ax = plt.subplots(figsize=(9.5, 5.2))
        for class_name in classes:
            sub = agg[agg["class_name"].astype(str) == class_name].sort_values("test_n_samples")
            ax.plot(sub["test_n_samples"], sub["mean_f1"], marker="o", label=class_name)
        ax.set_xlabel("Held-out test cell lines")
        ax.set_ylabel("Mean Test F1")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "per_class_mean_f1_by_test_size.png", dpi=180)
        plt.close(fig)


def write_aggregates(summary: pd.DataFrame, per_class: pd.DataFrame, output_dir: Path) -> None:
    ok = summary[summary["status"] == "ok"].copy()
    if ok.empty:
        return
    aggregate = ok.groupby("test_n_samples", as_index=False).agg(
        n_runs=("run_id", "count"),
        train_n_samples_mean=("train_n_samples", "mean"),
        test_macro_f1_mean=("test_macro_f1", "mean"),
        test_macro_f1_std=("test_macro_f1", "std"),
        test_macro_f1_median=("test_macro_f1", "median"),
        test_macro_f1_present_mean=("test_macro_f1_present_classes", "mean"),
        test_exact_match_mean=("test_exact_match_accuracy", "mean"),
        test_exact_match_std=("test_exact_match_accuracy", "std"),
        test_objectness_f1_mean=("test_objectness_f1", "mean"),
        runtime_sec_mean=("runtime_sec", "mean"),
    )
    aggregate.to_csv(output_dir / "aggregate_by_test_size.tsv", sep="\t", index=False)
    ok.sort_values(["test_macro_f1", "test_exact_match_accuracy", "test_objectness_f1"], ascending=[False, False, False]).to_csv(
        output_dir / "best_models.tsv", sep="\t", index=False
    )
    test_pc = per_class[(per_class["split"] == "test") & (per_class["status"] == "ok")].copy()
    if not test_pc.empty:
        test_pc.groupby(["test_n_samples", "class_name"], as_index=False).agg(
            n_runs=("run_id", "count"),
            support_mean=("support", "mean"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
        ).to_csv(output_dir / "aggregate_per_class_by_test_size.tsv", sep="\t", index=False)


def write_baseline(source_dir: Path, output_dir: Path) -> None:
    metrics_path = source_dir / "metrics_summary.tsv"
    per_class_path = source_dir / "per_class_metrics.tsv"
    if not metrics_path.exists() or not per_class_path.exists():
        return
    metrics = pd.read_csv(metrics_path, sep="\t")
    per_class = pd.read_csv(per_class_path, sep="\t")
    test_metrics = metrics[metrics["split"] == "test"].copy()
    test_per_class = per_class[per_class["split"] == "test"].copy()
    row = {
        "source_dir": str(source_dir),
        "test_macro_f1": float(test_per_class["f1"].astype(float).mean()) if not test_per_class.empty else float("nan"),
        "test_exact_match_accuracy": float(test_metrics["class_exact_match_labeled"].iloc[0]) if not test_metrics.empty else float("nan"),
        "test_any_match_accuracy": float(test_metrics["class_any_match_labeled"].iloc[0]) if not test_metrics.empty else float("nan"),
        "test_objectness_f1": float(test_metrics["objectness_f1"].iloc[0]) if not test_metrics.empty else float("nan"),
    }
    pd.DataFrame([row]).to_csv(output_dir / "baseline_current_general_model.tsv", sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runs").mkdir(exist_ok=True)

    class_names = _split_csv(args.class_names)
    embeddings, metadata, tabular_features, tabular_names, base_cfg = load_source(source_dir, class_names)
    samples = metadata["sample_id"].astype(str).to_numpy()
    targets = _multi_hot_targets(metadata, class_names, subtype_targets="general")
    labeled, background = _label_background_masks(metadata)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    test_sizes = _csv_ints(args.test_sizes)
    seeds = _csv_ints(args.seeds)
    write_baseline(source_dir, output_dir)

    readme = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "class_names": class_names,
        "n_rows": int(len(metadata)),
        "n_samples": int(pd.Series(samples).nunique()),
        "test_sizes": test_sizes,
        "seeds": seeds,
        "primary_metric": "test_macro_f1: mean of per-class F1 over all classes",
        "secondary_metric": "test_exact_match_accuracy: class_exact_match_labeled on labeled test candidates",
        "threshold_calibration": "train split only for each sweep run; no nested LOGO in sweep",
        "tabular_feature_names": tabular_names,
    }
    with (output_dir / "sweep_config.json").open("w", encoding="utf-8") as fh:
        json.dump(readme, fh, indent=2)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[pd.DataFrame] = []
    total_runs = len(test_sizes) * len(seeds)
    run_counter = 0
    for test_n in test_sizes:
        for split_seed in seeds:
            run_counter += 1
            run_id = f"test{test_n:02d}_seed{split_seed:03d}"
            run_dir = output_dir / "runs" / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            log.info("Run %d/%d %s", run_counter, total_runs, run_id)
            try:
                test_samples, split_status = sample_split(
                    samples,
                    targets,
                    labeled,
                    background,
                    int(test_n),
                    int(args.base_seed) + int(split_seed) + 1000 * int(test_n),
                    class_names,
                    int(args.max_split_attempts),
                )
                test_mask = np.isin(samples, np.asarray(test_samples, dtype=object))
                train_mask = ~test_mask
                write_split_table(run_dir, metadata, samples, test_samples, labeled, background)
                helper_args = make_model_args(base_cfg, args, int(args.model_seed_base) + int(split_seed) + 1000 * int(test_n))
                model, training_metrics = _train_model(
                    embeddings,
                    tabular_features,
                    metadata,
                    class_names,
                    train_mask=train_mask,
                    args=helper_args,
                    device=device,
                    epochs=int(helper_args.epochs),
                    patience=int(helper_args.patience),
                    seed_offset=0,
                    log_prefix=run_id,
                )
                training_metrics.to_csv(run_dir / "training_metrics.tsv", sep="\t", index=False)
                raw = predict_model(
                    model,
                    embeddings,
                    tabular_features,
                    metadata,
                    class_names,
                    device=device,
                    batch_size=int(helper_args.batch_size),
                    subtype_targets="general",
                )
                raw = apply_cluster_aggregation(raw, class_names, mode=str(helper_args.cluster_aggregation))
                raw.to_csv(run_dir / "raw_predictions.tsv", sep="\t", index=False)
                import time
                t0 = time.time()
                predictions, threshold_info = calibrate_and_annotate(raw, train_mask, class_names, helper_args, run_dir)
                runtime_sec = time.time() - t0
                predictions["split"] = np.where(test_mask, "test", "train")
                predictions.to_csv(run_dir / "classification_predictions.tsv", sep="\t", index=False)
                predictions.loc[train_mask].to_csv(run_dir / "train_predictions.tsv", sep="\t", index=False)
                predictions.loc[test_mask].to_csv(run_dir / "test_predictions.tsv", sep="\t", index=False)
                row, overall_metrics, per_class_metrics = summarize_run(
                    run_id,
                    int(test_n),
                    int(split_seed),
                    test_samples,
                    train_mask,
                    test_mask,
                    predictions,
                    class_names,
                    threshold_info,
                    training_metrics,
                    runtime_sec,
                )
                overall_metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
                per_class_metrics.to_csv(run_dir / "per_class_metrics.tsv", sep="\t", index=False)
                row["split_status"] = split_status
                summary_rows.append(row)
                pc = per_class_metrics.copy()
                pc.insert(0, "run_id", run_id)
                pc.insert(1, "status", "ok")
                pc.insert(2, "test_n_samples", int(test_n))
                pc.insert(3, "split_seed", int(split_seed))
                per_class_rows.append(pc)
            except Exception as exc:
                log.exception("Run failed: %s", run_id)
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "status": "failed",
                        "test_n_samples": int(test_n),
                        "split_seed": int(split_seed),
                        "error": str(exc),
                    }
                )
            summary = pd.DataFrame(summary_rows)
            summary.to_csv(output_dir / "sweep_summary.tsv", sep="\t", index=False)
            if per_class_rows:
                pd.concat(per_class_rows, ignore_index=True).to_csv(output_dir / "sweep_per_class.tsv", sep="\t", index=False)

    summary = pd.DataFrame(summary_rows)
    per_class = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()
    summary.to_csv(output_dir / "sweep_summary.tsv", sep="\t", index=False)
    if not per_class.empty:
        per_class.to_csv(output_dir / "sweep_per_class.tsv", sep="\t", index=False)
    write_aggregates(summary, per_class, output_dir)
    plot_summary(summary, per_class, output_dir)
    log.info("Done. Wrote sweep outputs to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/candidate_region_classifier_general")
    parser.add_argument("--output_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/general_sweep")
    parser.add_argument("--class_names", default="ecDNA,Seismic_Amplification,chromothripsis,BFB")
    parser.add_argument("--test_sizes", default=DEFAULT_TEST_SIZES)
    parser.add_argument("--seeds", default=DEFAULT_SEEDS)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--model_seed_base", type=int, default=4242)
    parser.add_argument("--max_split_attempts", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000000)
    parser.add_argument("--fast_thresholds", action="store_true", help="Use coarser threshold grids for quick exploratory sweeps.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
