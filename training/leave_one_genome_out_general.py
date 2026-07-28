"""Leave-one-genome-out sweep for the candidate-region general classifier."""

from __future__ import annotations

import argparse
import json
import logging
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
    predict_model,
    _train_model,
)

log = logging.getLogger(__name__)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "sample"


def load_manifest_samples(path: str | Path) -> list[str]:
    df = pd.read_csv(path, sep="\t").fillna("")
    if "sample_id" not in df.columns:
        raise ValueError(f"Manifest missing sample_id column: {path}")
    return sorted(df["sample_id"].astype(str).unique().tolist())


def plot_all_manifest_bars(summary_all: pd.DataFrame, output_dir: Path) -> None:
    df = summary_all.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df = df.sort_values(["status", "sample_id"], ascending=[True, True]).reset_index(drop=True)
    metrics = [
        ("test_objectness_f1", "Objectness F1"),
        ("test_macro_f1", "Macro F1"),
        ("test_exact_match_accuracy", "Exact-Match Accuracy"),
        ("test_any_match_accuracy", "Any-Match Accuracy"),
    ]
    x = np.arange(len(df))
    present = df["status"].astype(str).eq("ok").to_numpy()
    colors = np.where(present, "#4E79A7", "#B8B8B8")

    fig, axes = plt.subplots(4, 1, figsize=(16.5, 13.5), sharex=True)
    for ax, (metric, label) in zip(axes, metrics):
        vals = pd.to_numeric(df[metric], errors="coerce").fillna(0.0).to_numpy()
        ax.bar(x, vals, color=colors, width=0.78)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
        for pos in x[~present]:
            ax.text(pos, 0.03, "absent", rotation=90, va="bottom", ha="center", fontsize=7, color="#555555")
    axes[0].set_title("Leave-One-Genome-Out Metrics Across All Manifest Genomes")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(df["sample_id"], rotation=75, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "leave_one_genome_all_48_metric_bars.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16.5, 5.4))
    vals = pd.to_numeric(df["test_macro_f1"], errors="coerce").fillna(0.0).to_numpy()
    ax.bar(x, vals, color=colors, width=0.78)
    ok_vals = pd.to_numeric(df.loc[df["status"].astype(str).eq("ok"), "test_macro_f1"], errors="coerce").dropna()
    if not ok_vals.empty:
        ax.axhline(float(ok_vals.mean()), color="#E15759", linestyle=":", linewidth=1.6, label=f"present mean={ok_vals.mean():.3f}")
        ax.legend(fontsize=8)
    for pos in x[~present]:
        ax.text(pos, 0.03, "absent", rotation=90, va="bottom", ha="center", fontsize=7, color="#555555")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro F1")
    ax.set_title("Leave-One-Genome-Out Macro F1, Including Manifest Genomes With No Candidate Rows")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample_id"], rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "leave_one_genome_macro_f1_all_48.png", dpi=180)
    plt.close(fig)


def plot_ranked_present(summary_present: pd.DataFrame, output_dir: Path) -> None:
    df = summary_present.sort_values("test_macro_f1", ascending=True).reset_index(drop=True)
    if df.empty:
        return
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(15.5, 6.2))
    ax.bar(x, df["test_macro_f1"].astype(float), color="#4E79A7", label="macro F1")
    ax.scatter(x, df["test_exact_match_accuracy"].astype(float), color="#F28E2B", s=28, label="exact match")
    ax.scatter(x, df["test_objectness_f1"].astype(float), color="#59A14F", s=28, label="objectness F1")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Leave-One-Genome-Out Present-Genome Scores Ranked by Macro F1")
    ax.set_xticks(x)
    ax.set_xticklabels(df["sample_id"], rotation=75, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "leave_one_genome_present_43_ranked.png", dpi=180)
    plt.close(fig)


def aggregate_per_class(per_class: pd.DataFrame, output_dir: Path) -> None:
    if per_class.empty:
        return
    test = per_class[(per_class["status"] == "ok") & (per_class["split"] == "test")].copy()
    if test.empty:
        return
    agg = test.groupby("class_name", as_index=False).agg(
        support_mean=("support", "mean"),
        support_median=("support", "median"),
        f1_mean=("f1", "mean"),
        f1_median=("f1", "median"),
        f1_std=("f1", "std"),
        precision_mean=("precision", "mean"),
        recall_mean=("recall", "mean"),
    )
    agg.to_csv(output_dir / "leave_one_genome_per_class_aggregate.tsv", sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    class_names = _split_csv(args.class_names)
    embeddings, metadata, tabular_features, tabular_names, base_cfg = load_source(source_dir, class_names)
    samples = metadata["sample_id"].astype(str).to_numpy()
    present_samples = sorted(pd.unique(samples).tolist())
    manifest_samples = load_manifest_samples(args.manifest)
    absent_samples = sorted(set(manifest_samples).difference(present_samples))
    labeled, background = _label_background_masks(metadata)
    targets = _multi_hot_targets(metadata, class_names, subtype_targets="general")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    config = {
        "source_dir": str(source_dir),
        "manifest": str(args.manifest),
        "output_dir": str(output_dir),
        "n_manifest_samples": int(len(manifest_samples)),
        "n_present_candidate_samples": int(len(present_samples)),
        "absent_candidate_samples": absent_samples,
        "class_names": class_names,
        "tabular_feature_names": tabular_names,
        "threshold_calibration": "train split only per fold",
    }
    with (output_dir / "leave_one_genome_config.json").open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[pd.DataFrame] = []
    for fold_i, held_out in enumerate(present_samples, start=1):
        run_id = f"logo_{safe_name(held_out)}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(exist_ok=True)
        log.info("Fold %d/%d held_out=%s", fold_i, len(present_samples), held_out)
        test_mask = samples == held_out
        train_mask = ~test_mask
        t0 = time.time()
        try:
            write_split_table(run_dir, metadata, samples, [held_out], labeled, background)
            helper_args = make_model_args(base_cfg, args, int(args.model_seed_base) + fold_i)
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
            predictions, threshold_info = calibrate_and_annotate(raw, train_mask, class_names, helper_args, run_dir)
            predictions["split"] = np.where(test_mask, "test", "train")
            predictions.to_csv(run_dir / "classification_predictions.tsv", sep="\t", index=False)
            predictions.loc[train_mask].to_csv(run_dir / "train_predictions.tsv", sep="\t", index=False)
            predictions.loc[test_mask].to_csv(run_dir / "test_predictions.tsv", sep="\t", index=False)
            row, overall_metrics, per_class_metrics = summarize_run(
                run_id,
                1,
                fold_i,
                [held_out],
                train_mask,
                test_mask,
                predictions,
                class_names,
                threshold_info,
                training_metrics,
                time.time() - t0,
            )
            row["sample_id"] = held_out
            row["held_out_sample"] = held_out
            row["n_manifest_candidate_rows"] = int(test_mask.sum())
            row["status"] = "ok"
            overall_metrics.to_csv(run_dir / "metrics_summary.tsv", sep="\t", index=False)
            per_class_metrics.to_csv(run_dir / "per_class_metrics.tsv", sep="\t", index=False)
            summary_rows.append(row)
            pc = per_class_metrics.copy()
            pc.insert(0, "run_id", run_id)
            pc.insert(1, "sample_id", held_out)
            pc.insert(2, "status", "ok")
            per_class_rows.append(pc)
        except Exception as exc:
            log.exception("Fold failed for %s", held_out)
            summary_rows.append(
                {
                    "run_id": run_id,
                    "sample_id": held_out,
                    "held_out_sample": held_out,
                    "status": "failed",
                    "error": str(exc),
                    "test_n_samples": 1,
                }
            )
        pd.DataFrame(summary_rows).to_csv(output_dir / "leave_one_genome_summary_present43.tsv", sep="\t", index=False)
        if per_class_rows:
            pd.concat(per_class_rows, ignore_index=True).to_csv(output_dir / "leave_one_genome_per_class_present43.tsv", sep="\t", index=False)

    present_summary = pd.DataFrame(summary_rows)
    per_class = pd.concat(per_class_rows, ignore_index=True) if per_class_rows else pd.DataFrame()

    absent_rows = []
    for sample in absent_samples:
        row = {
            "run_id": f"absent_{safe_name(sample)}",
            "sample_id": sample,
            "held_out_sample": sample,
            "status": "absent_no_candidate_rows",
            "test_n_samples": 1,
            "train_n_samples": int(len(present_samples)),
            "test_samples": sample,
            "train_candidates": int(len(metadata)),
            "test_candidates": 0,
            "train_labeled": int(labeled.sum()),
            "test_labeled": 0,
            "test_empty": 0,
            "test_macro_f1": 0.0,
            "test_macro_f1_present_classes": 0.0,
            "test_exact_match_accuracy": 0.0,
            "test_any_match_accuracy": 0.0,
            "test_objectness_f1": 0.0,
            "test_objectness_precision": 0.0,
            "test_objectness_recall": 0.0,
        }
        for class_name in class_names:
            row[f"test_support_{class_name}"] = 0
            row[f"test_f1_{class_name}"] = 0.0
            row[f"test_precision_{class_name}"] = 0.0
            row[f"test_recall_{class_name}"] = 0.0
        absent_rows.append(row)
    absent_df = pd.DataFrame(absent_rows)
    summary_all = pd.concat([present_summary, absent_df], ignore_index=True, sort=False)
    summary_all = summary_all.sort_values("sample_id").reset_index(drop=True)

    present_summary.sort_values(["test_macro_f1", "test_exact_match_accuracy"], ascending=[False, False]).to_csv(
        output_dir / "leave_one_genome_best_to_worst_present43.tsv", sep="\t", index=False
    )
    present_summary.to_csv(output_dir / "leave_one_genome_summary_present43.tsv", sep="\t", index=False)
    summary_all.to_csv(output_dir / "leave_one_genome_summary_all48.tsv", sep="\t", index=False)
    if not per_class.empty:
        per_class.to_csv(output_dir / "leave_one_genome_per_class_present43.tsv", sep="\t", index=False)
        aggregate_per_class(per_class, output_dir)

    plot_all_manifest_bars(summary_all, output_dir)
    plot_ranked_present(present_summary[present_summary["status"] == "ok"].copy(), output_dir)

    ok = present_summary[present_summary["status"] == "ok"].copy()
    overview = {
        "n_ok_folds": int(len(ok)),
        "n_absent_manifest_samples": int(len(absent_samples)),
        "absent_manifest_samples": absent_samples,
        "mean_objectness_f1": float(ok["test_objectness_f1"].mean()) if not ok.empty else None,
        "mean_macro_f1": float(ok["test_macro_f1"].mean()) if not ok.empty else None,
        "median_macro_f1": float(ok["test_macro_f1"].median()) if not ok.empty else None,
        "mean_exact_match_accuracy": float(ok["test_exact_match_accuracy"].mean()) if not ok.empty else None,
        "mean_any_match_accuracy": float(ok["test_any_match_accuracy"].mean()) if not ok.empty else None,
        "hardest_present_sample_by_macro_f1": None if ok.empty else str(ok.sort_values("test_macro_f1").iloc[0]["sample_id"]),
        "easiest_present_sample_by_macro_f1": None if ok.empty else str(ok.sort_values("test_macro_f1", ascending=False).iloc[0]["sample_id"]),
    }
    with (output_dir / "leave_one_genome_overview.json").open("w", encoding="utf-8") as fh:
        json.dump(overview, fh, indent=2)
    log.info("Done. Wrote leave-one-genome-out outputs to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/candidate_region_classifier_general")
    parser.add_argument("--manifest", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/complex_sv_manifest.tsv")
    parser.add_argument("--output_dir", default="/data/KolmogorovLab/srinivasanbd/results/pipeline12/general_sweep/leave_one_genome_out")
    parser.add_argument("--class_names", default="ecDNA,Seismic_Amplification,chromothripsis,BFB")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000000)
    parser.add_argument("--fast_thresholds", action="store_true")
    parser.add_argument("--model_seed_base", type=int, default=7000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
