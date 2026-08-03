#!/usr/bin/env python3
"""Fit the localization scorer on all genomes using held-out-CV hyperparameters."""

from __future__ import annotations

import argparse
import json

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model import localization  # noqa: E402


def mode(values: pd.Series) -> object:
    """Return a deterministic mode, breaking ties lexicographically."""
    counts = values.astype(str).value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def select_cv_hyperparameters(cv_runs: Path) -> tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    epoch_rows: list[dict] = []
    calibration_rows: list[pd.DataFrame] = []
    for run in sorted(path for path in cv_runs.iterdir() if path.is_dir()):
        history_path = run / "training_history.tsv"
        calibration_path = run / "event_decoder_calibration.tsv"
        if not calibration_path.exists():
            calibration_path = run / "calibration_selected.tsv"
        if not history_path.exists() or not calibration_path.exists():
            continue
        history = pd.read_csv(history_path, sep="\t")
        if history.empty or "validation_loss" not in history:
            raise ValueError(f"invalid held-out history: {history_path}")
        best = history.loc[history.validation_loss.astype(float).idxmin()]
        epoch_rows.append({
            "fold": run.name,
            "epochs_run": int(history.epoch.max()),
            "best_epoch": int(best.epoch),
            "best_validation_loss": float(best.validation_loss),
        })
        selected = pd.read_csv(calibration_path, sep="\t")
        selected["fold"] = run.name
        calibration_rows.append(selected)

    epochs = pd.DataFrame(epoch_rows)
    calibrations = pd.concat(calibration_rows, ignore_index=True) if calibration_rows else pd.DataFrame()
    if len(epochs) < 2:
        raise ValueError(f"fewer than two complete CV runs found under {cv_runs}")
    if set(calibrations.label.astype(str)) != set(localization.CLASSES):
        raise ValueError("CV calibration files do not contain exactly the model classes")

    fixed_epochs = int(round(float(epochs.best_epoch.median())))
    final_rows = []
    for class_name in localization.CLASSES:
        rows = calibrations[calibrations.label.astype(str) == class_name]
        final_rows.append({
            "label": class_name,
            "threshold": float(rows.threshold.median()),
            "scale": float(rows.scale.median()),
            "region_mode": str(mode(rows.region_mode)),
            "nms_mode": str(mode(rows.nms_mode)),
            "maximum_per_sample": int(round(float(rows.maximum_per_sample.median()))),
            "selection_source": "median_or_mode_of_inner_validation_selected_CV_values",
            "n_cv_folds": int(rows.fold.nunique()),
        })
    return fixed_epochs, pd.DataFrame(final_rows), epochs, calibrations


def fit_fixed_epochs(
    features: np.ndarray,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    quality_array: np.ndarray,
    event_bags: dict[str, list[int]],
    config: dict,
    epochs: int,
    device: torch.device,
) -> tuple[localization.ProposalBagModel, dict, list[dict]]:
    """Train on every row for a fixed CV-selected duration, without validation."""
    all_indices = np.arange(len(candidates))
    normalized, stats = localization.normalize_features(features, all_indices)
    tensor_features = torch.from_numpy(normalized).to(device)
    quality = torch.from_numpy(quality_array).to(device)
    any_overlap = torch.from_numpy((quality_array > 0).any(axis=1)).to(device)
    index_tensor = torch.tensor(all_indices, dtype=torch.long, device=device)
    all_samples = set(candidates.sample_id.astype(str))
    event_rows = localization.event_rows_for_samples(labels, all_samples, event_bags, device)

    model = localization.ProposalBagModel(
        features.shape[1], int(config["hidden_dim"]), float(config["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    history: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(tensor_features)
        loss, parts = localization.split_loss(
            logits, index_tensor, quality, any_overlap, event_rows, config
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        history.append({"epoch": epoch, **{f"train_{key}": value for key, value in parts.items()}})
    return model, stats, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--embedding-bundle", required=True)
    parser.add_argument("--selected-embeddings", required=True)
    parser.add_argument("--tabular-features", required=True)
    parser.add_argument("--cv-runs", required=True, type=Path)
    parser.add_argument("--config", default=str(REPO / "configs/localization.json"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    config = json.loads(Path(args.config).read_text())
    localization.seed_all(int(config["seed"]))
    candidates = pd.read_csv(args.candidates).fillna(0)
    labels = pd.read_csv(args.labels, sep="\t")
    features, candidate_ids = localization.load_features(
        args.embedding_bundle,
        args.selected_embeddings,
        args.tabular_features,
        candidates,
    )
    candidates = (
        candidates.set_index(candidates.candidate_id.astype(str))
        .loc[candidate_ids]
        .reset_index(drop=True)
    )
    quality, _, event_bags = localization.build_supervision(candidates, labels)

    fixed_epochs, final_calibration, cv_epochs, cv_calibrations = select_cv_hyperparameters(
        args.cv_runs
    )
    model, stats, history = fit_fixed_epochs(
        features,
        candidates,
        labels,
        quality,
        event_bags,
        config,
        fixed_epochs,
        device,
    )

    calibration = {
        str(row.label): {
            "label": str(row.label),
            "threshold": float(row.threshold),
            "scale": float(row.scale),
            "region_mode": str(row.region_mode),
            "nms_mode": str(row.nms_mode),
            "maximum_per_sample": int(row.maximum_per_sample),
        }
        for row in final_calibration.itertuples()
    }
    all_samples = sorted(candidates.sample_id.astype(str).unique())
    labeled_samples = sorted(labels.sample_id.astype(str).unique())
    final_config = {**config, "epochs": fixed_epochs, "patience": None}
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "feature_mean": stats["mean"],
        "feature_std": stats["std"],
        "calibration": calibration,
        "config": final_config,
        "input_dim": int(features.shape[1]),
        "train_samples": all_samples,
        "labeled_train_samples": labeled_samples,
        "fixed_epochs": fixed_epochs,
        "epoch_selection_source": "median best epoch from genome-held-out CV",
        "decoder_selection_source": "median/mode of inner-validation-selected CV settings",
        "final_fit": True,
    }, args.output / "model.pt")
    final_calibration.to_csv(args.output / "decoder_calibration.tsv", sep="\t", index=False)
    cv_epochs.to_csv(args.output / "cv_epoch_selection.tsv", sep="\t", index=False)
    cv_calibrations.to_csv(args.output / "cv_decoder_selections.tsv", sep="\t", index=False)
    pd.DataFrame(history).to_csv(args.output / "training_history.tsv", sep="\t", index=False)
    metadata = {
        "model_type": "cv_calibrated_all_genome_localization_final_fit",
        "architecture": "ProposalBagModel with frozen event decoder",
        "training_genomes": len(all_samples),
        "training_sample_ids": all_samples,
        "labeled_training_genomes": len(labeled_samples),
        "training_labels": int(len(labels)),
        "classes": localization.CLASSES,
        "input_dim": int(features.shape[1]),
        "epochs": fixed_epochs,
        "epoch_source": f"median best epoch from {len(cv_epochs)} genome-held-out folds",
        "decoder_source": "per-class median numeric and mode categorical settings selected on each fold's inner validation genomes",
        "outer_test_labels_used_for_selection": False,
        "evaluation_warning": "Final-fit deployment artifact; use benchmark held-out results to estimate performance.",
        "device": str(device),
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print("\nFrozen decoder calibration:")
    print(final_calibration.to_string(index=False))


if __name__ == "__main__":
    main()
