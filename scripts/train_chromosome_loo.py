#!/usr/bin/env python3
"""Train one held-out-genome chromosome classifier."""

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
from genomic_features.chromosome_features import assemble_model_inputs  # noqa: E402
from architectures.chromosome import (  # noqa: E402
    CLASSES,
    calibrate_thresholds,
    canonical_chrom,
    train_model,
    validation_samples,
)


def build_targets(metadata: pd.DataFrame, label_path: str) -> np.ndarray:
    labels = pd.read_csv(label_path, sep="\t")
    required = {"sample_id", "chrom", "label"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Label table is missing columns: {sorted(missing)}")
    labels["chrom"] = labels["chrom"].map(canonical_chrom)
    observed = {
        (str(row.sample_id), str(row.chrom), str(row.label))
        for row in labels.itertuples(index=False)
    }
    unknown = sorted(set(labels["label"].astype(str)).difference(CLASSES))
    if unknown:
        raise ValueError(f"Unknown chromosome labels: {unknown}")
    targets = np.zeros((len(metadata), len(CLASSES)), dtype=np.float32)
    for row_index, row in metadata.reset_index(drop=True).iterrows():
        for class_index, class_name in enumerate(CLASSES):
            targets[row_index, class_index] = float(
                (str(row["sample_id"]), str(row["chrom"]), class_name) in observed
            )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--tabular", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--test-sample", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlabeled-weight", type=float, default=0.10)
    parser.add_argument("--validation-events-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=27)
    args = parser.parse_args()

    features, metadata = assemble_model_inputs(args.embedding_dir, args.tabular)
    targets = build_targets(metadata, args.labels)
    all_samples = sorted(metadata["sample_id"].astype(str).unique())
    labeled_samples = sorted(
        metadata.loc[targets.sum(axis=1) > 0, "sample_id"].astype(str).unique()
    )
    if args.test_sample not in labeled_samples:
        raise ValueError("--test-sample must be a genome with at least one label")

    calibration_pool = [
        sample for sample in labeled_samples if sample != args.test_sample
    ]
    validation = validation_samples(
        metadata,
        targets,
        calibration_pool,
        target_per_class=args.validation_events_per_class,
    )
    train_samples = [
        sample for sample in all_samples
        if sample != args.test_sample and sample not in validation
    ]
    model, normalization, history = train_model(
        features,
        metadata,
        targets,
        train_samples,
        validation,
        seed=args.seed,
        unlabeled_weight=args.unlabeled_weight,
    )

    normalized = np.clip(
        (features - normalization["mean"]) / normalization["std"], -8, 8
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(torch.as_tensor(normalized))
        ).numpy()
    validation_mask = metadata.sample_id.astype(str).isin(validation).to_numpy()
    test_mask = (
        metadata.sample_id.astype(str) == args.test_sample
    ).to_numpy()
    thresholds = calibrate_thresholds(
        targets[validation_mask], probabilities[validation_mask]
    )

    rows = []
    test_metadata = metadata.loc[test_mask].reset_index(drop=True)
    test_probabilities = probabilities[test_mask]
    test_targets = targets[test_mask]
    for row_index, row in test_metadata.iterrows():
        for class_index, class_name in enumerate(CLASSES):
            rows.append({
                "sample_id": row["sample_id"],
                "chrom": row["chrom"],
                "class": class_name,
                "probability": float(test_probabilities[row_index, class_index]),
                "threshold": float(thresholds[class_index]),
                "predicted": int(
                    test_probabilities[row_index, class_index]
                    >= thresholds[class_index]
                ),
                "truth": int(test_targets[row_index, class_index]),
            })

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "predictions.tsv", sep="\t", index=False)
    history.to_csv(output / "training_history.tsv", sep="\t", index=False)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": features.shape[1],
        "hidden_dim": 96,
        "dropout": 0.35,
        "classes": CLASSES,
        "mean": normalization["mean"],
        "std": normalization["std"],
        "thresholds": thresholds,
        "train_samples": train_samples,
        "validation_samples": validation,
        "test_samples": [args.test_sample],
        "unlabeled_weight": args.unlabeled_weight,
    }
    torch.save(checkpoint, output / "model.pt")
    summary = {
        "test_sample": args.test_sample,
        "train_samples": train_samples,
        "validation_samples": validation,
        "thresholds": dict(zip(CLASSES, map(float, thresholds))),
        "best_validation_loss": normalization["best_validation_loss"],
        "unlabeled_weight": args.unlabeled_weight,
    }
    (output / "run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
