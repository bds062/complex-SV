#!/usr/bin/env python3
"""Apply the all-48 final-fit chromosome model to prepared chromosome features."""

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
from architectures.chromosome import CLASSES, ChromosomeModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--tabular", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "models/chromosome_all48/model.pt"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    features, metadata = assemble_model_inputs(args.embedding_dir, args.tabular)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if list(checkpoint["classes"]) != CLASSES:
        raise ValueError("checkpoint class order does not match the packaged model")
    model = ChromosomeModel(
        input_dim=int(checkpoint["input_dim"]),
        hidden=int(checkpoint.get("hidden_dim", 96)),
        dropout=float(checkpoint.get("dropout", 0.35)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    normalized = np.clip(
        (features - np.asarray(checkpoint["mean"])) / np.asarray(checkpoint["std"]),
        -8,
        8,
    ).astype(np.float32)
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(torch.from_numpy(normalized).to(device))
        ).cpu().numpy()
    thresholds = np.asarray(checkpoint["thresholds"], dtype=np.float32)

    rows = []
    for row_index, record in metadata.reset_index(drop=True).iterrows():
        for class_index, class_name in enumerate(CLASSES):
            probability = float(probabilities[row_index, class_index])
            threshold = float(thresholds[class_index])
            rows.append({
                "sample_id": record["sample_id"],
                "chrom": record["chrom"],
                "class": class_name,
                "probability": probability,
                "threshold": threshold,
                "predicted": int(probability >= threshold),
            })
    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "chromosome_predictions.tsv", sep="\t", index=False)
    table.loc[table.predicted.astype(bool)].to_csv(
        output / "predicted_complex_sv.tsv", sep="\t", index=False
    )
    summary = {
        "checkpoint": str(checkpoint_path),
        "training_genomes": len(checkpoint.get("train_samples", [])),
        "chromosomes": len(metadata),
        "predicted_chromosome_classes": int(table.predicted.sum()),
        "device": str(device),
    }
    (output / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
