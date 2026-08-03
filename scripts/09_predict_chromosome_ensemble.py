#!/usr/bin/env python3
"""Apply the packaged five-fold chromosome classifiers to new chromosome embeddings."""

from __future__ import annotations

import argparse
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
        "--checkpoint-dir",
        default=str(REPO / "models/chromosome_fivefold"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-vote-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    x, meta = assemble_model_inputs(args.embedding_dir, args.tabular)
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No .pt checkpoints found under {args.checkpoint_dir}")

    probabilities = []
    binary_calls = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = ChromosomeModel(
            input_dim=int(checkpoint["input_dim"]),
            hidden=int(checkpoint.get("hidden_dim", 96)),
            dropout=float(checkpoint.get("dropout", 0.35)),
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()
        normalized = np.clip(
            (x - np.asarray(checkpoint["mean"])) / np.asarray(checkpoint["std"]), -8, 8
        ).astype(np.float32)
        with torch.no_grad():
            prob = torch.sigmoid(model(torch.from_numpy(normalized).to(device))).cpu().numpy()
        probabilities.append(prob)
        binary_calls.append(prob >= np.asarray(checkpoint["thresholds"])[None, :])

    probability_array = np.stack(probabilities)
    vote_array = np.stack(binary_calls)
    mean_probability = probability_array.mean(axis=0)
    vote_fraction = vote_array.mean(axis=0)
    rows = []
    for row_index, record in meta.reset_index(drop=True).iterrows():
        for class_index, class_name in enumerate(CLASSES):
            rows.append({
                "sample_id": record["sample_id"],
                "chrom": record["chrom"],
                "class": class_name,
                "mean_probability": float(mean_probability[row_index, class_index]),
                "vote_fraction": float(vote_fraction[row_index, class_index]),
                "predicted": int(vote_fraction[row_index, class_index] >= args.minimum_vote_fraction),
                "n_checkpoints": len(checkpoints),
            })
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "chromosome_predictions.tsv", sep="\t", index=False)
    table[table.predicted.astype(bool)].to_csv(output / "predicted_complex_sv.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
