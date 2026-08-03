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
MODEL_CODE = REPO / "final-results/chromosome_model/code"
sys.path.insert(0, str(MODEL_CODE))
from chromosome_model import CLASSES, SAFE_FEATURES, ChromosomeModel, canonical_chrom  # noqa: E402


def prepare_features(embedding_dir: Path, tabular_path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    bundle = np.load(embedding_dir / "embeddings.npz", allow_pickle=True)
    base = np.asarray(bundle["embeddings"], dtype=np.float32)
    meta = pd.read_csv(embedding_dir / "candidate_embeddings.tsv", sep="\t").fillna("")
    if len(meta) != len(base) or base.shape[1] != 402:
        raise ValueError(f"Expected matching metadata and 402D embeddings; got {len(meta)} rows and {base.shape}")
    meta["chrom"] = meta["chrom"].map(canonical_chrom)
    if meta.duplicated(["sample_id", "chrom"]).any():
        raise ValueError("Expected exactly one embedding per sample/chromosome")

    spans = (
        pd.to_numeric(meta["end_bp"], errors="coerce").fillna(0).to_numpy()
        - pd.to_numeric(meta["start_bp"], errors="coerce").fillna(0).to_numpy()
    ).clip(min=1)
    coordinates = np.column_stack([
        np.zeros(len(meta)), np.ones(len(meta)), np.full(len(meta), 0.5), np.ones(len(meta)),
        np.minimum(np.log1p(spans / 1_000_000.0) / 5.0, 1.0),
        np.minimum(np.log1p(spans / 1_000_000.0) / 6.0, 1.0),
        np.zeros(len(meta)), np.ones(len(meta)),
    ]).astype(np.float32)
    embedding_features = np.concatenate([base, base, np.zeros_like(base), coordinates], axis=1)

    tabular = pd.read_csv(tabular_path, sep="\t").fillna(0)
    tabular["chrom"] = tabular["chrom"].map(canonical_chrom)
    tabular = tabular.set_index(["sample_id", "chrom"])
    tabular_rows = []
    for row in meta.itertuples(index=False):
        values = tabular.loc[(str(row.sample_id), str(row.chrom))]
        tabular_rows.append([float(pd.to_numeric(values.get(name, 0), errors="coerce") or 0) for name in SAFE_FEATURES])
    x = np.concatenate([
        embedding_features,
        np.asarray(tabular_rows, dtype=np.float32),
        np.zeros((len(meta), 3), dtype=np.float32),
    ], axis=1)
    if x.shape[1] != 1254:
        raise ValueError(f"Expected 1,254 model inputs, found {x.shape[1]}")
    return x, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--tabular", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO / "final-results/chromosome_model/models/fivefold"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-vote-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    x, meta = prepare_features(Path(args.embedding_dir), Path(args.tabular))
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
