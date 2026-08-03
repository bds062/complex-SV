#!/usr/bin/env python3
"""Apply the packaged localization LOO checkpoints as a vote ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPO = Path(__file__).resolve().parents[1]
MODEL_CODE = REPO / "final-results/localization_model/code"
sys.path.insert(0, str(MODEL_CODE))
import candidate_bag_model as model_code  # noqa: E402
import frozen_event_decoder as event_code  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Candidate CSV used during feature preparation.")
    parser.add_argument("--embedding-bundle", required=True, help="embeddings.npz containing candidate_id.")
    parser.add_argument("--selected-embeddings", required=True, help="1,214D selected_embedding_features.npz.")
    parser.add_argument("--tabular-features", required=True, help="37D tabular_features.npz.")
    parser.add_argument(
        "--checkpoint-dir",
        default=str(REPO / "final-results/localization_model/models/loo"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-vote-fraction", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    candidates = pd.read_csv(args.candidates).fillna(0)
    x, ids = model_code.load_features(args.embedding_bundle, args.selected_embeddings, args.tabular_features, candidates)
    candidates = candidates.set_index(candidates.candidate_id.astype(str)).loc[ids].reset_index(drop=True)
    checkpoints = sorted(Path(args.checkpoint_dir).glob("*/model.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No */model.pt checkpoints found under {args.checkpoint_dir}")

    decoded_runs: list[pd.DataFrame] = []
    for model_index, checkpoint_path in enumerate(checkpoints):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = model_code.ProposalBagModel(
            int(checkpoint["input_dim"]), int(config["hidden_dim"]), float(config["dropout"])
        ).to(device)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        normalized = np.clip(
            (x - np.asarray(checkpoint["feature_mean"])) / np.asarray(checkpoint["feature_std"]), -8, 8
        ).astype(np.float32)
        with torch.no_grad():
            scores = torch.sigmoid(model(torch.from_numpy(normalized).to(device))).cpu().numpy()
        calibration_path = checkpoint_path.parent / "event_decoder_calibration.tsv"
        if not calibration_path.exists():
            raise FileNotFoundError(f"Missing frozen-decoder calibration: {calibration_path}")
        calibration_table = pd.read_csv(calibration_path, sep="\t").set_index("label")
        for class_index, class_name in enumerate(model_code.CLASSES):
            calibration = calibration_table.loc[class_name]
            events = event_code.prepare_events(candidates, scores[:, class_index])
            calls = event_code.decode_events(
                model_code,
                events,
                class_name,
                float(calibration["threshold"]),
                float(calibration["scale"]),
                str(calibration["region_mode"]),
                str(calibration["nms_mode"]),
                int(calibration["maximum_per_sample"]),
            )
            if not calls.empty:
                calls["checkpoint"] = checkpoint_path.parent.name
                decoded_runs.append(calls)

    all_calls = pd.concat(decoded_runs, ignore_index=True) if decoded_runs else pd.DataFrame()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_calls.to_csv(output / "per_checkpoint_calls.tsv", sep="\t", index=False)
    if all_calls.empty:
        pd.DataFrame().to_csv(output / "localization_predictions.tsv", sep="\t", index=False)
        return

    keys = ["sample_id", "chrom", "cluster_id", "label"]
    rows = []
    for values, group in all_calls.groupby(keys, sort=False):
        votes = int(group["checkpoint"].nunique())
        vote_fraction = votes / len(checkpoints)
        if vote_fraction < args.minimum_vote_fraction:
            continue
        representative = group.sort_values(
            ["score", "representative_candidate_id"], ascending=[False, True]
        ).iloc[0]
        rows.append({
            **dict(zip(keys, values)),
            "candidate_id": representative["representative_candidate_id"],
            "start": int(np.median(group["start"])),
            "end": int(np.median(group["end"])),
            "mean_score": float(group["score"].mean()),
            "vote_count": votes,
            "n_checkpoints": len(checkpoints),
            "vote_fraction": vote_fraction,
        })
    predictions = pd.DataFrame(rows)
    if not predictions.empty:
        predictions = predictions.sort_values(["sample_id", "mean_score"], ascending=[True, False])
    predictions.to_csv(output / "localization_predictions.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
