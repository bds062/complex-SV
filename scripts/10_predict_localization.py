#!/usr/bin/env python3
"""Apply the all-48 final-fit localization model to prepared candidates."""

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
from architectures import event_decoder as decoder  # noqa: E402
from architectures import localization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--embedding-bundle", required=True)
    parser.add_argument("--selected-embeddings", required=True)
    parser.add_argument("--tabular-features", required=True)
    parser.add_argument(
        "--checkpoint",
        default=str(REPO / "models/localization_all48/model.pt"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    candidates = pd.read_csv(args.candidates).fillna(0)
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

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = localization.ProposalBagModel(
        int(checkpoint["input_dim"]),
        int(config["hidden_dim"]),
        float(config["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    normalized = np.clip(
        (features - np.asarray(checkpoint["feature_mean"]))
        / np.asarray(checkpoint["feature_std"]),
        -8,
        8,
    ).astype(np.float32)
    with torch.no_grad():
        scores = torch.sigmoid(
            model(torch.from_numpy(normalized).to(device))
        ).cpu().numpy()

    calibration = checkpoint.get("calibration")
    if not calibration:
        raise ValueError("checkpoint does not contain final-fit decoder calibration")
    calls: list[pd.DataFrame] = []
    score_table = candidates[["candidate_id", "sample_id", "chrom", "start", "end"]].copy()
    for class_index, class_name in enumerate(localization.CLASSES):
        score_table[f"score_{class_name}"] = scores[:, class_index]
        values = calibration[class_name]
        events = decoder.prepare_events(candidates, scores[:, class_index])
        class_calls = decoder.decode_events(
            localization,
            events,
            class_name,
            float(values["threshold"]),
            float(values["scale"]),
            str(values["region_mode"]),
            str(values["nms_mode"]),
            int(values["maximum_per_sample"]),
        )
        if not class_calls.empty:
            class_calls["threshold"] = float(values["threshold"])
            class_calls["maximum_per_sample"] = int(values["maximum_per_sample"])
            calls.append(class_calls)

    prediction_columns = [
        "event_id", "representative_candidate_id", "sample_id", "chrom",
        "cluster_id", "cluster_size", "original_start", "original_end",
        "start", "end", "score", "label", "scale", "region_mode",
        "nms_mode", "threshold", "maximum_per_sample",
    ]
    predictions = (
        pd.concat(calls, ignore_index=True).reindex(columns=prediction_columns)
        if calls else pd.DataFrame(columns=prediction_columns)
    )
    if not predictions.empty:
        predictions = predictions.sort_values(
            ["sample_id", "score"], ascending=[True, False]
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output / "localized_complex_sv.tsv", sep="\t", index=False)
    score_table.to_csv(output / "candidate_scores.tsv", sep="\t", index=False)
    summary = {
        "checkpoint": str(checkpoint_path),
        "training_genomes": len(checkpoint.get("train_samples", [])),
        "candidate_regions": len(candidates),
        "predicted_events": len(predictions),
        "device": str(device),
    }
    (output / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
