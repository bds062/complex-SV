#!/usr/bin/env python3
"""Train one held-out-genome localization scorer and frozen event decoder."""

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
from model import localization as model_code  # noqa: E402
from model import event_decoder as event_code  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--embedding-bundle", required=True)
    parser.add_argument("--selected-embeddings", required=True)
    parser.add_argument("--tabular-features", required=True)
    parser.add_argument("--test-sample", required=True)
    parser.add_argument("--config", default=str(REPO / "configs/localization.json"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text())
    sample_seed = sum((index + 1) * ord(char) for index, char in enumerate(args.test_sample))
    model_code.seed_all(int(config["seed"]) + sample_seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    candidates = pd.read_csv(args.candidates)
    labels = pd.read_csv(args.labels, sep="\t")
    features, ids = model_code.load_features(
        args.embedding_bundle, args.selected_embeddings, args.tabular_features, candidates
    )
    candidates = candidates.set_index("candidate_id").loc[ids].reset_index()
    quality, any_overlap, event_bags = model_code.build_supervision(candidates, labels)

    samples = sorted(candidates.sample_id.astype(str).unique())
    if args.test_sample not in set(labels.sample_id.astype(str)):
        raise ValueError("--test-sample must be a genome with at least one training label")
    remaining = [sample for sample in samples if sample != args.test_sample]
    validation = model_code.select_inner_validation(
        remaining,
        labels,
        int(config["minimum_validation_events_per_class"]),
        int(config["seed"]) + sample_seed,
    )
    train = [sample for sample in remaining if sample not in validation]
    train_set, validation_set = set(train), set(validation)

    model, stats, history = model_code.fit_model(
        features,
        candidates,
        labels,
        quality,
        any_overlap,
        event_bags,
        train_set,
        validation_set,
        config,
        device,
    )
    normalized = np.clip(
        (features - stats["mean"]) / stats["std"], -8, 8
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(normalized).to(device))).cpu().numpy()

    validation_mask = candidates.sample_id.astype(str).isin(validation_set).to_numpy()
    test_mask = (candidates.sample_id.astype(str) == args.test_sample).to_numpy()
    validation_truth = labels[labels.sample_id.astype(str).isin(validation_set)]
    test_truth = labels[labels.sample_id.astype(str) == args.test_sample]

    calibration_rows = []
    sweep_rows = []
    prediction_rows = []
    calibration = {}
    for class_index, class_name in enumerate(model_code.CLASSES):
        validation_events = event_code.prepare_events(
            candidates.loc[validation_mask], scores[validation_mask, class_index]
        )
        selected, sweep = event_code.calibrate(
            model_code, validation_events, validation_truth, class_name, config
        )
        calibration[class_name] = selected
        calibration_rows.append(selected)
        sweep_rows.append(sweep)
        test_events = event_code.prepare_events(
            candidates.loc[test_mask], scores[test_mask, class_index]
        )
        prediction_rows.append(event_code.decode_events(
            model_code,
            test_events,
            class_name,
            float(selected["threshold"]),
            float(selected["scale"]),
            str(selected["region_mode"]),
            str(selected["nms_mode"]),
            int(selected["maximum_per_sample"]),
        ))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    matches = model_code.call_match_table(predictions, test_truth, 0.5)
    metrics = model_code.overall_metrics(predictions, matches, test_truth, 0.5)
    metrics.update({"test_sample": args.test_sample, "n_labels": len(test_truth)})

    predictions.to_csv(output / "test_predictions.tsv", sep="\t", index=False)
    matches.to_csv(output / "test_call_matches.tsv", sep="\t", index=False)
    pd.DataFrame([metrics]).to_csv(output / "metrics.tsv", sep="\t", index=False)
    pd.DataFrame(calibration_rows).to_csv(
        output / "event_decoder_calibration.tsv", sep="\t", index=False
    )
    pd.concat(sweep_rows, ignore_index=True).to_csv(
        output / "event_decoder_sweep.tsv", sep="\t", index=False
    )
    pd.DataFrame(history).to_csv(output / "training_history.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"split": "train", "samples": ",".join(sorted(train_set))},
        {"split": "validation", "samples": ",".join(sorted(validation_set))},
        {"split": "test", "samples": args.test_sample},
    ]).to_csv(output / "split_audit.tsv", sep="\t", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "feature_mean": stats["mean"],
            "feature_std": stats["std"],
            "calibration": calibration,
            "config": config,
            "input_dim": features.shape[1],
        },
        output / "model.pt",
    )
    (output / "run_summary.json").write_text(json.dumps({
        **metrics,
        "device": str(device),
        "train_samples": sorted(train_set),
        "validation_samples": sorted(validation_set),
    }, indent=2) + "\n")
    print(pd.DataFrame([metrics]).to_string(index=False))


if __name__ == "__main__":
    main()
