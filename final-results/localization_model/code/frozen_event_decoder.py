#!/usr/bin/env python3
"""Apply frozen Pipeline24 models to proposal-generator event clusters."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
P24 = HERE.parent
RESULTS = P24.parent


def load_pipeline():
    path = P24 / "four_class_f2_recommended.py"
    spec = importlib.util.spec_from_file_location("pipeline24_recommended", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def prepare_events(candidates: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Max-pool candidate scores and retain cluster geometry alternatives."""
    frame = candidates.copy()
    frame["score"] = scores
    keys = ["sample_id", "chrom", "cluster_id"]
    representative_rows = frame.groupby(keys, sort=False, dropna=False).score.idxmax()
    events = frame.loc[representative_rows].copy()
    geometry = frame.groupby(keys, sort=False, dropna=False).agg(
        envelope_start=("start", "min"),
        envelope_end=("end", "max"),
        cluster_size=("candidate_id", "size"),
    ).reset_index()
    events = events.merge(geometry, on=keys, how="left", validate="one_to_one")
    events["event_id"] = (
        events.sample_id.astype(str) + ":" + events.chrom.astype(str) + ":"
        + events.cluster_id.astype(str)
    )
    return events


def decode_events(
    module,
    events: pd.DataFrame,
    label: str,
    threshold: float,
    scale: float,
    region_mode: str,
    nms_mode: str,
    maximum_per_sample: int,
) -> pd.DataFrame:
    frame = events[events.score >= threshold].copy()
    if region_mode == "representative":
        frame["event_start"] = frame.start
        frame["event_end"] = frame.end
    elif region_mode == "envelope":
        frame["event_start"] = frame.envelope_start
        frame["event_end"] = frame.envelope_end
    else:
        raise ValueError(region_mode)
    frame = frame.sort_values("score", ascending=False)
    scaled = [
        module.scale_interval(int(row.event_start), int(row.event_end), float(scale))
        for row in frame.itertuples()
    ]
    frame["localized_start"] = [value[0] for value in scaled]
    frame["localized_end"] = [value[1] - 1 for value in scaled]
    kept = []
    for _, row in frame.iterrows():
        same_locus = [
            prior for prior in kept
            if str(prior.sample_id) == str(row.sample_id)
            and str(prior.chrom) == str(row.chrom)
        ]
        if nms_mode == "containment" and any(
            module.interval_containment(row, prior) >= 0.5 for prior in same_locus
        ):
            continue
        if sum(str(prior.sample_id) == str(row.sample_id) for prior in kept) >= maximum_per_sample:
            continue
        kept.append(row)
    columns = [
        "event_id", "candidate_id", "sample_id", "chrom", "cluster_id",
        "cluster_size", "event_start", "event_end", "localized_start",
        "localized_end", "score",
    ]
    if not kept:
        result = pd.DataFrame(columns=columns)
    else:
        result = pd.DataFrame(kept)[columns]
    result = result.rename(columns={
        "candidate_id": "representative_candidate_id",
        "event_start": "original_start",
        "event_end": "original_end",
        "localized_start": "start",
        "localized_end": "end",
    })
    result["label"] = label
    result["scale"] = scale
    result["region_mode"] = region_mode
    result["nms_mode"] = nms_mode
    return result


def calibrate(module, events, truth, label, config):
    # Compact grid keeps 37-fold LOO practical while spanning the useful range.
    quantiles = (0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99)
    thresholds = sorted(set(float(np.quantile(events.score, q)) for q in quantiles))
    rows = []
    for threshold in thresholds:
        for scale in (0.67, 1.0, 1.5):
            for region_mode in ("representative", "envelope"):
                for nms_mode in ("none", "containment"):
                    decoded = decode_events(
                        module, events, label, threshold, scale, region_mode,
                        nms_mode, 12,
                    )
                    for maximum in (1, 2, 4, 8, 12):
                        predictions = (
                            decoded.groupby("sample_id", sort=False).head(maximum)
                            if len(decoded) else decoded
                        )
                        metrics = module.class_metrics(predictions, truth, label, 0.5)
                        rows.append({
                            "label": label,
                            "threshold": threshold,
                            "scale": scale,
                            "region_mode": region_mode,
                            "nms_mode": nms_mode,
                            "maximum_per_sample": maximum,
                            **metrics,
                        })
    sweep = pd.DataFrame(rows)
    selected = sweep.sort_values(
        ["f1", "recall", "precision", "predictions"],
        ascending=[False, False, False, True],
    ).iloc[0].to_dict()
    return selected, sweep


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    module = load_pipeline()
    config = json.loads((P24 / "config_recommended.json").read_text())
    candidates = pd.read_csv(RESULTS / "pipeline18/merged_candidate_regions.csv")
    truth = pd.read_csv(P24 / "labels.tsv", sep="\t")
    features, ids = module.load_features(
        str(RESULTS / "pipeline18/candidate_region_classifier_general/embeddings.npz"),
        str(RESULTS / "pipeline18/candidate_region_classifier_general/selected_embedding_features.npz"),
        str(RESULTS / "pipeline18/candidate_region_classifier_general/tabular_features.npz"),
        candidates,
    )
    candidates = candidates.set_index("candidate_id").loc[ids].reset_index()

    sample_seed = sum((index + 1) * ord(char) for index, char in enumerate(args.test_sample))
    samples = sorted(candidates.sample_id.astype(str).unique())
    remaining = [sample for sample in samples if sample != args.test_sample]
    validation = set(module.select_inner_validation(
        remaining,
        truth,
        int(config["minimum_validation_events_per_class"]),
        int(config["seed"]) + sample_seed,
    ))

    checkpoint = torch.load(
        P24 / "loo_check/runs" / args.test_sample / "model.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = module.ProposalBagModel(
        int(checkpoint["input_dim"]),
        int(config["hidden_dim"]),
        float(config["dropout"]),
    )
    model.load_state_dict(checkpoint["model"])
    normalized = np.clip(
        (features - checkpoint["feature_mean"]) / checkpoint["feature_std"], -8, 8
    ).astype(np.float32)
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(normalized))).numpy()

    validation_mask = candidates.sample_id.astype(str).isin(validation)
    test_mask = candidates.sample_id.astype(str) == args.test_sample
    validation_truth = truth[truth.sample_id.astype(str).isin(validation)]
    test_truth = truth[truth.sample_id.astype(str) == args.test_sample]

    selected_rows, sweep_rows, prediction_rows = [], [], []
    for class_index, label in enumerate(module.CLASSES):
        validation_events = prepare_events(
            candidates[validation_mask],
            scores[validation_mask.to_numpy(), class_index],
        )
        selected, sweep = calibrate(
            module, validation_events, validation_truth, label, config
        )
        selected_rows.append(selected)
        sweep_rows.append(sweep)
        test_events = prepare_events(
            candidates[test_mask],
            scores[test_mask.to_numpy(), class_index],
        )
        prediction_rows.append(decode_events(
            module,
            test_events,
            label,
            float(selected["threshold"]),
            float(selected["scale"]),
            str(selected["region_mode"]),
            str(selected["nms_mode"]),
            int(selected["maximum_per_sample"]),
        ))

    predictions = pd.concat(prediction_rows, ignore_index=True)
    matches = module.call_match_table(predictions, test_truth, 0.5)
    metrics = module.overall_metrics(predictions, matches, test_truth, 0.5)
    metrics.update({"test_sample": args.test_sample, "n_labels": len(test_truth)})
    pd.DataFrame([metrics]).to_csv(args.output / "metrics.tsv", sep="\t", index=False)
    predictions.to_csv(args.output / "test_predictions.tsv", sep="\t", index=False)
    matches.to_csv(args.output / "test_call_matches.tsv", sep="\t", index=False)
    pd.DataFrame(selected_rows).to_csv(
        args.output / "calibration_selected.tsv", sep="\t", index=False
    )
    pd.concat(sweep_rows, ignore_index=True).to_csv(
        args.output / "calibration_sweeps.tsv", sep="\t", index=False
    )
    print(pd.DataFrame([metrics]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
