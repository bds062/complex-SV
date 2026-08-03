#!/usr/bin/env python3
"""Event-cluster pooling and deterministic localization decoding."""

from __future__ import annotations


import numpy as np
import pandas as pd





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
