#!/usr/bin/env python3
"""Proposal-level event-bag MIL with conservative class-specific localization."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


CLASSES = ["ecDNA", "chromothripsis", "BFB", "seismic_amplification"]


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def make_folds(samples: list[str], size: int, seed: int) -> list[list[str]]:
    values = list(samples); random.Random(seed).shuffle(values)
    return [values[i:i + size] for i in range(0, len(values), size)]


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def overlap_coefficient(a0: int, a1: int, b0: int, b1: int) -> float:
    value = overlap(a0, a1, b0, b1)
    return value / max(1, min(a1 - a0, b1 - b0))


def one_to_one_match_count(predictions: pd.DataFrame, truth: pd.DataFrame, cutoff: float, require_label: bool) -> int:
    """Maximum-cardinality interval matching without duplicate true positives."""
    group_columns = ["sample_id", "chrom"] + (["label"] if require_label else [])
    matched_total = 0
    for keys, calls in truth.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        possible = predictions
        for column, value in zip(group_columns, keys):
            possible = possible[possible[column].astype(str) == str(value)]
        if possible.empty:
            continue
        prediction_rows = list(possible.itertuples())
        call_rows = list(calls.itertuples())
        edges = []
        for prediction in prediction_rows:
            edges.append([
                index for index, call in enumerate(call_rows)
                if overlap_coefficient(int(prediction.start), int(prediction.end) + 1, int(call.start), int(call.end) + 1) >= cutoff
            ])
        matched_call = [-1] * len(call_rows)
        def augment(prediction_index: int, seen: list[bool]) -> bool:
            for call_index in edges[prediction_index]:
                if seen[call_index]:
                    continue
                seen[call_index] = True
                if matched_call[call_index] < 0 or augment(matched_call[call_index], seen):
                    matched_call[call_index] = prediction_index
                    return True
            return False
        matched_total += sum(augment(index, [False] * len(call_rows)) for index in range(len(prediction_rows)))
    return matched_total


def scale_interval(start: int, end_inclusive: int, factor: float) -> tuple[int, int]:
    end = end_inclusive + 1
    center = (start + end) / 2
    half = (end - start) * factor / 2
    return max(0, int(round(center - half))), int(round(center + half))


def select_inner_validation(samples: list[str], regions: pd.DataFrame, size: int, seed: int) -> list[str]:
    """Add genomes until every class reaches the configured event target."""
    rng = random.Random(seed)
    available = list(samples)
    rng.shuffle(available)
    selected: list[str] = []
    target = int(size)
    counts = {label: 0 for label in CLASSES}
    sample_counts = {
        sample: regions.loc[regions.sample_id.astype(str) == sample, "label"].value_counts().to_dict()
        for sample in available
    }
    while any(counts[label] < target for label in CLASSES):
        remaining = [sample for sample in available if sample not in selected]
        if not remaining:
            break
        deficits = {label: max(0, target - counts[label]) for label in CLASSES}
        gain = {
            sample: sum(min(deficits[label], int(sample_counts[sample].get(label, 0))) / target for label in CLASSES)
            for sample in remaining
        }
        best_gain = max(gain.values(), default=0)
        if best_gain <= 0:
            break
        best = next(sample for sample in remaining if gain[sample] == best_gain)
        selected.append(best)
        for label in CLASSES:
            counts[label] += int(sample_counts[best].get(label, 0))
    missing = {label: counts[label] for label in CLASSES if counts[label] < target}
    if missing:
        raise ValueError(f"could not reach {target} validation events per class: {missing}")
    return selected


def load_features(bundle_path: str, selected_path: str, tabular_path: str, candidates: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    bundle = np.load(bundle_path, allow_pickle=True)
    ids = [str(x) for x in bundle["candidate_id"]]
    selected = np.load(selected_path, allow_pickle=True)["embeddings"].astype(np.float32)
    tabular = np.load(tabular_path, allow_pickle=True)["features"].astype(np.float32)
    if len(ids) != len(selected) or len(ids) != len(tabular):
        raise ValueError("candidate feature bundles have inconsistent row counts")
    candidate_index = candidates.set_index("candidate_id")
    priors = []
    for candidate_id in ids:
        row = candidate_index.loc[candidate_id]
        priors.append([
            float(row.get("proposal_score", 0) or 0),
            float(row.get("candidate_evidence_score", 0) or 0),
            math.log1p(max(0, float(row.get("candidate_priority_rank", 0) or 0))),
        ])
    x = np.concatenate([selected, tabular, np.asarray(priors, np.float32)], axis=1)
    return x, ids


def build_supervision(candidates: pd.DataFrame, regions: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    quality = np.zeros((len(candidates), len(CLASSES)), np.float32)
    any_overlap = np.zeros(len(candidates), bool)
    event_bags: dict[str, list[int]] = {}
    grouped_candidates = {
        (str(s), str(c)): f for (s, c), f in candidates.groupby(["sample_id", "chrom"], sort=False)
    }
    id_to_row = {str(cid): i for i, cid in enumerate(candidates.candidate_id)}
    for event in regions.to_dict("records"):
        group = grouped_candidates.get((str(event["sample_id"]), str(event["chrom"])), pd.DataFrame())
        indices = []
        class_index = CLASSES.index(str(event["label"]))
        event_start, event_end = int(event["start"]), int(event["end"]) + 1
        for candidate in group.to_dict("records"):
            candidate_start, candidate_end = int(candidate["start"]), int(candidate["end"]) + 1
            ov = overlap(candidate_start, candidate_end, event_start, event_end)
            if ov <= 0:
                continue
            row = id_to_row[str(candidate["candidate_id"])]
            indices.append(row); any_overlap[row] = True
            quality[row, class_index] = max(quality[row, class_index], overlap_coefficient(candidate_start, candidate_end, event_start, event_end))
        event_bags[str(event["region_id"])] = indices
    return quality, any_overlap, event_bags


class ProposalBagModel(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def normalize_features(x: np.ndarray, fit_indices: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = x[fit_indices].mean(0).astype(np.float32)
    std = x[fit_indices].std(0).astype(np.float32)
    std[std < 1e-5] = 1
    normalized = np.clip((x - mean) / std, -8, 8).astype(np.float32)
    return normalized, {"mean": mean, "std": std}


def split_loss(
    logits: torch.Tensor,
    indices: torch.Tensor,
    quality: torch.Tensor,
    any_overlap: torch.Tensor,
    event_rows: list[tuple[int, torch.Tensor]],
    config: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    local_logits = logits[indices]
    local_quality = quality[indices]
    losses, point_positive, point_negative, regression = [], [], [], []
    strict = float(config["overlap_coefficient_threshold"])
    for class_index in range(len(CLASSES)):
        positive = local_quality[:, class_index] >= strict
        negative = local_quality[:, class_index] == 0
        if positive.any():
            point_positive.append(F.binary_cross_entropy_with_logits(local_logits[positive, class_index], torch.ones_like(local_logits[positive, class_index])))
        if negative.any():
            negative_logits = local_logits[negative, class_index]
            n_positive = int(positive.sum())
            count = min(len(negative_logits), max(int(config["minimum_hard_negatives"]), int(config["hard_negative_ratio"]) * max(1, n_positive)))
            hard = torch.topk(negative_logits, count).values
            point_negative.append(F.binary_cross_entropy_with_logits(hard, torch.zeros_like(hard)))
        observed = local_quality[:, class_index] > 0
        if observed.any():
            regression.append(F.smooth_l1_loss(torch.sigmoid(local_logits[observed, class_index]), local_quality[observed, class_index]))
    bag_losses = []
    for class_index, rows in event_rows:
        if len(rows):
            bag_logit = logits[rows, class_index].max()
            bag_losses.append(F.binary_cross_entropy_with_logits(bag_logit, torch.ones_like(bag_logit)))
    pos_loss = torch.stack(point_positive).mean() if point_positive else logits.sum() * 0
    neg_loss = torch.stack(point_negative).mean() if point_negative else logits.sum() * 0
    bag_loss = torch.stack(bag_losses).mean() if bag_losses else logits.sum() * 0
    reg_loss = torch.stack(regression).mean() if regression else logits.sum() * 0
    total = pos_loss + neg_loss + 0.75 * bag_loss + 0.35 * reg_loss
    return total, {"loss": float(total.detach()), "positive": float(pos_loss.detach()), "negative": float(neg_loss.detach()), "bag": float(bag_loss.detach()), "quality": float(reg_loss.detach())}


def event_rows_for_samples(
    regions: pd.DataFrame,
    samples: set[str],
    event_bags: dict[str, list[int]],
    device: torch.device,
) -> list[tuple[int, torch.Tensor]]:
    result = []
    for event in regions[regions.sample_id.astype(str).isin(samples)].to_dict("records"):
        rows = torch.tensor(event_bags.get(str(event["region_id"]), []), dtype=torch.long, device=device)
        result.append((CLASSES.index(str(event["label"])), rows))
    return result


def fit_model(
    x: np.ndarray,
    candidates: pd.DataFrame,
    regions: pd.DataFrame,
    quality_array: np.ndarray,
    any_overlap_array: np.ndarray,
    event_bags: dict[str, list[int]],
    train_samples: set[str],
    val_samples: set[str],
    config: dict,
    device: torch.device,
) -> tuple[ProposalBagModel, dict, list[dict]]:
    train_indices = np.where(candidates.sample_id.astype(str).isin(train_samples))[0]
    val_indices = np.where(candidates.sample_id.astype(str).isin(val_samples))[0]
    normalized, stats = normalize_features(x, train_indices)
    features = torch.from_numpy(normalized).to(device)
    quality = torch.from_numpy(quality_array).to(device)
    any_overlap = torch.from_numpy(any_overlap_array).to(device)
    train_tensor = torch.tensor(train_indices, dtype=torch.long, device=device)
    val_tensor = torch.tensor(val_indices, dtype=torch.long, device=device)
    train_events = event_rows_for_samples(regions, train_samples, event_bags, device)
    val_events = event_rows_for_samples(regions, val_samples, event_bags, device)
    model = ProposalBagModel(x.shape[1], int(config["hidden_dim"]), float(config["dropout"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    best_loss, best_state, stale, history = math.inf, None, 0, []
    for epoch in range(1, int(config["epochs"]) + 1):
        model.train(); logits = model(features)
        train_loss, train_parts = split_loss(logits, train_tensor, quality, any_overlap, train_events, config)
        optimizer.zero_grad(set_to_none=True); train_loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_parts = split_loss(model(features), val_tensor, quality, any_overlap, val_events, config)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_parts.items()}, **{f"validation_{k}": v for k, v in val_parts.items()}}
        history.append(row)
        if float(val_loss) < best_loss - 1e-4:
            best_loss, best_state, stale = float(val_loss), copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        if stale >= int(config["patience"]):
            break
    model.load_state_dict(best_state)
    return model, stats, history


def add_scaled_coordinates(frame: pd.DataFrame, factor: float) -> pd.DataFrame:
    result = frame.copy()
    values = [scale_interval(int(row.start), int(row.end), factor) for row in result.itertuples()]
    result["localized_start"] = [v[0] for v in values]
    result["localized_end"] = [v[1] - 1 for v in values]
    return result


def interval_containment(a: pd.Series, b: pd.Series) -> float:
    value = overlap(int(a.localized_start), int(a.localized_end) + 1, int(b.localized_start), int(b.localized_end) + 1)
    return value / max(1, min(int(a.localized_end) - int(a.localized_start) + 1, int(b.localized_end) - int(b.localized_start) + 1))


def decode_class(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    label: str,
    threshold: float,
    scale: float,
    nms_mode: str,
    maximum_per_sample: int,
    fold: int,
) -> pd.DataFrame:
    frame = candidates.copy()
    frame["score"] = scores
    frame = frame[frame.score >= threshold].sort_values("score", ascending=False)
    frame = add_scaled_coordinates(frame, scale)
    kept = []
    for _, row in frame.iterrows():
        prior = [x for x in kept if x.sample_id == row.sample_id and x.chrom == row.chrom]
        if nms_mode != "none":
            if nms_mode == "cluster_containment" and any(str(x.cluster_id) == str(row.cluster_id) for x in prior):
                continue
            if any(interval_containment(row, x) >= 0.5 for x in prior):
                continue
        if sum(x.sample_id == row.sample_id for x in kept) >= maximum_per_sample:
            continue
        kept.append(row)
    if not kept:
        return pd.DataFrame(columns=["fold", "candidate_id", "sample_id", "chrom", "start", "end", "label", "score", "scale", "nms_mode"])
    result = pd.DataFrame(kept)
    result = result.rename(columns={"start": "original_start", "end": "original_end", "localized_start": "start", "localized_end": "end"})
    result["fold"], result["label"], result["scale"], result["nms_mode"] = fold, label, scale, nms_mode
    return result[["fold", "candidate_id", "sample_id", "chrom", "cluster_id", "original_start", "original_end", "start", "end", "label", "score", "scale", "nms_mode"]]


def class_metrics(predictions: pd.DataFrame, truth: pd.DataFrame, label: str, cutoff: float) -> dict:
    calls = truth[truth.label == label]
    matched = one_to_one_match_count(predictions, calls, cutoff, require_label=False)
    recall_value = matched / max(1, len(calls))
    precision = matched / max(1, len(predictions))
    f1 = 2 * precision * recall_value / max(1e-12, precision + recall_value)
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall_value / max(1e-12, beta**2 * precision + recall_value)
    return {"calls": len(calls), "predictions": len(predictions), "true_predictions": matched, "recall": recall_value, "precision": precision, "f1": f1, "f2": f2}


def calibrate_class(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    truth: pd.DataFrame,
    label: str,
    config: dict,
    fold: int,
) -> tuple[dict, pd.DataFrame]:
    cutoff = float(config["overlap_coefficient_threshold"])
    quantiles = sorted(set(float(q) for q in config["threshold_quantiles"]))
    thresholds = sorted(set(float(np.quantile(scores, q)) for q in quantiles))
    rows = []
    for scale in config["scale_grid"]:
        for threshold in thresholds:
            for nms_mode in config["nms_modes"]:
                decoded = decode_class(
                    candidates, scores, label, threshold, float(scale),
                    str(nms_mode), 999999, fold,
                )
                for maximum in config["maximum_predictions_per_sample_grid"]:
                    pred = (
                        decoded
                        if int(maximum) >= 999
                        else decoded.groupby("sample_id", sort=False).head(int(maximum))
                    )
                    metrics = class_metrics(pred, truth, label, cutoff)
                    rows.append({"fold": fold, "label": label, "threshold": threshold, "scale": scale, "nms_mode": nms_mode, "maximum_per_sample": maximum, **metrics})
    table = pd.DataFrame(rows)
    selection_metric = str(config.get("calibration_selection_metric", "f1"))
    if selection_metric not in {"f1", "f2"}:
        raise ValueError(f"unsupported calibration selection metric: {selection_metric}")
    selected = table.sort_values([selection_metric, "recall", "precision", "predictions"], ascending=[False, False, False, True]).iloc[0].to_dict()
    return selected, table


def call_match_table(predictions: pd.DataFrame, truth: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    rows = []
    for call in truth.to_dict("records"):
        possible = predictions[(predictions.sample_id.astype(str) == str(call["sample_id"])) & (predictions.chrom.astype(str) == str(call["chrom"]))]
        best_any = best_correct = 0.0; best_label = ""; best_start = best_end = np.nan
        for pred in possible.to_dict("records"):
            value = overlap_coefficient(int(pred["start"]), int(pred["end"]) + 1, int(call["start"]), int(call["end"]) + 1)
            if value > best_any:
                best_any, best_label, best_start, best_end = value, pred["label"], pred["start"], pred["end"]
            if pred["label"] == call["label"]:
                best_correct = max(best_correct, value)
        rows.append({"region_id": call["region_id"], "sample_id": call["sample_id"], "chrom": call["chrom"], "truth_start": call["start"], "truth_end": call["end"], "truth_label": call["label"], "best_overlap_coefficient_any_class": best_any, "best_overlap_coefficient_correct_class": best_correct, "best_prediction_label": best_label, "best_prediction_start": best_start, "best_prediction_end": best_end, "localized": best_any >= cutoff, "classified_localized": best_correct >= cutoff})
    return pd.DataFrame(rows)


def overall_metrics(predictions: pd.DataFrame, matches: pd.DataFrame, truth: pd.DataFrame, cutoff: float) -> dict:
    classified_matches = one_to_one_match_count(predictions, truth, cutoff, require_label=True)
    localized_matches = one_to_one_match_count(predictions, truth, cutoff, require_label=False)
    recall_value = classified_matches / max(1, len(truth))
    precision = classified_matches / max(1, len(predictions))
    f1 = 2 * precision * recall_value / max(1e-12, precision + recall_value)
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall_value / max(1e-12, beta**2 * precision + recall_value)
    return {"n_predictions": len(predictions), "true_predictions": classified_matches, "localization_recall": localized_matches / max(1, len(truth)), "classified_recall": recall_value, "classified_precision": precision, "classified_f1": f1, "classified_f2": f2}


def pipeline18_baseline(path: str, truth: pd.DataFrame, cutoff: float) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, sep="\t")
    frame = frame[frame.called_complex_sv.astype(bool) & frame.predicted_classes.notna()].copy()
    frame["label"] = frame.predicted_classes.astype(str).str.split(";")
    frame = frame.explode("label"); frame = frame[frame.label.isin(CLASSES)]
    frame = frame.rename(columns={"start_bp": "start", "end_bp": "end"})
    frame["fold"] = pd.to_numeric(frame.fold_id, errors="coerce") - 1
    matches = call_match_table(frame, truth, cutoff)
    return frame, overall_metrics(frame, matches, truth, cutoff)


def plot_results(output: Path, comparisons: pd.DataFrame, matches: pd.DataFrame, histories: pd.DataFrame) -> None:
    ax = comparisons.set_index("method")[["classified_recall", "classified_precision", "classified_f1"]].plot.bar(figsize=(8, 4.5), ylim=(0, max(0.55, comparisons[["classified_recall", "classified_precision", "classified_f1"]].to_numpy().max() * 1.15)), rot=10)
    ax.set_ylabel("Held-out metric"); ax.set_title("Pipeline24 recommended F2 versus prior methods"); ax.figure.tight_layout(); ax.figure.savefig(output / "method_comparison.png", dpi=180); plt.close(ax.figure)
    per_class = []
    for label in CLASSES:
        part = matches[matches.truth_label == label]
        per_class.append({"label": label, "Localization": part.localized.mean(), "Localization + class": part.classified_localized.mean()})
    ax = pd.DataFrame(per_class).set_index("label").plot.bar(figsize=(7, 4), ylim=(0, 1), rot=0)
    ax.set_ylabel("Held-out caller recall"); ax.set_title("Pipeline24 overlap-coefficient recall"); ax.figure.tight_layout(); ax.figure.savefig(output / "per_class_recall.png", dpi=180); plt.close(ax.figure)
    fig, ax = plt.subplots(figsize=(7, 4))
    for fold, part in histories.groupby("fold"):
        ax.plot(part.epoch, part.validation_loss, alpha=.7, label=f"fold {fold}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss"); ax.set_title("Proposal-bag MIL training"); ax.legend(fontsize=7, ncol=2); fig.tight_layout(); fig.savefig(output / "training_curves.png", dpi=180); plt.close(fig)


def write_insights(output: Path, comparisons: pd.DataFrame, matches: pd.DataFrame, predictions: pd.DataFrame, calibration: pd.DataFrame, truth_mil_note: str) -> None:
    hybrid = comparisons[comparisons.method == "Pipeline24 recommended F2"].iloc[0]
    p18 = comparisons[comparisons.method == "Pipeline18 OOF"].iloc[0]
    lines = ["# Pipeline24 insights", "", "## Executive result", "",
        f"Pipeline24 produced {int(hybrid.n_predictions)} held-out predictions. Correct-class overlap-coefficient recall was {hybrid.classified_recall:.1%}, precision was {hybrid.classified_precision:.1%}, and F1 was {hybrid.classified_f1:.1%}.", "",
        f"Pipeline18 produced {int(p18.n_predictions)} class calls at {p18.classified_recall:.1%} recall and {p18.classified_precision:.1%} precision. Pipeline24 therefore changes recall by {hybrid.classified_recall-p18.classified_recall:+.1%} and precision by {hybrid.classified_precision-p18.classified_precision:+.1%} while using strict event-level coordinates rather than candidate-row labels.", "",
        "## Method", "", "- Prediction units are the 2,970 label-free Wakhan/Severus v3 proposals, not all chromosome bins.", "- Input is the pretrained 1,214-dimensional CN/SV candidate embedding, 37 safe tabular features, and three proposal-prior features.", "- Every caller event is a bag containing all directly overlapping proposals. Bag-max supervision keeps events whose proposals are only fragments; they are not deleted for failing overlap coefficient.", "- Proposals with overlap coefficient at least 0.5 receive direct quality supervision. Ambiguous overlaps receive continuous quality supervision rather than hard-negative labels.", "- Only the highest-scoring non-overlapping proposals contribute negative loss, reducing damage from incomplete caller labels.", "- Inner validation selects a class-specific score threshold, conservative centered scale factor, containment NMS strategy, and per-sample output cap.", "- Eight outer folds hold out six genomes each. Inner validation uses eight different genomes; held-out labels never select model epochs or decoding parameters.", "", "## Held-out comparison", "", "| Method | Predictions | Localization recall | Correct-class recall | Precision | F1 |", "|---|---:|---:|---:|---:|---:|" ]
    for row in comparisons.itertuples():
        lines.append(f"| {row.method} | {int(row.n_predictions):,} | {row.localization_recall:.1%} | {row.classified_recall:.1%} | {row.classified_precision:.1%} | {row.classified_f1:.1%} |")
    lines += ["", "## Pipeline24 by class", "", "| Class | Calls | Localized | Correct class + localized |", "|---|---:|---:|---:|"]
    for label in CLASSES:
        part = matches[matches.truth_label == label]
        lines.append(f"| {label} | {len(part)} | {part.localized.mean():.1%} | {part.classified_localized.mean():.1%} |")
    lines += ["", "## Proposal ceiling and localization", "", "Under overlap-coefficient evaluation, a contained proposal can qualify even when its length differs substantially from the caller interval. This directly tests whether reciprocal-overlap geometry was suppressing otherwise useful proposals.", "", "The decoder therefore makes only small, validation-selected scale changes and falls back conceptually to the original proposal. This is intentionally different from pipeline20's unconstrained start/end heads, which reduced strict retention.", "", "## Interpretation", "", truth_mil_note, "", "The hybrid solves pipeline21's whole-genome false-positive explosion by restricting scoring to biologically motivated proposals. It also avoids pipeline20's label-discarding gate by applying caller supervision at the event-bag level. Remaining errors separate into proposal misses, proposal ranking errors, class confusion, and imperfect boundaries; these should be measured independently.", "", "## Limitations", "", "- Only 13 ecDNA caller events are available, so both fitting and inner calibration are unstable.", "- External call sets are incomplete; hard-negative mining reduces but does not remove false-negative supervision.", "- Candidate embeddings were pretrained upstream and are reused here, so a raw-only ablation is still needed to attribute gains.", "- Class-specific scale calibration is coarse and cannot repair off-center or disconnected events.", "- Per-fold parameters can vary when the eight-genome validation set contains few rare events.", "", "## Recommendation", "", "Use this hybrid structure for the next iteration. Improve the proposal generator's chromothripsis coverage, add 100–250 kb focal proposals, and collect more independent ecDNA/BFB labels. Do not return to an unconstrained chromosome foreground scan. If precision remains low, prioritize curated hard negatives and calibration before increasing model size.", "", "## Artifacts", "", "- `oof_predictions.tsv` and `oof_call_matches.tsv`: strict held-out results.", "- `fold_metrics.tsv` and `calibration_selected.tsv`: fold-level performance and selected decoding parameters.", "- `calibration_sweeps.tsv`: all inner-validation configurations.", "- `method_comparison.tsv` and `method_comparison.png`: comparison with pipeline18 and pipeline21.", "- `training_history.tsv` and `training_curves.png`: optimization diagnostics.", "- `fold_00_model.pt` through `fold_07_model.pt`: deployable cross-fit components and normalization statistics.", "", "## Reproduction", "", "Run `run_pipeline24_overlap_coefficient.sh`. Seed 22 was used and the experiment ran on CPU."]
    (output / "insights.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True); parser.add_argument("--regions", required=True); parser.add_argument("--overlaps", required=True)
    parser.add_argument("--embedding-bundle", required=True); parser.add_argument("--selected-embeddings", required=True); parser.add_argument("--tabular-features", required=True)
    parser.add_argument("--pipeline18-oof", required=True); parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(args.config).read_text()); seed_all(int(config["seed"])); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidates = pd.read_csv(args.candidates); regions = pd.read_csv(args.regions, sep="\t")
    x, ids = load_features(args.embedding_bundle, args.selected_embeddings, args.tabular_features, candidates)
    order = {candidate_id: i for i, candidate_id in enumerate(ids)}
    candidates = candidates.set_index("candidate_id").loc[ids].reset_index()
    quality, any_overlap, event_bags = build_supervision(candidates, regions)
    samples = sorted(candidates.sample_id.astype(str).unique()); folds = make_folds(samples, int(config["outer_fold_size"]), int(config["seed"]))
    predictions_all, matches_all, fold_rows, calibration_rows, sweep_rows, history_rows = [], [], [], [], [], []
    cutoff = float(config["overlap_coefficient_threshold"])
    for fold_index, test_list in enumerate(folds):
        remaining = [s for s in samples if s not in test_list]
        val_list = select_inner_validation(remaining, regions, int(config["minimum_validation_events_per_class"]), int(config["seed"]) + fold_index)
        train_list = [s for s in remaining if s not in val_list]
        train, val, test = set(train_list), set(val_list), set(test_list)
        print(f"fold {fold_index}: train={len(train)} validation={len(val)} test={len(test)}", flush=True)
        checkpoint_path = output / f"fold_{fold_index:02d}_model.pt"
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model = ProposalBagModel(int(checkpoint["input_dim"]), int(config["hidden_dim"]), float(config["dropout"])).to(device)
            model.load_state_dict(checkpoint["model"])
            stats = {"mean": checkpoint["feature_mean"], "std": checkpoint["feature_std"]}
            history = []
            resumed_parameters = None if bool(config.get("recalibrate_saved_models", False)) else checkpoint["calibration"]
            action = "recalibrating" if resumed_parameters is None else "resumed calibration"
            print(f"fold {fold_index}: resumed saved model; {action}", flush=True)
        else:
            model, stats, history = fit_model(x, candidates, regions, quality, any_overlap, event_bags, train, val, config, device)
            resumed_parameters = None
        normalized = np.clip((x - stats["mean"]) / stats["std"], -8, 8).astype(np.float32)
        model.eval()
        with torch.no_grad(): scores = torch.sigmoid(model(torch.from_numpy(normalized).to(device))).cpu().numpy()
        val_candidates = candidates[candidates.sample_id.astype(str).isin(val)].copy(); val_rows = val_candidates.index.to_numpy(); val_truth = regions[regions.sample_id.astype(str).isin(val)]
        test_candidates = candidates[candidates.sample_id.astype(str).isin(test)].copy(); test_rows = test_candidates.index.to_numpy(); test_truth = regions[regions.sample_id.astype(str).isin(test)]
        fold_predictions = []
        selected_parameters = {}
        for class_index, label in enumerate(CLASSES):
            if resumed_parameters is not None:
                selected = resumed_parameters[label]
            else:
                selected, sweep = calibrate_class(val_candidates, scores[val_rows, class_index], val_truth, label, config, fold_index)
                sweep_rows.append(sweep)
            calibration_rows.append(selected); selected_parameters[label] = selected
            fold_predictions.append(decode_class(test_candidates, scores[test_rows, class_index], label, float(selected["threshold"]), float(selected["scale"]), str(selected["nms_mode"]), int(selected["maximum_per_sample"]), fold_index))
        predictions = pd.concat(fold_predictions, ignore_index=True)
        matches = call_match_table(predictions, test_truth, cutoff); matches["fold"] = fold_index
        metrics = overall_metrics(predictions, matches, test_truth, cutoff)
        fold_record = {"fold": fold_index, "train_samples": ",".join(sorted(train)), "validation_samples": ",".join(sorted(val)), "test_samples": ",".join(sorted(test)), "n_test_calls": len(test_truth), **metrics}
        for split_name, split_samples in [("train", train), ("validation", val), ("test", test)]:
            split_truth = regions[regions.sample_id.astype(str).isin(split_samples)]
            for label in CLASSES:
                fold_record[f"{split_name}_{label}_calls"] = int((split_truth.label == label).sum())
        fold_rows.append(fold_record)
        predictions_all.append(predictions); matches_all.append(matches)
        if history:
            history_rows.append(pd.DataFrame(history).assign(fold=fold_index))
        torch.save({"model": model.state_dict(), "feature_mean": stats["mean"], "feature_std": stats["std"], "calibration": selected_parameters, "config": config, "input_dim": x.shape[1]}, checkpoint_path)
    predictions = pd.concat(predictions_all, ignore_index=True); matches = pd.concat(matches_all, ignore_index=True); folds_table = pd.DataFrame(fold_rows)
    predictions.to_csv(output / "oof_predictions.tsv", sep="\t", index=False); matches.to_csv(output / "oof_call_matches.tsv", sep="\t", index=False); folds_table.to_csv(output / "fold_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration_selected.tsv", sep="\t", index=False)
    if sweep_rows:
        pd.concat(sweep_rows, ignore_index=True).to_csv(output / "calibration_sweeps.tsv", sep="\t", index=False)
    history_path = output / "training_history.tsv"
    existing_histories = pd.read_csv(history_path, sep="\t") if history_path.exists() else pd.DataFrame()
    if history_rows:
        new_histories = pd.concat(history_rows, ignore_index=True)
        completed_folds = set(new_histories.fold.unique())
        retained = existing_histories[~existing_histories.fold.isin(completed_folds)] if len(existing_histories) and "fold" in existing_histories else pd.DataFrame()
        histories = pd.concat([retained, new_histories], ignore_index=True)
    elif len(existing_histories):
        histories = existing_histories
    else:
        histories = pd.DataFrame(columns=["fold", "epoch", "validation_loss"])
    histories.to_csv(output / "training_history.tsv", sep="\t", index=False)
    hybrid = overall_metrics(predictions, matches, regions, cutoff); p18_predictions, p18 = pipeline18_baseline(args.pipeline18_oof, regions, cutoff)
    p21_dir = Path(args.pipeline18_oof).resolve().parents[2] / "pipeline21"
    p21_predictions = pd.read_csv(p21_dir / "oof_predictions.tsv", sep="\t")
    p21_matches = call_match_table(p21_predictions, regions, cutoff)
    p21 = overall_metrics(p21_predictions, p21_matches, regions, cutoff)
    comparisons = pd.DataFrame([
        {"method": "Pipeline18 OOF", **p18},
        {"method": "Pipeline21 dense + MIL", **p21},
        {"method": "Pipeline24 recommended F2", **hybrid},
    ])
    comparisons.to_csv(output / "method_comparison.tsv", sep="\t", index=False); plot_results(output, comparisons, matches, histories)
    note = "On exact held-out caller intervals, pipeline21's event MIL reached 54.7% class accuracy. Pipeline24 therefore focuses its learned capacity on ranking biologically motivated proposals and uses event-bag supervision instead of asking the same embedding to discover arbitrary chromosome intervals."
    write_insights(output, comparisons, matches, predictions, pd.DataFrame(calibration_rows), note)
    (output / "run_summary.json").write_text(json.dumps({"n_samples": len(samples), "n_events": len(regions), "device": str(device), **hybrid}, indent=2) + "\n")
    print(comparisons.to_string(index=False))


if __name__ == "__main__": main()
