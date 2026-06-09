"""Classification and boundary metrics for complex-SV predictions."""

from __future__ import annotations

import numpy as np

from model.prototypes import PrototypeCache
from utils import interval_iou, reciprocal_overlap


def boundary_iou(pred_start: int, pred_end: int, gt_start: int, gt_end: int) -> float:
    return interval_iou(pred_start, pred_end, gt_start, gt_end)


def classification_metrics(predictions: list[dict], ground_truth: list[dict]) -> dict:
    classes = sorted({str(x["sv_class"]) for x in ground_truth if str(x.get("sv_class", ""))})
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    matched_pred: set[int] = set()

    for gt in ground_truth:
        cls = str(gt["sv_class"])
        best_i = None
        best_ov = 0.0
        for i, pred in enumerate(predictions):
            if i in matched_pred:
                continue
            if str(pred.get("sample_id")) != str(gt.get("sample_id")) or str(pred.get("chrom")) != str(gt.get("chrom")):
                continue
            ov = reciprocal_overlap(int(pred["start_bp"]), int(pred["end_bp"]), int(gt["start_bp"]), int(gt["end_bp"]))
            if ov >= 0.5 and ov > best_ov:
                best_i = i
                best_ov = ov
        if best_i is None:
            counts[cls]["fn"] += 1
            continue
        matched_pred.add(best_i)
        pred_cls = str(predictions[best_i].get("sv_class", predictions[best_i].get("predicted_class", "unknown")))
        if pred_cls == cls:
            counts[cls]["tp"] += 1
        else:
            counts[cls]["fn"] += 1
            if pred_cls in counts:
                counts[pred_cls]["fp"] += 1

    for i, pred in enumerate(predictions):
        if i in matched_pred:
            continue
        pred_cls = str(pred.get("sv_class", pred.get("predicted_class", "")))
        if pred_cls in counts:
            counts[pred_cls]["fp"] += 1

    per_class = {}
    f1s = []
    for cls, c in counts.items():
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1, **c}
        f1s.append(f1)
    return {"per_class": per_class, "macro_f1": float(np.mean(f1s)) if f1s else 0.0}


def calibrate_tau(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    tau_grid: np.ndarray | None = None,
) -> float:
    if tau_grid is None:
        tau_grid = np.linspace(0.1, 0.9, 81)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    best_tau = float(tau_grid[0])
    best_score = -1.0
    for tau in tau_grid:
        scores = []
        for held in np.unique(labels):
            cache = PrototypeCache(embed_dim=embeddings.shape[1], tau=float(tau))
            for class_idx, name in enumerate(class_names):
                if class_idx == held:
                    continue
                idx = np.where(labels == class_idx)[0]
                if len(idx):
                    cache.add_class(name, embeddings[idx])
            preds = [cache.classify(embeddings[i])[0] for i in np.where(labels == held)[0]]
            scores.append(float(np.mean([p == "unknown" for p in preds])) if preds else 0.0)
        score = float(np.mean(scores)) if scores else 0.0
        if score > best_score:
            best_score = score
            best_tau = float(tau)
    return best_tau
