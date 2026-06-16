"""Multi-label objectness plus type-head training on frozen complex-SV embeddings.

This is a parallel classifier-head experiment for regions that can carry more
than one complex-SV label. Rows with the same sample/chrom/arm/start/end are
merged into one candidate with a multi-hot type target.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from discovery import embed_corpus
from training.train_classifier_head import enforce_candidate_resolution, load_embedding_table
from utils import set_seed

log = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = (
    "BFB",
    "chromothripsis",
    "seismic_amplification",
    "TIC",
)
CLASS_ALIASES = {
    "non_canonical_BFB": "BFB",
    "non_canonical_chromothripsis": "chromothripsis",
}
SCAN_EVIDENCE_VALUES = {"chromosome_scan", "chromosome_arm_scan"}


def _canonical_class_name(value: object) -> str:
    text = str(value).strip()
    return CLASS_ALIASES.get(text, text)


def _split_classes(value: object, *, collapse_aliases: bool = True) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []
    classes: list[str] = []
    for part in text.replace(",", ";").split(";"):
        cls = part.strip()
        if not cls:
            continue
        classes.append(_canonical_class_name(cls) if collapse_aliases else cls)
    return list(dict.fromkeys(classes))


def _sample_ids(metadata: pd.DataFrame) -> np.ndarray:
    if "sample_id" in metadata:
        return metadata["sample_id"].astype(str).to_numpy()
    return np.asarray(["sample"] * len(metadata), dtype=object)


def _label_background_masks(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    classes = metadata["sv_classes"].map(_split_classes) if "sv_classes" in metadata else metadata.get("sv_class", pd.Series([""] * len(metadata))).map(_split_classes)
    labeled = classes.map(bool).to_numpy()
    evidence = metadata["evidence"].astype(str) if "evidence" in metadata else pd.Series([""] * len(metadata))
    background = (~labeled) & evidence.isin(SCAN_EVIDENCE_VALUES).to_numpy()
    return labeled, background


def aggregate_multilabel_candidates(embeddings: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Merge duplicate region rows into one multi-label candidate."""
    meta = metadata.copy().reset_index(drop=True).fillna("")
    for col in ["sample_id", "chrom", "arm", "start_bp", "end_bp"]:
        if col not in meta.columns:
            meta[col] = "" if col in {"sample_id", "chrom", "arm"} else 0
    meta["start_bp"] = pd.to_numeric(meta["start_bp"], errors="coerce").fillna(0).astype(int)
    meta["end_bp"] = pd.to_numeric(meta["end_bp"], errors="coerce").fillna(0).astype(int)
    meta["_orig_idx"] = np.arange(len(meta), dtype=int)

    group_cols = ["sample_id", "chrom", "arm", "start_bp", "end_bp"]
    emb_rows: list[np.ndarray] = []
    out_rows: list[dict[str, Any]] = []
    for _key, grp in meta.groupby(group_cols, sort=False, dropna=False):
        idx = grp["_orig_idx"].astype(int).to_numpy()
        row = grp.iloc[0].drop(labels=["_orig_idx"]).to_dict()
        raw_class_values: list[str] = []
        class_values: list[str] = []
        for value in grp.get("sv_class", pd.Series([""] * len(grp))).tolist():
            raw_parts = _split_classes(value, collapse_aliases=False)
            raw_class_values.extend(raw_parts)
            class_values.extend(_canonical_class_name(part) for part in raw_parts)
        raw_classes = sorted(set(raw_class_values), key=raw_class_values.index)
        classes = sorted(set(class_values), key=class_values.index)
        candidate_ids = [str(x) for x in grp.get("candidate_id", pd.Series(dtype=str)).astype(str).tolist() if str(x)]
        label_ids = [str(x) for x in grp.get("label_id", pd.Series(dtype=str)).astype(str).tolist() if str(x)]
        row["candidate_id"] = ";".join(dict.fromkeys(candidate_ids)) or str(row.get("candidate_id", ""))
        row["label_id"] = ";".join(dict.fromkeys(label_ids))
        row["raw_sv_class"] = ";".join(raw_classes)
        row["raw_sv_classes"] = ";".join(raw_classes)
        row["sv_class"] = ";".join(classes)
        row["sv_classes"] = ";".join(classes)
        row["n_merged_rows"] = int(len(grp))
        row["merged_candidate_ids"] = ";".join(dict.fromkeys(candidate_ids))
        row["merged_label_ids"] = ";".join(dict.fromkeys(label_ids))
        emb_rows.append(np.asarray(embeddings[idx[0]], dtype=np.float32))
        out_rows.append(row)

    return np.stack(emb_rows, axis=0).astype(np.float32), pd.DataFrame(out_rows).fillna("")


def _parse_class_names(raw: str | None, metadata: pd.DataFrame) -> list[str]:
    observed: list[str] = []
    if "sv_classes" in metadata:
        for value in metadata["sv_classes"].tolist():
            observed.extend(_split_classes(value))
    if raw:
        names = list(dict.fromkeys(_canonical_class_name(part.strip()) for part in str(raw).split(",") if part.strip()))
    else:
        names = list(DEFAULT_CLASS_NAMES)
    unknown = sorted(set(observed).difference(names))
    if unknown:
        raise ValueError(f"Observed labels not present in --class_names: {unknown}; class_names={names}")
    return names


def _multi_hot_targets(metadata: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    targets = np.zeros((len(metadata), len(class_names)), dtype=np.float32)
    for row_i, value in enumerate(metadata.get("sv_classes", pd.Series([""] * len(metadata))).tolist()):
        for class_name in _split_classes(value):
            if class_name in class_to_idx:
                targets[row_i, class_to_idx[class_name]] = 1.0
    return targets


def _class_counts(targets: np.ndarray, class_names: list[str]) -> dict[str, int]:
    return {name: int(targets[:, i].sum()) for i, name in enumerate(class_names)}


class MultiLabelComplexSVClassifierHead(nn.Module):
    """Objectness plus independent sigmoid type logits for frozen embeddings."""

    def __init__(self, in_dim: int, num_classes: int, hidden_dims: list[int], dropout: float = 0.2, activation: str = "relu"):
        super().__init__()
        dims = [int(x) for x in hidden_dims if int(x) > 0]
        layers: list[nn.Module] = []
        prev = int(in_dim)
        act: type[nn.Module] = nn.GELU if str(activation).lower() == "gelu" else nn.ReLU
        for hidden in dims:
            layers.extend([nn.Linear(prev, hidden), nn.LayerNorm(hidden), act(), nn.Dropout(float(dropout))])
            prev = hidden
        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.objectness = nn.Linear(prev, 1)
        self.type_classifier = nn.Linear(prev, int(num_classes))

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(embedding)
        return self.objectness(h).squeeze(-1), self.type_classifier(h)


def _hidden_dims(raw: str) -> list[int]:
    text = str(raw or "").strip()
    if not text or text == "0":
        return []
    return [int(part) for part in text.split(",") if part.strip()]


def _objectness_weights(metadata: pd.DataFrame, train_idx: np.ndarray, labeled: np.ndarray, background: np.ndarray, background_weight: float) -> np.ndarray:
    weights = np.ones(len(train_idx), dtype=np.float32)
    train_background = background[train_idx]
    weights[train_background] = float(background_weight)
    if train_background.any():
        samples = _sample_ids(metadata)[train_idx]
        for sample in sorted(pd.unique(samples[train_background])):
            sample_bg = train_background & (samples == sample)
            if sample_bg.any():
                weights[sample_bg] = float(background_weight) / float(sample_bg.sum())
    train_labeled = labeled[train_idx]
    if train_labeled.any():
        weights[train_labeled] = 1.0
    return weights.astype(np.float32)


def _pos_weight(targets: np.ndarray, labeled: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if mode == "none" or not bool(labeled.any()):
        return None
    y = targets[labeled]
    pos = y.sum(axis=0).astype(np.float32)
    neg = max(float(y.shape[0]), 1.0) - pos
    weights = np.ones(y.shape[1], dtype=np.float32)
    present = pos > 0
    if mode == "inverse":
        weights[present] = neg[present] / np.maximum(pos[present], 1.0)
    elif mode == "inverse_sqrt":
        weights[present] = np.sqrt(neg[present] / np.maximum(pos[present], 1.0))
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    weights[~present] = 1.0
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def _smooth_binary_targets(targets: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    smoothing = float(label_smoothing)
    if smoothing <= 0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


def _make_model(args: argparse.Namespace, input_dim: int, n_classes: int, device: torch.device) -> MultiLabelComplexSVClassifierHead:
    return MultiLabelComplexSVClassifierHead(
        in_dim=int(input_dim),
        num_classes=int(n_classes),
        hidden_dims=_hidden_dims(args.hidden_dims),
        dropout=float(args.dropout),
        activation=str(args.activation),
    ).to(device)


def _train_model(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    train_mask: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    patience: int,
    seed_offset: int = 0,
    log_prefix: str = "train",
) -> tuple[MultiLabelComplexSVClassifierHead, pd.DataFrame]:
    set_seed(int(args.seed) + int(seed_offset))
    targets_np = _multi_hot_targets(metadata, class_names)
    labeled, background = _label_background_masks(metadata)
    usable_mask = train_mask & (labeled | background)
    train_idx = np.where(usable_mask)[0]
    if train_idx.size == 0:
        raise RuntimeError(f"No usable labeled/background rows for {log_prefix}")
    if not bool(labeled[train_idx].any()):
        raise RuntimeError(f"No positive labels available for {log_prefix}")

    x_train = torch.as_tensor(embeddings[train_idx], dtype=torch.float32, device=device)
    objectness_targets = torch.as_tensor(labeled[train_idx].astype(np.float32), dtype=torch.float32, device=device)
    type_targets = torch.as_tensor(targets_np[train_idx], dtype=torch.float32, device=device)
    type_mask_np = labeled[train_idx]
    type_mask = torch.as_tensor(type_mask_np, dtype=torch.bool, device=device)
    objectness_weights = torch.as_tensor(_objectness_weights(metadata, train_idx, labeled, background, float(args.background_weight)), dtype=torch.float32, device=device)
    pos_weight = _pos_weight(targets_np[train_idx], type_mask_np, str(args.class_weighting), device)

    model = _make_model(args, embeddings.shape[1], len(class_names), device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = int(patience)
    rows: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        objectness_logits, type_logits = model(x_train)
        bce_obj = F.binary_cross_entropy_with_logits(objectness_logits.view(-1), objectness_targets, reduction="none")
        objectness_loss = (bce_obj * objectness_weights).sum() / objectness_weights.sum().clamp_min(1e-8)
        if type_mask.any():
            smooth_targets = _smooth_binary_targets(type_targets[type_mask], float(args.label_smoothing))
            type_loss = F.binary_cross_entropy_with_logits(
                type_logits[type_mask],
                smooth_targets,
                pos_weight=pos_weight,
            )
        else:
            type_loss = torch.zeros((), dtype=objectness_loss.dtype, device=device)
        loss = objectness_loss + float(args.type_loss_weight) * type_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        opt.step()

        with torch.no_grad():
            objectness_prob = torch.sigmoid(objectness_logits)
            obj_acc = float(((objectness_prob >= 0.5) == objectness_targets.bool()).float().mean().item())
            if type_mask.any():
                type_prob = torch.sigmoid(type_logits[type_mask])
                exact = ((type_prob >= 0.5) == type_targets[type_mask].bool()).all(dim=1).float().mean().item()
            else:
                exact = 0.0
        loss_value = float(loss.detach().cpu().item())
        improved = loss_value < best_loss - 1e-6
        if improved:
            best_loss = loss_value
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = int(patience)
        else:
            patience_left -= 1
        rows.append(
            {
                "epoch": int(epoch),
                "loss": loss_value,
                "objectness_loss": float(objectness_loss.detach().cpu().item()),
                "type_loss": float(type_loss.detach().cpu().item()),
                "objectness_acc_at_0_5": obj_acc,
                "type_subset_acc_labeled_at_0_5": float(exact),
                "n_train": int(train_idx.size),
                "n_labeled": int(labeled[train_idx].sum()),
                "n_background": int(background[train_idx].sum()),
                "is_best": bool(improved),
                "split": log_prefix,
            }
        )
        if log_prefix == "final" and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(epochs)):
            log.info(
                "epoch=%d loss=%.4f obj=%.4f type=%.4f",
                epoch,
                loss_value,
                float(objectness_loss.detach().cpu().item()),
                float(type_loss.detach().cpu().item()),
            )
        if int(patience) > 0 and patience_left <= 0:
            if log_prefix == "final":
                log.info("Early stopping final head at epoch %d; best loss %.4f", epoch, best_loss)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_model(model: MultiLabelComplexSVClassifierHead, embeddings: np.ndarray, metadata: pd.DataFrame, class_names: list[str], device: torch.device, batch_size: int = 512) -> pd.DataFrame:
    model.eval()
    obj_logits: list[np.ndarray] = []
    type_logits_out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], int(batch_size)):
            batch = torch.as_tensor(embeddings[start : start + int(batch_size)], dtype=torch.float32, device=device)
            objectness_logits, type_logits = model(batch)
            obj_logits.append(objectness_logits.detach().cpu().numpy().astype(np.float32))
            type_logits_out.append(type_logits.detach().cpu().numpy().astype(np.float32))
    objectness_logit_np = np.concatenate(obj_logits, axis=0)
    objectness_prob_np = 1.0 / (1.0 + np.exp(-objectness_logit_np))
    type_logit_np = np.concatenate(type_logits_out, axis=0)
    type_prob_np = 1.0 / (1.0 + np.exp(-type_logit_np))
    out = metadata.copy()
    labeled, background = _label_background_masks(out)
    targets = _multi_hot_targets(out, class_names)
    out["is_labeled"] = labeled.astype(int)
    out["is_background_chromosome"] = background.astype(int)
    if "raw_sv_classes" in out:
        out["raw_true_classes"] = out["raw_sv_classes"].astype(str)
    out["true_classes"] = [";".join([class_names[i] for i, flag in enumerate(row) if flag > 0]) for row in targets]
    out["objectness_logit"] = objectness_logit_np.astype(float)
    out["objectness_prob"] = objectness_prob_np.astype(float)
    max_idx = np.argmax(type_prob_np, axis=1)
    out["top_type_class"] = [class_names[int(i)] for i in max_idx]
    out["max_type_probability"] = type_prob_np.max(axis=1).astype(float)
    out["max_type_logit"] = type_logit_np[np.arange(type_logit_np.shape[0]), max_idx].astype(float)
    for i, class_name in enumerate(class_names):
        out[f"type_logit_{class_name}"] = type_logit_np[:, i].astype(float)
        out[f"type_probability_{class_name}"] = type_prob_np[:, i].astype(float)
    return out


def _threshold_grid(min_value: float, max_value: float, steps: int) -> np.ndarray:
    return np.linspace(float(min_value), float(max_value), int(steps), dtype=np.float32)


def sweep_objectness_tau(score_df: pd.DataFrame, tau_grid: np.ndarray) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    eligible = (score_df["is_labeled"].astype(bool) | score_df["is_background_chromosome"].astype(bool)).to_numpy()
    df = score_df.loc[eligible].copy()
    if df.empty:
        return pd.DataFrame()
    y = df["is_labeled"].astype(bool).to_numpy()
    score = df["objectness_prob"].astype(float).to_numpy()
    rows = []
    for tau in tau_grid:
        called = score >= float(tau)
        tp = int((y & called).sum())
        fp = int(((~y) & called).sum())
        fn = int((y & (~called)).sum())
        tn = int(((~y) & (~called)).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        rows.append({"objectness_tau": float(tau), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def choose_tau(tau_df: pd.DataFrame, metric: str = "f1") -> float:
    if tau_df.empty:
        return 0.5
    metric = metric if metric in tau_df.columns else "f1"
    return float(tau_df.sort_values([metric, "precision", "recall", "objectness_tau"], ascending=[False, False, False, False]).iloc[0]["objectness_tau"])


def choose_type_thresholds(predictions: pd.DataFrame, class_names: list[str], tau_grid: np.ndarray) -> tuple[dict[str, float], pd.DataFrame]:
    labeled = predictions[predictions["is_labeled"].astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    thresholds: dict[str, float] = {}
    if labeled.empty:
        return {name: 0.5 for name in class_names}, pd.DataFrame()
    true_sets = [set(_split_classes(value)) for value in labeled["true_classes"].tolist()]
    for class_name in class_names:
        y = np.asarray([class_name in values for values in true_sets], dtype=bool)
        score = labeled[f"type_probability_{class_name}"].astype(float).to_numpy()
        best: dict[str, Any] | None = None
        for tau in tau_grid:
            pred = score >= float(tau)
            tp = int((y & pred).sum())
            fp = int(((~y) & pred).sum())
            fn = int((y & (~pred)).sum())
            tn = int(((~y) & (~pred)).sum())
            precision = tp / (tp + fp) if tp + fp else 1.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
            row = {"class_name": class_name, "type_tau": float(tau), "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
            rows.append(row)
            if best is None or (f1, precision, recall, float(tau)) > (best["f1"], best["precision"], best["recall"], best["type_tau"]):
                best = row
        thresholds[class_name] = float(best["type_tau"]) if best is not None and int(y.sum()) > 0 else 0.5
    return thresholds, pd.DataFrame(rows)


def annotate_predictions(predictions: pd.DataFrame, objectness_tau: float, type_thresholds: dict[str, float], class_names: list[str]) -> pd.DataFrame:
    out = predictions.copy()
    out["objectness_tau"] = float(objectness_tau)
    for class_name in class_names:
        out[f"type_threshold_{class_name}"] = float(type_thresholds.get(class_name, 0.5))
    pred_lists: list[list[str]] = []
    for _, row in out.iterrows():
        if float(row["objectness_prob"]) < float(objectness_tau):
            pred_lists.append([])
            continue
        classes = [class_name for class_name in class_names if float(row[f"type_probability_{class_name}"]) >= float(type_thresholds.get(class_name, 0.5))]
        pred_lists.append(classes)
    out["predicted_classes"] = [";".join(values) for values in pred_lists]
    out["predicted_class"] = out["predicted_classes"].mask(out["predicted_classes"].astype(str) == "", "none")
    out["called_complex_sv"] = out["predicted_classes"].astype(str) != ""
    exact: list[bool] = []
    any_match: list[bool] = []
    for true_value, pred_value, is_labeled in zip(out["true_classes"].tolist(), out["predicted_classes"].tolist(), out["is_labeled"].astype(bool).tolist()):
        if not is_labeled:
            exact.append(False)
            any_match.append(False)
            continue
        true_set = set(_split_classes(true_value))
        pred_set = set(_split_classes(pred_value))
        exact.append(true_set == pred_set)
        any_match.append(bool(true_set & pred_set))
    out["class_exact_match"] = exact
    out["class_any_match"] = any_match
    out["objectness_correct"] = (out["called_complex_sv"] == out["is_labeled"].astype(bool)) | ((~out["called_complex_sv"]) & out["is_background_chromosome"].astype(bool))
    return out


def predictions_to_distance_table(predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    out = predictions.copy()
    objectness = out["objectness_prob"].astype(float).to_numpy()
    max_type = out["max_type_probability"].astype(float).to_numpy()
    called = out["called_complex_sv"].astype(bool).to_numpy()
    out["distance_source"] = "multilabel_classifier_probability"
    out["classifier_objectness_distance"] = 1.0 - objectness
    out["nearest_prototype_class"] = out["top_type_class"].astype(str)
    out["nearest_prototype_distance"] = 1.0 - objectness
    out["prototype_confidence"] = np.where(called, objectness * max_type, 0.0)
    for class_name in class_names:
        prob = out[f"type_probability_{class_name}"].astype(float).to_numpy()
        joint = objectness * prob
        out[f"classifier_joint_probability_{class_name}"] = joint
        out[f"d_{class_name}"] = 1.0 - joint
    return out


def loso_to_leave_one_out(held_out_predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    if held_out_predictions.empty:
        return pd.DataFrame()
    labeled = held_out_predictions[held_out_predictions["is_labeled"].astype(bool)].copy()
    if labeled.empty:
        return pd.DataFrame()
    objectness = labeled["objectness_prob"].astype(float).to_numpy()
    max_type = labeled["max_type_probability"].astype(float).to_numpy()
    labeled["distance_source"] = "multilabel_classifier_probability"
    labeled["held_out_class"] = labeled["true_classes"].astype(str)
    labeled["nearest_prototype_class"] = labeled["top_type_class"].astype(str)
    labeled["nearest_prototype_distance"] = 1.0 - objectness
    labeled["leave_one_out_distance"] = 1.0 - objectness
    labeled["prototype_confidence"] = np.where(labeled["called_complex_sv"].astype(bool).to_numpy(), objectness * max_type, 0.0)
    labeled["leave_one_out_correct"] = labeled["class_exact_match"].astype(bool)
    for class_name in class_names:
        labeled[f"loo_d_{class_name}"] = 1.0 - objectness * labeled[f"type_probability_{class_name}"].astype(float).to_numpy()
    return labeled


def run_leave_one_sample_out(embeddings: np.ndarray, metadata: pd.DataFrame, class_names: list[str], args: argparse.Namespace, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled, _background = _label_background_masks(metadata)
    samples = _sample_ids(metadata)
    held_samples = sorted(pd.unique(samples[labeled]).tolist())
    rows: list[pd.DataFrame] = []
    metric_rows: list[pd.DataFrame] = []
    cv_epochs = int(args.cv_epochs) if args.cv_epochs is not None else int(args.epochs)
    cv_patience = int(args.cv_patience) if args.cv_patience is not None else int(args.patience)
    for fold_i, held_sample in enumerate(held_samples):
        train_mask = samples != held_sample
        if int((labeled & train_mask).sum()) == 0:
            log.warning("Skipping LOSO fold for %s; no labels remain in training fold", held_sample)
            continue
        model, metrics = _train_model(embeddings, metadata, class_names, train_mask, args, device, cv_epochs, cv_patience, seed_offset=1000 + fold_i, log_prefix=f"loso:{held_sample}")
        metrics["held_out_sample"] = held_sample
        metric_rows.append(metrics)
        pred = predict_model(model, embeddings, metadata, class_names, device=device, batch_size=int(args.batch_size))
        eval_labeled, eval_background = _label_background_masks(metadata)
        eval_mask = (samples == held_sample) & (eval_labeled | eval_background)
        fold = pred.loc[eval_mask].copy()
        fold["held_out_sample"] = held_sample
        fold["train_n_labeled"] = int((labeled & train_mask).sum())
        rows.append(fold)
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), pd.concat(metric_rows, ignore_index=True) if metric_rows else pd.DataFrame())


def _plot_training(metrics: pd.DataFrame, output_path: Path) -> None:
    if metrics.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(metrics["epoch"], metrics["loss"], label="total")
    axes[0].plot(metrics["epoch"], metrics["objectness_loss"], label="objectness")
    axes[0].plot(metrics["epoch"], metrics["type_loss"], label="multi-label type")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Multi-Label Classifier Training Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)
    axes[1].plot(metrics["epoch"], metrics["objectness_acc_at_0_5"], label="objectness acc @0.5")
    axes[1].plot(metrics["epoch"], metrics["type_subset_acc_labeled_at_0_5"], label="type subset acc @0.5")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Training Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_thresholds(objectness_tau_df: pd.DataFrame, type_threshold_df: pd.DataFrame, output_dir: Path, selected_objectness_tau: float) -> None:
    if not objectness_tau_df.empty:
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        ax.plot(objectness_tau_df["objectness_tau"], objectness_tau_df["precision"], label="precision")
        ax.plot(objectness_tau_df["objectness_tau"], objectness_tau_df["recall"], label="recall")
        ax.plot(objectness_tau_df["objectness_tau"], objectness_tau_df["f1"], label="F1")
        ax.axvline(float(selected_objectness_tau), color="black", linestyle=":", linewidth=1.2)
        ax.set_xlabel("Objectness tau")
        ax.set_ylabel("Score")
        ax.set_ylim(-0.03, 1.03)
        ax.set_title("Multi-Label Objectness Threshold Sweep")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "objectness_tau_sweep.png", dpi=180)
        plt.close(fig)
    if not type_threshold_df.empty:
        best = type_threshold_df.sort_values(["class_name", "f1", "precision", "recall"], ascending=[True, False, False, False]).groupby("class_name", sort=False).head(1)
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.bar(best["class_name"], best["type_tau"], color="#4E79A7")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Selected type threshold")
        ax.set_title("Per-Class Multi-Label Thresholds")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(output_dir / "type_thresholds.png", dpi=180)
        plt.close(fig)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings, metadata = load_embedding_table(args.embeddings_npz, args.metadata_tsv)
    embeddings, metadata = enforce_candidate_resolution(embeddings, metadata, args.candidate_resolution)
    embeddings, metadata = aggregate_multilabel_candidates(embeddings, metadata)
    class_names = _parse_class_names(args.class_names, metadata)
    targets = _multi_hot_targets(metadata, class_names)
    labeled, background = _label_background_masks(metadata)
    if int(labeled.sum()) < 4:
        raise RuntimeError("Need at least four labeled embeddings for multi-label classifier-head training")
    class_counts = _class_counts(targets, class_names)
    observed = [name for name, count in class_counts.items() if count > 0]
    if len(observed) < 2:
        raise RuntimeError(f"Need at least two observed classes; observed {class_counts}")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device=%s; input candidates=%d labels=%s background=%d", device, len(metadata), class_counts, int(background.sum()))

    final_model, training_metrics = _train_model(embeddings, metadata, class_names, np.ones(len(metadata), dtype=bool), args, device, int(args.epochs), int(args.patience), log_prefix="final")
    training_metrics.to_csv(out_dir / "training_metrics.tsv", sep="\t", index=False)
    _plot_training(training_metrics, out_dir / "training_curves.png")

    objectness_grid = _threshold_grid(args.tau_min, args.tau_max, args.tau_steps)
    type_grid = _threshold_grid(args.type_tau_min, args.type_tau_max, args.type_tau_steps)
    cv_annotated = pd.DataFrame()
    if bool(args.skip_loso):
        cv_predictions = pd.DataFrame()
        cv_training_metrics = pd.DataFrame()
        objectness_tau_df = pd.DataFrame()
        type_threshold_sweep = pd.DataFrame()
        selected_objectness_tau = float(args.tau) if args.tau is not None else 0.5
        type_thresholds = {name: float(args.type_tau) if args.type_tau is not None else 0.5 for name in class_names}
    else:
        cv_predictions, cv_training_metrics = run_leave_one_sample_out(embeddings, metadata, class_names, args, device)
        cv_training_metrics.to_csv(out_dir / "loso_training_metrics.tsv", sep="\t", index=False)
        objectness_tau_df = sweep_objectness_tau(cv_predictions, objectness_grid)
        selected_objectness_tau = float(args.tau) if args.tau is not None else choose_tau(objectness_tau_df, metric=str(args.tau_selection_metric))
        if args.type_tau is not None:
            type_thresholds = {name: float(args.type_tau) for name in class_names}
            type_threshold_sweep = pd.DataFrame()
        else:
            type_thresholds, type_threshold_sweep = choose_type_thresholds(cv_predictions, class_names, type_grid)
        cv_annotated = annotate_predictions(cv_predictions, selected_objectness_tau, type_thresholds, class_names) if not cv_predictions.empty else cv_predictions
        cv_predictions.to_csv(out_dir / "leave_one_sample_out_raw.tsv", sep="\t", index=False)
        cv_annotated.to_csv(out_dir / "leave_one_sample_out.tsv", sep="\t", index=False)
        objectness_tau_df.to_csv(out_dir / "objectness_tau_sweep.tsv", sep="\t", index=False)
        type_threshold_sweep.to_csv(out_dir / "type_threshold_sweep.tsv", sep="\t", index=False)
        _plot_thresholds(objectness_tau_df, type_threshold_sweep, out_dir, selected_objectness_tau)

    type_threshold_table = pd.DataFrame([{"class_name": name, "type_threshold": float(type_thresholds.get(name, 0.5))} for name in class_names])
    type_threshold_table.to_csv(out_dir / "type_thresholds.tsv", sep="\t", index=False)

    predictions = predict_model(final_model, embeddings, metadata, class_names, device=device, batch_size=int(args.batch_size))
    predictions = annotate_predictions(predictions, selected_objectness_tau, type_thresholds, class_names)
    predictions.to_csv(out_dir / "classification_predictions.tsv", sep="\t", index=False)
    called = predictions[predictions["called_complex_sv"].astype(bool)].copy()
    called.to_csv(out_dir / "predicted_complex_sv.tsv", sep="\t", index=False)

    compatibility_distances = predictions_to_distance_table(predictions, class_names)
    compatibility_distances.to_csv(out_dir / "prototype_distances.tsv", sep="\t", index=False)
    compatibility_loo = loso_to_leave_one_out(cv_annotated, class_names)
    compatibility_loo.to_csv(out_dir / "anchor_leave_one_out.tsv", sep="\t", index=False)
    classifier_distance_tau = 1.0 - float(selected_objectness_tau)
    embed_corpus.write_visualizations(embeddings, metadata, compatibility_distances, compatibility_loo, out_dir, tau=classifier_distance_tau)

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "input_dim": int(embeddings.shape[1]),
            "hidden_dims": _hidden_dims(args.hidden_dims),
            "dropout": float(args.dropout),
            "activation": str(args.activation),
            "class_names": class_names,
            "class_aliases": CLASS_ALIASES,
            "selected_objectness_tau": float(selected_objectness_tau),
            "type_thresholds": type_thresholds,
            "class_counts": class_counts,
            "n_labeled_candidates": int(labeled.sum()),
            "n_background": int(background.sum()),
            "config": vars(args),
        },
        out_dir / "multilabel_classification_head.pt",
    )
    summary = {
        "class_names": class_names,
        "class_aliases": CLASS_ALIASES,
        "class_counts": class_counts,
        "selected_objectness_tau": float(selected_objectness_tau),
        "classifier_distance_tau_for_legacy_plots": float(classifier_distance_tau),
        "type_thresholds": {name: float(value) for name, value in type_thresholds.items()},
        "n_rows_after_multilabel_merge": int(len(metadata)),
        "n_labeled_candidates": int(labeled.sum()),
        "n_background": int(background.sum()),
        "n_called_complex_sv": int(called.shape[0]),
        "config": vars(args),
    }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote multi-label classifier-head outputs to %s", out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings_npz", required=True)
    parser.add_argument("--metadata_tsv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_resolution", choices=("auto", "chromosome-arm", "any"), default="chromosome-arm")
    parser.add_argument("--class_names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--hidden_dims", default="128", help="Comma-separated hidden layer sizes; use 0 for linear heads.")
    parser.add_argument("--activation", choices=("relu", "gelu"), default="relu")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--cv_epochs", type=int, default=None)
    parser.add_argument("--cv_patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--background_weight", type=float, default=0.0)
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--class_weighting", choices=("none", "inverse", "inverse_sqrt"), default="inverse_sqrt")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=0.95)
    parser.add_argument("--tau_steps", type=int, default=91)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--tau_selection_metric", choices=("f1", "precision", "recall"), default="f1")
    parser.add_argument("--type_tau_min", type=float, default=0.05)
    parser.add_argument("--type_tau_max", type=float, default=0.95)
    parser.add_argument("--type_tau_steps", type=int, default=91)
    parser.add_argument("--type_tau", type=float, default=None, help="Override all per-class thresholds with one value.")
    parser.add_argument("--skip_loso", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
