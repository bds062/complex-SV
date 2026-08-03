"""Few-shot objectness plus type-head training on frozen complex-SV embeddings.

This trains only a small classifier head on saved prototype-mode embeddings. The
CN encoder, SV graph encoder, fusion embedding construction, and sample-residual
normalization stay frozen.
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
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from genomic_features import embedding
from architectures.heads import ComplexSVClassifierHead
from training.losses import complex_sv_objectness_type_loss
from utils import set_seed

log = logging.getLogger(__name__)

DEFAULT_CLASS_NAMES = ("BFB", "chromothripsis", "seismic_amplification", "TIC", "non_canonical_BFB", "non_canonical_chromothripsis")
SCAN_EVIDENCE_VALUES = {"chromosome_scan", "chromosome_arm_scan"}
CANDIDATE_RESOLUTION_CHOICES = ("auto", "chromosome-arm", "any")
ARM_VALUES = {"p", "q"}


def _metadata_from_npz(data: np.lib.npyio.NpzFile) -> pd.DataFrame:
    rows: dict[str, Any] = {}
    for key in data.files:
        if key == "embeddings":
            continue
        arr = data[key]
        if arr.ndim == 1:
            rows[key] = arr
    return pd.DataFrame(rows).fillna("")


def load_embedding_table(embeddings_npz: str | Path, metadata_tsv: str | Path | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(embeddings_npz, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    if metadata_tsv:
        metadata = pd.read_csv(metadata_tsv, sep="\t").fillna("")
    else:
        metadata = _metadata_from_npz(data)
    if len(metadata) != embeddings.shape[0]:
        raise ValueError(f"Metadata rows ({len(metadata)}) do not match embeddings ({embeddings.shape[0]})")
    return embeddings, metadata.reset_index(drop=True)


def _parse_class_names(raw: str | None, metadata: pd.DataFrame, labeled_mask: np.ndarray) -> list[str]:
    observed = sorted(pd.unique(metadata.loc[labeled_mask, "sv_class"].astype(str)).tolist()) if labeled_mask.any() else []
    if raw:
        names = [part.strip() for part in str(raw).split(",") if part.strip()]
    else:
        names = list(DEFAULT_CLASS_NAMES)
    unknown = sorted(set(observed).difference(names))
    if unknown:
        raise ValueError(f"Observed labels not present in --class_names: {unknown}; class_names={names}")
    return names


def _label_masks(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = metadata["sv_class"].astype(str) if "sv_class" in metadata else pd.Series([""] * len(metadata))
    evidence = metadata["evidence"].astype(str) if "evidence" in metadata else pd.Series([""] * len(metadata))
    labeled = labels.to_numpy() != ""
    background = (labels.to_numpy() == "") & evidence.isin(SCAN_EVIDENCE_VALUES).to_numpy()
    return labeled, background


def _clean_arm(value: object) -> str:
    text = str(value).strip().lower()
    return text if text in ARM_VALUES else ""


def _chrom_key(value: object) -> str:
    return str(value).strip().removeprefix("chr").removeprefix("CHR")


def _scope_series(metadata: pd.DataFrame) -> pd.Series:
    if "candidate_scope" in metadata:
        scope = metadata["candidate_scope"].astype(str)
    elif "label_scope" in metadata:
        scope = metadata["label_scope"].astype(str)
    else:
        scope = pd.Series([""] * len(metadata), index=metadata.index)
    return scope.fillna("")


def _row_keys(metadata: pd.DataFrame) -> pd.Series:
    samples = metadata["sample_id"].astype(str) if "sample_id" in metadata else pd.Series(["sample"] * len(metadata), index=metadata.index)
    chroms = metadata["chrom"].map(_chrom_key) if "chrom" in metadata else pd.Series([""] * len(metadata), index=metadata.index)
    return samples.astype(str) + "	" + chroms.astype(str)


def enforce_candidate_resolution(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    mode: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Prefer or require chromosome-arm candidates in classifier inputs."""
    mode = str(mode or "auto").strip().replace("_", "-")
    if mode not in CANDIDATE_RESOLUTION_CHOICES:
        choices = ", ".join(CANDIDATE_RESOLUTION_CHOICES)
        raise ValueError(f"candidate_resolution must be one of: {choices}")
    if mode == "any" or metadata.empty:
        return embeddings, metadata.reset_index(drop=True)

    out = metadata.copy().reset_index(drop=True)
    if "arm" not in out.columns:
        out["arm"] = ""
    out["arm"] = out["arm"].map(_clean_arm)

    labels = out["sv_class"].astype(str) if "sv_class" in out else pd.Series([""] * len(out), index=out.index)
    evidence = out["evidence"].astype(str) if "evidence" in out else pd.Series([""] * len(out), index=out.index)
    scope = _scope_series(out)
    labeled = labels.to_numpy() != ""
    background = (labels.to_numpy() == "") & evidence.isin(SCAN_EVIDENCE_VALUES).to_numpy()
    arm_scan = background & (evidence.to_numpy() == "chromosome_arm_scan") & out["arm"].isin(ARM_VALUES).to_numpy()
    whole_scan = background & (evidence.to_numpy() == "chromosome_scan")

    keep = np.ones(len(out), dtype=bool)
    if arm_scan.any() and whole_scan.any():
        arm_keys = set(_row_keys(out.loc[arm_scan]).tolist())
        whole_keys = _row_keys(out.loc[whole_scan])
        drop_idx = whole_keys[whole_keys.isin(arm_keys)].index.to_numpy(dtype=int)
        if drop_idx.size:
            keep[drop_idx] = False
            log.info("Dropped %d whole-chromosome scan row(s) because arm scan rows are present", int(drop_idx.size))

    if not keep.all():
        out = out.loc[keep].reset_index(drop=True)
        embeddings = np.asarray(embeddings, dtype=np.float32)[keep]
        labels = out["sv_class"].astype(str) if "sv_class" in out else pd.Series([""] * len(out), index=out.index)
        evidence = out["evidence"].astype(str) if "evidence" in out else pd.Series([""] * len(out), index=out.index)
        scope = _scope_series(out)
        labeled = labels.to_numpy() != ""
        background = (labels.to_numpy() == "") & evidence.isin(SCAN_EVIDENCE_VALUES).to_numpy()

    if mode == "chromosome-arm":
        arm_ok = out["arm"].isin(ARM_VALUES).to_numpy()
        label_arm_scope = scope.to_numpy() == "chromosome_arm"
        bad_labels = labeled & ((~label_arm_scope) | (~arm_ok))
        bad_background = background & ((evidence.to_numpy() != "chromosome_arm_scan") | (~arm_ok))
        if bad_labels.any() or bad_background.any():
            bad = out.loc[bad_labels | bad_background, [col for col in ["candidate_id", "sample_id", "chrom", "arm", "start_bp", "end_bp", "evidence", "sv_class", "label_scope", "candidate_scope"] if col in out.columns]].head(8)
            examples = bad.to_dict("records")
            raise ValueError(
                "Classifier candidate_resolution=chromosome-arm requires arm-level embeddings for all "
                "labeled test rows and unlabeled scan rows. The loaded metadata still contains "
                f"whole-chromosome or missing-arm rows, examples={examples}. Rerun prototype_mode.sh "
                "with PROTOTYPE_CANDIDATE_SOURCE=chromosome-arms and INFER_CANDIDATE_SOURCE=chromosome-arms."
            )

    return np.asarray(embeddings, dtype=np.float32), out.reset_index(drop=True)


def _sample_ids(metadata: pd.DataFrame) -> np.ndarray:
    if "sample_id" in metadata:
        return metadata["sample_id"].astype(str).to_numpy()
    return np.asarray(["sample"] * len(metadata), dtype=object)


def _type_targets(metadata: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    labels = metadata["sv_class"].astype(str).to_numpy() if "sv_class" in metadata else np.asarray([""] * len(metadata), dtype=object)
    targets = np.full(len(metadata), -1, dtype=np.int64)
    for i, label in enumerate(labels):
        if label:
            targets[i] = class_to_idx[label]
    return targets


def _class_counts(metadata: pd.DataFrame, labeled_mask: np.ndarray, class_names: list[str]) -> dict[str, int]:
    labels = metadata.loc[labeled_mask, "sv_class"].astype(str) if "sv_class" in metadata else pd.Series(dtype=str)
    counts = labels.value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in class_names}


def _class_weights(type_targets: np.ndarray, type_mask: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none" or not bool(type_mask.any()):
        return None
    counts = np.bincount(type_targets[type_mask], minlength=int(n_classes)).astype(np.float32)
    weights = np.ones(int(n_classes), dtype=np.float32)
    present = counts > 0
    if mode == "inverse":
        weights[present] = float(counts[present].sum()) / np.maximum(counts[present], 1.0)
    elif mode == "inverse_sqrt":
        weights[present] = np.sqrt(float(counts[present].sum()) / np.maximum(counts[present], 1.0))
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    if present.any():
        weights[present] = weights[present] / np.mean(weights[present])
    return torch.as_tensor(weights, dtype=torch.float32)


def _objectness_weights(
    metadata: pd.DataFrame,
    train_idx: np.ndarray,
    labeled_mask: np.ndarray,
    background_mask: np.ndarray,
    background_weight: float,
) -> np.ndarray:
    weights = np.ones(len(train_idx), dtype=np.float32)
    train_labeled = labeled_mask[train_idx]
    train_background = background_mask[train_idx]
    weights[train_background] = float(background_weight)
    if not train_background.any():
        return weights
    samples = _sample_ids(metadata)[train_idx]
    # Keep background chromosomes from one sample from dominating objectness.
    for sample in sorted(pd.unique(samples[train_background])):
        sample_bg = train_background & (samples == sample)
        if sample_bg.any():
            weights[sample_bg] = float(background_weight) / float(sample_bg.sum())
    if train_labeled.any():
        labels = metadata["sv_class"].astype(str).to_numpy()[train_idx]
        for class_name in sorted(pd.unique(labels[train_labeled])):
            cls_mask = train_labeled & (labels == class_name)
            if cls_mask.any():
                weights[cls_mask] = 1.0 / float(cls_mask.sum())
        pos_total = weights[train_labeled].sum()
        if pos_total > 0:
            weights[train_labeled] *= float(train_labeled.sum()) / pos_total
    return weights.astype(np.float32)


def _make_model(args: argparse.Namespace, input_dim: int, n_classes: int, device: torch.device) -> ComplexSVClassifierHead:
    return ComplexSVClassifierHead(
        in_dim=int(input_dim),
        num_classes=int(n_classes),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
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
) -> tuple[ComplexSVClassifierHead, pd.DataFrame]:
    set_seed(int(args.seed) + int(seed_offset))
    labeled_mask, background_mask = _label_masks(metadata)
    usable_mask = train_mask & (labeled_mask | background_mask)
    train_idx = np.where(usable_mask)[0]
    if train_idx.size == 0:
        raise RuntimeError(f"No usable labeled/background rows for {log_prefix}")
    if not bool(labeled_mask[train_idx].any()):
        raise RuntimeError(f"No positive labels available for {log_prefix}")

    all_type_targets = _type_targets(metadata, class_names)
    x_train = torch.as_tensor(embeddings[train_idx], dtype=torch.float32, device=device)
    objectness_targets_np = labeled_mask[train_idx].astype(np.float32)
    type_mask_np = labeled_mask[train_idx]
    type_targets_np = all_type_targets[train_idx].copy()
    type_targets_np[type_targets_np < 0] = 0
    weights_np = _objectness_weights(metadata, train_idx, labeled_mask, background_mask, float(args.background_weight))
    class_weight_t = _class_weights(type_targets_np, type_mask_np, len(class_names), str(args.class_weighting))
    if class_weight_t is not None:
        class_weight_t = class_weight_t.to(device)

    objectness_targets = torch.as_tensor(objectness_targets_np, dtype=torch.float32, device=device)
    type_targets = torch.as_tensor(type_targets_np, dtype=torch.long, device=device)
    type_mask = torch.as_tensor(type_mask_np, dtype=torch.bool, device=device)
    objectness_weights = torch.as_tensor(weights_np, dtype=torch.float32, device=device)

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
        loss, parts = complex_sv_objectness_type_loss(
            objectness_logits,
            type_logits,
            objectness_targets,
            type_targets,
            type_mask,
            objectness_weights=objectness_weights,
            class_weights=class_weight_t,
            type_loss_weight=float(args.type_loss_weight),
            label_smoothing=float(args.label_smoothing),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        opt.step()

        with torch.no_grad():
            objectness_prob = torch.sigmoid(objectness_logits)
            objectness_acc = float(((objectness_prob >= 0.5) == objectness_targets.bool()).float().mean().item())
            if type_mask.any():
                type_acc = float((torch.argmax(type_logits[type_mask], dim=1) == type_targets[type_mask]).float().mean().item())
            else:
                type_acc = 0.0
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
                "objectness_loss": float(parts["objectness_loss"].detach().cpu().item()),
                "type_loss": float(parts["type_loss"].detach().cpu().item()),
                "objectness_acc_at_0_5": objectness_acc,
                "type_acc_labeled": type_acc,
                "n_train": int(train_idx.size),
                "n_labeled": int(labeled_mask[train_idx].sum()),
                "n_background": int(background_mask[train_idx].sum()),
                "is_best": bool(improved),
                "split": log_prefix,
            }
        )
        if log_prefix == "final" and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(epochs)):
            log.info(
                "epoch=%d loss=%.4f obj=%.4f type=%.4f obj_acc=%.3f type_acc=%.3f",
                epoch,
                loss_value,
                float(parts["objectness_loss"].detach().cpu().item()),
                float(parts["type_loss"].detach().cpu().item()),
                objectness_acc,
                type_acc,
            )
        if int(patience) > 0 and patience_left <= 0:
            if log_prefix == "final":
                log.info("Early stopping final head at epoch %d; best loss %.4f", epoch, best_loss)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_model(
    model: ComplexSVClassifierHead,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    device: torch.device,
    batch_size: int = 512,
) -> pd.DataFrame:
    model.eval()
    objectness_logits_all: list[np.ndarray] = []
    type_probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], int(batch_size)):
            batch = torch.as_tensor(embeddings[start : start + int(batch_size)], dtype=torch.float32, device=device)
            objectness_logits, type_logits = model(batch)
            objectness_logits_all.append(objectness_logits.detach().cpu().numpy().astype(np.float32))
            type_probs_all.append(F.softmax(type_logits, dim=1).detach().cpu().numpy().astype(np.float32))
    objectness_logits_np = np.concatenate(objectness_logits_all, axis=0)
    objectness_prob_np = 1.0 / (1.0 + np.exp(-objectness_logits_np))
    type_probs_np = np.concatenate(type_probs_all, axis=0)
    type_idx = np.argmax(type_probs_np, axis=1)
    labeled_mask, background_mask = _label_masks(metadata)

    out = metadata.copy()
    out["is_labeled"] = labeled_mask.astype(int)
    out["is_background_chromosome"] = background_mask.astype(int)
    out["true_class"] = out["sv_class"].astype(str) if "sv_class" in out else ""
    out["objectness_logit"] = objectness_logits_np.astype(float)
    out["objectness_prob"] = objectness_prob_np.astype(float)
    out["type_predicted_class"] = [class_names[int(i)] for i in type_idx]
    out["max_type_probability"] = type_probs_np.max(axis=1).astype(float)
    for i, class_name in enumerate(class_names):
        out[f"type_probability_{class_name}"] = type_probs_np[:, i].astype(float)
    return out


def annotate_predictions(predictions: pd.DataFrame, objectness_tau: float) -> pd.DataFrame:
    out = predictions.copy()
    out["objectness_tau"] = float(objectness_tau)
    out["called_complex_sv"] = out["objectness_prob"].astype(float) >= float(objectness_tau)
    out["predicted_class"] = out["type_predicted_class"].where(out["called_complex_sv"], "none")
    out["objectness_correct"] = (out["called_complex_sv"] == out["is_labeled"].astype(bool)) | (
        (~out["called_complex_sv"]) & out["is_background_chromosome"].astype(bool)
    )
    out["class_correct"] = (
        out["is_labeled"].astype(bool)
        & out["called_complex_sv"].astype(bool)
        & (out["true_class"].astype(str) == out["predicted_class"].astype(str))
    )
    return out



def classifier_predictions_to_distance_table(predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Return old prototype-distance-shaped outputs for classifier predictions.

    These are compatibility pseudo-distances, not cosine distances to learned
    prototypes. Lower is more confident. ``nearest_prototype_distance`` is
    objectness distance, so the old PR/tau plots still correspond to the
    objectness threshold.
    """
    out = predictions.copy()
    objectness = out["objectness_prob"].astype(float).to_numpy()
    max_type = out["max_type_probability"].astype(float).to_numpy()
    called = out["predicted_class"].astype(str).to_numpy() != "none"
    out["distance_source"] = "classifier_probability"
    out["classifier_objectness_distance"] = 1.0 - objectness
    out["nearest_prototype_class"] = out["type_predicted_class"].astype(str)
    out["nearest_prototype_distance"] = 1.0 - objectness
    out["prototype_confidence"] = np.where(called, objectness * max_type, 0.0)
    for class_name in class_names:
        prob_col = f"type_probability_{class_name}"
        if prob_col not in out:
            continue
        type_prob = out[prob_col].astype(float).to_numpy()
        joint_prob = objectness * type_prob
        out[f"classifier_joint_probability_{class_name}"] = joint_prob
        out[f"d_{class_name}"] = 1.0 - joint_prob
    return out


def classifier_loso_to_leave_one_out(held_out_predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Return leave-one-out-shaped held-sample rows for legacy plots."""
    if held_out_predictions.empty:
        return pd.DataFrame()
    labeled = held_out_predictions[held_out_predictions["is_labeled"].astype(bool)].copy()
    if labeled.empty:
        return pd.DataFrame()
    objectness = labeled["objectness_prob"].astype(float).to_numpy()
    max_type = labeled["max_type_probability"].astype(float).to_numpy()
    called = labeled["predicted_class"].astype(str).to_numpy() != "none"
    labeled["distance_source"] = "classifier_probability"
    labeled["held_out_class"] = labeled["true_class"].astype(str)
    labeled["nearest_prototype_class"] = labeled["type_predicted_class"].astype(str)
    labeled["nearest_prototype_distance"] = 1.0 - objectness
    labeled["leave_one_out_distance"] = 1.0 - objectness
    labeled["prototype_confidence"] = np.where(called, objectness * max_type, 0.0)
    labeled["leave_one_out_correct"] = labeled["true_class"].astype(str) == labeled["predicted_class"].astype(str)
    for class_name in class_names:
        prob_col = f"type_probability_{class_name}"
        if prob_col not in labeled:
            continue
        joint_prob = objectness * labeled[prob_col].astype(float).to_numpy()
        labeled[f"loo_d_{class_name}"] = 1.0 - joint_prob
    return labeled


def sweep_objectness_tau(score_df: pd.DataFrame, tau_grid: np.ndarray) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    eligible = (score_df["is_labeled"].astype(bool) | score_df["is_background_chromosome"].astype(bool)).to_numpy()
    df = score_df.loc[eligible].copy()
    if df.empty:
        return pd.DataFrame()
    y = df["is_labeled"].astype(bool).to_numpy()
    score = df["objectness_prob"].astype(float).to_numpy()
    true_cls = df["true_class"].astype(str).to_numpy()
    type_cls = df["type_predicted_class"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for tau in tau_grid:
        called = score >= float(tau)
        tp = int((y & called).sum())
        fp = int(((~y) & called).sum())
        fn = int((y & (~called)).sum())
        tn = int(((~y) & (~called)).sum())
        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        correct_type = y & called & (true_cls == type_cls)
        typed_tp = int(correct_type.sum())
        typed_precision = typed_tp / (tp + fp) if tp + fp else 1.0
        typed_recall = typed_tp / int(y.sum()) if int(y.sum()) else 0.0
        typed_f1 = (2 * typed_precision * typed_recall / (typed_precision + typed_recall)) if typed_precision + typed_recall else 0.0
        type_accuracy_called_positives = typed_tp / tp if tp else 0.0
        rows.append(
            {
                "objectness_tau": float(tau),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "typed_precision": float(typed_precision),
                "typed_recall": float(typed_recall),
                "typed_f1": float(typed_f1),
                "type_accuracy_called_positives": float(type_accuracy_called_positives),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "typed_tp": typed_tp,
            }
        )
    return pd.DataFrame(rows)


def choose_tau(tau_df: pd.DataFrame, metric: str = "typed_f1") -> float:
    if tau_df.empty:
        return 0.5
    metric = metric if metric in tau_df.columns else "f1"
    ranked = tau_df.sort_values(
        [metric, "precision", "recall", "objectness_tau"],
        ascending=[False, False, False, False],
    )
    return float(ranked.iloc[0]["objectness_tau"])


def run_leave_one_sample_out(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_mask, background_mask = _label_masks(metadata)
    samples = _sample_ids(metadata)
    held_samples = sorted(pd.unique(samples[labeled_mask]).tolist())
    rows: list[pd.DataFrame] = []
    train_metrics: list[pd.DataFrame] = []
    cv_epochs = int(args.cv_epochs) if args.cv_epochs is not None else int(args.epochs)
    cv_patience = int(args.cv_patience) if args.cv_patience is not None else int(args.patience)

    for fold_i, held_sample in enumerate(held_samples):
        train_mask = samples != held_sample
        train_labeled = labeled_mask & train_mask
        if int(train_labeled.sum()) == 0:
            log.warning("Skipping LOSO fold for %s; no labels remain in training fold", held_sample)
            continue
        log.info("LOSO fold %s: train labels=%d held positives=%d held background=%d", held_sample, int(train_labeled.sum()), int((labeled_mask & ~train_mask).sum()), int((background_mask & ~train_mask).sum()))
        model, metrics = _train_model(
            embeddings,
            metadata,
            class_names,
            train_mask=train_mask,
            args=args,
            device=device,
            epochs=cv_epochs,
            patience=cv_patience,
            seed_offset=1000 + fold_i,
            log_prefix=f"loso:{held_sample}",
        )
        metrics["held_out_sample"] = held_sample
        train_metrics.append(metrics)
        pred = predict_model(model, embeddings, metadata, class_names, device=device, batch_size=int(args.batch_size))
        eval_mask = (samples == held_sample) & (labeled_mask | background_mask)
        fold = pred.loc[eval_mask].copy()
        fold["held_out_sample"] = held_sample
        fold["train_n_labeled"] = int(train_labeled.sum())
        fold["train_class_counts"] = json.dumps(_class_counts(metadata.loc[train_mask].reset_index(drop=True), _label_masks(metadata.loc[train_mask].reset_index(drop=True))[0], class_names), sort_keys=True)
        rows.append(fold)

    cv_predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cv_metrics = pd.concat(train_metrics, ignore_index=True) if train_metrics else pd.DataFrame()
    return cv_predictions, cv_metrics


def _plot_training(metrics: pd.DataFrame, output_path: Path) -> None:
    if metrics.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(metrics["epoch"], metrics["loss"], label="total")
    axes[0].plot(metrics["epoch"], metrics["objectness_loss"], label="objectness")
    axes[0].plot(metrics["epoch"], metrics["type_loss"], label="type")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Classifier Training Loss")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].plot(metrics["epoch"], metrics["objectness_acc_at_0_5"], label="objectness acc @0.5")
    axes[1].plot(metrics["epoch"], metrics["type_acc_labeled"], label="type acc, labels")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Training Accuracy")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_tau(tau_df: pd.DataFrame, output_path: Path, selected_tau: float) -> None:
    if tau_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(tau_df["objectness_tau"], tau_df["precision"], label="precision")
    ax.plot(tau_df["objectness_tau"], tau_df["recall"], label="recall")
    ax.plot(tau_df["objectness_tau"], tau_df["f1"], label="F1")
    if "typed_f1" in tau_df:
        ax.plot(tau_df["objectness_tau"], tau_df["typed_f1"], label="typed F1")
    ax.axvline(float(selected_tau), color="black", linestyle=":", linewidth=1.2, label=f"selected tau={selected_tau:.3g}")
    ax.set_xlabel("Objectness tau")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Leave-One-Sample-Out Threshold Sweep")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_validation_confusion(predictions: pd.DataFrame, class_names: list[str], output_path: Path) -> None:
    if predictions.empty:
        return
    labeled = predictions[predictions["is_labeled"].astype(bool)].copy()
    if labeled.empty:
        return
    cols = class_names + ["none"]
    table = pd.crosstab(labeled["true_class"], labeled["predicted_class"]).reindex(index=class_names, columns=cols, fill_value=0)
    fig, ax = plt.subplots(figsize=(7.2, max(3.2, 0.55 * len(class_names) + 2.0)))
    im = ax.imshow(table.to_numpy(dtype=float), cmap="viridis")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("GT class")
    ax.set_title("Held-Out Label Predictions")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, str(int(table.iat[i, j])), ha="center", va="center", color="white" if table.iat[i, j] > table.to_numpy().max() * 0.45 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Count")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_objectness_scores(predictions: pd.DataFrame, output_path: Path, selected_tau: float) -> None:
    if predictions.empty:
        return
    eligible = predictions[predictions["is_labeled"].astype(bool) | predictions["is_background_chromosome"].astype(bool)].copy()
    if eligible.empty:
        return
    eligible = eligible.sort_values("objectness_prob", ascending=True).reset_index(drop=True)
    colors = np.where(eligible["is_labeled"].astype(bool), "#c43b3b", "#4a78b8")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.scatter(np.arange(len(eligible)), eligible["objectness_prob"].astype(float), c=colors, s=18, alpha=0.85, linewidths=0)
    ax.axhline(float(selected_tau), color="black", linestyle=":", linewidth=1.2, label=f"tau={selected_tau:.3g}")
    ax.set_xlabel("Held-out rows sorted by objectness")
    ax.set_ylabel("Objectness probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Held-Out Objectness Scores")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embeddings, metadata = load_embedding_table(args.embeddings_npz, args.metadata_tsv)
    embeddings, metadata = enforce_candidate_resolution(embeddings, metadata, args.candidate_resolution)
    labeled_mask, background_mask = _label_masks(metadata)
    if int(labeled_mask.sum()) < 4:
        raise RuntimeError("Need at least four labeled embeddings for few-shot classifier-head training")
    class_names = _parse_class_names(args.class_names, metadata, labeled_mask)
    class_counts = _class_counts(metadata, labeled_mask, class_names)
    observed_classes = [name for name, count in class_counts.items() if count > 0]
    if len(observed_classes) < 2:
        raise RuntimeError(f"Need at least two observed classes; observed {class_counts}")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    log.info("Using device=%s; input embeddings=%s labels=%s background=%s", device, embeddings.shape, class_counts, int(background_mask.sum()))
    if int(background_mask.sum()) == 0:
        log.warning("No unlabeled chromosome/arm scan backgrounds were found; objectness will train only on positives.")

    all_train_mask = np.ones(len(metadata), dtype=bool)
    final_model, training_metrics = _train_model(
        embeddings,
        metadata,
        class_names,
        train_mask=all_train_mask,
        args=args,
        device=device,
        epochs=int(args.epochs),
        patience=int(args.patience),
        seed_offset=0,
        log_prefix="final",
    )
    training_metrics.to_csv(out_dir / "training_metrics.tsv", sep="\t", index=False)
    _plot_training(training_metrics, out_dir / "training_curves.png")

    tau_grid = np.linspace(float(args.tau_min), float(args.tau_max), int(args.tau_steps), dtype=np.float32)
    cv_annotated = pd.DataFrame()
    if bool(args.skip_loso):
        cv_predictions = pd.DataFrame()
        cv_training_metrics = pd.DataFrame()
        tau_df = pd.DataFrame()
        selected_tau = float(args.tau) if args.tau is not None else 0.5
    else:
        cv_predictions, cv_training_metrics = run_leave_one_sample_out(embeddings, metadata, class_names, args, device)
        cv_training_metrics.to_csv(out_dir / "loso_training_metrics.tsv", sep="\t", index=False)
        tau_df = sweep_objectness_tau(cv_predictions, tau_grid)
        selected_tau = float(args.tau) if args.tau is not None else choose_tau(tau_df, metric=str(args.tau_selection_metric))
        cv_annotated = annotate_predictions(cv_predictions, selected_tau) if not cv_predictions.empty else cv_predictions
        cv_predictions.to_csv(out_dir / "leave_one_sample_out_raw.tsv", sep="\t", index=False)
        cv_annotated.to_csv(out_dir / "leave_one_sample_out.tsv", sep="\t", index=False)
        tau_df.to_csv(out_dir / "objectness_tau_sweep.tsv", sep="\t", index=False)
        _plot_tau(tau_df, out_dir / "objectness_tau_sweep.png", selected_tau)
        _plot_validation_confusion(cv_annotated, class_names, out_dir / "held_out_prediction_summary.png")
        _plot_objectness_scores(cv_annotated, out_dir / "held_out_objectness_scores.png", selected_tau)

    predictions = predict_model(final_model, embeddings, metadata, class_names, device=device, batch_size=int(args.batch_size))
    predictions = annotate_predictions(predictions, selected_tau)
    predictions.to_csv(out_dir / "classification_predictions.tsv", sep="\t", index=False)
    called = predictions[predictions["called_complex_sv"].astype(bool)].copy()
    called.to_csv(out_dir / "predicted_complex_sv.tsv", sep="\t", index=False)

    compatibility_distances = classifier_predictions_to_distance_table(predictions, class_names)
    compatibility_distances.to_csv(out_dir / "prototype_distances.tsv", sep="\t", index=False)
    compatibility_loo = classifier_loso_to_leave_one_out(cv_annotated, class_names)
    compatibility_loo.to_csv(out_dir / "anchor_leave_one_out.tsv", sep="\t", index=False)
    # Legacy visualization helpers expect lower-is-better distance tau.
    classifier_distance_tau = 1.0 - float(selected_tau)
    embedding.write_visualizations(
        embeddings,
        metadata,
        compatibility_distances,
        compatibility_loo,
        out_dir,
        tau=classifier_distance_tau,
    )

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "input_dim": int(embeddings.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "class_names": class_names,
            "selected_objectness_tau": float(selected_tau),
            "class_counts": class_counts,
            "n_background": int(background_mask.sum()),
            "config": vars(args),
        },
        out_dir / "classification_head.pt",
    )
    summary = {
        "class_names": class_names,
        "class_counts": class_counts,
        "selected_objectness_tau": float(selected_tau),
        "classifier_distance_tau_for_legacy_plots": float(1.0 - float(selected_tau)),
        "tau_selection_metric": str(args.tau_selection_metric),
        "n_labeled": int(labeled_mask.sum()),
        "n_background": int(background_mask.sum()),
        "n_called_complex_sv": int(called.shape[0]),
    }
    if not tau_df.empty:
        best_row = tau_df.loc[(tau_df["objectness_tau"] - float(selected_tau)).abs().idxmin()].to_dict()
        summary["selected_tau_metrics"] = {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) for k, v in best_row.items()}
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote classifier-head outputs to %s", out_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings_npz", required=True, help="Input embeddings.npz from prototype-mode inference/anchors.")
    parser.add_argument("--metadata_tsv", default=None, help="Optional candidate_embeddings.tsv; otherwise metadata is read from NPZ arrays.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate_resolution",
        choices=CANDIDATE_RESOLUTION_CHOICES,
        default="auto",
        help=(
            "Candidate resolution policy. chromosome-arm requires labeled/test rows and "
            "unlabeled scan rows to be arm-level; auto drops duplicate whole-chromosome "
            "scan rows when arm scans are present; any preserves all rows."
        ),
    )
    parser.add_argument("--class_names", default=",".join(DEFAULT_CLASS_NAMES), help="Comma-separated output class order for the type head.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Set to 0 for a linear-only head.")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--cv_epochs", type=int, default=None, help="Epochs per leave-one-sample-out fold; defaults to --epochs.")
    parser.add_argument("--cv_patience", type=int, default=None, help="Patience per leave-one-sample-out fold; defaults to --patience.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--background_weight", type=float, default=0.25, help="Weak per-sample weight assigned to unlabeled chromosome/arm scan negatives.")
    parser.add_argument("--type_loss_weight", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument("--class_weighting", choices=("none", "inverse", "inverse_sqrt"), default="inverse_sqrt")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=0.95)
    parser.add_argument("--tau_steps", type=int, default=91)
    parser.add_argument("--tau", type=float, default=None, help="Override calibrated objectness tau for final predictions.")
    parser.add_argument("--tau_selection_metric", choices=("f1", "typed_f1", "precision", "recall"), default="typed_f1")
    parser.add_argument("--skip_loso", action="store_true", help="Skip leave-one-sample-out calibration and use --tau or 0.5.")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
