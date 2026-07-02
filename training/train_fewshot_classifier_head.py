"""Train hierarchical few-shot prototypical classifier on frozen complex-SV embeddings.

Architecture
------------
Level 2 (family): 4 family prototypes built by pooling canonical + non-canonical examples.
  BFB_family             <- BFB + non_canonical_BFB
  chromothripsis_family  <- chromothripsis + non_canonical_chromothripsis
  seismic_amplification  <- seismic_amplification  (no subtypes)
  TIC                    <- TIC                    (no subtypes)

Level 3 (subtype): 2-class prototypes within BFB_family and chromothripsis_family only.

Training: single shared MetricProjection trained with a combined LOO loss at both levels
simultaneously. Level 2 pulls family members together; Level 3 pushes canonical and
non-canonical apart within each family. One optimizer, one backward pass per epoch.

Inference: two-pass distance lookup — nearest family (Level 2) then nearest subtype within
that family (Level 3). No additional learned parameters.
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

from discovery import embed_corpus
from model.heads import MetricProjection
from training.train_classifier_head import DEFAULT_CLASS_NAMES, enforce_candidate_resolution, load_embedding_table
from utils import set_seed

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hierarchy definition
# ---------------------------------------------------------------------------

FAMILY_MAP: dict[str, str] = {
    "BFB": "BFB_family",
    "non_canonical_BFB": "BFB_family",
    "chromothripsis": "chromothripsis_family",
    "non_canonical_chromothripsis": "chromothripsis_family",
    "seismic_amplification": "seismic_amplification",
    "TIC": "TIC",
}

FAMILY_SUBTYPES: dict[str, list[str]] = {
    "BFB_family": ["BFB", "non_canonical_BFB"],
    "chromothripsis_family": ["chromothripsis", "non_canonical_chromothripsis"],
}

FAMILY_NAMES: list[str] = ["BFB_family", "chromothripsis_family", "seismic_amplification", "TIC"]

SCAN_EVIDENCE_VALUES = {"chromosome_scan", "chromosome_arm_scan"}


def _build_family_names(class_names: list[str]) -> list[str]:
    """Return ordered family names covering all class_names, adding unknown classes as own families."""
    seen: dict[str, None] = {}
    for cn in class_names:
        fam = FAMILY_MAP.get(cn, cn)
        seen[fam] = None
    # preserve the canonical order from FAMILY_NAMES, then append unknowns
    result = [f for f in FAMILY_NAMES if f in seen]
    for f in seen:
        if f not in result:
            result.append(f)
    return result


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def _label_masks(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    labels = metadata["sv_class"].astype(str) if "sv_class" in metadata else pd.Series([""] * len(metadata))
    evidence = metadata["evidence"].astype(str) if "evidence" in metadata else pd.Series([""] * len(metadata))
    labeled = labels.to_numpy() != ""
    background = (labels.to_numpy() == "") & evidence.isin(SCAN_EVIDENCE_VALUES).to_numpy()
    return labeled, background


def _parse_class_names(raw: str | None, metadata: pd.DataFrame, labeled_mask: np.ndarray) -> list[str]:
    observed = sorted(pd.unique(metadata.loc[labeled_mask, "sv_class"].astype(str)).tolist()) if labeled_mask.any() else []
    names = [part.strip() for part in str(raw or "").split(",") if part.strip()] or list(DEFAULT_CLASS_NAMES)
    unknown = sorted(set(observed).difference(names))
    if unknown:
        raise ValueError(f"Observed labels not present in --class_names: {unknown}; class_names={names}")
    return names


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
            targets[i] = class_to_idx.get(label, -1)
    return targets


def _class_counts(metadata: pd.DataFrame, labeled_mask: np.ndarray, class_names: list[str]) -> dict[str, int]:
    labels = metadata.loc[labeled_mask, "sv_class"].astype(str) if "sv_class" in metadata else pd.Series(dtype=str)
    counts = labels.value_counts().to_dict()
    return {name: int(counts.get(name, 0)) for name in class_names}


def _parameter_count(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _class_weights(targets: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    counts = np.bincount(targets[targets >= 0], minlength=int(n_classes)).astype(np.float32)
    present = counts > 0
    weights = np.ones(int(n_classes), dtype=np.float32)
    if mode == "inverse":
        weights[present] = float(counts[present].sum()) / np.maximum(counts[present], 1.0)
    elif mode == "inverse_sqrt":
        weights[present] = np.sqrt(float(counts[present].sum()) / np.maximum(counts[present], 1.0))
    else:
        raise ValueError(f"Unknown class weighting mode: {mode}")
    if present.any():
        weights[present] = weights[present] / np.mean(weights[present])
    return torch.as_tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Hierarchy helpers (torch, used during training)
# ---------------------------------------------------------------------------

def _family_targets_torch(
    y_class: torch.Tensor,
    class_names: list[str],
    family_names: list[str],
) -> torch.Tensor:
    """Map per-example class indices to family indices."""
    result = torch.full_like(y_class, -1)
    for class_idx, cn in enumerate(class_names):
        fam = FAMILY_MAP.get(cn, cn)
        if fam in family_names:
            fam_idx = family_names.index(fam)
            result[y_class == class_idx] = fam_idx
    return result


def _subtype_mask_and_targets(
    y_class: torch.Tensor,
    class_names: list[str],
    family_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (bool mask, remapped 2-class subtype targets) for examples in family_name."""
    subtypes = FAMILY_SUBTYPES.get(family_name, [])
    mask = torch.zeros(len(y_class), dtype=torch.bool, device=y_class.device)
    sub_targets = torch.full_like(y_class, -1)
    for sub_idx, sub_name in enumerate(subtypes):
        if sub_name in class_names:
            ci = class_names.index(sub_name)
            m = y_class == ci
            mask |= m
            sub_targets[m] = sub_idx
    return mask, sub_targets[mask]


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model(input_dim: int, projection_dim: int, hidden_dim: int, dropout: float, device: torch.device) -> MetricProjection:
    return MetricProjection(
        in_dim=int(input_dim),
        embed_dim=int(projection_dim),
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
    ).to(device)


# ---------------------------------------------------------------------------
# LOO prototype logits (used at both Level 2 and Level 3)
# ---------------------------------------------------------------------------

def _loo_prototype_logits(
    projected: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squared-Euclidean LOO prototype logits. Returns (logits [N, C], usable mask [N])."""
    dim = int(projected.shape[1])
    sums = torch.zeros((int(n_classes), dim), dtype=projected.dtype, device=projected.device)
    sums.index_add_(0, targets, projected)
    counts = torch.bincount(targets, minlength=int(n_classes)).to(projected.device)
    logits = torch.full((projected.shape[0], int(n_classes)), -1.0e9, dtype=projected.dtype, device=projected.device)
    usable = torch.zeros(projected.shape[0], dtype=torch.bool, device=projected.device)

    for i in range(projected.shape[0]):
        query_class = int(targets[i].detach().cpu().item())
        for class_idx in range(int(n_classes)):
            count = int(counts[class_idx].detach().cpu().item())
            if count <= 0:
                continue
            if class_idx == query_class:
                if count <= 1:
                    continue
                proto = (sums[class_idx] - projected[i]) / float(count - 1)
            else:
                proto = sums[class_idx] / float(count)
            logits[i, class_idx] = -torch.sum((projected[i] - proto) ** 2) / float(temperature)
        usable[i] = logits[i, query_class] > -1.0e8
    return logits, usable


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_projection(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    family_names: list[str],
    train_mask: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    epochs: int,
    patience: int,
    seed_offset: int = 0,
    log_prefix: str = "train",
) -> tuple[MetricProjection, pd.DataFrame]:
    set_seed(int(args.seed) + int(seed_offset))
    labeled_mask, _ = _label_masks(metadata)
    targets_all = _type_targets(metadata, class_names)
    train_idx = np.where(train_mask & labeled_mask)[0]
    if train_idx.size == 0:
        raise RuntimeError(f"No labeled rows available for {log_prefix}")
    train_targets_np = targets_all[train_idx]

    # Need ≥2 families with ≥2 examples each for Level 2 LOO
    family_targets_np = np.array([
        family_names.index(FAMILY_MAP.get(class_names[t], class_names[t]))
        if t >= 0 and FAMILY_MAP.get(class_names[t], class_names[t]) in family_names else -1
        for t in train_targets_np
    ], dtype=np.int64)
    fam_counts = np.bincount(family_targets_np[family_targets_np >= 0], minlength=len(family_names))
    if int((fam_counts >= 2).sum()) < 2:
        raise RuntimeError(
            f"Need at least two families with two examples each for {log_prefix}; family_counts={fam_counts.tolist()}"
        )

    model = make_model(embeddings.shape[1], int(args.projection_dim), int(args.hidden_dim), float(args.dropout), device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    x_train = torch.as_tensor(embeddings[train_idx], dtype=torch.float32, device=device)
    y_class = torch.as_tensor(train_targets_np, dtype=torch.long, device=device)
    y_family = _family_targets_torch(y_class, class_names, family_names)

    # Class weights for Level 2 (family-level)
    family_weight_t = _class_weights(family_targets_np, len(family_names), str(args.class_weighting))
    if family_weight_t is not None:
        family_weight_t = family_weight_t.to(device)

    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience_left = int(patience)
    rows: list[dict[str, Any]] = []

    for epoch in range(1, int(epochs) + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        # Noise augmentation
        if float(args.embedding_noise_sigma) > 0:
            x_input = x_train + float(args.embedding_noise_sigma) * torch.randn_like(x_train)
        else:
            x_input = x_train
        projected = model(x_input)

        # Level 2: family LOO loss
        usable_fam = y_family >= 0
        if not usable_fam.any():
            raise RuntimeError(f"No family-labeled rows for {log_prefix}")
        logits_L2, usable_L2 = _loo_prototype_logits(
            projected[usable_fam], y_family[usable_fam], len(family_names), float(args.temperature)
        )
        L2 = F.cross_entropy(
            logits_L2[usable_L2],
            y_family[usable_fam][usable_L2],
            weight=family_weight_t,
            label_smoothing=float(args.label_smoothing),
        )
        family_acc = float((torch.argmax(logits_L2[usable_L2], dim=1) == y_family[usable_fam][usable_L2]).float().mean().item())

        # Level 3: per-family subtype LOO losses
        L3 = torch.zeros((), dtype=projected.dtype, device=device)
        subtype_accs: list[float] = []
        for family_name in FAMILY_SUBTYPES:
            if family_name not in family_names:
                continue
            fmask, sub_targets = _subtype_mask_and_targets(y_class, class_names, family_name)
            if not fmask.any():
                continue
            sub_counts = torch.bincount(sub_targets, minlength=2)
            if int((sub_counts >= 2).sum()) < 2:
                continue
            sub_proj = projected[fmask]
            logits_L3, usable_L3 = _loo_prototype_logits(sub_proj, sub_targets, 2, float(args.temperature))
            if not usable_L3.any():
                continue
            sub_weight = _class_weights(sub_targets.detach().cpu().numpy(), 2, str(args.class_weighting))
            if sub_weight is not None:
                sub_weight = sub_weight.to(device)
            L3 = L3 + F.cross_entropy(
                logits_L3[usable_L3],
                sub_targets[usable_L3],
                weight=sub_weight,
                label_smoothing=float(args.label_smoothing),
            )
            subtype_accs.append(
                float((torch.argmax(logits_L3[usable_L3], dim=1) == sub_targets[usable_L3]).float().mean().item())
            )

        loss = L2 + float(args.hierarchical_lambda) * L3
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(args.grad_clip))
        opt.step()

        loss_value = float(loss.detach().cpu().item())
        subtype_acc = float(np.mean(subtype_accs)) if subtype_accs else float("nan")
        improved = loss_value < best_loss - 1e-6
        if improved:
            best_loss = loss_value
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = int(patience)
        else:
            patience_left -= 1

        rows.append({
            "epoch": int(epoch),
            "loss": loss_value,
            "family_loo_acc": family_acc,
            "subtype_loo_acc": subtype_acc,
            "n_train_labeled": int(train_idx.size),
            "n_usable_family": int(usable_L2.detach().cpu().numpy().sum()),
            "is_best": bool(improved),
            "split": log_prefix,
        })
        if log_prefix == "final" and (epoch == 1 or epoch % int(args.log_every) == 0 or epoch == int(epochs)):
            log.info(
                "epoch=%d loss=%.4f family_acc=%.3f subtype_acc=%.3f",
                epoch, loss_value, family_acc, subtype_acc,
            )
        if int(patience) > 0 and patience_left <= 0:
            if log_prefix == "final":
                log.info("Early stopping at epoch %d; best loss %.4f", epoch, best_loss)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def project_embeddings(
    model: MetricProjection,
    embeddings: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, embeddings.shape[0], int(batch_size)):
            batch = torch.as_tensor(embeddings[start : start + int(batch_size)], dtype=torch.float32, device=device)
            rows.append(model(batch).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0)


def compute_hierarchical_prototypes(
    projected: np.ndarray,
    type_targets: np.ndarray,
    class_names: list[str],
    family_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Compute L2-normalized family and subtype prototype means.

    Returns
    -------
    family_prototypes : [F, D]
    family_counts : [F]
    subtype_prototypes : dict family_name -> [S, D]
    subtype_counts : dict family_name -> [S]
    """
    dim = int(projected.shape[1])
    n_fam = len(family_names)
    family_prototypes = np.zeros((n_fam, dim), dtype=np.float32)
    family_counts = np.zeros(n_fam, dtype=np.int64)

    for fam_idx, fam_name in enumerate(family_names):
        member_class_indices = [
            ci for ci, cn in enumerate(class_names)
            if FAMILY_MAP.get(cn, cn) == fam_name
        ]
        mask = np.zeros(len(type_targets), dtype=bool)
        for ci in member_class_indices:
            mask |= type_targets == ci
        if mask.any():
            proto = projected[mask].mean(axis=0)
            norm = float(np.linalg.norm(proto))
            family_prototypes[fam_idx] = proto / norm if norm > 0 else proto
            family_counts[fam_idx] = int(mask.sum())

    subtype_prototypes: dict[str, np.ndarray] = {}
    subtype_counts: dict[str, np.ndarray] = {}
    for fam_name, subtypes in FAMILY_SUBTYPES.items():
        if fam_name not in family_names:
            continue
        n_sub = len(subtypes)
        sub_protos = np.zeros((n_sub, dim), dtype=np.float32)
        sub_cnts = np.zeros(n_sub, dtype=np.int64)
        for sub_idx, sub_name in enumerate(subtypes):
            if sub_name in class_names:
                ci = class_names.index(sub_name)
                mask = type_targets == ci
                if mask.any():
                    proto = projected[mask].mean(axis=0)
                    norm = float(np.linalg.norm(proto))
                    sub_protos[sub_idx] = proto / norm if norm > 0 else proto
                    sub_cnts[sub_idx] = int(mask.sum())
        subtype_prototypes[fam_name] = sub_protos
        subtype_counts[fam_name] = sub_cnts

    return family_prototypes, family_counts, subtype_prototypes, subtype_counts


def _softmax_logits(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    exp[~np.isfinite(exp)] = 0.0
    denom = exp.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return exp / denom


def predict_hierarchical(
    model: MetricProjection,
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    family_prototypes: np.ndarray,
    family_counts: np.ndarray,
    subtype_prototypes: dict[str, np.ndarray],
    subtype_counts: dict[str, np.ndarray],
    distance_tau: float,
    objectness_scale: float,
    temperature: float,
    device: torch.device,
    batch_size: int = 512,
    family_names: list[str] | None = None,
) -> pd.DataFrame:
    if family_names is None:
        family_names = _build_family_names(class_names)

    projected = project_embeddings(model, embeddings, device=device, batch_size=batch_size)
    N = projected.shape[0]

    # ---- Level 2: family distances ----
    absent_fam = np.asarray(family_counts) <= 0
    family_distances = np.sum(
        (projected[:, None, :] - family_prototypes[None, :, :]) ** 2, axis=2
    ).astype(np.float32)  # [N, F]
    family_distances[:, absent_fam] = np.inf

    family_logits = -family_distances / float(temperature)
    family_logits[:, absent_fam] = -1.0e9
    family_probs = _softmax_logits(family_logits)  # [N, F]

    nearest_family_idx = np.argmin(family_distances, axis=1)  # [N]
    nearest_family_distance = family_distances[np.arange(N), nearest_family_idx].astype(np.float32)
    nearest_family_distance[~np.isfinite(nearest_family_distance)] = np.inf

    objectness_prob = 1.0 / (1.0 + np.exp(
        -float(objectness_scale) * (float(distance_tau) - nearest_family_distance)
    ))
    objectness_prob[~np.isfinite(objectness_prob)] = 0.0

    # ---- Level 3: subtype distances (vectorised per-family) ----
    type_predicted_class = np.full(N, "", dtype=object)
    max_type_probability = np.zeros(N, dtype=np.float32)
    is_canonical = np.full(N, "", dtype=object)

    # initialise per-class probability and distance columns
    type_prob_cols: dict[str, np.ndarray] = {cn: np.zeros(N, dtype=np.float32) for cn in class_names}
    proto_dist_cols: dict[str, np.ndarray] = {cn: np.full(N, np.inf, dtype=np.float32) for cn in class_names}

    for fam_idx, fam_name in enumerate(family_names):
        fam_mask = nearest_family_idx == fam_idx
        if not fam_mask.any():
            continue
        global_indices = np.where(fam_mask)[0]
        fam_proj = projected[fam_mask]  # [M, D]

        if fam_name in FAMILY_SUBTYPES:
            subtypes = FAMILY_SUBTYPES[fam_name]
            sub_protos = subtype_prototypes.get(fam_name)
            sub_cnts = subtype_counts.get(fam_name)
            if sub_protos is None or sub_cnts is None:
                # fallback: treat family as its own single class
                type_predicted_class[global_indices] = fam_name
                max_type_probability[global_indices] = family_probs[global_indices, fam_idx]
                is_canonical[global_indices] = "True"
                continue

            sub_absent = sub_cnts <= 0
            sub_dists = np.sum(
                (fam_proj[:, None, :] - sub_protos[None, :, :]) ** 2, axis=2
            ).astype(np.float32)  # [M, S]
            sub_dists[:, sub_absent] = np.inf

            sub_logits = -sub_dists / float(temperature)
            sub_logits[:, sub_absent] = -1.0e9
            sub_probs = _softmax_logits(sub_logits)  # [M, S]

            nearest_sub = np.argmin(sub_dists, axis=1)  # [M]

            for m_i, g_i in enumerate(global_indices):
                sub_idx = int(nearest_sub[m_i])
                sub_name = subtypes[sub_idx] if sub_idx < len(subtypes) else fam_name
                type_predicted_class[g_i] = sub_name
                fam_prob = float(family_probs[g_i, fam_idx])
                max_type_probability[g_i] = float(fam_prob * sub_probs[m_i, sub_idx])
                is_canonical[g_i] = str("non_canonical" not in sub_name)

            # populate per-class columns for all subtypes
            for si, sn in enumerate(subtypes):
                if sn in type_prob_cols:
                    fam_prob_vec = family_probs[global_indices, fam_idx]
                    type_prob_cols[sn][global_indices] = (fam_prob_vec * sub_probs[:, si]).astype(np.float32)
                    proto_dist_cols[sn][global_indices] = sub_dists[:, si]
        else:
            # seismic_amplification / TIC: family IS the final class
            type_predicted_class[global_indices] = fam_name
            max_type_probability[global_indices] = family_probs[global_indices, fam_idx].astype(np.float32)
            is_canonical[global_indices] = "True"
            if fam_name in type_prob_cols:
                type_prob_cols[fam_name][global_indices] = family_probs[global_indices, fam_idx].astype(np.float32)
                proto_dist_cols[fam_name][global_indices] = nearest_family_distance[global_indices]

    # ---- Multi-label: per-family objectness (for predicted_families column) ----
    objectness_tau_default = 0.5
    per_fam_obj = 1.0 / (1.0 + np.exp(
        -float(objectness_scale) * (float(distance_tau) - family_distances)
    ))  # [N, F]
    per_fam_obj[~np.isfinite(per_fam_obj)] = 0.0
    predicted_families_list: list[str] = []
    for i in range(N):
        active = [
            family_names[fi]
            for fi in range(len(family_names))
            if float(per_fam_obj[i, fi]) >= objectness_tau_default and not np.isinf(family_distances[i, fi])
        ]
        predicted_families_list.append(",".join(active))

    # ---- Assemble output DataFrame ----
    labeled_mask, background_mask = _label_masks(metadata)
    out = metadata.copy()
    out["is_labeled"] = labeled_mask.astype(int)
    out["is_background_chromosome"] = background_mask.astype(int)
    out["true_class"] = out["sv_class"].astype(str) if "sv_class" in out else ""
    out["nearest_prototype_class"] = [family_names[int(fi)] for fi in nearest_family_idx]
    out["nearest_prototype_distance"] = nearest_family_distance.astype(float)
    out["objectness_logit"] = (float(objectness_scale) * (float(distance_tau) - nearest_family_distance)).astype(float)
    out["objectness_prob"] = objectness_prob.astype(float)
    out["type_predicted_class"] = type_predicted_class.tolist()
    out["max_type_probability"] = max_type_probability.astype(float)
    out["predicted_families"] = predicted_families_list
    out["is_canonical"] = is_canonical.tolist()

    for cn in class_names:
        out[f"type_probability_{cn}"] = type_prob_cols[cn]
        out[f"prototype_distance_{cn}"] = proto_dist_cols[cn]

    for fi, fn in enumerate(family_names):
        out[f"family_probability_{fn}"] = family_probs[:, fi].astype(float)
        out[f"family_distance_{fn}"] = family_distances[:, fi].astype(float)

    return out


def annotate_predictions(
    predictions: pd.DataFrame,
    distance_tau: float,
    objectness_scale: float,
    objectness_tau: float = 0.5,
) -> pd.DataFrame:
    out = predictions.copy()
    nearest = out["nearest_prototype_distance"].astype(float).to_numpy()
    objectness_logit = float(objectness_scale) * (float(distance_tau) - nearest)
    objectness_prob = 1.0 / (1.0 + np.exp(-objectness_logit))
    objectness_prob[~np.isfinite(objectness_prob)] = 0.0
    out["selected_distance_tau"] = float(distance_tau)
    out["objectness_tau"] = float(objectness_tau)
    out["objectness_logit"] = objectness_logit.astype(float)
    out["objectness_prob"] = objectness_prob.astype(float)
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


def fewshot_predictions_to_distance_table(predictions: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    out = predictions.copy()
    objectness = out["objectness_prob"].astype(float).to_numpy()
    max_type = out["max_type_probability"].astype(float).to_numpy()
    called = out["predicted_class"].astype(str).to_numpy() != "none"
    out["distance_source"] = "fewshot_hierarchical_prototype"
    out["prototype_confidence"] = np.where(called, objectness * max_type, 0.0)
    for class_name in class_names:
        src = f"prototype_distance_{class_name}"
        if src in out:
            out[f"d_{class_name}"] = out[src].astype(float)
    return out


# ---------------------------------------------------------------------------
# Tau calibration
# ---------------------------------------------------------------------------

def sweep_distance_tau(score_df: pd.DataFrame, tau_grid: np.ndarray) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame()
    eligible = (score_df["is_labeled"].astype(bool) | score_df["is_background_chromosome"].astype(bool)).to_numpy()
    df = score_df.loc[eligible].copy()
    if df.empty:
        return pd.DataFrame()
    y = df["is_labeled"].astype(bool).to_numpy()
    distance = df["nearest_prototype_distance"].astype(float).to_numpy()
    true_cls = df["true_class"].astype(str).to_numpy()
    type_cls = df["type_predicted_class"].astype(str).to_numpy()
    rows: list[dict[str, Any]] = []
    for tau in tau_grid:
        called = distance <= float(tau)
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
        rows.append({
            "distance_tau": float(tau),
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
        })
    return pd.DataFrame(rows)


def choose_distance_tau(tau_df: pd.DataFrame, metric: str = "typed_f1") -> float:
    if tau_df.empty:
        return 0.5
    metric = metric if metric in tau_df.columns else "f1"
    ranked = tau_df.sort_values(
        [metric, "precision", "recall", "distance_tau"],
        ascending=[False, False, False, True],
    )
    return float(ranked.iloc[0]["distance_tau"])


# ---------------------------------------------------------------------------
# Leave-one-sample-out cross-validation
# ---------------------------------------------------------------------------

def run_leave_one_sample_out(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    family_names: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labeled_mask, background_mask = _label_masks(metadata)
    samples = _sample_ids(metadata)
    targets = _type_targets(metadata, class_names)
    held_samples = sorted(pd.unique(samples[labeled_mask]).tolist())
    rows: list[pd.DataFrame] = []
    train_metrics: list[pd.DataFrame] = []
    cv_epochs = int(args.cv_epochs) if args.cv_epochs is not None else int(args.epochs)
    cv_patience = int(args.cv_patience) if args.cv_patience is not None else int(args.patience)

    for fold_i, held_sample in enumerate(held_samples):
        train_mask = samples != held_sample
        train_labels = labeled_mask & train_mask
        if int(train_labels.sum()) == 0:
            log.warning("Skipping LOSO fold for %s; no labels remain", held_sample)
            continue
        try:
            model, metrics = train_projection(
                embeddings,
                metadata,
                class_names,
                family_names,
                train_mask=train_mask,
                args=args,
                device=device,
                epochs=cv_epochs,
                patience=cv_patience,
                seed_offset=1000 + fold_i,
                log_prefix=f"loso:{held_sample}",
            )
        except RuntimeError as exc:
            log.warning("Skipping LOSO fold for %s: %s", held_sample, exc)
            continue
        metrics["held_out_sample"] = held_sample
        train_metrics.append(metrics)

        projected_train = project_embeddings(model, embeddings[train_labels], device=device, batch_size=int(args.batch_size))
        fam_protos, fam_counts, sub_protos, sub_counts = compute_hierarchical_prototypes(
            projected_train, targets[train_labels], class_names, family_names
        )
        pred = predict_hierarchical(
            model,
            embeddings,
            metadata,
            class_names,
            fam_protos,
            fam_counts,
            sub_protos,
            sub_counts,
            distance_tau=0.5,
            objectness_scale=float(args.objectness_scale),
            temperature=float(args.temperature),
            device=device,
            batch_size=int(args.batch_size),
            family_names=family_names,
        )
        eval_mask = (samples == held_sample) & (labeled_mask | background_mask)
        fold = pred.loc[eval_mask].copy()
        fold["held_out_sample"] = held_sample
        fold["train_n_labeled"] = int(train_labels.sum())
        fold["train_class_counts"] = json.dumps(
            _class_counts(
                metadata.loc[train_mask].reset_index(drop=True),
                _label_masks(metadata.loc[train_mask].reset_index(drop=True))[0],
                class_names,
            ),
            sort_keys=True,
        )
        rows.append(fold)

    cv_predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    cv_metrics = pd.concat(train_metrics, ignore_index=True) if train_metrics else pd.DataFrame()
    return cv_predictions, cv_metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_training(metrics: pd.DataFrame, output_path: Path) -> None:
    if metrics.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(metrics["epoch"], metrics["loss"], label="total loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Hierarchical Few-Shot Loss")
    axes[0].grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[1].plot(metrics["epoch"], metrics["family_loo_acc"], label="family LOO acc (L2)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_title("Level 2: Family LOO Accuracy")
    axes[1].grid(alpha=0.2)
    axes[1].legend(fontsize=8)
    axes[2].plot(metrics["epoch"], metrics["subtype_loo_acc"], label="subtype LOO acc (L3)", color="tab:orange")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].set_title("Level 3: Subtype LOO Accuracy")
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_tau(tau_df: pd.DataFrame, output_path: Path, selected_tau: float) -> None:
    if tau_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.plot(tau_df["distance_tau"], tau_df["precision"], label="precision")
    ax.plot(tau_df["distance_tau"], tau_df["recall"], label="recall")
    ax.plot(tau_df["distance_tau"], tau_df["f1"], label="F1")
    if "typed_f1" in tau_df:
        ax.plot(tau_df["distance_tau"], tau_df["typed_f1"], label="typed F1")
    ax.axvline(float(selected_tau), color="black", linestyle=":", linewidth=1.2, label=f"selected tau={selected_tau:.3g}")
    ax.set_xlabel("Nearest-family distance threshold")
    ax.set_ylabel("Score")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("LOSO Distance Sweep (family-level tau)")
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
    ax.set_xlabel("Predicted class (Level 3 subtype)")
    ax.set_ylabel("True class")
    ax.set_title("Held-Out Hierarchical Few-Shot Predictions")
    max_value = max(float(table.to_numpy().max()), 1.0)
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, str(int(table.iat[i, j])), ha="center", va="center",
                    color="white" if table.iat[i, j] > max_value * 0.45 else "black", fontsize=9)
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
    eligible = eligible.sort_values("nearest_prototype_distance", ascending=True).reset_index(drop=True)
    colors = np.where(eligible["is_labeled"].astype(bool), "#c43b3b", "#4a78b8")
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.scatter(np.arange(len(eligible)), eligible["nearest_prototype_distance"].astype(float),
               c=colors, s=18, alpha=0.85, linewidths=0)
    ax.axhline(float(selected_tau), color="black", linestyle=":", linewidth=1.2, label=f"family tau={selected_tau:.3g}")
    ax.set_xlabel("Held-out rows sorted by nearest-family distance")
    ax.set_ylabel("Nearest-family distance")
    ax.set_title("Held-Out Hierarchical Few-Shot Distances")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_haplotype_summary(predictions: pd.DataFrame, output_path: Path) -> None:
    if "haplotype" not in predictions.columns:
        return
    called = predictions[predictions.get("called_complex_sv", pd.Series(dtype=bool)).astype(bool)].copy()
    if called.empty:
        return
    classes = sorted(called["type_predicted_class"].astype(str).unique())
    hap_order = ["HP1", "HP2", "bilateral"]
    colors = {"HP1": "#4E79A7", "HP2": "#E15759", "bilateral": "#76B7B2"}

    counts = (
        called.groupby(["type_predicted_class", "haplotype"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=classes, columns=hap_order, fill_value=0)
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.6 * len(classes) + 2.0)))

    x = np.arange(len(classes))
    bottom = np.zeros(len(classes))
    for hap in hap_order:
        vals = counts[hap].to_numpy(dtype=float) if hap in counts else np.zeros(len(classes))
        axes[0].bar(x, vals, bottom=bottom, label=hap, color=colors[hap], alpha=0.85)
        bottom += vals
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    axes[0].set_ylabel("Called complex SVs")
    axes[0].set_title("Haplotype by Predicted Class")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    if "haplotype_score" in called.columns:
        for hap in hap_order:
            subset = called[called["haplotype"] == hap]["haplotype_score"].astype(float)
            if not subset.empty:
                axes[1].scatter(
                    subset.values,
                    np.random.default_rng(0).uniform(-0.3, 0.3, len(subset)),
                    label=hap, color=colors[hap], s=25, alpha=0.75, linewidths=0,
                )
        axes[1].axvline(0.25, color="gray", linestyle="--", linewidth=0.8)
        axes[1].axvline(-0.25, color="gray", linestyle="--", linewidth=0.8)
        axes[1].set_xlabel("Haplotype score (−1=HP2, +1=HP1)")
        axes[1].set_yticks([])
        axes[1].set_title("Haplotype Score Distribution")
        axes[1].legend(fontsize=8)
        axes[1].grid(axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)



def _write_prediction_view(predictions: pd.DataFrame, class_names: list[str], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for row in predictions.to_dict("records"):
        out = {
            "candidate_id": row.get("candidate_id", ""),
            "sample_id": row.get("sample_id", ""),
            "chrom": row.get("chrom", ""),
            "arm": row.get("arm", ""),
            "start_bp": int(row.get("start_bp", 0)),
            "end_bp": int(row.get("end_bp", 0)),
            "predicted_class": row.get("predicted_class", "none"),
            "called_complex_sv": bool(row.get("called_complex_sv", False)),
            "objectness_prob": float(row.get("objectness_prob", 0.0)),
            "nearest_prototype_distance": float(row.get("nearest_prototype_distance", np.inf)),
            "nearest_family": row.get("nearest_prototype_class", ""),
            "type_predicted_class": row.get("type_predicted_class", ""),
            "max_type_probability": float(row.get("max_type_probability", 0.0)),
            "predicted_families": row.get("predicted_families", ""),
            "is_canonical": row.get("is_canonical", ""),
            "evidence": row.get("evidence", ""),
            "candidate_scope": row.get("candidate_scope", row.get("label_scope", "")),
            "true_class": row.get("true_class", ""),
        }
        for class_name in class_names:
            out[f"type_probability_{class_name}"] = float(row.get(f"type_probability_{class_name}", 0.0))
        rows.append(out)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(int(args.seed))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    embeddings, metadata = load_embedding_table(args.embeddings_npz, args.metadata_tsv)
    embeddings, metadata = enforce_candidate_resolution(embeddings, metadata, args.candidate_resolution)
    labeled_mask, background_mask = _label_masks(metadata)
    if int(labeled_mask.sum()) < 4:
        raise RuntimeError("Need at least four labeled embeddings for hierarchical few-shot training")
    class_names = _parse_class_names(args.class_names, metadata, labeled_mask)
    family_names = _build_family_names(class_names)
    class_counts = _class_counts(metadata, labeled_mask, class_names)
    log.info(
        "device=%s embeddings=%s class_counts=%s family_names=%s background=%d",
        device, embeddings.shape, class_counts, family_names, int(background_mask.sum()),
    )

    all_train_mask = np.ones(len(metadata), dtype=bool)
    final_model, training_metrics = train_projection(
        embeddings,
        metadata,
        class_names,
        family_names,
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

    tau_grid = np.linspace(float(args.distance_tau_min), float(args.distance_tau_max), int(args.distance_tau_steps), dtype=np.float32)
    cv_annotated = pd.DataFrame()
    if bool(args.skip_loso):
        cv_predictions = pd.DataFrame()
        cv_training_metrics = pd.DataFrame()
        tau_df = pd.DataFrame()
        selected_distance_tau = float(args.distance_tau) if args.distance_tau is not None else 0.5
    else:
        cv_predictions, cv_training_metrics = run_leave_one_sample_out(
            embeddings, metadata, class_names, family_names, args, device
        )
        cv_training_metrics.to_csv(out_dir / "loso_training_metrics.tsv", sep="\t", index=False)
        tau_df = sweep_distance_tau(cv_predictions, tau_grid)
        selected_distance_tau = (
            float(args.distance_tau)
            if args.distance_tau is not None
            else choose_distance_tau(tau_df, metric=str(args.tau_selection_metric))
        )
        cv_annotated = (
            annotate_predictions(cv_predictions, selected_distance_tau, float(args.objectness_scale), objectness_tau=0.5)
            if not cv_predictions.empty
            else cv_predictions
        )
        cv_predictions.to_csv(out_dir / "leave_one_sample_out_raw.tsv", sep="\t", index=False)
        cv_annotated.to_csv(out_dir / "leave_one_sample_out.tsv", sep="\t", index=False)
        tau_df.to_csv(out_dir / "distance_tau_sweep.tsv", sep="\t", index=False)
        _plot_tau(tau_df, out_dir / "distance_tau_sweep.png", selected_distance_tau)
        _plot_validation_confusion(cv_annotated, class_names, out_dir / "held_out_prediction_summary.png")
        _plot_objectness_scores(cv_annotated, out_dir / "held_out_distance_scores.png", selected_distance_tau)
        _plot_haplotype_summary(cv_annotated, out_dir / "held_out_haplotype_summary.png")

    targets = _type_targets(metadata, class_names)
    projected_labeled = project_embeddings(final_model, embeddings[labeled_mask], device=device, batch_size=int(args.batch_size))
    fam_protos, fam_counts, sub_protos, sub_counts = compute_hierarchical_prototypes(
        projected_labeled, targets[labeled_mask], class_names, family_names
    )

    predictions = predict_hierarchical(
        final_model,
        embeddings,
        metadata,
        class_names,
        fam_protos,
        fam_counts,
        sub_protos,
        sub_counts,
        distance_tau=selected_distance_tau,
        objectness_scale=float(args.objectness_scale),
        temperature=float(args.temperature),
        device=device,
        batch_size=int(args.batch_size),
        family_names=family_names,
    )
    predictions = annotate_predictions(predictions, selected_distance_tau, float(args.objectness_scale), objectness_tau=0.5)
    predictions.to_csv(out_dir / "classification_predictions.tsv", sep="\t", index=False)
    called = predictions[predictions["called_complex_sv"].astype(bool)].copy()
    called.to_csv(out_dir / "predicted_complex_sv.tsv", sep="\t", index=False)
    _write_prediction_view(predictions, class_names, out_dir / "predictions.tsv")
    _plot_haplotype_summary(predictions, out_dir / "haplotype_summary.png")

    compatibility_distances = fewshot_predictions_to_distance_table(predictions, class_names)
    compatibility_distances.to_csv(out_dir / "prototype_distances.tsv", sep="\t", index=False)
    compatibility_loo = (
        fewshot_predictions_to_distance_table(cv_annotated, class_names)
        if not cv_annotated.empty
        else pd.DataFrame()
    )
    compatibility_loo.to_csv(out_dir / "anchor_leave_one_out.tsv", sep="\t", index=False)
    embed_corpus.write_visualizations(
        embeddings,
        metadata,
        compatibility_distances,
        compatibility_loo,
        out_dir,
        tau=float(selected_distance_tau),
    )

    parameter_count = _parameter_count(final_model)
    architecture = {
        "model": "MetricProjection + hierarchical family/subtype prototypes",
        "input_dim": int(embeddings.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "projection_dim": int(args.projection_dim),
        "dropout": float(args.dropout),
        "normalization": "L2-normalized projected embeddings and prototypes",
        "level2_distance": "squared Euclidean to family prototypes",
        "level3_distance": "squared Euclidean to subtype prototypes within family",
        "family_names": family_names,
        "family_subtypes": FAMILY_SUBTYPES,
        "objectness": f"sigmoid({float(args.objectness_scale):.6g} * (family_tau - nearest_family_distance))",
    }

    # serialise subtype prototype tensors for checkpoint
    sub_protos_serialised = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in sub_protos.items()}
    sub_counts_serialised = {k: torch.as_tensor(v, dtype=torch.long) for k, v in sub_counts.items()}

    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "input_dim": int(embeddings.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "projection_dim": int(args.projection_dim),
            "dropout": float(args.dropout),
            "class_names": class_names,
            "class_counts": class_counts,
            "family_map": FAMILY_MAP,
            "family_names": family_names,
            "family_prototype_vectors": torch.as_tensor(fam_protos, dtype=torch.float32),
            "family_prototype_counts": torch.as_tensor(fam_counts, dtype=torch.long),
            "subtype_prototype_vectors": sub_protos_serialised,
            "subtype_prototype_counts": sub_counts_serialised,
            "selected_distance_tau": float(selected_distance_tau),
            "objectness_tau": 0.5,
            "objectness_scale": float(args.objectness_scale),
            "temperature": float(args.temperature),
            "hierarchical_lambda": float(args.hierarchical_lambda),
            "embedding_noise_sigma": float(args.embedding_noise_sigma),
            "parameter_count": int(parameter_count),
            "architecture": architecture,
            "config": vars(args),
        },
        out_dir / "fewshot_classification_head.pt",
    )

    summary: dict[str, Any] = {
        "architecture": architecture,
        "parameter_count": int(parameter_count),
        "class_names": class_names,
        "class_counts": class_counts,
        "family_names": family_names,
        "family_counts": {fn: int(fam_counts[fi]) for fi, fn in enumerate(family_names)},
        "selected_distance_tau": float(selected_distance_tau),
        "objectness_tau": 0.5,
        "objectness_scale": float(args.objectness_scale),
        "temperature": float(args.temperature),
        "hierarchical_lambda": float(args.hierarchical_lambda),
        "embedding_noise_sigma": float(args.embedding_noise_sigma),
        "tau_selection_metric": str(args.tau_selection_metric),
        "n_labeled": int(labeled_mask.sum()),
        "n_background": int(background_mask.sum()),
        "n_called_complex_sv": int(called.shape[0]),
    }
    if not tau_df.empty:
        best_row = tau_df.loc[(tau_df["distance_tau"] - float(selected_distance_tau)).abs().idxmin()].to_dict()
        summary["selected_tau_metrics"] = {
            k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
            for k, v in best_row.items()
        }
    with (out_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote hierarchical few-shot outputs to %s; called=%d/%d", out_dir, int(called.shape[0]), len(predictions))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings_npz", required=True)
    parser.add_argument("--metadata_tsv", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_resolution", choices=("auto", "chromosome-arm", "any"), default="chromosome-arm")
    parser.add_argument("--class_names", default=",".join(DEFAULT_CLASS_NAMES))
    parser.add_argument("--projection_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--objectness_scale", type=float, default=12.0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--cv_epochs", type=int, default=220)
    parser.add_argument("--cv_patience", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--label_smoothing", type=float, default=0.02)
    parser.add_argument("--class_weighting", choices=("none", "inverse", "inverse_sqrt"), default="inverse_sqrt")
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--distance_tau_min", type=float, default=0.0)
    parser.add_argument("--distance_tau_max", type=float, default=4.0)
    parser.add_argument("--distance_tau_steps", type=int, default=161)
    parser.add_argument("--distance_tau", type=float, default=None)
    parser.add_argument("--tau_selection_metric", choices=("f1", "typed_f1", "precision", "recall"), default="typed_f1")
    parser.add_argument("--skip_loso", action="store_true")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=25)
    parser.add_argument("--embedding_noise_sigma", type=float, default=0.02,
                        help="Gaussian noise std added to training embeddings each epoch (0=disabled).")
    parser.add_argument("--hierarchical_lambda", type=float, default=0.5,
                        help="Weight for Level 3 subtype LOO loss relative to Level 2 family loss.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
