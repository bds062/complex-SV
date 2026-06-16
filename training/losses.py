"""Loss functions for supervised fusion/head training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def episodic_loss(
    support_embs: torch.Tensor,
    support_labels: torch.Tensor,
    query_embs: torch.Tensor,
    query_labels: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    classes = torch.unique(support_labels)
    prototypes = []
    for cls in classes:
        prototypes.append(support_embs[support_labels == cls].mean(dim=0))
    prototypes = F.normalize(torch.stack(prototypes, dim=0), p=2, dim=-1)
    query_embs = F.normalize(query_embs, p=2, dim=-1)
    logits = -torch.cdist(query_embs, prototypes, p=2).pow(2)
    remapped = torch.empty_like(query_labels)
    for i, cls in enumerate(classes):
        remapped[query_labels == cls] = i
    loss = F.cross_entropy(logits, remapped)
    acc = float((torch.argmax(logits, dim=1) == remapped).float().mean().item())
    return loss, acc


def boundary_loss(pred_offsets: torch.Tensor, gt_offsets: torch.Tensor, gaussian_sigma: float = 3.0) -> torch.Tensor:
    huber = F.smooth_l1_loss(pred_offsets, gt_offsets, beta=0.1, reduction="none")
    weight = torch.exp(-0.5 * (((pred_offsets - gt_offsets) * 128.0 / float(gaussian_sigma)) ** 2))
    return ((1.0 - weight) * huber).mean()


def recon_cn_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=recon.device)
    denom = mask.sum().clamp_min(1).to(dtype=recon.dtype) * recon.shape[-1]
    return (((recon - target) ** 2) * mask.unsqueeze(-1)).sum() / denom


def recon_graph_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(dtype=torch.bool, device=recon.device)
    denom = mask.sum().clamp_min(1).to(dtype=recon.dtype) * recon.shape[-1]
    return (((recon - target) ** 2) * mask.unsqueeze(-1)).sum() / denom


def leave_one_out_prototype_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, float]:
    """Prototype cross-entropy where each labeled example is held out of its class prototype."""
    embeddings = F.normalize(embeddings, p=2, dim=-1)
    labels = labels.to(dtype=torch.long, device=embeddings.device)
    classes = torch.unique(labels)
    logits_rows: list[torch.Tensor] = []
    targets: list[int] = []

    for row_i in range(embeddings.shape[0]):
        row_logits = []
        target = None
        usable = True
        for class_i, cls in enumerate(classes):
            mask = labels == cls
            if labels[row_i] == cls:
                mask = mask.clone()
                mask[row_i] = False
                target = class_i
            if not bool(mask.any()):
                usable = False
                break
            proto = F.normalize(embeddings[mask].mean(dim=0), p=2, dim=0)
            row_logits.append(torch.sum(embeddings[row_i] * proto) / float(temperature))
        if usable and target is not None:
            logits_rows.append(torch.stack(row_logits))
            targets.append(int(target))

    if not logits_rows:
        raise ValueError("No usable leave-one-out prototype examples; need at least two labels per class")

    logits = torch.stack(logits_rows, dim=0)
    target_tensor = torch.as_tensor(targets, dtype=torch.long, device=embeddings.device)
    loss = F.cross_entropy(logits, target_tensor)
    acc = float((torch.argmax(logits, dim=1) == target_tensor).float().mean().item())
    return loss, acc


def background_prototype_repulsion_loss(
    background_embeddings: torch.Tensor,
    labeled_embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.95,
    temperature: float = 0.05,
) -> torch.Tensor:
    """Weakly push unlabeled background chromosomes away from class prototypes."""
    if background_embeddings.numel() == 0:
        return torch.zeros((), dtype=labeled_embeddings.dtype, device=labeled_embeddings.device)
    labeled_embeddings = F.normalize(labeled_embeddings, p=2, dim=-1)
    background_embeddings = F.normalize(background_embeddings, p=2, dim=-1)
    labels = labels.to(dtype=torch.long, device=labeled_embeddings.device)
    prototypes = []
    for cls in torch.unique(labels):
        prototypes.append(F.normalize(labeled_embeddings[labels == cls].mean(dim=0), p=2, dim=0))
    if not prototypes:
        return torch.zeros((), dtype=labeled_embeddings.dtype, device=labeled_embeddings.device)
    prototypes_t = torch.stack(prototypes, dim=0)
    max_sim = torch.max(background_embeddings @ prototypes_t.T, dim=1).values
    return F.softplus((max_sim - float(margin)) / float(temperature)).mean()


def complex_sv_objectness_type_loss(
    objectness_logits: torch.Tensor,
    type_logits: torch.Tensor,
    objectness_targets: torch.Tensor,
    type_targets: torch.Tensor,
    type_mask: torch.Tensor,
    objectness_weights: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    type_loss_weight: float = 1.0,
    label_smoothing: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Train any-complex-SV objectness plus type logits for positive anchors.

    Objectness is supervised on labeled positives and weak unlabeled chromosome
    backgrounds. Type cross-entropy is evaluated only where ``type_mask`` is true.
    """
    objectness_targets = objectness_targets.to(dtype=objectness_logits.dtype, device=objectness_logits.device).view(-1)
    type_targets = type_targets.to(dtype=torch.long, device=objectness_logits.device).view(-1)
    type_mask = type_mask.to(dtype=torch.bool, device=objectness_logits.device).view(-1)
    if objectness_weights is None:
        objectness_weights = torch.ones_like(objectness_targets)
    else:
        objectness_weights = objectness_weights.to(dtype=objectness_logits.dtype, device=objectness_logits.device).view(-1)
    bce = F.binary_cross_entropy_with_logits(objectness_logits.view(-1), objectness_targets, reduction="none")
    objectness_loss = (bce * objectness_weights).sum() / objectness_weights.sum().clamp_min(1e-8)
    if type_mask.any():
        if class_weights is not None:
            class_weights = class_weights.to(dtype=type_logits.dtype, device=type_logits.device)
        type_loss = F.cross_entropy(
            type_logits[type_mask],
            type_targets[type_mask],
            weight=class_weights,
            label_smoothing=float(label_smoothing),
        )
    else:
        type_loss = torch.zeros((), dtype=objectness_loss.dtype, device=objectness_loss.device)
    total = objectness_loss + float(type_loss_weight) * type_loss
    return total, {"objectness_loss": objectness_loss, "type_loss": type_loss}

