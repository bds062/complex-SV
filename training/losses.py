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
