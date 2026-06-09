"""Positive-unlabeled risk helpers."""

from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F


def pu_risk(model_scores: torch.Tensor, labels: torch.Tensor, prior: float = 0.15) -> torch.Tensor:
    """Non-negative PU risk using binary logistic losses."""
    scores = model_scores.view(-1)
    labels = labels.to(dtype=torch.float32, device=scores.device).view(-1)
    pos = labels == 1
    unl = labels == 0
    if pos.sum() == 0:
        return F.binary_cross_entropy(scores, labels)
    positive_risk = prior * F.binary_cross_entropy(scores[pos], torch.ones_like(scores[pos]))
    negative_on_pos = prior * F.binary_cross_entropy(scores[pos], torch.zeros_like(scores[pos]))
    negative_on_unl = (
        F.binary_cross_entropy(scores[unl], torch.zeros_like(scores[unl])) if unl.any() else torch.zeros((), device=scores.device)
    )
    negative_risk = torch.clamp(negative_on_unl - negative_on_pos, min=0.0)
    return positive_risk + negative_risk


def assign_pu_weights(df: pd.DataFrame, shatterseek_positive_ids: set[str], prior: float = 0.15) -> pd.DataFrame:
    out = df.copy()
    ids = out.get("candidate_id", out.index.to_series()).astype(str)
    out["pu_weight"] = ids.map(lambda x: 1.0 if x in shatterseek_positive_ids else float(prior))
    return out
