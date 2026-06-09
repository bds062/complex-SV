"""Prediction heads for fused complex-SV embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryHead(nn.Module):
    """Regress fractional start/end offsets within a candidate interval."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Sigmoid(),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class ConfidenceHead(nn.Module):
    """Binary confidence that a candidate is any known complex SV."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embed_dim, 1), nn.Sigmoid())

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class EmbeddingProjection(nn.Module):
    """Project fused features to the normalized pre-classification embedding."""

    def __init__(self, in_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(fused), p=2, dim=-1)
