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


class MetricProjection(nn.Module):
    """Small trainable projection for frozen prototype-mode embeddings."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int = 64,
        hidden_dim: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        in_dim = int(in_dim)
        embed_dim = int(embed_dim)
        hidden_dim = int(hidden_dim)
        if hidden_dim > 0:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:
            self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(embedding), p=2, dim=-1)


class ComplexSVClassifierHead(nn.Module):
    """Objectness plus complex-SV type classifier for frozen embeddings."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        in_dim = int(in_dim)
        hidden_dim = int(hidden_dim)
        num_classes = int(num_classes)
        if hidden_dim > 0:
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(float(dropout)),
            )
            head_dim = hidden_dim
        else:
            self.backbone = nn.Identity()
            head_dim = in_dim
        self.objectness = nn.Linear(head_dim, 1)
        self.type_classifier = nn.Linear(head_dim, num_classes)

    def forward(self, embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(embedding)
        objectness_logit = self.objectness(h).squeeze(-1)
        type_logits = self.type_classifier(h)
        return objectness_logit, type_logits

