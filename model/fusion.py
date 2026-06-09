"""Stage 4 multimodal fusion Transformer."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from config import FusionConfig
except ImportError:  # pragma: no cover
    FusionConfig = Any  # type: ignore


class FusionTransformer(nn.Module):
    """
    Fuse CN bin embeddings with graph and segment context tokens.

    Inputs:
        cn_bin_embs: [B, 128, cn_embed_dim]
        graph_regional_emb: [B, graph_embed_dim]
        graph_global_emb: [B, graph_global_dim]
        segment_stats: [B, segment_stats_dim]

    Returns:
        [B, d_model]
    """

    def __init__(self, cfg: FusionConfig | Any):
        super().__init__()
        self.d_model = int(getattr(cfg, "d_model", 256))
        n_heads = int(getattr(cfg, "n_heads", 8))
        n_layers = int(getattr(cfg, "n_layers", 3))
        dropout = float(getattr(cfg, "dropout", 0.2))
        cn_embed_dim = int(getattr(cfg, "cn_embed_dim", 256))
        graph_embed_dim = int(getattr(cfg, "graph_embed_dim", 64))
        graph_global_dim = int(getattr(cfg, "graph_global_dim", 64))
        segment_stats_dim = int(getattr(cfg, "segment_stats_dim", 18))

        if self.d_model % n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})")

        self.cn_proj = nn.Linear(cn_embed_dim, self.d_model)
        self.graph_regional_proj = nn.Linear(graph_embed_dim, self.d_model)
        self.graph_global_proj = nn.Linear(graph_global_dim, self.d_model)
        self.segment_proj = nn.Linear(segment_stats_dim, self.d_model)
        self.type_embed = nn.Parameter(torch.randn(1, 4, self.d_model) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=self.d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        cn_bin_embs: torch.Tensor,
        graph_regional_emb: torch.Tensor,
        graph_global_emb: torch.Tensor,
        segment_stats: torch.Tensor,
    ) -> torch.Tensor:
        if cn_bin_embs.ndim != 3:
            raise ValueError(f"cn_bin_embs must be [B, L, D], got {tuple(cn_bin_embs.shape)}")

        graph_token = self.graph_regional_proj(graph_regional_emb).unsqueeze(1)
        global_token = self.graph_global_proj(graph_global_emb).unsqueeze(1)
        segment_token = self.segment_proj(segment_stats).unsqueeze(1)
        cn_tokens = self.cn_proj(cn_bin_embs)

        context = torch.cat([graph_token, global_token, segment_token], dim=1)
        context = context + self.type_embed[:, :3]
        cn_tokens = cn_tokens + self.type_embed[:, 3:4]
        tokens = torch.cat([context, cn_tokens], dim=1)
        encoded = self.norm(self.encoder(tokens))
        return encoded.mean(dim=1)
