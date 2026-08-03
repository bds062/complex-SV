
"""
Stage 2a copy-number Transformer encoder and masked autoencoder.

This module contains model definitions only. It is imported by pretrain_cn.py,
fusion-time SVModel components, and post-pretraining visualization scripts.

Input convention
----------------
Copy-number tensors use shape [B, n_bins, n_cn_channels]. The channel order is
provided by genomic_features.cn_resampler.CN_CHANNELS.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

try:
    from config import CNEncoderConfig
except ImportError:  # Allows direct local imports during early development.
    CNEncoderConfig = Any  # type: ignore[misc,assignment]

try:
    from genomic_features.cn_resampler import CN_CHANNELS
except ImportError:  # Keeps this module importable during isolated prototyping.
    CN_CHANNELS = [
        "cn_total",
        "cn_hp1",
        "cn_hp2",
        "log_coverage_total",
        "coverage_hp1_fraction",
        "coverage_hp2_fraction",
        "confidence_hp1",
        "confidence_hp2",
        "loh",
        "allele_imbalance",
        "breakpoint_count",
    ]

CN_INPUT_DIM = len(CN_CHANNELS)
D_MODEL_CN = 256


class CNTransformerEncoder(nn.Module):
    """
    Transformer encoder for resampled copy-number profiles.

    forward(x) expects x of shape [B, n_bins, CN_INPUT_DIM] and returns:
        cls_embedding: [B, d_model]
        bin_embeddings: [B, n_bins, d_model]
    """

    def __init__(self, cfg: CNEncoderConfig | Any):
        super().__init__()
        self.d_model = int(getattr(cfg, "d_model", D_MODEL_CN))
        self.n_bins = max(
            int(getattr(cfg, "n_bins_arm", 0)),
            int(getattr(cfg, "n_bins_region", 0)),
            int(getattr(cfg, "seq_len", 256)),
        )
        self.dropout_p = float(getattr(cfg, "dropout", 0.1))
        ff_dim = int(getattr(cfg, "ff_dim", getattr(cfg, "d_ff", 1024)))
        n_heads = int(getattr(cfg, "n_heads", 8))
        n_layers = int(getattr(cfg, "n_layers", 6))

        if self.d_model % n_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({n_heads})")

        self.input_proj = nn.Linear(CN_INPUT_DIM, self.d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_bins + 1, self.d_model) * 0.02)
        self.dropout = nn.Dropout(self.dropout_p)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=self.dropout_p,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[-1] != CN_INPUT_DIM:
            raise ValueError(f"CNTransformerEncoder expects [B, L, {CN_INPUT_DIM}], got {tuple(x.shape)}")

        batch_size, seq_len, _ = x.shape
        if seq_len + 1 > self.pos_embed.shape[1]:
            raise ValueError(f"Input length {seq_len} exceeds configured maximum {self.pos_embed.shape[1] - 1}")

        tokens = self.input_proj(x)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : seq_len + 1]
        tokens = self.dropout(tokens)
        encoded = self.final_norm(self.encoder(tokens))
        return encoded[:, 0], encoded[:, 1:]


class CNMaskedAutoencoder(nn.Module):
    """
    Masked copy-number autoencoder wrapping CNTransformerEncoder.

    forward(x, mask) expects:
        x: [B, n_bins, CN_INPUT_DIM]
        mask: [B, n_bins] bool, True marks masked positions

    Returns:
        recon: [B, n_bins, CN_INPUT_DIM]
        cls_emb: [B, d_model]
        bin_embs: [B, n_bins, d_model]
    """

    def __init__(self, cfg: CNEncoderConfig | Any):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(getattr(cfg, "d_model", D_MODEL_CN))
        self.encoder = CNTransformerEncoder(cfg)
        self.mask_token = nn.Parameter(torch.randn(1, 1, CN_INPUT_DIM) * 0.02)
        self.recon_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, CN_INPUT_DIM),
        )
        self.cls_recon_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, CN_INPUT_DIM),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mask.ndim != 2:
            raise ValueError(f"mask must be [B, L], got {tuple(mask.shape)}")
        if x.shape[:2] != mask.shape:
            raise ValueError(f"x and mask sequence dimensions must match: x={tuple(x.shape)}, mask={tuple(mask.shape)}")
        if x.shape[-1] != CN_INPUT_DIM:
            raise ValueError(f"x must have {CN_INPUT_DIM} CN channels, got {x.shape[-1]}")

        mask = mask.to(dtype=torch.bool, device=x.device)
        x_masked = torch.where(
            mask.unsqueeze(-1),
            self.mask_token.to(dtype=x.dtype, device=x.device).expand_as(x),
            x,
        )
        cls_emb, bin_embs = self.encoder(x_masked)
        recon = self.recon_head(bin_embs)
        return recon, cls_emb, bin_embs

    def reconstruct_from_cls(self, cls_emb: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Return a coarse full-sequence reconstruction from CLS only."""
        token = self.cls_recon_head(cls_emb).unsqueeze(1)
        return token.expand(-1, seq_len, -1)
