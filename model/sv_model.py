"""Top-level trainable complex-SV model assembly."""

from __future__ import annotations

from collections.abc import Iterator

import torch
import torch.nn as nn

from .fusion import FusionTransformer
from .heads import BoundaryHead, ConfidenceHead, EmbeddingProjection


class SVModel(nn.Module):
    """Assemble pretrained encoders, fusion, projection, and prediction heads."""

    def __init__(
        self,
        cn_encoder: nn.Module,
        graph_encoder: nn.Module,
        fusion: FusionTransformer,
        embedding_proj: EmbeddingProjection,
        boundary_head: BoundaryHead,
        confidence_head: ConfidenceHead,
    ):
        super().__init__()
        self.cn_encoder = cn_encoder
        self.graph_encoder = graph_encoder
        self.fusion = fusion
        self.embedding_proj = embedding_proj
        self.boundary_head = boundary_head
        self.confidence_head = confidence_head

    def forward(
        self,
        cn_bins: torch.Tensor,
        graph_data,
        segment_stats: torch.Tensor,
        sv_node_indices: list[int] | list[list[int]],
        mask_cn: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if cn_bins.ndim != 3:
            raise ValueError(f"cn_bins must be [B, L, C], got {tuple(cn_bins.shape)}")

        batch_size = cn_bins.shape[0]
        if mask_cn is None:
            mask_cn = torch.zeros(cn_bins.shape[:2], dtype=torch.bool, device=cn_bins.device)
        cn_recon, cn_cls_emb, cn_bin_embs = self.cn_encoder(cn_bins, mask_cn)

        if graph_data is None:
            graph_embed_dim = self.fusion.graph_regional_proj.in_features
            graph_global_dim = self.fusion.graph_global_proj.in_features
            graph_regional = torch.zeros(batch_size, graph_embed_dim, dtype=cn_bins.dtype, device=cn_bins.device)
            graph_global = torch.zeros(batch_size, graph_global_dim, dtype=cn_bins.dtype, device=cn_bins.device)
            graph_recon = None
        else:
            graph_data = graph_data.to(cn_bins.device)
            graph_mask = torch.zeros(graph_data["sv"].x.shape[0], dtype=torch.bool, device=cn_bins.device)
            graph_recon, node_h = self.graph_encoder(graph_data, graph_mask)
            graph_global_one = self.graph_encoder.global_embed(node_h)
            if batch_size == 1 and (not sv_node_indices or isinstance(sv_node_indices[0], int)):
                node_lists = [sv_node_indices]  # type: ignore[list-item]
            else:
                node_lists = sv_node_indices  # type: ignore[assignment]
            regional = [self.graph_encoder.regional_embed(node_h, idxs) for idxs in node_lists]
            graph_regional = torch.stack(regional, dim=0)
            graph_global = graph_global_one.unsqueeze(0).expand(batch_size, -1)

        fused = self.fusion(cn_bin_embs, graph_regional, graph_global, segment_stats)
        embedding = self.embedding_proj(fused)
        return {
            "embedding": embedding,
            "boundary": self.boundary_head(embedding),
            "confidence": self.confidence_head(embedding),
            "cn_recon": cn_recon if mask_cn is not None else None,
            "graph_recon": graph_recon,
            "graph_regional_emb": graph_regional,
            "cn_cls_emb": cn_cls_emb,
        }

    def freeze_encoders(self) -> None:
        for param in self.encoder_parameters():
            param.requires_grad = False

    def unfreeze_encoders(self) -> None:
        for param in self.encoder_parameters():
            param.requires_grad = True

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.cn_encoder.parameters()
        yield from self.graph_encoder.parameters()

    def non_encoder_parameters(self) -> Iterator[nn.Parameter]:
        for module in [self.fusion, self.embedding_proj, self.boundary_head, self.confidence_head]:
            yield from module.parameters()
