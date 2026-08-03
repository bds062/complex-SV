"""
Stage 2b heterogeneous graph Transformer encoder and graph MAE.

This module contains model definitions only.  Graph construction and parsing
live in complex_sv.genomic_features.*.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import TransformerConv
except ImportError as exc:  # pragma: no cover - dependency message only
    raise ImportError(
        "PyTorch Geometric is required for complex_sv.pretrain.graph_encoder. "
        "Install torch-geometric and matching PyG wheels for your torch build."
    ) from exc

try:
    from config import GraphEncoderConfig
    from genomic_features.graph_builder import EDGE_ATTR_DIM, EDGE_TYPES
    from genomic_features.severus_parser import N_FEAT
except ImportError:  # Allows direct local imports during early development.
    GraphEncoderConfig = Any  # type: ignore[misc,assignment]
    EDGE_PROXIMITY = ("sv", "proximal_to", "sv")
    EDGE_MATE = ("sv", "mate_of", "sv")
    EDGE_INTERCHROM = ("sv", "interchrom_mate_of", "sv")
    EDGE_CLUSTER = ("sv", "cluster_linked", "sv")
    EDGE_PHASE = ("sv", "phase_linked", "sv")
    EDGE_TYPES = (EDGE_PROXIMITY, EDGE_MATE, EDGE_INTERCHROM, EDGE_CLUSTER, EDGE_PHASE)
    EDGE_ATTR_DIM = 3
    N_FEAT = 46


D_MODEL_GRAPH = 128
EMBED_DIM = 64
GRAPH_AUX_TARGET_DIM = 6


def _edge_key(edge_type: tuple[str, str, str]) -> str:
    return "__".join(edge_type)


class HeteroGraphTransformerEncoder(nn.Module):
    """
    Stacked heterogeneous graph Transformer encoder.

    Each layer applies one TransformerConv per edge type and combines their
    messages with learned per-edge-type gates before residual + LayerNorm.

    forward(x_dict, edge_index_dict, edge_attr_dict) returns node embeddings
    of shape [N_sv, d_model].
    """

    def __init__(self, cfg: GraphEncoderConfig | Any):
        super().__init__()
        self.d_model = int(getattr(cfg, "d_model", D_MODEL_GRAPH))
        self.n_heads = int(getattr(cfg, "n_heads", 8))
        self.n_layers = int(getattr(cfg, "n_layers", 4))
        self.dropout_p = float(getattr(cfg, "dropout", 0.1))
        self.edge_attr_dim = int(getattr(cfg, "edge_attr_dim", EDGE_ATTR_DIM))

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

        self.input_proj = nn.Linear(N_FEAT, self.d_model)
        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            layer = nn.ModuleDict()
            for edge_type in EDGE_TYPES:
                layer[_edge_key(edge_type)] = TransformerConv(
                    in_channels=self.d_model,
                    out_channels=self.d_model // self.n_heads,
                    heads=self.n_heads,
                    dropout=self.dropout_p,
                    edge_dim=self.edge_attr_dim,
                    concat=True,
                )
            self.layers.append(layer)

        self.norms = nn.ModuleList([nn.LayerNorm(self.d_model) for _ in range(self.n_layers)])
        self.drop = nn.Dropout(self.dropout_p)
        self.edge_gates = nn.Parameter(torch.ones(self.n_layers, len(EDGE_TYPES)))

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> torch.Tensor:
        if "sv" not in x_dict:
            raise KeyError("x_dict must contain node type 'sv'")

        h = self.input_proj(x_dict["sv"])

        for layer_idx, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            gated_messages: list[torch.Tensor] = []
            gates = torch.sigmoid(self.edge_gates[layer_idx])

            for edge_idx, edge_type in enumerate(EDGE_TYPES):
                edge_index = edge_index_dict.get(edge_type)
                if edge_index is None:
                    edge_index = torch.zeros((2, 0), dtype=torch.long, device=h.device)
                else:
                    edge_index = edge_index.to(device=h.device)
                edge_attr = edge_attr_dict.get(edge_type)
                if edge_attr is None:
                    edge_attr = torch.zeros(
                        edge_index.shape[1],
                        self.edge_attr_dim,
                        dtype=h.dtype,
                        device=h.device,
                    )
                    if edge_attr.numel() > 0:
                        edge_attr[:, 0] = 1.0
                else:
                    edge_attr = edge_attr.to(device=h.device, dtype=h.dtype)
                    if edge_attr.ndim == 1:
                        edge_attr = edge_attr.unsqueeze(-1)
                    if edge_attr.shape[1] < self.edge_attr_dim:
                        pad = torch.zeros(
                            edge_attr.shape[0],
                            self.edge_attr_dim - edge_attr.shape[1],
                            dtype=edge_attr.dtype,
                            device=edge_attr.device,
                        )
                        edge_attr = torch.cat([edge_attr, pad], dim=1)
                    elif edge_attr.shape[1] > self.edge_attr_dim:
                        edge_attr = edge_attr[:, : self.edge_attr_dim]

                conv = layer[_edge_key(edge_type)]
                msg = conv(h, edge_index, edge_attr=edge_attr)
                gated_messages.append(gates[edge_idx] * msg)

            msg_sum = torch.stack(gated_messages, dim=0).sum(dim=0)
            h = norm(h + self.drop(msg_sum))

        return h


class AttentionReadout(nn.Module):
    """
    Attention-weighted regional readout over variable-length SV node subsets.

    forward(node_embeds) accepts [K, d_model] and returns [embed_dim].
    forward_batched(node_embeds, mask) accepts [B, K, d_model] plus a bool mask
    where True means valid node and returns [B, embed_dim].
    """

    def __init__(self, d_model: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attn_score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )
        self.global_proj = nn.Linear(d_model, d_model // 2)
        self.out_proj = nn.Sequential(
            nn.Linear(d_model + d_model // 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, node_embeds: torch.Tensor) -> torch.Tensor:
        if node_embeds.ndim != 2:
            raise ValueError(f"node_embeds must be [K, D], got {tuple(node_embeds.shape)}")
        if node_embeds.shape[0] == 0:
            raise ValueError("AttentionReadout cannot pool an empty node set")
        scores = self.attn_score(node_embeds)
        weights = torch.softmax(scores, dim=0)
        attended = (weights * node_embeds).sum(dim=0)
        global_ctx = self.global_proj(node_embeds.mean(dim=0))
        return self.out_proj(torch.cat([attended, global_ctx], dim=-1))

    def forward_with_weights(self, node_embeds: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Like forward() but also returns per-node attention weights [K]."""
        if node_embeds.ndim != 2:
            raise ValueError(f"node_embeds must be [K, D], got {tuple(node_embeds.shape)}")
        if node_embeds.shape[0] == 0:
            raise ValueError("AttentionReadout cannot pool an empty node set")
        scores = self.attn_score(node_embeds)
        weights = torch.softmax(scores, dim=0)
        attended = (weights * node_embeds).sum(dim=0)
        global_ctx = self.global_proj(node_embeds.mean(dim=0))
        return self.out_proj(torch.cat([attended, global_ctx], dim=-1)), weights.squeeze(-1)

    def forward_batched(self, node_embeds: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if node_embeds.ndim != 3:
            raise ValueError(f"node_embeds must be [B, K, D], got {tuple(node_embeds.shape)}")
        if valid_mask.shape != node_embeds.shape[:2]:
            raise ValueError("valid_mask must have shape [B, K]")
        valid_mask = valid_mask.to(dtype=torch.bool, device=node_embeds.device)
        scores = self.attn_score(node_embeds).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        weights = torch.where(valid_mask.unsqueeze(-1), weights, torch.zeros_like(weights))
        attended = (weights * node_embeds).sum(dim=1)
        denom = valid_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=node_embeds.dtype)
        mean_ctx = (node_embeds * valid_mask.unsqueeze(-1)).sum(dim=1) / denom
        global_ctx = self.global_proj(mean_ctx)
        return self.out_proj(torch.cat([attended, global_ctx], dim=-1))


class SVGraphMAE(nn.Module):
    """
    Heterogeneous graph masked autoencoder.

    Methods:
        encode(data) -> [N, d_model]
        regional_embed(node_h, node_indices) -> [embed_dim]
        global_embed(node_h) -> [embed_dim]
        forward(data, mask) -> (recon [N, N_FEAT], node_h [N, d_model])
    """

    def __init__(self, cfg: GraphEncoderConfig | Any):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(getattr(cfg, "d_model", D_MODEL_GRAPH))
        self.embed_dim = int(getattr(cfg, "embed_dim", EMBED_DIM))
        dropout = float(getattr(cfg, "dropout", 0.1))

        self.encoder = HeteroGraphTransformerEncoder(cfg)
        self.recon_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, N_FEAT),
        )
        self.readout = AttentionReadout(self.d_model, self.embed_dim, dropout)
        self.global_proj = nn.Sequential(
            nn.Linear(self.d_model, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )
        self.aux_head = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.GELU(),
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, GRAPH_AUX_TARGET_DIM),
        )

    def _edge_attr_dict(self, data: HeteroData) -> dict[tuple[str, str, str], torch.Tensor]:
        return {
            edge_type: data[edge_type].edge_attr
            for edge_type in EDGE_TYPES
            if edge_type in data.edge_types and hasattr(data[edge_type], "edge_attr")
        }

    def encode(self, data: HeteroData) -> torch.Tensor:
        return self.encoder(data.x_dict, data.edge_index_dict, self._edge_attr_dict(data))

    def forward(self, data: HeteroData, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(dtype=torch.bool, device=data["sv"].x.device)
        x_orig = data["sv"].x.clone()
        data["sv"].x = x_orig.clone()
        data["sv"].x[mask] = 0.0
        node_h = self.encode(data)
        recon = self.recon_head(node_h)
        data["sv"].x = x_orig
        return recon, node_h

    def graph_aux(self, node_h: torch.Tensor, batch: torch.Tensor | None = None) -> torch.Tensor:
        if batch is None:
            pooled = self.readout(node_h).unsqueeze(0)
            return self.aux_head(pooled)

        batch = batch.to(device=node_h.device, dtype=torch.long)
        pooled: list[torch.Tensor] = []
        n_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        for graph_idx in range(n_graphs):
            idx = torch.nonzero(batch == graph_idx, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            pooled.append(self.readout(node_h[idx]))
        if not pooled:
            raise ValueError("graph_aux requires at least one graph with SV nodes")
        return self.aux_head(torch.stack(pooled, dim=0))

    def regional_embed(self, node_h: torch.Tensor, node_indices: Iterable[int]) -> torch.Tensor:
        idx = torch.as_tensor(list(node_indices), dtype=torch.long, device=node_h.device)
        if idx.numel() == 0:
            return self.global_embed(node_h)
        return self.readout(node_h[idx])

    def regional_embed_with_weights(
        self, node_h: torch.Tensor, node_indices: Iterable[int]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Like regional_embed() but also returns per-node attention weights [K], or None if no nodes."""
        idx_list = list(node_indices)
        if not idx_list:
            return self.global_embed(node_h), None
        idx = torch.as_tensor(idx_list, dtype=torch.long, device=node_h.device)
        embedding, weights = self.readout.forward_with_weights(node_h[idx])
        return embedding, weights

    def global_embed(self, node_h: torch.Tensor) -> torch.Tensor:
        if node_h.ndim != 2 or node_h.shape[0] == 0:
            raise ValueError("node_h must be a non-empty [N, D] tensor")
        return F.normalize(self.global_proj(node_h.mean(dim=0)), p=2, dim=-1)
