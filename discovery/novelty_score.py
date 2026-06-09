"""Dual reconstruction-error novelty scores for candidate regions."""

from __future__ import annotations

import torch


def compute_cn_recon_error(model, cn_bins: torch.Tensor) -> float:
    """Return unmasked CN reconstruction MSE for a candidate tensor."""
    model.eval()
    with torch.no_grad():
        mask = torch.zeros(cn_bins.shape[:2], dtype=torch.bool, device=cn_bins.device)
        recon, _cls, _bins = model.cn_encoder(cn_bins, mask) if hasattr(model, "cn_encoder") else model(cn_bins, mask)
        return float(torch.mean((recon - cn_bins) ** 2).item())


def compute_graph_recon_error(model, graph_data, sv_node_indices: list[int]) -> float:
    """Return unmasked graph node reconstruction MSE over candidate SV nodes."""
    graph_model = model.graph_encoder if hasattr(model, "graph_encoder") else model
    if graph_data is None or not sv_node_indices:
        return 0.0
    graph_model.eval()
    with torch.no_grad():
        x = graph_data["sv"].x
        mask = torch.zeros(x.shape[0], dtype=torch.bool, device=x.device)
        recon, _node_h = graph_model(graph_data, mask)
        idx = torch.as_tensor(sv_node_indices, dtype=torch.long, device=x.device)
        return float(torch.mean((recon[idx] - x[idx]) ** 2).item())


def combined_novelty_score(cn_error: float, graph_error: float, cn_p95: float, graph_p95: float) -> float:
    """Mean of percentile-normalized CN and graph reconstruction errors."""
    cn_norm = min(float(cn_error) / max(float(cn_p95), 1e-12), 2.0)
    graph_norm = min(float(graph_error) / max(float(graph_p95), 1e-12), 2.0)
    return float(max(0.0, min((cn_norm + graph_norm) / 2.0, 1.0)))
