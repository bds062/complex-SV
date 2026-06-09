"""Few-shot prototype cache for complex SV class lookup."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from utils import cosine_distance, l2_normalize, torch_load_checkpoint
except ImportError:  # pragma: no cover
    from ..utils import cosine_distance, l2_normalize, torch_load_checkpoint  # type: ignore


class PrototypeCache:
    """
    Store class prototypes and classify L2-normalized query embeddings.

    The cache is intentionally not an nn.Module. Updating prototypes only
    requires embedding new support examples and saving this object.
    """

    def __init__(self, embed_dim: int = 128, tau: float = 0.5):
        self.embed_dim = int(embed_dim)
        self.tau = float(tau)
        self.prototypes: dict[str, torch.Tensor] = {}

    def add_class(self, class_name: str, embeddings: torch.Tensor) -> None:
        embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be [K, D], got {tuple(embeddings.shape)}")
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot build a prototype from zero embeddings")
        if not self.prototypes and self.embed_dim != int(embeddings.shape[1]):
            self.embed_dim = int(embeddings.shape[1])
        if embeddings.shape[1] != self.embed_dim:
            raise ValueError(f"Expected embedding dim {self.embed_dim}, got {embeddings.shape[1]}")
        proto = l2_normalize(embeddings.mean(dim=0), dim=0).detach().cpu()
        self.prototypes[str(class_name)] = proto

    def remove_class(self, class_name: str) -> None:
        self.prototypes.pop(str(class_name), None)

    def class_names(self) -> list[str]:
        return sorted(self.prototypes)

    def n_classes(self) -> int:
        return len(self.prototypes)

    def classify(self, query: torch.Tensor) -> tuple[str, float, dict[str, float]]:
        if not self.prototypes:
            return "unknown", 0.0, {}

        query = torch.as_tensor(query, dtype=torch.float32)
        if query.ndim == 2 and query.shape[0] == 1:
            query = query.squeeze(0)
        if query.ndim != 1:
            raise ValueError(f"query must be [D] or [1, D], got {tuple(query.shape)}")
        if query.shape[0] != self.embed_dim:
            raise ValueError(f"Expected query dim {self.embed_dim}, got {query.shape[0]}")

        query = F.normalize(query, p=2, dim=0)
        labels = self.class_names()
        protos = torch.stack([self.prototypes[label] for label in labels], dim=0)
        distances = cosine_distance(query.unsqueeze(0), protos).squeeze(0)
        distance_dict = {label: float(distances[i].item()) for i, label in enumerate(labels)}

        best_idx = int(torch.argmin(distances).item())
        best_label = labels[best_idx]
        d_min = float(distances[best_idx].item())
        confidence = max(0.0, (self.tau - d_min) / max(self.tau, 1e-12))
        if d_min >= self.tau:
            return "unknown", confidence, distance_dict
        return best_label, confidence, distance_dict

    def classify_batch(self, queries: torch.Tensor) -> list[tuple[str, float, dict[str, float]]]:
        queries = torch.as_tensor(queries, dtype=torch.float32)
        if queries.ndim != 2:
            raise ValueError(f"queries must be [B, D], got {tuple(queries.shape)}")
        return [self.classify(row) for row in queries]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embed_dim": self.embed_dim,
                "tau": self.tau,
                "prototypes": {k: v.detach().cpu() for k, v in self.prototypes.items()},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "PrototypeCache":
        obj = torch_load_checkpoint(path, map_location="cpu")
        cache = cls(embed_dim=int(obj.get("embed_dim", 128)), tau=float(obj.get("tau", 0.5)))
        raw = obj.get("prototypes", {})
        cache.prototypes = {
            str(label): F.normalize(torch.as_tensor(vec, dtype=torch.float32), p=2, dim=0)
            for label, vec in raw.items()
        }
        return cache
