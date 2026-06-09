"""N-way K-shot episode sampler."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator

import torch


class EpisodeSampler:
    def __init__(
        self,
        labeled_dataset: list[dict],
        n_way: int = 2,
        k_shot: int = 5,
        n_query: int = 10,
        seed: int = 42,
    ):
        self.labeled_dataset = list(labeled_dataset)
        self.n_way = int(n_way)
        self.k_shot = int(k_shot)
        self.n_query = int(n_query)
        self.rng = random.Random(seed)
        self.by_class: dict[str, list[dict]] = defaultdict(list)
        for row in self.labeled_dataset:
            self.by_class[str(row["class_name"])].append(row)

    def sample_episode(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        eligible = [cls for cls, rows in self.by_class.items() if len(rows) >= self.k_shot + self.n_query]
        if len(eligible) < self.n_way:
            raise ValueError(
                f"Need at least {self.n_way} classes with {self.k_shot + self.n_query} examples each; "
                f"found {len(eligible)}"
            )
        classes = self.rng.sample(eligible, self.n_way)
        support, support_labels, query, query_labels = [], [], [], []
        for label_idx, cls in enumerate(classes):
            rows = self.rng.sample(self.by_class[cls], self.k_shot + self.n_query)
            for row in rows[: self.k_shot]:
                support.append(torch.as_tensor(row["embedding"], dtype=torch.float32))
                support_labels.append(label_idx)
            for row in rows[self.k_shot :]:
                query.append(torch.as_tensor(row["embedding"], dtype=torch.float32))
                query_labels.append(label_idx)
        return (
            torch.stack(support, dim=0),
            torch.as_tensor(support_labels, dtype=torch.long),
            torch.stack(query, dim=0),
            torch.as_tensor(query_labels, dtype=torch.long),
        )

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        while True:
            yield self.sample_episode()
