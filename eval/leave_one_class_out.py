"""Leave-one-class-out evaluation for prototype embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from model.prototypes import PrototypeCache


def run_loco_eval(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    tau: float,
    output_dir: str | Path,
) -> pd.DataFrame:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    rows = []
    for held in sorted(np.unique(labels)):
        cache = PrototypeCache(embed_dim=embeddings.shape[1], tau=tau)
        for class_idx, name in enumerate(class_names):
            if class_idx == held:
                continue
            idx = np.where(labels == class_idx)[0]
            if len(idx):
                cache.add_class(name, embeddings[idx])
        held_idx = np.where(labels == held)[0]
        known_idx = np.where(labels != held)[0]
        held_preds = [cache.classify(embeddings[i])[0] for i in held_idx]
        known_preds = [cache.classify(embeddings[i])[0] for i in known_idx]
        rows.append(
            {
                "held_out_class": class_names[int(held)] if int(held) < len(class_names) else str(held),
                "unknown_recall": float(np.mean([p == "unknown" for p in held_preds])) if held_preds else 0.0,
                "false_unknown_rate": float(np.mean([p == "unknown" for p in known_preds])) if known_preds else 0.0,
                "n_held_out": int(len(held_idx)),
                "n_known": int(len(known_idx)),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "leave_one_class_out.tsv", sep="\t", index=False)
    return df
