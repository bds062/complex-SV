"""Cluster unknown or low-confidence candidate embeddings for review."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _cosine_similarity_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.clip(norm, 1e-12, None)
    return x @ x.T


def cluster_outliers(
    embeddings_npz: str | Path,
    output_dir: str | Path,
    min_cluster_size: int = 10,
    min_samples: int = 5,
    confidence_threshold: float = 0.3,
) -> pd.DataFrame:
    data = np.load(embeddings_npz, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    meta = {k: data[k] for k in data.files if k != "embeddings"}
    df = pd.DataFrame(meta)
    if "sv_class" not in df:
        df["sv_class"] = ""
    if "confidence" not in df:
        df["confidence"] = 1.0
    if "novelty_score" not in df:
        df["novelty_score"] = 0.0

    mask = (df["sv_class"].astype(str) == "unknown") | (pd.to_numeric(df["confidence"], errors="coerce").fillna(1) < confidence_threshold)
    use_embeddings = embeddings[mask.to_numpy()]
    use_df = df.loc[mask].reset_index(drop=True)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(use_df) == 0:
        report = pd.DataFrame(columns=["cluster", "n_members", "mean_similarity", "mean_novelty_score", "flagged"])
        report.to_csv(out_dir / "clusters_report.tsv", sep="\t", index=False)
        return report

    try:
        import umap
        import hdbscan

        reduced = umap.UMAP(n_components=10, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(use_embeddings)
        labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean").fit_predict(reduced)
        xy = umap.UMAP(n_components=2, metric="cosine", n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(use_embeddings)
        np.savez(out_dir / "cluster_embedding_metrics.npz", embeddings=use_embeddings, umap10=reduced, xy=xy, cluster=labels)
    except ImportError:
        labels = np.full(len(use_df), -1, dtype=int)

    rows = []
    known_novelty = pd.to_numeric(df.loc[~mask, "novelty_score"], errors="coerce").dropna()
    novelty_cut = float(np.percentile(known_novelty, 90)) if len(known_novelty) else 0.0
    for cluster_id in sorted(set(labels.tolist())):
        if cluster_id < 0:
            continue
        idx = np.where(labels == cluster_id)[0]
        sim = _cosine_similarity_matrix(use_embeddings[idx])
        tri = sim[np.triu_indices_from(sim, k=1)]
        mean_sim = float(np.mean(tri)) if tri.size else 1.0
        mean_novelty = float(pd.to_numeric(use_df.loc[idx, "novelty_score"], errors="coerce").fillna(0).mean())
        flagged = bool(mean_sim > 0.6 and mean_novelty > novelty_cut)
        rows.append(
            {
                "cluster": int(cluster_id),
                "n_members": int(len(idx)),
                "mean_similarity": mean_sim,
                "mean_novelty_score": mean_novelty,
                "flagged": flagged,
                "representative_sample_ids": ",".join(use_df.loc[idx, "sample_id"].astype(str).head(10)),
            }
        )
        if flagged:
            arrays = {col: use_df.loc[idx, col].astype(str).to_numpy() for col in use_df.columns}
            np.savez(out_dir / f"cluster_{cluster_id}.npz", embeddings=use_embeddings[idx], **arrays)

    report = pd.DataFrame(rows)
    report.to_csv(out_dir / "clusters_report.tsv", sep="\t", index=False)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--min_cluster_size", type=int, default=10)
    parser.add_argument("--min_samples", type=int, default=5)
    args = parser.parse_args(argv)
    cluster_outliers(args.embeddings, args.output_dir, args.min_cluster_size, args.min_samples)


if __name__ == "__main__":
    main()
