"""Run complex-SV prototype-mode inference from a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from discovery import embed_corpus
from model.prototypes import PrototypeCache


def _load_embedding_metadata(output_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = np.load(output_dir / "embeddings.npz", allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    meta = pd.read_csv(output_dir / "candidate_embeddings.tsv", sep="\t").fillna("")
    return embeddings, meta


def _write_predictions(distances: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for row in distances.to_dict("records"):
        pred = row.get("predicted_class", row.get("sv_class", "none")) or "none"
        out = {
            "candidate_id": row.get("candidate_id", ""),
            "sample_id": row["sample_id"],
            "chrom": row["chrom"],
            "arm": row.get("arm", ""),
            "start_bp": int(row["start_bp"]),
            "end_bp": int(row["end_bp"]),
            "sv_class": pred,
            "confidence": float(row.get("prototype_confidence", 0.0)),
            "novelty_score": float(row.get("novelty_score", 0.0)),
            "evidence": row.get("evidence", ""),
            "candidate_scope": row.get("candidate_scope", row.get("label_scope", "")),
            "embedding_mode": row.get("embedding_mode", ""),
        }
        for key, value in row.items():
            if str(key).startswith("d_"):
                out[key] = value
        rows.append(out)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_args = SimpleNamespace(
        manifest=args.manifest,
        labels=args.labels,
        cn_checkpoint=args.cn_checkpoint,
        graph_checkpoint=args.graph_checkpoint,
        output_dir=str(out_dir),
        candidate_source=args.candidate_source,
        prototypes_name="prototypes.pt",
        tau=args.tau,
        embedding_normalization=args.embedding_normalization,
        sample_baseline_min_candidates=args.sample_baseline_min_candidates,
        strict=args.strict,
    )
    embed_corpus.run(embed_args)

    embeddings, meta = _load_embedding_metadata(out_dir)
    tau_for_outputs = float(args.tau)
    if args.prototypes:
        cache = PrototypeCache.load(args.prototypes)
        tau_for_outputs = float(cache.tau)
        distances = embed_corpus.write_distance_table(
            cache,
            embeddings,
            meta,
            out_dir / "prototype_distances.tsv",
        )
        loo_path = out_dir / "anchor_leave_one_out.tsv"
        leave_one_out = pd.read_csv(loo_path, sep="\t").fillna("") if loo_path.exists() else pd.DataFrame()
        embed_corpus.write_visualizations(embeddings, meta, distances, leave_one_out, out_dir, tau=tau_for_outputs)
    else:
        distances = pd.read_csv(out_dir / "prototype_distances.tsv", sep="\t").fillna("")

    _write_predictions(distances, out_dir / "predictions.tsv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--labels", required=False)
    parser.add_argument("--cn_checkpoint", required=True)
    parser.add_argument("--graph_checkpoint", required=False)
    parser.add_argument("--prototypes", required=False, help="Optional existing PrototypeCache .pt")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--candidate_source",
        choices=embed_corpus.CANDIDATE_SOURCE_CHOICES,
        default="labels",
        help="Region source to pass through to discovery.embed_corpus.",
    )
    parser.add_argument("--tau", type=float, default=embed_corpus.DEFAULT_TAU)
    parser.add_argument(
        "--embedding_normalization",
        choices=embed_corpus.EMBEDDING_NORMALIZATION_CHOICES,
        default="none",
        help="Optional post-encoder normalization before prototype scoring.",
    )
    parser.add_argument(
        "--sample_baseline_min_candidates",
        type=int,
        default=3,
        help="Minimum same-sample background candidates for sample_residual baselines before falling back.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
