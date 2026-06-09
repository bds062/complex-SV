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
        pred = row.get("predicted_class", row.get("sv_class", "unknown")) or "unknown"
        out = {
            "candidate_id": row.get("candidate_id", ""),
            "sample_id": row["sample_id"],
            "chrom": row["chrom"],
            "start_bp": int(row["start_bp"]),
            "end_bp": int(row["end_bp"]),
            "sv_class": pred,
            "confidence": float(row.get("prototype_confidence", 0.0)),
            "novelty_score": float(row.get("novelty_score", 0.0)),
            "evidence": row.get("evidence", ""),
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
        strict=args.strict,
    )
    embed_corpus.run(embed_args)

    embeddings, meta = _load_embedding_metadata(out_dir)
    if args.prototypes:
        cache = PrototypeCache.load(args.prototypes)
        distances = embed_corpus.write_distance_table(
            cache,
            embeddings,
            meta,
            out_dir / "prototype_distances.tsv",
        )
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
    parser.add_argument("--candidate_source", choices=["labels", "proposals", "all"], default="labels")
    parser.add_argument("--tau", type=float, default=0.5)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
