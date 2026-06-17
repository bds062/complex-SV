"""Apply a trained few-shot prototypical classifier to saved complex-SV embeddings."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discovery import embed_corpus
from model.heads import MetricProjection
from training.train_classifier_head import enforce_candidate_resolution, load_embedding_table
from training.train_fewshot_classifier_head import (
    annotate_predictions,
    fewshot_predictions_to_distance_table,
    predict_with_prototypes,
    _parameter_count,
    _write_prediction_view,
)
from utils import torch_load_checkpoint

log = logging.getLogger(__name__)


def _load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[MetricProjection, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
    missing = [
        key
        for key in [
            "model_state_dict",
            "input_dim",
            "projection_dim",
            "hidden_dim",
            "dropout",
            "class_names",
            "prototype_vectors",
            "prototype_counts",
        ]
        if key not in checkpoint
    ]
    if missing:
        raise KeyError(f"Few-shot checkpoint missing required key(s): {missing}; checkpoint={checkpoint_path}")
    model = MetricProjection(
        in_dim=int(checkpoint["input_dim"]),
        embed_dim=int(checkpoint["projection_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, checkpoint = _load_model(args.checkpoint, device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    prototypes = checkpoint["prototype_vectors"].detach().cpu().numpy()
    prototype_counts = checkpoint["prototype_counts"].detach().cpu().numpy()
    distance_tau = float(args.distance_tau) if args.distance_tau is not None else float(checkpoint.get("selected_distance_tau", 0.5))
    objectness_tau = float(args.objectness_tau) if args.objectness_tau is not None else float(checkpoint.get("objectness_tau", 0.5))
    objectness_scale = float(args.objectness_scale) if args.objectness_scale is not None else float(checkpoint.get("objectness_scale", 12.0))
    temperature = float(args.temperature) if args.temperature is not None else float(checkpoint.get("temperature", 0.1))

    embeddings, metadata = load_embedding_table(args.embeddings_npz, args.metadata_tsv)
    embeddings, metadata = enforce_candidate_resolution(embeddings, metadata, args.candidate_resolution)
    if embeddings.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError(
            f"Embedding dimension mismatch: embeddings have {embeddings.shape[1]}, "
            f"checkpoint expects {checkpoint['input_dim']}. Use the same encoder/embedding mode used for training."
        )

    log.info(
        "Applying few-shot checkpoint=%s to %d candidate(s), classes=%s, distance_tau=%.4g",
        args.checkpoint,
        len(metadata),
        class_names,
        distance_tau,
    )
    predictions = predict_with_prototypes(
        model,
        embeddings,
        metadata,
        class_names,
        prototypes,
        prototype_counts,
        distance_tau=distance_tau,
        objectness_scale=objectness_scale,
        temperature=temperature,
        device=device,
        batch_size=int(args.batch_size),
    )
    predictions = annotate_predictions(predictions, distance_tau, objectness_scale, objectness_tau=objectness_tau)
    predictions.to_csv(out_dir / "classification_predictions.tsv", sep="\t", index=False)
    called = predictions[predictions["called_complex_sv"].astype(bool)].copy()
    called.to_csv(out_dir / "predicted_complex_sv.tsv", sep="\t", index=False)
    _write_prediction_view(predictions, class_names, out_dir / "predictions.tsv")

    compatibility_distances = fewshot_predictions_to_distance_table(predictions, class_names)
    compatibility_distances.to_csv(out_dir / "prototype_distances.tsv", sep="\t", index=False)
    if not args.skip_visualizations:
        embed_corpus.write_visualizations(
            embeddings,
            metadata,
            compatibility_distances,
            pd.DataFrame(),
            out_dir,
            tau=float(distance_tau),
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embeddings_npz": str(Path(args.embeddings_npz).resolve()),
        "metadata_tsv": str(Path(args.metadata_tsv).resolve()) if args.metadata_tsv else "",
        "class_names": class_names,
        "prototype_counts": {name: int(prototype_counts[i]) for i, name in enumerate(class_names)},
        "selected_distance_tau": float(distance_tau),
        "objectness_tau": float(objectness_tau),
        "objectness_scale": float(objectness_scale),
        "temperature": float(temperature),
        "n_candidates": int(len(predictions)),
        "n_called_complex_sv": int(called.shape[0]),
        "candidate_resolution": str(args.candidate_resolution),
        "parameter_count": int(checkpoint.get("parameter_count", _parameter_count(model))),
        "architecture": checkpoint.get("architecture", {}),
        "config": vars(args),
    }
    with (out_dir / "inference_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote few-shot inference outputs to %s; called=%d/%d", out_dir, int(called.shape[0]), len(predictions))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained fewshot_classification_head.pt")
    parser.add_argument("--embeddings_npz", required=True, help="Frozen embeddings.npz from prototype/inference embedding stage")
    parser.add_argument("--metadata_tsv", default=None, help="candidate_embeddings.tsv matching embeddings_npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_resolution", choices=("auto", "chromosome-arm", "any"), default="chromosome-arm")
    parser.add_argument("--distance_tau", type=float, default=None, help="Override checkpoint nearest-prototype distance threshold")
    parser.add_argument("--objectness_tau", type=float, default=None, help="Override checkpoint objectness probability threshold")
    parser.add_argument("--objectness_scale", type=float, default=None, help="Override checkpoint distance-to-objectness sigmoid scale")
    parser.add_argument("--temperature", type=float, default=None, help="Override checkpoint distance softmax temperature")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_visualizations", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
