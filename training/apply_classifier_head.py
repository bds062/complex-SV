"""Apply a trained single-label classifier head to saved complex-SV embeddings."""

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
from model.heads import ComplexSVClassifierHead
from training.train_classifier_head import (
    annotate_predictions,
    classifier_predictions_to_distance_table,
    enforce_candidate_resolution,
    load_embedding_table,
    predict_model,
)
from utils import torch_load_checkpoint

log = logging.getLogger(__name__)


def _load_head(checkpoint_path: str | Path, device: torch.device) -> tuple[ComplexSVClassifierHead, dict[str, Any]]:
    checkpoint = torch_load_checkpoint(checkpoint_path, map_location=device)
    missing = [key for key in ["model_state_dict", "input_dim", "class_names"] if key not in checkpoint]
    if missing:
        raise KeyError(f"Classifier checkpoint missing required key(s): {missing}; checkpoint={checkpoint_path}")
    class_names = [str(name) for name in checkpoint["class_names"]]
    model = ComplexSVClassifierHead(
        in_dim=int(checkpoint["input_dim"]),
        num_classes=len(class_names),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.2)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, checkpoint


def _write_call_table(predictions: pd.DataFrame, class_names: list[str], output_path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for row in predictions.to_dict("records"):
        out = {
            "candidate_id": row.get("candidate_id", ""),
            "sample_id": row.get("sample_id", ""),
            "chrom": row.get("chrom", ""),
            "arm": row.get("arm", ""),
            "start_bp": int(row.get("start_bp", 0)),
            "end_bp": int(row.get("end_bp", 0)),
            "predicted_class": row.get("predicted_class", "none"),
            "called_complex_sv": bool(row.get("called_complex_sv", False)),
            "objectness_prob": float(row.get("objectness_prob", 0.0)),
            "type_predicted_class": row.get("type_predicted_class", ""),
            "max_type_probability": float(row.get("max_type_probability", 0.0)),
            "evidence": row.get("evidence", ""),
            "candidate_scope": row.get("candidate_scope", row.get("label_scope", "")),
            "true_class": row.get("true_class", ""),
        }
        for class_name in class_names:
            out[f"type_probability_{class_name}"] = float(row.get(f"type_probability_{class_name}", 0.0))
        rows.append(out)
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    model, checkpoint = _load_head(args.checkpoint, device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    objectness_tau = float(args.tau) if args.tau is not None else float(checkpoint.get("selected_objectness_tau", 0.5))

    embeddings, metadata = load_embedding_table(args.embeddings_npz, args.metadata_tsv)
    embeddings, metadata = enforce_candidate_resolution(embeddings, metadata, args.candidate_resolution)
    if embeddings.shape[1] != int(checkpoint["input_dim"]):
        raise ValueError(
            f"Embedding dimension mismatch: embeddings have {embeddings.shape[1]}, "
            f"checkpoint expects {checkpoint['input_dim']}. Use the same encoder/embedding mode used for training."
        )

    log.info(
        "Applying single-label classifier checkpoint=%s to %d candidate(s), classes=%s, objectness_tau=%.4g",
        args.checkpoint,
        len(metadata),
        class_names,
        objectness_tau,
    )
    predictions = predict_model(model, embeddings, metadata, class_names, device=device, batch_size=int(args.batch_size))
    predictions = annotate_predictions(predictions, objectness_tau)
    predictions.to_csv(out_dir / "classification_predictions.tsv", sep="\t", index=False)

    called = predictions[predictions["called_complex_sv"].astype(bool)].copy()
    called.to_csv(out_dir / "predicted_complex_sv.tsv", sep="\t", index=False)
    _write_call_table(predictions, class_names, out_dir / "predictions.tsv")

    compatibility_distances = classifier_predictions_to_distance_table(predictions, class_names)
    compatibility_distances.to_csv(out_dir / "prototype_distances.tsv", sep="\t", index=False)

    if not args.skip_visualizations:
        classifier_distance_tau = 1.0 - float(objectness_tau)
        embed_corpus.write_visualizations(
            embeddings,
            metadata,
            compatibility_distances,
            pd.DataFrame(),
            out_dir,
            tau=classifier_distance_tau,
        )

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embeddings_npz": str(Path(args.embeddings_npz).resolve()),
        "metadata_tsv": str(Path(args.metadata_tsv).resolve()) if args.metadata_tsv else "",
        "class_names": class_names,
        "selected_objectness_tau": float(objectness_tau),
        "classifier_distance_tau_for_legacy_plots": float(1.0 - objectness_tau),
        "n_candidates": int(len(predictions)),
        "n_called_complex_sv": int(called.shape[0]),
        "candidate_resolution": str(args.candidate_resolution),
        "config": vars(args),
    }
    with (out_dir / "inference_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote classifier-head inference outputs to %s; called=%d/%d", out_dir, int(called.shape[0]), len(predictions))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Trained classification_head.pt")
    parser.add_argument("--embeddings_npz", required=True, help="Frozen embeddings.npz from prototype/inference embedding stage")
    parser.add_argument("--metadata_tsv", default=None, help="candidate_embeddings.tsv matching embeddings_npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--candidate_resolution", choices=("auto", "chromosome-arm", "any"), default="chromosome-arm")
    parser.add_argument("--tau", type=float, default=None, help="Override checkpoint objectness threshold")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip_visualizations", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
