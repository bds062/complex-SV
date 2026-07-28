#!/usr/bin/env python3
"""Apply a trained candidate-region classifier to prepared unlabeled candidates."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.train_candidate_region_classifier import (  # noqa: E402
    CandidateRegionTabularClassifierHead,
    annotate_predictions_with_secondary,
    apply_cluster_aggregation,
    cluster_prediction_table,
    predict_model,
    predictions_to_distance_table,
)
from utils import torch_load_checkpoint  # noqa: E402


log = logging.getLogger(__name__)


def load_inputs(embedding_dir: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    embedding_path = embedding_dir / "embeddings.npz"
    metadata_path = embedding_dir / "candidate_embeddings.tsv"
    tabular_path = embedding_dir / "tabular_features.npz"
    for path in [embedding_path, metadata_path, tabular_path]:
        if not path.exists():
            raise FileNotFoundError(path)
    embedding_data = np.load(embedding_path, allow_pickle=True)
    tabular_data = np.load(tabular_path, allow_pickle=True)
    embeddings = np.asarray(embedding_data["embeddings"], dtype=np.float32)
    tabular = np.asarray(tabular_data["features"], dtype=np.float32)
    feature_names = [str(value) for value in tabular_data["feature_names"].tolist()]
    metadata = pd.read_csv(metadata_path, sep="\t").fillna("")
    if not (len(metadata) == embeddings.shape[0] == tabular.shape[0]):
        raise ValueError(
            f"Row mismatch: metadata={len(metadata)} embeddings={embeddings.shape[0]} tabular={tabular.shape[0]}"
        )
    return embeddings, tabular, metadata, feature_names


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch_load_checkpoint(args.checkpoint, map_location=device)
    class_names = [str(value) for value in checkpoint["class_names"]]
    embeddings, tabular, metadata, feature_names = load_inputs(Path(args.embedding_dir))

    expected_features = [str(value) for value in checkpoint.get("tabular_feature_names", [])]
    expected_embedding_dim = int(checkpoint.get("embedding_dim", checkpoint["input_dim"]))
    if embeddings.shape[1] != expected_embedding_dim:
        raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match checkpoint {expected_embedding_dim}")
    if tabular.shape[1] != int(checkpoint["tabular_dim"]) or feature_names != expected_features:
        raise ValueError(
            "Tabular features do not match checkpoint: "
            f"prepared={feature_names} checkpoint={expected_features}"
        )

    config = checkpoint.get("config", {}) or {}
    prepared_embedding_features = set(metadata.get("embedding_features", pd.Series(dtype=str)).astype(str))
    expected_embedding_features = str(checkpoint.get("embedding_features", config.get("embedding_features", "")))
    if expected_embedding_features and prepared_embedding_features and prepared_embedding_features != {expected_embedding_features}:
        raise ValueError(
            f"Embedding feature mode mismatch: prepared={sorted(prepared_embedding_features)} "
            f"checkpoint={expected_embedding_features}"
        )
    prepared_normalization = set(metadata.get("embedding_normalization", pd.Series(dtype=str)).astype(str))
    expected_normalization = str(config.get("embedding_normalization", ""))
    if expected_normalization and prepared_normalization and prepared_normalization != {expected_normalization}:
        raise ValueError(
            f"Embedding normalization mismatch: prepared={sorted(prepared_normalization)} "
            f"checkpoint={expected_normalization}"
        )

    model = CandidateRegionTabularClassifierHead(
        embedding_dim=expected_embedding_dim,
        tabular_dim=int(checkpoint["tabular_dim"]),
        num_classes=len(class_names),
        hidden_dims=[int(value) for value in checkpoint.get("hidden_dims", [])],
        tabular_hidden_dim=int(checkpoint.get("tabular_hidden_dim", 32)),
        dropout=float(checkpoint.get("dropout", 0.2)),
        activation=str(checkpoint.get("activation", "relu")),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    raw = predict_model(
        model,
        embeddings,
        tabular,
        metadata,
        class_names,
        device=device,
        batch_size=int(args.batch_size),
        subtype_targets=str(checkpoint.get("subtype_targets", "general")),
    )
    aggregated = apply_cluster_aggregation(
        raw,
        class_names,
        mode=str(checkpoint.get("cluster_aggregation", "max")),
    )
    predictions = annotate_predictions_with_secondary(
        aggregated,
        float(checkpoint.get("selected_objectness_tau", 0.5)),
        {name: float(checkpoint.get("type_thresholds", {}).get(name, 0.5)) for name in class_names},
        class_names,
        secondary_min=checkpoint.get("selected_secondary_min"),
        secondary_delta=checkpoint.get("selected_secondary_delta"),
        rescue_type_tau=checkpoint.get("selected_rescue_type_tau"),
        rescue_objectness_floor=checkpoint.get("selected_rescue_objectness_floor"),
        rescue_margin=checkpoint.get("selected_rescue_margin"),
        subtype_thresholds=checkpoint.get("subtype_thresholds", {}),
        subtype_targets=str(checkpoint.get("subtype_targets", "general")),
    )

    raw.to_csv(output_dir / "row_raw_predictions.tsv", sep="\t", index=False)
    predictions.to_csv(output_dir / "classification_predictions.tsv", sep="\t", index=False)
    predictions[predictions["called_complex_sv"].astype(bool)].to_csv(
        output_dir / "predicted_complex_sv.tsv", sep="\t", index=False
    )
    cluster_prediction_table(predictions, class_names).to_csv(
        output_dir / "cluster_predictions.tsv", sep="\t", index=False
    )
    predictions_to_distance_table(predictions, class_names).to_csv(
        output_dir / "prototype_distances.tsv", sep="\t", index=False
    )
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "embedding_dir": str(Path(args.embedding_dir).resolve()),
        "class_names": class_names,
        "n_candidates": int(len(predictions)),
        "n_called_complex_sv": int(predictions["called_complex_sv"].astype(bool).sum()),
        "selected_objectness_tau": float(checkpoint.get("selected_objectness_tau", 0.5)),
        "type_thresholds": checkpoint.get("type_thresholds", {}),
        "secondary_min": checkpoint.get("selected_secondary_min"),
        "secondary_delta": checkpoint.get("selected_secondary_delta"),
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Wrote %d predictions (%d called) to %s", len(predictions), summary["n_called_complex_sv"], output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--embedding_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
