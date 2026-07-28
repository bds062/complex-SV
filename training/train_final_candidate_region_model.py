#!/usr/bin/env python3
"""Fit a deployable neural model on all rows using cross-fold calibration."""

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

from training.sweep_candidate_region_general_splits import (  # noqa: E402
    calibrate_and_annotate,
    load_source,
    make_model_args,
)
from training.train_candidate_region_classifier import (  # noqa: E402
    _train_model,
    annotate_predictions_with_secondary,
    apply_cluster_aggregation,
    predict_model,
)
from training.train_multilabel_classifier_head import _hidden_dims  # noqa: E402


log = logging.getLogger(__name__)


def jsonable(values: dict) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = [value.strip() for value in args.class_names.split(",") if value.strip()]
    embeddings, metadata, tabular, tabular_names, base_cfg = load_source(source_dir, class_names)
    helper_args = make_model_args(base_cfg, args, int(args.model_seed_base))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    oof = pd.read_csv(args.cross_fold_predictions, sep="\t").fillna("")
    if set(oof["candidate_id"].astype(str)) != set(metadata["candidate_id"].astype(str)):
        raise ValueError("Cross-fold predictions do not contain exactly one held-out prediction for every candidate")
    oof = oof.set_index(oof["candidate_id"].astype(str)).loc[metadata["candidate_id"].astype(str)].reset_index(drop=True)
    calibration_dir = output_dir / "final_calibration"
    calibration_dir.mkdir(exist_ok=True)
    _, threshold_info = calibrate_and_annotate(
        oof,
        np.ones(len(oof), dtype=bool),
        class_names,
        helper_args,
        calibration_dir,
    )

    final_model, training_metrics = _train_model(
        embeddings,
        tabular,
        metadata,
        class_names,
        train_mask=np.ones(len(metadata), dtype=bool),
        args=helper_args,
        device=device,
        epochs=int(helper_args.epochs),
        patience=int(helper_args.patience),
        seed_offset=0,
        log_prefix="final_all_genomes",
    )
    training_metrics.to_csv(output_dir / "final_training_metrics.tsv", sep="\t", index=False)
    raw = predict_model(
        final_model,
        embeddings,
        tabular,
        metadata,
        class_names,
        device=device,
        batch_size=int(helper_args.batch_size),
        subtype_targets="general",
    )
    raw = apply_cluster_aggregation(raw, class_names, mode=str(helper_args.cluster_aggregation))
    predictions = annotate_predictions_with_secondary(
        raw,
        threshold_info["selected_objectness_tau"],
        threshold_info["type_thresholds"],
        class_names,
        secondary_min=threshold_info.get("selected_secondary_min"),
        secondary_delta=threshold_info.get("selected_secondary_delta"),
        rescue_type_tau=threshold_info.get("selected_rescue_type_tau"),
        rescue_objectness_floor=threshold_info.get("selected_rescue_objectness_floor"),
        rescue_margin=threshold_info.get("selected_rescue_margin"),
        subtype_thresholds={},
        subtype_targets="general",
    )
    predictions.to_csv(output_dir / "final_all_genome_predictions.tsv", sep="\t", index=False)

    embedding_modes = metadata.get("embedding_features", pd.Series(["local_context_diff_coords"])).astype(str)
    embedding_features = str(embedding_modes.mode().iloc[0]) if not embedding_modes.empty else "local_context_diff_coords"
    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "input_dim": int(embeddings.shape[1]),
        "embedding_dim": int(embeddings.shape[1]),
        "tabular_dim": int(tabular.shape[1]),
        "tabular_feature_names": tabular_names,
        "tabular_hidden_dim": int(helper_args.tabular_hidden_dim),
        "hidden_dims": _hidden_dims(helper_args.hidden_dims),
        "dropout": float(helper_args.dropout),
        "activation": str(helper_args.activation),
        "class_names": class_names,
        "subtype_targets": "general",
        "subtype_thresholding": "off",
        "subtype_thresholds": {},
        "cluster_aggregation": str(helper_args.cluster_aggregation),
        "threshold_calibration": "cross_fold_out_of_fold",
        "embedding_features": embedding_features,
        "config": jsonable(vars(helper_args)),
        **threshold_info,
    }
    torch.save(checkpoint, output_dir / "candidate_region_classifier.pt")
    summary = {
        "training_rows": int(len(metadata)),
        "training_samples": int(metadata["sample_id"].nunique()),
        "class_names": class_names,
        "calibration_rows": int(len(oof)),
        "calibration_source": str(Path(args.cross_fold_predictions).resolve()),
        "checkpoint": str((output_dir / "candidate_region_classifier.pt").resolve()),
        **threshold_info,
    }
    (output_dir / "final_model_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log.info("Wrote final all-genome checkpoint to %s", output_dir / "candidate_region_classifier.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--cross_fold_predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_names", default="ecDNA,chromothripsis,BFB")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log_every", type=int, default=1000000)
    parser.add_argument("--fast_thresholds", action="store_true")
    parser.add_argument("--model_seed_base", type=int, default=16000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
