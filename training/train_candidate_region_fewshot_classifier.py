"""Train a prototypical few-shot candidate-region complex-SV classifier.

This is a parallel experiment to train_candidate_region_classifier.py. It uses
the same candidate-region embeddings, tabular features, splits, LOGO threshold
calibration, subtype threshold bands, and plotting-compatible outputs, but
replaces the neural classifier head with class prototypes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discovery import embed_corpus
from training.train_candidate_region_classifier import (
    DEFAULT_CLASS_NAMES,
    _class_counts,
    _jsonable_config,
    _label_background_masks,
    _multi_hot_targets,
    _normalize_base_class_columns,
    _split_csv,
    _threshold_grid,
    _uses_binary_subtype_targets,
    annotate_predictions_with_secondary,
    apply_cluster_aggregation,
    choose_rescue_thresholding,
    choose_secondary_thresholding,
    choose_subtype_thresholds_from_predictions,
    choose_tau_from_sweep,
    choose_type_thresholds_from_predictions,
    cluster_prediction_table,
    embed_candidate_table,
    make_tabular_features,
    metric_tables,
    plot_per_class_metrics,
    plot_split_metrics,
    read_candidate_regions,
    rescue_sweep_table_for_selected,
    secondary_sweep_table_for_selected,
    select_embedding_features,
    select_test_samples,
    selected_tabular_feature_names,
    write_tabular_feature_table,
)
from training.train_multilabel_classifier_head import (
    _plot_thresholds,
    predictions_to_distance_table,
    sweep_objectness_tau,
)
from utils import set_seed

log = logging.getLogger(__name__)


@dataclass
class Prototype:
    class_name: str
    prototype_name: str
    prototype_kind: str
    n_members: int
    vector: np.ndarray


def _safe_logit(prob: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    arr = np.asarray(prob, dtype=np.float32)
    arr = np.clip(arr, eps, 1.0 - eps)
    return np.log(arr / (1.0 - arr)).astype(np.float32)


def _l2_normalize(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, eps, None)


def _weighted_centroid(features: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0:
        raise ValueError("Cannot build a prototype from an empty feature matrix")
    if weights is None:
        centroid = np.mean(x, axis=0)
    else:
        w = np.asarray(weights, dtype=np.float32).reshape(-1)
        w = np.clip(w, 1e-6, None)
        centroid = np.average(x, axis=0, weights=w)
    norm = float(np.linalg.norm(centroid))
    if norm > 1e-12:
        centroid = centroid / norm
    return centroid.astype(np.float32)


def _prototype_weights(targets: np.ndarray, class_i: int, subtype_targets: str, weighting: str) -> np.ndarray:
    mode = str(weighting or "auto").lower()
    if mode in {"off", "none", "binary", "0"}:
        return np.ones(targets.shape[0], dtype=np.float32)
    if mode == "auto" and _uses_binary_subtype_targets(subtype_targets):
        return np.ones(targets.shape[0], dtype=np.float32)
    return np.clip(np.asarray(targets[:, class_i], dtype=np.float32), 1e-6, None)


def make_prototype_features(
    embeddings: np.ndarray,
    tabular_features: np.ndarray,
    tabular_weight: float,
    final_l2_normalize: bool,
) -> np.ndarray:
    emb = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
    tab = np.asarray(tabular_features, dtype=np.float32)
    if tab.ndim != 2 or tab.shape[0] != emb.shape[0]:
        raise ValueError(f"tabular_features shape {tab.shape} does not match embeddings shape {emb.shape}")
    if tab.shape[1] > 0:
        features = np.concatenate([emb, float(tabular_weight) * tab], axis=1).astype(np.float32)
    else:
        features = emb.astype(np.float32)
    if final_l2_normalize:
        features = _l2_normalize(features)
    return features.astype(np.float32)


def _cluster_centers(
    features: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_clusters <= 1 or features.shape[0] <= 1:
        return [(np.arange(features.shape[0]), _weighted_centroid(features, weights))]
    try:
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=int(n_clusters), random_state=int(seed), n_init=20)
        labels = km.fit_predict(features)
    except Exception as exc:
        log.warning("KMeans prototype clustering failed; using one containing-class prototype: %s", exc)
        return [(np.arange(features.shape[0]), _weighted_centroid(features, weights))]

    centers: list[tuple[np.ndarray, np.ndarray]] = []
    for cluster_i in range(int(n_clusters)):
        idx = np.where(labels == cluster_i)[0]
        if idx.size == 0:
            continue
        centers.append((idx, _weighted_centroid(features[idx], weights[idx])))
    return centers


def build_prototypes(
    features: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    train_mask: np.ndarray,
    subtype_targets: str,
    containing_prototypes: int,
    min_prototype_members: int,
    min_cluster_members: int,
    subtype_weighting: str,
    seed: int,
) -> tuple[list[Prototype], pd.DataFrame]:
    targets = _multi_hot_targets(metadata, class_names, subtype_targets=subtype_targets)
    labeled, _background = _label_background_masks(metadata)
    train_labeled = np.asarray(train_mask, dtype=bool) & labeled
    label_counts = (targets > 0).sum(axis=1)
    prototypes: list[Prototype] = []
    rows: list[dict[str, Any]] = []

    for class_i, class_name in enumerate(class_names):
        positive = train_labeled & (targets[:, class_i] > 0)
        only = positive & (label_counts == 1)
        all_idx = np.where(positive)[0]
        only_idx = np.where(only)[0]
        all_weights = _prototype_weights(targets, class_i, subtype_targets, subtype_weighting)

        if only_idx.size >= int(min_prototype_members):
            proto = _weighted_centroid(features[only_idx], all_weights[only_idx])
            prototypes.append(
                Prototype(
                    class_name=class_name,
                    prototype_name=f"{class_name}__only",
                    prototype_kind="only_class",
                    n_members=int(only_idx.size),
                    vector=proto,
                )
            )

        if all_idx.size >= int(min_prototype_members):
            requested_k = max(1, int(containing_prototypes))
            max_by_members = max(1, int(all_idx.size) // max(1, int(min_cluster_members)))
            n_clusters = min(requested_k, int(all_idx.size), max_by_members if requested_k > 1 else requested_k)
            n_clusters = max(1, n_clusters)
            centers = _cluster_centers(features[all_idx], all_weights[all_idx], n_clusters, seed + class_i)
            for cluster_i, (relative_idx, center) in enumerate(centers, start=1):
                suffix = "all" if len(centers) == 1 else f"all_k{cluster_i}"
                prototypes.append(
                    Prototype(
                        class_name=class_name,
                        prototype_name=f"{class_name}__{suffix}",
                        prototype_kind="contains_class",
                        n_members=int(len(relative_idx)),
                        vector=center,
                    )
                )

        rows.append(
            {
                "class_name": class_name,
                "n_train_positive": int(all_idx.size),
                "n_train_only_class": int(only_idx.size),
                "n_prototypes": int(sum(p.class_name == class_name for p in prototypes)),
            }
        )

    if not prototypes:
        raise RuntimeError("No prototypes were built; training split has no usable positive labels")
    return prototypes, pd.DataFrame(rows)


def prototype_table(prototypes: list[Prototype]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "class_name": proto.class_name,
                "prototype_name": proto.prototype_name,
                "prototype_kind": proto.prototype_kind,
                "n_members": int(proto.n_members),
                "prototype_norm": float(np.linalg.norm(proto.vector)),
            }
            for proto in prototypes
        ]
    )


def prototype_arrays(prototypes: list[Prototype]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vectors = np.stack([proto.vector for proto in prototypes], axis=0).astype(np.float32)
    classes = np.asarray([proto.class_name for proto in prototypes], dtype=object)
    names = np.asarray([proto.prototype_name for proto in prototypes], dtype=object)
    kinds = np.asarray([proto.prototype_kind for proto in prototypes], dtype=object)
    return vectors, classes, names, kinds


def _score_from_distance(distance: np.ndarray, temperature: float, transform: str) -> np.ndarray:
    if str(transform).lower() == "cosine_shift":
        similarity = 1.0 - np.asarray(distance, dtype=np.float32)
        return np.clip((similarity + 1.0) / 2.0, 1e-6, 1.0 - 1e-6).astype(np.float32)
    temp = max(float(temperature), 1e-6)
    return np.clip(np.exp(-np.asarray(distance, dtype=np.float32) / temp), 1e-6, 1.0 - 1e-6).astype(np.float32)


def predict_with_prototypes(
    prototypes: list[Prototype],
    features: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    subtype_targets: str,
    temperature: float,
    score_transform: str,
) -> pd.DataFrame:
    vectors, proto_classes, proto_names, _proto_kinds = prototype_arrays(prototypes)
    x = _l2_normalize(features)
    p = _l2_normalize(vectors)
    similarity = np.clip(x @ p.T, -1.0, 1.0)
    distances = 1.0 - similarity

    n = x.shape[0]
    class_scores = np.zeros((n, len(class_names)), dtype=np.float32)
    class_distances = np.ones((n, len(class_names)), dtype=np.float32) * np.inf
    best_proto_names: list[list[str]] = [[""] * len(class_names) for _ in range(n)]

    for class_i, class_name in enumerate(class_names):
        cols = np.where(proto_classes == class_name)[0]
        if cols.size == 0:
            class_scores[:, class_i] = 1e-6
            class_distances[:, class_i] = np.inf
            continue
        sub_dist = distances[:, cols]
        best_relative = np.argmin(sub_dist, axis=1)
        best_cols = cols[best_relative]
        best_dist = sub_dist[np.arange(n), best_relative]
        class_distances[:, class_i] = best_dist.astype(np.float32)
        class_scores[:, class_i] = _score_from_distance(best_dist, temperature, score_transform)
        for row_i, proto_col in enumerate(best_cols):
            best_proto_names[row_i][class_i] = str(proto_names[int(proto_col)])

    objectness_prob = np.max(class_scores, axis=1)
    objectness_logit = _safe_logit(objectness_prob)
    type_logits = _safe_logit(class_scores)
    max_idx = np.argmax(class_scores, axis=1)

    out = metadata.copy()
    labeled, background = _label_background_masks(out)
    targets = _multi_hot_targets(out, class_names, subtype_targets=subtype_targets)
    out["is_labeled"] = labeled.astype(int)
    out["is_background_chromosome"] = background.astype(int)
    if "raw_sv_classes" in out:
        out["raw_true_classes"] = out["raw_sv_classes"].astype(str)
    out["true_classes"] = [";".join([class_names[i] for i, flag in enumerate(row) if flag > 0]) for row in targets]
    if _uses_binary_subtype_targets(subtype_targets):
        _normalize_base_class_columns(out, ["raw_sv_classes", "raw_true_classes", "sv_classes", "sv_class", "true_classes"])

    out["objectness_logit"] = objectness_logit.astype(float)
    out["objectness_prob"] = objectness_prob.astype(float)
    out["top_type_class"] = [class_names[int(i)] for i in max_idx]
    out["max_type_probability"] = class_scores.max(axis=1).astype(float)
    out["max_type_logit"] = type_logits[np.arange(type_logits.shape[0]), max_idx].astype(float)
    out["nearest_fewshot_prototype"] = [best_proto_names[row_i][int(max_idx[row_i])] for row_i in range(n)]
    out["nearest_fewshot_distance"] = class_distances[np.arange(n), max_idx].astype(float)
    out["fewshot_score_transform"] = str(score_transform)
    out["fewshot_temperature"] = float(temperature)
    for class_i, class_name in enumerate(class_names):
        out[f"type_logit_{class_name}"] = type_logits[:, class_i].astype(float)
        out[f"type_probability_{class_name}"] = class_scores[:, class_i].astype(float)
        out[f"prototype_distance_{class_name}"] = class_distances[:, class_i].astype(float)
        out[f"nearest_prototype_{class_name}"] = [best_proto_names[row_i][class_i] for row_i in range(n)]
    return out


def load_existing_embeddings(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame] | None:
    if args.embeddings_npz and args.metadata_tsv:
        emb_path = Path(args.embeddings_npz)
        meta_path = Path(args.metadata_tsv)
        if not emb_path.exists():
            raise FileNotFoundError(f"--embeddings_npz not found: {emb_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"--metadata_tsv not found: {meta_path}")
        data = np.load(emb_path, allow_pickle=True)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        metadata = pd.read_csv(meta_path, sep="\t").fillna("")
        log.info("Using external embeddings: %s and %s", emb_path, meta_path)
        return embeddings, metadata
    return None


def run_logo_calibration(
    features: np.ndarray,
    metadata: pd.DataFrame,
    class_names: list[str],
    outer_train_mask: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = _multi_hot_targets(metadata, class_names, subtype_targets=str(args.subtype_targets))
    labeled, background = _label_background_masks(metadata)
    samples = metadata["sample_id"].astype(str).to_numpy()
    eligible_outer = outer_train_mask & (labeled | background)
    held_samples = sorted(pd.unique(samples[eligible_outer]).tolist())
    rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []

    log.info("Running few-shot LOGO threshold calibration on %d training genome(s)", len(held_samples))
    for fold_i, held_sample in enumerate(held_samples, start=1):
        eval_mask = eligible_outer & (samples == held_sample)
        fold_train_mask = outer_train_mask & (samples != held_sample)
        n_fold_pos = int((fold_train_mask & labeled).sum())
        n_fold_background = int((fold_train_mask & background).sum())
        if n_fold_pos == 0 or int(eval_mask.sum()) == 0:
            log.warning(
                "Skipping LOGO fold held_out=%s; train_pos=%d eval_candidates=%d",
                held_sample,
                n_fold_pos,
                int(eval_mask.sum()),
            )
            continue
        prototypes, class_summary = build_prototypes(
            features,
            metadata,
            class_names,
            fold_train_mask,
            subtype_targets=str(args.subtype_targets),
            containing_prototypes=int(args.containing_prototypes),
            min_prototype_members=int(args.min_prototype_members),
            min_cluster_members=int(args.min_cluster_members),
            subtype_weighting=str(args.prototype_subtype_weighting),
            seed=int(args.seed) + 2000 + fold_i,
        )
        pred = predict_with_prototypes(
            prototypes,
            features,
            metadata,
            class_names,
            subtype_targets=str(args.subtype_targets),
            temperature=float(args.prototype_temperature),
            score_transform=str(args.score_transform),
        )
        pred = apply_cluster_aggregation(pred, class_names, mode=str(args.cluster_aggregation))
        fold = pred.loc[eval_mask].copy()
        fold["held_out_sample"] = held_sample
        fold["calibration_split"] = "logo"
        fold["logo_fold"] = int(fold_i)
        fold["logo_train_n_labeled"] = n_fold_pos
        fold["logo_train_n_empty"] = n_fold_background
        rows.append(fold)

        metric_row: dict[str, Any] = {
            "held_out_sample": held_sample,
            "calibration_split": "logo",
            "logo_fold": int(fold_i),
            "n_train_labeled": n_fold_pos,
            "n_train_empty": n_fold_background,
            "n_eval_candidates": int(eval_mask.sum()),
            "n_prototypes": int(len(prototypes)),
            "n_classes_with_prototypes": int(class_summary["n_prototypes"].astype(int).gt(0).sum()),
        }
        for class_name in class_names:
            metric_row[f"n_prototypes_{class_name}"] = int((prototype_table(prototypes)["class_name"] == class_name).sum())
            metric_row[f"n_train_positive_{class_name}"] = int((targets[fold_train_mask, class_names.index(class_name)] > 0).sum())
        metric_rows.append(metric_row)

    calibration_predictions = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    calibration_metrics = pd.DataFrame(metric_rows)
    return calibration_predictions, calibration_metrics


def _write_threshold_tables(
    output_dir: Path,
    class_names: list[str],
    type_thresholds: dict[str, float],
    subtype_thresholds: dict[str, dict[str, float]],
    selected_secondary_min: float | None,
    selected_secondary_delta: float | None,
) -> None:
    pd.DataFrame(
        [
            {
                "class_name": name,
                "type_threshold": float(type_thresholds.get(name, 0.5)),
                "secondary_threshold": (
                    max(float(type_thresholds.get(name, 0.5)) + float(selected_secondary_delta), float(selected_secondary_min))
                    if selected_secondary_min is not None and selected_secondary_delta is not None
                    else float(type_thresholds.get(name, 0.5))
                ),
            }
            for name in class_names
        ]
    ).to_csv(output_dir / "type_thresholds.tsv", sep="\t", index=False)
    pd.DataFrame(
        [
            {
                "class_name": name,
                "has_subtypes": bool(float(subtype_thresholds.get(name, {}).get("has_subtypes", 0.0)) > 0.0),
                "noncanonicalB_threshold": float(subtype_thresholds.get(name, {}).get("noncanonicalB", type_thresholds.get(name, 0.5))),
                "noncanonical_threshold": float(subtype_thresholds.get(name, {}).get("noncanonical", 1.01)),
                "canonical_threshold": float(subtype_thresholds.get(name, {}).get("canonical", 1.01)),
                "type_threshold": float(type_thresholds.get(name, 0.5)),
            }
            for name in class_names
        ]
    ).to_csv(output_dir / "subtype_thresholds.tsv", sep="\t", index=False)


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = _split_csv(args.class_names)
    if not class_names:
        raise ValueError("--class_names must include at least one class")

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    existing = load_existing_embeddings(args)
    if existing is not None:
        embeddings, metadata = existing
    elif args.reuse_embeddings and (output_dir / "embeddings.npz").exists() and (output_dir / "candidate_embeddings.tsv").exists():
        data = np.load(output_dir / "embeddings.npz", allow_pickle=True)
        embeddings = np.asarray(data["embeddings"], dtype=np.float32)
        metadata = pd.read_csv(output_dir / "candidate_embeddings.tsv", sep="\t").fillna("")
        log.info("Reusing embeddings from %s", output_dir)
    else:
        manifest = embed_corpus.read_manifest(args.manifest)
        candidates_df = read_candidate_regions(args.candidate_regions, class_names)
        embeddings, metadata = embed_candidate_table(
            candidates_df,
            manifest,
            args.cn_checkpoint,
            args.graph_checkpoint,
            output_dir,
            args.embedding_normalization,
            int(args.sample_baseline_min_candidates),
            bool(args.strict),
            device,
        )

    embeddings, metadata, selected_embedding_features = select_embedding_features(
        embeddings,
        metadata,
        str(args.embedding_features),
        output_dir,
    )
    log.info("Using embedding_features=%s; embedding_dim=%d", selected_embedding_features, embeddings.shape[1])

    tabular_feature_names = selected_tabular_feature_names(str(args.tabular_features))
    tabular_features = make_tabular_features(metadata, tabular_feature_names)
    write_tabular_feature_table(metadata, tabular_features, tabular_feature_names, output_dir)
    features = make_prototype_features(
        embeddings,
        tabular_features,
        tabular_weight=float(args.tabular_weight),
        final_l2_normalize=bool(args.final_l2_normalize),
    )
    post_model_embeddings = features.astype(np.float32)
    np.savez(
        output_dir / "fewshot_feature_matrix.npz",
        features=post_model_embeddings,
        embedding_dim=int(embeddings.shape[1]),
        tabular_dim=int(tabular_features.shape[1]),
        tabular_weight=float(args.tabular_weight),
    )
    np.savez(
        output_dir / "post_model_embeddings.npz",
        embeddings=post_model_embeddings,
        model_space="fewshot_prototype_feature_space",
        embedding_dim=int(embeddings.shape[1]),
        tabular_dim=int(tabular_features.shape[1]),
        tabular_weight=float(args.tabular_weight),
        final_l2_normalize=bool(args.final_l2_normalize),
    )

    targets = _multi_hot_targets(metadata, class_names, subtype_targets=str(args.subtype_targets))
    labeled, background = _label_background_masks(metadata)
    class_counts = _class_counts(targets, class_names)
    samples = metadata["sample_id"].astype(str).to_numpy()
    test_samples = select_test_samples(
        samples,
        required=str(args.required_test_sample),
        explicit=_split_csv(args.test_samples),
        n_test_samples=int(args.n_test_samples),
        seed=int(args.seed),
    )
    test_mask = np.isin(samples, np.asarray(test_samples, dtype=object))
    train_mask = ~test_mask
    if int((train_mask & labeled).sum()) < 2:
        raise RuntimeError("Training split has fewer than two positive candidate regions")

    split_rows = []
    for sample_id in sorted(pd.unique(samples)):
        sample_mask = samples == sample_id
        split_rows.append(
            {
                "sample_id": sample_id,
                "split": "test" if sample_id in test_samples else "train",
                "n_candidates": int(sample_mask.sum()),
                "n_positive": int((sample_mask & labeled).sum()),
                "n_empty": int((sample_mask & background).sum()),
            }
        )
    pd.DataFrame(split_rows).to_csv(output_dir / "sample_splits.tsv", sep="\t", index=False)
    log.info(
        "Split: train_candidates=%d train_pos=%d train_empty=%d test_samples=%s test_candidates=%d",
        int(train_mask.sum()),
        int((train_mask & labeled).sum()),
        int((train_mask & background).sum()),
        ",".join(test_samples),
        int(test_mask.sum()),
    )

    objectness_grid = _threshold_grid(args.tau_min, args.tau_max, args.tau_steps)
    type_grid = _threshold_grid(args.type_tau_min, args.type_tau_max, args.type_tau_steps)
    threshold_calibration = str(args.threshold_calibration).lower()
    objectness_tau_df = pd.DataFrame()
    type_threshold_sweep = pd.DataFrame()
    subtype_threshold_sweep = pd.DataFrame()
    rescue_threshold_sweep = pd.DataFrame()
    secondary_threshold_sweep = pd.DataFrame()
    calibration_raw = pd.DataFrame()
    calibration_annotated = pd.DataFrame()
    calibration_training_metrics = pd.DataFrame()
    selected_objectness_tau: float | None = None
    type_thresholds: dict[str, float] = {}
    subtype_thresholds: dict[str, dict[str, float]] = {}
    selected_rescue_type_tau: float | None = None
    selected_rescue_objectness_floor: float | None = None
    selected_rescue_margin: float | None = None
    selected_secondary_min: float | None = None
    selected_secondary_delta: float | None = None

    if threshold_calibration == "logo":
        calibration_raw, calibration_training_metrics = run_logo_calibration(
            features,
            metadata,
            class_names,
            train_mask,
            args,
        )
        if calibration_raw.empty:
            raise RuntimeError("Few-shot LOGO threshold calibration produced no held-out predictions")
        calibration_raw.to_csv(output_dir / "logo_calibration_raw.tsv", sep="\t", index=False)
        calibration_training_metrics.to_csv(output_dir / "logo_training_metrics.tsv", sep="\t", index=False)
        objectness_tau_df = sweep_objectness_tau(calibration_raw, objectness_grid)
        selected_objectness_tau = float(args.tau) if args.tau is not None else choose_tau_from_sweep(
            objectness_tau_df,
            metric=str(args.tau_selection_metric),
            tie_break=str(args.threshold_tie_break),
        )
        if args.type_tau is not None:
            type_thresholds = {name: float(args.type_tau) for name in class_names}
        else:
            type_thresholds, type_threshold_sweep = choose_type_thresholds_from_predictions(
                calibration_raw,
                class_names,
                type_grid,
                tie_break=str(args.threshold_tie_break),
            )
        if str(args.subtype_thresholding).lower() == "optimize" and not _uses_binary_subtype_targets(args.subtype_targets):
            subtype_thresholds, subtype_threshold_sweep = choose_subtype_thresholds_from_predictions(
                calibration_raw,
                class_names,
                type_thresholds,
                type_grid,
                tie_break=str(args.threshold_tie_break),
            )
        selected_rescue_type_tau, selected_rescue_objectness_floor, selected_rescue_margin, rescue_threshold_sweep = choose_rescue_thresholding(
            calibration_raw,
            float(selected_objectness_tau),
            type_thresholds,
            class_names,
            args,
            subtype_thresholds=subtype_thresholds,
        )
        selected_secondary_min, selected_secondary_delta, secondary_threshold_sweep = choose_secondary_thresholding(
            calibration_raw,
            float(selected_objectness_tau),
            type_thresholds,
            class_names,
            args,
            subtype_thresholds=subtype_thresholds,
            rescue_type_tau=selected_rescue_type_tau,
            rescue_objectness_floor=selected_rescue_objectness_floor,
            rescue_margin=selected_rescue_margin,
        )
        calibration_annotated = annotate_predictions_with_secondary(
            calibration_raw,
            float(selected_objectness_tau),
            type_thresholds,
            class_names,
            secondary_min=selected_secondary_min,
            secondary_delta=selected_secondary_delta,
            rescue_type_tau=selected_rescue_type_tau,
            rescue_objectness_floor=selected_rescue_objectness_floor,
            rescue_margin=selected_rescue_margin,
            subtype_thresholds=subtype_thresholds,
            subtype_targets=str(args.subtype_targets),
        )
        if rescue_threshold_sweep.empty:
            rescue_threshold_sweep = rescue_sweep_table_for_selected(
                calibration_annotated,
                class_names,
                selected_rescue_type_tau,
                selected_rescue_objectness_floor,
                selected_rescue_margin,
                str(args.rescue_thresholding),
            )
        if secondary_threshold_sweep.empty:
            secondary_threshold_sweep = secondary_sweep_table_for_selected(
                calibration_annotated,
                class_names,
                selected_secondary_min,
                selected_secondary_delta,
                str(args.secondary_thresholding),
            )
        calibration_annotated.to_csv(output_dir / "logo_calibration_predictions.tsv", sep="\t", index=False)
        objectness_tau_df.to_csv(output_dir / "objectness_tau_sweep_logo.tsv", sep="\t", index=False)
        objectness_tau_df.to_csv(output_dir / "objectness_tau_sweep_calibration.tsv", sep="\t", index=False)
        type_threshold_sweep.to_csv(output_dir / "type_threshold_sweep_logo.tsv", sep="\t", index=False)
        type_threshold_sweep.to_csv(output_dir / "type_threshold_sweep_calibration.tsv", sep="\t", index=False)
        subtype_threshold_sweep.to_csv(output_dir / "subtype_threshold_sweep_logo.tsv", sep="\t", index=False)
        subtype_threshold_sweep.to_csv(output_dir / "subtype_threshold_sweep_calibration.tsv", sep="\t", index=False)
        rescue_threshold_sweep.to_csv(output_dir / "rescue_threshold_sweep_logo.tsv", sep="\t", index=False)
        rescue_threshold_sweep.to_csv(output_dir / "rescue_threshold_sweep_calibration.tsv", sep="\t", index=False)
        secondary_threshold_sweep.to_csv(output_dir / "secondary_threshold_sweep_logo.tsv", sep="\t", index=False)
        secondary_threshold_sweep.to_csv(output_dir / "secondary_threshold_sweep_calibration.tsv", sep="\t", index=False)
        logo_overall, logo_per_class = metric_tables(calibration_annotated, class_names, "logo")
        logo_overall.to_csv(output_dir / "logo_metrics_summary.tsv", sep="\t", index=False)
        logo_per_class.to_csv(output_dir / "logo_per_class_metrics.tsv", sep="\t", index=False)

    final_prototypes, final_class_summary = build_prototypes(
        features,
        metadata,
        class_names,
        train_mask,
        subtype_targets=str(args.subtype_targets),
        containing_prototypes=int(args.containing_prototypes),
        min_prototype_members=int(args.min_prototype_members),
        min_cluster_members=int(args.min_cluster_members),
        subtype_weighting=str(args.prototype_subtype_weighting),
        seed=int(args.seed),
    )
    proto_df = prototype_table(final_prototypes)
    proto_df.to_csv(output_dir / "fewshot_prototypes.tsv", sep="\t", index=False)
    final_class_summary.to_csv(output_dir / "fewshot_class_summary.tsv", sep="\t", index=False)
    vectors, proto_classes, proto_names, proto_kinds = prototype_arrays(final_prototypes)
    np.savez(
        output_dir / "fewshot_prototypes.npz",
        prototypes=vectors,
        class_names=proto_classes,
        prototype_names=proto_names,
        prototype_kinds=proto_kinds,
    )

    row_raw_predictions = predict_with_prototypes(
        final_prototypes,
        features,
        metadata,
        class_names,
        subtype_targets=str(args.subtype_targets),
        temperature=float(args.prototype_temperature),
        score_transform=str(args.score_transform),
    )
    row_raw_predictions.to_csv(output_dir / "row_raw_predictions.tsv", sep="\t", index=False)
    raw_predictions = apply_cluster_aggregation(row_raw_predictions, class_names, mode=str(args.cluster_aggregation))
    raw_predictions.to_csv(output_dir / "cluster_aggregated_raw_predictions.tsv", sep="\t", index=False)
    train_raw = raw_predictions.loc[train_mask].copy()
    train_objectness_tau_df = sweep_objectness_tau(train_raw, objectness_grid)
    train_objectness_tau_df.to_csv(output_dir / "objectness_tau_sweep_in_sample_train.tsv", sep="\t", index=False)
    train_type_thresholds, train_type_threshold_sweep = choose_type_thresholds_from_predictions(
        train_raw,
        class_names,
        type_grid,
        tie_break=str(args.threshold_tie_break),
    )
    train_type_threshold_sweep.to_csv(output_dir / "type_threshold_sweep_in_sample_train.tsv", sep="\t", index=False)
    if str(args.subtype_thresholding).lower() == "optimize" and not _uses_binary_subtype_targets(args.subtype_targets):
        train_subtype_thresholds, train_subtype_threshold_sweep = choose_subtype_thresholds_from_predictions(
            train_raw,
            class_names,
            train_type_thresholds,
            type_grid,
            tie_break=str(args.threshold_tie_break),
        )
    else:
        train_subtype_thresholds, train_subtype_threshold_sweep = {}, pd.DataFrame()
    train_subtype_threshold_sweep.to_csv(output_dir / "subtype_threshold_sweep_in_sample_train.tsv", sep="\t", index=False)

    if threshold_calibration == "train":
        objectness_tau_df = train_objectness_tau_df
        selected_objectness_tau = float(args.tau) if args.tau is not None else choose_tau_from_sweep(
            objectness_tau_df,
            metric=str(args.tau_selection_metric),
            tie_break=str(args.threshold_tie_break),
        )
        if args.type_tau is not None:
            type_thresholds = {name: float(args.type_tau) for name in class_names}
        else:
            type_thresholds = train_type_thresholds
            type_threshold_sweep = train_type_threshold_sweep
        if str(args.subtype_thresholding).lower() == "optimize" and not _uses_binary_subtype_targets(args.subtype_targets):
            subtype_thresholds = train_subtype_thresholds
            subtype_threshold_sweep = train_subtype_threshold_sweep
        selected_rescue_type_tau, selected_rescue_objectness_floor, selected_rescue_margin, rescue_threshold_sweep = choose_rescue_thresholding(
            train_raw,
            float(selected_objectness_tau),
            type_thresholds,
            class_names,
            args,
            subtype_thresholds=subtype_thresholds,
        )
        selected_secondary_min, selected_secondary_delta, secondary_threshold_sweep = choose_secondary_thresholding(
            train_raw,
            float(selected_objectness_tau),
            type_thresholds,
            class_names,
            args,
            subtype_thresholds=subtype_thresholds,
            rescue_type_tau=selected_rescue_type_tau,
            rescue_objectness_floor=selected_rescue_objectness_floor,
            rescue_margin=selected_rescue_margin,
        )
        train_calibration_annotated = annotate_predictions_with_secondary(
            train_raw,
            float(selected_objectness_tau),
            type_thresholds,
            class_names,
            secondary_min=selected_secondary_min,
            secondary_delta=selected_secondary_delta,
            rescue_type_tau=selected_rescue_type_tau,
            rescue_objectness_floor=selected_rescue_objectness_floor,
            rescue_margin=selected_rescue_margin,
            subtype_thresholds=subtype_thresholds,
            subtype_targets=str(args.subtype_targets),
        )
        if rescue_threshold_sweep.empty:
            rescue_threshold_sweep = rescue_sweep_table_for_selected(
                train_calibration_annotated,
                class_names,
                selected_rescue_type_tau,
                selected_rescue_objectness_floor,
                selected_rescue_margin,
                str(args.rescue_thresholding),
            )
        if secondary_threshold_sweep.empty:
            secondary_threshold_sweep = secondary_sweep_table_for_selected(
                train_calibration_annotated,
                class_names,
                selected_secondary_min,
                selected_secondary_delta,
                str(args.secondary_thresholding),
            )
        objectness_tau_df.to_csv(output_dir / "objectness_tau_sweep_train.tsv", sep="\t", index=False)
        objectness_tau_df.to_csv(output_dir / "objectness_tau_sweep_calibration.tsv", sep="\t", index=False)
        type_threshold_sweep.to_csv(output_dir / "type_threshold_sweep_train.tsv", sep="\t", index=False)
        type_threshold_sweep.to_csv(output_dir / "type_threshold_sweep_calibration.tsv", sep="\t", index=False)
        subtype_threshold_sweep.to_csv(output_dir / "subtype_threshold_sweep_train.tsv", sep="\t", index=False)
        subtype_threshold_sweep.to_csv(output_dir / "subtype_threshold_sweep_calibration.tsv", sep="\t", index=False)
        rescue_threshold_sweep.to_csv(output_dir / "rescue_threshold_sweep_train.tsv", sep="\t", index=False)
        rescue_threshold_sweep.to_csv(output_dir / "rescue_threshold_sweep_calibration.tsv", sep="\t", index=False)
        secondary_threshold_sweep.to_csv(output_dir / "secondary_threshold_sweep_train.tsv", sep="\t", index=False)
        secondary_threshold_sweep.to_csv(output_dir / "secondary_threshold_sweep_calibration.tsv", sep="\t", index=False)

    if selected_objectness_tau is None:
        raise RuntimeError("No objectness threshold was selected")
    _plot_thresholds(objectness_tau_df, type_threshold_sweep, output_dir, float(selected_objectness_tau))

    predictions = annotate_predictions_with_secondary(
        raw_predictions,
        float(selected_objectness_tau),
        type_thresholds,
        class_names,
        secondary_min=selected_secondary_min,
        secondary_delta=selected_secondary_delta,
        rescue_type_tau=selected_rescue_type_tau,
        rescue_objectness_floor=selected_rescue_objectness_floor,
        rescue_margin=selected_rescue_margin,
        subtype_thresholds=subtype_thresholds,
        subtype_targets=str(args.subtype_targets),
    )
    predictions["split"] = np.where(test_mask, "test", "train")
    predictions.to_csv(output_dir / "classification_predictions.tsv", sep="\t", index=False)
    predictions.loc[train_mask].to_csv(output_dir / "train_predictions.tsv", sep="\t", index=False)
    predictions.loc[test_mask].to_csv(output_dir / "test_predictions.tsv", sep="\t", index=False)
    predictions[predictions["called_complex_sv"].astype(bool)].to_csv(output_dir / "predicted_complex_sv.tsv", sep="\t", index=False)
    cluster_predictions = cluster_prediction_table(predictions, class_names)
    cluster_predictions.to_csv(output_dir / "cluster_predictions.tsv", sep="\t", index=False)
    _write_threshold_tables(
        output_dir,
        class_names,
        type_thresholds,
        subtype_thresholds,
        selected_secondary_min,
        selected_secondary_delta,
    )

    overall_tables: list[pd.DataFrame] = []
    per_class_tables: list[pd.DataFrame] = []
    for split_name, split_mask in [("train", train_mask), ("test", test_mask), ("all", np.ones(len(metadata), dtype=bool))]:
        overall, per_class = metric_tables(predictions.loc[split_mask].copy(), class_names, split_name)
        if not overall.empty:
            overall_tables.append(overall)
        if not per_class.empty:
            per_class_tables.append(per_class)
    overall_metrics = pd.concat(overall_tables, ignore_index=True) if overall_tables else pd.DataFrame()
    per_class_metrics = pd.concat(per_class_tables, ignore_index=True) if per_class_tables else pd.DataFrame()
    overall_metrics.to_csv(output_dir / "metrics_summary.tsv", sep="\t", index=False)
    per_class_metrics.to_csv(output_dir / "per_class_metrics.tsv", sep="\t", index=False)
    plot_split_metrics(overall_metrics, output_dir / "split_metrics.png")
    plot_per_class_metrics(per_class_metrics, output_dir / "per_class_metrics.png")

    compatibility_distances = predictions_to_distance_table(predictions, class_names)
    compatibility_distances["distance_source"] = "fewshot_prototype_probability"
    for class_name in class_names:
        if f"prototype_distance_{class_name}" in predictions:
            compatibility_distances[f"prototype_distance_{class_name}"] = predictions[f"prototype_distance_{class_name}"].astype(float).to_numpy()
            compatibility_distances[f"nearest_prototype_{class_name}"] = predictions[f"nearest_prototype_{class_name}"].astype(str).to_numpy()
    compatibility_distances.to_csv(output_dir / "prototype_distances.tsv", sep="\t", index=False)
    try:
        embed_corpus.write_visualizations(
            embeddings,
            metadata,
            compatibility_distances,
            pd.DataFrame(),
            output_dir,
            tau=1.0 - float(selected_objectness_tau),
        )
    except Exception as exc:
        log.warning("Few-shot candidate-region visualization failed: %s", exc)

    checkpoint = {
        "model_type": "candidate_region_fewshot_prototype",
        "prototypes": torch.as_tensor(vectors, dtype=torch.float32),
        "prototype_class_names": [str(x) for x in proto_classes.tolist()],
        "prototype_names": [str(x) for x in proto_names.tolist()],
        "prototype_kinds": [str(x) for x in proto_kinds.tolist()],
        "class_names": class_names,
        "selected_objectness_tau": float(selected_objectness_tau),
        "type_thresholds": {name: float(value) for name, value in type_thresholds.items()},
        "subtype_targets": str(args.subtype_targets),
        "subtype_thresholding": str(args.subtype_thresholding),
        "subtype_thresholds": {
            name: {level: float(value) for level, value in levels.items()}
            for name, levels in subtype_thresholds.items()
        },
        "cluster_aggregation": str(args.cluster_aggregation),
        "threshold_calibration": threshold_calibration,
        "rescue_thresholding": str(args.rescue_thresholding),
        "selected_rescue_type_tau": None if selected_rescue_type_tau is None else float(selected_rescue_type_tau),
        "selected_rescue_objectness_floor": None if selected_rescue_objectness_floor is None else float(selected_rescue_objectness_floor),
        "selected_rescue_margin": None if selected_rescue_margin is None else float(selected_rescue_margin),
        "secondary_thresholding": str(args.secondary_thresholding),
        "selected_secondary_min": None if selected_secondary_min is None else float(selected_secondary_min),
        "selected_secondary_delta": None if selected_secondary_delta is None else float(selected_secondary_delta),
        "class_counts_train": _class_counts(_multi_hot_targets(metadata.loc[train_mask].reset_index(drop=True), class_names, subtype_targets=str(args.subtype_targets)), class_names),
        "class_counts_all": class_counts,
        "test_samples": test_samples,
        "embedding_features": selected_embedding_features,
        "embedding_dim": int(embeddings.shape[1]),
        "tabular_dim": int(tabular_features.shape[1]),
        "tabular_feature_names": tabular_feature_names,
        "prototype_temperature": float(args.prototype_temperature),
        "score_transform": str(args.score_transform),
        "tabular_weight": float(args.tabular_weight),
        "config": _jsonable_config(args),
    }
    torch.save(checkpoint, output_dir / "candidate_region_classifier_fewshot.pt")

    summary = {
        "model_type": "candidate_region_fewshot_prototype",
        "class_names": class_names,
        "class_counts_all": class_counts,
        "selected_objectness_tau": float(selected_objectness_tau),
        "type_thresholds": {name: float(value) for name, value in type_thresholds.items()},
        "subtype_targets": str(args.subtype_targets),
        "subtype_thresholding": str(args.subtype_thresholding),
        "subtype_thresholds": {
            name: {level: float(value) for level, value in levels.items()}
            for name, levels in subtype_thresholds.items()
        },
        "cluster_aggregation": str(args.cluster_aggregation),
        "threshold_calibration": threshold_calibration,
        "rescue_thresholding": str(args.rescue_thresholding),
        "selected_rescue_type_tau": None if selected_rescue_type_tau is None else float(selected_rescue_type_tau),
        "selected_rescue_objectness_floor": None if selected_rescue_objectness_floor is None else float(selected_rescue_objectness_floor),
        "selected_rescue_margin": None if selected_rescue_margin is None else float(selected_rescue_margin),
        "secondary_thresholding": str(args.secondary_thresholding),
        "selected_secondary_min": None if selected_secondary_min is None else float(selected_secondary_min),
        "selected_secondary_delta": None if selected_secondary_delta is None else float(selected_secondary_delta),
        "calibration_n_candidates": int(len(calibration_raw)),
        "calibration_samples": sorted(pd.unique(calibration_raw["held_out_sample"].astype(str)).tolist()) if "held_out_sample" in calibration_raw else [],
        "n_candidates": int(len(metadata)),
        "n_positive_candidates": int(labeled.sum()),
        "n_empty_candidates": int(background.sum()),
        "test_samples": test_samples,
        "required_test_sample": str(args.required_test_sample),
        "embedding_features": selected_embedding_features,
        "embedding_dim": int(embeddings.shape[1]),
        "tabular_dim": int(tabular_features.shape[1]),
        "tabular_feature_names": tabular_feature_names,
        "prototype_temperature": float(args.prototype_temperature),
        "score_transform": str(args.score_transform),
        "containing_prototypes": int(args.containing_prototypes),
        "tabular_weight": float(args.tabular_weight),
        "n_prototypes": int(len(final_prototypes)),
        "metrics": overall_metrics.to_dict("records") if not overall_metrics.empty else [],
        "config": _jsonable_config(args),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Wrote few-shot candidate-region classifier outputs to %s", output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="", help="Complex-SV manifest TSV")
    parser.add_argument("--candidate_regions", default="", help="merged_candidate_regions.csv from gen_candidates.py")
    parser.add_argument("--cn_checkpoint", default="")
    parser.add_argument("--graph_checkpoint", default=None)
    parser.add_argument("--embeddings_npz", default="", help="Optional existing embeddings.npz to reuse instead of embedding candidates")
    parser.add_argument("--metadata_tsv", default="", help="Optional candidate_embeddings.tsv matching --embeddings_npz")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_names", default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--required_test_sample", default="H1395,H1437,ME180,SCC154,ZR7530,SNU1245")
    parser.add_argument("--test_samples", default="")
    parser.add_argument("--n_test_samples", type=int, default=6)
    parser.add_argument("--reuse_embeddings", action="store_true")
    parser.add_argument("--embedding_features", choices=("full", "coords"), default="full")
    parser.add_argument("--embedding_normalization", choices=embed_corpus.EMBEDDING_NORMALIZATION_CHOICES, default="none")
    parser.add_argument("--sample_baseline_min_candidates", type=int, default=3)
    parser.add_argument("--tabular_features", default="safe")
    parser.add_argument("--tabular_weight", type=float, default=1.0)
    parser.add_argument("--final_l2_normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--containing_prototypes", type=int, default=1)
    parser.add_argument("--min_prototype_members", type=int, default=1)
    parser.add_argument("--min_cluster_members", type=int, default=2)
    parser.add_argument("--prototype_subtype_weighting", choices=("auto", "on", "off", "binary"), default="auto")
    parser.add_argument("--prototype_temperature", type=float, default=0.25)
    parser.add_argument("--score_transform", choices=("exp", "cosine_shift"), default="exp")
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=0.95)
    parser.add_argument("--tau_steps", type=int, default=91)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--tau_selection_metric", choices=("f1", "precision", "recall"), default="f1")
    parser.add_argument("--threshold_calibration", choices=("logo", "train"), default="logo")
    parser.add_argument("--threshold_tie_break", choices=("low", "high"), default="low")
    parser.add_argument("--type_tau_min", type=float, default=0.05)
    parser.add_argument("--type_tau_max", type=float, default=0.95)
    parser.add_argument("--type_tau_steps", type=int, default=91)
    parser.add_argument("--type_tau", type=float, default=None)
    parser.add_argument("--subtype_targets", choices=("general", "specific"), default="general")
    parser.add_argument("--subtype_thresholding", choices=("off", "optimize"), default="off")
    parser.add_argument("--cluster_aggregation", choices=("off", "max"), default="max")
    parser.add_argument("--rescue_thresholding", choices=("off", "fixed", "optimize"), default="optimize")
    parser.add_argument("--rescue_type_tau", type=float, default=0.85)
    parser.add_argument("--rescue_objectness_floor", type=float, default=0.0)
    parser.add_argument("--rescue_margin", type=float, default=0.0)
    parser.add_argument("--rescue_type_tau_min", type=float, default=0.60)
    parser.add_argument("--rescue_type_tau_max", type=float, default=0.98)
    parser.add_argument("--rescue_type_tau_steps", type=int, default=20)
    parser.add_argument("--rescue_objectness_floor_grid", default="0,0.001,0.005,0.01,0.02,0.05,0.10")
    parser.add_argument("--rescue_margin_min", type=float, default=0.0)
    parser.add_argument("--rescue_margin_max", type=float, default=0.30)
    parser.add_argument("--rescue_margin_steps", type=int, default=7)
    parser.add_argument("--rescue_min_recall", type=float, default=0.85)
    parser.add_argument("--rescue_min_precision", type=float, default=0.60)
    parser.add_argument("--rescue_max_empty_fp_rate", type=float, default=0.75)
    parser.add_argument("--secondary_thresholding", choices=("off", "fixed", "optimize"), default="optimize")
    parser.add_argument("--secondary_min", type=float, default=0.55)
    parser.add_argument("--secondary_delta", type=float, default=0.15)
    parser.add_argument("--secondary_min_min", type=float, default=0.30)
    parser.add_argument("--secondary_min_max", type=float, default=0.90)
    parser.add_argument("--secondary_min_steps", type=int, default=13)
    parser.add_argument("--secondary_delta_min", type=float, default=0.0)
    parser.add_argument("--secondary_delta_max", type=float, default=0.50)
    parser.add_argument("--secondary_delta_steps", type=int, default=11)
    parser.add_argument("--secondary_min_recall", type=float, default=0.85)
    parser.add_argument("--secondary_min_precision", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    main()
