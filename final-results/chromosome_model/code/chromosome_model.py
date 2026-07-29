#!/usr/bin/env python3
"""Train/evaluate the Pipeline27 whole-chromosome multi-label classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
SAFE_FEATURES = [
    "n_windows", "n_segments", "n_segments_ge_100kb", "component_length",
    "segment_frequency", "ploidy", "segment_len_q25", "segment_len_q50",
    "segment_len_q75", "oscillating_two_state", "oscillating_segment_fraction",
    "oscillating_transition_fraction", "n_TCN_0", "n_TCN_1", "n_TCN_2",
    "n_TCN_3", "n_TCN_4", "n_TCN_5", "n_TCN_6", "n_TCN_7",
    "n_TCN_7_10", "n_TCN_11_20", "n_TCN_20_40", "n_TCN_gt_40",
    "n_breakpoints", "n_FB_lowCN_2Mb", "FB_first_index", "FB_last_index",
    "FB_lowCN_first_index", "FB_lowCN_last_index", "n_DEL", "n_DUP", "n_FB",
    "n_INTER_CHR", "n_INV_LIKE", "n_sBND", "n_interchromosomal_SV",
]


class ChromosomeModel(nn.Module):
    """Pipeline24 MLP head, applied once per chromosome."""

    def __init__(self, input_dim: int = 1254, hidden: int = 96, dropout: float = 0.35):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def canonical_chrom(value: object) -> str:
    text = str(value).strip()
    return text if text.startswith("chr") else f"chr{text}"


def load_dataset() -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    emb_dir = HERE / "chromosome_embeddings"
    bundle = np.load(emb_dir / "embeddings.npz", allow_pickle=True)
    embeddings = bundle["embeddings"].astype(np.float32)
    metadata = pd.read_csv(emb_dir / "candidate_embeddings.tsv", sep="\t").fillna("")
    if len(metadata) != len(embeddings):
        raise ValueError("embedding/metadata row mismatch")
    metadata["chrom"] = metadata["chrom"].map(canonical_chrom)
    tabular_table = pd.read_csv(HERE / "chromosome_tabular.tsv", sep="\t").fillna(0)
    tabular_table["chrom"] = tabular_table["chrom"].map(canonical_chrom)
    tabular_table = tabular_table.set_index(["sample_id", "chrom"])

    labels = pd.read_csv(HERE / "chromosome_labels.tsv", sep="\t")
    labels["chrom"] = labels["chrom"].map(canonical_chrom)
    target_map: dict[tuple[str, str], set[str]] = {}
    for row in labels.itertuples(index=False):
        target_map.setdefault((str(row.sample_id), str(row.chrom)), set()).add(str(row.label))

    # Positive chromosomes have one duplicate anchor per class. Their embeddings are
    # identical by construction, so retain exactly one row per sample/chromosome.
    keep_rows: list[int] = []
    records: list[dict] = []
    for (sample, chrom), group in metadata.groupby(["sample_id", "chrom"], sort=True):
        idx = int(group.index[0])
        if len(group) != 1:
            raise ValueError(f"expected one chromosome embedding: {sample}/{chrom}")
        classes = target_map.get((str(sample), str(chrom)), set())
        record = group.iloc[0].to_dict()
        record["sample_id"] = str(sample)
        record["chrom"] = str(chrom)
        record["sv_classes"] = ";".join(c for c in CLASSES if c in classes)
        record["is_labeled_chromosome"] = bool(classes)
        keep_rows.append(idx)
        records.append(record)

    meta = pd.DataFrame(records).reset_index(drop=True)
    base = embeddings[np.asarray(keep_rows)]
    if base.shape[1] != 402:
        raise ValueError(f"expected 402-dimensional base embeddings, found {base.shape[1]}")
    spans = (
        pd.to_numeric(meta["end_bp"], errors="coerce").fillna(0).to_numpy()
        - pd.to_numeric(meta["start_bp"], errors="coerce").fillna(0).to_numpy()
    ).clip(min=1)
    coordinates = np.column_stack([
        np.zeros(len(meta)), np.ones(len(meta)), np.full(len(meta), 0.5),
        np.ones(len(meta)),
        np.minimum(np.log1p(spans / 1_000_000.0) / 5.0, 1.0),
        np.minimum(np.log1p(spans / 1_000_000.0) / 6.0, 1.0),
        np.zeros(len(meta)), np.ones(len(meta)),
    ]).astype(np.float32)
    # Pipeline24 full representation: local, chromosome context,
    # local-minus-context, and eight relative-coordinate features.
    emb = np.concatenate([base, base, np.zeros_like(base), coordinates], axis=1)

    tabular_rows = []
    for row in meta.itertuples(index=False):
        key = (str(row.sample_id), str(row.chrom))
        if key not in tabular_table.index:
            raise KeyError(f"missing chromosome summary for {key}")
        values = tabular_table.loc[key]
        tabular_rows.append([
            float(pd.to_numeric(values.get(name, 0), errors="coerce") or 0)
            for name in SAFE_FEATURES
        ])
    tabular = np.asarray(tabular_rows, dtype=np.float32)
    # Preserve Pipeline24's 1,254 input contract without using proposal features.
    neutral_proposal_slots = np.zeros((len(meta), 3), dtype=np.float32)
    x = np.concatenate([emb, tabular, neutral_proposal_slots], axis=1)
    if x.shape[1] != 1254:
        raise ValueError(f"expected 1,254 inputs, found {x.shape[1]}")

    y = np.zeros((len(meta), len(CLASSES)), dtype=np.float32)
    for i, value in enumerate(meta["sv_classes"]):
        observed = set(str(value).split(";")) if str(value) else set()
        for j, class_name in enumerate(CLASSES):
            y[i, j] = float(class_name in observed)
    return x, meta, y


def grouped_folds(meta: pd.DataFrame, y: np.ndarray, n_folds: int = 5) -> dict[str, int]:
    sample_names = sorted(meta.sample_id.unique())
    sample_vectors = {
        sample: y[meta.sample_id.to_numpy() == sample].sum(axis=0)
        for sample in sample_names
    }
    labeled = [s for s in sample_names if sample_vectors[s].sum() > 0]
    labeled.sort(key=lambda s: (-float(sample_vectors[s].sum()), s))
    fold_counts = np.zeros((n_folds, len(CLASSES)), dtype=float)
    fold_sizes = np.zeros(n_folds, dtype=int)
    assignment: dict[str, int] = {}
    target = sum((sample_vectors[s] for s in labeled), np.zeros(len(CLASSES))) / n_folds
    for sample in labeled:
        vector = sample_vectors[sample]
        scores = []
        for fold in range(n_folds):
            after = fold_counts[fold] + vector
            imbalance = float(np.square((after - target) / np.maximum(target, 1)).sum())
            # Enforce balanced genome counts first, then optimize class balance.
            scores.append((fold_sizes[fold], imbalance, fold))
        fold = min(scores)[2]
        assignment[sample] = int(fold)
        fold_counts[fold] += vector
        fold_sizes[fold] += 1
    return assignment


def validation_samples(
    meta: pd.DataFrame,
    y: np.ndarray,
    available: list[str],
    target_per_class: int = 5,
) -> list[str]:
    sample_vectors = {
        sample: y[meta.sample_id.to_numpy() == sample].sum(axis=0).astype(int)
        for sample in available
    }
    total = sum(sample_vectors.values(), np.zeros(len(CLASSES), dtype=int))
    target = np.minimum(total, target_per_class)
    selected: list[str] = []
    counts = np.zeros(len(CLASSES), dtype=int)
    while np.any(counts < target):
        remaining = [s for s in available if s not in selected]
        if not remaining:
            break
        deficit = np.maximum(target - counts, 0)
        ranked = []
        for sample in remaining:
            gain = np.minimum(sample_vectors[sample], deficit)
            score = float((gain / np.maximum(target, 1)).sum())
            ranked.append((-score, -int(gain.sum()), sample))
        _, neg_gain, best = min(ranked)
        if neg_gain == 0:
            break
        selected.append(best)
        counts += sample_vectors[best]
    if not selected:
        selected = available[: max(1, min(5, len(available)))]
    return selected


def standardize(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0).astype(np.float32)
    std = x[train_mask].std(axis=0).astype(np.float32)
    std[std < 1e-5] = 1.0
    return np.clip((x - mean) / std, -8, 8).astype(np.float32), mean, std


def fbeta_counts(y_true: np.ndarray, y_pred: np.ndarray, beta: float) -> tuple[float, int, int, int]:
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    b2 = beta * beta
    score = (1 + b2) * tp / max((1 + b2) * tp + b2 * fn + fp, 1)
    return float(score), tp, fp, fn


def calibrate_thresholds(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(len(CLASSES), 0.5, dtype=np.float32)
    grid = np.linspace(0.02, 0.98, 97)
    for j in range(len(CLASSES)):
        best = (-1.0, -1.0, 0.5)
        for threshold in grid:
            pred = probabilities[:, j] >= threshold
            f2, tp, fp, fn = fbeta_counts(y[:, j], pred, beta=2)
            precision = tp / max(tp + fp, 1)
            candidate = (f2, precision, -float(threshold))
            if candidate > best:
                best = candidate
                thresholds[j] = threshold
    return thresholds


def train_model(
    x: np.ndarray,
    meta: pd.DataFrame,
    y: np.ndarray,
    train_samples: list[str],
    validation: list[str],
    seed: int,
    unlabeled_weight: float,
) -> tuple[ChromosomeModel, dict, pd.DataFrame]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "2"))))

    samples = meta.sample_id.astype(str).to_numpy()
    train_mask = np.isin(samples, train_samples)
    val_mask = np.isin(samples, validation)
    x_norm, mean, std = standardize(x, train_mask)
    xt = torch.as_tensor(x_norm[train_mask])
    yt = torch.as_tensor(y[train_mask])
    xv = torch.as_tensor(x_norm[val_mask])
    yv = torch.as_tensor(y[val_mask])

    train_any = y[train_mask].sum(axis=1) > 0
    element_weights = np.full(yt.shape, float(unlabeled_weight), dtype=np.float32)
    # Missing classes on an annotated chromosome are more credible negatives,
    # but still not treated as perfectly complete labels.
    element_weights[train_any, :] = 0.25
    element_weights[y[train_mask] > 0] = 1.0
    positive = y[train_mask].sum(axis=0)
    effective_negative = ((1 - y[train_mask]) * element_weights).sum(axis=0)
    positive_scale = np.sqrt(effective_negative / np.maximum(positive, 1.0))
    positive_scale = np.clip(positive_scale, 1.0, 8.0)
    for j in range(len(CLASSES)):
        element_weights[y[train_mask, j] > 0, j] *= positive_scale[j]
    wt = torch.as_tensor(element_weights)

    val_any = y[val_mask].sum(axis=1) > 0
    val_weights = np.full(yv.shape, float(unlabeled_weight), dtype=np.float32)
    val_weights[val_any, :] = 0.25
    val_weights[y[val_mask] > 0] = 1.0
    wv = torch.as_tensor(val_weights)

    model = ChromosomeModel(input_dim=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-3)
    best_state = None
    best_loss = math.inf
    bad_epochs = 0
    history = []
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        logits = model(xt)
        raw = F.binary_cross_entropy_with_logits(logits, yt, reduction="none")
        loss = (raw * wt).sum() / wt.sum().clamp_min(1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_raw = F.binary_cross_entropy_with_logits(model(xv), yv, reduction="none")
            val_loss = float((val_raw * wv).sum() / wv.sum().clamp_min(1))
        history.append({"epoch": epoch + 1, "train_loss": float(loss), "validation_loss": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= 35:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {"mean": mean, "std": std, "best_validation_loss": best_loss}, pd.DataFrame(history)


def evaluate(args: argparse.Namespace) -> None:
    x, meta, y = load_dataset()
    labeled_samples = sorted(
        meta.loc[y.sum(axis=1) > 0, "sample_id"].astype(str).unique()
    )
    all_samples = sorted(meta.sample_id.astype(str).unique())
    folds = grouped_folds(meta, y)
    pd.DataFrame(
        [{"sample_id": sample, "fold": fold} for sample, fold in sorted(folds.items())]
    ).to_csv(HERE / "fivefold_assignments.tsv", sep="\t", index=False)

    if args.split == "loo":
        if args.test_sample not in labeled_samples:
            raise ValueError(f"unknown labeled test sample: {args.test_sample}")
        test_samples = [args.test_sample]
        run_name = args.test_sample
    else:
        test_samples = sorted(s for s, fold in folds.items() if fold == args.fold)
        run_name = f"fold_{args.fold}"

    calibration_pool = [s for s in labeled_samples if s not in test_samples]
    validation = validation_samples(meta, y, calibration_pool, target_per_class=5)
    # Label-free genomes are retained as weak background in every training split.
    train_samples = [s for s in all_samples if s not in test_samples and s not in validation]
    model, norm, history = train_model(
        x, meta, y, train_samples, validation,
        seed=27 + (args.fold if args.split == "grouped5" else labeled_samples.index(args.test_sample)),
        unlabeled_weight=args.unlabeled_weight,
    )

    x_norm = np.clip((x - norm["mean"]) / norm["std"], -8, 8).astype(np.float32)
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(torch.as_tensor(x_norm))).numpy()
    val_mask = meta.sample_id.astype(str).isin(validation).to_numpy()
    test_mask = meta.sample_id.astype(str).isin(test_samples).to_numpy()
    thresholds = calibrate_thresholds(y[val_mask], probabilities[val_mask])
    predictions = probabilities[test_mask] >= thresholds

    rows = []
    test_meta = meta.loc[test_mask].reset_index(drop=True)
    for i, row in test_meta.iterrows():
        for j, class_name in enumerate(CLASSES):
            rows.append({
                "split": args.split,
                "run": run_name,
                "sample_id": row["sample_id"],
                "chrom": row["chrom"],
                "class": class_name,
                "probability": float(probabilities[test_mask][i, j]),
                "threshold": float(thresholds[j]),
                "predicted": int(predictions[i, j]),
                "truth": int(y[test_mask][i, j]),
            })
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    prediction_table = pd.DataFrame(rows)
    prediction_table.to_csv(out / "predictions.tsv", sep="\t", index=False)
    history.to_csv(out / "training_history.tsv", sep="\t", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": x.shape[1],
        "hidden_dim": 96,
        "dropout": 0.35,
        "classes": CLASSES,
        "mean": norm["mean"],
        "std": norm["std"],
        "thresholds": thresholds,
        "train_samples": train_samples,
        "validation_samples": validation,
        "test_samples": test_samples,
        "unlabeled_weight": args.unlabeled_weight,
    }
    torch.save(checkpoint, out / "model.pt")
    summary = {
        "split": args.split,
        "run": run_name,
        "n_train_samples": len(train_samples),
        "n_validation_samples": len(validation),
        "n_test_samples": len(test_samples),
        "train_samples": train_samples,
        "validation_samples": validation,
        "test_samples": test_samples,
        "thresholds": dict(zip(CLASSES, map(float, thresholds))),
        "best_validation_loss": norm["best_validation_loss"],
        "unlabeled_weight": args.unlabeled_weight,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["loo", "grouped5"], required=True)
    parser.add_argument("--test-sample")
    parser.add_argument("--fold", type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--unlabeled-weight", type=float, default=0.10)
    args = parser.parse_args()
    if args.split == "loo" and not args.test_sample:
        parser.error("--test-sample is required for loo")
    if args.split == "grouped5" and args.fold is None:
        parser.error("--fold is required for grouped5")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
