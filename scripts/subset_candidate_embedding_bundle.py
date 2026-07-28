#!/usr/bin/env python3
"""Subset a prepared candidate embedding bundle by candidate ID."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ["ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"]


def has_label(value: object) -> bool:
    return str(value).strip().lower() not in {"", "0", "false", "none", "nan", "<na>"}


def replace_labels(metadata: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    out = metadata.copy()
    for class_name in CLASS_NAMES:
        out[class_name] = candidates.get(class_name, pd.Series([""] * len(candidates))).astype(str).to_numpy()
    classes = [
        ";".join(name for name in CLASS_NAMES if has_label(row.get(name, "")))
        for row in candidates.to_dict("records")
    ]
    out["sv_class"] = classes
    out["sv_classes"] = classes
    out["raw_sv_classes"] = classes
    positive = pd.Series(classes).ne("").to_numpy()
    out["evidence"] = np.where(positive, "candidate_region_label", "candidate_region_empty")
    out["label_scope"] = np.where(positive, "region", "empty_candidate_region")
    out["candidate_scope"] = "candidate_region"
    out["label_id"] = np.where(positive, out["candidate_id"].astype(str), "")
    return out


def subset_npz(source: Path, output: Path, indices: np.ndarray, source_rows: int) -> None:
    data = np.load(source, allow_pickle=True)
    payload: dict[str, np.ndarray] = {}
    for key in data.files:
        value = np.asarray(data[key])
        payload[key] = value[indices] if value.ndim > 0 and value.shape[0] == source_rows else value
    np.savez(output, **payload)


def run(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.candidate_regions).fillna("")
    metadata = pd.read_csv(source_dir / "candidate_embeddings.tsv", sep="\t").fillna("")
    source_ids = metadata["candidate_id"].astype(str).tolist()
    index_by_id = {candidate_id: index for index, candidate_id in enumerate(source_ids)}
    target_ids = targets["candidate_id"].astype(str).tolist()
    missing = [candidate_id for candidate_id in target_ids if candidate_id not in index_by_id]
    if missing:
        raise ValueError(f"{len(missing)} target candidate IDs are absent from source embeddings; examples={missing[:8]}")
    indices = np.asarray([index_by_id[candidate_id] for candidate_id in target_ids], dtype=int)
    subset_metadata = metadata.iloc[indices].reset_index(drop=True)
    if subset_metadata["candidate_id"].astype(str).tolist() != target_ids:
        raise RuntimeError("Candidate order mismatch after subsetting")
    subset_metadata = replace_labels(subset_metadata, targets)
    subset_metadata.to_csv(output_dir / "candidate_embeddings.tsv", sep="\t", index=False)
    targets.to_csv(output_dir / "candidate_regions_from_csv.tsv", sep="\t", index=False)

    for name in ["embeddings.npz", "selected_embedding_features.npz", "tabular_features.npz"]:
        source = source_dir / name
        if source.exists():
            subset_npz(source, output_dir / name, indices, len(metadata))
    for name in ["embedding_features.txt", "tabular_feature_names.txt"]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)

    source_summary = source_dir / "training_summary.json"
    summary = json.loads(source_summary.read_text()) if source_summary.exists() else {}
    summary["n_candidates"] = int(len(targets))
    summary["subset_source_dir"] = str(source_dir.resolve())
    summary["subset_candidate_regions"] = str(Path(args.candidate_regions).resolve())
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    config = dict(summary.get("config", {}))
    config["candidate_regions"] = str(Path(args.candidate_regions).resolve())
    config["output_dir"] = str(output_dir.resolve())
    summary["config"] = config
    print(f"Subset {len(targets)} of {len(metadata)} candidate embeddings into {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_dir", required=True)
    parser.add_argument("--candidate_regions", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
