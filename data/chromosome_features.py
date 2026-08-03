#!/usr/bin/env python3
"""Whole-chromosome Wakhan and Severus feature construction."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

TCN_BINS = [
    ("n_TCN_0", 0, 0), ("n_TCN_1", 1, 1), ("n_TCN_2", 2, 2),
    ("n_TCN_3", 3, 3), ("n_TCN_4", 4, 4), ("n_TCN_5", 5, 5),
    ("n_TCN_6", 6, 6), ("n_TCN_7", 7, 7), ("n_TCN_7_10", 8, 10),
    ("n_TCN_11_20", 11, 20), ("n_TCN_20_40", 21, 40),
]

def oscillation(cn: np.ndarray) -> tuple[float, float, float]:
    if len(cn) < 2:
        return 0.0, 0.0, 0.0
    rounded = np.rint(cn).astype(int)
    top = set(pd.Series(rounded).value_counts().head(2).index)
    fraction = float(np.mean([value in top for value in rounded]))
    transitions = rounded[1:] != rounded[:-1]
    alternating = [transitions[i] and rounded[i - 1] == rounded[i + 1]
                   for i in range(1, len(rounded) - 1)]
    transition_fraction = float(np.mean(transitions))
    two_state = float(len(top) == 2 and fraction >= 0.8 and
                      (np.mean(alternating) if alternating else 0) >= 0.5)
    return two_state, fraction, transition_fraction

def sv_types(row: pd.Series) -> set[str]:
    values: set[str] = set()
    sv_type = str(row.get("sv_type_str", "")).upper()
    bnd_type = str(row.get("bnd_type", "")).upper()
    if sv_type == "DEL" or "DEL_LIKE" in bnd_type: values.add("DEL")
    if sv_type == "DUP" or "DUP_LIKE" in bnd_type: values.add("DUP")
    if float(row.get("is_foldback_like", 0) or 0) > 0 or float(row.get("is_foldback", 0) or 0) > 0: values.add("FB")
    if float(row.get("has_interchrom_mate", 0) or 0) > 0: values.add("INTER_CHR")
    if float(row.get("is_inv_like_bnd", 0) or 0) > 0 or sv_type == "INV": values.add("INV_LIKE")
    if sv_type == "SBND": values.add("sBND")
    return values

def summarize(sample_id: str, chrom: str, cn: pd.DataFrame, sv: pd.DataFrame) -> dict:
    cn, sv = cn.sort_values("start").copy(), sv.sort_values("pos").copy()
    starts = pd.to_numeric(cn.start, errors="coerce").fillna(0).to_numpy()
    ends = pd.to_numeric(cn.end, errors="coerce").fillna(0).to_numpy()
    lengths = np.maximum(ends - starts, 1)
    total_cn = pd.to_numeric(cn.cn_total, errors="coerce").fillna(0).to_numpy()
    span = max(int(ends.max() - starts.min()), 1)
    two_state, osc_fraction, transition_fraction = oscillation(total_cn)
    row = {
        "sample_id": sample_id, "chrom": chrom, "n_windows": 1,
        "n_segments": len(cn), "n_segments_ge_100kb": int((lengths >= 100_000).sum()),
        "component_length": span, "segment_frequency": len(cn) / span,
        "ploidy": float(np.average(total_cn, weights=lengths)),
        "segment_len_q25": float(np.quantile(lengths, .25)),
        "segment_len_q50": float(np.quantile(lengths, .50)),
        "segment_len_q75": float(np.quantile(lengths, .75)),
        "oscillating_two_state": two_state,
        "oscillating_segment_fraction": osc_fraction,
        "oscillating_transition_fraction": transition_fraction,
        "n_breakpoints": len(sv),
    }
    rounded = np.rint(total_cn).astype(int)
    for name, low, high in TCN_BINS:
        row[name] = int(((rounded >= low) & (rounded <= high)).sum())
    row["n_TCN_gt_40"] = int((rounded > 40).sum())
    types = [sv_types(sv_row) for _, sv_row in sv.iterrows()]
    for name in ["DEL", "DUP", "FB", "INTER_CHR", "INV_LIKE", "sBND"]:
        row[f"n_{name}"] = int(sum(name in value for value in types))
    fb = [i + 1 for i, value in enumerate(types) if "FB" in value]
    row.update({"FB_first_index": min(fb) if fb else 0,
                "FB_last_index": max(fb) if fb else 0,
                "n_FB_lowCN_2Mb": 0, "FB_lowCN_first_index": 0,
                "FB_lowCN_last_index": 0})
    inter = pd.to_numeric(sv["has_interchrom_mate"], errors="coerce").fillna(0) > 0 if len(sv) else pd.Series(dtype=bool)
    row["n_interchromosomal_SV"] = int(sv.loc[inter, "sv_id"].nunique()) if len(sv) else 0
    return row


def assemble_model_inputs(
    embedding_dir: str | Path,
    tabular_path: str | Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Assemble the 1,254-dimensional whole-chromosome model input."""
    from model.chromosome import SAFE_FEATURES, canonical_chrom

    embedding_dir = Path(embedding_dir)
    bundle = np.load(embedding_dir / "embeddings.npz", allow_pickle=True)
    base = np.asarray(bundle["embeddings"], dtype=np.float32)
    metadata = pd.read_csv(
        embedding_dir / "candidate_embeddings.tsv", sep="\t"
    ).fillna("")
    if len(metadata) != len(base) or base.shape[1] != 402:
        raise ValueError(
            f"Expected matching metadata and 402D embeddings; "
            f"got {len(metadata)} rows and {base.shape}"
        )
    metadata["chrom"] = metadata["chrom"].map(canonical_chrom)
    if metadata.duplicated(["sample_id", "chrom"]).any():
        raise ValueError("Expected exactly one embedding per sample/chromosome")

    spans = (
        pd.to_numeric(metadata["end_bp"], errors="coerce").fillna(0).to_numpy()
        - pd.to_numeric(metadata["start_bp"], errors="coerce").fillna(0).to_numpy()
    ).clip(min=1)
    coordinates = np.column_stack([
        np.zeros(len(metadata)), np.ones(len(metadata)),
        np.full(len(metadata), 0.5), np.ones(len(metadata)),
        np.minimum(np.log1p(spans / 1_000_000.0) / 5.0, 1.0),
        np.minimum(np.log1p(spans / 1_000_000.0) / 6.0, 1.0),
        np.zeros(len(metadata)), np.ones(len(metadata)),
    ]).astype(np.float32)
    embedding_features = np.concatenate(
        [base, base, np.zeros_like(base), coordinates], axis=1
    )

    tabular = pd.read_csv(tabular_path, sep="\t").fillna(0)
    tabular["chrom"] = tabular["chrom"].map(canonical_chrom)
    tabular = tabular.set_index(["sample_id", "chrom"])
    tabular_rows = []
    for row in metadata.itertuples(index=False):
        values = tabular.loc[(str(row.sample_id), str(row.chrom))]
        tabular_rows.append([
            float(pd.to_numeric(values.get(name, 0), errors="coerce") or 0)
            for name in SAFE_FEATURES
        ])
    features = np.concatenate([
        embedding_features,
        np.asarray(tabular_rows, dtype=np.float32),
        np.zeros((len(metadata), 3), dtype=np.float32),
    ], axis=1)
    if features.shape[1] != 1254:
        raise ValueError(f"Expected 1,254 model inputs, found {features.shape[1]}")
    return features, metadata
