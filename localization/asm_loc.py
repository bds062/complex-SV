"""ASM-Loc-inspired chromosome localization for complex structural variants.

The genomic sequence analogue of a video snippet is a fixed-width genomic bin.
The model combines a local convolution (nearby-bin structure), Transformer
attention (longer chromosome context), class/foreground heads, and a boundary
distance head.  It is intentionally independent of the expensive pretrained
candidate encoders: the localizer cheaply scans a genome, then pipeline18 embeds
and classifies only the resulting proposals.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


CLASS_NAMES = ("ecDNA", "chromothripsis", "BFB")
FEATURE_NAMES = (
    "cn_mean",
    "cn_max",
    "cn_min",
    "cn_std",
    "cn_change_count",
    "cn_segment_count",
    "cn_small_segment_count",
    "cn_high_fraction",
    "cn_loh_fraction",
    "bp_count",
    "bp_unique_sv_count",
    "bp_del_count",
    "bp_dup_count",
    "bp_fb_count",
    "bp_inter_count",
    "bp_inv_like_count",
    "bp_sbnd_count",
    "bp_support_sum",
)
COUNT_FEATURES = {
    "cn_change_count",
    "cn_segment_count",
    "cn_small_segment_count",
    "bp_count",
    "bp_unique_sv_count",
    "bp_del_count",
    "bp_dup_count",
    "bp_fb_count",
    "bp_inter_count",
    "bp_inv_like_count",
    "bp_sbnd_count",
    "bp_support_sum",
}


@dataclass
class ModelConfig:
    input_dim: int = len(FEATURE_NAMES)
    num_classes: int = len(CLASS_NAMES)
    hidden_dim: int = 96
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.15
    max_boundary_bins: int = 64


@dataclass
class TrainConfig:
    epochs: int = 35
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 16
    chunk_bins: int = 256
    chunk_overlap_bins: int = 64
    short_event_boost: float = 3.0
    boundary_loss_weight: float = 0.5
    patience: int = 7
    seed: int = 17


def chromosome_key(value: object) -> tuple[int, int | str]:
    text = str(value).removeprefix("chr")
    if text.isdigit():
        return (0, int(text))
    return (1, {"X": 23, "Y": 24, "M": 25, "MT": 25}.get(text, text))


def interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    """Half-open interval overlap length."""
    return max(0, min(int(end_a), int(end_b)) - max(int(start_a), int(start_b)))


def _weighted_mean(values: np.ndarray, weights: np.ndarray, default: float = 0.0) -> float:
    total = float(weights.sum())
    return float(np.dot(values, weights) / total) if total > 0 else float(default)


def _cn_features(chrom_cna: pd.DataFrame, start: int, end: int, high_cn: float) -> dict[str, float]:
    overlap = chrom_cna[(chrom_cna["start"] < end) & (chrom_cna["end"] > start)]
    if overlap.empty:
        return {name: 0.0 for name in FEATURE_NAMES if name.startswith("cn_")}
    left = np.maximum(overlap["start"].to_numpy(dtype=np.int64), start)
    right = np.minimum(overlap["end"].to_numpy(dtype=np.int64), end)
    lengths = np.maximum(right - left, 0).astype(np.float64)
    tcn = pd.to_numeric(overlap["TCN"], errors="coerce").fillna(2.0).to_numpy(dtype=np.float64)
    cn1 = pd.to_numeric(overlap.get("CN1", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    mean = _weighted_mean(tcn, lengths, 2.0)
    variance = _weighted_mean((tcn - mean) ** 2, lengths)
    original_lengths = (
        pd.to_numeric(overlap["end"], errors="coerce")
        - pd.to_numeric(overlap["start"], errors="coerce")
    ).clip(lower=0)
    ordered_tcn = tcn[np.argsort(left)]
    return {
        "cn_mean": mean,
        "cn_max": float(np.max(tcn)),
        "cn_min": float(np.min(tcn)),
        "cn_std": float(math.sqrt(max(variance, 0.0))),
        "cn_change_count": float(np.count_nonzero(np.diff(ordered_tcn))),
        "cn_segment_count": float(len(overlap)),
        "cn_small_segment_count": float((original_lengths < 2_000_000).sum()),
        "cn_high_fraction": _weighted_mean((tcn >= high_cn).astype(float), lengths),
        "cn_loh_fraction": _weighted_mean((cn1 <= 0).astype(float), lengths),
    }


def _bp_features(chrom_bps: pd.DataFrame, start: int, end: int) -> dict[str, float]:
    overlap = chrom_bps[(chrom_bps["pos"] >= start) & (chrom_bps["pos"] < end)]
    types = overlap.get("SV_TYPE", pd.Series(dtype=str)).astype(str)
    return {
        "bp_count": float(len(overlap)),
        "bp_unique_sv_count": float(overlap.get("sv_id", pd.Series(dtype=str)).nunique()),
        "bp_del_count": float(types.eq("DEL").sum()),
        "bp_dup_count": float(types.eq("DUP").sum()),
        "bp_fb_count": float(types.eq("FB").sum()),
        "bp_inter_count": float(types.eq("INTER_CHR").sum()),
        "bp_inv_like_count": float(types.eq("INV_LIKE").sum()),
        "bp_sbnd_count": float(types.eq("sBND").sum()),
        "bp_support_sum": float(pd.to_numeric(overlap.get("supp", 0), errors="coerce").fillna(0).sum()),
    }


def build_sample_bins(
    sample_id: str,
    cna: pd.DataFrame,
    breakpoints: pd.DataFrame,
    bin_size: int = 1_000_000,
    high_copy_ratio: float = 2.0,
    high_copy_floor: float = 6.0,
) -> pd.DataFrame:
    """Convert parsed CNA/SV tables into one ordered feature row per genomic bin."""
    if cna.empty:
        return pd.DataFrame()
    lengths = (cna["end"] - cna["start"]).clip(lower=0).astype(float)
    sample_ploidy = _weighted_mean(
        pd.to_numeric(cna["TCN"], errors="coerce").fillna(2.0).to_numpy(float),
        lengths.to_numpy(float),
        2.0,
    )
    high_cn = max(float(high_copy_floor), sample_ploidy * float(high_copy_ratio))
    rows: list[dict] = []
    for chrom in sorted(cna["chrom"].astype(str).unique(), key=chromosome_key):
        chrom_cna = cna[cna["chrom"].astype(str).eq(chrom)].sort_values("start")
        chrom_bps = breakpoints[breakpoints["chrom"].astype(str).eq(chrom)]
        chrom_start = max(0, int(chrom_cna["start"].min()))
        chrom_end = int(chrom_cna["end"].max()) + 1
        first_bin = (chrom_start // bin_size) * bin_size
        for start in range(first_bin, chrom_end, bin_size):
            end = min(start + bin_size, chrom_end)
            row = {
                "sample_id": str(sample_id),
                "chrom": chrom,
                "bin_index": int(start // bin_size),
                "start": int(start),
                "end": int(end),
            }
            row.update(_cn_features(chrom_cna, start, end, high_cn))
            row.update(_bp_features(chrom_bps, start, end))
            rows.append(row)
    return pd.DataFrame(rows)


def add_targets(
    bins: pd.DataFrame,
    regions: pd.DataFrame,
    class_names: Iterable[str] = CLASS_NAMES,
    max_boundary_bins: int = 64,
    bin_size: int = 1_000_000,
) -> pd.DataFrame:
    """Attach foreground, class, and normalized left/right boundary targets."""
    out = bins.copy()
    classes = list(class_names)
    out["foreground_target"] = 0.0
    out["left_boundary_target"] = 0.0
    out["right_boundary_target"] = 0.0
    out["target_event_bins"] = 0
    for name in classes:
        out[f"target_{name}"] = 0.0

    grouped_regions = {
        (str(sample), str(chrom)): frame
        for (sample, chrom), frame in regions.groupby(
            [regions["sample_id"].astype(str), regions["chrom"].astype(str)], sort=False
        )
    }
    for key, index in out.groupby(
        [out["sample_id"].astype(str), out["chrom"].astype(str)], sort=False
    ).groups.items():
        calls = grouped_regions.get(key)
        if calls is None:
            continue
        for row_i in index:
            b_start, b_end = int(out.at[row_i, "start"]), int(out.at[row_i, "end"])
            best_overlap = 0
            best_call = None
            for call in calls.to_dict("records"):
                overlap = interval_overlap(b_start, b_end, int(call["start"]), int(call["end"]) + 1)
                if overlap > best_overlap:
                    best_overlap, best_call = overlap, call
                if overlap > 0 and str(call["label"]) in classes:
                    out.at[row_i, f"target_{call['label']}"] = max(
                        float(out.at[row_i, f"target_{call['label']}"]),
                        overlap / max(1, b_end - b_start),
                    )
            if best_call is None:
                continue
            event_start = int(best_call["start"])
            event_end = int(best_call["end"]) + 1
            out.at[row_i, "foreground_target"] = best_overlap / max(1, b_end - b_start)
            out.at[row_i, "left_boundary_target"] = np.clip(
                (b_start - event_start) / bin_size / max_boundary_bins, 0.0, 1.0
            )
            out.at[row_i, "right_boundary_target"] = np.clip(
                (event_end - b_end) / bin_size / max_boundary_bins, 0.0, 1.0
            )
            out.at[row_i, "target_event_bins"] = max(1, math.ceil((event_end - event_start) / bin_size))
    return out


class ChromosomeChunkDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_names: list[str],
        class_names: list[str],
        chunk_bins: int,
        overlap_bins: int,
        include_targets: bool = True,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.feature_names = feature_names
        self.class_names = class_names
        self.chunk_bins = int(chunk_bins)
        self.include_targets = include_targets
        self.chunks: list[np.ndarray] = []
        step = max(1, int(chunk_bins) - int(overlap_bins))
        for _, indices in self.frame.groupby(["sample_id", "chrom"], sort=False).groups.items():
            ordered = np.asarray(list(indices), dtype=np.int64)
            for offset in range(0, len(ordered), step):
                chunk = ordered[offset : offset + chunk_bins]
                self.chunks.append(chunk)
                if offset + chunk_bins >= len(ordered):
                    break

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        indices = self.chunks[item]
        rows = self.frame.iloc[indices]
        length = len(rows)
        x = np.zeros((self.chunk_bins, len(self.feature_names)), dtype=np.float32)
        x[:length] = rows[self.feature_names].to_numpy(dtype=np.float32)
        mask = np.zeros(self.chunk_bins, dtype=bool)
        mask[:length] = True
        result = {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "row_index": torch.from_numpy(
                np.pad(indices, (0, self.chunk_bins - length), constant_values=-1)
            ),
        }
        if self.include_targets:
            fg = np.zeros(self.chunk_bins, dtype=np.float32)
            classes = np.zeros((self.chunk_bins, len(self.class_names)), dtype=np.float32)
            boundary = np.zeros((self.chunk_bins, 2), dtype=np.float32)
            duration = np.zeros(self.chunk_bins, dtype=np.float32)
            fg[:length] = rows["foreground_target"].to_numpy(dtype=np.float32)
            classes[:length] = rows[[f"target_{name}" for name in self.class_names]].to_numpy(dtype=np.float32)
            boundary[:length] = rows[["left_boundary_target", "right_boundary_target"]].to_numpy(dtype=np.float32)
            duration[:length] = rows["target_event_bins"].to_numpy(dtype=np.float32)
            result.update(
                foreground=torch.from_numpy(fg),
                classes=torch.from_numpy(classes),
                boundary=torch.from_numpy(boundary),
                duration=torch.from_numpy(duration),
            )
        return result


class ASMLocGenome(nn.Module):
    """Local convolution plus inter-bin attention and localization heads."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv1d(config.hidden_dim, config.hidden_dim, 5, padding=2, groups=config.hidden_dim),
            nn.Conv1d(config.hidden_dim, config.hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.foreground_head = nn.Linear(config.hidden_dim, 1)
        self.class_head = nn.Linear(config.hidden_dim, config.num_classes)
        self.boundary_head = nn.Linear(config.hidden_dim, 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.input_projection(x)
        hidden = hidden + self.local(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.context(hidden, src_key_padding_mask=~mask)
        hidden = self.norm(hidden)
        return {
            "foreground_logits": self.foreground_head(hidden).squeeze(-1),
            "class_logits": self.class_head(hidden),
            "boundary": torch.sigmoid(self.boundary_head(hidden)),
        }


def normalize_features(
    train: pd.DataFrame,
    feature_names: list[str],
    stats: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    out = train.copy()
    values = out[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    for name in feature_names:
        if name in COUNT_FEATURES:
            values[name] = np.log1p(values[name].clip(lower=0))
    if stats is None:
        means = values.mean()
        stds = values.std(ddof=0).replace(0, 1.0)
        stats = {"mean": means.to_dict(), "std": stds.to_dict()}
    for name in feature_names:
        values[name] = (values[name] - float(stats["mean"][name])) / max(float(stats["std"][name]), 1e-6)
    out.loc[:, feature_names] = values.astype(np.float32)
    return out, stats


def _localization_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: TrainConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    mask = batch["mask"]
    fg_target = batch["foreground"]
    class_target = batch["classes"]
    positive = mask & (fg_target > 0)
    background = mask & ~positive
    # Dynamic segment sampling analogue: shorter events contribute more.
    duration = batch["duration"].clamp(min=1)
    short_weight = 1.0 + config.short_event_boost / torch.sqrt(duration)
    fg_weight = torch.where(positive, short_weight, torch.ones_like(short_weight))
    n_pos = positive.sum().float().clamp(min=1)
    n_neg = background.sum().float().clamp(min=1)
    fg_weight = fg_weight * torch.where(positive, (n_neg / n_pos).clamp(max=20), 1.0)
    fg_loss = F.binary_cross_entropy_with_logits(
        outputs["foreground_logits"][mask], fg_target[mask], weight=fg_weight[mask]
    )
    if positive.any():
        class_loss = F.binary_cross_entropy_with_logits(
            outputs["class_logits"][positive],
            class_target[positive],
            weight=short_weight[positive, None],
        )
        boundary_loss = F.smooth_l1_loss(outputs["boundary"][positive], batch["boundary"][positive])
    else:
        class_loss = fg_loss * 0
        boundary_loss = fg_loss * 0
    total = fg_loss + class_loss + config.boundary_loss_weight * boundary_loss
    return total, {
        "loss": float(total.detach()),
        "foreground_loss": float(fg_loss.detach()),
        "class_loss": float(class_loss.detach()),
        "boundary_loss": float(boundary_loss.detach()),
    }


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def fit_model(
    train_loader,
    validation_loader,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output: str | Path,
    feature_names: list[str],
    class_names: list[str],
    feature_stats: dict,
    device: torch.device,
) -> dict:
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    model = ASMLocGenome(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    stale = 0
    history: list[dict] = []
    for epoch in range(1, train_config.epochs + 1):
        epoch_row: dict[str, float] = {"epoch": epoch}
        for split, loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(split == "train")
            totals: list[dict[str, float]] = []
            for raw_batch in loader:
                batch = _move_batch(raw_batch, device)
                with torch.set_grad_enabled(split == "train"):
                    outputs = model(batch["x"], batch["mask"])
                    loss, parts = _localization_loss(outputs, batch, train_config)
                if split == "train":
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                totals.append(parts)
            for name in totals[0] if totals else ["loss"]:
                epoch_row[f"{split}_{name}"] = float(np.mean([row[name] for row in totals])) if totals else float("nan")
        history.append(epoch_row)
        value = epoch_row["validation_loss"]
        if value < best_loss - 1e-4:
            best_loss = value
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= train_config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training produced no checkpoint")
    checkpoint = {
        "model_state_dict": best_state,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "feature_names": feature_names,
        "class_names": class_names,
        "feature_stats": feature_stats,
        "best_validation_loss": best_loss,
        "history": history,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    pd.DataFrame(history).to_csv(output.with_suffix(".training.tsv"), sep="\t", index=False)
    return checkpoint


@torch.no_grad()
def predict_bins(model: ASMLocGenome, loader, n_rows: int, device: torch.device) -> dict[str, np.ndarray]:
    model.eval()
    sums = {
        "foreground_probability": np.zeros(n_rows, dtype=np.float64),
        "class_probability": np.zeros((n_rows, model.config.num_classes), dtype=np.float64),
        "boundary": np.zeros((n_rows, 2), dtype=np.float64),
    }
    counts = np.zeros(n_rows, dtype=np.float64)
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        outputs = model(batch["x"], batch["mask"])
        fg = torch.sigmoid(outputs["foreground_logits"]).cpu().numpy()
        classes = torch.sigmoid(outputs["class_logits"]).cpu().numpy()
        boundary = outputs["boundary"].cpu().numpy()
        indices = raw_batch["row_index"].numpy()
        masks = raw_batch["mask"].numpy()
        for batch_i in range(indices.shape[0]):
            valid = masks[batch_i]
            rows = indices[batch_i, valid]
            sums["foreground_probability"][rows] += fg[batch_i, valid]
            sums["class_probability"][rows] += classes[batch_i, valid]
            sums["boundary"][rows] += boundary[batch_i, valid]
            counts[rows] += 1
    counts = np.maximum(counts, 1)
    sums["foreground_probability"] /= counts
    sums["class_probability"] /= counts[:, None]
    sums["boundary"] /= counts[:, None]
    return {name: value.astype(np.float32) for name, value in sums.items()}


def _nms(proposals: pd.DataFrame, threshold: float) -> pd.DataFrame:
    kept: list[pd.Series] = []
    for _, candidate in proposals.sort_values("localization_score", ascending=False).iterrows():
        reject = False
        for other in kept:
            if candidate["sample_id"] != other["sample_id"] or candidate["chrom"] != other["chrom"]:
                continue
            overlap = interval_overlap(candidate["start"], candidate["end"], other["start"], other["end"])
            union = max(candidate["end"], other["end"]) - min(candidate["start"], other["start"])
            if union > 0 and overlap / union >= threshold:
                reject = True
                break
        if not reject:
            kept.append(candidate)
    return pd.DataFrame(kept).sort_values(["sample_id", "chrom", "start"]).reset_index(drop=True) if kept else proposals.iloc[:0]


def proposals_from_predictions(
    bins: pd.DataFrame,
    class_names: list[str],
    foreground_threshold: float = 0.45,
    class_threshold: float = 0.40,
    merge_gap_bins: int = 1,
    min_bins: int = 1,
    bin_size: int = 1_000_000,
    max_boundary_bins: int = 64,
    nms_iou: float = 0.65,
) -> pd.DataFrame:
    """Turn bin activation sequences into boundary-refined candidate events."""
    rows: list[dict] = []
    for (sample, chrom), chrom_df in bins.groupby(["sample_id", "chrom"], sort=False):
        ordered = chrom_df.sort_values("start").reset_index(drop=True)
        active: list[int] = []
        last_hit = -10
        for idx, row in ordered.iterrows():
            class_max = max(float(row[f"probability_{name}"]) for name in class_names)
            hit = float(row["foreground_probability"]) >= foreground_threshold and class_max >= class_threshold
            if hit:
                if active and idx - last_hit > merge_gap_bins + 1:
                    _emit_proposal(rows, ordered, active, sample, chrom, class_names, min_bins, bin_size, max_boundary_bins)
                    active = []
                active.append(idx)
                last_hit = idx
            elif active and idx - last_hit > merge_gap_bins:
                _emit_proposal(rows, ordered, active, sample, chrom, class_names, min_bins, bin_size, max_boundary_bins)
                active = []
        if active:
            _emit_proposal(rows, ordered, active, sample, chrom, class_names, min_bins, bin_size, max_boundary_bins)
    proposals = pd.DataFrame(rows)
    if proposals.empty:
        return pd.DataFrame(
            columns=["candidate_id", "sample_id", "chrom", "start", "end", "localization_score", "localization_classes"]
        )
    proposals = _nms(proposals, nms_iou)
    proposals.insert(
        0,
        "candidate_id",
        [f"{row.sample_id}:asmloc:{idx + 1:04d}" for idx, row in enumerate(proposals.itertuples())],
    )
    return proposals


def _emit_proposal(
    rows: list[dict],
    ordered: pd.DataFrame,
    indices: list[int],
    sample: str,
    chrom: str,
    class_names: list[str],
    min_bins: int,
    bin_size: int,
    max_boundary_bins: int,
) -> None:
    if len(indices) < min_bins:
        return
    selected = ordered.iloc[indices]
    fg = selected["foreground_probability"].to_numpy(float)
    best_local = int(np.argmax(fg))
    anchor = selected.iloc[best_local]
    left_bins = float(anchor["left_boundary_prediction"]) * max_boundary_bins
    right_bins = float(anchor["right_boundary_prediction"]) * max_boundary_bins
    start = max(0, int(anchor["start"] - left_bins * bin_size))
    end = int(anchor["end"] + right_bins * bin_size)
    # Never shrink away the activated support.
    start = min(start, int(selected["start"].min()))
    end = max(end, int(selected["end"].max()))
    class_scores = {name: float(selected[f"probability_{name}"].max()) for name in class_names}
    rows.append(
        {
            "sample_id": sample,
            "chrom": chrom,
            "start": start,
            "end": end,
            "n_active_bins": len(indices),
            "localization_score": float(fg.max()),
            "localization_mean_score": float(fg.mean()),
            "localization_classes": ";".join(name for name, score in class_scores.items() if score >= 0.4),
            **{f"localization_probability_{name}": score for name, score in class_scores.items()},
        }
    )


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[ASMLocGenome, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ASMLocGenome(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def write_summary(path: str | Path, **values) -> None:
    Path(path).write_text(json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8")
