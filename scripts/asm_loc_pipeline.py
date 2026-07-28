#!/usr/bin/env python3
"""Train and run chromosome-scale complex-SV localization.

Subcommands:
  build-dataset  Parse CNA/SV inputs into cheap chromosome-bin features.
  train          Train an ASM-Loc-inspired sequence localizer.
  predict        Scan bin features and emit boundary-refined event proposals.
  materialize    Add pipeline18-compatible region features to proposals.
  evaluate       Measure proposal recall and boundary IoU against call regions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from localization.asm_loc import (  # noqa: E402
    CLASS_NAMES,
    FEATURE_NAMES,
    ASMLocGenome,
    ChromosomeChunkDataset,
    ModelConfig,
    TrainConfig,
    add_targets,
    build_sample_bins,
    fit_model,
    interval_overlap,
    load_checkpoint,
    normalize_features,
    predict_bins,
    proposals_from_predictions,
    write_summary,
)
from candidate_generator_v3 import resolve_cna_vcf  # noqa: E402
from process_vcfs import (  # noqa: E402
    _chrom_arm,
    _summarize_candidate_intervals,
    assign_linked_cluster_ids,
    calculate_ploidy,
    get_bps,
    read_centromere_bed,
    read_cna_vcf_to_dataframe,
)


def safe_name(value: object) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("._") or "sample"


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "," if path.suffix == ".csv" else "\t"
    return pd.read_csv(path, sep=sep, compression="infer").fillna("")


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if path.suffix == ".csv" else "\t"
    frame.to_csv(path, sep=sep, index=False, compression="infer")


def _teacher_regions(path: str | Path, objectness: float, type_probability: float) -> pd.DataFrame:
    table = read_table(path)
    required = {"sample_id", "chrom", "start_bp", "end_bp", "objectness_prob"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Teacher predictions missing columns: {sorted(missing)}")
    rows: list[dict] = []
    for row in table.to_dict("records"):
        if float(row.get("objectness_prob", 0) or 0) < objectness:
            continue
        for name in CLASS_NAMES:
            probability = float(row.get(f"type_probability_{name}", 0) or 0)
            if probability >= type_probability:
                rows.append(
                    {
                        "region_id": f"teacher:{row.get('candidate_id', len(rows))}:{name}",
                        "sample_id": str(row["sample_id"]),
                        "chrom": str(row["chrom"]),
                        "start": int(row["start_bp"]),
                        "end": int(row["end_bp"]),
                        "label": name,
                        "source": "pipeline18_cross_fold_teacher",
                    }
                )
    return pd.DataFrame(rows)


def load_training_regions(args: argparse.Namespace) -> pd.DataFrame:
    regions = read_table(args.regions)
    required = {"sample_id", "chrom", "start", "end", "label"}
    missing = required.difference(regions.columns)
    if missing:
        raise ValueError(f"Regions table missing columns: {sorted(missing)}")
    regions = regions[regions["label"].astype(str).isin(CLASS_NAMES)].copy()
    if args.teacher_predictions:
        teacher = _teacher_regions(
            args.teacher_predictions,
            args.teacher_objectness,
            args.teacher_type_probability,
        )
        if not teacher.empty:
            # Do not add a teacher proposal where a direct region already overlaps.
            direct_keys = {
                key: group
                for key, group in regions.groupby(["sample_id", "chrom"], sort=False)
            }
            keep = []
            for row in teacher.to_dict("records"):
                existing = direct_keys.get((row["sample_id"], row["chrom"]))
                overlaps = False
                if existing is not None:
                    overlaps = any(
                        interval_overlap(row["start"], row["end"] + 1, call.start, call.end + 1) > 0
                        for call in existing.itertuples()
                    )
                keep.append(not overlaps)
            teacher = teacher.loc[keep]
            regions = pd.concat([regions, teacher], ignore_index=True, sort=False)
    return regions


def build_dataset(args: argparse.Namespace) -> None:
    manifest = read_table(args.manifest)
    regions = load_training_regions(args)
    frames: list[pd.DataFrame] = []
    failures: list[dict] = []
    for index, row in enumerate(manifest.to_dict("records"), start=1):
        sample = str(row["sample_id"])
        print(f"[{index}/{len(manifest)}] building bins for {sample}", flush=True)
        try:
            cna = read_cna_vcf_to_dataframe(resolve_cna_vcf(row["wakhan_root"]))
            bps = get_bps(row["severus_vcf"])
            bins = build_sample_bins(
                sample,
                cna,
                bps,
                bin_size=args.bin_size,
                high_copy_ratio=args.high_copy_ratio,
                high_copy_floor=args.high_copy_floor,
            )
            bins = add_targets(
                bins,
                regions,
                max_boundary_bins=args.max_boundary_bins,
                bin_size=args.bin_size,
            )
            frames.append(bins)
        except Exception as exc:
            if not args.keep_going:
                raise
            failures.append({"sample_id": sample, "error": str(exc)})
            print(f"  ERROR: {exc}", flush=True)
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_table(merged, args.output)
    summary_path = Path(args.output).with_suffix("").with_suffix(".summary.json")
    write_summary(
        summary_path,
        manifest_samples=len(manifest),
        output_samples=int(merged["sample_id"].nunique()) if not merged.empty else 0,
        bins=len(merged),
        positive_bins=int((merged.get("foreground_target", 0) > 0).sum()),
        direct_regions=int((regions.get("source", "") != "pipeline18_cross_fold_teacher").sum()),
        teacher_regions=int((regions.get("source", "") == "pipeline18_cross_fold_teacher").sum()),
        failed_samples=failures,
        bin_size=args.bin_size,
        feature_names=list(FEATURE_NAMES),
    )


def _split_samples(frame: pd.DataFrame, validation_samples: str, fraction: float, seed: int) -> tuple[list[str], list[str]]:
    samples = sorted(frame["sample_id"].astype(str).unique())
    explicit = [value.strip() for value in validation_samples.split(",") if value.strip()]
    if explicit:
        missing = sorted(set(explicit).difference(samples))
        if missing:
            raise ValueError(f"Validation samples absent from dataset: {missing}")
        valid = explicit
    else:
        rng = np.random.default_rng(seed)
        shuffled = list(rng.permutation(samples))
        valid = shuffled[: max(1, round(len(samples) * fraction))]
    train = [sample for sample in samples if sample not in set(valid)]
    if not train:
        raise ValueError("Validation selection leaves no training samples")
    return train, valid


def train(args: argparse.Namespace) -> None:
    frame = read_table(args.dataset)
    feature_names = list(FEATURE_NAMES)
    class_names = list(CLASS_NAMES)
    train_samples, valid_samples = _split_samples(
        frame, args.validation_samples, args.validation_fraction, args.seed
    )
    train_frame = frame[frame["sample_id"].astype(str).isin(train_samples)].copy()
    valid_frame = frame[frame["sample_id"].astype(str).isin(valid_samples)].copy()
    train_frame, stats = normalize_features(train_frame, feature_names)
    valid_frame, _ = normalize_features(valid_frame, feature_names, stats=stats)
    train_config = TrainConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        chunk_bins=args.chunk_bins,
        chunk_overlap_bins=args.chunk_overlap_bins,
        short_event_boost=args.short_event_boost,
        boundary_loss_weight=args.boundary_loss_weight,
        patience=args.patience,
        seed=args.seed,
    )
    model_config = ModelConfig(
        input_dim=len(feature_names),
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_boundary_bins=args.max_boundary_bins,
    )
    train_dataset = ChromosomeChunkDataset(
        train_frame, feature_names, class_names, args.chunk_bins, args.chunk_overlap_bins
    )
    valid_dataset = ChromosomeChunkDataset(
        valid_frame, feature_names, class_names, args.chunk_bins, args.chunk_overlap_bins
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = fit_model(
        train_loader,
        valid_loader,
        model_config,
        train_config,
        args.output,
        feature_names,
        class_names,
        stats,
        device,
    )
    split = {
        "train_samples": train_samples,
        "validation_samples": valid_samples,
        "train_bins": len(train_frame),
        "validation_bins": len(valid_frame),
        "best_validation_loss": checkpoint["best_validation_loss"],
    }
    Path(args.output).with_suffix(".split.json").write_text(json.dumps(split, indent=2) + "\n")


def predict(args: argparse.Namespace) -> None:
    raw = read_table(args.dataset)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    feature_names = list(checkpoint["feature_names"])
    class_names = list(checkpoint["class_names"])
    frame, _ = normalize_features(raw, feature_names, checkpoint["feature_stats"])
    train_config = checkpoint["train_config"]
    dataset = ChromosomeChunkDataset(
        frame,
        feature_names,
        class_names,
        args.chunk_bins or int(train_config["chunk_bins"]),
        args.chunk_overlap_bins if args.chunk_overlap_bins is not None else int(train_config["chunk_overlap_bins"]),
        include_targets=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    values = predict_bins(model, loader, len(frame), device)
    scored = raw.copy()
    scored["foreground_probability"] = values["foreground_probability"]
    scored["left_boundary_prediction"] = values["boundary"][:, 0]
    scored["right_boundary_prediction"] = values["boundary"][:, 1]
    for index, name in enumerate(class_names):
        scored[f"probability_{name}"] = values["class_probability"][:, index]
    proposals = proposals_from_predictions(
        scored,
        class_names,
        foreground_threshold=args.foreground_threshold,
        class_threshold=args.class_threshold,
        merge_gap_bins=args.merge_gap_bins,
        min_bins=args.min_bins,
        bin_size=args.bin_size,
        max_boundary_bins=int(checkpoint["model_config"]["max_boundary_bins"]),
        nms_iou=args.nms_iou,
    )
    write_table(scored, args.bin_predictions)
    write_table(proposals, args.output)
    write_summary(
        Path(args.output).with_suffix(".summary.json"),
        bins=len(scored),
        proposals=len(proposals),
        samples=int(proposals["sample_id"].nunique()) if not proposals.empty else 0,
        foreground_threshold=args.foreground_threshold,
        class_threshold=args.class_threshold,
        nms_iou=args.nms_iou,
    )


def materialize(args: argparse.Namespace) -> None:
    proposals = read_table(args.proposals)
    manifest = read_table(args.manifest)
    centromeres = read_centromere_bed(args.centromeres)
    manifest_by_sample = {str(row["sample_id"]): row for row in manifest.to_dict("records")}
    frames: list[pd.DataFrame] = []
    for index, (sample, group) in enumerate(proposals.groupby("sample_id", sort=False), start=1):
        print(f"[{index}/{proposals['sample_id'].nunique()}] materializing {sample}", flush=True)
        manifest_row = manifest_by_sample.get(str(sample))
        if manifest_row is None:
            raise KeyError(f"Proposal sample {sample!r} absent from manifest")
        cna_vcf = resolve_cna_vcf(manifest_row["wakhan_root"])
        cna = read_cna_vcf_to_dataframe(cna_vcf)
        bps = get_bps(manifest_row["severus_vcf"])
        intervals = group.copy()
        intervals["arm"] = [
            _chrom_arm(row.chrom, int(row.start), int(row.end), centromeres)
            for row in intervals.itertuples()
        ]
        intervals["n_windows"] = pd.to_numeric(intervals.get("n_active_bins", 1), errors="coerce").fillna(1).astype(int)
        intervals["component_intervals"] = [
            f"{int(row.start)}-{int(row.end)}" for row in intervals.itertuples()
        ]
        summary = _summarize_candidate_intervals(
            intervals,
            cna,
            bps,
            sample_ploidy=calculate_ploidy(cna),
            apply_candidate_filter=False,
        )
        localization_columns = [
            column
            for column in intervals.columns
            if column.startswith("localization_") or column in {"candidate_id", "n_active_bins"}
        ]
        metadata = intervals.set_index(["chrom", "start", "end"])[localization_columns]
        summary = summary.join(metadata, on=["chrom", "start", "end"])
        summary = assign_linked_cluster_ids(summary, cna, bps)
        summary.insert(0, "sample_id", str(sample))
        summary.insert(1, "candidate_id", summary.pop("candidate_id"))
        summary.insert(2, "wakhan_sample_id", str(manifest_row.get("wakhan_sample_id", "")))
        summary.insert(3, "wakhan_root", str(manifest_row["wakhan_root"]))
        summary.insert(4, "cna_vcf", str(cna_vcf))
        summary.insert(5, "severus_vcf", str(manifest_row["severus_vcf"]))
        summary.insert(6, "discovery_source", "asm_loc_sliding_window")
        for name in ("ecDNA", "Seismic_Amplification", "chromothripsis", "BFB"):
            summary[name] = ""
        frames.append(summary)
    merged = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    write_table(merged, args.output)


def evaluate(args: argparse.Namespace) -> None:
    proposals = read_table(args.proposals)
    calls = read_table(args.regions)
    by_key = {
        key: frame
        for key, frame in proposals.groupby(
            [proposals["sample_id"].astype(str), proposals["chrom"].astype(str)], sort=False
        )
    }
    rows = []
    for call in calls.to_dict("records"):
        candidates = by_key.get((str(call["sample_id"]), str(call["chrom"])), pd.DataFrame())
        best_iou = 0.0
        best_coverage = 0.0
        best_id = ""
        for candidate in candidates.to_dict("records"):
            overlap = interval_overlap(
                int(call["start"]), int(call["end"]) + 1, int(candidate["start"]), int(candidate["end"])
            )
            union = max(int(call["end"]) + 1, int(candidate["end"])) - min(int(call["start"]), int(candidate["start"]))
            coverage = overlap / max(1, int(call["end"]) + 1 - int(call["start"]))
            iou = overlap / max(1, union)
            if (iou, coverage) > (best_iou, best_coverage):
                best_iou, best_coverage, best_id = iou, coverage, str(candidate.get("candidate_id", ""))
        rows.append({**call, "best_candidate_id": best_id, "best_iou": best_iou, "best_call_coverage": best_coverage})
    details = pd.DataFrame(rows)
    write_table(details, args.output)
    summary = {
        "calls": len(details),
        "proposals": len(proposals),
        "mean_best_iou": float(details["best_iou"].mean()),
        "median_best_iou": float(details["best_iou"].median()),
        "recall_coverage_0.5": float((details["best_call_coverage"] >= 0.5).mean()),
        "recall_iou_0.3": float((details["best_iou"] >= 0.3).mean()),
        "recall_iou_0.5": float((details["best_iou"] >= 0.5).mean()),
    }
    write_summary(Path(args.output).with_suffix(".summary.json"), **summary)
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-dataset")
    build.add_argument("--manifest", required=True)
    build.add_argument("--regions", required=True)
    build.add_argument("--teacher_predictions", default="")
    build.add_argument("--teacher_objectness", type=float, default=0.95)
    build.add_argument("--teacher_type_probability", type=float, default=0.90)
    build.add_argument("--output", required=True)
    build.add_argument("--bin_size", type=int, default=1_000_000)
    build.add_argument("--max_boundary_bins", type=int, default=64)
    build.add_argument("--high_copy_ratio", type=float, default=2.0)
    build.add_argument("--high_copy_floor", type=float, default=6.0)
    build.add_argument("--keep_going", action="store_true")
    build.set_defaults(func=build_dataset)

    training = sub.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--validation_samples", default="")
    training.add_argument("--validation_fraction", type=float, default=0.20)
    training.add_argument("--epochs", type=int, default=35)
    training.add_argument("--patience", type=int, default=7)
    training.add_argument("--learning_rate", type=float, default=3e-4)
    training.add_argument("--weight_decay", type=float, default=1e-3)
    training.add_argument("--batch_size", type=int, default=16)
    training.add_argument("--chunk_bins", type=int, default=256)
    training.add_argument("--chunk_overlap_bins", type=int, default=64)
    training.add_argument("--short_event_boost", type=float, default=3.0)
    training.add_argument("--boundary_loss_weight", type=float, default=0.5)
    training.add_argument("--hidden_dim", type=int, default=96)
    training.add_argument("--num_heads", type=int, default=4)
    training.add_argument("--num_layers", type=int, default=3)
    training.add_argument("--dropout", type=float, default=0.15)
    training.add_argument("--max_boundary_bins", type=int, default=64)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--device", default="auto")
    training.set_defaults(func=train)

    inference = sub.add_parser("predict")
    inference.add_argument("--dataset", required=True)
    inference.add_argument("--checkpoint", required=True)
    inference.add_argument("--output", required=True)
    inference.add_argument("--bin_predictions", required=True)
    inference.add_argument("--foreground_threshold", type=float, default=0.45)
    inference.add_argument("--class_threshold", type=float, default=0.40)
    inference.add_argument("--merge_gap_bins", type=int, default=1)
    inference.add_argument("--min_bins", type=int, default=1)
    inference.add_argument("--nms_iou", type=float, default=0.65)
    inference.add_argument("--bin_size", type=int, default=1_000_000)
    inference.add_argument("--chunk_bins", type=int, default=None)
    inference.add_argument("--chunk_overlap_bins", type=int, default=None)
    inference.add_argument("--batch_size", type=int, default=32)
    inference.add_argument("--workers", type=int, default=0)
    inference.add_argument("--device", default="auto")
    inference.set_defaults(func=predict)

    material = sub.add_parser("materialize")
    material.add_argument("--manifest", required=True)
    material.add_argument("--proposals", required=True)
    material.add_argument("--output", required=True)
    material.add_argument(
        "--centromeres",
        default="/data/KolmogorovLab/srinivasanbd/results/grch38.cen_coord.curated.bed",
    )
    material.set_defaults(func=materialize)

    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--proposals", required=True)
    evaluation.add_argument("--regions", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.set_defaults(func=evaluate)
    return root


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
