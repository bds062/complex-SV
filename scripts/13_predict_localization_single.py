#!/usr/bin/env python3
"""Run the final localization model and prediction plots for one genome."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CENTROMERES = REPO / "genomic_features/grch38.cen_coord.curated.bed"
CHROM_SIZES = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555,
    "5": 181538259, "6": 170805979, "7": 159345973, "8": 145138636,
    "9": 138394717, "10": 133797422, "11": 135086622, "12": 133275309,
    "13": 114364328, "14": 107043718, "15": 101991189, "16": 90338345,
    "17": 83257441, "18": 80373285, "19": 58617616, "20": 64444167,
    "21": 46709983, "22": 50818468, "X": 156040895, "Y": 57227415,
}
FINAL_COLUMNS = [
    "prediction_id", "sample_name", "chromosome", "chromosome_arm",
    "region", "start", "end", "prediction", "score", "threshold",
    "score_margin", "plot_status", "plot_path", "event_id",
    "representative_candidate_id", "cluster_id", "cluster_size",
    "original_start", "original_end", "boundary_scale", "region_mode",
    "nms_mode", "maximum_per_sample",
]


def command(argv: list[str]) -> None:
    print("+", " ".join(argv), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHON_BIN", sys.executable)
    subprocess.run(argv, check=True, env=env)


def absolute_path(value: str) -> str:
    return str(Path(value).expanduser().resolve())


def chrom_key(value: object) -> str:
    return str(value).strip().removeprefix("chr").removeprefix("CHR")


def load_centromeres() -> dict[str, tuple[int, int]]:
    frame = pd.read_csv(CENTROMERES, sep="\t", header=None, names=["chrom", "start", "end"])
    return {chrom_key(row.chrom): (int(row.start), int(row.end)) for row in frame.itertuples(index=False)}


def arm_context(chrom: str, start: int, end: int, centromeres: dict[str, tuple[int, int]]) -> tuple[str, int, int]:
    key = chrom_key(chrom)
    chrom_end = int(CHROM_SIZES.get(key, max(end, start + 1)))
    if key not in centromeres:
        return "", 0, chrom_end
    cen_start, cen_end = centromeres[key]
    if start < cen_end and end > cen_start:
        return "", 0, chrom_end
    if (start + end) / 2 < cen_start:
        return "p", 0, cen_start
    return "q", cen_end, chrom_end


def prepare_prediction_tables(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if raw.empty:
        plot_columns = [
            "prediction_id", "sample_id", "chrom", "arm", "start_bp", "end_bp",
            "context_start_bp", "context_end_bp", "highlight_start_bp",
            "highlight_end_bp", "prediction", "predicted_class",
            "predicted_classes", "score",
        ]
        return pd.DataFrame(columns=FINAL_COLUMNS), pd.DataFrame(columns=plot_columns)

    frame = raw.copy().sort_values(["score", "chrom", "start"], ascending=[False, True, True]).reset_index(drop=True)
    frame["prediction_id"] = [f"{row.sample_id}:prediction_{index:04d}" for index, row in enumerate(frame.itertuples(), start=1)]
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame["score_margin"] = frame["score"] - frame["threshold"]
    centromeres = load_centromeres()
    final_rows: list[dict[str, object]] = []
    plot_rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        start, end = int(row.start), int(row.end)
        arm, context_start, context_end = arm_context(str(row.chrom), start, end, centromeres)
        region = f"{row.chrom}:{start}-{end}"
        score, threshold, margin = float(row.score), float(row.threshold), float(row.score_margin)
        final_rows.append({
            "prediction_id": row.prediction_id, "sample_name": row.sample_id,
            "chromosome": row.chrom, "chromosome_arm": arm, "region": region,
            "start": start, "end": end, "prediction": row.label,
            "score": score, "threshold": threshold, "score_margin": margin,
            "plot_status": "pending", "plot_path": "", "event_id": row.event_id,
            "representative_candidate_id": row.representative_candidate_id,
            "cluster_id": row.cluster_id, "cluster_size": int(row.cluster_size),
            "original_start": int(row.original_start), "original_end": int(row.original_end),
            "boundary_scale": float(row.scale), "region_mode": row.region_mode,
            "nms_mode": row.nms_mode, "maximum_per_sample": int(row.maximum_per_sample),
        })
        plot_rows.append({
            "prediction_id": row.prediction_id, "sample_id": row.sample_id,
            "chrom": row.chrom, "arm": arm, "start_bp": start, "end_bp": end,
            "context_start_bp": context_start, "context_end_bp": context_end,
            "highlight_start_bp": start, "highlight_end_bp": end,
            "highlight_label": f"{row.label} prediction: {region}",
            "prediction": row.label, "predicted_class": row.label,
            "predicted_classes": row.label, "called_complex_sv": True,
            "is_labeled": "", "sv_class": "", "true_classes": "",
            "evidence": "localized_event", "split": "inference",
            "score": score, "objectness_prob": score, "threshold": threshold,
            "score_margin": margin,
            "score_text": f"score={score:.4f}; threshold={threshold:.4f}; margin={margin:.4f}; prediction_id={row.prediction_id}",
        })
    return pd.DataFrame(final_rows, columns=FINAL_COLUMNS), pd.DataFrame(plot_rows)


def attach_plot_index(final: pd.DataFrame, index_path: Path, output_dir: Path) -> pd.DataFrame:
    if final.empty or not index_path.exists():
        if not final.empty:
            final["plot_status"] = "not_plotted"
        return final
    index = pd.read_csv(index_path, sep="\t").fillna("")
    if "prediction_id" not in index.columns:
        final["plot_status"] = "plot_index_missing_prediction_id"
        return final
    lookup = index.set_index(index["prediction_id"].astype(str))
    statuses, paths = [], []
    for prediction_id in final["prediction_id"].astype(str):
        if prediction_id not in lookup.index:
            statuses.append("missing_from_plot_index")
            paths.append("")
            continue
        row = lookup.loc[prediction_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        statuses.append(str(row.get("plot_status", "")))
        raw_path = str(row.get("plot_path", ""))
        if raw_path:
            path = Path(raw_path)
            try:
                raw_path = path.resolve().relative_to(output_dir.resolve()).as_posix()
            except ValueError:
                raw_path = str(path)
        paths.append(raw_path)
    final["plot_status"] = statuses
    final["plot_path"] = paths
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-name", "--sample-id", dest="sample_name", required=True)
    parser.add_argument(
        "--wakhan-root", "--bed-root", "--wakhan-file", dest="wakhan_root", required=True,
        help="Wakhan *_copynumbers_segments prefix or either HP_1/HP_2 BED file.",
    )
    parser.add_argument("--severus-vcf", required=True, help="Matching Severus VCF for this genome.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("balanced", "sensitive"), default="balanced")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a CUDA device string.")
    parser.add_argument("--checkpoint", default=str(REPO / "models/localization_all48/model.pt"))
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    manifest_path = output_dir / "input_manifest.tsv"
    severus_vcf = Path(args.severus_vcf).expanduser().resolve()
    if not severus_vcf.is_file():
        raise FileNotFoundError(f"Severus VCF not found: {severus_vcf}")
    manifest = pd.DataFrame([{
        "sample_id": args.sample_name,
        "wakhan_root": absolute_path(args.wakhan_root),
        "severus_vcf": str(severus_vcf),
    }])
    manifest.to_csv(manifest_path, sep="\t", index=False)

    candidate_dir = output_dir / "candidates"
    feature_dir = output_dir / "features"
    call_dir = output_dir / "localized_calls"
    command([str(REPO / "candidate_generator/run.sh"), str(manifest_path), str(candidate_dir), "--profile", args.profile])
    candidate_table = candidate_dir / "merged_candidate_regions.csv"
    command([str(REPO / "scripts/04_embed_candidates.sh"), str(manifest_path), str(candidate_table), str(feature_dir), "--device", args.device])
    command([
        sys.executable, str(REPO / "scripts/10_predict_localization.py"),
        "--candidates", str(candidate_table),
        "--embedding-bundle", str(feature_dir / "embeddings.npz"),
        "--selected-embeddings", str(feature_dir / "selected_embedding_features.npz"),
        "--tabular-features", str(feature_dir / "tabular_features.npz"),
        "--checkpoint", absolute_path(args.checkpoint),
        "--output", str(call_dir), "--device", args.device,
    ])

    raw = pd.read_csv(call_dir / "localized_complex_sv.tsv", sep="\t")
    final, plot_input = prepare_prediction_tables(raw)
    plot_input_path = call_dir / "prediction_plot_input.tsv"
    plot_input.to_csv(plot_input_path, sep="\t", index=False)
    if not plot_input.empty:
        command([
            sys.executable, str(REPO / "scripts/plot_localized_predictions.py"),
            "--manifest", str(manifest_path), "--predictions", str(plot_input_path),
            "--output_dir", str(plots_dir), "--plot_scope", "all",
            "--group_by_column", "prediction", "--dpi", str(args.dpi),
            "--centromeres", str(CENTROMERES),
        ])
        final = attach_plot_index(final, plots_dir / "selected_predictions.tsv", output_dir)
    final_path = output_dir / "predictions.tsv"
    final.to_csv(final_path, sep="\t", index=False)
    plot_count = int((final["plot_status"] == "ok").sum()) if not final.empty else 0
    summary = {
        "sample_name": args.sample_name, "wakhan_root": manifest.iloc[0]["wakhan_root"],
        "severus_vcf": str(severus_vcf), "checkpoint": absolute_path(args.checkpoint),
        "candidate_profile": args.profile, "predictions": len(final),
        "plots_written": plot_count, "prediction_table": str(final_path),
        "plot_directory": str(plots_dir),
    }
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(final)} prediction(s) to {final_path}")
    print(f"Wrote {plot_count} plot(s) under {plots_dir}")


if __name__ == "__main__":
    main()
