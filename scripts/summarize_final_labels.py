#!/usr/bin/env python3
"""Create compact summary figures for the TSVs in final_labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "BFB": "#2a9d8f",
    "chromothripsis": "#e76f51",
    "seismic_amplification": "#457b9d",
    "ecDNA": "#f4a261",
}


def classify_file(path: Path) -> tuple[str, str]:
    name = path.stem
    status = "noncanonical" if "noncanonical" in name else "canonical"
    if name.startswith("bfbarchitect"):
        return "BFB", status
    if name.startswith("chromothripsis"):
        return "chromothripsis", status
    if name.startswith("seismic_amplification"):
        return "seismic_amplification", status
    if name.startswith("coral_ecDNA"):
        return "ecDNA", status
    raise ValueError(f"Unrecognized final-label file: {path.name}")


def read_tables(label_dir: Path) -> tuple[pd.DataFrame, dict[tuple[str, str], pd.DataFrame]]:
    tables: dict[tuple[str, str], pd.DataFrame] = {}
    records: list[dict[str, object]] = []
    for path in sorted(label_dir.glob("*.tsv")):
        source, status = classify_file(path)
        frame = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        for column in ["start", "end"]:
            frame[column] = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
        frame["size_bp"] = frame["end"] - frame["start"] + 1
        frame["source"] = source
        frame["status"] = status
        tables[(source, status)] = frame
        numeric = frame["size_bp"].dropna()
        records.append({
            "source": source,
            "status": status,
            "n_calls": len(frame),
            "n_samples": frame["sample_id"].nunique() if "sample_id" in frame else 0,
            "mean_size_mb": numeric.mean() / 1e6 if len(numeric) else np.nan,
            "median_size_mb": numeric.median() / 1e6 if len(numeric) else np.nan,
            "min_size_mb": numeric.min() / 1e6 if len(numeric) else np.nan,
            "max_size_mb": numeric.max() / 1e6 if len(numeric) else np.nan,
        })
    if not records:
        raise ValueError(f"No TSV files found in {label_dir}")
    summary = pd.DataFrame(records).sort_values(["source", "status"])
    return summary, tables


def save_counts(summary: pd.DataFrame, out: Path) -> None:
    labels = summary["source"].drop_duplicates().tolist()
    x = np.arange(len(labels))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ["n_calls", "n_samples"], ["Calls", "Unique samples"]):
        for offset, status in [(-width / 2, "canonical"), (width / 2, "noncanonical")]:
            values = []
            for label in labels:
                row = summary[(summary.source == label) & (summary.status == status)]
                values.append(float(row.iloc[0][metric]) if not row.empty else 0)
            ax.bar(x + offset, values, width, label=status, color="#264653" if status == "canonical" else "#b56576")
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_ylabel(title)
        ax.set_title(f"{title} by label class")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)
    fig.suptitle("Final-label inventory", fontsize=14)
    fig.savefig(out / "label_counts.png", dpi=220)
    plt.close(fig)


def save_sizes(tables: dict[tuple[str, str], pd.DataFrame], out: Path) -> None:
    order = [key for key in [("BFB", "canonical"), ("BFB", "noncanonical"),
                             ("chromothripsis", "canonical"), ("chromothripsis", "noncanonical"),
                             ("seismic_amplification", "canonical"), ("seismic_amplification", "noncanonical"),
                             ("ecDNA", "canonical"), ("ecDNA", "noncanonical")] if key in tables]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    positions, values, colors, labels = [], [], [], []
    for i, key in enumerate(order, 1):
        vals = tables[key]["size_bp"].dropna().to_numpy() / 1e6
        if len(vals):
            positions.append(i)
            values.append(vals)
            colors.append(COLORS[key[0]])
            labels.append(f"{key[0]}\n{key[1]}")
    bp = ax.boxplot(values, positions=positions, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for pos, vals in zip(positions, values):
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(pos + jitter, vals, s=18, color="#222222", alpha=0.65, zorder=3)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("Interval size (Mb; log scale)")
    ax.set_yscale("log")
    ax.set_title("Final-label interval sizes")
    ax.grid(axis="y", which="both", alpha=0.2)
    fig.savefig(out / "interval_size_distributions.png", dpi=220)
    plt.close(fig)


def save_sample_profile(tables: dict[tuple[str, str], pd.DataFrame], out: Path) -> None:
    frames = []
    for (source, status), frame in tables.items():
        counts = frame.groupby("sample_id").size().rename(f"{source}:{status}")
        frames.append(counts)
    matrix = pd.concat(frames, axis=1).fillna(0)
    matrix["total"] = matrix.sum(axis=1)
    matrix = matrix.sort_values("total", ascending=True).drop(columns="total")
    fig, ax = plt.subplots(figsize=(12, max(6, 0.28 * len(matrix) + 1)), constrained_layout=True)
    left = np.zeros(len(matrix))
    for column in matrix.columns:
        source, status = column.split(":", 1)
        ax.barh(matrix.index, matrix[column], left=left, label=f"{source} ({status})", color=COLORS[source], alpha=0.95 if status == "canonical" else 0.5)
        left += matrix[column].to_numpy()
    ax.set_xlabel("Number of calls")
    ax.set_title("Calls per sample")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)
    fig.savefig(out / "calls_per_sample.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_characteristics(tables: dict[tuple[str, str], pd.DataFrame], out: Path) -> None:
    specs = [
        (("BFB", "canonical"), "bfb_score", "BFB score"),
        (("chromothripsis", "canonical"), "confidence", "ShatterSeek confidence"),
        (("chromothripsis", "canonical"), "oscillating_cn_fraction", "Oscillating CN fraction"),
        (("ecDNA", "canonical"), "discordant_long_reads_total", "CoRaL discordant long reads"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, (key, column, title) in zip(axes.flat, specs):
        frame = tables.get(key)
        if frame is None or column not in frame:
            ax.text(0.5, 0.5, "not available", ha="center", va="center")
            ax.set_title(title)
            continue
        vals = pd.to_numeric(frame[column], errors="coerce").dropna()
        ax.hist(vals, bins=min(12, max(4, len(vals))), color=COLORS[key[0]], alpha=0.8, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel(column)
        ax.set_ylabel("Calls")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Caller-specific characteristics", fontsize=14)
    fig.savefig(out / "caller_characteristics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, default=Path("final_labels"))
    parser.add_argument("--output-dir", type=Path, default=Path("final_labels/summary_plots"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary, tables = read_tables(args.label_dir)
    summary.to_csv(args.output_dir / "final_label_summary.tsv", sep="\t", index=False)
    save_counts(summary, args.output_dir)
    save_sizes(tables, args.output_dir)
    save_sample_profile(tables, args.output_dir)
    save_characteristics(tables, args.output_dir)
    print(summary.to_string(index=False))
    print(f"plots={args.output_dir}")


if __name__ == "__main__":
    main()
