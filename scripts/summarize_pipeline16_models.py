#!/usr/bin/env python3
"""Summarize neural, few-shot, and hybrid pipeline16 cross-fold results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


MODEL_DIRS = {
    "Neural": Path("."),
    "Few-shot": Path("candidate_region_classifier_fewshot_general"),
    "Hybrid": Path("candidate_region_classifier_hybrid_general_fewshot"),
}
PROFILES = ("sensitive", "balanced")
METRICS = (
    ("objectness_f1_aggregate", "Objectness F1"),
    ("macro_f1_aggregate", "Macro class F1"),
    ("exact_match_aggregate", "Exact match"),
    ("any_match_aggregate", "Any match"),
)
COLORS = {"sensitive": "#4E79A7", "balanced": "#E15759"}


def run(base: Path) -> None:
    rows: list[dict[str, object]] = []
    class_rows: list[pd.DataFrame] = []
    for profile in PROFILES:
        profile_dir = base / profile
        for model_name, relative_dir in MODEL_DIRS.items():
            model_dir = profile_dir / relative_dir
            overview_path = model_dir / "cross_fold_overview.json"
            class_path = model_dir / "aggregate_per_class_metrics.tsv"
            if not overview_path.exists() or not class_path.exists():
                raise FileNotFoundError(f"Incomplete {profile} {model_name} outputs: {model_dir}")
            overview = json.loads(overview_path.read_text())
            rows.append(
                {
                    "profile": profile,
                    "model": model_name,
                    "n_folds": int(overview["n_folds"]),
                    "n_ok_folds": int(overview["n_ok_folds"]),
                    **{key: overview.get(key) for key, _ in METRICS},
                }
            )
            per_class = pd.read_csv(class_path, sep="\t")
            per_class.insert(0, "model", model_name)
            per_class.insert(0, "profile", profile)
            class_rows.append(per_class)

    summary = pd.DataFrame(rows)
    summary.to_csv(base / "all_model_comparison.tsv", sep="\t", index=False)
    pd.concat(class_rows, ignore_index=True).to_csv(
        base / "all_model_comparison_per_class.tsv", sep="\t", index=False
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    x = np.arange(len(MODEL_DIRS), dtype=float)
    width = 0.34
    for ax, (metric_key, title) in zip(axes.flat, METRICS):
        for index, profile in enumerate(PROFILES):
            values = (
                summary.loc[summary["profile"].eq(profile)]
                .set_index("model")
                .loc[list(MODEL_DIRS), metric_key]
                .astype(float)
                .to_numpy()
            )
            offset = (index - 0.5) * width
            bars = ax.bar(x + offset, values, width, color=COLORS[profile], label=profile.title())
            ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=9)
        ax.set_title(title, fontsize=14)
        ax.set_xticks(x, list(MODEL_DIRS), fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Held-out score", fontsize=12)
    axes[1, 0].set_ylabel("Held-out score", fontsize=12)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle("Pipeline16 candidate-profile and model comparison", fontsize=16, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(base / "all_model_comparison.png", dpi=300, bbox_inches="tight")
    fig.savefig(base / "all_model_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.base)
