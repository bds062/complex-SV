#!/usr/bin/env python3
"""Rank and visualize observed-label false positives from Pipeline27 LOO."""
from __future__ import annotations

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
P27 = HERE.parent
RESULTS = P27.parent
ROOT = RESULTS.parent
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
DISPLAY = {"BFB":"BFB", "chromothripsis":"Chromothripsis", "ecDNA":"ecDNA", "seismic_amplification":"Seismic amplification"}
COLORS = {"BFB":"#4E79A7", "chromothripsis":"#F28E2B", "ecDNA":"#E15759", "seismic_amplification":"#59A14F"}

def build_tables(top_per_class: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = pd.read_csv(P27 / "loo/oof_predictions.tsv", sep="\t")
    fp = pred[(pred.predicted == 1) & (pred.truth == 0)].copy()
    fp["margin"] = fp.probability - fp.threshold
    truths = (pred[pred.truth == 1].groupby(["sample_id","chrom"])["class"]
              .agg(lambda x: ";".join(sorted(set(x)))).rename("true_classes_on_chromosome"))
    fp = fp.merge(truths, on=["sample_id","chrom"], how="left")
    fp["true_classes_on_chromosome"] = fp.true_classes_on_chromosome.fillna("")
    fp["fp_context"] = np.where(fp.true_classes_on_chromosome.eq(""),
                                 "unlabeled chromosome", "wrong class on labeled chromosome")
    fp = fp.sort_values(["class","margin"], ascending=[True,False]).reset_index(drop=True)
    fp["class_fp_rank"] = fp.groupby("class").cumcount() + 1
    fp["overall_fp_rank"] = fp.margin.rank(method="first", ascending=False).astype(int)
    fp.to_csv(HERE / "all_observed_false_positives.tsv", sep="\t", index=False)
    top = fp.groupby("class", sort=False).head(top_per_class).copy()
    top.to_csv(HERE / "top_observed_false_positives.tsv", sep="\t", index=False)
    return fp, top

def make_summary_plots(fp: pd.DataFrame, top: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    counts = fp.groupby(["class","fp_context"]).size().unstack(fill_value=0).reindex(CLASSES)
    bottom = np.zeros(len(CLASSES))
    context_colors = {"unlabeled chromosome":"#9C755F", "wrong class on labeled chromosome":"#BAB0AC"}
    for context in ["unlabeled chromosome", "wrong class on labeled chromosome"]:
        values = counts.get(context, pd.Series(0, index=CLASSES)).to_numpy()
        axes[0].bar(range(len(CLASSES)), values, bottom=bottom, label=context,
                    color=context_colors[context])
        bottom += values
    axes[0].set_xticks(range(len(CLASSES)), [DISPLAY[c] for c in CLASSES], rotation=18)
    axes[0].set_ylabel("Observed-label false-positive chromosome-class calls")
    axes[0].set_title("Where Pipeline27 false positives occur")
    axes[0].legend(frameon=False, fontsize=9)
    for i, value in enumerate(bottom): axes[0].text(i, value+.6, str(int(value)), ha="center")

    ordered = top.sort_values(["class","margin"], ascending=[False,True]).reset_index(drop=True)
    labels = [f"{r.sample_id} {r.chrom} · {DISPLAY[r['class']]}" for _,r in ordered.iterrows()]
    y = np.arange(len(ordered))
    axes[1].barh(y, ordered.probability, color=[COLORS[c] for c in ordered["class"]], alpha=.82)
    axes[1].scatter(ordered.threshold, y, color="#C00000", marker="|", s=180,
                    linewidths=2.2, label="fold threshold", zorder=3)
    axes[1].set_yticks(y, labels, fontsize=8)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Held-out sigmoid probability")
    axes[1].set_title("Strongest false positives per class\n(red marker = decision threshold)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "false_positive_overview.png", dpi=190)
    plt.close(fig)

    matrix = fp.pivot_table(index="sample_id", columns="class", values="predicted", aggfunc="sum", fill_value=0)
    matrix = matrix.reindex(columns=CLASSES, fill_value=0)
    matrix = matrix.loc[matrix.sum(1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, max(6, .27*len(matrix)+2)))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlOrRd", vmin=0,
                      vmax=max(1, int(matrix.to_numpy().max())))
    ax.set_xticks(range(len(CLASSES)), [DISPLAY[c] for c in CLASSES], rotation=20, ha="right")
    ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
    ax.set_title("False-positive chromosomes per held-out genome and class")
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Held-out genome")
    fig.colorbar(image, ax=ax, label="Number of false-positive chromosomes", shrink=.7)
    fig.tight_layout()
    fig.savefig(HERE / "false_positive_sample_heatmap.png", dpi=190)
    plt.close(fig)

def build_plot_input(top: pd.DataFrame) -> Path:
    meta = pd.read_csv(P27 / "chromosome_embeddings/candidate_embeddings.tsv", sep="\t")
    meta = meta[["sample_id","chrom","start_bp","end_bp"]].drop_duplicates(["sample_id","chrom"])
    plot = top.merge(meta, on=["sample_id","chrom"], how="left")
    plot["context_start_bp"] = plot.start_bp
    plot["context_end_bp"] = plot.end_bp
    plot["highlight_start_bp"] = plot.start_bp
    plot["highlight_end_bp"] = plot.end_bp
    plot["highlight_label"] = "Pipeline27 whole-chromosome prediction"
    plot["predicted_class"] = plot["class"]
    plot["predicted_classes"] = plot["class"]
    plot["called_complex_sv"] = True
    plot["is_labeled"] = plot.true_classes_on_chromosome.ne("")
    plot["sv_class"] = plot.true_classes_on_chromosome
    plot["true_classes"] = plot.true_classes_on_chromosome
    plot["evidence"] = "whole_chromosome"
    plot["score"] = plot.probability
    plot["objectness_prob"] = plot.probability
    plot["score_text"] = [
        f"P={r.probability:.3f}; threshold={r.threshold:.3f}; margin={r.margin:.3f}; "
        f"observed truth={r.true_classes_on_chromosome or 'none'}"
        for r in plot.itertuples(index=False)
    ]
    path = HERE / "top_false_positive_plot_input.tsv"
    plot.to_csv(path, sep="\t", index=False)
    return path

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--top-per-class",type=int,default=4); parser.add_argument("--dpi",type=int,default=160)
    args=parser.parse_args(); HERE.mkdir(parents=True, exist_ok=True)
    fp, top = build_tables(args.top_per_class); make_summary_plots(fp, top)
    plot_input = build_plot_input(top)
    plot_dir = HERE / "chromosome_plots"
    if plot_dir.exists():
        shutil.rmtree(plot_dir)
    subprocess.run([
        sys.executable, str(ROOT / "complex-SV/discovery/plot_predicted_chromosomes.py"),
        "--manifest", str(RESULTS / "pipeline18/complex_sv_manifest.tsv"),
        "--prototype_distances", str(plot_input), "--output_dir", str(plot_dir),
        "--plot_scope", "all", "--group_by_column", "predicted_class", "--dpi", str(args.dpi),
        "--centromeres", str(RESULTS / "grch38.cen_coord.curated.bed")
    ], check=True)
    print(fp.groupby(["class","fp_context"]).size().to_string())
    print(f"Wrote {len(fp)} false positives and {len(top)} top chromosome plots")

if __name__ == "__main__": main()
