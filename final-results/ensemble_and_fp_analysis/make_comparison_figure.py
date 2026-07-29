#!/usr/bin/env python3
"""Plot overlap between packaged localization and chromosome-model false positives."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


HERE=Path(__file__).resolve().parent
DISPLAY={"BFB":"BFB","chromothripsis":"Chromothripsis","ecDNA":"ecDNA","seismic_amplification":"Seismic"}
COLORS={"BFB":"#4E79A7","chromothripsis":"#F28E2B","ecDNA":"#59A14F","seismic_amplification":"#E15759"}


def main():
    per=pd.read_csv(HERE/"per_class_overlap.tsv",sep="\t")
    scores=pd.read_csv(HERE/"shared_false_positive_scores.tsv",sep="\t")
    fig,axes=plt.subplots(1,2,figsize=(14,5.8))
    x=np.arange(len(per)); width=.34
    axes[0].bar(x-width/2,per.loc_unique_keys,width,color="#4E79A7")
    axes[0].bar(x+width/2,per.chrom_fp,width,color="#E15759")
    # The same exact-key overlap is cross-hatched from zero to its shared height
    # on both model bars, making the common portion visually proportional.
    axes[0].bar(x-width/2,per.shared,width,facecolor="none",edgecolor="#202020",hatch="////",linewidth=.7)
    axes[0].bar(x+width/2,per.shared,width,facecolor="none",edgecolor="#202020",hatch="////",linewidth=.7)
    axes[0].set_xticks(x,[DISPLAY[c] for c in per["class"]],rotation=15)
    axes[0].set_ylabel("Unique sample/chromosome/class FP keys")
    axes[0].set_title("False-positive overlap by class",fontweight="bold")
    axes[0].legend(handles=[
        Patch(facecolor="#4E79A7",label="Localization FP keys"),
        Patch(facecolor="#E15759",label="Chromosome FP keys"),
        Patch(facecolor="white",edgecolor="#202020",hatch="////",label="Shared exact FP keys"),
    ],frameon=False)
    axes[0].grid(axis="y",alpha=.18)

    for label,part in scores.groupby("class"):
        axes[1].scatter(part.localization_score,part.chromosome_probability,s=58,alpha=.8,
                        color=COLORS[label],edgecolor="white",linewidth=.5,label=DISPLAY[label])
    rho,p=spearmanr(scores.localization_score,scores.chromosome_probability)
    axes[1].set_xlabel("Localization-model score")
    axes[1].set_ylabel("Chromosome-model probability")
    axes[1].set_title(f"Scores for 37 shared exact FP keys\nSpearman ρ={rho:.3f}",fontweight="bold")
    axes[1].legend(frameon=False); axes[1].grid(alpha=.18)
    fig.tight_layout(); fig.savefig(HERE/"false_positive_overlap.png",dpi=210,bbox_inches="tight"); plt.close(fig)


if __name__=="__main__": main()
