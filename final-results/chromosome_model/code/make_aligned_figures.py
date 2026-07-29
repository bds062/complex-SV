#!/usr/bin/env python3
"""Generate visually aligned model figures for either summer-results package."""
from __future__ import annotations

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
METRICS = ROOT / "metrics"
PREDICTIONS = ROOT / "predictions"
MODEL = ROOT.name
CLASSES = ["BFB", "chromothripsis", "ecDNA", "seismic_amplification"]
DISPLAY = {"BFB":"BFB", "chromothripsis":"Chromothripsis", "ecDNA":"ecDNA", "seismic_amplification":"Seismic amplification"}
COLORS = {
    "precision":"#F28E2B", "recall":"#4E79A7", "f1":"#59A14F", "f2":"#B07AA1",
    "proposal_miss":"#9C755F", "representation_miss":"#9C755F", "event_geometry":"#BAB0AC",
    "score_threshold":"#E15759", "nms":"#F28E2B", "output_cap":"#EDC948",
    "one_to_one_collision":"#B07AA1", "recovered":"#59A14F",
}

plt.rcParams.update({
    "font.family":"DejaVu Sans", "font.size":10.5, "axes.titlesize":15,
    "axes.titleweight":"bold", "axes.labelsize":11, "xtick.labelsize":9.5,
    "ytick.labelsize":9.5, "legend.fontsize":9.5, "figure.facecolor":"white",
    "axes.facecolor":"white", "axes.edgecolor":"#455A64", "text.color":"#263238",
    "axes.labelcolor":"#37474F", "xtick.color":"#455A64", "ytick.color":"#455A64",
})

def clean_axis(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color="#90A4AE", alpha=.18, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#B0BEC5")

def metric_inputs():
    if MODEL == "localization_model":
        summary=json.loads((METRICS/"run_summary.json").read_text())
        overall={"precision":summary["classified_precision"], "recall":summary["classified_recall"],
                 "f1":summary["classified_f1"], "f2":summary["classified_f2"]}
        frame=pd.read_csv(METRICS/"per_class_metrics.tsv",sep="\t").rename(columns={"label":"class"})
        subtitle=(f"Localized event classification · {summary['true_predictions']}/108 one-to-one matches · "
                  f"{summary['n_predictions']} emitted intervals")
        title="Localization model · LOO held-out performance"
    elif MODEL == "chrom_model":
        summary=json.loads((METRICS/"loo_run_summary.json").read_text())
        overall={key:summary[key] for key in ["precision","recall","f1","f2"]}
        frame=pd.read_csv(METRICS/"loo_per_class_metrics.tsv",sep="\t")
        subtitle=(f"Whole-chromosome classification · {summary['tp']}/99 chromosome-class positives recovered · "
                  f"{summary['predictions']} positive calls")
        title="Chromosome model · LOO held-out performance"
    else:
        raise ValueError(f"unsupported package: {MODEL}")
    return overall, frame.set_index("class").loc[CLASSES].reset_index(), title, subtitle

def plot_metrics():
    overall, frame, title, subtitle = metric_inputs()
    fig, axes=plt.subplots(1,2,figsize=(15,5.8),gridspec_kw={"width_ratios":[.88,1.45]})
    fig.suptitle(title, x=.055, y=.985, ha="left", fontsize=17, fontweight="bold")
    fig.text(.055,.925,subtitle,ha="left",fontsize=10.5,color="#607D8B")
    labels=["Precision","Recall","F1","F2"]; keys=["precision","recall","f1","f2"]
    values=[overall[key] for key in keys]
    bars=axes[0].bar(labels,values,color=[COLORS[k] for k in keys],width=.68)
    axes[0].bar_label(bars,labels=[f"{v:.3f}" for v in values],padding=4,fontweight="bold",fontsize=10)
    axes[0].set_ylim(0,1); axes[0].set_ylabel("Held-out score"); axes[0].set_title("Overall",loc="left")
    clean_axis(axes[0])
    x=np.arange(len(CLASSES)); width=.24
    for offset,key,label in [(-width,"precision","Precision"),(0,"recall","Recall"),(width,"f1","F1")]:
        bars=axes[1].bar(x+offset,frame[key],width,label=label,color=COLORS[key])
        axes[1].bar_label(bars,labels=[f"{v:.2f}" for v in frame[key]],padding=3,fontsize=8)
    axes[1].set_xticks(x,[DISPLAY[c] for c in CLASSES],rotation=10)
    axes[1].set_ylim(0,1); axes[1].set_ylabel("Held-out score"); axes[1].set_title("By event class",loc="left")
    axes[1].legend(frameon=False,ncol=3,loc="upper right"); clean_axis(axes[1])
    fig.subplots_adjust(left=.055,right=.985,top=.84,bottom=.13,wspace=.24)
    fig.savefig(FIGURES/"loo_held_out_metrics.png",dpi=220,facecolor="white")
    plt.close(fig)

def loss_inputs():
    if MODEL == "localization_model":
        flow=[("Caller\nlabels",108),("Candidate\ncovered",106),("Geometry\ncompatible",94),
              ("Above class\nthreshold",58),("After\nNMS",58),("After output\ncaps",56),("One-to-one\nmatches",53)]
        annotations=[("−2","Candidate proposal miss","loss"),("−12","Event geometry","loss"),
                     ("−36","Class score threshold","loss"),("−0","NMS","zero"),
                     ("−2","Output cap","loss"),("−3","Assignment collision","loss")]
        raw=pd.read_csv(METRICS/"loss_by_class.tsv",sep="\t")
        stages=["proposal_miss","event_geometry","score_threshold","nms","output_cap","one_to_one_collision","recovered"]
        subtitle="Localized caller intervals · exclusive first-failure counts · overlap coefficient ≥ 0.5"
        note="Largest loss: 36 labels fail class scoring/thresholding after compatible event geometry."
    else:
        flow=[("Source interval\nlabels",108),("Chromosome-class\ntargets",99),
              ("Chromosome\nrepresented",99),("Above class\nthreshold",54),("Final\nmatches",54)]
        annotations=[("9 merged","Same-class events collapsed","collapse"),("−0","Representation","zero"),
                     ("−45","Class score threshold","loss"),("−0","Post-processing","zero")]
        pred=pd.read_csv(PREDICTIONS/"loo_oof_predictions.tsv",sep="\t")
        positive=pred[pred.truth==1].copy()
        positive["loss_stage"]=np.where(positive.predicted==1,"recovered","score_threshold")
        raw=(positive.groupby(["class","loss_stage"]).size().rename("labels").reset_index()
             .rename(columns={"class":"truth_label"}))
        stages=["representation_miss","score_threshold","nms","output_cap","one_to_one_collision","recovered"]
        subtitle="Chromosome-class targets · no proposals, boundaries, overlap filter, NMS, or output cap"
        note="The 108→99 change is label deduplication, not model loss; 45 of 99 targets fall below class thresholds."
    return flow,annotations,raw,stages,subtitle,note

def plot_loss_pipeline():
    flow,annotations,raw,stages,subtitle,note=loss_inputs()
    fig=plt.figure(figsize=(15,8.4)); grid=fig.add_gridspec(2,1,height_ratios=[1.02,1],hspace=.43)
    ax=fig.add_subplot(grid[0]); ax.set_xlim(-.7,12.7); ax.set_ylim(-1.15,1.35); ax.axis("off")
    xs=np.linspace(0,12,len(flow))
    for i,((label,count),x) in enumerate(zip(flow,xs)):
        final=i==len(flow)-1; initial=i==0
        face="#4E79A7" if initial else (COLORS["recovered"] if final else "#EAF0F5")
        text_color="white" if initial or final else "#263238"
        box=FancyBboxPatch((x-.56,-.14),1.12,.62,boxstyle="round,pad=.04,rounding_size=.07",
                           facecolor=face,edgecolor="#455A64",linewidth=1.15)
        ax.add_patch(box); ax.text(x,.17,str(count),ha="center",va="center",fontsize=17,fontweight="bold",color=text_color)
        ax.text(x,-.38,label,ha="center",va="top",fontsize=10.3,linespacing=1.12)
        if i<len(flow)-1:
            nx=xs[i+1]; ax.add_patch(FancyArrowPatch((x+.59,.17),(nx-.59,.17),arrowstyle="-|>",mutation_scale=14,
                                                     linewidth=1.3,color="#607D8B"))
            main,detail,kind=annotations[i]; mid=(x+nx)/2
            color="#607D8B" if kind in {"zero","collapse"} else COLORS["score_threshold"]
            ax.text(mid,.72,main,ha="center",va="center",fontsize=11.5,fontweight="bold",color=color)
            ax.text(mid,.55,detail,ha="center",va="top",fontsize=8.2,color="#607D8B")
    ax.text(-.55,1.19,"Label retention through inference",fontsize=17,fontweight="bold",ha="left")
    ax.text(-.55,.99,subtitle,fontsize=10.3,color="#607D8B",ha="left")
    ax.text(6,-.97,note,ha="center",va="center",fontsize=9.5,color="#455A64",
            bbox=dict(boxstyle="round,pad=.38",facecolor="#F7F9FA",edgecolor="#CFD8DC"))

    ax2=fig.add_subplot(grid[1]); pivot=raw.pivot_table(index="truth_label",columns="loss_stage",values="labels",fill_value=0)
    y=np.arange(len(CLASSES)); left=np.zeros(len(CLASSES)); labels={
        "proposal_miss":"proposal miss","representation_miss":"representation miss","event_geometry":"event geometry",
        "score_threshold":"threshold miss","nms":"NMS","output_cap":"output cap",
        "one_to_one_collision":"assignment collision","recovered":"recovered"}
    for stage in stages:
        values=np.array([int(pivot.loc[c,stage]) if c in pivot.index and stage in pivot.columns else 0 for c in CLASSES])
        if values.sum()==0: continue
        bars=ax2.barh(y,values,left=left,color=COLORS[stage],edgecolor="white",linewidth=.7,label=labels[stage])
        for bar,value in zip(bars,values):
            if value: ax2.text(bar.get_x()+bar.get_width()/2,bar.get_y()+bar.get_height()/2,str(value),ha="center",va="center",
                               fontsize=9,fontweight="bold",color="white" if stage!="event_geometry" else "#263238")
        left+=values
    ax2.set_yticks(y,[DISPLAY[c] for c in CLASSES]); ax2.invert_yaxis(); ax2.set_xlabel("Observed labels")
    ax2.set_title("Exclusive outcome by event class",loc="left"); clean_axis(ax2,"x")
    ax2.legend(loc="upper center",bbox_to_anchor=(.5,-.18),ncol=4,frameon=False)
    fig.subplots_adjust(left=.09,right=.985,top=.97,bottom=.13)
    fig.savefig(FIGURES/"label_loss_pipeline.png",dpi=220,facecolor="white")
    plt.close(fig)

def main():
    FIGURES.mkdir(parents=True,exist_ok=True); plot_metrics(); plot_loss_pipeline()
    print(f"Wrote aligned figures for {MODEL} to {FIGURES}")
if __name__=="__main__": main()
