"""Compare called complex-SV candidates across model output directories."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib_venn import venn3


METHODS = (
    ("classification", "Single-label NN", "#4C78A8"),
    ("multilabel", "Multilabel NN", "#F58518"),
    ("fewshot", "Few-shot proto", "#54A24B"),
)
SCAN_EVIDENCE = {"chromosome_scan", "chromosome_arm_scan"}
COLLAPSE_CLASS = {
    "non_canonical_BFB": "BFB",
    "non_canonical_chromothripsis": "chromothripsis",
}
CLASS_ORDER = ("BFB", "chromothripsis", "seismic_amplification", "TIC")


def _read_calls(method: str, path: Path) -> pd.DataFrame:
    table_path = path / "predicted_complex_sv.tsv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing called table for {method}: {table_path}")
    df = pd.read_csv(table_path, sep="\t").fillna("")
    if "called_complex_sv" in df:
        called = df["called_complex_sv"].astype(str).str.lower().isin({"true", "1", "yes"})
        df = df.loc[called].copy()
    df["method"] = method
    for col in ["candidate_id", "sample_id", "chrom", "arm", "start_bp", "end_bp", "evidence", "predicted_class", "predicted_classes"]:
        if col not in df:
            df[col] = ""
    df["candidate_key"] = df.apply(_candidate_key, axis=1)
    df["class_list"] = df.apply(_classes_for_row, axis=1)
    return df


def _candidate_key(row: pd.Series) -> str:
    parts = [
        str(row.get("sample_id", "")),
        str(row.get("chrom", "")),
        str(row.get("arm", "")),
        str(row.get("start_bp", "")),
        str(row.get("end_bp", "")),
    ]
    if not any(parts):
        return str(row.get("candidate_id", ""))
    return "|".join(parts)


def _classes_for_row(row: pd.Series) -> list[str]:
    raw = str(row.get("predicted_classes", "") or row.get("predicted_class", "")).strip()
    if not raw or raw == "none":
        return []
    parts = [part.strip() for part in raw.replace(",", ";").split(";") if part.strip() and part.strip() != "none"]
    return parts


def _collapse_class(name: str) -> str:
    return COLLAPSE_CLASS.get(str(name), str(name))


def _aggregate_method_calls(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in df.groupby("candidate_key", sort=False):
        first = group.iloc[0]
        classes = sorted({cls for classes in group["class_list"] for cls in classes})
        collapsed = sorted({_collapse_class(cls) for cls in classes})
        rows.append(
            {
                "candidate_key": key,
                "candidate_id": ";".join(sorted(set(group["candidate_id"].astype(str)))),
                "sample_id": first.get("sample_id", ""),
                "chrom": first.get("chrom", ""),
                "arm": first.get("arm", ""),
                "start_bp": first.get("start_bp", ""),
                "end_bp": first.get("end_bp", ""),
                "evidence": first.get("evidence", ""),
                "classes": ";".join(classes),
                "classes_collapsed": ";".join(collapsed),
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _scope_filter(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "all_calls":
        return df.copy()
    if scope == "unlabeled_scan":
        if "is_labeled" in df:
            unlabeled = ~df["is_labeled"].astype(str).str.lower().isin({"true", "1", "yes"})
        else:
            unlabeled = df["evidence"].astype(str).isin(SCAN_EVIDENCE)
        scan = df["evidence"].astype(str).isin(SCAN_EVIDENCE)
        return df.loc[unlabeled & scan].copy()
    raise ValueError(f"Unknown scope: {scope}")


def _sets_for_scope(method_tables: dict[str, pd.DataFrame], scope: str) -> dict[str, set[str]]:
    return {method: set(_scope_filter(df, scope)["candidate_key"]) for method, df in method_tables.items()}


def _combo_counts(sets: dict[str, set[str]]) -> pd.DataFrame:
    methods = [method for method, _, _ in METHODS]
    union = sorted(set().union(*(sets[method] for method in methods)))
    rows: list[dict[str, Any]] = []
    for key in union:
        present = tuple(method for method in methods if key in sets[method])
        rows.append(
            {
                "combination": "&".join(present),
                "n_models": len(present),
                "candidate_key": key,
                **{method: int(method in present) for method in methods},
            }
        )
    return pd.DataFrame(rows)


def _overlap_summary(sets: dict[str, set[str]], scope: str) -> pd.DataFrame:
    methods = [method for method, _, _ in METHODS]
    rows: list[dict[str, Any]] = []
    for mask in range(1, 1 << len(methods)):
        present = [methods[i] for i in range(len(methods)) if mask & (1 << i)]
        absent = [methods[i] for i in range(len(methods)) if not mask & (1 << i)]
        shared = set.intersection(*(sets[method] for method in present)) if present else set()
        for method in absent:
            shared = shared.difference(sets[method])
        rows.append(
            {
                "scope": scope,
                "combination": "&".join(present),
                "n_models": len(present),
                "count": int(len(shared)),
            }
        )
    return pd.DataFrame(rows)


def _pairwise_summary(sets: dict[str, set[str]], scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a, label_a, _ in METHODS:
        for b, label_b, _ in METHODS:
            inter = len(sets[a] & sets[b])
            union = len(sets[a] | sets[b])
            rows.append(
                {
                    "scope": scope,
                    "method_a": a,
                    "method_b": b,
                    "label_a": label_a,
                    "label_b": label_b,
                    "intersection": int(inter),
                    "union": int(union),
                    "jaccard": float(inter / union) if union else 1.0,
                }
            )
    return pd.DataFrame(rows)


def _draw_venn(summary: pd.DataFrame, scope: str, output_path: Path) -> None:
    counts = dict(zip(summary["combination"], summary["count"]))
    labels = {method: label for method, label, _ in METHODS}
    subsets = (
        int(counts.get("classification", 0)),
        int(counts.get("multilabel", 0)),
        int(counts.get("classification&multilabel", 0)),
        int(counts.get("fewshot", 0)),
        int(counts.get("classification&fewshot", 0)),
        int(counts.get("multilabel&fewshot", 0)),
        int(counts.get("classification&multilabel&fewshot", 0)),
    )
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    diagram = venn3(
        subsets=subsets,
        set_labels=(
            labels["classification"],
            labels["multilabel"],
            labels["fewshot"],
        ),
        set_colors=("#4C78A8", "#F58518", "#54A24B"),
        alpha=0.45,
        ax=ax,
    )
    for text in diagram.set_labels:
        if text is not None:
            text.set_fontsize(11)
            text.set_fontweight("bold")
    for text in diagram.subset_labels:
        if text is not None:
            text.set_fontsize(12)
            text.set_fontweight("bold")
    title = "All Called Candidates" if scope == "all_calls" else "Unlabeled Chromosome-Arm Scan Calls"
    ax.set_title(f"Pipeline6 Model Call Overlap: {title}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_upset(combo_rows: pd.DataFrame, scope: str, output_path: Path) -> None:
    if combo_rows.empty:
        return
    methods = [method for method, _, _ in METHODS]
    labels = {method: label for method, label, _ in METHODS}
    counts = combo_rows.groupby(methods, dropna=False).size().reset_index(name="count")
    counts["n_models"] = counts[methods].sum(axis=1)
    counts = counts.sort_values(["count", "n_models"], ascending=[False, False]).reset_index(drop=True)
    x = np.arange(len(counts))
    fig = plt.figure(figsize=(max(7.8, 0.75 * len(counts)), 5.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.4], hspace=0.05)
    ax_bar = fig.add_subplot(gs[0])
    ax_mat = fig.add_subplot(gs[1], sharex=ax_bar)
    ax_bar.bar(x, counts["count"], color="#333333", width=0.68)
    ax_bar.set_ylabel("Candidates")
    ax_bar.grid(axis="y", alpha=0.18)
    title = "All Calls" if scope == "all_calls" else "Unlabeled Scan Calls"
    ax_bar.set_title(f"Pipeline6 Call Combinations: {title}")
    ax_bar.tick_params(axis="x", labelbottom=False)
    for i, value in enumerate(counts["count"]):
        ax_bar.text(i, value + max(counts["count"].max() * 0.015, 0.25), str(int(value)), ha="center", va="bottom", fontsize=9)
    for row_i, method in enumerate(methods):
        y = len(methods) - 1 - row_i
        present = counts[method].astype(bool).to_numpy()
        ax_mat.scatter(x[present], np.full(present.sum(), y), s=75, color="#111111")
        ax_mat.scatter(x[~present], np.full((~present).sum(), y), s=45, color="#d0d0d0")
        for i in x[present]:
            present_rows = [len(methods) - 1 - j for j, m in enumerate(methods) if bool(counts.loc[i, m])]
            if present_rows:
                ax_mat.plot([i, i], [min(present_rows), max(present_rows)], color="#111111", linewidth=1.8)
    ax_mat.set_yticks(range(len(methods)))
    ax_mat.set_yticklabels([labels[method] for method in reversed(methods)])
    ax_mat.set_xticks(x)
    ax_mat.set_xticklabels([""] * len(x))
    ax_mat.set_ylim(-0.6, len(methods) - 0.4)
    ax_mat.spines[["top", "right", "bottom"]].set_visible(False)
    ax_mat.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _draw_jaccard(pairwise: pd.DataFrame, scope: str, output_path: Path) -> None:
    methods = [method for method, _, _ in METHODS]
    labels = [label for _, label, _ in METHODS]
    matrix = pairwise.pivot(index="method_a", columns="method_b", values="jaccard").reindex(index=methods, columns=methods).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(5.7, 4.8))
    im = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(methods)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="black", fontsize=11)
    title = "All Calls" if scope == "all_calls" else "Unlabeled Scan Calls"
    ax.set_title(f"Pairwise Jaccard: {title}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _class_distribution(method_tables: dict[str, pd.DataFrame], scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for method, label, _ in METHODS:
        scoped = _scope_filter(method_tables[method], scope)
        for _, row in scoped.iterrows():
            classes = row.get("class_list", [])
            for cls in classes:
                rows.append({"scope": scope, "method": method, "method_label": label, "class": _collapse_class(cls)})
    if not rows:
        return pd.DataFrame(columns=["scope", "method", "method_label", "class", "count"])
    return pd.DataFrame(rows).groupby(["scope", "method", "method_label", "class"], as_index=False).size().rename(columns={"size": "count"})


def _draw_class_distribution(class_counts: pd.DataFrame, scope: str, output_path: Path) -> None:
    methods = [method for method, _, _ in METHODS]
    labels = {method: label for method, label, _ in METHODS}
    classes = [cls for cls in CLASS_ORDER if cls in set(class_counts["class"])]
    classes += sorted(set(class_counts["class"]).difference(classes))
    if not classes:
        return
    table = class_counts.pivot_table(index="method", columns="class", values="count", aggfunc="sum", fill_value=0).reindex(index=methods, columns=classes, fill_value=0)
    x = np.arange(len(methods))
    bottom = np.zeros(len(methods), dtype=float)
    palette = {
        "BFB": "#4C78A8",
        "chromothripsis": "#F58518",
        "seismic_amplification": "#54A24B",
        "TIC": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    for cls in classes:
        values = table[cls].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=cls, color=palette.get(cls, None))
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([labels[method] for method in methods], rotation=20, ha="right")
    ax.set_ylabel("Class calls, collapsed")
    title = "All Calls" if scope == "all_calls" else "Unlabeled Scan Calls"
    ax.set_title(f"Predicted Class Mix: {title}")
    ax.grid(axis="y", alpha=0.18)
    ax.legend(fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _agreement_table(method_tables: dict[str, pd.DataFrame], output_path: Path) -> pd.DataFrame:
    method_aggs = {method: _aggregate_method_calls(df) for method, df in method_tables.items()}
    keys = sorted(set().union(*(set(df["candidate_key"]) for df in method_aggs.values())))
    info_by_key: dict[str, dict[str, Any]] = {}
    for df in method_aggs.values():
        for row in df.to_dict("records"):
            info_by_key.setdefault(row["candidate_key"], row)
    rows: list[dict[str, Any]] = []
    for key in keys:
        base = info_by_key[key]
        out = {
            "candidate_key": key,
            "sample_id": base.get("sample_id", ""),
            "chrom": base.get("chrom", ""),
            "arm": base.get("arm", ""),
            "start_bp": base.get("start_bp", ""),
            "end_bp": base.get("end_bp", ""),
            "evidence": base.get("evidence", ""),
        }
        n_called = 0
        class_sets: list[set[str]] = []
        for method, _, _ in METHODS:
            row = method_aggs[method].loc[method_aggs[method]["candidate_key"] == key]
            called = not row.empty
            out[f"{method}_called"] = int(called)
            if called:
                n_called += 1
                classes = str(row.iloc[0]["classes_collapsed"]).split(";") if row.iloc[0]["classes_collapsed"] else []
            else:
                classes = []
            class_sets.append(set(classes))
            out[f"{method}_classes"] = ";".join(classes)
        nonempty = [classes for classes in class_sets if classes]
        out["n_models_called"] = int(n_called)
        out["any_class_agreement"] = int(bool(nonempty) and bool(set.intersection(*nonempty))) if len(nonempty) >= 2 else 0
        rows.append(out)
    result = pd.DataFrame(rows)
    result.to_csv(output_path, sep="\t", index=False)
    return result


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dirs = {
        "classification": Path(args.classification_dir),
        "multilabel": Path(args.multilabel_dir),
        "fewshot": Path(args.fewshot_dir),
    }
    method_tables = {method: _read_calls(method, path) for method, path in input_dirs.items()}
    for method, df in method_tables.items():
        _aggregate_method_calls(df).to_csv(output_dir / f"{method}_called_candidates.tsv", sep="\t", index=False)

    all_summaries: list[pd.DataFrame] = []
    all_pairwise: list[pd.DataFrame] = []
    all_class_counts: list[pd.DataFrame] = []
    for scope in ["all_calls", "unlabeled_scan"]:
        sets = _sets_for_scope(method_tables, scope)
        combo_rows = _combo_counts(sets)
        combo_rows.to_csv(output_dir / f"candidate_combinations_{scope}.tsv", sep="\t", index=False)
        summary = _overlap_summary(sets, scope)
        pairwise = _pairwise_summary(sets, scope)
        class_counts = _class_distribution(method_tables, scope)
        summary.to_csv(output_dir / f"overlap_summary_{scope}.tsv", sep="\t", index=False)
        pairwise.to_csv(output_dir / f"pairwise_overlap_{scope}.tsv", sep="\t", index=False)
        class_counts.to_csv(output_dir / f"class_distribution_{scope}.tsv", sep="\t", index=False)
        _draw_venn(summary, scope, output_dir / f"called_overlap_venn_{scope}.png")
        _draw_upset(combo_rows, scope, output_dir / f"called_overlap_upset_{scope}.png")
        _draw_jaccard(pairwise, scope, output_dir / f"pairwise_jaccard_{scope}.png")
        _draw_class_distribution(class_counts, scope, output_dir / f"class_distribution_{scope}.png")
        all_summaries.append(summary)
        all_pairwise.append(pairwise)
        all_class_counts.append(class_counts)

    pd.concat(all_summaries, ignore_index=True).to_csv(output_dir / "overlap_summary.tsv", sep="\t", index=False)
    pd.concat(all_pairwise, ignore_index=True).to_csv(output_dir / "pairwise_overlap.tsv", sep="\t", index=False)
    pd.concat(all_class_counts, ignore_index=True).to_csv(output_dir / "class_distribution.tsv", sep="\t", index=False)
    agreement = _agreement_table(method_tables, output_dir / "model_agreement_by_candidate.tsv")
    readme = [
        "Pipeline6 model call overlap analysis",
        "",
        "Candidate overlap is keyed by sample_id, chrom, arm, start_bp, and end_bp.",
        "all_calls includes label anchors and unlabeled scan calls.",
        "unlabeled_scan keeps only unlabeled chromosome/chromosome-arm scan calls.",
        "Class distribution collapses non_canonical_BFB into BFB and non_canonical_chromothripsis into chromothripsis.",
        "",
        f"Union candidates across all called sets: {len(agreement)}",
    ]
    (output_dir / "README.txt").write_text("\n".join(readme) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classification_dir", required=True)
    parser.add_argument("--multilabel_dir", required=True)
    parser.add_argument("--fewshot_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
