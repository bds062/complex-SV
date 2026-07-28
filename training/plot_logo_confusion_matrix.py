"""Aggregate held-out LOGO candidate-region predictions into confusion matrices."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


NONE_LABEL = "none"


def split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def split_classes(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "empty", "unlabeled"}:
        return []
    return [part.strip() for part in text.replace(",", ";").split(";") if part.strip()]


def normalize_combo(value: object, class_names: list[str]) -> str:
    values = set(split_classes(value))
    if not values:
        return NONE_LABEL
    ordered = [name for name in class_names if name in values]
    ordered.extend(sorted(values.difference(ordered)))
    return ";".join(ordered)


def combo_key(label: str, class_names: list[str]) -> tuple[int, int, list[int], str]:
    if label == NONE_LABEL:
        return (0, 0, [], label)
    parts = split_classes(label)
    ranks = [class_names.index(part) if part in class_names else len(class_names) for part in parts]
    return (1, len(parts), ranks, label)


def ordered_labels(values: Iterable[str], class_names: list[str]) -> list[str]:
    return sorted(set(values), key=lambda value: combo_key(value, class_names))


def wrap_label(label: str, width: int = 19) -> str:
    if label == NONE_LABEL:
        return label
    return "\n".join(textwrap_piece(part, width) for part in label.split(";"))


def textwrap_piece(text: str, width: int) -> str:
    pieces = re.split(r"(_|-)", text)
    lines: list[str] = []
    line = ""
    for piece in pieces:
        if not piece:
            continue
        if len(line) + len(piece) > width and line:
            lines.append(line)
            line = piece
        else:
            line += piece
    if line:
        lines.append(line)
    return "\n".join(lines)


def load_logo_predictions(runs_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(runs_dir.glob("logo_*/test_predictions.tsv")):
        df = pd.read_csv(path, sep="\t")
        df.insert(0, "logo_fold", path.parent.name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No logo_*/test_predictions.tsv files found under {runs_dir}")
    return pd.concat(frames, ignore_index=True)


def add_confusion_columns(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["true_combo"] = out.get("true_classes", pd.Series("", index=out.index)).map(lambda value: normalize_combo(value, class_names))
    pred_source = out["predicted_classes"] if "predicted_classes" in out else out.get("predicted_class", pd.Series("", index=out.index))
    out["pred_combo"] = pred_source.map(lambda value: normalize_combo(value, class_names))
    out["is_labeled_bool"] = boolish(out.get("is_labeled", pd.Series(False, index=out.index)))
    out["is_background_bool"] = boolish(out.get("is_background_chromosome", pd.Series(False, index=out.index)))
    out["class_exact_bool"] = boolish(out.get("class_exact_match", pd.Series(False, index=out.index)))
    out["objectness_correct_bool"] = boolish(out.get("objectness_correct", pd.Series(False, index=out.index)))

    def classify(row: pd.Series) -> str:
        true_set = set(split_classes(row["true_combo"]))
        pred_set = set(split_classes(row["pred_combo"]))
        if not true_set and not pred_set:
            return "true_empty_pred_none"
        if not true_set and pred_set:
            return "empty_false_positive"
        if true_set and not pred_set:
            return "false_negative"
        if true_set == pred_set:
            return "exact"
        if true_set & pred_set:
            return "partial_overlap"
        return "wrong_class"

    out["aggregate_error_type"] = out.apply(classify, axis=1)
    out["missing_classes"] = out.apply(
        lambda row: ";".join(name for name in class_names if name in set(split_classes(row["true_combo"])) - set(split_classes(row["pred_combo"]))),
        axis=1,
    )
    out["extra_classes"] = out.apply(
        lambda row: ";".join(name for name in class_names if name in set(split_classes(row["pred_combo"])) - set(split_classes(row["true_combo"]))),
        axis=1,
    )
    return out


def confusion_table(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    matrix = pd.crosstab(df["true_combo"], df["pred_combo"])
    return matrix.reindex(index=labels, columns=labels, fill_value=0).astype(int)


def row_fraction(matrix: pd.DataFrame) -> pd.DataFrame:
    denom = matrix.sum(axis=1).replace(0, np.nan)
    return matrix.div(denom, axis=0).fillna(0.0)


def plot_heatmap(matrix: pd.DataFrame, output_path: Path, title: str, normalize: bool = False) -> None:
    values = matrix.to_numpy(dtype=float)
    if normalize:
        plot_values = row_fraction(matrix).to_numpy(dtype=float)
        cmap = "Blues"
        vmax = 1.0
    else:
        plot_values = values
        cmap = "YlOrRd"
        vmax = max(1.0, float(np.nanmax(plot_values)) if plot_values.size else 1.0)

    n_rows, n_cols = matrix.shape
    fig_w = max(9.0, n_cols * 1.05)
    fig_h = max(7.0, n_rows * 0.85)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(plot_values, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, pad=14)
    ax.set_xlabel("Predicted class set")
    ax.set_ylabel("True class set")
    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_rows))
    ax.set_xticklabels([wrap_label(label) for label in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels([wrap_label(label) for label in matrix.index], fontsize=8)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(n_rows):
        for j in range(n_cols):
            count = int(values[i, j])
            if count == 0:
                continue
            if normalize:
                text = f"{count}\n{plot_values[i, j]:.0%}"
            else:
                text = str(count)
            color = "white" if plot_values[i, j] > (0.55 * vmax if not normalize else 0.55) else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Row fraction" if normalize else "Count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def biggest_confusions(df: pd.DataFrame, labels: list[str]) -> pd.DataFrame:
    totals = df.groupby("true_combo").size().rename("true_total")
    rows: list[dict[str, object]] = []
    off = df[df["true_combo"] != df["pred_combo"]].copy()
    grouped = off.groupby(["true_combo", "pred_combo", "aggregate_error_type", "missing_classes", "extra_classes"], dropna=False)
    for keys, part in grouped:
        true_combo, pred_combo, error_type, missing, extra = keys
        examples = []
        for _, row in part.head(8).iterrows():
            examples.append(f"{row.get('sample_id', '')}:{row.get('chrom', '')}{row.get('arm', '')}:{row.get('start_bp', '')}-{row.get('end_bp', '')}")
        count = int(len(part))
        true_total = int(totals.get(true_combo, count))
        rows.append(
            {
                "true_combo": true_combo,
                "pred_combo": pred_combo,
                "count": count,
                "true_total": true_total,
                "row_fraction": count / true_total if true_total else math.nan,
                "error_type": error_type,
                "missing_classes": missing,
                "extra_classes": extra,
                "examples": ";".join(examples),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["true_combo"] = pd.Categorical(out["true_combo"], categories=labels, ordered=True)
    out["pred_combo"] = pd.Categorical(out["pred_combo"], categories=labels, ordered=True)
    return out.sort_values(["count", "row_fraction", "true_combo", "pred_combo"], ascending=[False, False, True, True]).reset_index(drop=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs_dir", required=True, help="Directory containing logo_*/test_predictions.tsv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--class_names", default="ecDNA,Seismic_Amplification,chromothripsis,BFB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    class_names = split_csv(args.class_names)
    df = add_confusion_columns(load_logo_predictions(runs_dir), class_names)
    df.to_csv(output_dir / "logo_aggregate_predictions.tsv", sep="\t", index=False)

    labels_all = ordered_labels(list(df["true_combo"]) + list(df["pred_combo"]), class_names)
    all_matrix = confusion_table(df, labels_all)
    all_matrix.to_csv(output_dir / "logo_aggregate_confusion_matrix.tsv", sep="\t")
    row_fraction(all_matrix).to_csv(output_dir / "logo_aggregate_confusion_matrix_row_fraction.tsv", sep="\t")
    plot_heatmap(all_matrix, output_dir / "logo_aggregate_confusion_matrix_counts.png", "Aggregate LOGO Confusion Matrix, All Candidate Rows")
    plot_heatmap(all_matrix, output_dir / "logo_aggregate_confusion_matrix_row_fraction.png", "Aggregate LOGO Confusion Matrix, Row-Normalized", normalize=True)

    labeled = df[df["is_labeled_bool"]].copy()
    labels_labeled = ordered_labels(list(labeled["true_combo"]) + list(labeled["pred_combo"]), class_names)
    labeled_matrix = confusion_table(labeled, labels_labeled)
    labeled_matrix.to_csv(output_dir / "logo_aggregate_confusion_matrix_labeled_only.tsv", sep="\t")
    row_fraction(labeled_matrix).to_csv(output_dir / "logo_aggregate_confusion_matrix_labeled_only_row_fraction.tsv", sep="\t")
    plot_heatmap(
        labeled_matrix,
        output_dir / "logo_aggregate_confusion_matrix_labeled_only_counts.png",
        "Aggregate LOGO Confusion Matrix, Labeled Complex-SV Rows",
    )
    plot_heatmap(
        labeled_matrix,
        output_dir / "logo_aggregate_confusion_matrix_labeled_only_row_fraction.png",
        "Aggregate LOGO Confusion Matrix, Labeled Complex-SV Rows, Row-Normalized",
        normalize=True,
    )

    confusions = biggest_confusions(df, labels_all)
    confusions.to_csv(output_dir / "logo_biggest_confusions.tsv", sep="\t", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "n_rows", "value": int(len(df))},
            {"metric": "n_labeled", "value": int(df["is_labeled_bool"].sum())},
            {"metric": "n_background", "value": int(df["is_background_bool"].sum())},
            {"metric": "n_exact_combo", "value": int((df["true_combo"] == df["pred_combo"]).sum())},
            {"metric": "n_off_diagonal", "value": int((df["true_combo"] != df["pred_combo"]).sum())},
            {"metric": "n_partial_overlap", "value": int((df["aggregate_error_type"] == "partial_overlap").sum())},
            {"metric": "n_wrong_class", "value": int((df["aggregate_error_type"] == "wrong_class").sum())},
            {"metric": "n_false_negative", "value": int((df["aggregate_error_type"] == "false_negative").sum())},
            {"metric": "n_empty_false_positive", "value": int((df["aggregate_error_type"] == "empty_false_positive").sum())},
        ]
    )
    summary.to_csv(output_dir / "logo_aggregate_confusion_summary.tsv", sep="\t", index=False)
    print(f"Wrote aggregate LOGO confusion outputs to {output_dir}")


if __name__ == "__main__":
    main()
