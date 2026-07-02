#!/usr/bin/env python
"""Plot class-specific UMAP highlights from candidate-region embeddings."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd


DEFAULT_CLASS_ORDER = ["BFB", "chromothripsis", "ecDNA", "Seismic_Amplification"]
CLASS_COLORS = {
    "BFB": "#E15759",
    "chromothripsis": "#4E79A7",
    "ecDNA": "#59A14F",
    "Seismic_Amplification": "#F28E2B",
}
FALLBACK_COLORS = [
    "#76B7B2",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
]
NULL_LABELS = {"", "nan", "none", "null", "na"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one UMAP/PCA projection plot per complex-SV class. Each plot has "
            "two panels: rows containing the class and rows labeled only with that class."
        )
    )
    parser.add_argument(
        "embedding",
        help="Classifier output directory containing embeddings.npz, or an embeddings.npz file.",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="candidate_embeddings.tsv matching embeddings.npz. Defaults to <embedding_dir>/candidate_embeddings.tsv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for class-specific UMAP plots. Defaults to <embedding_dir>/class_specific_umaps.",
    )
    parser.add_argument(
        "--label-column",
        default="sv_classes",
        help="Metadata column containing true labels. Defaults to sv_classes.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help=(
            "Classes to plot. Accepts space-separated names or comma/semicolon-separated lists. "
            "Defaults to observed base classes, ordered as BFB, chromothripsis, ecDNA, Seismic_Amplification."
        ),
    )
    parser.add_argument(
        "--title-prefix",
        default="",
        help="Optional prefix added to plot titles.",
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="Annotate highlighted points when a panel has at most --max-annotations points.",
    )
    parser.add_argument(
        "--max-annotations",
        type=int,
        default=35,
        help="Maximum highlighted points to annotate per panel when --annotate is set.",
    )
    return parser.parse_args(argv)


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path | None, Path]:
    embedding_path = Path(args.embedding)
    if embedding_path.is_dir():
        embedding_npz = embedding_path / "embeddings.npz"
        base_dir = embedding_path
    else:
        embedding_npz = embedding_path
        base_dir = embedding_path.parent

    if not embedding_npz.exists():
        raise FileNotFoundError(f"Embedding NPZ not found: {embedding_npz}")

    metadata_path = Path(args.metadata) if args.metadata else base_dir / "candidate_embeddings.tsv"
    if not metadata_path.exists():
        metadata_path = None

    output_dir = Path(args.output_dir) if args.output_dir else base_dir / "class_specific_umaps"
    output_dir.mkdir(parents=True, exist_ok=True)
    return embedding_npz, metadata_path, output_dir


def _decode_array(values: np.ndarray) -> np.ndarray:
    out: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            text = value.decode("utf-8", errors="replace")
        else:
            text = str(value)
        if text.lower() == "nan":
            text = ""
        out.append(text)
    return np.asarray(out, dtype=object)


def load_inputs(embedding_npz: Path, metadata_path: Path | None) -> tuple[np.ndarray, pd.DataFrame]:
    arrays = np.load(embedding_npz, allow_pickle=True)
    if "embeddings" not in arrays:
        raise KeyError(f"{embedding_npz} does not contain an 'embeddings' array")
    embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={embeddings.shape}")
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

    if metadata_path is not None:
        metadata = pd.read_csv(metadata_path, sep="\t").fillna("")
    else:
        cols: dict[str, np.ndarray] = {}
        for key in arrays.files:
            if key == "embeddings":
                continue
            values = np.asarray(arrays[key])
            if values.shape[:1] == embeddings.shape[:1]:
                cols[key] = _decode_array(values)
        metadata = pd.DataFrame(cols)

    if len(metadata) != embeddings.shape[0]:
        raise ValueError(
            f"Metadata row count ({len(metadata)}) does not match embeddings ({embeddings.shape[0]})"
        )
    return embeddings, metadata


def reduce_embeddings_2d(embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    n = int(embeddings.shape[0])
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32), "empty"
    if n == 1:
        return np.zeros((1, 2), dtype=np.float32), "single point"

    if n >= 5:
        try:
            import umap

            xy = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            ).fit_transform(embeddings)
            return np.asarray(xy, dtype=np.float32), "UMAP"
        except Exception as exc:
            print(f"[plot_class_specific_umaps] UMAP failed, falling back to PCA: {exc}")

    from sklearn.decomposition import PCA

    n_components = min(2, embeddings.shape[0], embeddings.shape[1])
    xy_small = PCA(n_components=n_components, random_state=42).fit_transform(embeddings)
    xy = np.zeros((embeddings.shape[0], 2), dtype=np.float32)
    xy[:, :n_components] = xy_small
    return xy, "PCA"


def split_base_classes(value: object) -> list[str]:
    text = str(value).strip()
    if text.lower() in NULL_LABELS:
        return []

    classes: list[str] = []
    seen: set[str] = set()
    for part in text.replace(",", ";").split(";"):
        token = part.strip()
        if not token:
            continue
        base = token.split(":", 1)[0].strip()
        if base.lower() in NULL_LABELS or base in seen:
            continue
        classes.append(base)
        seen.add(base)
    return classes


def parse_class_args(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    classes: list[str] = []
    seen: set[str] = set()
    for value in values:
        for token in re.split(r"[;,]", str(value)):
            cls = token.strip()
            if not cls or cls in seen:
                continue
            classes.append(cls)
            seen.add(cls)
    return classes


def choose_label_column(metadata: pd.DataFrame, requested: str) -> str:
    if requested in metadata.columns:
        return requested
    for candidate in ["sv_classes", "sv_class", "true_classes", "raw_sv_classes", "raw_true_classes"]:
        if candidate in metadata.columns:
            print(
                f"[plot_class_specific_umaps] Requested label column {requested!r} not found; using {candidate!r}"
            )
            return candidate
    raise ValueError(
        "Could not find a label column. Pass --label-column or provide metadata with sv_classes/sv_class."
    )


def infer_classes(label_sets: list[set[str]], requested: list[str]) -> list[str]:
    if requested:
        return requested
    observed = {cls for labels in label_sets for cls in labels}
    ordered = [cls for cls in DEFAULT_CLASS_ORDER if cls in observed]
    ordered.extend(sorted(observed.difference(ordered), key=str.lower))
    if not ordered:
        raise ValueError("No labeled complex-SV classes were found in the metadata.")
    return ordered


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("_") or "class"


def class_color(class_name: str, index: int) -> str:
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]
    return FALLBACK_COLORS[index % len(FALLBACK_COLORS)]


def candidate_label(row: pd.Series) -> str:
    sample = str(row.get("sample_id", "")).strip()
    chrom = str(row.get("chrom", "")).strip()
    arm = str(row.get("arm", "")).strip()
    start = str(row.get("start_bp", row.get("start", ""))).strip()
    parts = [part for part in [sample, chrom + arm if chrom else arm, start] if part]
    if parts:
        return "_".join(parts)
    return str(row.get("candidate_id", "")).strip()


def axis_limits(xy: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_min, x_max = float(np.min(xy[:, 0])), float(np.max(xy[:, 0]))
    y_min, y_max = float(np.min(xy[:, 1])), float(np.max(xy[:, 1]))
    x_pad = max((x_max - x_min) * 0.06, 0.5)
    y_pad = max((y_max - y_min) * 0.06, 0.5)
    return (x_min - x_pad, x_max + x_pad), (y_min - y_pad, y_max + y_pad)


def plot_panel(
    ax: plt.Axes,
    xy: np.ndarray,
    metadata: pd.DataFrame,
    mask: np.ndarray,
    class_name: str,
    color: str,
    title: str,
    annotate: bool,
    max_annotations: int,
) -> None:
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c="#C9CED6",
        s=22,
        alpha=0.42,
        linewidths=0,
        label=f"other / unlabeled (n={int((~mask).sum())})",
    )
    if bool(mask.any()):
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            c=color,
            s=58,
            alpha=0.94,
            edgecolors="black",
            linewidths=0.45,
            label=f"{class_name} (n={int(mask.sum())})",
        )
        if annotate and int(mask.sum()) <= max_annotations:
            highlight_rows = metadata.reset_index(drop=True).loc[mask]
            highlight_xy = xy[mask]
            for (_, row), point in zip(highlight_rows.iterrows(), highlight_xy, strict=False):
                ax.annotate(
                    candidate_label(row),
                    (point[0], point[1]),
                    fontsize=6.5,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

    ax.set_title(title)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, frameon=True)


def plot_class_umap(
    xy: np.ndarray,
    metadata: pd.DataFrame,
    label_sets: list[set[str]],
    class_name: str,
    class_index: int,
    method: str,
    output_path: Path,
    title_prefix: str,
    annotate: bool,
    max_annotations: int,
) -> tuple[int, int]:
    contains = np.asarray([class_name in labels for labels in label_sets], dtype=bool)
    only = np.asarray([labels == {class_name} for labels in label_sets], dtype=bool)
    color = class_color(class_name, class_index)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.4), sharex=True, sharey=True)
    prefix = f"{title_prefix.strip()} " if title_prefix.strip() else ""
    fig.suptitle(f"{prefix}{class_name} embedding projection ({method})", fontsize=13)

    plot_panel(
        axes[0],
        xy,
        metadata,
        contains,
        class_name,
        color,
        f"Contains {class_name}",
        annotate,
        max_annotations,
    )
    plot_panel(
        axes[1],
        xy,
        metadata,
        only,
        class_name,
        color,
        f"Only {class_name}",
        annotate,
        max_annotations,
    )
    xlim, ylim = axis_limits(xy)
    for ax in axes:
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return int(contains.sum()), int(only.sum())


def write_coordinates(
    xy: np.ndarray,
    metadata: pd.DataFrame,
    label_sets: list[set[str]],
    output_path: Path,
) -> None:
    out = metadata.copy()
    out.insert(0, "umap_x", xy[:, 0])
    out.insert(1, "umap_y", xy[:, 1])
    out.insert(2, "base_classes", [";".join(sorted(labels)) for labels in label_sets])
    out.to_csv(output_path, sep="\t", index=False)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    embedding_npz, metadata_path, output_dir = resolve_paths(args)
    embeddings, metadata = load_inputs(embedding_npz, metadata_path)
    label_column = choose_label_column(metadata, args.label_column)
    label_sets = [set(split_base_classes(value)) for value in metadata[label_column].tolist()]
    classes = infer_classes(label_sets, parse_class_args(args.classes))

    xy, method = reduce_embeddings_2d(embeddings)
    write_coordinates(xy, metadata, label_sets, output_dir / "class_specific_umap_coordinates.tsv")

    summary_rows: list[dict[str, object]] = []
    for class_index, class_name in enumerate(classes):
        output_path = output_dir / f"class_umap_{safe_name(class_name)}.png"
        n_contains, n_only = plot_class_umap(
            xy,
            metadata,
            label_sets,
            class_name,
            class_index,
            method,
            output_path,
            args.title_prefix,
            args.annotate,
            int(args.max_annotations),
        )
        summary_rows.append(
            {
                "class": class_name,
                "n_contains_class": n_contains,
                "n_only_class": n_only,
                "projection_method": method,
                "output_png": str(output_path),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "class_specific_umap_summary.tsv", sep="\t", index=False)
    print(f"[plot_class_specific_umaps] Wrote {len(summary)} plots to {output_dir}")


if __name__ == "__main__":
    main()
