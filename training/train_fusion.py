"""Supervised fusion training entry point.

This scaffold intentionally refuses to train on the current weak 10-anchor,
single-class setup. Use discovery.embed_corpus prototype mode until there are
enough interval-level examples and at least two classes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def validate_training_labels(labels_file: str | Path, n_way: int, k_shot: int, n_query: int) -> pd.DataFrame:
    labels = pd.read_csv(labels_file, sep="\t").fillna("")
    labels = labels[labels["sv_class"].astype(str) != ""].copy()
    class_counts = labels["sv_class"].astype(str).value_counts()
    enough = class_counts[class_counts >= int(k_shot) + int(n_query)]
    if len(enough) < int(n_way):
        raise RuntimeError(
            "Not enough labels for supervised episodic fusion training. "
            f"Need at least {n_way} classes with {k_shot + n_query} examples each; "
            f"observed counts: {class_counts.to_dict()}. "
            "Run prototype mode with discovery.embed_corpus instead."
        )
    if "label_scope" in labels:
        interval_level = labels["label_scope"].astype(str).isin({"interval", "region", "breakpoint"}).all()
    else:
        interval_level = False
    if not interval_level:
        raise RuntimeError(
            "Supervised boundary/fusion training requires interval-level labels. "
            "Current labels include broad chromosome-scope anchors; use prototype mode."
        )
    return labels


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels_file", required=True)
    parser.add_argument("--cn_checkpoint", required=True)
    parser.add_argument("--graph_checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_way", type=int, default=2)
    parser.add_argument("--k_shot", type=int, default=5)
    parser.add_argument("--n_query", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validate_training_labels(args.labels_file, args.n_way, args.k_shot, args.n_query)
    raise NotImplementedError("Full supervised fusion optimization will be enabled after sufficient labels are available.")


if __name__ == "__main__":
    main()
