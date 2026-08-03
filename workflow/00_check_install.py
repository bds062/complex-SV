#!/usr/bin/env python3
"""Validate the Python environment and packaged model artifacts."""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


REQUIRED_MODULES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scikit-learn": "sklearn",
    "PyTorch": "torch",
    "PyTorch Geometric": "torch_geometric",
    "matplotlib": "matplotlib",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: inferred from this script).",
    )
    parser.add_argument("--skip-models", action="store_true", help="Only check Python dependencies.")
    args = parser.parse_args()

    failures: list[str] = []
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    for display_name, module_name in REQUIRED_MODULES.items():
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"[ok] {display_name}: {version}")
        except Exception as exc:
            failures.append(f"{display_name}: {exc}")
            print(f"[missing] {display_name}: {exc}")

    if not args.skip_models:
        expected = [
            args.repo / "models/pretrained_featurizer/cn_encoder.pt",
            args.repo / "models/pretrained_featurizer/sv_graph_encoder.pt",
            args.repo / "models/localization_all48/model.pt",
            args.repo / "models/localization_all48/decoder_calibration.tsv",
            args.repo / "models/localization_all48/metadata.json",
            args.repo / "models/localization_all48/cv_epoch_selection.tsv",
            args.repo / "models/localization_all48/cv_decoder_selections.tsv",
            args.repo / "models/chromosome_all48/model.pt",
            args.repo / "models/localization_loo",
            args.repo / "models/chromosome_fivefold",
        ]
        for path in expected:
            if path.exists():
                print(f"[ok] model artifact: {path.relative_to(args.repo)}")
            else:
                failures.append(f"missing model artifact: {path}")
                print(f"[missing] model artifact: {path.relative_to(args.repo)}")

    if failures:
        print("\nEnvironment validation failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        raise SystemExit(1)
    print("\nInstallation and packaged artifacts are ready.")


if __name__ == "__main__":
    main()
