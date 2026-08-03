#!/usr/bin/env python3
"""Discover completed Wakhan/Severus samples and append them to a manifest."""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path


FIELDS = ("sample_id", "wakhan_sample_id", "wakhan_root", "severus_vcf")
SCORE_DIRECTORY = re.compile(
    r"^(?P<first>\d+(?:\.\d+)?)_(?P<second>\d+(?:\.\d+)?)_(?P<third>\d+(?:\.\d+)?)$"
)
HP1_SUFFIX = "_copynumbers_segments_HP_1.bed"


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != list(FIELDS):
            raise ValueError(
                f"manifest columns must be exactly {list(FIELDS)}; observed {reader.fieldnames}"
            )
        rows = [{field: str(row[field]).strip() for field in FIELDS} for row in reader]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("manifest contains duplicate sample_id values")
    return rows


def score_key(path: Path) -> tuple[float, float, float, str] | None:
    match = SCORE_DIRECTORY.fullmatch(path.name)
    if not match:
        return None
    # The third value is the primary Wakhan solution score. The remaining
    # values provide deterministic tie-breaking only.
    return (
        float(match.group("third")),
        float(match.group("second")),
        float(match.group("first")),
        path.name,
    )


def highest_scoring_solution(sample_dir: Path) -> Path:
    wakhan_dir = sample_dir / "sv_cna_v2" / "wakhan"
    if not wakhan_dir.is_dir():
        raise FileNotFoundError(f"Wakhan directory not found: {wakhan_dir}")
    candidates = [
        (key, path)
        for path in wakhan_dir.iterdir()
        if path.is_dir() and (key := score_key(path)) is not None
    ]
    if not candidates:
        raise FileNotFoundError(f"no numeric Wakhan solution directories found in {wakhan_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_wakhan_root(solution_dir: Path) -> tuple[str, Path]:
    bed_dir = solution_dir / "bed_output"
    hp1_files = sorted(bed_dir.glob(f"*{HP1_SUFFIX}"))
    valid: list[Path] = []
    for hp1 in hp1_files:
        root = hp1.with_name(hp1.name[: -len(HP1_SUFFIX)])
        hp2 = root.with_name(f"{root.name}_copynumbers_segments_HP_2.bed")
        cna_vcf = solution_dir / "vcf_output" / f"{root.name}_wakhan_cna_integers.vcf"
        if hp2.is_file() and cna_vcf.is_file():
            valid.append(root)
    if len(valid) != 1:
        raise ValueError(
            f"expected one complete HP1/HP2/CNA-VCF Wakhan root in {bed_dir}; found {len(valid)}"
        )
    return valid[0].name, valid[0]


def build_row(manifest_id: str, sample_dir: Path) -> dict[str, str]:
    solution = highest_scoring_solution(sample_dir)
    wakhan_sample_id, wakhan_root = resolve_wakhan_root(solution)
    severus_vcf = sample_dir / "sv_cna_v2" / "severus" / "somatic_SVs" / "severus_somatic.vcf"
    if not severus_vcf.is_file():
        raise FileNotFoundError(f"Severus VCF not found: {severus_vcf}")
    return {
        "sample_id": manifest_id,
        "wakhan_sample_id": wakhan_sample_id,
        "wakhan_root": str(wakhan_root.absolute()),
        "severus_vcf": str(severus_vcf.absolute()),
    }


def parse_sample_spec(value: str) -> tuple[str, str]:
    if "=" in value:
        manifest_id, directory_name = value.split("=", 1)
    else:
        manifest_id = directory_name = value
    if not manifest_id or not directory_name or "/" in directory_name:
        raise argparse.ArgumentTypeError("sample must be SAMPLE_ID or MANIFEST_ID=DIRECTORY_NAME")
    return manifest_id, directory_name


def sample_directory_is_recorded(sample_dir: Path, rows: list[dict[str, str]]) -> bool:
    wanted = sample_dir.resolve()
    for row in rows:
        root = Path(row["wakhan_root"]).resolve()
        if root == wanted or wanted in root.parents:
            return True
    return False


def atomic_write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="Current four-column TSV manifest.")
    parser.add_argument(
        "--samples-root",
        action="append",
        required=True,
        type=Path,
        help="Directory whose immediate children are sample run directories; repeatable.",
    )
    parser.add_argument(
        "--sample",
        action="append",
        type=parse_sample_spec,
        help="Only add this sample. Accepts SAMPLE_ID or MANIFEST_ID=DIRECTORY_NAME; repeatable.",
    )
    parser.add_argument("--output", type=Path, help="Write a new manifest instead of updating MANIFEST in place.")
    parser.add_argument("--dry-run", action="store_true", help="Report additions without writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    existing_ids = {row["sample_id"] for row in rows}
    additions: list[dict[str, str]] = []
    skipped: list[tuple[str, str]] = []

    for samples_root in args.samples_root:
        if not samples_root.is_dir():
            raise FileNotFoundError(f"samples root not found: {samples_root}")

    if args.sample:
        for manifest_id, directory_name in args.sample:
            if manifest_id in existing_ids:
                continue
            errors: list[str] = []
            for samples_root in args.samples_root:
                sample_dir = samples_root / directory_name
                if not sample_dir.is_dir():
                    continue
                if sample_directory_is_recorded(sample_dir, rows + additions):
                    break
                try:
                    row = build_row(manifest_id, sample_dir)
                except (FileNotFoundError, ValueError) as error:
                    errors.append(str(error))
                    continue
                additions.append(row)
                existing_ids.add(manifest_id)
                break
            else:
                detail = "; ".join(errors) if errors else "sample directory was not found under any root"
                raise RuntimeError(f"cannot add requested sample {manifest_id}: {detail}")
    else:
        for samples_root in args.samples_root:
            requested = [(path.name, path) for path in sorted(samples_root.iterdir()) if path.is_dir()]
            for manifest_id, sample_dir in requested:
                if manifest_id in existing_ids or sample_directory_is_recorded(sample_dir, rows + additions):
                    continue
                try:
                    row = build_row(manifest_id, sample_dir)
                except (FileNotFoundError, ValueError) as error:
                    skipped.append((manifest_id, str(error)))
                    continue
                additions.append(row)
                existing_ids.add(manifest_id)

    additions.sort(key=lambda row: row["sample_id"])
    for row in additions:
        solution = Path(row["wakhan_root"]).parents[1].name
        third_score = score_key(Path(solution))[0]
        print(f"ADD\t{row['sample_id']}\tthird_score={third_score:g}\t{row['wakhan_root']}")
    print(f"Existing rows: {len(rows)}; additions: {len(additions)}; incomplete directories skipped: {len(skipped)}")

    if args.dry_run:
        return
    destination = args.output or args.manifest
    atomic_write(destination, rows + additions)
    print(f"Wrote {len(rows) + len(additions)} rows to {destination}")


if __name__ == "__main__":
    main()
