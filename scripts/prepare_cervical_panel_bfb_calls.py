#!/usr/bin/env python3
"""Convert the curated cervical-panel BFB CSV into the standard BFB call TSV."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


REGION_PATTERN = re.compile(r"^(chr[^:\s]+):(\d+)-(\d+)$", re.IGNORECASE)


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input_csv)
    output_path = Path(args.output_tsv)
    audit_path = Path(args.audit_tsv)
    summary_path = Path(args.summary_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_path).fillna("")
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {input_path}; found {frame.shape[1]}")

    sample_column = str(frame.columns[0])
    region_column = str(frame.columns[1])
    call_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    seen: set[tuple[str, str, int, int]] = set()

    for row_index, row in frame.iterrows():
        sample = str(row.iloc[0]).strip()
        region_text = str(row.iloc[1]).strip()
        status = "included"
        reason = ""
        match = REGION_PATTERN.fullmatch(region_text)
        if not sample:
            status, reason = "excluded", "missing sample name"
        elif region_text.lower().startswith("no amplification"):
            status, reason = "excluded", "second column explicitly reports no amplification"
        elif match is None:
            status, reason = "excluded", "second column is not a chr:start-end interval"

        if status == "included" and match is not None:
            chrom = match.group(1)
            start, end = sorted((int(match.group(2)), int(match.group(3))))
            key = (sample, chrom, start, end)
            if key in seen:
                status, reason = "excluded", "duplicate sample and interval"
            else:
                seen.add(key)
                call_rows.append(
                    {
                        "sample_id": sample,
                        "call_id": f"CervicalPanelBFB:{row_index + 2}",
                        "amplified_region": f"{chrom}:{start}-{end}",
                        "chrom": chrom,
                        "start": start,
                        "end": end,
                        "source": "CervicalPanelBFB",
                        "source_csv_row": int(row_index + 2),
                    }
                )
        audit_rows.append(
            {
                "source_csv_row": int(row_index + 2),
                "sample_id": sample,
                "second_column_value": region_text.replace("\n", " "),
                "status": status,
                "reason": reason,
            }
        )

    calls = pd.DataFrame(call_rows)
    audit = pd.DataFrame(audit_rows)
    calls.to_csv(output_path, sep="\t", index=False)
    audit.to_csv(audit_path, sep="\t", index=False)
    summary = {
        "input_csv": str(input_path),
        "sample_column": sample_column,
        "region_column": region_column,
        "coordinate_column_index_one_based": 2,
        "input_rows": int(len(frame)),
        "included_calls": int(len(calls)),
        "included_samples": int(calls["sample_id"].nunique()) if not calls.empty else 0,
        "excluded_rows": int(audit["status"].eq("excluded").sum()),
        "rule": "Use the second CSV column as the BFB interval; exclude explicit no-amplification, malformed, and duplicate rows.",
        "prediction_column_used": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--audit_tsv", required=True)
    parser.add_argument("--summary_json", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
