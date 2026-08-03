#!/usr/bin/env python3
"""Build the 37 safe whole-chromosome Wakhan/Severus summary features."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from genomic_features.severus_parser import parse_severus  # noqa: E402
from genomic_features.wakhan_parser import parse_wakhan  # noqa: E402

from genomic_features.chromosome_features import summarize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, sep="\t").fillna("")
    rows = []
    for record in manifest.to_dict("records"):
        sample_id = str(record["sample_id"])
        cn = parse_wakhan(record["wakhan_root"])
        sv = parse_severus(record["severus_vcf"], sample_id=sample_id)
        for chrom, cn_chrom in cn.groupby("chrom", sort=False):
            sv_chrom = sv[sv.chrom.astype(str) == str(chrom)]
            rows.append(summarize(sample_id, str(chrom), cn_chrom, sv_chrom))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    main()
