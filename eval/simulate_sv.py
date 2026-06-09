"""Small synthetic SV generators for smoke-testing the pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def simulate_chromothripsis(
    chrom: str,
    region_start: int,
    region_end: int,
    n_fragments: int = 20,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cuts = np.sort(rng.integers(region_start, region_end, size=max(2, n_fragments - 1)))
    edges = np.concatenate([[region_start], cuts, [region_end]])
    segments = []
    svs = []
    for i in range(len(edges) - 1):
        cn_total = 1.0 if i % 2 == 0 else 3.0
        segments.append(
            {
                "sample_id": "synthetic_chromothripsis",
                "chrom": chrom,
                "start": int(edges[i]),
                "end": int(edges[i + 1]),
                "cn_total": cn_total,
                "cn_hp1": cn_total,
                "cn_hp2": 0.0,
                "log_coverage_total": 1.0,
                "coverage_hp1_fraction": 1.0,
                "coverage_hp2_fraction": 0.0,
                "confidence_hp1": 1.0,
                "confidence_hp2": 1.0,
                "loh": 1,
                "allele_imbalance": 1.0,
                "breakpoint_count": 1,
            }
        )
        if i < len(edges) - 2:
            svs.append(
                {
                    "sample_id": "synthetic_chromothripsis",
                    "sv_id": f"ct_{i}",
                    "mate_id": "",
                    "cluster_id": "ct_cluster",
                    "phase_set": 1,
                    "chrom": chrom,
                    "pos": int(edges[i + 1] - 1),
                    "end": int(min(edges[i + 1] + 1, region_end)),
                    "sv_type_str": "BND",
                    "sv_type": 3,
                    "maj_gt": 1,
                    "qual": 60.0,
                    "log_svlen": 0.0,
                    "mapq": 60.0,
                    "vaf_mean": 0.4,
                    "vaf_std": 0.0,
                    "hvaf_hp1": 0.4,
                    "hvaf_hp2": 0.0,
                    "hvaf_unph": 0.0,
                    "dr_mean": 20.0,
                    "dv_mean": 12.0,
                    "supp_total": 12.0,
                    "supp_hp1": 12.0,
                    "supp_hp2": 0.0,
                    "ref_total": 20.0,
                    "phase_balance": 1.0,
                    "n_samples_gt": 1.0,
                    "is_precise": 1.0,
                    "is_vntr": 0.0,
                    "is_bnd": 1.0,
                    "has_phase": 1.0,
                    "hp_concordant": 0.0,
                    "strand_1_plus": 1.0,
                    "strand_1_minus": 0.0,
                    "strand_2_plus": 0.0,
                    "strand_2_minus": 1.0,
                }
            )
    return pd.DataFrame(segments), pd.DataFrame(svs)


def simulate_bfb(chrom: str, fold_back_site: int, n_cycles: int = 4, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = max(0, int(fold_back_site) - 5_000_000)
    end = int(fold_back_site) + 5_000_000
    wakhan, severus = simulate_chromothripsis(chrom, start, end, n_fragments=max(6, n_cycles * 2), seed=seed)
    wakhan["sample_id"] = "synthetic_bfb"
    severus["sample_id"] = "synthetic_bfb"
    wakhan["cn_total"] = np.linspace(2, 2 + n_cycles, len(wakhan))
    wakhan["cn_hp1"] = wakhan["cn_total"]
    wakhan["cn_hp2"] = 0.0
    severus["sv_type_str"] = "DUP"
    severus["sv_type"] = 2
    severus["is_bnd"] = 0.0
    return wakhan, severus


def write_simulated_wakhan(wakhan_df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    wakhan_df.to_csv(path, sep="\t", index=False)


def write_simulated_vcf(severus_df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("##fileformat=VCFv4.2\n")
        fh.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE\n")
        for row in severus_df.to_dict("records"):
            pos_1 = int(row["pos"]) + 1
            info = (
                f"SVTYPE={row['sv_type_str']};END={int(row['end'])};MAPQ={float(row['mapq'])};"
                f"CLUSTERID={row['cluster_id']};PHASESETID={row['phase_set']};"
                f"SUPP_READS={int(row['supp_total'])};REF_READS={int(row['ref_total'])};PRECISE;STRANDS=+-"
            )
            fh.write(
                f"{row['chrom']}\t{pos_1}\t{row['sv_id']}\tN\t<{row['sv_type_str']}>\t{row['qual']}\tPASS\t"
                f"{info}\tGT:VAF:DR:DV:hVAF\t0/1:{row['vaf_mean']}:{int(row['dr_mean'])}:{int(row['dv_mean'])}:0.4,0,0\n"
            )
