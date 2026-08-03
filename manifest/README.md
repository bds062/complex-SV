# Cohort manifest

`complex_sv_manifest.tsv` is the 48-genome manifest used for the reported
experiments consisting of 5 CASTLE lines + 43 new lines. Its columns are `sample_id`, `wakhan_sample_id`, `wakhan_root`,
and `severus_vcf`.
## Append newly processed samples

New runs are expected beneath a common directory with this structure (as per the 43 new cell lines):

```text
SAMPLES_ROOT/
  SAMPLE_ID/
    sv_cna_v2/
      wakhan/
        *_*_SCORE/
          bed_output/
          vcf_output/
      severus/somatic_SVs/severus_somatic.vcf
```

Update a manifest in place:

```bash
python manifest/update_manifest.py manifest/complex_sv_manifest.tsv \
  --samples-root /data/KolmogorovLab/HPV_Dean_Collab/nano_nf_runs
```

The updater scans immediate sample directories and appends completed samples
that are not already represented. When Wakhan has multiple numeric solution
directories, the solution with the largest score is always selected.

To avoid changing the publication manifest directly, use `--output`:

```bash
python manifest/update_manifest.py manifest/complex_sv_manifest.tsv \
  --samples-root /path/to/nano_nf_runs \
  --output work/expanded_manifest.tsv
```

Restrict discovery to named samples with repeatable `--sample`. If a manifest
ID differs from its directory name, use `MANIFEST_ID=DIRECTORY_NAME`:

```bash
python manifest/update_manifest.py work/expanded_manifest.tsv \
  --samples-root /path/to/nano_nf_runs \
  --sample NEW_SAMPLE \
  --sample DISPLAY_ID=run_directory
```

During automatic scanning, incomplete directories are counted and skipped. A sample is appended only when
the selected solution contains paired HP1/HP2 Wakhan BED files, the matching
integer-CNA VCF, and the Severus somatic VCF. Writes are atomic, existing row
order is preserved, and new rows are sorted by sample ID. Use `--dry-run` to
inspect proposed additions without writing.
