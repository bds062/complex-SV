# Label generation

Generates method-based calls into supervised label resolutions used by complex-SV. Independent of model predictions and candidate proposals.

## Inputs

Provide a directory containing these tab-separated files:

| Broad class | Required files |
|---|---|
| BFB | `bfbarchitect_calls.tsv`, `bfbarchitect_noncanonical_calls.tsv` |
| chromothripsis | `chromothripsis_calls.tsv`, `chromothripsis_noncanonical_calls.tsv` |
| ecDNA | `coral_ecDNA_calls.tsv` |
| seismic amplification | `seismic_amplification_calls.tsv`, `seismic_amplification_noncanonical_calls.tsv` |

BFB, chromothripsis, and seismic-amplification tables require `sample_id`,
`chrom`, `start`, and `end`. The CoRAL table requires `sample_id` and either
those coordinate columns or a `region` field containing one or more
semicolon-separated `chr:start-end` intervals.

The canonical and noncanonical filenames describe annotation provenance only.
Both contribute the same broad class target. Any internal canonicality columns
are ignored. Coordinates must use GRCh38.

These source TSVs are curated caller outputs:

- BFB/BFB-like events: BFBArchitect;
- chromothripsis and seismic amplification: ShatterSeek-derived criteria;
- ecDNA: CoRAL circular-amplicon calls.

Caller installation and execution are upstream prerequisites. Because these
tools have independent environments and versioning, this repository begins at
their reviewed TSV outputs and records the source filename and row for every
derived label.

## Run

```bash
label_generator/run.sh \
  /path/to/curated_label_tsvs \
  outputs/labels
```

## Outputs

- `interval_labels.tsv`: localized labels used by the candidate/MIL model;
- `chromosome_labels.tsv`: class targets expanded to the full GRCh38 chromosome;
- `embedding_labels.tsv`: chromosome labels in the embedding input schema;
- `label_summary.tsv`: counts by class and resolution;
- `label_manifest.json`: source files, coordinate convention, and provenance policy.

The generator validates required columns, chromosomes, interval geometry,
GRCh38 chromosome bounds, and duplicate class intervals. It fails rather than
silently discarding malformed records.

The exact publication label tables are retained under `benchmarks/*/labels/`.
Those tables should be used to reproduce the reported cross-validation; this
generator is for rebuilding the inputs from the curated caller exports.
