# Chromosome-scale ASM-Loc pipeline

This pipeline separates two jobs that pipeline18 currently conflates:

1. **Localization / proposal generation** scans every chromosome using inexpensive
   1 Mb CNA/SV feature bins.
2. **Event typing** embeds only the localized proposals and applies the trained
   pipeline18 `ecDNA/chromothripsis/BFB` classifier.

The design adapts the useful parts of ASM-Loc to a genome: fixed genomic bins are
the snippets, a local convolution captures neighboring-bin structure, Transformer
attention captures longer chromosome context, short events receive higher loss
weight, and a boundary-distance head refines activation runs into interval
proposals. Pipeline18 cross-fold predictions may be included as conservative
pseudo-regions, but direct external regions take precedence.

## Run

From `complex-SV`:

```bash
bash scripts/run_pipeline19_asm_loc.sh
```

By default this reads `../results/pipeline18` and writes
`../results/pipeline19_asm_loc`. Important overrides are:

```bash
OUTPUT_DIR=/path/to/output \
DEVICE=cuda \
BIN_SIZE=1000000 \
bash scripts/run_pipeline19_asm_loc.sh
```

Set `TEACHER_PREDICTIONS=/nonexistent` to disable pipeline18 pseudo-region
refinement. The model accepts teacher calls only when objectness is at least
0.95 and a type probability is at least 0.90; it discards teacher intervals
overlapping a direct caller region.

## Main outputs

- `chromosome_bins.tsv.gz`: model-ready ordered chromosome bins.
- `asm_loc_model.pt`: localizer checkpoint and feature normalization.
- `bin_predictions.tsv.gz`: per-bin foreground, class, and boundary scores.
- `localized_proposals.tsv`: cheap boundary-refined proposals.
- `localization_recall.summary.json`: coverage and IoU localization metrics.
- `merged_candidate_regions.csv`: pipeline18-compatible candidate table.
- `final_predictions/predicted_complex_sv.tsv`: final typed calls.

## Validation and tuning

The default training split holds out 20% of genomes, chosen deterministically,
and records them in `asm_loc_model.split.json`. Tune localization thresholds
against held-out genomes, not the full training set. The most useful parameters
on `predict` are:

- `--foreground_threshold` (default 0.45)
- `--class_threshold` (default 0.40)
- `--merge_gap_bins` (default 1)
- `--nms_iou` (default 0.65)

Use 500 kb bins if small focal events are frequently missed; use 2 Mb bins if
runtime and memory matter more than sub-megabase boundaries. `BIN_SIZE` must be
the same for dataset construction and prediction.

## Avoiding optimistic evaluation

The optional teacher input should be the pipeline18 **cross-fold** predictions,
not final in-sample predictions. Localization recall is reported against all
external regions for quick diagnostics, but the held-out genomes in the split
file are the appropriate set for model selection. A future production pass can
retrain after thresholds and hyperparameters are frozen.
