# Standalone candidate-region discovery

`generate_standalone_candidate_regions.py` proposes complex-SV regions using only
Wakhan CNA and Severus SV inputs from a manifest. It never reads BFBArchitect,
ShatterSeek, or CoRaL output.

## Discovery signals

The generator combines complementary, class-agnostic proposal families before
recomputing the standard regional CNA/SV features:

- runs of short CNA segments
- focal high-copy runs
- individual and clustered Severus foldbacks
- adaptive SV clusters at 2 Mb and 10 Mb scales
- joint CNA/SV density windows at 5 Mb, 20 Mb, and 50 Mb scales
- unusually complex chromosome arms

Every candidate receives an evidence-only score, within-genome rank, and tier.
The `sensitive` profile retains every proposal. The `balanced` profile keeps
regions with stronger CNA, SV, foldback, or high-copy support. Both complete
tables are written regardless of the selected profile.

## Run discovery only

```bash
python complex-SV/scripts/generate_standalone_candidate_regions.py \
  results/pipeline12/complex_sv_manifest.tsv \
  --output_dir results/my_standalone_candidates \
  --profile sensitive \
  --keep_going
```

The selected table is `merged_candidate_regions.csv`. The unfiltered and
balanced alternatives are `merged_candidate_regions_sensitive.csv` and
`merged_candidate_regions_balanced.csv`.

## Run discovery and classification

```bash
MANIFEST=results/my_genome_manifest.tsv \
OUTPUT_DIR=results/my_standalone_run \
CHECKPOINT=results/my_model/candidate_region_classifier.pt \
PROFILE=balanced \
bash complex-SV/scripts/run_standalone_candidate_classifier.sh
```

This runs candidate discovery, prepares full local/context/difference/coordinate
embeddings and safe tabular features, then applies the checkpoint's calibrated
objectness, type, rescue, secondary-call, subtype, and cluster-aggregation rules.
Predictions are written under `OUTPUT_DIR/predictions`.

The checkpoint must have been trained with the same embedding normalization,
embedding feature mode, and tabular features. The apply command checks those
contracts. For final deployment, retrain on candidates emitted by this generator;
older checkpoints can be used for smoke tests but saw a different candidate and
negative distribution during training.

## Offline evaluation only

External caller intervals can be used after discovery to measure proposal recall.
They are not part of deployment:

```bash
python complex-SV/scripts/evaluate_candidate_region_recall.py \
  --candidates results/my_standalone_candidates/merged_candidate_regions.csv \
  --external_regions results/pipeline15/external_regions.tsv \
  --output_dir results/my_standalone_candidates/evaluation
```

The BFB input-concordance audit is similarly offline:

```bash
python complex-SV/scripts/audit_bfbarchitect_severus_concordance.py \
  --manifest results/pipeline12/complex_sv_manifest.tsv \
  --bfb_calls results/bfbarchitect3/bfb_calls.tsv \
  --output_dir results/my_standalone_candidates/bfb_input_audit
```
