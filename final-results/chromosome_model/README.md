# Final whole-chromosome complex-SV model result

This directory packages Pipeline27, the whole-chromosome multi-label complex-SV classifier.

## Headline leave-one-genome-out result

- 1,099 chromosomes encoded across 48 genomes
- 99 chromosome-class positives on 81 chromosomes across 37 labeled genomes
- 169 held-out positive calls: 54 TP, 115 observed-label FP, and 45 FN
- Precision: 0.320
- Recall: 0.545
- F1: 0.403
- F2: 0.478

Grouped five-fold evaluation reached precision 0.226, recall 0.566, F1 0.323, and F2 0.435. The difference measures small-data and split instability.

## Contents

- `TRAINING.md`: complete representation, supervision, fitting, calibration, and evaluation procedure.
- `labels/`: exact four source label tables plus chromosome-collapsed targets.
- `models/loo/`: 37 leave-one-genome-out evaluation checkpoints.
- `models/fivefold/`: five balanced grouped-fold evaluation checkpoints.
- `models/run_metadata/`: fold-specific train, validation, test, and threshold metadata.
- `metrics/`: aggregate and per-class LOO/five-fold metrics and fold assignments.
- `predictions/`: strict held-out chromosome-class predictions.
- `figures/loo_held_out_metrics.png`: aligned overall and per-class LOO performance.
- `figures/label_loss_pipeline.png`: aligned label-retention flow and per-class outcomes.
- `figures/`: false-positive summaries and top genome plots.
- `figures/false_positives/`: top 10 observed-label held-out false positives for each of the four classes.
- `false_positive_audit/`: all 115 observed-label false positives and ranked examples.
- `code/`: exact Pipeline27 implementation and Slurm workflow snapshots.
- `MANIFEST.sha256`: checksums for every packaged artifact except the manifest itself.

No proposal generator, sliding windows, boundary localizer, overlap test, NMS, or output cap is used. A positive means that a class occurs somewhere on a chromosome; this model does not recover event boundaries.

The packaged checkpoints are cross-validation artifacts, not a single deployment checkpoint trained on all labels. For deployment, select thresholds from inner out-of-fold predictions and then train one final head on all labeled genomes plus weak unlabeled-chromosome background.
