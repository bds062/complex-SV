# Chromosome-level ensemble experiment

Both final models were reduced to the same sample/chromosome/class unit. The localization model contributes a positive chromosome call when at least one localized event of that class survives its own held-out decoder.

The OR and AND rules use the models' original held-out calls. The stacked ensemble is leakage-safe at the meta-model level: for every held-out genome, a regularized logistic combiner and its F1 threshold are fitted using only the other genomes' out-of-fold predictions.

| Method | Precision | Recall | F1 | F2 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Chromosome model | 0.320 | 0.545 | 0.403 | 0.478 | 54 | 115 | 45 |
| Localization→chromosome | 0.364 | 0.566 | 0.443 | 0.509 | 56 | 98 | 43 |
| OR ensemble | 0.276 | 0.677 | 0.392 | 0.524 | 67 | 176 | 32 |
| AND ensemble | 0.537 | 0.434 | 0.480 | 0.452 | 43 | 37 | 56 |
| LOO stacked ensemble | 0.467 | 0.424 | 0.444 | 0.432 | 42 | 48 | 57 |

Best observed method: **AND ensemble** (F1=0.480; change versus chromosome model=+0.077).

Accuracy is retained in the TSV but is not emphasized because the large number of negative chromosome/class combinations makes it misleadingly high.
