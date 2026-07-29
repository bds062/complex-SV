# Final complex-SV results

This directory is a direct working snapshot of the final localization,
chromosome-level, ensemble, and false-positive analyses. It intentionally
includes intermediate tables, training code, model artifacts, writeups, and
figures so the analysis can be cleaned and reorganized later.

## Result packages

- `localization_model/`: best semi-localized event model. See `TRAINING.md` for
  the full training/evaluation procedure and `README.md` for metrics and
  packaged artifacts.
- `chromosome_model/`: best chromosome-level multilabel model. See
  `TRAINING.md` and `README.md`.
- `ensemble_and_fp_analysis/`: chromosome-level ensemble evaluation and direct
  comparison of localization-model and chromosome-model false positives.

## False-positive plots

- Localization model:
  `localization_model/figures/false_positives/`
- Chromosome model:
  `chromosome_model/figures/false_positives/`
- Agreement/AND ensemble:
  `ensemble_and_fp_analysis/shared_false_positives/`

The ensemble FP gallery contains the chromosome-class false positives called
by both component models. Each figure spans the complete chromosome and
highlights the localization-model interval.

## Ensemble summary

At the common held-out sample/chromosome/class unit:

| Method | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|
| Chromosome model | 0.320 | 0.545 | 0.403 | 0.478 |
| Localization aggregated to chromosome | 0.364 | 0.566 | 0.443 | 0.509 |
| OR ensemble | 0.276 | 0.677 | 0.392 | 0.524 |
| AND/agreement ensemble | 0.538 | 0.434 | 0.480 | 0.452 |
| LOO stacked ensemble | 0.467 | 0.424 | 0.444 | 0.432 |

The AND ensemble is the best F1-oriented rule, while OR is the
recall/F2-oriented rule. Full predictions, per-class metrics, methodology, and
the comparison figure are in
`ensemble_and_fp_analysis/chromosome_level_ensemble/`.
