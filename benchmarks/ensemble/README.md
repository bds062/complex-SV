# Chromosome-level ensemble benchmark

Localization calls were collapsed to chromosome-class calls and combined with
the chromosome model on the same held-out genomes.

| Method | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|
| Chromosome model | 0.320 | 0.545 | 0.403 | 0.478 |
| Localization aggregated to chromosome | 0.364 | 0.566 | 0.443 | 0.509 |
| OR ensemble | 0.276 | 0.677 | 0.392 | 0.524 |
| AND ensemble | 0.538 | 0.434 | 0.480 | 0.452 |
| LOO stacked ensemble | 0.467 | 0.424 | 0.444 | 0.432 |

The retained `chromosome_level_ensemble/` directory contains the complete
prediction table, overall and per-class metrics, the comparison figure, and
methodological notes.
