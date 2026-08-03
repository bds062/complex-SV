# Localization benchmark

Genome-level leave-one-out evaluation produced 163 interval predictions and 53
one-to-one correct-class matches among 108 labels at overlap coefficient >= 0.5.

| Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|
| 0.325 | 0.491 | 0.391 | 0.445 |

This directory contains the exact label tables, held-out predictions, aggregate
metrics, primary figures, and the complete training protocol. Release
checkpoints are stored in `../../models/localization_loo/`.
