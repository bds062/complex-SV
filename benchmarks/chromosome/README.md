# Chromosome-level benchmark

Genome-level leave-one-out evaluation of the multilabel chromosome classifier
produced the following aggregate result:

| Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|
| 0.320 | 0.545 | 0.403 | 0.478 |

This directory contains source labels, held-out predictions, aggregate metrics,
primary figures, and the complete training protocol. The five-fold release
ensemble is stored in `../../models/chromosome_fivefold/`.
