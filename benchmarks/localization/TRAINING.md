# Localization model: training and evaluation

## Prediction task

The model assigns one or more complex-SV classes to label-free candidate
intervals and returns a calibrated event interval. The supported classes are
BFB, chromothripsis, ecDNA, and seismic amplification.

## Inputs

Wakhan supplies haplotype-specific copy-number segments and Severus supplies
structural-variant breakpoints. A class-agnostic proposal generator combines
short copy-number segments, focal amplification, foldback intervals, multiscale
SV clusters, joint CNA/SV density windows, and complex chromosome arms.

The evaluated cohort contains 2,970 proposals from 48 genomes. The training
labels contain 108 events across 37 labeled genomes: 42 BFB, 36
chromothripsis, 13 ecDNA, and 17 seismic-amplification events.

## Representation and architecture

Each proposal is represented by 1,254 values:

- a 1,214-dimensional frozen CN/SV local-context embedding;
- 37 regional CNA/SV summary features;
- proposal score, evidence score, and log priority rank.

Training-fold statistics standardize each input dimension, after which values
are clipped to [-8, 8]. The classifier is:

```text
1,254 -> Linear(96) -> LayerNorm -> GELU -> Dropout(0.35)
      -> Linear(48) -> GELU -> Dropout(0.35) -> four logits
```

Optimization uses AdamW (learning rate 0.001, weight decay 0.003), gradient
norm clipping at 5, a maximum of 300 epochs, and early stopping with patience
35.

## Multiple-instance supervision

All proposals overlapping one caller event form an event bag. A proposal with
overlap coefficient at least 0.5 receives direct positive supervision for that
class. Proposals with nonzero subthreshold overlap receive continuous
overlap-quality supervision. Bag-max loss preserves events represented only by
proposal fragments. Hard-negative mining retains the highest-scoring
nonoverlapping proposals with a 6:1 negative-to-positive target and a minimum
of 64 negatives.

## Frozen event decoder

Candidate scores are max-pooled by genome, chromosome, and proposal cluster.
Validation genomes select a class-specific score threshold, representative or
cluster-envelope geometry, a centered scale in {0.67, 1.0, 1.5}, containment
NMS, and a per-genome output cap. This stage is deterministic calibration; it
does not fit a second localization network.

## Evaluation protocol

Evaluation holds out one complete labeled genome at a time. Validation genomes
are selected from the remaining cohort until approximately five events per
class are represented. The test genome is excluded from feature
standardization, optimization, early stopping, and decoder calibration.

Predictions and caller events are matched one-to-one at overlap coefficient
>= 0.5. A prediction cannot claim more than one event, and an event cannot be
claimed by more than one prediction.

## Result

| Predictions | TP | Labels | Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|---:|---:|---:|
| 163 | 53 | 108 | 0.325 | 0.491 | 0.391 | 0.445 |

The main loss occurs during score thresholding and class ranking, rather than
proposal generation or NMS. These estimates are based on a small cell-line
cohort and require confirmation on an independent dataset.
