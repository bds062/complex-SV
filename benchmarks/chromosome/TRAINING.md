# Chromosome model: training and evaluation

## Prediction task

The chromosome model predicts four independent labels for every
genome-chromosome pair: BFB, chromothripsis, ecDNA, and seismic amplification.
The output is multilabel, so one chromosome may carry several event classes.
The model does not estimate event boundaries.

The 108 interval labels collapse to 99 chromosome-class positives on 81
chromosomes. The evaluated dataset contains 1,099 chromosomes from 48 genomes;
1,018 chromosomes have no observed class label and are treated as weak
unlabeled background.

## Representation and architecture

The frozen encoders produce a 402-dimensional chromosome representation: 256
CN features, 64 regional SV features, 64 global SV features, and 18 segment
statistics. The chromosome input preserves the localization model's contract:

```text
[base402, base402, zeros402, coordinates8] + tabular37 + zeros3 = 1,254
```

The classifier is:

```text
1,254 -> Linear(96) -> LayerNorm -> GELU -> Dropout(0.35)
      -> Linear(48) -> GELU -> Dropout(0.35) -> four independent logits
```

Optimization uses AdamW (learning rate 0.001, weight decay 0.003), gradient
norm clipping at 5, a maximum of 300 epochs, and early stopping with patience
35.

## Positive-unlabeled loss

Caller labels are incomplete. Observed positive chromosome-class pairs receive
weight 1.0. A missing class on a chromosome labeled for another class receives
weight 0.25. A class on a completely unlabeled chromosome receives weight
0.10. Positive weights are additionally scaled by the square root of effective
negative mass divided by positive count, clipped to [1, 8].

## Validation and evaluation

Complete genomes define train, validation, and test splits. Validation genomes
are selected until approximately five positive chromosome-class pairs per
class are represented. Each class threshold is selected on validation data by
F2, with precision and then lower threshold used as tie breakers.

Leave-one-genome-out evaluation scores every chromosome in the held-out genome.
No held-out chromosome participates in standardization, fitting, early
stopping, or threshold calibration.

## Result

| Predictions | TP | FP | FN | Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 169 | 54 | 115 | 45 | 0.320 | 0.545 | 0.403 | 0.478 |

Per-class F1 is 0.513 for BFB, 0.254 for chromothripsis, 0.255 for ecDNA, and
0.541 for seismic amplification. Some observed false positives may represent
real but unlabeled events, so measured precision is conditional on label
completeness.
