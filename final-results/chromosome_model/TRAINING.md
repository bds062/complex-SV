# Training the whole-chromosome complex-SV model

## Task definition

Pipeline27 changes the prediction unit from a candidate interval to an entire `(genome, chromosome)` pair. It predicts four independent labels:

1. BFB
2. chromothripsis
3. ecDNA
4. seismic amplification

The output is multi-label, not softmax multiclass. Seventeen positive chromosomes carry more than one class. Multiple source events of the same class on one chromosome collapse to one target.

The selected 108 interval labels collapse to 99 chromosome-class positives on 81 chromosomes:

| Class | Chromosome-class positives | Positive samples | Source interval labels |
|---|---:|---:|---:|
| BFB | 37 | 24 | 42 |
| Chromothripsis | 32 | 23 | 36 |
| ecDNA | 13 | 9 | 13 |
| Seismic amplification | 17 | 9 | 17 |

The exact source and collapsed tables are in `labels/`.

## Inputs

1. Run Wakhan to obtain haplotype-specific copy-number segments.
2. Run Severus to obtain SV calls and breakpoint features.
3. Construct one full-chromosome region for every chromosome represented by Wakhan.
4. Encode each chromosome once with the same pretrained CN and SV encoders used upstream by Pipeline18/Pipeline24.

There are 1,099 chromosome rows across 48 genomes. Eighty-one chromosomes have at least one observed class label; the remaining 1,018 are weak unlabeled background.

There is no candidate proposal generator. The source label coordinates are never passed to the encoder and do not determine the chromosome boundaries.

## Chromosome representation

The frozen base encoder produces 402 features:

- 256-dimensional CN chromosome representation;
- 64-dimensional regional SV graph representation;
- 64-dimensional global SV graph representation;
- 18 CN/SV segment statistics.

Pipeline24 used a 1,214-dimensional local/context representation. For a whole chromosome, local and context are identical. Pipeline27 therefore constructs the exact equivalent efficiently:

```text
[base402, base402, zeros402, coordinates8] = 1,214 features
```

The eight coordinates encode relative start, end, center, width, log chromosome span, log context span, mixed-arm status, and chromosome-context status. Pipeline27 then adds:

- 37 whole-chromosome safe CN/SV summary features;
- three constant-zero slots replacing Pipeline24's proposal score, evidence score, and proposal-rank inputs.

The final input remains 1,254-dimensional while containing no proposal information. Features are standardized using training genomes only and clipped to `[-8, 8]`.

Chromosomes with more than 500 Severus nodes use the encoder's deterministic even subsample of 500 nodes. Interchromosomal mate indicators remain in the SV node features.

## Neural architecture

```text
1,254 inputs
    → Linear(1,254, 96)
    → LayerNorm
    → GELU
    → Dropout(0.35)
    → Linear(96, 48)
    → GELU
    → Dropout(0.35)
    → Linear(48, 4 independent logits)
    → Sigmoid probabilities
```

This is the Pipeline24 head topology applied once per chromosome. There is no separate objectness gate because a second threshold would create an unnecessary recall bottleneck.

## Positive-unlabeled supervision

Caller tables are incomplete, so absence of a label is not treated as a fully reliable negative.

For each chromosome-class loss element:

- observed positive: base weight 1.00;
- absent class on a chromosome labeled for another class: weight 0.25;
- class on a completely unlabeled chromosome: weight 0.10.

Positive weights are multiplied by the square root of effective negative mass divided by the number of training positives for that class, clipped to `[1, 8]`. The loss is element-wise weighted binary cross-entropy with logits, normalized by total weight.

The 0.10 weak-negative setting balances two errors: treating incomplete labels as ground-truth negatives versus allowing unrestricted whole-genome false-positive calls.

## Optimization

- Optimizer: AdamW
- Learning rate: 0.001
- Weight decay: 0.003
- Maximum epochs: 300
- Early-stopping patience: 35 validation epochs
- Gradient norm clipping: 5.0
- Hidden width: 96, then 48
- Dropout: 0.35
- Base seed: 27 with deterministic fold offsets

The selected checkpoint minimizes weighted validation BCE. No held-out test labels affect training or checkpoint selection.

## Validation and threshold calibration

Validation genomes are selected greedily from the non-test labeled genomes until approximately five chromosome-class positives per class are present. Label-free genomes remain available as weak training background.

After checkpoint selection, each class receives its own threshold. Thresholds are swept from 0.02 through 0.98 in 0.01 steps on validation chromosomes and selected by F2, then precision, then the lower threshold.

There is no geometry calibration, overlap coefficient, reciprocal overlap, NMS, one-to-one interval assignment, or prediction cap.

## Genome-level evaluation

### Leave one genome out

Thirty-seven labeled genomes are held out one at a time. All chromosomes from the test genome are scored. The other labeled genomes provide training and validation data; the 11 label-free genomes provide weak background when not otherwise excluded.

Aggregate chromosome-class result:

| Predictions | TP | FP | FN | Precision | Recall | F1 | F2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 169 | 54 | 115 | 45 | 0.320 | 0.545 | 0.403 | 0.478 |

Per-class LOO results:

| Class | Precision | Recall | F1 | F2 |
|---|---:|---:|---:|---:|
| BFB | 0.382 | 0.784 | 0.513 | 0.647 |
| Chromothripsis | 0.258 | 0.250 | 0.254 | 0.252 |
| ecDNA | 0.167 | 0.538 | 0.255 | 0.372 |
| Seismic amplification | 0.500 | 0.588 | 0.541 | 0.568 |

### Grouped five-fold

The 37 labeled genomes are assigned to balanced outer folds of 7, 7, 8, 8, and 7 genomes, with class balance optimized after enforcing fold size. This evaluation reached precision 0.226, recall 0.566, F1 0.323, and F2 0.435.

All splitting is by genome. Chromosomes from one genome never appear in both training and test sets.

## False-positive interpretation

LOO produced 115 observed-label false-positive chromosome-class calls:

| Class | False positives |
|---|---:|
| BFB | 47 |
| Chromothripsis | 23 |
| ecDNA | 35 |
| Seismic amplification | 10 |

Seventy-five occur on chromosomes with no label of any class. Forty are wrong-class calls on chromosomes labeled for another event. In particular, 23 of 35 ecDNA false positives are on chromosomes labeled for a different class, showing that class separation—not only event detection—is a major limitation.

These are observed-label false positives. Some chromosomes may contain genuine events missing from the caller tables, so measured precision is a lower bound under label incompleteness.

## Interpretation relative to Pipeline24

Pipeline27 removes proposal and localization failures and raises chromosome-class recall to 0.545. Its F1 is not directly comparable with Pipeline24's interval-level F1 because the prediction target is easier and different. Pipeline27 answers, “Which complex-SV classes occur somewhere on this chromosome?” Pipeline24 attempts to return event intervals.

BFB and seismic amplification are the strongest chromosome-level classes. Chromothripsis recall remains only 0.25, while ecDNA suffers severe class confusion and low precision. The five-fold decline shows that limited independent genomes and validation composition remain important sources of instability.

## Reproduction and source artifacts

The exact code snapshot is in `code/`. The original working results remain in `results/pipeline27/`.

Large reusable embeddings are not duplicated in this summer-results package. They are located at:

```text
results/pipeline27/chromosome_embeddings/
results/pipeline27/chromosome_tabular.tsv
```

Upstream inputs and frozen encoders are:

```text
results/pipeline18/complex_sv_manifest.tsv
results/pipeline3/cn_pretrain_chrom/cn_encoder.pt
results/pipeline3/sv3/graph_encoder.pt
```

The embedding Slurm job requires the project's `envs/env2` Python environment and its bundled C++ runtime, as recorded in `code/prepare_embeddings.sbatch`. The LOO and grouped five-fold arrays run independently after the shared chromosome embeddings are produced.

For a future train-all deployment model, calibrate class thresholds exclusively from pooled inner out-of-fold predictions, then fit one final head using all 37 labeled genomes plus weak label-free background. Do not deploy one of the packaged held-out evaluation checkpoints as if it were a train-all model.
