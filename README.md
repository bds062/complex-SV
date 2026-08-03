# complex-SV

A research pipeline for detecting complex structural-variant events from
[Wakhan](https://github.com/KolmogorovLab/Wakhan) copy-number segments and
[Severus](https://github.com/KolmogorovLab/Severus) structural-variant calls.

The repository supports two related tasks:

- **localized event detection**: propose genomic intervals, classify them as
  BFB, chromothripsis, ecDNA, or seismic amplification, and return calibrated
  event boundaries;
- **chromosome-level detection**: predict which of the four event classes occur
  anywhere on each chromosome. This model is multilabel and does not localize
  an event.

The numbered commands in [`workflow/`](workflow/) are the supported model entry
points. The upstream [label generator](label_generator/README.md),
[candidate generator](candidate_generator/README.md), and replaceable
[integration-test configuration](test/README.md) are documented separately.
Held-out predictions, aggregate metrics, labels, and primary figures are
provided in [`benchmarks/`](benchmarks/).

> **Status:** This is research software, not a clinical diagnostic. The
> packaged supervised models were evaluated by genome-level cross-validation
> on a small cancer cell-line cohort. Calls on new cohorts require independent
> validation.

## Method overview

```text
Wakhan segments ──> masked CN encoder ─┐
                                      ├─> candidate/chromosome representation
Severus VCF ──────> masked graph MAE ──┘
                                      │
           localized path             ├─> proposals ─> MIL scorer ─> frozen decoder
           chromosome path            └─> chromosome MLP ─> multilabel calls
```

The frozen encoders generate a 402-dimensional base representation: 256 CN
features, 64 regional SV features, 64 global SV features, and 18 segment
statistics. The final heads receive 1,254 inputs after adding local/context
features, coordinates, 37 safe tabular summaries, and three proposal-prior
slots. The chromosome model sets the proposal slots to zero.

The selected localization evaluation reached precision 0.325, recall 0.491,
F1 0.391, and F2 0.445 at overlap coefficient >= 0.5. The chromosome model
reached precision 0.320, recall 0.545, F1 0.403, and F2 0.478. See the
model-specific training notes for the full protocol:

- [localization model](benchmarks/localization/TRAINING.md)
- [chromosome model](benchmarks/chromosome/TRAINING.md)
- [chromosome-level ensemble](benchmarks/ensemble/README.md)

## 1. Installation

The recommended installation uses the supplied Conda environment, which is a
superset of the KolmogorovLab/Severus environment:

```bash
git clone git@github.com:srinivasabd/complex-SV.git
cd complex-SV

conda env create --file environment.yml
conda activate complex-SV
python workflow/00_check_install.py
```

An existing Severus environment can be extended, but a separate environment is
recommended so the validated SV-calling installation is not changed. See the
[environment guide](ENVIRONMENT.md) for Severus reuse, GPU/CUDA verification,
and installation alternatives.

The project is currently designed for GRCh38. Candidate generation uses the
bundled [centromere coordinates](data/grch38.cen_coord.curated.bed).

## 2. Input files

Every downstream command uses a tab-separated manifest with one row per genome:

```tsv
sample_id	wakhan_root	severus_vcf
sample_A	/absolute/path/sample_A	/absolute/path/sample_A.severus.vcf
sample_B	/absolute/path/sample_B	/absolute/path/sample_B.severus.vcf
```

`wakhan_root` may be a Wakhan BED root, a
`*_copynumbers_segments` prefix, or either member of the paired
`*_copynumbers_segments_HP_1.bed` and `HP_2.bed` files. `severus_vcf`
must point to the Severus VCF for the same genome. Sample identifiers must be
unique and consistent with the training-label tables.

The original 48-genome publication manifest and a utility for appending new
samples are provided in [`manifest/`](manifest/README.md). The utility selects
the Wakhan solution with the largest third score component and validates all
required files before updating a manifest.

Use absolute paths in manifests and pretraining lists. It makes Slurm jobs
independent of their launch directory.

## 3. Pretrain the autoencoders

You can skip this section and use the checkpoints shipped in
[`models/pretrained/`](models/pretrained/).

### 3.1 Copy-number masked autoencoder

Create a text file containing one Wakhan root per line:

```text
/absolute/path/sample_A
/absolute/path/sample_B
```

Then run:

```bash
workflow/01_pretrain_cn_encoder.sh \
  inputs/wakhan_roots.txt \
  outputs/pretraining/cn \
  --epochs 100 \
  --batch_size 64
```

The trainer samples fixed-base-pair windows, resamples paired haplotype copy
number into fixed-length tensors, masks bins, and minimizes reconstruction MSE
only over masked bins. Its primary artifact is
`outputs/pretraining/cn/cn_encoder.pt`; metadata, logs, a loss plot, and
optional window embeddings are written beside it.

### 3.2 Severus graph masked autoencoder

Create a text file containing one Severus VCF per line:

```text
/absolute/path/sample_A.severus.vcf
/absolute/path/sample_B.severus.vcf
```

Then run:

```bash
workflow/02_pretrain_sv_encoder.sh \
  inputs/severus_vcfs.txt \
  outputs/pretraining/sv \
  --epochs 100 \
  --batch_size 32
```

The graph trainer builds regional breakpoint graphs, masks SV nodes, and
optimizes node reconstruction plus graph-level density, foldback, and
interchromosomal objectives. Its primary artifact is
`outputs/pretraining/sv/graph_encoder.pt`.

Run either numbered command with `--help` after its two positional arguments
to inspect advanced window, masking, architecture, and sampling settings. A
newly pretrained checkpoint can replace the default path in the embedding
commands by passing a later duplicate `--cn_checkpoint` or
`--graph_checkpoint` argument.

## 4. Use the shipped localization models

This path returns semi-localized event intervals.

### 4.1 Generate label-free proposals

```bash
candidate_generator/run.sh \
  inputs/manifest.tsv \
  outputs/localized/candidates \
  --profile balanced \
  --keep_going
```

The generator combines short-segment runs, focal high-copy runs, Severus
foldbacks, multiscale SV clusters, joint CNA/SV windows, and complex chromosome
arms. It does not read class labels or external complex-SV callers. The selected
table is `merged_candidate_regions.csv`; the sensitive and balanced tables are
also retained.

Use `--profile sensitive` when proposal recall is more important than runtime
and false-positive burden.

### 4.2 Embed candidates

```bash
workflow/04_embed_candidates.sh \
  inputs/manifest.tsv \
  outputs/localized/candidates/merged_candidate_regions.csv \
  outputs/localized/features \
  --device cuda
```

This writes candidate metadata, the 402-dimensional encoder bundle, the
1,214-dimensional selected local/context representation, and the 37 safe
tabular features.

### 4.3 Apply the packaged ensemble

```bash
python workflow/06_predict_localization_ensemble.py \
  --candidates outputs/localized/candidates/merged_candidate_regions.csv \
  --embedding-bundle outputs/localized/features/embeddings.npz \
  --selected-embeddings outputs/localized/features/selected_embedding_features.npz \
  --tabular-features outputs/localized/features/tabular_features.npz \
  --output outputs/localized/predictions \
  --device cuda
```

`localization_predictions.tsv` contains consensus interval calls.
`per_checkpoint_calls.tsv` preserves every component-model call for auditing.
The default requires at least half of the 37 LOO models to call an event cluster.
Change `--minimum-vote-fraction` to explore the precision/recall tradeoff.

The original per-fold thresholds were selected without each fold's held-out
genome. However, using all LOO models as a vote ensemble on a new cohort is a
research extrapolation; the 0.5 ensemble-vote threshold itself has not been
validated on an independent cohort.

## 5. Train the localization model

Training labels are tab-separated and use inclusive interval ends:

```tsv
region_id	sample_id	chrom	start	end	label
event_001	sample_A	chr8	127000000	133000000	ecDNA
```

Allowed labels are `ecDNA`, `chromothripsis`, `BFB`, and
`seismic_amplification`.

First run candidate generation and feature extraction for the full training
cohort. Then reproduce the selected event-bag MIL and frozen-decoder
cross-validation:

```bash
workflow/05_train_localization_model.sh \
  outputs/train/candidates/merged_candidate_regions.csv \
  inputs/training_labels.tsv \
  outputs/train/features/embeddings.npz \
  outputs/train/features/selected_embedding_features.npz \
  outputs/train/features/tabular_features.npz \
  AU565 \
  outputs/train/localization_cv/AU565
```

Run step 05 once for every labeled genome, using that genome as
`TEST_SAMPLE`. These independent jobs can be launched as a Slurm array. Each
run trains its own scorer, selects validation genomes by event counts, and
calibrates its frozen event-cluster decoder without the test genome.

For every caller event, all overlapping proposals form a bag. Candidates with
overlap coefficient >= 0.5 receive direct class supervision; partial overlaps
receive continuous overlap-quality supervision. Hard-negative mining reduces
damage from incomplete labels. Validation genomes select early stopping,
class-specific thresholds, representative-versus-envelope geometry, centered
boundary scale, containment NMS mode, and per-genome output caps. Test genomes never participate in fitting or calibration.

Each run writes held-out predictions and matches, training history, decoder
calibration and sweep tables, split membership, metrics, and one checkpoint.
Aggregate only after all labeled test genomes finish. Split strictly by genome.

## 6. Use the shipped chromosome models

This path predicts classes per chromosome and intentionally skips proposal
generation, localization, geometry adjustment, and NMS.

### 6.1 Prepare one representation per chromosome

```bash
workflow/07_embed_chromosomes.sh \
  inputs/manifest.tsv \
  outputs/chromosome
```

### 6.2 Apply the packaged five-fold ensemble

```bash
python workflow/09_predict_chromosome_ensemble.py \
  --embedding-dir outputs/chromosome/embeddings \
  --tabular outputs/chromosome/chromosome_tabular.tsv \
  --output outputs/chromosome/predictions \
  --device cuda
```

The output is multilabel: one chromosome may receive multiple classes.
`chromosome_predictions.tsv` contains all chromosome-class probabilities and
vote fractions; `predicted_complex_sv.tsv` contains calls passing the default
majority-vote rule.

## 7. Train the chromosome model

After step 07, provide a label table with columns `sample_id`, `chrom`, and
`label`. Run one independent job per labeled test genome:

```bash
workflow/08_train_chromosome_model.sh \
  outputs/chromosome/embeddings \
  outputs/chromosome/chromosome_tabular.tsv \
  inputs/chromosome_labels.tsv \
  AU565 \
  outputs/chromosome_cv/AU565
```

The model uses weighted multilabel BCE. Observed positives receive weight 1.0,
missing classes on another labeled chromosome receive weight 0.25, and fully
unlabeled chromosomes receive weight 0.10. Per-class F2 thresholds are selected
on validation genomes. Split by genome: chromosomes from a test genome must
never occur in training or calibration. The independent LOO jobs can be
launched as a Slurm array.

## 8. Which model should I use?

Use the localization model when the required output is an event interval or
when several same-class events may occur on one chromosome. Its recall is
bounded by proposal generation and its precision is affected by thresholding
and event geometry.

Use the chromosome model when the question is only whether a chromosome
contains a class. It avoids proposal and localization losses and can detect
multilabel chromosomes, but it cannot identify event boundaries.

For precision-oriented chromosome screening, require agreement between the
chromosome model and localization calls aggregated to chromosome. In the held-
out analysis this AND rule achieved F1 0.480. For recall-oriented screening,
use the OR rule, which achieved F2 0.524. Those figures are cross-validation
results, not external-cohort estimates.

## 9. Repository layout

```text
benchmarks/     labels, held-out predictions, metrics, and primary figures
candidate_generator/ label-free Wakhan/Severus proposal pipeline
configs/        versioned method configuration
data/           Wakhan/Severus parsing and feature construction
discovery/      candidate and chromosome embedding extraction
label_generator/ curated caller TSV normalization and provenance
manifest/       publication cohort manifest and sample-discovery updater
model/          localization, event-decoder, and chromosome architectures
models/         pretrained encoders and supervised release ensembles
pretrain/       masked-autoencoder implementations and trainers
test/           configurable single-genome integration tests
training/       candidate feature preparation and shared loss functions
workflow/       supported numbered training and inference commands
```

Large generated embeddings and run outputs are intentionally ignored by Git.
Only publication inputs, aggregate results, and reusable checkpoints belong in
the repository.
