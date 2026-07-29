# Training the selected localization model

## Selected method

The selected result is the original Pipeline24 proposal-bag neural scorer followed by the frozen event-cluster decoder. “Frozen” means that event clusters are constructed deterministically from the trained candidate scores; no second event-representation network is fitted.

The strict leave-one-genome-out result is 53 correct one-to-one matches among 108 caller labels from 163 predictions: precision 0.325, recall 0.491, F1 0.391, and F2 0.445. This is the preferred result until more independent labeled genomes are available.

## Training labels

The exact selected-model training table has 108 events across 37 labeled genomes:

| Class | Events |
|---|---:|
| BFB | 42 |
| Chromothripsis | 36 |
| ecDNA | 13 |
| Seismic amplification | 17 |

The four exact class-specific tables are in `labels/`. A later set of 42 chromothripsis calls was evaluated but was not used by the selected model; see “Model-selection decision” below.

## Inputs and candidate generation

1. Run Wakhan to obtain haplotype-specific copy-number segments.
2. Run Severus to obtain structural-variant calls and breakpoint graphs.
3. Apply the label-free standalone candidate generator.
4. The experiment contains 2,970 candidate intervals from 48 cell lines.
5. Candidates are grouped into 1,698 event clusters by sample, chromosome, and proposal-generator `cluster_id`.

The model never scans arbitrary chromosome windows. Localization begins with Wakhan/Severus candidate proposals.

## Candidate representation

Each candidate has 1,254 input features:

- 1,214-dimensional pretrained CN/SV embedding;
- 37 safe tabular CN/SV features;
- proposal score, candidate evidence score, and log-transformed priority rank.

Within each outer fold, features are standardized using training genomes only and clipped to the range [-8, 8].

## Neural scorer

The candidate scorer is:

```text
1,254 inputs
    → Linear(1,254, 96)
    → LayerNorm
    → GELU
    → Dropout(0.35)
    → Linear(96, 48)
    → GELU
    → Dropout(0.35)
    → Linear(48, 4 class logits)
```

Training uses AdamW with learning rate 0.001 and weight decay 0.003. Training runs for at most 300 epochs with early stopping after 35 validation epochs without improvement.

## Supervision

- Every caller event forms a bag containing all directly overlapping candidates.
- A candidate with overlap coefficient at least 0.5 receives direct positive supervision for that class.
- Candidates with nonzero but subthreshold overlap receive continuous overlap-quality supervision rather than being forced negative.
- Bag-max loss preserves caller events represented only by proposal fragments.
- Hard-negative mining uses the highest-scoring nonoverlapping candidates, with a 6:1 negative-to-positive target and at least 64 negatives.
- The total loss combines point-positive, hard-negative, event-bag, and overlap-quality terms.

## Genome-level cross-validation

Evaluation is leave-one-labeled-genome-out across 37 genomes. For each held-out genome:

1. Reserve the test genome before fitting or calibration.
2. Select validation genomes until the validation set contains at least five labels from every class.
3. Train on the remaining candidate genomes, including label-free genomes as background.
4. Select the early-stopping checkpoint by validation loss.
5. Score candidates in the held-out genome.

The 37 packaged checkpoints in `models/loo/` are cross-validation artifacts. They are not a single train-all deployment checkpoint.

## Frozen event decoder

For each class:

1. Group candidates by sample, chromosome, and proposal cluster.
2. Use the maximum candidate class score as the event score.
3. Use the maximum-scoring candidate as representative geometry; validation may instead select the cluster envelope.
4. Calibrate on validation labels using one-to-one F1.
5. Select from score thresholds, centered scales {0.67, 1.0, 1.5}, representative/envelope geometry, no NMS or containment NMS, and per-sample caps {1, 2, 4, 8, 12}.
6. Apply the selected decoder once to the held-out genome.

Evaluation uses maximum-cardinality one-to-one matching at overlap coefficient >= 0.5. Multiple predictions cannot claim the same caller label.

## Where labels are lost

The selected model retains labels through these stages:

```text
108 caller labels
  → 106 candidate-covered          (2 proposal misses)
  → 94 event-geometry compatible   (12 geometry losses)
  → 58 above class threshold       (36 score/ranking losses)
  → 58 after NMS                   (0 NMS losses)
  → 56 after output caps           (2 cap losses)
  → 53 one-to-one matches          (3 assignment collisions)
```

The largest apparent bottleneck is thresholding, but an optimistic test-label threshold oracle reaches only F1 0.417. Many false events rank above missed labels, so better score separation—not simply a lower threshold—is required.

## Model-selection decision

Six additional chromothripsis labels were tested later, giving 42 chromothripsis and 114 total labels. Exact retraining reduced paired 37-genome F1 from 0.400 to 0.331 and full 39-genome F1 to 0.321. Five additions were LOW-confidence calls. All six had proposal overlap coefficient at least 0.973, but both old and retrained models recovered only the same two additions.

Early-stopping epochs and validation loss were essentially unchanged, and validation membership changed in only one fold. The decline is attributed to small-sample instability and negative transfer through the shared representation. The updated retraining was therefore not selected.

## Reproduction

The packaged source files in `code/` are snapshots of the exact implementation:

- `candidate_bag_model.py`: representation, supervision, fitting, matching, and metrics;
- `frozen_event_decoder.py`: event clustering, decoding, and F1 calibration;
- `config.json`: selected training configuration;
- `analyze_label_losses.py`: stage-by-stage loss audit;
- `make_summary_plots.py`: package summary figures.

The original feature bundles remain under `results/pipeline18/candidate_region_classifier_general/`, and the candidate table is `results/pipeline18/merged_candidate_regions.csv`.

For a future train-all deployment model, choose calibration parameters exclusively from pooled inner out-of-fold predictions, then train the neural scorer on all labeled genomes. Do not select parameters using the external test set.
