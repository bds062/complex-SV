# False-positive comparison: localization versus chromosome model

The models are compared at the common `(sample, chromosome, predicted class)` unit. Localization can emit multiple interval events per key, whereas the chromosome model emits at most one call per key.

## Main result

- Localization: 110 false-positive intervals representing 105 unique chromosome-class keys.
- Chromosome model: 115 false-positive chromosome-class keys.
- Exact overlap: 37 keys.
- Exact-key Jaccard similarity: 0.202.
- The shared keys cover 35.2% of localization keys and 32.2% of chromosome-model keys.
- Ignoring predicted class, 43 genomic sample/chromosome loci overlap, with locus-level Jaccard 0.331.
- Scores on the 37 shared keys have Spearman correlation 0.561.

The failure sets are therefore related, but not very similar. BFB has the greatest agreement; seismic amplification has no exact shared FP key.

Only seven localization FP interval calls occur on chromosomes carrying a same-class truth label. Those are localization-specific geometry, duplicate, or assignment errors. Most differences instead arise because the models use different prediction units and learn different decision boundaries.

## Top plotted examples

The two top-FP galleries share 10 exact sample/chromosome/class examples out of 64 unique gallery keys (Jaccard 0.156).

## Artifacts

- `false_positive_overlap.png`: per-class overlap and shared-key score comparison.
- `per_class_overlap.tsv`: per-class counts and similarity.
- `shared_false_positive_chromosome_classes.tsv`: exact shared keys.
- `localization_only_false_positive_keys.tsv` and `chromosome_only_false_positive_keys.tsv`.
- `shared_false_positive_scores.tsv`: scores for the 37 shared keys.
