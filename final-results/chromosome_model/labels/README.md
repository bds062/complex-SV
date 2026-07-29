# Training labels

The four class-specific TSV files and `all_labels.tsv` are the exact 108 interval labels selected for Pipeline24 and Pipeline27.

Pipeline27 generalizes these intervals to chromosome-level multi-label targets. `chromosome_labels.tsv` contains the 99 deduplicated `(sample, chromosome, class)` positives. `embedding_labels.tsv` is an archival full-chromosome label-schema table; labels were not supplied during chromosome embedding, preventing coordinate leakage. `label_summary.tsv` records the collapse counts.
