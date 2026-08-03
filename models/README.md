# Packaged models

`pretrained/` contains the frozen representation encoders used by the final
experiments:

- `cn_encoder.pt`: masked copy-number autoencoder trained on Wakhan haplotype
  segment windows;
- `sv_graph_encoder.pt`: masked graph autoencoder trained on regional Severus
  breakpoint graphs.

The supervised checkpoints remain beside their provenance and metrics:

- `final-results/localization_model/models/loo/`: 37 localization LOO models;
- `final-results/chromosome_model/models/fivefold/`: five chromosome models;
- `final-results/chromosome_model/models/loo/`: 37 chromosome LOO models.

LOO and five-fold files are evaluation/cross-fit artifacts, not a single
train-all model. The numbered inference scripts use them as vote ensembles and
report the vote fraction. This is useful for research application to new
genomes, but the ensemble threshold has not been validated on an independent
external cohort.
