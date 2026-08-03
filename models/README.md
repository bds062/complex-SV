# Model release

The repository includes the frozen representation encoders and supervised
cross-validation ensembles used in the reported experiments.

| Directory | Contents |
|---|---|
| `pretrained/` | CN masked autoencoder and Severus graph autoencoder |
| `localization_loo/` | 37 genome-held-out localization scorers and decoder calibrations |
| `chromosome_fivefold/` | Five chromosome-level multilabel classifiers |

Verify encoder integrity from this directory with:

```bash
cd models/pretrained
sha256sum -c SHA256SUMS
```

The supervised files are cross-validation components rather than train-all
deployment checkpoints. The inference commands report ensemble vote fractions;
their ensemble decision thresholds require validation on an independent cohort.
