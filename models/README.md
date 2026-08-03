# Model release

The repository includes frozen representation encoders, final-fit deployment
models, and the cross-validation components used for the reported benchmarks.

| Directory | Contents |
|---|---|
| `pretrained_featurizer/` | CN masked autoencoder and Severus graph autoencoder |
| `localization_all48/` | Localization model fit using all 48 cohort genomes with held-out-CV-selected duration and decoder settings |
| `chromosome_all48/` | Multilabel chromosome model fit using all 48 cohort genomes |
| `localization_loo/` | Genome-held-out localization scorers and decoder calibrations used for evaluation |
| `chromosome_fivefold/` | Five chromosome-level classifiers used for evaluation |

Use `scripts/10_predict_localization.py` and
`scripts/11_predict_chromosomes.py` to apply the all-48 checkpoints directly,
or `scripts/12_predict_new_dataset.sh` to run both paths from a new manifest.
The all-48 models are final fits and do not provide an independent performance
estimate; use the held-out results in `benchmarks/` for that purpose. The
localization fit is reproducible with `scripts/train_localization_all.py`; it
requires the original feature files, labels, and completed held-out run
directories.

`SHA256SUMS` is an integrity manifest. It lets a user detect a truncated,
corrupted, or inadvertently modified model file; it is not read during model
inference and does not affect predictions. From the repository root, verify all
packaged artifacts with:

```bash
sha256sum --check models/SHA256SUMS
```
