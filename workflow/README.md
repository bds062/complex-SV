# Numbered workflow

Run commands from the repository root. Set `PYTHON_BIN` when the desired
interpreter is not named `python`.

| Step | Command | Purpose |
|---:|---|---|
| 00 | `00_check_install.py` | Validate dependencies and packaged models |
| 01 | `01_pretrain_cn_encoder.sh` | Pretrain the Wakhan CN autoencoder |
| 02 | `02_pretrain_sv_encoder.sh` | Pretrain the Severus graph autoencoder |
| 03 | `03_generate_candidates.sh` | Generate label-free candidate intervals |
| 04 | `04_embed_candidates.sh` | Build localization-model features |
| 05 | `05_train_localization_model.sh` | Train one localization LOO split |
| 06 | `06_predict_localization_ensemble.py` | Apply shipped localization models |
| 07 | `07_embed_chromosomes.sh` | Build whole-chromosome features |
| 08 | `08_train_chromosome_model.sh` | Train one chromosome LOO split |
| 09 | `09_predict_chromosome_ensemble.py` | Apply shipped chromosome models |

The [root manual](../README.md) defines input schemas, complete commands,
training behavior, output files, and limitations. Scripts 05 and 08 are
evaluation/reproduction paths. Scripts 06 and 09 apply packaged cross-fit
components as research ensembles to new genomes.
