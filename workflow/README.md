# Numbered workflow

Run commands from the repository root. Set `PYTHON_BIN` when the desired
interpreter is not named `python`.

| Step | Command | Purpose |
|---:|---|---|
| 00 | `00_check_install.py` | Validate dependencies and packaged models |
| 01 | `01_pretrain_cn_encoder.sh` | Pretrain the Wakhan CN autoencoder |
| 02 | `02_pretrain_sv_encoder.sh` | Pretrain the Severus graph autoencoder |
| 03 | `03_generate_candidates.sh` | Compatibility entry point for `candidate_generator/run.sh` |
| 04 | `04_embed_candidates.sh` | Build localization-model features |
| 05 | `05_train_localization_model.sh` | Train one localization LOO split |
| 06 | `06_predict_localization_ensemble.py` | Apply the held-out localization ensemble for evaluation |
| 07 | `07_embed_chromosomes.sh` | Build whole-chromosome features |
| 08 | `08_train_chromosome_model.sh` | Train one chromosome LOO split |
| 09 | `09_predict_chromosome_ensemble.py` | Apply the held-out chromosome ensemble for evaluation |
| 10 | `10_predict_localization.py` | Apply the final localization model fit on all 48 genomes |
| 11 | `11_predict_chromosomes.py` | Apply the final chromosome model fit on all 48 genomes |
| 12 | `12_predict_new_dataset.sh` | Run both final models on a new manifest |

The [root manual](../README.md) defines input schemas, complete commands,
training behavior, output files, and limitations. Scripts 05, 06, 08, and 09
are evaluation/reproduction paths. Scripts 10--12 are the supported application
paths for new datasets. `train_localization_all.py` creates a final all-genome
localization fit after deriving its fixed epoch count and decoder settings from
completed genome-held-out runs.
