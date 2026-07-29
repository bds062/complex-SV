# Top LOO false positives

Model: `localization_model`

| Class | Plotted | Total available false positives |
|---|---:|---:|
| BFB | 10 | 49 |
| chromothripsis | 10 | 31 |
| ecDNA | 10 | 26 |
| seismic_amplification | 4 | 4 |

Plot filenames begin with their one-based within-class rank. Plots are ranked by held-out score for localization and by probability-minus-threshold margin for chromosome classification.
Localization false positives use strict same-class, one-to-one matching at overlap coefficient >= 0.5. Seismic amplification has only four such false positives, so all four are shown.
