# Updated chromothripsis Pipeline24 frozen-event results

## Overall

```
                                         method  labels  n_predictions  true_predictions  localization_recall  classified_recall  classified_precision  classified_f1  classified_f2
                Old model / original 108 labels     108            163                53                0.574              0.491                 0.325          0.391          0.445
Old model / updated labels on shared 37 genomes     112            163                55                0.571              0.491                 0.337          0.400          0.450
Retrained / updated labels on shared 37 genomes     112            172                47                0.491              0.420                 0.273          0.331          0.379
            Retrained / full updated 39 genomes     114            179                47                0.482              0.412                 0.263          0.321          0.370
```

## Per class

```
      evaluation                 label  calls  predictions  true_predictions  recall  precision    f1    f2
      old_shared                 ecDNA     13           33                 7   0.538      0.212 0.304 0.412
retrained_shared                 ecDNA     13           34                 6   0.462      0.176 0.255 0.349
  retrained_full                 ecDNA     13           34                 6   0.462      0.176 0.255 0.349
      old_shared        chromothripsis     40           41                12   0.300      0.293 0.296 0.299
retrained_shared        chromothripsis     40           47                 8   0.200      0.170 0.184 0.193
  retrained_full        chromothripsis     42           50                 8   0.190      0.160 0.174 0.183
      old_shared                   BFB     42           75                26   0.619      0.347 0.444 0.535
retrained_shared                   BFB     42           76                24   0.571      0.316 0.407 0.492
  retrained_full                   BFB     42           80                24   0.571      0.300 0.393 0.484
      old_shared seismic_amplification     17           14                10   0.588      0.714 0.645 0.610
retrained_shared seismic_amplification     17           15                 9   0.529      0.600 0.562 0.542
  retrained_full seismic_amplification     17           15                 9   0.529      0.600 0.562 0.542
```

## Design

- The candidate-bag neural model is retrained from scratch in every leave-one-genome-out fold.
- The architecture, feature inputs, hard-negative/event-bag loss, event clustering, max score pooling, representative/envelope choices, scale grid, NMS, output caps, and one-to-one F1 calibration are unchanged.
- The only intended data change is 42 chromothripsis labels instead of 36; the 42 BFB, 13 ecDNA, and 17 seismic-amplification labels are unchanged.
- The paired 37-genome rows distinguish improvement from merely adding two newly labeled evaluation genomes.

## Interpretation

Retraining does not help. On the fair shared-genome comparison, F1 falls from 0.400 to 0.331; on all 39 updated genomes it is 0.321. The old model already recovers two of the six added labels, so merely updating the truth table raises its F1 from 0.391 to 0.400 without changing the model.

Five of the six additions are LOW-confidence ShatterSeek calls. All six have proposal overlap coefficient at least 0.973, so candidate generation is not limiting them. Both the old and retrained models recover only the same two additions (MS751 chr21 and QGU chr2). Validation membership changes in only one of 37 paired folds, ruling out split changes as the main cause.

The extra chromothripsis supervision perturbs the shared neural representation and degrades every class on paired evaluation: chromothripsis F1 0.296 to 0.184, BFB 0.444 to 0.407, ecDNA 0.304 to 0.255, and seismic amplification 0.645 to 0.563. This is negative transfer/small-sample instability, not a proposal recall problem. Keep the prior frozen model as the preferred checkpoint.

Diagnostics: `added_label_diagnostics.tsv`, `validation_split_comparison.tsv`, and `paired_fold_metrics.tsv`.
