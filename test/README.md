# Integration tests

The tests exercise the packaged release on one real genome. The committed
defaults point to the AU565 inputs on the Kolmogorov Lab HPC. No source data or
generated test output is copied into Git.

## Configure paths

Copy the default configuration and replace paths as needed:

```bash
cp test/paths.hpc.env test/paths.local.env
$EDITOR test/paths.local.env
```

Every setting can instead be overridden in the environment. For example:

```bash
SAMPLE_ID=sample_A \
WAKHAN_ROOT=/data/sample_A/wakhan/sample_A \
SEVERUS_VCF=/data/sample_A/severus.vcf \
LABEL_SOURCE_DIR=/data/curated_labels \
TEST_OUTPUT_DIR=/scratch/complex_sv_test \
test/run_test.sh quick
```

The path configuration contains:

- one Wakhan root and matching Severus VCF;
- the directory holding the seven curated label TSVs;
- a writable output directory;
- device, proposal profile, and candidate-cap test settings.

## Quick test

```bash
test/run_test.sh quick test/paths.hpc.env
```

This checks dependencies and model artifacts, rebuilds all four-class label
tables, and runs the candidate generator. The default cap of 25 proposals
keeps it suitable as a smoke test; it is not a benchmark run.

## Full test

```bash
test/run_test.sh full test/paths.hpc.env
```

The full mode additionally embeds candidates, applies the localization
ensemble, embeds every chromosome, and applies the chromosome ensemble. Set
`TEST_DEVICE=cuda` when a compatible GPU is available. The full test verifies
integration and checkpoint compatibility; its capped single-genome predictions
must not be interpreted as held-out performance.

Both modes fail immediately for missing inputs, malformed labels, dependency
problems, or unsuccessful pipeline stages. All outputs are written beneath
`TEST_OUTPUT_DIR`, which defaults outside the repository.
