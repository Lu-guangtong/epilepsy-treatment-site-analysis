# External bipolar-pair analysis

Run commands from the repository root. The final cohort contains 13 ds004100 participants. Common-band feature tables cover delta through high gamma. Ripple and exploratory fast-ripple features use separate native-rate tables.

## Required inputs

The cohort root contains `labels/external_pair_labels_13_1024hz.csv`. The feature directory contains the following files.

- `patient_features_label_blind.csv`
- `native_ripple_1024hz_patient_features_label_blind.csv`
- `native_fast_ripple_1024hz_patient_features_label_blind.csv`

The label table supplies `patient`, `bipolar_channel`, `label_binary` and `primary_status`. Feature columns follow `feature__<band>__<method>__<summary>`. Graph analysis also requires pair midpoint coordinates and shaft identifiers in the label table.

## Patient-held-out models

```shell
python external_pair/run_pair_lopo.py --data-root <cohort-root> --feature-root <feature-directory> --output-root <model-output>
```

Use `--bands` to select frequency bands. Outputs include predictions, selected features, inner C scores, thresholds and patient-macro metrics for each band.

## Spatial diffusion

```shell
python external_pair/run_graph_diffusion.py --prediction-root <model-output> --labels <label-csv> --output-root <graph-output>
```

## Full-refit permutations

```shell
python external_pair/run_permutation_tests.py prepare --output-base <permutation-base> --run-dir <permutation-base/run_01> --labels <label-csv> --feature-root <feature-directory> --reference-run <model-output> --graph-run <graph-output> --permutations 1000
python external_pair/run_permutation_tests.py validate --output-base <permutation-base> --run-dir <permutation-base/run_01>
python external_pair/run_permutation_tests.py run --output-base <permutation-base> --run-dir <permutation-base/run_01>
python external_pair/run_permutation_tests.py summarize --output-base <permutation-base> --run-dir <permutation-base/run_01>
```

The preparation stage freezes input and code hashes. Validation checks the numerical engine against the reference model before the full run. `status` reports progress and `benchmark` compares worker counts. Output directories are isolated from inputs and protected by a single-writer lock. Resume checks verify the saved configuration, sources and cache.

## Shared functions

`pair_analysis/` contains signal filters and resampling, local feature construction and seizure aggregation, model primitives, graph functions and output-path checks. `features.extract_recording_features` accepts a recording record, a bipolar-pair table and extraction settings. `features.aggregate_patient_features` aggregates the resulting seizure-level rows.

These feature functions require independently verified recording membership, pair definitions, onset metadata and sampling settings. The complete reviewed configuration from raw recordings to the final cohort feature tables is required separately.

## Tests

```shell
python -m pytest external_pair -q -p no:cacheprovider
```
