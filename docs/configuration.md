# Configuration

## Private inputs

Public YAMLs contain `release_example_only: true`. `demo001` and `demo002` are invented examples. The `expected` values describe the study analysis, including 84 eligible institutional regions and 22 positives.

Create a private configuration tree outside the code directory, preserving the relative filenames. Replace the example cohort, paths, individual rules and expected counts with the values for the authorised dataset. Remove `release_example_only` from reviewed private copies. `SEEG_ALLOW_EXAMPLE_CONFIG=1` enables template loading for synthetic checks.

Set `SEEG_PRIVATE_CONFIG` to a JSON containing the cohort fields illustrated in `configs/cohort.example.json`. This supplies patient membership, trial counts, anatomical folder mappings, manual pair exclusions and continuous-recording settings.

The annotation workbook is expected at `annotations/clinical_annotations.xlsx` below `SEEG_DATA_ROOT`. Its worksheets and column names follow the readers in `scripts/`. Recordings and intermediate CSV/NPZ inputs must follow the schemas written by the preceding stages.

## Environment variables

| Variable | Purpose |
|---|---|
| `SEEG_CONFIG_ROOT` | Private YAML directory, defaulting to the bundled `configs/` |
| `SEEG_PRIVATE_CONFIG` | Absolute path to the private cohort JSON |
| `SEEG_OUTPUT_ROOT` | Writable analysis workspace, defaulting to `work/` |
| `SEEG_DATA_ROOT` | Read-only institutional annotation and reference-input directory |
| `SEEG_AUX_ROOT` | Read-only anatomical auxiliary directory, including `brainstorm_electrode_region_labels/` |
| `SEEG_BRAINSTORM_ROOT` | Brainstorm protocol parent containing `data/` and `anat/` |
| `SEEG_EXTERNAL_REGION_ROOT` | Prepared external regional mapping inputs |
| `SEEG_EXTERNAL_PREPARATION_ROOT` | Prepared JSON inputs for external atlas mapping |
| `SEEG_ALLOW_EXAMPLE_CONFIG` | Set to `1` for synthetic template inspection |

Configure the source roots required by the selected stage. Output directories must be separate from the configured source roots. The automatically resolved `${SEEG_REPO_ROOT}` placeholder points to the code directory.

Intermediate inputs and outputs share `SEEG_OUTPUT_ROOT/data_derived/`. Regional feature and model outputs use its `regional_analysis/` subdirectory. Individual commands also provide output options with stage-specific path checks.

## Analysis configurations

| Configuration | Purpose |
|---|---|
| `electrode_pair_modeling.yaml` | Local features and bipolar-pair models |
| `region_features.yaml` | Regional connectivity feature extraction |
| `regional_analysis/region_models.yaml` | Primary regional models and inference |
| `cross_scale_models.yaml` | Cross-scale feature extraction and modelling |
| `region_representation_comparison.yaml` | Regional representation comparison |
| `regional_analysis/cross_scale_analysis.yaml` | Cross-scale and residual-error summaries |
| `regional_analysis/coverage_models.yaml` | Coverage controls, reference sensitivity and patient-deletion refits |
| `electrode_graph_smoothing.yaml` | Coordinates and spatial diffusion |
| `regional_analysis/shaft_graph_controls.yaml` | Shaft-conditioned spatial controls |
| `excluded_contact_graph_extension.yaml` | Harmonic extension to excluded tissue |
| `region_contact_residual_burden.yaml`, `regional_false_negatives.yaml` | Regional residual-error analyses |
| `patient_specific_notch.yaml` | Individual narrowband filter rules |
| `external_region_ds004100.yaml` | External regional workflow |

The external regional mapping directory contains `labels_both_end_ge1_20260914_v1/` with `rule.json`, `cohort.json`, `region_labels_primary.json` and `preprocessed_signal_manifest_active.json`, plus anatomy, region-count and pair-audit JSON files at its root. Existing data-rule identifiers are checked against the supplied manifests.
