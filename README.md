# Multiscale treatment-site analysis in focal epilepsy

Research code for *Interpreting treated sites in focal epilepsy across scales and neighbourhoods*.

## Setup

Use Python 3.12 and install the dependencies in an isolated environment.

```shell
python -m pip install -r requirements.txt
```

Dependencies reflect the development environment. Installation in a clean environment has not been verified.

Institutional workflows require private recordings, clinical annotations, anatomical tables and cohort settings. Configure these inputs using [docs/configuration.md](docs/configuration.md). Public configurations contain invented participant examples and aggregate study checks.

## Analysis workflow

Entry points below are in `scripts/` unless another directory is shown.

| Stage | Entry points |
|---|---|
| Segmentation and time axes | `split_continuous_edf.py`, `convert_single_trial_edf_to_fif.py`, `normalize_segment_timebase.py` |
| Signal preparation | `resample_segments_10k_to_2k.py`, `clean_contacts_from_resampled.py`, `build_bipolar_from_clean_contacts.py`, `notch_bipolar_line_noise.py`, `apply_patient_specific_notch.py` |
| Local features | `compute_line_length_matrices.py`, `compute_band_power_matrices.py`, `extract_electrode_pair_features.py` |
| Regional features | `compute_network_matrices.py`, `extract_region_network_features.py`, `build_region_features.py` |
| Patient-held-out models | `run_electrode_pair_lopo_modeling.py`, `run_region_lopo_modeling.py` |
| Statistical summaries | `compute_pair_max_t.py` and regional correction and secondary metrics from `run_region_lopo_modeling.py` |
| Cross-scale comparisons | `extract_cross_scale_features.py`, `run_cross_scale_models.py`, `run_region_representation_comparison.py`, `run_cross_scale_analysis.py` |
| Spatial diffusion and controls | `build_bipolar_coordinate_manifest.py`, `run_electrode_graph_smoothing.py`, `run_shaft_graph_controls.py` |
| Supplementary analyses | `run_region_coverage.py`, `analyze_region_contact_residual_burden.py`, `analyze_regional_false_negatives.py`, `run_excluded_contact_graph_extension.py` |
| External analyses | See [external_pair/README.md](external_pair/README.md). Regional entries are `build_external_dk_mapping.py`, `run_external_region_classification.py`, `run_external_region_permutations.py` |

The primary institutional regional specification is `configs/regional_analysis/region_models.yaml`. `region_features.yaml` defines extraction inputs. `modeling.top_k_features` controls feature selection, while `labels.rule.top_k` controls the treatment-reference region rule.

Each stage requires the preceding signal, feature or prediction tables. Use a new output directory when data, labels, scientific settings or code change.

Institutional and external models are fitted separately. Treatment annotations provide the clinical reference labels. The institutional and external bipolar-pair models select features on each outer training set before inner C tuning. External regional models repeat feature selection in each inner training fold.

## Tests

```shell
python -m pytest external_pair -q -p no:cacheprovider
```

Tests use synthetic arrays and temporary files to check numerical calculations, feature aggregation, path protection, configuration and checkpoint handling. Full clinical results require the original inputs and a separate reproduction run.

## Data and permissions

Institutional workflows require authorised access to recordings, annotations and participant-specific settings. The external data source is OpenNeuro ds004100. External pair evaluation starts from prepared feature tables. The complete reviewed raw-recording-to-feature configuration remains to be supplied. EDF/FIF outputs can retain recording dates and annotations and require a separate privacy review before sharing.

See [RIGHTS_NOTICE.md](RIGHTS_NOTICE.md) for software and data permissions.
