"""Evaluate cross-scale representations and regional residual errors."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = output_root()
SCRIPT_DIR = REPO_ROOT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import analyze_regional_false_negatives as fn_concentration  # noqa: E402
import analyze_region_contact_residual_burden as residual  # noqa: E402
import run_cross_scale_models as cross_model  # noqa: E402
import run_region_representation_comparison as cross_driver  # noqa: E402
import run_region_lopo_modeling as region_model  # noqa: E402


RUN_ID = "regional_analysis"
ALGORITHM_VERSION = "cross_scale_analysis_tie_corrected_v1"
AP_IMPLEMENTATION = "sklearn_equivalent_tie_grouped_v2"
AP_DEFINITION = "sklearn.metrics.average_precision_score"
CONFIG_PATH = config_root() / RUN_ID / "cross_scale_analysis.yaml"
ANALYSIS_ROOT = ROOT / "data_derived" / RUN_ID
OUTPUT_ROOT = ANALYSIS_ROOT / "cross_scale_analysis"
SMOKE_ROOT = ROOT / "checks" / RUN_ID
SOURCE_PATH = Path(__file__).resolve()


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(path: Path) -> None:
    """Restrict writes to the analysis output or a dedicated test directory."""
    assert_output_path(path)
    resolved = _resolve(path)
    formal = _resolve(OUTPUT_ROOT)
    smoke = _resolve(SMOKE_ROOT)
    if resolved == formal:
        return
    if _is_relative_to(resolved, smoke) and resolved != smoke:
        return
    raise SystemExit(f"Refusing to write outside the cross-scale analysis directories: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = expand_config(yaml.safe_load(handle))
    if not isinstance(payload, dict):
        raise SystemExit(f"Configuration must be a mapping: {path}")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path: Path) -> tuple[str, int]:
    """Hash a file or tree, including relative filenames for directories."""
    target = _resolve(path)
    if target.is_file():
        return file_sha256(target), 1
    if not target.is_dir():
        raise SystemExit(f"Missing input/protected path: {target}")
    digest = hashlib.sha256()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest(), len(files)


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        suffix=".csv",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(path)


def atomic_json(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ["numpy", "pandas", "scikit-learn", "PyYAML"]:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def snapshot_paths(paths: Iterable[Path | str]) -> dict[str, dict[str, object]]:
    snapshot: dict[str, dict[str, object]] = {}
    for value in paths:
        path = _resolve(value)
        digest, n_files = path_sha256(path)
        snapshot[str(path)] = {"sha256": digest, "n_files": n_files}
    return snapshot


def input_paths(config: Mapping[str, object]) -> dict[str, Path]:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise SystemExit("Configuration inputs must be a mapping")
    required = [
        "corrected_region_predictions",
        "corrected_region_labels",
        "local_signal_region_features",
        "native_contact_predictions",
        "reused_network_predictions",
        "reused_network_metrics",
        "reused_network_folds",
        "reused_network_comparison",
    ]
    missing_keys = [key for key in required if key not in inputs]
    if missing_keys:
        raise SystemExit(f"Configuration inputs are missing keys: {missing_keys}")
    paths = {key: _resolve(str(inputs[key])) for key in required}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise SystemExit(f"Downstream inputs are missing: {missing_files}")
    return paths


def validate_contract(config: Mapping[str, object]) -> None:
    if str(config.get("run_id")) != RUN_ID:
        raise SystemExit(f"run_id must be {RUN_ID}")
    if str(config.get("average_precision_implementation")) != AP_IMPLEMENTATION:
        raise SystemExit(f"average_precision_implementation must be {AP_IMPLEMENTATION}")
    if str(config.get("rank_tie_method")) != "average":
        raise SystemExit("rank_tie_method must be average")
    if region_model.AVERAGE_PRECISION_IMPLEMENTATION != AP_IMPLEMENTATION:
        raise SystemExit("Loaded region model does not expose the reviewed AP implementation")
    if region_model.AVERAGE_PRECISION_DEFINITION != AP_DEFINITION:
        raise SystemExit("Loaded region model has an unexpected AP definition")
    modeling = config.get("modeling")
    if not isinstance(modeling, Mapping):
        raise SystemExit("Configuration modeling section must be a mapping")
    if str(modeling.get("univariate_orientation_rule")) != region_model.UNIVARIATE_ORIENTATION_RULE:
        raise SystemExit("Local-to-region orientation must be learned from inner training patients")


def build_execution_fingerprint(
    config: Mapping[str, object], config_path: Path, inputs: Mapping[str, Path]
) -> tuple[str, dict[str, object]]:
    source_paths = {
        "driver": SOURCE_PATH,
        "region_model": Path(region_model.__file__).resolve(),
        "cross_model": Path(cross_model.__file__).resolve(),
        "cross_driver": Path(cross_driver.__file__).resolve(),
        "residual_analysis": Path(residual.__file__).resolve(),
        "fn_concentration": Path(fn_concentration.__file__).resolve(),
    }
    payload: dict[str, object] = {
        "run_id": RUN_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "average_precision_implementation": AP_IMPLEMENTATION,
        "rank_tie_method": "average",
        "config": {"path": str(_resolve(config_path)), "sha256": file_sha256(config_path)},
        "inputs": {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in sorted(inputs.items())
        },
        "sources": {
            key: {"path": str(path), "sha256": file_sha256(path)}
            for key, path in sorted(source_paths.items())
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "configured_analysis": config.get("analysis"),
        "configured_modeling": config.get("modeling"),
        "frequency_bands": config.get("frequency_bands"),
    }
    return canonical_hash(payload), payload


def _add_canonical_ap_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    alias_pairs = [
        ("AUPRC", "average_precision"),
        ("normalized_AUPRC", "normalized_average_precision"),
        ("cross_AUPRC", "cross_average_precision"),
        ("native_AUPRC", "native_average_precision"),
        ("delta_cross_minus_native_AUPRC", "delta_cross_minus_native_average_precision"),
        ("cross_normalized_AUPRC", "cross_normalized_average_precision"),
        ("native_normalized_AUPRC", "native_normalized_average_precision"),
        (
            "delta_cross_minus_native_normalized_AUPRC",
            "delta_cross_minus_native_normalized_average_precision",
        ),
    ]
    for alias, canonical in alias_pairs:
        if alias in output.columns and canonical not in output.columns:
            output[canonical] = output[alias]
        if canonical in output.columns and alias not in output.columns:
            output[alias] = output[canonical]
    output["average_precision_definition"] = AP_DEFINITION
    output["AUPRC_semantics"] = "compatibility_alias_for_average_precision_not_trapezoidal_auc"
    return output


def build_patient_region_rankings(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "frequency_band",
        "patient",
        "desikan_killiany_label",
        "y_true",
        "y_score",
        "rank_within_patient",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise SystemExit(f"Corrected region predictions lack ranking fields: {missing}")
    output = predictions.copy()
    groups = output.groupby(["frequency_band", "patient"], sort=False)
    output["rank_within_patient_recomputed"] = groups["y_score"].rank(
        method="average", ascending=False
    )
    if not np.allclose(
        output["rank_within_patient"].to_numpy(float),
        output["rank_within_patient_recomputed"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise SystemExit("Stored corrected region ranks disagree with average-tie recomputation")
    output["score_percentile_within_patient"] = groups["y_score"].rank(
        method="average", pct=True, ascending=True
    )
    output["n_regions_within_patient"] = groups["y_score"].transform("size").astype(int)
    output["score_tie_size"] = output.groupby(
        ["frequency_band", "patient", "y_score"], sort=False
    )["y_score"].transform("size").astype(int)
    output["is_top3_average_rank"] = output["rank_within_patient_recomputed"].le(3.0)
    output["is_top20pct_average_rank"] = output["rank_within_patient_recomputed"].le(
        np.ceil(0.20 * output["n_regions_within_patient"])
    )
    output["rank_tie_method"] = "average"
    output["average_precision_implementation"] = AP_IMPLEMENTATION
    output = output.sort_values(
        ["frequency_band", "patient", "rank_within_patient_recomputed", "desikan_killiany_label"],
        kind="mergesort",
    ).reset_index(drop=True)
    top3 = output[output["is_top3_average_rank"]].copy()
    return output, top3.reset_index(drop=True)


def _validate_native_region_predictions(
    predictions: pd.DataFrame, bands: Sequence[str], expected: Mapping[str, object]
) -> None:
    required = {
        "frequency_band",
        "patient",
        "heldout_patient",
        "desikan_killiany_label",
        "y_true",
        "y_score",
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise SystemExit(f"Corrected region predictions are missing columns: {missing}")
    if len(predictions) != int(expected["region_prediction_rows"]):
        raise SystemExit("Corrected region prediction row count is not 672")
    if set(predictions["frequency_band"].astype(str)) != set(bands):
        raise SystemExit("Corrected region predictions do not cover all configured bands")
    if not predictions["patient"].astype(str).eq(predictions["heldout_patient"].astype(str)).all():
        raise SystemExit("Corrected region predictions contain non-heldout rows")
    keys = ["frequency_band", "patient", "desikan_killiany_label"]
    if predictions.duplicated(keys).any():
        raise SystemExit("Corrected region predictions contain duplicate identities")


def build_local_comparison(
    local_metrics: pd.DataFrame, native_metrics: pd.DataFrame
) -> pd.DataFrame:
    native = native_metrics.set_index("frequency_band")
    records: list[dict[str, object]] = []
    for row in local_metrics.sort_values("frequency_band").itertuples(index=False):
        band = str(row.frequency_band)
        reference = native.loc[band]
        record: dict[str, object] = {
            "direction": "local_signal_to_region",
            "target_level": "brain_region",
            "frequency_band": band,
            "native_method": "wpli2_plus_imcoh_to_region_corrected",
            "cross_method": "line_length_plus_band_power_to_region",
            "execution_status": "rerun_with_tie_corrected_nested_lopo",
        }
        for canonical, alias in [
            ("average_precision", "AUPRC"),
            ("normalized_average_precision", "normalized_AUPRC"),
            ("AUROC", "AUROC"),
        ]:
            cross_value = float(getattr(row, canonical))
            native_value = float(reference[canonical])
            record[f"cross_{canonical}"] = cross_value
            record[f"native_{canonical}"] = native_value
            record[f"delta_cross_minus_native_{canonical}"] = cross_value - native_value
            record[f"cross_{alias}"] = cross_value
            record[f"native_{alias}"] = native_value
            record[f"delta_cross_minus_native_{alias}"] = cross_value - native_value
        records.append(record)
    return _add_canonical_ap_columns(pd.DataFrame.from_records(records))


def build_network_reuse_tables(
    comparison: pd.DataFrame, audit: Mapping[str, object]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    network = comparison.copy()
    network["direction"] = "network_to_contact"
    network["target_level"] = "bipolar_contact"
    network["execution_status"] = "reused_verified_not_retrained"
    network = _add_canonical_ap_columns(network)
    rows: list[dict[str, object]] = []
    source_files = audit.get("source_files", {})
    if not isinstance(source_files, Mapping):
        raise SystemExit("Network reuse audit lacks source file hashes")
    for role, item in source_files.items():
        if not isinstance(item, Mapping):
            raise SystemExit("Malformed network reuse source audit")
        rows.append(
            {
                "direction": "network_to_contact",
                "reuse_status": "reused_verified_not_retrained",
                "model_fit_calls": 0,
                "source_role": str(role),
                "source_path": str(item["path"]),
                "source_sha256": str(item["sha256"]),
            }
        )
    native = audit.get("native_electrode_predictions")
    if isinstance(native, Mapping):
        rows.append(
            {
                "direction": "network_to_contact",
                "reuse_status": "reused_verified_not_retrained",
                "model_fit_calls": 0,
                "source_role": "native_contact_predictions",
                "source_path": str(native["path"]),
                "source_sha256": str(native["sha256"]),
            }
        )
    return network, pd.DataFrame.from_records(rows)


def bootstrap_local_comparison(
    local_predictions: pd.DataFrame,
    native_predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = cross_model._bootstrap_direction_metrics(
        "electrode_to_region",
        local_predictions,
        native_predictions,
        n_bootstrap,
        seed,
    )
    bootstrap["direction"] = "local_signal_to_region"
    bootstrap = _add_canonical_ap_columns(bootstrap)
    ci_records: list[dict[str, object]] = []
    statistics = [
        "cross_average_precision",
        "native_average_precision",
        "delta_cross_minus_native_average_precision",
        "cross_normalized_average_precision",
        "native_normalized_average_precision",
        "delta_cross_minus_native_normalized_average_precision",
        "cross_AUROC",
        "native_AUROC",
        "delta_cross_minus_native_AUROC",
    ]
    for band, frame in bootstrap.groupby("frequency_band", sort=True):
        for statistic in statistics:
            values = frame[statistic].dropna().to_numpy(float)
            ci_records.append(
                {
                    "direction": "local_signal_to_region",
                    "frequency_band": str(band),
                    "statistic": statistic,
                    "n_valid_bootstrap": len(values),
                    "ci_lower_95": float(np.quantile(values, 0.025)),
                    "ci_upper_95": float(np.quantile(values, 0.975)),
                }
            )
    ci = _add_canonical_ap_columns(pd.DataFrame.from_records(ci_records))
    return bootstrap, ci


def run_local_signal_to_region(
    config: dict[str, object], inputs: Mapping[str, Path], n_bootstrap: int
) -> tuple[dict[str, pd.DataFrame], dict[str, object], pd.DataFrame, pd.DataFrame]:
    expected = config["expected"]
    modeling = config["modeling"]
    analysis = config["analysis"]
    if not isinstance(expected, Mapping) or not isinstance(modeling, Mapping) or not isinstance(analysis, Mapping):
        raise SystemExit("expected/modeling/analysis sections must be mappings")
    bands = [str(value) for value in config["frequency_bands"]]
    features = pd.read_csv(inputs["local_signal_region_features"])
    labels, label_counts = cross_driver.load_locked_region_labels(
        inputs["corrected_region_labels"], expected
    )
    validation = cross_driver.validate_region_inputs(features, labels, config)
    local_predictions, local_metrics, local_folds = cross_driver.run_electrode_to_region(
        features, labels, config
    )
    cross_driver.validate_region_outputs(local_predictions, local_metrics, local_folds, config)
    local_predictions["direction"] = "local_signal_to_region"
    local_predictions["legacy_direction_alias"] = "electrode_to_region"
    local_predictions["average_precision_implementation"] = AP_IMPLEMENTATION
    local_metrics["direction"] = "local_signal_to_region"
    local_folds["direction"] = "local_signal_to_region"
    local_metrics = _add_canonical_ap_columns(local_metrics)
    local_folds["average_precision_implementation"] = AP_IMPLEMENTATION

    native_predictions = pd.read_csv(inputs["corrected_region_predictions"])
    native_predictions = native_predictions[
        native_predictions["frequency_band"].astype(str).isin(bands)
    ].copy()
    _validate_native_region_predictions(native_predictions, bands, expected)
    cross_model._validate_prediction_alignment(
        local_predictions,
        native_predictions,
        cross_model.REGION_KEYS,
    )
    native_metrics = _add_canonical_ap_columns(
        region_model.compute_prediction_metrics_by_band(native_predictions)
    )
    local_patient_metrics = _add_canonical_ap_columns(
        region_model.compute_patient_prediction_metrics_by_band(local_predictions)
    )
    local_patient_metrics["direction"] = "local_signal_to_region"
    comparison = build_local_comparison(local_metrics, native_metrics)
    bootstrap, bootstrap_ci = bootstrap_local_comparison(
        local_predictions,
        native_predictions,
        n_bootstrap=n_bootstrap,
        seed=int(analysis["random_seed"]),
    )
    frames = {
        "local_signal_to_region_predictions_by_band.csv": local_predictions,
        "local_signal_to_region_metrics_by_band.csv": local_metrics,
        "local_signal_to_region_patient_metrics_by_band.csv": local_patient_metrics,
        "local_signal_to_region_fold_details_by_band.csv": local_folds,
        "local_signal_to_region_bootstrap_results.csv": bootstrap,
        "local_signal_to_region_bootstrap_ci.csv": bootstrap_ci,
    }
    audit = {
        "label_counts": label_counts,
        "input_validation": validation,
        "n_predictions": len(local_predictions),
        "n_folds": len(local_folds),
        "n_metric_rows": len(local_metrics),
        "n_bootstrap": n_bootstrap,
        "model_fitted": True,
        "fit_scope": "local_signal_to_region_only",
    }
    return frames, audit, comparison, native_predictions


def run_residual_analyses(
    config: dict[str, object],
    inputs: Mapping[str, Path],
    *,
    n_bootstrap: int,
    n_random_rankings: int,
    n_fn_permutations: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    analysis = config["analysis"]
    expected = config["expected"]
    expected_fn = config["expected_fn"]
    if not isinstance(analysis, Mapping) or not isinstance(expected, Mapping) or not isinstance(expected_fn, Mapping):
        raise SystemExit("analysis/expected/expected_fn sections must be mappings")
    bands = [str(value) for value in config["frequency_bands"]]
    labels = residual.load_region_labels(inputs["corrected_region_labels"])
    region_predictions = residual.load_region_predictions(
        inputs["corrected_region_predictions"], bands
    )
    contact_predictions = residual.load_contact_predictions(
        inputs["native_contact_predictions"], bands
    )
    contact_table, region_table = residual.build_joint_tables(
        region_predictions, contact_predictions, labels
    )
    residual.validate_full_run(contact_table, region_table, labels, bands, expected)

    patient_burden, band_burden = residual.summarize_suspicious_burden(region_table, "yes")
    patient_status, patient_status_comparison, status_comparison = (
        residual.compare_region_status_fn_rates(region_table)
    )
    associations = residual.region_score_associations(region_table)
    capture_patient, capture_curves = residual.build_capture_analysis(
        region_table,
        n_random_rankings=n_random_rankings,
        grid_points=int(analysis["capture_curve_grid_points"]),
        seed=int(analysis["random_seed"]),
    )
    capture_band = residual.summarize_capture(capture_patient)
    observed = residual.build_observed_metrics(
        patient_burden,
        patient_status_comparison,
        associations,
        capture_patient,
    )
    bootstrap, bootstrap_ci = residual.patient_bootstrap(
        region_table,
        patient_burden,
        patient_status_comparison,
        capture_patient,
        observed,
        n_iterations=n_bootstrap,
        seed=int(analysis["random_seed"]),
    )

    fn_patient = fn_concentration.build_patient_band_summary(contact_table)
    fn_band = fn_concentration.build_band_summary(fn_patient)
    fn_concentration.validate_full_run(
        contact_table, fn_patient, fn_band, bands, expected_fn
    )
    fn_bootstrap, fn_bootstrap_ci = fn_concentration.patient_bootstrap(
        fn_patient,
        bands=bands,
        n_iterations=n_bootstrap,
        seed=int(analysis["random_seed"]),
    )
    fn_permutation, fn_permutation_summary = fn_concentration.within_patient_permutation(
        contact_table,
        fn_band,
        bands=bands,
        n_permutations=n_fn_permutations,
        seed=int(analysis["random_seed"]),
    )

    frames = {
        "region_contact_predictions_by_band.csv": contact_table,
        "region_contact_residuals_by_band.csv": region_table,
        "suspicious_region_patient_band_burden.csv": patient_burden,
        "suspicious_region_burden_by_band.csv": band_burden,
        "region_status_patient_band_burden.csv": patient_status,
        "region_status_patient_band_fn_rate_comparison.csv": patient_status_comparison,
        "region_status_fn_rate_comparison_by_band.csv": status_comparison,
        "region_score_residual_association_by_band.csv": associations,
        "residual_fn_capture_patient_band.csv": capture_patient,
        "residual_fn_capture_curves.csv": capture_curves,
        "residual_fn_capture_by_band.csv": capture_band,
        "residual_bootstrap_results.csv": bootstrap,
        "residual_bootstrap_ci.csv": bootstrap_ci,
        "fn_concentration_patient_band.csv": fn_patient,
        "fn_concentration_by_band.csv": fn_band,
        "fn_concentration_bootstrap_results.csv": fn_bootstrap,
        "fn_concentration_bootstrap_ci.csv": fn_bootstrap_ci,
        "fn_concentration_permutation_results.csv": fn_permutation,
        "fn_concentration_permutation_summary.csv": fn_permutation_summary,
    }
    audit = {
        "n_contact_prediction_rows": len(contact_table),
        "n_region_prediction_rows": len(region_table),
        "n_bootstrap": n_bootstrap,
        "n_random_region_rankings": n_random_rankings,
        "n_fn_permutations": n_fn_permutations,
        "contact_model_refitted": False,
        "contact_predictions_reused_sha256": file_sha256(inputs["native_contact_predictions"]),
        "interpretation": "posthoc_region_ranking_and_residual_false_negative_analysis",
    }
    return frames, audit


def run_analysis(
    config: dict[str, object],
    inputs: Mapping[str, Path],
    *,
    n_local_bootstrap: int,
    n_residual_bootstrap: int,
    n_random_rankings: int,
    n_fn_permutations: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, object]]:
    local_frames, local_audit, local_comparison, native_region_predictions = (
        run_local_signal_to_region(config, inputs, n_local_bootstrap)
    )
    _, _, reused_comparison, network_audit = cross_driver.validate_reused_network_branch(config)
    network_comparison, network_audit_table = build_network_reuse_tables(
        reused_comparison, network_audit
    )
    residual_frames, residual_audit = run_residual_analyses(
        config,
        inputs,
        n_bootstrap=n_residual_bootstrap,
        n_random_rankings=n_random_rankings,
        n_fn_permutations=n_fn_permutations,
    )
    rankings, top3 = build_patient_region_rankings(native_region_predictions)
    comparison = pd.concat([network_comparison, local_comparison], ignore_index=True, sort=False)
    expected_rows = 2 * len(config["frequency_bands"])
    if len(comparison) != expected_rows:
        raise SystemExit(f"Cross-scale comparison row mismatch: {len(comparison)} != {expected_rows}")
    frames = {
        **local_frames,
        **residual_frames,
        "cross_scale_method_comparison.csv": comparison,
        "network_to_contact_reuse_audit.csv": network_audit_table,
        "patient_region_rankings_by_band.csv": rankings,
        "patient_region_top3_by_band.csv": top3,
    }
    summary = {
        "status": "computed",
        "run_id": RUN_ID,
        "algorithm_version": ALGORITHM_VERSION,
        "average_precision_implementation": AP_IMPLEMENTATION,
        "average_precision_definition": AP_DEFINITION,
        "rank_tie_method": "average",
        "local_signal_to_region": local_audit,
        "network_to_contact": {
            **network_audit,
            "model_fit_calls_in_this_runner": 0,
        },
        "residual_false_negative": residual_audit,
        "evidence_boundary": (
            "agreement_with_clinician_ablation_selection_only; no independent SOZ, "
            "outcome, external-cohort, or clinical-utility endpoint"
        ),
        "generated_reports": False,
        "generated_formal_figures": False,
    }
    return frames, summary


def output_paths(out_dir: Path, frame_names: Iterable[str]) -> dict[str, Path]:
    paths = {name: out_dir / name for name in frame_names}
    paths["run_summary.json"] = out_dir / "run_summary.json"
    paths["run_manifest.json"] = out_dir / "run_manifest.json"
    return paths


def ensure_new_output_root(out_dir: Path) -> None:
    assert_output_path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(
            f"Output directory is not empty: {out_dir}. Use --resume only for a complete matching run."
        )


def verify_complete_resume(out_dir: Path, fingerprint: str) -> dict[str, object]:
    manifest_path = out_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit("Cannot resume: complete run_manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("execution_fingerprint") != fingerprint:
        raise SystemExit("Cannot resume: configuration/input/code fingerprint changed")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise SystemExit("Cannot resume: manifest output hashes are missing")
    for name, item in outputs.items():
        if not isinstance(item, Mapping):
            raise SystemExit("Cannot resume: malformed output hash entry")
        path = out_dir / str(name)
        if not path.is_file() or file_sha256(path) != str(item.get("sha256")):
            raise SystemExit(f"Cannot resume: output is missing or changed: {path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--n-local-bootstrap", type=int)
    parser.add_argument("--n-residual-bootstrap", type=int)
    parser.add_argument("--n-random-rankings", type=int)
    parser.add_argument("--n-fn-permutations", type=int)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _positive_count(value: int, label: str) -> int:
    if value <= 0:
        raise SystemExit(f"{label} must be positive")
    return value


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_contract(config)
    out_dir = _resolve(args.out or str(config["output_root"]))
    assert_allowed_output_root(out_dir)
    inputs = input_paths(config)
    fingerprint, fingerprint_payload = build_execution_fingerprint(config, args.config, inputs)
    protected_values = config.get("protected_input_roots")
    if not isinstance(protected_values, list) or not protected_values:
        raise SystemExit("protected_input_roots must be a non-empty list")
    protected_before = snapshot_paths([Path(str(value)) for value in protected_values])

    if args.resume:
        if not args.apply:
            raise SystemExit("--resume requires --apply")
        manifest = verify_complete_resume(out_dir, fingerprint)
        if manifest.get("protected_input_after") != protected_before:
            raise SystemExit("Cannot resume: protected input outputs changed after the completed run")
        print(
            json.dumps(
                {
                    "mode": "resume",
                    "status": "complete_outputs_verified",
                    "output_root": str(out_dir),
                    "execution_fingerprint": fingerprint,
                },
                indent=2,
            )
        )
        return

    analysis = config["analysis"]
    if not isinstance(analysis, Mapping):
        raise SystemExit("analysis must be a mapping")
    shared_bootstrap = args.n_bootstrap
    n_local_bootstrap = _positive_count(
        int(
            args.n_local_bootstrap
            if args.n_local_bootstrap is not None
            else shared_bootstrap
            if shared_bootstrap is not None
            else analysis["local_bootstrap_iterations"]
        ),
        "local bootstrap count",
    )
    n_residual_bootstrap = _positive_count(
        int(
            args.n_residual_bootstrap
            if args.n_residual_bootstrap is not None
            else shared_bootstrap
            if shared_bootstrap is not None
            else analysis["residual_bootstrap_iterations"]
        ),
        "residual bootstrap count",
    )
    n_random_rankings = _positive_count(
        int(
            args.n_random_rankings
            if args.n_random_rankings is not None
            else analysis["random_region_rankings"]
        ),
        "random ranking count",
    )
    n_fn_permutations = _positive_count(
        int(
            args.n_fn_permutations
            if args.n_fn_permutations is not None
            else analysis["fn_within_patient_permutations"]
        ),
        "FN permutation count",
    )
    if args.apply:
        ensure_new_output_root(out_dir)

    frames, summary = run_analysis(
        config,
        inputs,
        n_local_bootstrap=n_local_bootstrap,
        n_residual_bootstrap=n_residual_bootstrap,
        n_random_rankings=n_random_rankings,
        n_fn_permutations=n_fn_permutations,
    )
    protected_after_compute = snapshot_paths([Path(str(value)) for value in protected_values])
    if protected_after_compute != protected_before:
        raise SystemExit("Protected legacy outputs changed while downstream analysis was computing")

    summary.update(
        {
            "status": "complete" if args.apply else "computed_not_written",
            "output_root": str(out_dir),
            "execution_fingerprint": fingerprint,
            "n_machine_tables": len(frames),
        }
    )
    if args.apply:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in frames.items():
            atomic_csv(frame, out_dir / name)
        atomic_json(summary, out_dir / "run_summary.json")
        protected_after = snapshot_paths([Path(str(value)) for value in protected_values])
        if protected_after != protected_before:
            raise SystemExit("Protected legacy outputs changed during downstream output writing")
        output_hashes: dict[str, dict[str, object]] = {}
        for path in sorted(item for item in out_dir.iterdir() if item.is_file()):
            if path.name == "run_manifest.json":
                continue
            output_hashes[path.name] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
        manifest = {
            "status": "complete",
            "run_id": RUN_ID,
            "algorithm_version": ALGORITHM_VERSION,
            "execution_fingerprint": fingerprint,
            "fingerprint_payload": fingerprint_payload,
            "command": [sys.executable, *sys.argv],
            "protected_input_before": protected_before,
            "protected_input_after": protected_after,
            "outputs": output_hashes,
        }
        atomic_json(manifest, out_dir / "run_manifest.json")

    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "status": summary["status"],
                "output_root": str(out_dir),
                "machine_tables": len(frames),
                "local_signal_to_region_rows": summary["local_signal_to_region"]["n_predictions"],
                "region_residual_rows": summary["residual_false_negative"]["n_region_prediction_rows"],
                "network_to_contact_model_fit_calls": 0,
                "execution_fingerprint": fingerprint,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
