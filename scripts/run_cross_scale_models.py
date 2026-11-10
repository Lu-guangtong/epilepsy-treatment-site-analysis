"""Fit cross-scale representations with patient-held-out evaluation."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml

import run_electrode_pair_lopo_modeling as electrode_model
import run_region_lopo_modeling as region_model


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "cross_scale_models.yaml"
OUTPUT_ROOT = ROOT / "data_derived" / "cross_scale_models"
CHECK_ROOT = ROOT / "checks"
DIRECTIONS = ["network_to_electrode", "electrode_to_region"]
ELECTRODE_KEYS = ["frequency_band", "patient", "bipolar_channel"]
REGION_KEYS = [
    "frequency_band",
    "patient",
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
]


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(out_dir: Path) -> None:
    assert_output_path(out_dir)
    resolved = _resolve(out_dir)
    output_root = _resolve(OUTPUT_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if resolved == AUX_ROOT or _is_relative_to(resolved, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {resolved}")
    if not (
        resolved == output_root
        or _is_relative_to(resolved, output_root)
        or resolved == tmp_root
        or _is_relative_to(resolved, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_permutation_fingerprint(
    config_path: Path,
    feature_paths: Iterable[Path],
    observed_summary: dict[str, object],
    parameters: dict[str, object],
) -> str:
    payload = {
        "config_sha256": _sha256(config_path),
        "implementation_sha256": {
            str(path): _sha256(path)
            for path in [
                Path(__file__).resolve(),
                REPO_ROOT / "scripts" / "run_electrode_pair_lopo_modeling.py",
                REPO_ROOT / "scripts" / "run_region_lopo_modeling.py",
            ]
        },
        "feature_sha256": {
            str(path): _sha256(path) for path in sorted(feature_paths)
        },
        "observed_files": observed_summary["files"],
        "parameters": parameters,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _preflight_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            f"Output exists; pass --overwrite or --resume: {existing[0]}"
        )


def load_feature_inputs(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_root = Path(str(config["feature_root"]))
    network_path = (
        feature_root / "network_to_electrode_patient_features_unlabeled.csv"
    )
    region_path = (
        feature_root / "electrode_to_region_patient_features_unlabeled.csv"
    )
    missing = [path for path in [network_path, region_path] if not path.exists()]
    if missing:
        raise SystemExit(
            "Cross-level features are missing; run extract_cross_scale_features.py "
            f"first. Missing={missing[0]}"
        )
    return pd.read_csv(network_path), pd.read_csv(region_path)


def feature_input_paths(config: dict[str, object]) -> tuple[Path, Path]:
    feature_root = Path(str(config["feature_root"]))
    return (
        feature_root / "network_to_electrode_patient_features_unlabeled.csv",
        feature_root / "electrode_to_region_patient_features_unlabeled.csv",
    )


def load_labels(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit = Path(str(config["audit_workbook"]))
    electrode_labels = electrode_model.load_electrode_labels(audit)
    region_labels = region_model.build_label_table_from_region_rows(
        region_model.load_region_summary_rows(audit)
    )
    return electrode_labels, region_labels


def validate_modeling_inputs(
    network_features: pd.DataFrame,
    region_features: pd.DataFrame,
    electrode_labels: pd.DataFrame,
    region_labels: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    bands = [str(value) for value in config["frequency_bands"]]
    expected_electrodes = int(config["expected"]["primary_electrode_rows"])
    expected_regions = int(config["expected"]["primary_region_rows"])
    network_data = electrode_model.join_primary_features_labels(
        network_features, electrode_labels
    )
    region_data = region_model.join_features_and_labels(
        region_features, region_labels
    )
    if len(network_data) != expected_electrodes:
        raise SystemExit(
            f"Expected {expected_electrodes} primary electrode rows, got "
            f"{len(network_data)}"
        )
    if len(region_data) != expected_regions:
        raise SystemExit(
            f"Expected {expected_regions} primary region rows, got {len(region_data)}"
        )
    network_methods = [
        str(value) for value in config["network_to_electrode"]["methods"]
    ]
    expected_network = int(
        config["network_to_electrode"]["expected_candidates_per_method_per_band"]
    )
    network_coverage = electrode_model.validate_candidate_coverage(
        network_features,
        bands,
        network_methods,
        expected_network,
    )
    source_methods = [
        str(value) for value in config["electrode_to_region"]["methods"]
    ]
    aggregations = [
        str(value)
        for value in config["electrode_to_region"]["contact_aggregations"]
    ]
    expected_region = int(
        config["electrode_to_region"][
            "expected_candidates_per_method_aggregation_per_band"
        ]
    )
    variant_methods = [
        f"{method}_{aggregation}"
        for method in source_methods
        for aggregation in aggregations
    ]
    region_coverage = region_model.validate_feature_method_coverage(
        region_features,
        bands,
        variant_methods,
        expected_per_method_per_band=expected_region,
    )
    return {
        "primary_electrode_rows": len(network_data),
        "primary_region_rows": len(region_data),
        "network_to_electrode_candidate_coverage": network_coverage,
        "electrode_to_region_candidate_coverage": region_coverage,
    }


def run_observed_models(
    network_features: pd.DataFrame,
    region_features: pd.DataFrame,
    electrode_labels: pd.DataFrame,
    region_labels: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling = config["modeling"]
    bands = [str(value) for value in config["frequency_bands"]]
    top_k = int(modeling["top_k_features"])
    c_values = [float(value) for value in modeling["c_values"]]
    non_missing = float(modeling["feature_non_missing_threshold"])
    correlation = float(
        modeling["correlation_pruning_spearman_abs_electrode_target"]
    )
    electrode_predictions, electrode_folds = electrode_model.run_lopo_by_band(
        network_features,
        electrode_labels,
        bands,
        top_k,
        c_values,
        non_missing,
        correlation,
    )
    electrode_metrics = electrode_model.metrics_by_band(electrode_predictions)
    electrode_predictions.insert(0, "direction", "network_to_electrode")
    electrode_predictions.insert(1, "target_level", "bipolar_electrode_point")
    electrode_metrics.insert(0, "direction", "network_to_electrode")
    electrode_metrics.insert(1, "target_level", "bipolar_electrode_point")
    electrode_fold_frame = electrode_folds.copy()
    electrode_fold_frame.insert(0, "direction", "network_to_electrode")
    electrode_fold_frame.insert(1, "target_level", "bipolar_electrode_point")

    region_predictions, region_folds = region_model.run_lopo_predictions_by_band(
        region_features,
        region_labels,
        bands,
        top_k_features=top_k,
        c_values=c_values,
        random_seed=int(modeling["random_seed"]),
        non_missing_threshold=non_missing,
    )
    region_metrics = region_model.compute_prediction_metrics_by_band(
        region_predictions
    )
    region_predictions.insert(0, "direction", "electrode_to_region")
    region_predictions.insert(1, "target_level", "brain_region")
    region_metrics.insert(0, "direction", "electrode_to_region")
    region_metrics.insert(1, "target_level", "brain_region")
    region_fold_frame = pd.DataFrame.from_records(region_folds)
    region_fold_frame.insert(0, "direction", "electrode_to_region")
    region_fold_frame.insert(1, "target_level", "brain_region")

    predictions = pd.concat(
        [electrode_predictions, region_predictions],
        ignore_index=True,
        sort=False,
    )
    metrics = pd.concat(
        [electrode_metrics, region_metrics],
        ignore_index=True,
        sort=False,
    )
    folds = pd.concat(
        [electrode_fold_frame, region_fold_frame],
        ignore_index=True,
        sort=False,
    )
    return predictions, metrics, folds


def _normalize_metric_row(direction: str, row: pd.Series) -> dict[str, float]:
    if direction == "network_to_electrode":
        return {
            "AUPRC": float(row["macro_AUPRC"]),
            "normalized_AUPRC": float(row["macro_normalized_AUPRC"]),
            "AUROC": float(row["macro_AUROC"]),
        }
    return {
        "AUPRC": float(row["AUPRC"]),
        "normalized_AUPRC": float(row["normalized_AUPRC"]),
        "AUROC": float(row["AUROC"]),
    }


def _validate_prediction_alignment(
    cross: pd.DataFrame,
    native: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    cross_keys = cross[keys + ["y_true"]].copy()
    native_keys = native[keys + ["y_true"]].copy()
    if cross_keys.duplicated(keys).any() or native_keys.duplicated(keys).any():
        raise SystemExit("Duplicate prediction identities during native comparison")
    aligned = cross.merge(
        native,
        on=keys,
        suffixes=("_cross", "_native"),
        how="inner",
        validate="one_to_one",
    )
    if len(aligned) != len(cross) or len(aligned) != len(native):
        raise SystemExit(
            "Cross/native predictions do not cover the same target samples: "
            f"cross={len(cross)} native={len(native)} aligned={len(aligned)}"
        )
    if not np.array_equal(
        aligned["y_true_cross"].to_numpy(dtype=int),
        aligned["y_true_native"].to_numpy(dtype=int),
    ):
        raise SystemExit("Cross/native ground-truth labels differ")
    return aligned


def load_native_outputs(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        Path(str(config["native_electrode_predictions"])),
        Path(str(config["native_electrode_metrics"])),
        Path(str(config["native_region_predictions"])),
        Path(str(config["native_region_metrics"])),
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing native formal result: {missing[0]}")
    bands = {str(value) for value in config["frequency_bands"]}
    frames = [pd.read_csv(path) for path in paths]
    return tuple(
        frame[frame["frequency_band"].astype(str).isin(bands)].copy()
        for frame in frames
    )


def build_observed_comparison(
    cross_predictions: pd.DataFrame,
    cross_metrics: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    (
        native_electrode_predictions,
        native_electrode_metrics,
        native_region_predictions,
        native_region_metrics,
    ) = load_native_outputs(config)
    cross_electrode_predictions = cross_predictions[
        cross_predictions["direction"] == "network_to_electrode"
    ].copy()
    cross_region_predictions = cross_predictions[
        cross_predictions["direction"] == "electrode_to_region"
    ].copy()
    _validate_prediction_alignment(
        cross_electrode_predictions,
        native_electrode_predictions,
        ELECTRODE_KEYS,
    )
    _validate_prediction_alignment(
        cross_region_predictions,
        native_region_predictions,
        REGION_KEYS,
    )
    native_by_direction = {
        "network_to_electrode": native_electrode_metrics.set_index(
            "frequency_band"
        ),
        "electrode_to_region": native_region_metrics.set_index("frequency_band"),
    }
    native_names = {
        "network_to_electrode": "line_length_plus_band_power_to_electrode",
        "electrode_to_region": "wpli2_plus_imcoh_to_region",
    }
    cross_names = {
        "network_to_electrode": "wpli2_plus_imcoh_to_electrode",
        "electrode_to_region": "line_length_plus_band_power_to_region",
    }
    rows: list[dict[str, object]] = []
    for row in cross_metrics.to_dict("records"):
        direction = str(row["direction"])
        band = str(row["frequency_band"])
        cross_values = _normalize_metric_row(direction, pd.Series(row))
        native_row = native_by_direction[direction].loc[band]
        native_values = _normalize_metric_row(direction, native_row)
        record: dict[str, object] = {
            "direction": direction,
            "target_level": row["target_level"],
            "frequency_band": band,
            "native_method": native_names[direction],
            "cross_method": cross_names[direction],
        }
        for metric in ["AUPRC", "normalized_AUPRC", "AUROC"]:
            record[f"native_{metric}"] = native_values[metric]
            record[f"cross_{metric}"] = cross_values[metric]
            record[f"delta_cross_minus_native_{metric}"] = (
                cross_values[metric] - native_values[metric]
            )
        if direction == "network_to_electrode":
            record["native_permutation_p_raw"] = native_row.get(
                "permutation_p_raw_macro_AUPRC", np.nan
            )
            record["native_permutation_p_fwer_maxT"] = native_row.get(
                "permutation_p_fwer_maxT_macro_AUPRC", np.nan
            )
        else:
            record["native_permutation_p_raw"] = native_row.get(
                "permutation_p_AUPRC", np.nan
            )
            record["native_permutation_p_fwer_maxT"] = np.nan
        rows.append(record)
    native_predictions = {
        "network_to_electrode": native_electrode_predictions,
        "electrode_to_region": native_region_predictions,
    }
    return pd.DataFrame.from_records(rows), native_predictions


def _bootstrap_direction_metrics(
    direction: str,
    cross: pd.DataFrame,
    native: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    bands = sorted(cross["frequency_band"].astype(str).unique())
    rows: list[dict[str, object]] = []
    for band_index, band in enumerate(bands, start=1):
        cross_band = cross[cross["frequency_band"] == band]
        native_band = native[native["frequency_band"] == band]
        keys = ELECTRODE_KEYS if direction == "network_to_electrode" else REGION_KEYS
        _validate_prediction_alignment(cross_band, native_band, keys)
        patients = sorted(cross_band["patient"].astype(str).unique())
        rng = np.random.default_rng(seed + band_index * 100000)
        for bootstrap_id in range(1, n_bootstrap + 1):
            sampled = rng.choice(patients, size=len(patients), replace=True)
            cross_samples: list[pd.DataFrame] = []
            native_samples: list[pd.DataFrame] = []
            for sample_index, patient in enumerate(sampled):
                patient_name = f"{patient}__bootstrap{sample_index}"
                cross_frame = cross_band[cross_band["patient"] == patient].copy()
                native_frame = native_band[native_band["patient"] == patient].copy()
                cross_frame["patient"] = patient_name
                native_frame["patient"] = patient_name
                cross_samples.append(cross_frame)
                native_samples.append(native_frame)
            cross_sample = pd.concat(cross_samples, ignore_index=True)
            native_sample = pd.concat(native_samples, ignore_index=True)
            if direction == "network_to_electrode":
                cross_values = electrode_model.patient_macro_metrics(cross_sample)
                native_values = electrode_model.patient_macro_metrics(native_sample)
                metric_map = {
                    "AUPRC": "macro_AUPRC",
                    "normalized_AUPRC": "macro_normalized_AUPRC",
                    "AUROC": "macro_AUROC",
                }
            else:
                cross_values = region_model.compute_prediction_metrics(cross_sample)
                native_values = region_model.compute_prediction_metrics(native_sample)
                metric_map = {
                    "AUPRC": "AUPRC",
                    "normalized_AUPRC": "normalized_AUPRC",
                    "AUROC": "AUROC",
                }
            record: dict[str, object] = {
                "direction": direction,
                "frequency_band": band,
                "bootstrap_id": bootstrap_id,
            }
            for output_name, source_name in metric_map.items():
                cross_value = float(cross_values[source_name])
                native_value = float(native_values[source_name])
                record[f"cross_{output_name}"] = cross_value
                record[f"native_{output_name}"] = native_value
                record[f"delta_cross_minus_native_{output_name}"] = (
                    cross_value - native_value
                )
            rows.append(record)
    return pd.DataFrame.from_records(rows)


def run_observed_bootstrap(
    cross_predictions: pd.DataFrame,
    native_predictions: dict[str, pd.DataFrame],
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    for offset, direction in enumerate(DIRECTIONS):
        cross = cross_predictions[
            cross_predictions["direction"] == direction
        ].copy()
        frames.append(
            _bootstrap_direction_metrics(
                direction,
                cross,
                native_predictions[direction],
                n_bootstrap,
                seed + offset * 10_000_000,
            )
        )
    bootstrap = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    ci_rows: list[dict[str, object]] = []
    for (direction, band), frame in bootstrap.groupby(
        ["direction", "frequency_band"], sort=True
    ):
        for metric in ["AUPRC", "normalized_AUPRC", "AUROC"]:
            for family in ["cross", "native", "delta_cross_minus_native"]:
                column = f"{family}_{metric}"
                values = frame[column].dropna()
                ci_rows.append(
                    {
                        "direction": direction,
                        "frequency_band": band,
                        "estimate": column,
                        "ci_low": (
                            float(values.quantile(0.025))
                            if not values.empty
                            else np.nan
                        ),
                        "ci_high": (
                            float(values.quantile(0.975))
                            if not values.empty
                            else np.nan
                        ),
                    }
                )
    return bootstrap, pd.DataFrame.from_records(ci_rows)


def _observed_paths(out_dir: Path) -> dict[str, Path]:
    observed = out_dir / "observed"
    return {
        "predictions": observed / "observed_predictions.csv",
        "metrics": observed / "observed_metrics_by_band.csv",
        "folds": observed / "observed_fold_details.csv",
        "comparison": observed / "observed_method_comparison.csv",
        "bootstrap": observed / "observed_bootstrap_results.csv",
        "bootstrap_ci": observed / "observed_bootstrap_ci.csv",
        "summary": observed / "observed_run_summary.json",
    }


def write_observed_outputs(
    out_dir: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    comparison: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    validation: dict[str, object],
    config_path: Path,
    run_parameters: dict[str, object],
    overwrite: bool,
) -> dict[str, object]:
    paths = _observed_paths(out_dir)
    _preflight_outputs(paths.values(), overwrite)
    frames = {
        "predictions": predictions,
        "metrics": metrics,
        "folds": folds,
        "comparison": comparison,
        "bootstrap": bootstrap,
        "bootstrap_ci": bootstrap_ci,
    }
    for key, frame in frames.items():
        _atomic_write_csv(frame, paths[key])
    summary: dict[str, object] = {
        "status": "observed_complete",
        "config": str(config_path),
        "run_parameters": run_parameters,
        "validation": validation,
        "files": {
            key: {
                "path": str(paths[key]),
                "rows": len(frame),
                "sha256": _sha256(paths[key]),
            }
            for key, frame in frames.items()
        },
    }
    _atomic_write_json(summary, paths["summary"])
    return summary


def validate_observed_gate(
    out_dir: Path,
    config: dict[str, object],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, object],
]:
    paths = _observed_paths(out_dir)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Observed gate is incomplete: missing {missing[0]}")
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    if summary.get("status") != "observed_complete":
        raise SystemExit("Observed run summary is not marked observed_complete")
    for key in [
        "predictions",
        "metrics",
        "folds",
        "comparison",
        "bootstrap",
        "bootstrap_ci",
    ]:
        details = summary["files"][key]
        path = paths[key]
        if _sha256(path) != details["sha256"]:
            raise SystemExit(f"Observed gate checksum mismatch: {path}")
        frame = pd.read_csv(path)
        if len(frame) != int(details["rows"]):
            raise SystemExit(f"Observed gate row-count mismatch: {path}")
    predictions = pd.read_csv(paths["predictions"])
    metrics = pd.read_csv(paths["metrics"])
    comparison = pd.read_csv(paths["comparison"])
    expected_predictions = (
        int(config["expected"]["network_to_electrode_prediction_rows"])
        + int(config["expected"]["electrode_to_region_prediction_rows"])
    )
    expected_folds = int(config["expected"]["folds_per_direction"]) * 2
    if len(predictions) != expected_predictions:
        raise SystemExit(
            f"Observed prediction count mismatch: {len(predictions)} != "
            f"{expected_predictions}"
        )
    if len(pd.read_csv(paths["folds"])) != expected_folds:
        raise SystemExit("Observed fold count mismatch")
    return predictions, metrics, comparison, summary


def _permutation_seed(base_seed: int, direction: str, permutation_id: int) -> int:
    direction_offset = 100_000_000 if direction == "electrode_to_region" else 0
    return int(base_seed + direction_offset + permutation_id)


def _run_permutation_chunk(
    direction: str,
    permutation_ids: list[int],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bands: list[str],
    observed: dict[str, float],
    seed: int,
    top_k: int,
    c_values: list[float],
    non_missing_threshold: float,
    correlation_threshold: float,
    run_fingerprint: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for permutation_id in permutation_ids:
        permutation_seed = _permutation_seed(seed, direction, permutation_id)
        rng = np.random.default_rng(permutation_seed)
        if direction == "network_to_electrode":
            permuted = electrode_model.permute_labels_within_patient(labels, rng)
            predictions, _ = electrode_model.run_lopo_by_band(
                features,
                permuted,
                bands,
                top_k,
                c_values,
                non_missing_threshold,
                correlation_threshold,
            )
            metrics = electrode_model.metrics_by_band(predictions)
            metric_values = metrics.set_index("frequency_band")[
                "macro_AUPRC"
            ].to_dict()
        elif direction == "electrode_to_region":
            permuted = region_model.permute_labels_within_patient(labels, rng)
            predictions, _ = region_model.run_lopo_predictions_by_band(
                features,
                permuted,
                bands,
                top_k_features=top_k,
                c_values=c_values,
                random_seed=permutation_seed,
                non_missing_threshold=non_missing_threshold,
            )
            metrics = region_model.compute_prediction_metrics_by_band(predictions)
            metric_values = metrics.set_index("frequency_band")["AUPRC"].to_dict()
        else:
            raise SystemExit(f"Unsupported permutation direction: {direction}")
        null_max = float(max(metric_values.values()))
        for band in bands:
            value = float(metric_values[band])
            rows.append(
                {
                    "direction": direction,
                    "run_fingerprint": run_fingerprint,
                    "permutation_id": int(permutation_id),
                    "seed": permutation_seed,
                    "frequency_band": band,
                    "AUPRC": value,
                    "max_AUPRC_across_bands": null_max,
                    "null_ge_observed": value >= float(observed[band]),
                    "max_null_ge_observed": null_max >= float(observed[band]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _validate_permutation_checkpoint(
    frame: pd.DataFrame,
    direction: str,
    permutation_ids: list[int],
    bands: list[str],
    run_fingerprint: str,
) -> None:
    required = {
        "direction",
        "run_fingerprint",
        "permutation_id",
        "seed",
        "frequency_band",
        "AUPRC",
        "max_AUPRC_across_bands",
        "null_ge_observed",
        "max_null_ge_observed",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Permutation checkpoint missing columns: {missing}")
    if set(frame["direction"].astype(str)) != {direction}:
        raise SystemExit("Permutation checkpoint direction mismatch")
    if set(frame["run_fingerprint"].astype(str)) != {run_fingerprint}:
        raise SystemExit("Permutation checkpoint run fingerprint mismatch")
    expected_pairs = {
        (int(permutation_id), band)
        for permutation_id in permutation_ids
        for band in bands
    }
    actual_pairs = {
        (int(row.permutation_id), str(row.frequency_band))
        for row in frame[["permutation_id", "frequency_band"]].itertuples(
            index=False
        )
    }
    if actual_pairs != expected_pairs or len(frame) != len(expected_pairs):
        raise SystemExit("Permutation checkpoint task identities are incomplete")
    numeric = frame[["AUPRC", "max_AUPRC_across_bands"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise SystemExit("Permutation checkpoint contains non-finite metrics")


def _run_permutation_chunk_to_checkpoint(
    direction: str,
    permutation_ids: list[int],
    checkpoint_file: Path,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bands: list[str],
    observed: dict[str, float],
    seed: int,
    top_k: int,
    c_values: list[float],
    non_missing_threshold: float,
    correlation_threshold: float,
    run_fingerprint: str,
) -> str:
    frame = _run_permutation_chunk(
        direction,
        permutation_ids,
        features,
        labels,
        bands,
        observed,
        seed,
        top_k,
        c_values,
        non_missing_threshold,
        correlation_threshold,
        run_fingerprint,
    )
    _validate_permutation_checkpoint(
        frame, direction, permutation_ids, bands, run_fingerprint
    )
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_file.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(checkpoint_file)
    return str(checkpoint_file)


def run_permutations_with_checkpoints(
    direction: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bands: list[str],
    observed_metrics: pd.DataFrame,
    n_permutations: int,
    seed: int,
    top_k: int,
    c_values: list[float],
    non_missing_threshold: float,
    correlation_threshold: float,
    n_jobs: int,
    checkpoint_dir: Path,
    checkpoint_chunk_size: int,
    resume: bool,
    run_fingerprint: str,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()
    if n_jobs < 1 or checkpoint_chunk_size < 1:
        raise SystemExit("n_jobs and checkpoint_chunk_size must be positive")
    assert_allowed_output_root(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    direction_metrics = observed_metrics[
        observed_metrics["direction"] == direction
    ].copy()
    primary_column = (
        "macro_AUPRC" if direction == "network_to_electrode" else "AUPRC"
    )
    observed = direction_metrics.set_index("frequency_band")[
        primary_column
    ].to_dict()
    permutation_ids = list(range(1, n_permutations + 1))
    chunks = [
        permutation_ids[start : start + checkpoint_chunk_size]
        for start in range(0, len(permutation_ids), checkpoint_chunk_size)
    ]
    files = {
        tuple(chunk): checkpoint_dir
        / f"permutations_{chunk[0]:04d}_{chunk[-1]:04d}.csv"
        for chunk in chunks
    }
    pending: list[list[int]] = []
    for chunk in chunks:
        path = files[tuple(chunk)]
        if not path.exists():
            pending.append(chunk)
            continue
        if not resume:
            raise SystemExit(
                f"Permutation checkpoint exists; pass --resume: {path}"
            )
        _validate_permutation_checkpoint(
            pd.read_csv(path), direction, chunk, bands, run_fingerprint
        )
    worker_args = (
        features,
        labels,
        bands,
        observed,
        seed,
        top_k,
        c_values,
        non_missing_threshold,
        correlation_threshold,
        run_fingerprint,
    )
    if pending and n_jobs == 1:
        for completed, chunk in enumerate(pending, start=1):
            _run_permutation_chunk_to_checkpoint(
                direction,
                chunk,
                files[tuple(chunk)],
                *worker_args,
            )
            print(
                f"{direction}_permutation_checkpoint="
                f"{completed}/{len(pending)}"
            )
    elif pending:
        with ProcessPoolExecutor(
            max_workers=min(n_jobs, len(pending)),
            max_tasks_per_child=1,
        ) as executor:
            futures = {
                executor.submit(
                    _run_permutation_chunk_to_checkpoint,
                    direction,
                    chunk,
                    files[tuple(chunk)],
                    *worker_args,
                ): chunk
                for chunk in pending
            }
            completed = 0
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    future.result()
                except Exception as error:
                    raise SystemExit(
                        "Permutation worker failed after completed checkpoints "
                        f"were saved; rerun with --resume. direction={direction} "
                        f"chunk={chunk[0]}-{chunk[-1]} error={error}"
                    ) from error
                completed += 1
                print(
                    f"{direction}_permutation_checkpoint="
                    f"{completed}/{len(pending)}"
                )
    frames: list[pd.DataFrame] = []
    for chunk in chunks:
        path = files[tuple(chunk)]
        if not path.exists():
            raise SystemExit(f"Missing completed checkpoint: {path}")
        frame = pd.read_csv(path)
        _validate_permutation_checkpoint(
            frame, direction, chunk, bands, run_fingerprint
        )
        frames.append(frame)
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["permutation_id", "frequency_band"], kind="stable")
        .reset_index(drop=True)
    )


def add_cross_permutation_p_values(
    comparison: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (direction, band), frame in permutation.groupby(
        ["direction", "frequency_band"], sort=True
    ):
        raw = (int(frame["null_ge_observed"].sum()) + 1) / (len(frame) + 1)
        fwer = (int(frame["max_null_ge_observed"].sum()) + 1) / (
            len(frame) + 1
        )
        if fwer + 1e-15 < raw:
            raise SystemExit("maxT-FWER p-value is smaller than raw p-value")
        rows.append(
            {
                "direction": direction,
                "frequency_band": band,
                "cross_permutation_p_raw_AUPRC": raw,
                "cross_permutation_p_fwer_maxT_AUPRC": fwer,
            }
        )
    return comparison.merge(
        pd.DataFrame.from_records(rows),
        on=["direction", "frequency_band"],
        how="left",
        validate="one_to_one",
    )


def write_final_outputs(
    out_dir: Path,
    permutation: pd.DataFrame,
    final_comparison: pd.DataFrame,
    observed_summary: dict[str, object],
    n_permutations: int,
    n_jobs: int,
    overwrite: bool,
) -> dict[str, object]:
    paths = {
        "permutation": out_dir / "permutation_results.csv",
        "comparison": out_dir
        / "final_method_comparison_with_permutation.csv",
        "summary": out_dir / "cross_level_run_summary.json",
    }
    _preflight_outputs(paths.values(), overwrite)
    _atomic_write_csv(permutation, paths["permutation"])
    _atomic_write_csv(final_comparison, paths["comparison"])
    summary: dict[str, object] = {
        "status": "complete",
        "observed_status": observed_summary["status"],
        "n_permutations_per_direction": n_permutations,
        "permutation_n_jobs": n_jobs,
        "files": {
            "permutation": {
                "path": str(paths["permutation"]),
                "rows": len(permutation),
                "sha256": _sha256(paths["permutation"]),
            },
            "comparison": {
                "path": str(paths["comparison"]),
                "rows": len(final_comparison),
                "sha256": _sha256(paths["comparison"]),
            },
        },
    }
    _atomic_write_json(summary, paths["summary"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--stage",
        choices=["observed", "permutation", "all"],
        default="all",
    )
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--checkpoint-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_allowed_output_root(args.out)
    modeling = config["modeling"]
    bands = [str(value) for value in config["frequency_bands"]]
    n_permutations = (
        int(modeling["n_permutations_default"])
        if args.n_permutations is None
        else int(args.n_permutations)
    )
    n_bootstrap = (
        int(modeling["n_bootstrap_default"])
        if args.n_bootstrap is None
        else int(args.n_bootstrap)
    )
    n_jobs = (
        int(modeling["permutation_n_jobs_default"])
        if args.n_jobs is None
        else int(args.n_jobs)
    )
    checkpoint_size = (
        int(modeling["permutation_checkpoint_size"])
        if args.checkpoint_size is None
        else int(args.checkpoint_size)
    )
    required_feature_paths = feature_input_paths(config)
    if args.dry_run and not all(path.exists() for path in required_feature_paths):
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "stage": args.stage,
                    "feature_inputs_ready": False,
                    "missing_feature_inputs": [
                        str(path)
                        for path in required_feature_paths
                        if not path.exists()
                    ],
                    "frequency_bands": bands,
                    "n_permutations": n_permutations,
                    "n_bootstrap": n_bootstrap,
                    "n_jobs": n_jobs,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    network_features, region_features = load_feature_inputs(config)
    electrode_labels, region_labels = load_labels(config)
    validation = validate_modeling_inputs(
        network_features,
        region_features,
        electrode_labels,
        region_labels,
        config,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "stage": args.stage,
                    "frequency_bands": bands,
                    "n_permutations": n_permutations,
                    "n_bootstrap": n_bootstrap,
                    "n_jobs": n_jobs,
                    "validation": validation,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.stage in {"observed", "all"}:
        observed_paths = _observed_paths(args.out)
        if args.resume and observed_paths["summary"].exists():
            _, observed_metrics, comparison, observed_summary = (
                validate_observed_gate(args.out, config)
            )
        else:
            predictions, observed_metrics, folds = run_observed_models(
                network_features,
                region_features,
                electrode_labels,
                region_labels,
                config,
            )
            comparison, native_predictions = build_observed_comparison(
                predictions, observed_metrics, config
            )
            bootstrap, bootstrap_ci = run_observed_bootstrap(
                predictions,
                native_predictions,
                n_bootstrap,
                int(modeling["random_seed"]),
            )
            observed_summary = write_observed_outputs(
                args.out,
                predictions,
                observed_metrics,
                folds,
                comparison,
                bootstrap,
                bootstrap_ci,
                validation,
                args.config,
                {
                    "n_bootstrap": n_bootstrap,
                    "top_k_features": int(modeling["top_k_features"]),
                    "c_values": [
                        float(value) for value in modeling["c_values"]
                    ],
                    "feature_non_missing_threshold": float(
                        modeling["feature_non_missing_threshold"]
                    ),
                    "electrode_target_spearman_threshold": float(
                        modeling[
                            "correlation_pruning_spearman_abs_electrode_target"
                        ]
                    ),
                    "random_seed": int(modeling["random_seed"]),
                },
                overwrite=args.overwrite or args.resume,
            )
            print(
                comparison[
                    [
                        "direction",
                        "frequency_band",
                        "native_AUPRC",
                        "cross_AUPRC",
                        "delta_cross_minus_native_AUPRC",
                    ]
                ].to_string(index=False)
            )
        if args.stage == "observed":
            return
    else:
        _, observed_metrics, comparison, observed_summary = validate_observed_gate(
            args.out, config
        )

    if n_permutations <= 0:
        print("Observed results complete; n_permutations=0, permutation skipped.")
        return
    final_paths = [
        args.out / "permutation_results.csv",
        args.out / "final_method_comparison_with_permutation.csv",
        args.out / "cross_level_run_summary.json",
    ]
    existing_final = [path for path in final_paths if path.exists()]
    if existing_final and not args.overwrite:
        raise SystemExit(
            "Final permutation output already exists; refusing to rerun the "
            f"expensive stage without --overwrite: {existing_final[0]}"
        )
    permutation_parameters = {
        "frequency_bands": bands,
        "n_permutations": n_permutations,
        "top_k_features": int(modeling["top_k_features"]),
        "c_values": [float(value) for value in modeling["c_values"]],
        "feature_non_missing_threshold": float(
            modeling["feature_non_missing_threshold"]
        ),
        "electrode_target_spearman_threshold": float(
            modeling["correlation_pruning_spearman_abs_electrode_target"]
        ),
        "random_seed": int(modeling["random_seed"]),
    }
    run_fingerprint = build_permutation_fingerprint(
        args.config,
        required_feature_paths,
        observed_summary,
        permutation_parameters,
    )
    permutation_frames: list[pd.DataFrame] = []
    common = {
        "bands": bands,
        "observed_metrics": observed_metrics,
        "n_permutations": n_permutations,
        "seed": int(modeling["random_seed"]),
        "top_k": int(modeling["top_k_features"]),
        "c_values": [float(value) for value in modeling["c_values"]],
        "non_missing_threshold": float(
            modeling["feature_non_missing_threshold"]
        ),
        "correlation_threshold": float(
            modeling["correlation_pruning_spearman_abs_electrode_target"]
        ),
        "n_jobs": n_jobs,
        "checkpoint_chunk_size": checkpoint_size,
        "resume": args.resume,
        "run_fingerprint": run_fingerprint,
    }
    permutation_frames.append(
        run_permutations_with_checkpoints(
            "network_to_electrode",
            network_features,
            electrode_labels,
            checkpoint_dir=args.out
            / ".permutation_checkpoints"
            / "network_to_electrode",
            **common,
        )
    )
    permutation_frames.append(
        run_permutations_with_checkpoints(
            "electrode_to_region",
            region_features,
            region_labels,
            checkpoint_dir=args.out
            / ".permutation_checkpoints"
            / "electrode_to_region",
            **common,
        )
    )
    permutation = pd.concat(permutation_frames, ignore_index=True)
    expected_permutation_rows = n_permutations * len(bands) * len(DIRECTIONS)
    if len(permutation) != expected_permutation_rows:
        raise SystemExit(
            f"Permutation row mismatch: {len(permutation)} != "
            f"{expected_permutation_rows}"
        )
    final_comparison = add_cross_permutation_p_values(
        comparison, permutation
    )
    final_summary = write_final_outputs(
        args.out,
        permutation,
        final_comparison,
        observed_summary,
        n_permutations,
        n_jobs,
        overwrite=args.overwrite,
    )
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
