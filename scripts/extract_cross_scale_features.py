"""Extract local and connectivity features for cross-scale comparisons."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

import extract_electrode_pair_features as electrode_features
import extract_region_network_features as region_features


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "cross_scale_models.yaml"
OUTPUT_ROOT = ROOT / "data_derived" / "cross_scale_models"
FEATURE_ROOT = OUTPUT_ROOT / "features"
CHECK_ROOT = ROOT / "checks"
NETWORK_TASK_KEYS = ["patient", "trial", "window", "frequency_band", "method"]
METRIC_TASK_KEYS = ["patient", "trial", "window", "frequency_band"]
REGION_KEYS = [
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
]
PATIENT_REGION_KEYS = ["patient", *REGION_KEYS]
NODE_FEATURES = [
    "same_region_mean",
    "same_region_q90",
    "other_region_mean",
    "other_region_q90",
    "global_connection_mean_normalized",
    "global_connection_q90",
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


def normalize_region(value: object) -> str:
    return region_features.normalize_region_label(value)


def _region_tuple(row: pd.Series | dict[str, object]) -> tuple[str, str, bool]:
    return (
        normalize_region(row["desikan_killiany_label"]),
        str(row["tissue_class"]).strip(),
        bool(row["is_white_csf_ventricle_unknown"]),
    )


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_q90(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.quantile(finite, 0.90)) if finite.size else float("nan")


def _aggregate(values: pd.Series, operation: str) -> float:
    finite = values[np.isfinite(values.to_numpy(dtype=float))].to_numpy(dtype=float)
    if finite.size == 0:
        return float("nan")
    if operation == "trial_mean":
        return float(np.mean(finite))
    if operation == "trial_median":
        return float(np.median(finite))
    if operation == "trial_q90":
        return float(np.quantile(finite, 0.90))
    if operation == "trial_iqr":
        if finite.size < 2:
            return float("nan")
        return float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
    raise ValueError(operation)


def load_channel_audit(config: dict[str, object]) -> pd.DataFrame:
    audit = electrode_features.load_nonclinical_channel_audit(
        Path(str(config["audit_workbook"]))
    )
    audit["desikan_killiany_label"] = audit["desikan_killiany_label"].map(
        normalize_region
    )
    return audit


def validate_network_manifest(config: dict[str, object]) -> pd.DataFrame:
    path = Path(str(config["network_matrix_manifest"]))
    frame = pd.read_csv(path)
    required = set(NETWORK_TASK_KEYS + ["out_file", "qc_status", "n_channels"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    expected_rows = int(config["expected"]["network_matrix_rows"])
    if len(frame) != expected_rows:
        raise SystemExit(f"{path}: expected {expected_rows} rows, got {len(frame)}")
    if set(frame["qc_status"].astype(str)) != {"computed"}:
        raise SystemExit(f"{path}: not every network matrix is computed")
    if frame.duplicated(NETWORK_TASK_KEYS).any():
        raise SystemExit(f"{path}: duplicate network matrix tasks")
    expected_methods = set(config["network_to_electrode"]["methods"])
    expected_bands = set(config["frequency_bands"])
    expected_windows = set(config["windows"])
    if set(frame["method"].astype(str)) != expected_methods:
        raise SystemExit(f"{path}: unexpected network methods")
    if set(frame["frequency_band"].astype(str)) != expected_bands:
        raise SystemExit(f"{path}: unexpected network frequency bands")
    if set(frame["window"].astype(str)) != expected_windows:
        raise SystemExit(f"{path}: unexpected network windows")
    for key, group in frame.groupby(
        ["patient", "trial", "window", "frequency_band"], sort=False
    ):
        if set(group["method"].astype(str)) != expected_methods:
            raise SystemExit(f"Incomplete network methods for task {key}")
    missing_files = [value for value in frame["out_file"] if not Path(str(value)).exists()]
    if missing_files:
        raise SystemExit(f"Missing network matrix, first={missing_files[0]}")
    return frame.sort_values(NETWORK_TASK_KEYS, kind="stable").reset_index(drop=True)


def _load_metric_manifest(
    path: Path,
    expected_method: str,
    expected_all_rows: int,
    bands: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(METRIC_TASK_KEYS + ["out_file", "qc_status", "method"])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if len(frame) != expected_all_rows:
        raise SystemExit(f"{path}: expected {expected_all_rows} rows, got {len(frame)}")
    if set(frame["qc_status"].astype(str)) != {"computed"}:
        raise SystemExit(f"{path}: not every metric matrix is computed")
    if set(frame["method"].astype(str)) != {expected_method}:
        raise SystemExit(f"{path}: unexpected metric method")
    if frame.duplicated(METRIC_TASK_KEYS).any():
        raise SystemExit(f"{path}: duplicate metric tasks")
    selected = frame[frame["frequency_band"].astype(str).isin(bands)].copy()
    missing_files = [
        value for value in selected["out_file"] if not Path(str(value)).exists()
    ]
    if missing_files:
        raise SystemExit(f"Missing metric matrix, first={missing_files[0]}")
    return selected


def build_metric_task_table(config: dict[str, object]) -> pd.DataFrame:
    bands = {str(value) for value in config["frequency_bands"]}
    expected_all = int(
        config["expected"]["metric_matrix_rows_per_method_all_bands"]
    )
    expected_selected = int(
        config["expected"]["metric_matrix_rows_per_method_selected_bands"]
    )
    line = _load_metric_manifest(
        Path(str(config["line_length_manifest"])),
        "line_length_rate",
        expected_all,
        bands,
    )
    power = _load_metric_manifest(
        Path(str(config["band_power_manifest"])),
        "absolute_band_power_multitaper",
        expected_all,
        bands,
    )
    if len(line) != expected_selected or len(power) != expected_selected:
        raise SystemExit(
            "Unexpected selected metric task counts: "
            f"line={len(line)} power={len(power)} expected={expected_selected}"
        )
    line = line[METRIC_TASK_KEYS + ["out_file"]].rename(
        columns={"out_file": "line_length_file"}
    )
    power = power[METRIC_TASK_KEYS + ["out_file"]].rename(
        columns={"out_file": "band_power_file"}
    )
    merged = line.merge(
        power,
        on=METRIC_TASK_KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if set(merged["_merge"].astype(str)) != {"both"}:
        raise SystemExit("Line-length and band-power selected tasks do not align")
    return (
        merged.drop(columns="_merge")
        .sort_values(METRIC_TASK_KEYS, kind="stable")
        .reset_index(drop=True)
    )


def read_network_npz(path: Path) -> tuple[np.ndarray, list[str], dict[str, object]]:
    with np.load(path, allow_pickle=False) as saved:
        required = [
            "matrix",
            "channel_names",
            "patient",
            "trial",
            "window",
            "frequency_band",
            "method",
        ]
        missing = [field for field in required if field not in saved.files]
        if missing:
            raise SystemExit(f"{path}: missing NPZ fields {missing}")
        matrix = np.asarray(saved["matrix"], dtype=np.float64)
        channels = saved["channel_names"].astype(str).tolist()
        metadata = {
            field: np.asarray(saved[field]).item()
            for field in required
            if field not in {"matrix", "channel_names"}
        }
    if matrix.shape != (len(channels), len(channels)):
        raise SystemExit(f"{path}: network matrix/channel shape mismatch")
    return matrix, channels, metadata


def compute_node_feature_arrays(
    matrix: np.ndarray,
    channel_regions: list[tuple[str, str, bool]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrix = np.asarray(matrix, dtype=np.float64)
    n_nodes = len(channel_regions)
    if matrix.shape != (n_nodes, n_nodes):
        raise SystemExit("Network matrix and region metadata have incompatible shapes")
    outputs = {
        feature: np.full(n_nodes, np.nan, dtype=np.float64)
        for feature in NODE_FEATURES
    }
    same_counts = np.zeros(n_nodes, dtype=int)
    other_counts = np.zeros(n_nodes, dtype=int)
    global_counts = np.zeros(n_nodes, dtype=int)
    indices = np.arange(n_nodes)
    for node in range(n_nodes):
        nonself = indices != node
        same = np.asarray(
            [
                nonself[index] and channel_regions[index] == channel_regions[node]
                for index in indices
            ],
            dtype=bool,
        )
        other = nonself & ~same
        same_values = matrix[node, same]
        other_values = matrix[node, other]
        global_values = matrix[node, nonself]
        same_counts[node] = int(np.isfinite(same_values).sum())
        other_counts[node] = int(np.isfinite(other_values).sum())
        global_counts[node] = int(np.isfinite(global_values).sum())
        outputs["same_region_mean"][node] = _finite_mean(same_values)
        outputs["same_region_q90"][node] = _finite_q90(same_values)
        outputs["other_region_mean"][node] = _finite_mean(other_values)
        outputs["other_region_q90"][node] = _finite_q90(other_values)
        outputs["global_connection_mean_normalized"][node] = _finite_mean(
            global_values
        )
        outputs["global_connection_q90"][node] = _finite_q90(global_values)
    qc = {
        "same_region_finite_edges": same_counts,
        "other_region_finite_edges": other_counts,
        "global_finite_edges": global_counts,
    }
    return outputs, qc


def _validate_network_npz_metadata(
    path: Path,
    metadata: dict[str, object],
    record: pd.Series,
    window: str,
) -> None:
    for field in ["patient", "trial", "window", "frequency_band", "method"]:
        expected = window if field == "window" else record[field]
        actual = metadata[field]
        if str(actual) != str(expected):
            raise SystemExit(
                f"{path}: NPZ {field}={actual!r} does not match manifest {expected!r}"
            )


def extract_network_to_electrode_trial_features(
    manifest: pd.DataFrame,
    audit: pd.DataFrame,
    config: dict[str, object],
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    patient_filter = None if patients is None else {str(value).lower() for value in patients}
    trial_filter = None if trials is None else {int(value) for value in trials}
    selected = manifest.copy()
    if patient_filter is not None:
        selected = selected[selected["patient"].isin(patient_filter)]
    if trial_filter is not None:
        selected = selected[selected["trial"].isin(trial_filter)]
    epsilon = float(config["network_to_electrode"]["epsilon"])
    top_fraction = float(config["network_to_electrode"]["rank_top_fraction"])
    rows: list[dict[str, object]] = []
    group_keys = ["patient", "trial", "frequency_band", "method"]
    for (patient, trial, band, method), group in selected.groupby(
        group_keys, sort=True
    ):
        if set(group["window"].astype(str)) != {"baseline", "early_ictal"}:
            raise SystemExit(
                f"Incomplete network windows for {patient} trial{trial:02d} {band} {method}"
            )
        patient_audit = audit[
            (audit["patient"] == patient) & (~audit["remove_as_bad_channel"])
        ].copy()
        expected_channels = patient_audit["bipolar_channel"].astype(str).tolist()
        channel_regions = [
            _region_tuple(record) for record in patient_audit.to_dict("records")
        ]
        primary_channels = patient_audit.loc[
            ~patient_audit["is_white_csf_ventricle_unknown"], "bipolar_channel"
        ].astype(str).tolist()
        primary_indices = np.asarray(
            [expected_channels.index(channel) for channel in primary_channels],
            dtype=int,
        )
        indexed = group.set_index("window")
        feature_by_window: dict[str, dict[str, np.ndarray]] = {}
        qc_by_window: dict[str, dict[str, np.ndarray]] = {}
        for window in ["baseline", "early_ictal"]:
            record = indexed.loc[window]
            path = Path(str(record["out_file"]))
            raw_matrix, channels, metadata = read_network_npz(path)
            _validate_network_npz_metadata(path, metadata, record, window)
            if channels != expected_channels:
                raise SystemExit(
                    f"{path}: channel order differs from the final non-bad audit order"
                )
            transformed = region_features.transform_matrix_values(
                raw_matrix, str(method)
            )
            if not np.allclose(
                transformed,
                transformed.T,
                equal_nan=True,
                rtol=1e-10,
                atol=1e-12,
            ):
                raise SystemExit(
                    f"{path}: transformed connectivity matrix is not symmetric"
                )
            feature_by_window[window], qc_by_window[window] = (
                compute_node_feature_arrays(transformed, channel_regions)
            )
        feature_arrays: dict[str, np.ndarray] = {}
        for feature in NODE_FEATURES:
            baseline = feature_by_window["baseline"][feature][primary_indices]
            early = feature_by_window["early_ictal"][feature][primary_indices]
            feature_arrays[f"baseline__{feature}"] = baseline
            feature_arrays[f"early_ictal__{feature}"] = early
            feature_arrays[f"contrast__{feature}__delta"] = early - baseline
            feature_arrays[f"contrast__{feature}__log_ratio"] = np.log(
                (early + epsilon) / (baseline + epsilon)
            )
        rank_top_arrays: dict[str, np.ndarray] = {}
        for feature_name, values in feature_arrays.items():
            rank = electrode_features.percentile_rank(values)
            rank_top_arrays[f"rank_top25__{feature_name}"] = np.where(
                np.isfinite(rank),
                (rank >= 1.0 - top_fraction).astype(float),
                np.nan,
            )
        for index, channel in enumerate(primary_channels):
            row: dict[str, object] = {
                "patient": str(patient),
                "trial": int(trial),
                "bipolar_channel": channel,
                "frequency_band": str(band),
                "method": str(method),
                "qc__baseline_same_region_finite_edges": int(
                    qc_by_window["baseline"]["same_region_finite_edges"][
                        primary_indices[index]
                    ]
                ),
                "qc__early_same_region_finite_edges": int(
                    qc_by_window["early_ictal"]["same_region_finite_edges"][
                        primary_indices[index]
                    ]
                ),
                "qc__baseline_global_finite_edges": int(
                    qc_by_window["baseline"]["global_finite_edges"][
                        primary_indices[index]
                    ]
                ),
                "qc__early_global_finite_edges": int(
                    qc_by_window["early_ictal"]["global_finite_edges"][
                        primary_indices[index]
                    ]
                ),
            }
            row.update(
                {name: float(values[index]) for name, values in feature_arrays.items()}
            )
            row.update(
                {name: float(values[index]) for name, values in rank_top_arrays.items()}
            )
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def aggregate_network_to_electrode_patient_features(
    trial_features: pd.DataFrame,
    audit: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    identity = {"patient", "trial", "bipolar_channel", "frequency_band", "method"}
    base_features = sorted(
        column
        for column in trial_features.columns
        if column not in identity
        and not column.startswith("qc__")
        and not column.startswith("rank_top25__")
    )
    top_features = sorted(
        column for column in trial_features.columns if column.startswith("rank_top25__")
    )
    operations = [
        str(value)
        for value in config["network_to_electrode"]["patient_aggregations"]
    ]
    records: dict[tuple[str, str], dict[str, object]] = {}
    dictionary: list[dict[str, object]] = []
    for (patient, channel, band, method), group in trial_features.groupby(
        ["patient", "bipolar_channel", "frequency_band", "method"], sort=True
    ):
        key = (str(patient), str(channel))
        row = records.setdefault(
            key, {"patient": str(patient), "bipolar_channel": str(channel)}
        )
        prefix = f"feature__{band}__{method}"
        for feature in base_features:
            for operation in operations:
                column = f"{prefix}__{feature}__{operation}"
                row[column] = _aggregate(group[feature], operation)
                dictionary.append(
                    {
                        "feature": column,
                        "frequency_band": band,
                        "method": method,
                        "trial_feature": feature,
                        "patient_aggregation": operation,
                    }
                )
        for feature in top_features:
            column = f"{prefix}__{feature}__trial_top25_fraction"
            row[column] = _aggregate(group[feature], "trial_mean")
            dictionary.append(
                {
                    "feature": column,
                    "frequency_band": band,
                    "method": method,
                    "trial_feature": feature,
                    "patient_aggregation": "trial_top25_fraction",
                }
            )
        row[f"qc__{band}__{method}__n_trials"] = int(group["trial"].nunique())
    output = pd.DataFrame.from_records(list(records.values()))
    metadata = audit[
        (~audit["remove_as_bad_channel"])
        & (~audit["is_white_csf_ventricle_unknown"])
    ][
        [
            "patient",
            "bipolar_channel",
            "desikan_killiany_label",
            "tissue_class",
            "is_white_csf_ventricle_unknown",
        ]
    ].copy()
    output = metadata.merge(
        output,
        on=["patient", "bipolar_channel"],
        how="inner",
        validate="one_to_one",
    )
    dictionary_frame = pd.DataFrame.from_records(dictionary).drop_duplicates("feature")
    return output, dictionary_frame


def aggregate_contacts_by_region(
    values: np.ndarray,
    region_to_indices: dict[tuple[str, str, bool], list[int]],
    aggregation: str,
) -> tuple[list[tuple[str, str, bool]], np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise SystemExit("Electrode metric matrix must be channel by subwindow")
    keys = sorted(region_to_indices)
    rows: list[np.ndarray] = []
    for key in keys:
        indices = np.asarray(region_to_indices[key], dtype=int)
        subset = values[indices]
        if aggregation == "contact_mean":
            rows.append(np.mean(subset, axis=0))
        elif aggregation == "contact_q90":
            rows.append(np.quantile(subset, 0.90, axis=0))
        else:
            raise SystemExit(f"Unsupported contact aggregation: {aggregation}")
    return keys, np.stack(rows, axis=0)


def read_metric_npz(
    path: Path,
    expected: dict[str, object],
) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        required = [
            "matrix",
            "channel_names",
            "subwindow_start_sec_from_onset",
            "subwindow_end_sec_from_onset",
            "patient",
            "trial",
            "window",
            "frequency_band",
            "method",
        ]
        missing = [field for field in required if field not in saved.files]
        if missing:
            raise SystemExit(f"{path}: missing metric NPZ fields {missing}")
        output = {
            field: np.array(saved[field], copy=True)
            for field in required
            if field not in {"patient", "trial", "window", "frequency_band", "method"}
        }
        for field in ["patient", "trial", "window", "frequency_band", "method"]:
            actual = np.asarray(saved[field]).item()
            if str(actual) != str(expected[field]):
                raise SystemExit(
                    f"{path}: NPZ {field}={actual!r} does not match "
                    f"expected {expected[field]!r}"
                )
        if "qc_flag_mask" in saved.files:
            output["qc_flag_mask"] = np.array(saved["qc_flag_mask"], copy=True)
    matrix = np.asarray(output["matrix"], dtype=np.float64)
    channels = output["channel_names"].astype(str).tolist()
    if matrix.ndim != 2 or matrix.shape[0] != len(channels):
        raise SystemExit(f"{path}: metric matrix/channel shape mismatch")
    if not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise SystemExit(f"{path}: metric matrix must be finite and strictly positive")
    return output


def _region_method_trial_rows(
    patient: str,
    trial: int,
    band: str,
    source_method: str,
    baseline_file: Path,
    early_file: Path,
    expected_channels: list[str],
    region_to_indices: dict[tuple[str, str, bool], list[int]],
    contact_aggregation: str,
    rank_top_fraction: float,
) -> list[dict[str, object]]:
    baseline = read_metric_npz(
        baseline_file,
        {
            "patient": patient,
            "trial": trial,
            "window": "baseline",
            "frequency_band": band,
            "method": source_method,
        },
    )
    early = read_metric_npz(
        early_file,
        {
            "patient": patient,
            "trial": trial,
            "window": "early_ictal",
            "frequency_band": band,
            "method": source_method,
        },
    )
    baseline_names = baseline["channel_names"].astype(str).tolist()
    early_names = early["channel_names"].astype(str).tolist()
    if baseline_names != expected_channels or early_names != expected_channels:
        raise SystemExit(
            f"Channel order mismatch for {patient} trial{trial:02d} {band} {source_method}"
        )
    region_keys, baseline_values = aggregate_contacts_by_region(
        baseline["matrix"], region_to_indices, contact_aggregation
    )
    early_keys, early_values = aggregate_contacts_by_region(
        early["matrix"], region_to_indices, contact_aggregation
    )
    if early_keys != region_keys:
        raise SystemExit("Baseline and early region keys differ")
    baseline_log = electrode_features.strict_log10(baseline_values)
    early_log = electrode_features.strict_log10(early_values)
    baseline_centers = (
        baseline["subwindow_start_sec_from_onset"]
        + baseline["subwindow_end_sec_from_onset"]
    ) / 2.0
    early_centers = (
        early["subwindow_start_sec_from_onset"]
        + early["subwindow_end_sec_from_onset"]
    ) / 2.0
    baseline_features = electrode_features.compute_window_features(
        baseline_log, baseline_centers
    )
    early_features = electrode_features.compute_window_features(
        early_log, early_centers
    )
    contrasts = electrode_features.compute_contrast_features(
        baseline_log,
        early_log,
        baseline_features,
        early_features,
        early_centers,
    )
    feature_arrays: dict[str, np.ndarray] = {}
    feature_arrays.update(
        {f"baseline__{key}": value for key, value in baseline_features.items()}
    )
    feature_arrays.update(
        {f"early__{key}": value for key, value in early_features.items()}
    )
    feature_arrays.update(
        {f"contrast__{key}": value for key, value in contrasts.items()}
    )
    for source in electrode_features.RANK_SOURCES:
        rank = electrode_features.percentile_rank(feature_arrays[source])
        feature_arrays[f"rank_percentile__{source}"] = rank
        feature_arrays[f"rank_top25__{source}"] = np.where(
            np.isfinite(rank),
            (rank >= 1.0 - rank_top_fraction).astype(float),
            np.nan,
        )
    method_variant = f"{source_method}_{contact_aggregation}"
    rows: list[dict[str, object]] = []
    for index, (region, tissue, is_wcu) in enumerate(region_keys):
        region_indices = np.asarray(region_to_indices[region_keys[index]], dtype=int)
        baseline_qc = baseline.get("qc_flag_mask")
        early_qc = early.get("qc_flag_mask")
        row: dict[str, object] = {
            "patient": patient,
            "trial": int(trial),
            "desikan_killiany_label": region,
            "tissue_class": tissue,
            "is_white_csf_ventricle_unknown": bool(is_wcu),
            "frequency_band": band,
            "method": method_variant,
            "source_method": source_method,
            "contact_aggregation": contact_aggregation,
            "qc__n_region_contacts": len(region_indices),
            "qc__baseline_n_subwindows": baseline_log.shape[1],
            "qc__early_n_subwindows": early_log.shape[1],
            "qc__baseline_artifact_flag_fraction": (
                float(np.mean(baseline_qc[region_indices] != 0))
                if baseline_qc is not None
                else np.nan
            ),
            "qc__early_artifact_flag_fraction": (
                float(np.mean(early_qc[region_indices] != 0))
                if early_qc is not None
                else np.nan
            ),
        }
        row.update(
            {name: float(values[index]) for name, values in feature_arrays.items()}
        )
        rows.append(row)
    return rows


def extract_electrode_to_region_trial_features(
    tasks: pd.DataFrame,
    audit: pd.DataFrame,
    config: dict[str, object],
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    patient_filter = None if patients is None else {str(value).lower() for value in patients}
    trial_filter = None if trials is None else {int(value) for value in trials}
    selected = tasks.copy()
    if patient_filter is not None:
        selected = selected[selected["patient"].isin(patient_filter)]
    if trial_filter is not None:
        selected = selected[selected["trial"].isin(trial_filter)]
    top_fraction = float(config["electrode_to_region"]["rank_top_fraction"])
    contact_aggregations = [
        str(value)
        for value in config["electrode_to_region"]["contact_aggregations"]
    ]
    rows: list[dict[str, object]] = []
    for (patient, trial, band), group in selected.groupby(
        ["patient", "trial", "frequency_band"], sort=True
    ):
        if set(group["window"].astype(str)) != {"baseline", "early_ictal"}:
            raise SystemExit(
                f"Incomplete metric windows for {patient} trial{trial:02d} {band}"
            )
        patient_audit = audit[
            (audit["patient"] == patient) & (~audit["remove_as_bad_channel"])
        ].copy()
        expected_channels = patient_audit["bipolar_channel"].astype(str).tolist()
        region_to_indices: dict[tuple[str, str, bool], list[int]] = {}
        for index, record in enumerate(patient_audit.to_dict("records")):
            region_to_indices.setdefault(_region_tuple(record), []).append(index)
        indexed = group.set_index("window")
        for source_method, column in [
            ("line_length_rate", "line_length_file"),
            ("absolute_band_power_multitaper", "band_power_file"),
        ]:
            for aggregation in contact_aggregations:
                rows.extend(
                    _region_method_trial_rows(
                        str(patient),
                        int(trial),
                        str(band),
                        source_method,
                        Path(str(indexed.loc["baseline", column])),
                        Path(str(indexed.loc["early_ictal", column])),
                        expected_channels,
                        region_to_indices,
                        aggregation,
                        top_fraction,
                    )
                )
    return pd.DataFrame.from_records(rows)


def aggregate_electrode_to_region_patient_features(
    trial_features: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_trial_features = [
        *(f"baseline__{name}" for name in electrode_features.WINDOW_FEATURES),
        *(f"early__{name}" for name in electrode_features.WINDOW_FEATURES),
        *(f"contrast__{name}" for name in electrode_features.CONTRAST_FEATURES),
        *(f"rank_percentile__{name}" for name in electrode_features.RANK_SOURCES),
    ]
    top_features = [
        f"rank_top25__{name}" for name in electrode_features.RANK_SOURCES
    ]
    operations = [
        str(value)
        for value in config["electrode_to_region"]["patient_aggregations"]
    ]
    records: dict[tuple[object, ...], dict[str, object]] = {}
    dictionary: list[dict[str, object]] = []
    group_columns = [
        "patient",
        *REGION_KEYS,
        "frequency_band",
        "method",
        "source_method",
        "contact_aggregation",
    ]
    for key, group in trial_features.groupby(group_columns, sort=True, dropna=False):
        patient, region, tissue, is_wcu, band, method, source_method, aggregation = key
        region_key = (str(patient), str(region), str(tissue), bool(is_wcu))
        row = records.setdefault(
            region_key,
            {
                "patient": str(patient),
                "desikan_killiany_label": str(region),
                "tissue_class": str(tissue),
                "is_white_csf_ventricle_unknown": bool(is_wcu),
            },
        )
        for feature in base_trial_features:
            if feature.startswith("baseline__"):
                window = "baseline"
                feature_tail = feature.removeprefix("baseline__")
                prefix = f"trial__{window}__{band}__{method}__{feature_tail}"
            elif feature.startswith("early__"):
                window = "early_ictal"
                feature_tail = feature.removeprefix("early__")
                prefix = f"trial__{window}__{band}__{method}__{feature_tail}"
            elif feature.startswith("contrast__"):
                feature_tail = feature.removeprefix("contrast__")
                prefix = f"contrast__{band}__{method}__{feature_tail}"
            else:
                feature_tail = feature.removeprefix("rank_percentile__")
                prefix = (
                    f"trial__rank_percentile__{band}__{method}__{feature_tail}"
                )
            for operation in operations:
                column = f"{prefix}__{operation}"
                row[column] = _aggregate(group[feature], operation)
                dictionary.append(
                    {
                        "feature": column,
                        "frequency_band": band,
                        "method": method,
                        "source_method": source_method,
                        "contact_aggregation": aggregation,
                        "trial_feature": feature,
                        "patient_aggregation": operation,
                    }
                )
        for feature in top_features:
            feature_tail = feature.removeprefix("rank_top25__")
            column = (
                f"trial__rank_top25__{band}__{method}__{feature_tail}"
                "__trial_top25_fraction"
            )
            row[column] = _aggregate(group[feature], "trial_mean")
            dictionary.append(
                {
                    "feature": column,
                    "frequency_band": band,
                    "method": method,
                    "source_method": source_method,
                    "contact_aggregation": aggregation,
                    "trial_feature": feature,
                    "patient_aggregation": "trial_top25_fraction",
                }
            )
        for feature in electrode_features.DIRECTIONAL_CONTRASTS:
            source = f"contrast__{feature}"
            column = (
                f"contrast__{band}__{method}__{feature}__trial_positive_fraction"
            )
            finite = group[source][
                np.isfinite(group[source].to_numpy(dtype=float))
            ].to_numpy(dtype=float)
            row[column] = (
                float(np.mean(finite > 0.0)) if finite.size else float("nan")
            )
            dictionary.append(
                {
                    "feature": column,
                    "frequency_band": band,
                    "method": method,
                    "source_method": source_method,
                    "contact_aggregation": aggregation,
                    "trial_feature": source,
                    "patient_aggregation": "trial_positive_fraction",
                }
            )
        row[f"qc__{band}__{method}__n_trials"] = int(group["trial"].nunique())
        row[f"qc__{band}__{method}__n_region_contacts_min"] = int(
            group["qc__n_region_contacts"].min()
        )
        row[f"qc__{band}__{method}__n_region_contacts_max"] = int(
            group["qc__n_region_contacts"].max()
        )
        for window in ["baseline", "early"]:
            source = f"qc__{window}_artifact_flag_fraction"
            finite_qc = group[source][
                np.isfinite(group[source].to_numpy(dtype=float))
            ].to_numpy(dtype=float)
            row[f"qc__{band}__{method}__{window}_artifact_fraction_mean"] = (
                float(np.mean(finite_qc)) if finite_qc.size else np.nan
            )
            row[f"qc__{band}__{method}__{window}_artifact_fraction_max"] = (
                float(np.max(finite_qc)) if finite_qc.size else np.nan
            )
    output = pd.DataFrame.from_records(list(records.values()))
    dictionary_frame = pd.DataFrame.from_records(dictionary).drop_duplicates("feature")
    return output, dictionary_frame


def _electrode_feature_band(column: str) -> str | None:
    parts = str(column).split("__")
    return parts[1] if len(parts) >= 4 and parts[0] == "feature" else None


def _electrode_feature_method(column: str) -> str | None:
    parts = str(column).split("__")
    return parts[2] if len(parts) >= 4 and parts[0] == "feature" else None


def _region_feature_band(column: str) -> str | None:
    parts = str(column).split("__")
    if parts[0] == "trial" and len(parts) >= 4:
        return parts[2]
    if parts[0] == "contrast" and len(parts) >= 4:
        return parts[1]
    return None


def _region_feature_method(column: str) -> str | None:
    parts = str(column).split("__")
    if parts[0] == "trial" and len(parts) >= 5:
        return parts[3]
    if parts[0] == "contrast" and len(parts) >= 4:
        return parts[2]
    return None


def validate_feature_outputs(
    network_patient: pd.DataFrame,
    network_dictionary: pd.DataFrame,
    region_patient: pd.DataFrame,
    region_dictionary: pd.DataFrame,
    config: dict[str, object],
    full_run: bool,
) -> dict[str, object]:
    bands = [str(value) for value in config["frequency_bands"]]
    network_methods = [
        str(value) for value in config["network_to_electrode"]["methods"]
    ]
    expected_network = int(
        config["network_to_electrode"]["expected_candidates_per_method_per_band"]
    )
    for band in bands:
        for method in network_methods:
            count = sum(
                _electrode_feature_band(column) == band
                and _electrode_feature_method(column) == method
                for column in network_dictionary["feature"].astype(str)
            )
            if count != expected_network:
                raise SystemExit(
                    f"Network-to-electrode candidate count mismatch for {band}/{method}: "
                    f"{count} != {expected_network}"
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
    for band in bands:
        for source_method in source_methods:
            for aggregation in aggregations:
                method = f"{source_method}_{aggregation}"
                count = sum(
                    _region_feature_band(column) == band
                    and _region_feature_method(column) == method
                    for column in region_dictionary["feature"].astype(str)
                )
                if count != expected_region:
                    raise SystemExit(
                        f"Electrode-to-region candidate count mismatch for "
                        f"{band}/{method}: {count} != {expected_region}"
                    )
    if full_run:
        expected_electrodes = int(config["expected"]["primary_electrode_rows"])
        if len(network_patient) != expected_electrodes:
            raise SystemExit(
                f"Expected {expected_electrodes} network-to-electrode rows, "
                f"got {len(network_patient)}"
            )
        expected_regions = int(config["expected"]["region_feature_rows"])
        if len(region_patient) != expected_regions:
            raise SystemExit(
                f"Expected {expected_regions} label-free region rows, "
                f"got {len(region_patient)}"
            )
    return {
        "network_to_electrode_patient_rows": len(network_patient),
        "network_to_electrode_candidate_columns": len(network_dictionary),
        "electrode_to_region_patient_rows": len(region_patient),
        "electrode_to_region_candidate_columns": len(region_dictionary),
    }
def _atomic_write_csv(frame: pd.DataFrame, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(
    out_dir: Path,
    network_trial: pd.DataFrame,
    network_patient: pd.DataFrame,
    network_dictionary: pd.DataFrame,
    region_trial: pd.DataFrame,
    region_patient: pd.DataFrame,
    region_dictionary: pd.DataFrame,
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    assert_allowed_output_root(out_dir)
    outputs = {
        "network_to_electrode_trial_features_long.csv": network_trial,
        "network_to_electrode_patient_features_unlabeled.csv": network_patient,
        "network_to_electrode_feature_dictionary.csv": network_dictionary,
        "electrode_to_region_trial_features_long.csv": region_trial,
        "electrode_to_region_patient_features_unlabeled.csv": region_patient,
        "electrode_to_region_feature_dictionary.csv": region_dictionary,
    }
    output_paths = {name: out_dir / name for name in outputs}
    summary_path = out_dir / "cross_level_feature_summary.json"
    existing = [
        path for path in [*output_paths.values(), summary_path] if path.exists()
    ]
    if existing and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {existing[0]}")
    for name, frame in outputs.items():
        _atomic_write_csv(frame, output_paths[name], overwrite=True)
    summary["files"] = {
        name: {
            "rows": len(frame),
            "sha256": _sha256(output_paths[name]),
        }
        for name, frame in outputs.items()
    }
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=FEATURE_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--patients", nargs="+")
    parser.add_argument("--trials", nargs="+", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_allowed_output_root(args.out)
    network_manifest = validate_network_manifest(config)
    metric_tasks = build_metric_task_table(config)
    audit = load_channel_audit(config)
    if not args.apply:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "network_matrix_rows": len(network_manifest),
                    "metric_task_rows": len(metric_tasks),
                    "frequency_bands": config["frequency_bands"],
                    "network_methods": config["network_to_electrode"]["methods"],
                    "electrode_methods": config["electrode_to_region"]["methods"],
                    "out": str(args.out),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    network_trial = extract_network_to_electrode_trial_features(
        network_manifest,
        audit,
        config,
        patients=args.patients,
        trials=args.trials,
    )
    network_patient, network_dictionary = (
        aggregate_network_to_electrode_patient_features(
            network_trial, audit, config
        )
    )
    region_trial = extract_electrode_to_region_trial_features(
        metric_tasks,
        audit,
        config,
        patients=args.patients,
        trials=args.trials,
    )
    region_patient, region_dictionary = (
        aggregate_electrode_to_region_patient_features(region_trial, config)
    )
    full_run = args.patients is None and args.trials is None
    validation = validate_feature_outputs(
        network_patient,
        network_dictionary,
        region_patient,
        region_dictionary,
        config,
        full_run=full_run,
    )
    summary = {
        "status": "features_complete",
        "config": str(args.config),
        "full_run": full_run,
        "network_trial_rows": len(network_trial),
        "electrode_to_region_trial_rows": len(region_trial),
        **validation,
    }
    write_outputs(
        args.out,
        network_trial,
        network_patient,
        network_dictionary,
        region_trial,
        region_patient,
        region_dictionary,
        summary,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
