"""Extract unlabeled region-level network features from electrode matrices."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "region_features.yaml"
FEATURE_ROOT = ROOT / "data_derived" / "region_features"
CHECK_ROOT = ROOT / "checks"
REGIONAL_ANALYSIS_ROOT = ROOT / "data_derived" / "regional_analysis"
EPSILON = 1e-6
RANK_TIE_METHOD = "average"
TOP_FRACTION_THRESHOLD = 0.75
TRIAL_FEATURES = (
    "within_mean",
    "within_q90",
    "out_mean",
    "out_q90",
    "node_strength_mean_normalized",
    "node_strength_q90_normalized",
)


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
    out_dir = _resolve(out_dir)
    aux_root = _resolve(AUX_ROOT)
    feature_root = _resolve(FEATURE_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    analysis_root = _resolve(REGIONAL_ANALYSIS_ROOT)
    if out_dir == aux_root or _is_relative_to(out_dir, aux_root):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == feature_root
        or _is_relative_to(out_dir, feature_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
        or out_dir == analysis_root
        or _is_relative_to(out_dir, analysis_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def load_config(config_path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def truthy(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "y", "1"}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_region_label(value: object) -> str:
    text = clean_text(value)
    if not text or text.lower() == "nan":
        return "nan"
    return text


def header_map(ws: object) -> dict[str, int]:
    return {
        clean_text(ws.cell(1, col).value): col
        for col in range(1, ws.max_column + 1)
        if ws.cell(1, col).value is not None
    }


def normalize_names(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip() for value in values if str(value).strip()}
    return normalized or None


def normalize_ints(values: Iterable[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(value) for value in values}


def load_kept_channel_regions(audit_workbook: Path, patient: str) -> dict[str, tuple[str, str, bool]]:
    wb = openpyxl.load_workbook(audit_workbook, read_only=True, data_only=True)
    ws = wb[f"{patient}_bipolar"]
    headers = header_map(ws)
    required = [
        "bipolar_channel",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
        "remove_as_bad_channel",
    ]
    missing = [column for column in required if column not in headers]
    if missing:
        raise SystemExit(f"{ws.title}: missing columns {missing}")
    output: dict[str, tuple[str, str, bool]] = {}
    for row_idx in range(2, ws.max_row + 1):
        channel = clean_text(ws.cell(row_idx, headers["bipolar_channel"]).value)
        if not channel:
            continue
        if truthy(ws.cell(row_idx, headers["remove_as_bad_channel"]).value):
            continue
        output[channel] = (
            normalize_region_label(ws.cell(row_idx, headers["desikan_killiany_label"]).value),
            clean_text(ws.cell(row_idx, headers["tissue_class"]).value),
            truthy(ws.cell(row_idx, headers["is_white_csf_ventricle_unknown"]).value),
        )
    wb.close()
    return output


def transform_matrix_values(matrix: np.ndarray, method: str) -> np.ndarray:
    transformed = np.array(matrix, dtype=float, copy=True)
    if method in {"wpli2_debiased", "aec_traditional"}:
        transformed = np.maximum(transformed, 0.0)
    elif method == "aec_negative_magnitude":
        transformed = np.maximum(-transformed, 0.0)
    elif method in {"imcoh", "aec_absolute"}:
        transformed = np.abs(transformed)
    else:
        raise SystemExit(f"Unsupported method for region features: {method}")
    np.fill_diagonal(transformed, np.nan)
    return transformed


def finite_values(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values)]


def q90(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.nanquantile(finite, 0.90))


def mean_or_nan(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.nanmean(finite))


def compute_region_feature_rows(
    matrix: np.ndarray,
    region_to_indices: dict[tuple[str, str, bool], list[int]],
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    n_nodes_total = matrix.shape[0]
    all_indices = np.arange(n_nodes_total)
    rows: list[dict[str, object]] = []
    for (region, tissue, is_wcu), indices in sorted(region_to_indices.items()):
        idx = np.asarray(indices, dtype=int)
        outside = np.setdiff1d(all_indices, idx, assume_unique=False)
        n_region_nodes = int(idx.size)
        if n_region_nodes >= 2:
            sub = matrix[np.ix_(idx, idx)]
            within_mask = np.triu(np.ones(sub.shape, dtype=bool), k=1)
            within_values = sub[within_mask]
            within_notes = ""
        else:
            within_values = np.asarray([], dtype=float)
            within_notes = "requires_at_least_2_nodes"
        out_values = matrix[np.ix_(idx, outside)].ravel() if n_region_nodes >= 1 else np.asarray([], dtype=float)
        n_within_edges = int(within_values.size)
        n_out_edges = int(out_values.size)
        all_region_edges = np.concatenate([within_values, out_values]) if n_within_edges or n_out_edges else np.asarray([])
        n_nan_edges = int(np.isnan(all_region_edges).sum())

        node_means = []
        for node_idx in idx:
            node_edges = np.delete(matrix[node_idx, :], node_idx)
            finite = finite_values(node_edges)
            node_means.append(float(np.nanmean(finite)) if finite.size else float("nan"))
        node_means_array = np.asarray(node_means, dtype=float)
        feature_values = {
            "within_mean": mean_or_nan(within_values),
            "within_q90": q90(within_values),
            "out_mean": mean_or_nan(out_values),
            "out_q90": q90(out_values),
            "node_strength_mean_normalized": mean_or_nan(node_means_array),
            "node_strength_q90_normalized": q90(node_means_array),
        }
        for feature_name in TRIAL_FEATURES:
            notes = within_notes if feature_name.startswith("within_") and within_notes else ""
            rows.append(
                {
                    **metadata,
                    "desikan_killiany_label": region,
                    "tissue_class": tissue,
                    "is_white_csf_ventricle_unknown": bool(is_wcu),
                    "feature_family": "trial",
                    "feature_name": feature_name,
                    "feature_value": feature_values[feature_name],
                    "n_region_nodes": n_region_nodes,
                    "n_within_edges": n_within_edges,
                    "n_out_edges": n_out_edges,
                    "n_nan_edges": n_nan_edges,
                    "qc_notes": notes,
                }
            )
    return rows


REGION_KEYS = ["desikan_killiany_label", "tissue_class", "is_white_csf_ventricle_unknown"]
PAIR_KEYS = ["patient", "trial", *REGION_KEYS, "frequency_band", "method", "feature_name"]


def build_contrast_features(trial_features: pd.DataFrame, epsilon: float = EPSILON) -> pd.DataFrame:
    if trial_features.empty:
        return pd.DataFrame()
    for column in ["n_region_nodes", "n_within_edges", "n_out_edges", "n_nan_edges"]:
        if column not in trial_features.columns:
            trial_features[column] = 0
    keep_cols = [
        "patient",
        "trial",
        *REGION_KEYS,
        "window",
        "frequency_band",
        "method",
        "feature_name",
        "feature_value",
        "n_region_nodes",
        "n_within_edges",
        "n_out_edges",
        "n_nan_edges",
    ]
    data = trial_features[keep_cols].copy()
    base = data[data["window"] == "baseline"].rename(columns={"feature_value": "baseline_value"})
    early = data[data["window"] == "early_ictal"].rename(columns={"feature_value": "early_ictal_value"})
    merged = base.drop(columns=["window"]).merge(
        early.drop(columns=["window", "n_region_nodes", "n_within_edges", "n_out_edges", "n_nan_edges"]),
        on=PAIR_KEYS,
        how="inner",
    )
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        baseline = float(row["baseline_value"]) if pd.notna(row["baseline_value"]) else float("nan")
        early_value = float(row["early_ictal_value"]) if pd.notna(row["early_ictal_value"]) else float("nan")
        common = {
            "patient": row["patient"],
            "trial": int(row["trial"]),
            "desikan_killiany_label": row["desikan_killiany_label"],
            "tissue_class": row["tissue_class"],
            "is_white_csf_ventricle_unknown": bool(row["is_white_csf_ventricle_unknown"]),
            "feature_family": "contrast",
            "window": "baseline_to_early_ictal",
            "frequency_band": row["frequency_band"],
            "method": row["method"],
            "feature_name": row["feature_name"],
            "baseline_value": baseline,
            "early_ictal_value": early_value,
            "n_region_nodes": int(row["n_region_nodes"]),
            "n_within_edges": int(row["n_within_edges"]),
            "n_out_edges": int(row["n_out_edges"]),
            "n_nan_edges": int(row["n_nan_edges"]),
            "qc_notes": "",
        }
        rows.append({**common, "contrast_type": "delta", "feature_value": early_value - baseline})
        if np.isfinite(baseline) and np.isfinite(early_value):
            if baseline < 0.0 or early_value < 0.0:
                raise SystemExit(
                    "Log-ratio requires non-negative connectivity features: "
                    f"baseline={baseline}, early_ictal={early_value}"
                )
            log_ratio = float(np.log((early_value + epsilon) / (baseline + epsilon)))
        else:
            log_ratio = float("nan")
        rows.append({**common, "contrast_type": "log_ratio", "feature_value": log_ratio})
    return pd.DataFrame.from_records(rows)


def feature_id(row: pd.Series) -> str:
    if row.get("feature_family") == "contrast":
        return (
            f"contrast__{row['frequency_band']}__{row['method']}__"
            f"{row['feature_name']}__{row['contrast_type']}"
        )
    return f"trial__{row['window']}__{row['frequency_band']}__{row['method']}__{row['feature_name']}"


def _iqr(values: pd.Series) -> float:
    finite = values.dropna()
    if finite.empty:
        return float("nan")
    return float(finite.quantile(0.75) - finite.quantile(0.25))


def _q90_series(values: pd.Series) -> float:
    finite = values.dropna()
    if finite.empty:
        return float("nan")
    return float(finite.quantile(0.90))


def _top25_fraction(frame: pd.DataFrame) -> dict[tuple[object, ...], float]:
    counts: dict[tuple[object, ...], list[bool]] = defaultdict(list)
    group_cols = ["patient", "trial", "feature_id"]
    region_cols = ["patient", *REGION_KEYS, "feature_id"]
    for _, trial_frame in frame.groupby(group_cols, dropna=False):
        valid = trial_frame[pd.notna(trial_frame["feature_value"])].copy()
        if valid.empty:
            continue
        n_valid = len(valid)
        if n_valid == 1:
            percentile = pd.Series(0.5, index=valid.index, dtype=float)
        else:
            average_rank = valid["feature_value"].rank(
                method=RANK_TIE_METHOD,
                ascending=True,
            )
            percentile = (average_rank - 1.0) / float(n_valid - 1)
        is_top = percentile >= TOP_FRACTION_THRESHOLD
        for idx, row in valid.iterrows():
            key = tuple(row[col] for col in region_cols)
            counts[key].append(bool(is_top.loc[idx]))
    return {key: float(np.mean(values)) for key, values in counts.items() if values}


def aggregate_patient_region_features(trial_features: pd.DataFrame, contrast_features: pd.DataFrame) -> pd.DataFrame:
    frames = [trial_features.copy()]
    if contrast_features is not None and not contrast_features.empty:
        frames.append(contrast_features.copy())
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame()
    combined["feature_id"] = combined.apply(feature_id, axis=1)
    index_cols = ["patient", *REGION_KEYS]
    top25 = _top25_fraction(combined)
    rows: dict[tuple[object, ...], dict[str, object]] = {}
    for (patient, region, tissue, is_wcu, fid), group in combined.groupby([*index_cols, "feature_id"], dropna=False):
        key = (patient, region, tissue, bool(is_wcu))
        row = rows.setdefault(
            key,
            {
                "patient": patient,
                "desikan_killiany_label": region,
                "tissue_class": tissue,
                "is_white_csf_ventricle_unknown": bool(is_wcu),
            },
        )
        values = group["feature_value"]
        row[f"{fid}__trial_mean"] = float(values.mean(skipna=True))
        row[f"{fid}__trial_median"] = float(values.median(skipna=True))
        row[f"{fid}__trial_iqr"] = _iqr(values)
        row[f"{fid}__trial_q90"] = _q90_series(values)
        top_key = (patient, region, tissue, bool(is_wcu), fid)
        row[f"{fid}__trial_top25_fraction_within_patient"] = top25.get(top_key, float("nan"))
        row[f"qc__{fid}__n_region_nodes_max"] = int(group["n_region_nodes"].max())
        row[f"qc__{fid}__n_nan_edges_sum"] = int(group["n_nan_edges"].sum())
    return pd.DataFrame.from_records(list(rows.values()))


def patient_feature_method(column: str) -> str | None:
    parts = str(column).split("__")
    if parts[0] == "trial" and len(parts) >= 5:
        return parts[3]
    if parts[0] == "contrast" and len(parts) >= 4:
        return parts[2]
    return None


def validate_extracted_feature_tables(
    trial_features: pd.DataFrame,
    contrast_features: pd.DataFrame,
    patient_features: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, int]:
    expected_rows = {
        "trial": config.get("expected_trial_feature_rows"),
        "contrast": config.get("expected_contrast_feature_rows"),
        "patient_region": config.get("expected_patient_region_rows"),
    }
    actual_rows = {
        "trial": len(trial_features),
        "contrast": len(contrast_features),
        "patient_region": len(patient_features),
    }
    for name, expected in expected_rows.items():
        if expected is not None and actual_rows[name] != int(expected):
            raise SystemExit(
                f"Unexpected {name} feature row count: {actual_rows[name]} != {expected}"
            )
    for name, frame in [("trial", trial_features), ("contrast", contrast_features)]:
        finite = frame["feature_value"].dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise SystemExit(f"{name} features contain Inf values")
    if patient_features.duplicated(["patient", *REGION_KEYS]).any():
        raise SystemExit("Duplicate patient-region rows in final feature matrix")

    expected_methods = {str(value) for value in config["network_methods"]}
    counts = {method: 0 for method in sorted(expected_methods)}
    unknown: list[str] = []
    for column in patient_features.columns:
        if not column.startswith(("trial__", "contrast__")):
            continue
        method = patient_feature_method(column)
        if method not in expected_methods:
            unknown.append(column)
        else:
            counts[method] += 1
        finite = patient_features[column].dropna().to_numpy(dtype=float)
        if not np.isfinite(finite).all():
            raise SystemExit(f"Patient feature contains Inf values: {column}")
    if unknown:
        raise SystemExit(f"Unknown method patient feature columns: {unknown[:5]}")
    per_band = config.get("expected_candidate_features_per_method_per_band")
    if per_band is not None:
        expected_per_method = int(per_band) * len(config["frequency_bands"])
        mismatched = {
            method: count for method, count in counts.items() if count != expected_per_method
        }
        if mismatched:
            raise SystemExit(
                f"Unexpected patient feature counts by method: {mismatched}; "
                f"expected_each={expected_per_method}"
            )
    return counts


def build_region_index(channel_names: list[str], channel_regions: dict[str, tuple[str, str, bool]]) -> dict[tuple[str, str, bool], list[int]]:
    output: dict[tuple[str, str, bool], list[int]] = defaultdict(list)
    missing = [channel for channel in channel_names if channel not in channel_regions]
    if missing:
        raise SystemExit(f"Channels missing from final bipolar audit: {missing[:5]}")
    for idx, channel in enumerate(channel_names):
        output[channel_regions[channel]].append(idx)
    return dict(output)


MANIFEST_KEY_COLUMNS = ["patient", "trial", "window", "frequency_band", "method"]
MATRIX_GROUP_COLUMNS = ["patient", "trial", "window", "frequency_band"]
MANIFEST_REQUIRED_COLUMNS = {*MANIFEST_KEY_COLUMNS, "out_file", "qc_status"}


def normalize_method_specs(methods: object) -> list[tuple[str, str]]:
    if not isinstance(methods, list) or not methods:
        raise SystemExit("Manifest methods must be a non-empty list")
    specs: list[tuple[str, str]] = []
    for value in methods:
        if isinstance(value, str):
            source_method = output_method = value
        elif isinstance(value, dict):
            source_method = clean_text(value.get("source_method"))
            output_method = clean_text(value.get("output_method"))
            if not source_method or not output_method:
                raise SystemExit(f"Invalid derived method specification: {value}")
        else:
            raise SystemExit(f"Invalid method specification: {value}")
        specs.append((source_method, output_method))
    outputs = [output for _, output in specs]
    if len(outputs) != len(set(outputs)):
        raise SystemExit(f"Duplicate output methods in manifest specification: {outputs}")
    return specs


def _read_declared_manifest(
    path: Path,
    method_specs: list[tuple[str, str]],
    strict: bool,
) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing network matrix manifest: {path}")
    manifest = pd.read_csv(path)
    missing_columns = sorted(MANIFEST_REQUIRED_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise SystemExit(f"{path}: missing required columns {missing_columns}")
    available_methods = set(manifest["method"].astype(str).unique())
    source_methods = {source for source, _ in method_specs}
    missing_methods = sorted(source_methods - available_methods)
    if missing_methods:
        raise SystemExit(f"{path}: missing configured methods {missing_methods}")
    source_rows = manifest[manifest["method"].astype(str).isin(source_methods)].copy()
    if strict:
        unexpected_methods = sorted(available_methods - source_methods)
        if unexpected_methods:
            raise SystemExit(f"{path}: undeclared methods present {unexpected_methods}")
        noncomputed = source_rows[source_rows["qc_status"].astype(str) != "computed"]
        if not noncomputed.empty:
            raise SystemExit(f"{path}: contains {len(noncomputed)} non-computed matrix rows")
    else:
        source_rows = source_rows[source_rows["qc_status"].astype(str) == "computed"].copy()
    frames: list[pd.DataFrame] = []
    for source_method, output_method in method_specs:
        selected = source_rows[source_rows["method"].astype(str) == source_method].copy()
        selected["source_method"] = source_method
        selected["method"] = output_method
        frames.append(selected)
    selected = pd.concat(frames, ignore_index=True, sort=False)
    selected["source_manifest"] = str(path)
    return selected


def validate_manifest_grid(manifest: pd.DataFrame, config: dict[str, object]) -> None:
    expected_methods = {str(value) for value in config["network_methods"]}
    duplicate_mask = manifest.duplicated(MANIFEST_KEY_COLUMNS, keep=False)
    if duplicate_mask.any():
        duplicates = manifest.loc[duplicate_mask, MANIFEST_KEY_COLUMNS].head(5).to_dict("records")
        raise SystemExit(f"Duplicate network matrix tasks: {duplicates}")
    # A signed source matrix may intentionally be reused by multiple sign-preserving
    # derived methods, such as positive AEC and negative-AEC magnitude.
    actual_methods = set(manifest["method"].astype(str).unique())
    if actual_methods != expected_methods:
        raise SystemExit(
            f"Network methods do not match config: actual={sorted(actual_methods)} "
            f"expected={sorted(expected_methods)}"
        )
    for key, group in manifest.groupby(MATRIX_GROUP_COLUMNS, sort=False):
        group_methods = set(group["method"].astype(str))
        if group_methods != expected_methods or len(group) != len(expected_methods):
            raise SystemExit(
                f"Incomplete method grid for {key}: actual={sorted(group_methods)} "
                f"expected={sorted(expected_methods)}"
            )
    expected_rows = config.get("expected_network_matrix_rows")
    if expected_rows is not None and len(manifest) != int(expected_rows):
        raise SystemExit(f"Unexpected network matrix row count: {len(manifest)} != {expected_rows}")
    missing_files = [path for path in manifest["out_file"].astype(str) if not Path(path).exists()]
    if missing_files:
        raise SystemExit(f"Missing network matrix files: {missing_files[:5]}")


def load_manifest(config: dict[str, object]) -> pd.DataFrame:
    source_specs = config.get("network_matrix_manifests")
    frames: list[pd.DataFrame] = []
    if source_specs is not None:
        if not isinstance(source_specs, list) or not source_specs:
            raise SystemExit("network_matrix_manifests must be a non-empty list")
        declared_methods: set[str] = set()
        for spec in source_specs:
            if not isinstance(spec, dict) or "path" not in spec or "methods" not in spec:
                raise SystemExit(f"Invalid network matrix manifest specification: {spec}")
            method_specs = normalize_method_specs(spec["methods"])
            output_methods = {output for _, output in method_specs}
            overlap = declared_methods & output_methods
            if overlap:
                raise SystemExit(f"Methods declared in multiple manifests: {sorted(overlap)}")
            declared_methods.update(output_methods)
            frames.append(
                _read_declared_manifest(Path(str(spec["path"])), method_specs, strict=True)
            )
    else:
        path = Path(str(config["network_matrix_manifest"]))
        method_specs = [(str(value), str(value)) for value in config["network_methods"]]
        frames.append(_read_declared_manifest(path, method_specs, strict=False))
    manifest = pd.concat(frames, ignore_index=True, sort=False)
    validate_manifest_grid(manifest, config)
    return manifest


def validate_matrix_metadata_and_channels(
    data: np.lib.npyio.NpzFile,
    record: dict[str, object],
    reference_channels: dict[tuple[object, ...], list[str]],
) -> tuple[np.ndarray, list[str]]:
    matrix = np.asarray(data["matrix"], dtype=float)
    channel_names = [str(value) for value in data["channel_names"].tolist()]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SystemExit(f"{record['out_file']}: matrix is not square: {matrix.shape}")
    if matrix.shape[0] != len(channel_names):
        raise SystemExit(
            f"{record['out_file']}: matrix/channel mismatch {matrix.shape[0]} != {len(channel_names)}"
        )
    metadata_checks = {
        "patient": str(record["patient"]),
        "trial": int(record["trial"]),
        "window": str(record["window"]),
        "frequency_band": str(record["frequency_band"]),
        "method": str(record.get("source_method", record["method"])),
    }
    for field, expected in metadata_checks.items():
        actual = data[field].item()
        if actual != expected:
            raise SystemExit(
                f"{record['out_file']}: NPZ {field}={actual!r} does not match manifest {expected!r}"
            )
    group_key = tuple(record[column] for column in MATRIX_GROUP_COLUMNS)
    reference = reference_channels.setdefault(group_key, channel_names)
    if channel_names != reference:
        raise SystemExit(
            f"Cross-method channel order mismatch for {group_key}: "
            f"method={record['method']}"
        )
    return matrix, channel_names


def filter_manifest(
    manifest: pd.DataFrame,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
    windows: Iterable[str] | None = None,
    bands: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
) -> pd.DataFrame:
    output = manifest.copy()
    filters = {
        "patient": normalize_names(patients),
        "window": normalize_names(windows),
        "frequency_band": normalize_names(bands),
        "method": normalize_names(methods),
    }
    for column, values in filters.items():
        if values is not None:
            output = output[output[column].isin(values)]
    trial_filter = normalize_ints(trials)
    if trial_filter is not None:
        output = output[output["trial"].astype(int).isin(trial_filter)]
    return output.sort_values(["patient", "trial", "window", "frequency_band", "method"]).reset_index(drop=True)


def extract_trial_features(config_path: Path = CONFIG_PATH, **filters: object) -> pd.DataFrame:
    config = load_config(config_path)
    manifest = filter_manifest(load_manifest(config), **filters)
    workbook = Path(str(config["audit_workbook"]))
    channel_regions_by_patient: dict[str, dict[str, tuple[str, str, bool]]] = {}
    reference_channels: dict[tuple[object, ...], list[str]] = {}
    rows: list[dict[str, object]] = []
    for npz_path, source_group in manifest.groupby("out_file", sort=False):
        records = source_group.to_dict("records")
        patient = str(records[0]["patient"])
        if patient not in channel_regions_by_patient:
            channel_regions_by_patient[patient] = load_kept_channel_regions(workbook, patient)
        npz_file = Path(str(npz_path))
        with np.load(npz_file, allow_pickle=False) as data:
            raw_matrix = np.asarray(data["matrix"], dtype=float)
            channel_names = [str(value) for value in data["channel_names"].tolist()]
            for record in records:
                validate_matrix_metadata_and_channels(data, record, reference_channels)
        region_index = build_region_index(channel_names, channel_regions_by_patient[patient])
        for record in records:
            matrix = transform_matrix_values(raw_matrix, str(record["method"]))
            metadata = {
                "patient": patient,
                "trial": int(record["trial"]),
                "window": record["window"],
                "frequency_band": record["frequency_band"],
                "method": record["method"],
                "source_method": record.get("source_method", record["method"]),
                "source_npz": str(npz_file),
            }
            rows.extend(compute_region_feature_rows(matrix, region_index, metadata))
    return pd.DataFrame.from_records(rows)


def write_outputs(
    trial_features: pd.DataFrame,
    contrast_features: pd.DataFrame,
    patient_features: pd.DataFrame,
    out_dir: Path,
    overwrite: bool,
) -> None:
    assert_allowed_output_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "region_trial_features_long.csv": trial_features,
        "region_trial_contrast_features_long.csv": contrast_features,
        "region_patient_features_unlabeled.csv": patient_features,
    }
    for filename in outputs:
        out_file = out_dir / filename
        if out_file.exists() and not overwrite:
            raise SystemExit(f"Output already exists; pass --overwrite: {out_file}")
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--trials", nargs="*", type=int, default=None)
    parser.add_argument("--windows", nargs="*", default=None)
    parser.add_argument("--bands", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    out_dir = Path(args.out) if args.out is not None else Path(str(config["region_features_root"]))
    filters = {
        "patients": args.patients,
        "trials": args.trials,
        "windows": args.windows,
        "bands": args.bands,
        "methods": args.methods,
    }
    trial_features = extract_trial_features(args.config, **filters)
    contrast_features = build_contrast_features(trial_features)
    patient_features = aggregate_patient_region_features(trial_features, contrast_features)
    candidate_counts = validate_extracted_feature_tables(
        trial_features,
        contrast_features,
        patient_features,
        config,
    )
    print(f"trial_feature_rows={len(trial_features)}")
    print(f"contrast_feature_rows={len(contrast_features)}")
    print(f"patient_region_rows={len(patient_features)}")
    print(f"candidate_feature_counts_by_method={candidate_counts}")
    if args.dry_run:
        return
    write_outputs(trial_features, contrast_features, patient_features, out_dir, args.overwrite)
    print(f"wrote={out_dir}")


if __name__ == "__main__":
    main()
