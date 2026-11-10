"""Extract label-free bipolar-contact features from raw metric matrices."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "electrode_pair_modeling.yaml"
FEATURE_ROOT = ROOT / "data_derived" / "electrode_pair_features"
CHECK_ROOT = ROOT / "checks"
TASK_KEYS = ["patient", "trial", "window", "frequency_band"]
PATIENTS = private_value("cohort", "patients")
WINDOW_FEATURES = [
    "log_mean",
    "log_median",
    "log_q75",
    "log_q90",
    "log_iqr",
    "log_mad",
    "log_peak",
    "log_peak_prominence",
    "log_theil_sen_slope_per_sec",
]
CONTRAST_FEATURES = [
    "median_log10_change",
    "q90_log10_change",
    "peak_log10_change",
    "robust_median_shift",
    "early_exceed_baseline_q95_fraction",
    "early_positive_excess_over_baseline_q95_mean",
    "log_iqr_change",
    "log_mad_change",
    "log_slope_change",
    "early_peak_latency_fraction",
]
RANK_SOURCES = [
    "early__log_median",
    "early__log_q90",
    "contrast__median_log10_change",
    "contrast__robust_median_shift",
    "contrast__early_positive_excess_over_baseline_q95_mean",
]
DIRECTIONAL_CONTRASTS = [
    "median_log10_change",
    "q90_log10_change",
    "peak_log10_change",
    "robust_median_shift",
    "log_iqr_change",
    "log_mad_change",
    "log_slope_change",
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
    out_dir = _resolve(out_dir)
    feature_root = _resolve(FEATURE_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == feature_root
        or _is_relative_to(out_dir, feature_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def truthy(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "y", "1"}


def clean_text(value: object) -> str:
    return "" if value is None else str(value).strip()


def header_map(ws: object) -> dict[str, int]:
    return {
        clean_text(ws.cell(1, column).value): column
        for column in range(1, ws.max_column + 1)
        if ws.cell(1, column).value is not None
    }


def load_nonclinical_channel_audit(audit_workbook: Path) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(audit_workbook, read_only=True, data_only=True)
    rows: list[dict[str, object]] = []
    required = [
        "patient_id",
        "bipolar_channel",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
        "remove_as_bad_channel",
    ]
    for patient in PATIENTS:
        sheet_name = f"{patient}_bipolar"
        if sheet_name not in workbook.sheetnames:
            raise SystemExit(f"Missing audit worksheet: {sheet_name}")
        ws = workbook[sheet_name]
        headers = header_map(ws)
        missing = [column for column in required if column not in headers]
        if missing:
            raise SystemExit(f"{sheet_name}: missing columns {missing}")
        for row_idx in range(2, ws.max_row + 1):
            channel = clean_text(ws.cell(row_idx, headers["bipolar_channel"]).value)
            if not channel:
                continue
            row_patient = clean_text(ws.cell(row_idx, headers["patient_id"]).value).lower()
            if row_patient != patient:
                raise SystemExit(f"{sheet_name} row {row_idx}: patient mismatch {row_patient}")
            rows.append(
                {
                    "patient": patient,
                    "bipolar_channel": channel,
                    "desikan_killiany_label": clean_text(
                        ws.cell(row_idx, headers["desikan_killiany_label"]).value
                    ),
                    "tissue_class": clean_text(ws.cell(row_idx, headers["tissue_class"]).value),
                    "is_white_csf_ventricle_unknown": truthy(
                        ws.cell(row_idx, headers["is_white_csf_ventricle_unknown"]).value
                    ),
                    "remove_as_bad_channel": truthy(
                        ws.cell(row_idx, headers["remove_as_bad_channel"]).value
                    ),
                }
            )
    workbook.close()
    output = pd.DataFrame.from_records(rows)
    if output.duplicated(["patient", "bipolar_channel"]).any():
        raise SystemExit("Duplicate patient/bipolar_channel rows in audit workbook")
    return output


def load_metric_manifest(path: Path, expected_method: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = set(TASK_KEYS + ["out_file", "qc_status", "method"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing manifest columns {missing}")
    if set(frame["qc_status"]) != {"computed"}:
        raise SystemExit(f"{path}: not all matrix tasks are computed")
    if set(frame["method"]) != {expected_method}:
        raise SystemExit(f"{path}: unexpected methods {sorted(frame['method'].unique())}")
    if frame.duplicated(TASK_KEYS).any():
        raise SystemExit(f"{path}: duplicate matrix tasks")
    missing_files = [value for value in frame["out_file"] if not Path(str(value)).exists()]
    if missing_files:
        raise SystemExit(f"{path}: missing matrix files, first={missing_files[0]}")
    return frame


def build_aligned_task_table(config: dict[str, object]) -> pd.DataFrame:
    line = load_metric_manifest(Path(str(config["line_length_manifest"])), "line_length_rate")
    power = load_metric_manifest(
        Path(str(config["band_power_manifest"])), "absolute_band_power_multitaper"
    )
    expected = int(config["expected"]["matrix_tasks_per_method"])
    if len(line) != expected or len(power) != expected:
        raise SystemExit(f"Unexpected matrix task counts: line={len(line)} power={len(power)}")
    line = line[TASK_KEYS + ["out_file"]].rename(columns={"out_file": "line_length_file"})
    power = power[TASK_KEYS + ["out_file"]].rename(columns={"out_file": "band_power_file"})
    merged = line.merge(power, on=TASK_KEYS, how="outer", validate="one_to_one", indicator=True)
    if set(merged["_merge"]) != {"both"}:
        raise SystemExit("Line-length and band-power task sets do not match")
    return merged.drop(columns="_merge").sort_values(TASK_KEYS, kind="stable").reset_index(drop=True)


def strict_log10(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix <= 0.0):
        raise SystemExit("Metric matrix must be finite, two-dimensional, and strictly positive")
    return np.log10(matrix)


def theil_sen_slopes(values: np.ndarray, times_sec: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    times = np.asarray(times_sec, dtype=np.float64)
    if values.ndim != 2 or times.ndim != 1 or values.shape[1] != times.size or times.size < 2:
        raise SystemExit("Theil-Sen slope inputs have incompatible shapes")
    left, right = np.triu_indices(times.size, k=1)
    time_delta = times[right] - times[left]
    if np.any(time_delta <= 0.0):
        raise SystemExit("Subwindow times must be strictly increasing")
    pairwise = (values[:, right] - values[:, left]) / time_delta[np.newaxis, :]
    return np.median(pairwise, axis=1)


def compute_window_features(values_log10: np.ndarray, centers_sec: np.ndarray) -> dict[str, np.ndarray]:
    median = np.median(values_log10, axis=1)
    q25 = np.quantile(values_log10, 0.25, axis=1)
    q75 = np.quantile(values_log10, 0.75, axis=1)
    q90 = np.quantile(values_log10, 0.90, axis=1)
    peak = np.max(values_log10, axis=1)
    return {
        "log_mean": np.mean(values_log10, axis=1),
        "log_median": median,
        "log_q75": q75,
        "log_q90": q90,
        "log_iqr": q75 - q25,
        "log_mad": np.median(np.abs(values_log10 - median[:, np.newaxis]), axis=1),
        "log_peak": peak,
        "log_peak_prominence": peak - median,
        "log_theil_sen_slope_per_sec": theil_sen_slopes(values_log10, centers_sec),
    }


def compute_contrast_features(
    baseline_log10: np.ndarray,
    early_log10: np.ndarray,
    baseline_features: dict[str, np.ndarray],
    early_features: dict[str, np.ndarray],
    early_centers_from_onset: np.ndarray,
    early_window_duration_sec: float = 10.0,
) -> dict[str, np.ndarray]:
    baseline_q95 = np.quantile(baseline_log10, 0.95, axis=1)
    baseline_mad = baseline_features["log_mad"]
    denominator = 1.4826 * baseline_mad
    median_change = early_features["log_median"] - baseline_features["log_median"]
    robust_shift = np.divide(
        median_change,
        denominator,
        out=np.full_like(median_change, np.nan),
        where=denominator > 0.0,
    )
    excess = np.maximum(early_log10 - baseline_q95[:, np.newaxis], 0.0)
    peak_indices = np.argmax(early_log10, axis=1)
    peak_latency = early_centers_from_onset[peak_indices] / early_window_duration_sec
    return {
        "median_log10_change": median_change,
        "q90_log10_change": early_features["log_q90"] - baseline_features["log_q90"],
        "peak_log10_change": early_features["log_peak"] - baseline_features["log_peak"],
        "robust_median_shift": robust_shift,
        "early_exceed_baseline_q95_fraction": np.mean(
            early_log10 > baseline_q95[:, np.newaxis], axis=1
        ),
        "early_positive_excess_over_baseline_q95_mean": np.mean(excess, axis=1),
        "log_iqr_change": early_features["log_iqr"] - baseline_features["log_iqr"],
        "log_mad_change": early_features["log_mad"] - baseline_features["log_mad"],
        "log_slope_change": (
            early_features["log_theil_sen_slope_per_sec"]
            - baseline_features["log_theil_sen_slope_per_sec"]
        ),
        "early_peak_latency_fraction": np.clip(peak_latency, 0.0, 1.0),
    }


def percentile_rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    valid = series.notna()
    output = np.full(len(series), np.nan, dtype=np.float64)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return output
    if n_valid == 1:
        output[valid.to_numpy()] = 0.5
        return output
    ranks = series[valid].rank(method="average").to_numpy(dtype=np.float64)
    output[valid.to_numpy()] = (ranks - 1.0) / (n_valid - 1.0)
    return output


def _read_metric_file(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        required = [
            "matrix",
            "channel_names",
            "subwindow_start_sec_from_onset",
            "subwindow_end_sec_from_onset",
        ]
        missing = [key for key in required if key not in saved.files]
        if missing:
            raise SystemExit(f"{path}: missing NPZ fields {missing}")
        output = {key: np.array(saved[key], copy=True) for key in required}
        if "qc_flag_mask" in saved.files:
            output["qc_flag_mask"] = np.array(saved["qc_flag_mask"], copy=True)
    return output


def _method_trial_rows(
    patient: str,
    trial: int,
    band: str,
    method: str,
    baseline_file: Path,
    early_file: Path,
    expected_channels: list[str],
    eligible_channels: list[str],
    rank_top_fraction: float,
) -> list[dict[str, object]]:
    baseline = _read_metric_file(baseline_file)
    early = _read_metric_file(early_file)
    baseline_names = baseline["channel_names"].astype(str).tolist()
    early_names = early["channel_names"].astype(str).tolist()
    if baseline_names != expected_channels or early_names != expected_channels:
        raise SystemExit(f"Channel order mismatch for {patient} trial{trial:02d} {band} {method}")
    indices = np.asarray([expected_channels.index(channel) for channel in eligible_channels], dtype=int)
    baseline_values = strict_log10(baseline["matrix"][indices])
    early_values = strict_log10(early["matrix"][indices])
    baseline_centers = (
        baseline["subwindow_start_sec_from_onset"]
        + baseline["subwindow_end_sec_from_onset"]
    ) / 2.0
    early_centers = (
        early["subwindow_start_sec_from_onset"] + early["subwindow_end_sec_from_onset"]
    ) / 2.0
    baseline_features = compute_window_features(baseline_values, baseline_centers)
    early_features = compute_window_features(early_values, early_centers)
    contrasts = compute_contrast_features(
        baseline_values,
        early_values,
        baseline_features,
        early_features,
        early_centers,
    )
    feature_arrays: dict[str, np.ndarray] = {}
    feature_arrays.update({f"baseline__{key}": value for key, value in baseline_features.items()})
    feature_arrays.update({f"early__{key}": value for key, value in early_features.items()})
    feature_arrays.update({f"contrast__{key}": value for key, value in contrasts.items()})
    for source in RANK_SOURCES:
        rank = percentile_rank(feature_arrays[source])
        feature_arrays[f"rank_percentile__{source}"] = rank
        feature_arrays[f"rank_top25__{source}"] = np.where(
            np.isfinite(rank), (rank >= 1.0 - rank_top_fraction).astype(float), np.nan
        )

    baseline_qc = baseline.get("qc_flag_mask")
    early_qc = early.get("qc_flag_mask")
    rows: list[dict[str, object]] = []
    for idx, channel in enumerate(eligible_channels):
        row: dict[str, object] = {
            "patient": patient,
            "trial": trial,
            "bipolar_channel": channel,
            "frequency_band": band,
            "method": method,
            "qc__baseline_n_subwindows": baseline_values.shape[1],
            "qc__early_n_subwindows": early_values.shape[1],
            "qc__baseline_artifact_flag_fraction": (
                float(np.mean(baseline_qc[indices[idx]] != 0)) if baseline_qc is not None else np.nan
            ),
            "qc__early_artifact_flag_fraction": (
                float(np.mean(early_qc[indices[idx]] != 0)) if early_qc is not None else np.nan
            ),
        }
        row.update({key: float(values[idx]) for key, values in feature_arrays.items()})
        rows.append(row)
    return rows


def extract_trial_features(
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
    rows: list[dict[str, object]] = []
    top_fraction = float(config["features"]["rank_top_fraction"])
    for (patient, trial, band), group in selected.groupby(
        ["patient", "trial", "frequency_band"], sort=True
    ):
        if set(group["window"]) != {"baseline", "early_ictal"} or len(group) != 2:
            raise SystemExit(f"Incomplete windows for {patient} trial{trial:02d} {band}")
        patient_audit = audit[audit["patient"] == patient]
        expected_channels = patient_audit.loc[
            ~patient_audit["remove_as_bad_channel"], "bipolar_channel"
        ].tolist()
        eligible_channels = patient_audit.loc[
            (~patient_audit["remove_as_bad_channel"])
            & (~patient_audit["is_white_csf_ventricle_unknown"]),
            "bipolar_channel",
        ].tolist()
        indexed = group.set_index("window")
        for method, column in [
            ("line_length_rate", "line_length_file"),
            ("absolute_band_power_multitaper", "band_power_file"),
        ]:
            rows.extend(
                _method_trial_rows(
                    str(patient),
                    int(trial),
                    str(band),
                    method,
                    Path(str(indexed.loc["baseline", column])),
                    Path(str(indexed.loc["early_ictal", column])),
                    expected_channels,
                    eligible_channels,
                    top_fraction,
                )
            )
    return pd.DataFrame.from_records(rows)


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
        return float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if finite.size >= 2 else float("nan")
    raise ValueError(operation)


def aggregate_patient_features(
    trial_features: pd.DataFrame,
    audit: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base_trial_features = [
        *(f"baseline__{name}" for name in WINDOW_FEATURES),
        *(f"early__{name}" for name in WINDOW_FEATURES),
        *(f"contrast__{name}" for name in CONTRAST_FEATURES),
        *(f"rank_percentile__{name}" for name in RANK_SOURCES),
    ]
    top_features = [f"rank_top25__{name}" for name in RANK_SOURCES]
    operations = [str(value) for value in config["features"]["patient_aggregations"]]
    records: dict[tuple[str, str], dict[str, object]] = {}
    dictionary: dict[str, dict[str, object]] = {}
    for (patient, channel, band, method), group in trial_features.groupby(
        ["patient", "bipolar_channel", "frequency_band", "method"], sort=True
    ):
        key = (str(patient), str(channel))
        row = records.setdefault(key, {"patient": patient, "bipolar_channel": channel})
        prefix = f"feature__{band}__{method}"
        for feature in base_trial_features:
            for operation in operations:
                column = f"{prefix}__{feature}__{operation}"
                row[column] = _aggregate(group[feature], operation)
                dictionary[column] = {
                        "feature": column,
                        "frequency_band": band,
                        "method": method,
                        "trial_feature": feature,
                        "patient_aggregation": operation,
                    }
        for feature in top_features:
            column = f"{prefix}__{feature}__trial_top25_fraction"
            row[column] = float(group[feature].mean())
            dictionary[column] = {
                    "feature": column,
                    "frequency_band": band,
                    "method": method,
                    "trial_feature": feature,
                    "patient_aggregation": "trial_top25_fraction",
                }
        for feature in DIRECTIONAL_CONTRASTS:
            source = f"contrast__{feature}"
            column = f"{prefix}__{source}__trial_positive_fraction"
            finite = group[source].dropna()
            row[column] = float((finite > 0.0).mean()) if not finite.empty else float("nan")
            dictionary[column] = {
                    "feature": column,
                    "frequency_band": band,
                    "method": method,
                    "trial_feature": source,
                    "patient_aggregation": "trial_positive_fraction",
                }
    output = pd.DataFrame.from_records(list(records.values()))
    eligible_audit = audit[
        (~audit["remove_as_bad_channel"]) & (~audit["is_white_csf_ventricle_unknown"])
    ][
        [
            "patient",
            "bipolar_channel",
            "desikan_killiany_label",
            "tissue_class",
            "is_white_csf_ventricle_unknown",
        ]
    ]
    output = eligible_audit.merge(
        output, on=["patient", "bipolar_channel"], how="inner", validate="one_to_one"
    )
    n_trials = trial_features.groupby(["patient", "bipolar_channel"])["trial"].nunique()
    output["qc__n_trials"] = [n_trials.loc[(row.patient, row.bipolar_channel)] for row in output.itertuples()]
    feature_dictionary = pd.DataFrame.from_records(list(dictionary.values()))
    return output, feature_dictionary


def validate_full_outputs(
    trial_features: pd.DataFrame,
    patient_features: pd.DataFrame,
    dictionary: pd.DataFrame,
    config: dict[str, object],
) -> None:
    expected = config["expected"]
    if len(trial_features) != int(expected["trial_feature_rows"]):
        raise SystemExit(f"Unexpected trial feature rows: {len(trial_features)}")
    if len(patient_features) != int(expected["patient_feature_rows"]):
        raise SystemExit(f"Unexpected patient feature rows: {len(patient_features)}")
    counts = dictionary.groupby(["frequency_band", "method"]).size()
    expected_count = int(config["features"]["expected_candidates_per_method_per_band"])
    if not (counts == expected_count).all():
        raise SystemExit(f"Unexpected candidate feature counts: {counts.to_dict()}")


def write_outputs(
    out_dir: Path,
    trial_features: pd.DataFrame,
    patient_features: pd.DataFrame,
    dictionary: pd.DataFrame,
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    assert_allowed_output_root(out_dir)
    outputs = {
        "electrode_trial_features_long.csv": trial_features,
        "electrode_patient_features_unlabeled.csv": patient_features,
        "electrode_feature_dictionary.csv": dictionary,
    }
    for filename in [*outputs, "electrode_feature_extraction_summary.json"]:
        path = out_dir / filename
        if path.exists() and not overwrite:
            raise SystemExit(f"Output already exists; pass --overwrite: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")
    (out_dir / "electrode_feature_extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--patients", nargs="*")
    parser.add_argument("--trials", nargs="*", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_allowed_output_root(args.out)
    tasks = build_aligned_task_table(config)
    audit = load_nonclinical_channel_audit(Path(str(config["audit_workbook"])))
    kept = audit[~audit["remove_as_bad_channel"]]
    primary = kept[~kept["is_white_csf_ventricle_unknown"]]
    print(f"aligned_tasks={len(tasks)}")
    print(f"channels_after_bad_removal={len(kept)} primary_channels={len(primary)}")
    print(f"patients={primary.groupby('patient').size().to_dict()}")
    if args.dry_run:
        return
    trial_features = extract_trial_features(
        tasks,
        audit,
        config,
        patients=args.patients,
        trials=args.trials,
    )
    patient_features, dictionary = aggregate_patient_features(trial_features, audit, config)
    if args.patients is None and args.trials is None:
        validate_full_outputs(trial_features, patient_features, dictionary, config)
    summary = {
        "config": str(args.config),
        "n_aligned_matrix_tasks_per_method": len(tasks),
        "n_trial_feature_rows": len(trial_features),
        "n_patient_feature_rows": len(patient_features),
        "n_candidate_features": len(dictionary),
        "candidate_counts_by_band_method": {
            f"{band}|{method}": int(count)
            for (band, method), count in dictionary.groupby(["frequency_band", "method"]).size().items()
        },
        "labels_read_during_feature_extraction": False,
        "metric_transform": "log10_strictly_positive",
        "qc_flags_used_for_exclusion": False,
    }
    write_outputs(
        args.out,
        trial_features,
        patient_features,
        dictionary,
        summary,
        overwrite=args.overwrite,
    )
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
