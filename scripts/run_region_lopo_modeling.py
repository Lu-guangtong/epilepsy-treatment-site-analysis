"""Run patient-isolated LOPO modeling on region-level network features."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
from typing import Iterable

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import openpyxl
import pandas as pd
import psutil
import yaml
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler
from threadpoolctl import threadpool_limits


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "regional_analysis" / "region_models.yaml"
FEATURE_ROOT = ROOT / "data_derived" / "region_features"
MODELING_ROOT = ROOT / "data_derived" / "region_modeling"
MODELING_BY_BAND_ROOT = ROOT / "data_derived" / "region_modeling_by_band"
CHECK_ROOT = ROOT / "checks"
REGIONAL_ANALYSIS_ROOT = ROOT / "data_derived" / "regional_analysis"
UNIVARIATE_ORIENTATION_RULE = "inner_training_positive_vs_negative_median"
PRIMARY_REGION_METRIC = "patient_macro_AUPRC"
PRIMARY_REGION_METRIC_SCOPE = "within_patient_then_macro_across_patients"
AVERAGE_PRECISION_IMPLEMENTATION = "sklearn_equivalent_tie_grouped_v2"
AVERAGE_PRECISION_DEFINITION = "sklearn.metrics.average_precision_score"
REGION_KEYS = ["patient", "desikan_killiany_label", "tissue_class", "is_white_csf_ventricle_unknown"]
NON_FEATURE_COLUMNS = {
    "patient",
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
    "label_binary",
    "suspicious_region_status",
    "inclusion_status",
    "n_bipolar_pairs_after_bad_channel_removal",
}

_PERMUTATION_WORKER_STATE: dict[str, object] = {}


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
    modeling_root = _resolve(MODELING_ROOT)
    modeling_by_band_root = _resolve(MODELING_BY_BAND_ROOT)
    feature_root = _resolve(FEATURE_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    analysis_root = _resolve(REGIONAL_ANALYSIS_ROOT)
    if out_dir == aux_root or _is_relative_to(out_dir, aux_root):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == modeling_root
        or _is_relative_to(out_dir, modeling_root)
        or out_dir == modeling_by_band_root
        or _is_relative_to(out_dir, modeling_by_band_root)
        or out_dir == feature_root
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


def load_region_summary_rows(audit_workbook: Path) -> pd.DataFrame:
    wb = openpyxl.load_workbook(audit_workbook, read_only=True, data_only=True)
    rows: list[dict[str, object]] = []
    for sheet_name in wb.sheetnames:
        if not sheet_name.endswith("_region_summary"):
            continue
        patient = sheet_name.removesuffix("_region_summary")
        ws = wb[sheet_name]
        headers = header_map(ws)
        required = [
            "display_order",
            "region_group",
            "desikan_killiany_label",
            "tissue_class",
            "is_white_csf_ventricle_unknown",
            "suspicious_region_status",
            "n_bipolar_pairs_after_bad_channel_removal",
            "n_doctor_burned_pairs_after_bad_channel_removal",
        ]
        missing = [column for column in required if column not in headers]
        if missing:
            raise SystemExit(f"{sheet_name}: missing columns {missing}")
        for row_idx in range(2, ws.max_row + 1):
            if not any(ws.cell(row_idx, col).value is not None for col in range(1, ws.max_column + 1)):
                continue
            rows.append(
                {
                    "patient": patient,
                    "display_order": int(ws.cell(row_idx, headers["display_order"]).value),
                    "region_group": clean_text(
                        ws.cell(row_idx, headers["region_group"]).value
                    ),
                    "desikan_killiany_label": normalize_region_label(
                        ws.cell(row_idx, headers["desikan_killiany_label"]).value
                    ),
                    "tissue_class": clean_text(ws.cell(row_idx, headers["tissue_class"]).value),
                    "is_white_csf_ventricle_unknown": truthy(
                        ws.cell(row_idx, headers["is_white_csf_ventricle_unknown"]).value
                    ),
                    "suspicious_region_status": clean_text(
                        ws.cell(row_idx, headers["suspicious_region_status"]).value
                    ).lower(),
                    "n_bipolar_pairs_after_bad_channel_removal": int(
                        ws.cell(row_idx, headers["n_bipolar_pairs_after_bad_channel_removal"]).value or 0
                    ),
                    "n_doctor_burned_pairs_after_bad_channel_removal": int(
                        ws.cell(
                            row_idx,
                            headers["n_doctor_burned_pairs_after_bad_channel_removal"],
                        ).value
                        or 0
                    ),
                }
            )
    wb.close()
    return pd.DataFrame.from_records(rows)


def build_label_table_from_region_rows(region_rows: pd.DataFrame) -> pd.DataFrame:
    output = region_rows[
        [
            *REGION_KEYS,
            "suspicious_region_status",
            "n_bipolar_pairs_after_bad_channel_removal",
        ]
    ].copy()
    labels: list[float] = []
    statuses: list[str] = []
    for row in output.to_dict("records"):
        status = str(row["suspicious_region_status"]).strip().lower()
        n_nodes = int(row["n_bipolar_pairs_after_bad_channel_removal"])
        if n_nodes == 0:
            labels.append(np.nan)
            statuses.append("excluded_no_signal_coverage")
        elif status == "yes":
            labels.append(1)
            statuses.append("primary")
        elif status == "no":
            labels.append(0)
            statuses.append("primary")
        elif status == "not_applicable":
            labels.append(np.nan)
            statuses.append("excluded_not_applicable")
        else:
            raise SystemExit(f"Unexpected region status: {status}")
    output["label_binary"] = labels
    output["inclusion_status"] = statuses
    return output


def build_ablation_top4_label_table(
    region_rows: pd.DataFrame,
    rule: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked_rule_values = {
        "ranking_field": "n_doctor_burned_pairs_after_bad_channel_removal",
        "ranking_direction": "descending",
        "tie_break_field": "display_order",
        "tie_break_direction": "ascending",
        "expand_boundary_ties": False,
        "burned_rate_numerator": "n_doctor_burned_pairs_after_bad_channel_removal",
        "burned_rate_denominator": "n_bipolar_pairs_after_bad_channel_removal",
        "threshold_logic": "or",
    }
    for key, expected_value in locked_rule_values.items():
        if rule.get(key) != expected_value:
            raise SystemExit(
                f"Unsupported labels.rule value for {key}: {rule.get(key)!r} != "
                f"{expected_value!r}"
            )
    required = {
        *REGION_KEYS,
        "display_order",
        "region_group",
        "suspicious_region_status",
        "n_bipolar_pairs_after_bad_channel_removal",
        "n_doctor_burned_pairs_after_bad_channel_removal",
    }
    missing = sorted(required.difference(region_rows.columns))
    if missing:
        raise SystemExit(f"Region rows missing ablation-rule columns: {missing}")

    top_k = int(rule["top_k"])
    rate_threshold = float(rule["burned_rate_strictly_greater_than"])
    count_threshold = int(rule["burned_count_strictly_greater_than"])
    eligible_group = str(rule.get("eligible_region_group", "region_model_eligible"))
    if top_k < 1:
        raise SystemExit("labels.rule.top_k must be positive")
    if not 0.0 <= rate_threshold <= 1.0:
        raise SystemExit("labels.rule burned-rate threshold must be in [0, 1]")

    rank_by_index: dict[int, int] = {}
    for _, patient_rows in region_rows.groupby("patient", sort=True):
        eligible = patient_rows.loc[
            patient_rows["region_group"].eq(eligible_group)
            & ~patient_rows["is_white_csf_ventricle_unknown"].map(truthy)
            & patient_rows["n_bipolar_pairs_after_bad_channel_removal"].gt(0)
        ].sort_values(
            ["n_doctor_burned_pairs_after_bad_channel_removal", "display_order"],
            ascending=[False, True],
            kind="mergesort",
        )
        for rank, index in enumerate(eligible.index, start=1):
            rank_by_index[int(index)] = rank

    audit_records: list[dict[str, object]] = []
    label_records: list[dict[str, object]] = []
    for index, row in region_rows.iterrows():
        n_pairs = int(row["n_bipolar_pairs_after_bad_channel_removal"])
        n_burned = int(row["n_doctor_burned_pairs_after_bad_channel_removal"])
        if n_pairs < 0 or n_burned < 0 or n_burned > n_pairs:
            raise SystemExit(
                f"Invalid after-bad counts for {row['patient']} / "
                f"{row['desikan_killiany_label']}: burned={n_burned}, pairs={n_pairs}"
            )
        burned_rate = float(n_burned / n_pairs) if n_pairs else float("nan")
        rank = rank_by_index.get(int(index))
        in_top_k = rank is not None and rank <= top_k
        passes_rate = bool(np.isfinite(burned_rate) and burned_rate > rate_threshold)
        passes_count = n_burned > count_threshold
        is_not_applicable = (
            str(row["region_group"]) != eligible_group
            or truthy(row["is_white_csf_ventricle_unknown"])
        )

        if is_not_applicable:
            derived_status = "not_applicable"
            inclusion_status = "excluded_not_applicable"
            label_binary = float("nan")
            reason = "excluded_not_applicable"
        elif n_pairs == 0:
            derived_status = "no"
            inclusion_status = "excluded_no_signal_coverage"
            label_binary = float("nan")
            reason = "excluded_no_signal_coverage"
        elif in_top_k and (passes_rate or passes_count):
            derived_status = "yes"
            inclusion_status = "primary"
            label_binary = 1.0
            reason = "top_k_and_threshold_pass"
        else:
            derived_status = "no"
            inclusion_status = "primary"
            label_binary = 0.0
            reason = "outside_top_k" if not in_top_k else "threshold_not_met"

        original_status = clean_text(row["suspicious_region_status"]).lower()
        key_values = {key: row[key] for key in REGION_KEYS}
        label_records.append(
            {
                **key_values,
                "suspicious_region_status": derived_status,
                "label_binary": label_binary,
                "inclusion_status": inclusion_status,
                "n_bipolar_pairs_after_bad_channel_removal": n_pairs,
            }
        )
        audit_records.append(
            {
                **key_values,
                "display_order": int(row["display_order"]),
                "region_group": str(row["region_group"]),
                "original_suspicious_region_status": original_status,
                "derived_suspicious_region_status": derived_status,
                "inclusion_status": inclusion_status,
                "n_bipolar_pairs_after_bad_channel_removal": n_pairs,
                "n_doctor_burned_pairs_after_bad_channel_removal": n_burned,
                "burned_rate_after_bad_channel_removal": burned_rate,
                "burned_count_rank_within_patient": rank,
                "is_within_exact_top_k": in_top_k,
                "passes_strict_burned_rate_threshold": passes_rate,
                "passes_strict_burned_count_threshold": passes_count,
                "label_changed_from_audit_status": original_status != derived_status,
                "derivation_reason": reason,
            }
        )

    labels = pd.DataFrame.from_records(label_records)
    audit = pd.DataFrame.from_records(audit_records).sort_values(
        ["patient", "display_order"], kind="mergesort"
    )
    return labels, audit


def validate_ablation_rule_expectations(
    labels: pd.DataFrame,
    audit: pd.DataFrame,
    expected: dict[str, object],
) -> None:
    primary = labels.loc[labels["inclusion_status"].eq("primary")]
    actual = {
        "label_rows": len(labels),
        "primary_rows": len(primary),
        "positive_rows": int(primary["label_binary"].sum()),
        "negative_rows": int((primary["label_binary"] == 0).sum()),
        "changed_rows": int(audit["label_changed_from_audit_status"].sum()),
    }
    for key, value in actual.items():
        expected_value = int(expected[key])
        if value != expected_value:
            raise SystemExit(f"Ablation-rule expectation failed for {key}: {value} != {expected_value}")

    expected_changes = {
        (str(item["patient"]), str(item["desikan_killiany_label"]), str(item["from"]), str(item["to"]))
        for item in expected.get("changes", [])
    }
    changed = audit.loc[audit["label_changed_from_audit_status"]]
    actual_changes = {
        (
            str(row.patient),
            str(row.desikan_killiany_label),
            str(row.original_suspicious_region_status),
            str(row.derived_suspicious_region_status),
        )
        for row in changed.itertuples(index=False)
    }
    if actual_changes != expected_changes:
        raise SystemExit(
            f"Ablation-rule changed-label set mismatch: {sorted(actual_changes)} != "
            f"{sorted(expected_changes)}"
        )


def feature_columns(features: pd.DataFrame) -> list[str]:
    numeric_columns = [
        column
        for column in features.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.startswith("qc__")
        and pd.api.types.is_numeric_dtype(features[column])
    ]
    return sorted(numeric_columns)


def feature_band(column: str) -> str | None:
    parts = str(column).split("__")
    if len(parts) < 3:
        return None
    if parts[0] == "trial" and len(parts) >= 4:
        return parts[2]
    if parts[0] == "contrast":
        return parts[1]
    return None


def feature_method(column: str) -> str | None:
    parts = str(column).split("__")
    if parts[0] == "trial" and len(parts) >= 5:
        return parts[3]
    if parts[0] == "contrast" and len(parts) >= 4:
        return parts[2]
    return None


def feature_columns_for_band(features: pd.DataFrame, band: str) -> list[str]:
    return [column for column in feature_columns(features) if feature_band(column) == band]


def select_feature_methods(features: pd.DataFrame, network_methods: Iterable[str]) -> pd.DataFrame:
    selected_methods = {str(value) for value in network_methods}
    keep: list[str] = []
    for column in features.columns:
        method = feature_method(column)
        if method is None or method in selected_methods:
            keep.append(column)
    return features.loc[:, keep].copy()


def validate_feature_method_coverage(
    features: pd.DataFrame,
    frequency_bands: Iterable[str],
    network_methods: Iterable[str],
    expected_per_method_per_band: int | None = None,
) -> dict[str, dict[str, int]]:
    expected_methods = {str(value) for value in network_methods}
    if not expected_methods:
        raise SystemExit("No network methods configured")
    coverage: dict[str, dict[str, int]] = {}
    for band in [str(value) for value in frequency_bands]:
        candidates = feature_columns_for_band(features, band)
        counts = {method: 0 for method in sorted(expected_methods)}
        unknown: list[str] = []
        for column in candidates:
            method = feature_method(column)
            if method not in expected_methods:
                unknown.append(column)
            else:
                counts[method] += 1
        if unknown:
            raise SystemExit(f"Unknown method feature columns for {band}: {unknown[:5]}")
        missing = [method for method, count in counts.items() if count == 0]
        if missing:
            raise SystemExit(f"Missing method feature candidates for {band}: {missing}")
        if expected_per_method_per_band is not None:
            mismatched = {
                method: count
                for method, count in counts.items()
                if count != int(expected_per_method_per_band)
            }
            if mismatched:
                raise SystemExit(
                    f"Unexpected feature candidate counts for {band}: {mismatched}; "
                    f"expected_each={expected_per_method_per_band}"
                )
        coverage[band] = counts
    return coverage


def assert_selected_features_match_band(selected_features: list[str], band: str) -> None:
    mismatched = [feature for feature in selected_features if feature_band(feature) != band]
    if mismatched:
        raise SystemExit(f"Selected cross-band features for {band}: {mismatched}")


def join_features_and_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    merged = features.merge(labels, on=REGION_KEYS, how="inner", validate="one_to_one")
    return merged[merged["inclusion_status"] == "primary"].copy()


def _fit_predict_scores(
    train: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    c_value: float,
) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    x_train = scaler.fit_transform(imputer.fit_transform(train[columns]))
    x_test = scaler.transform(imputer.transform(test[columns]))
    model = LogisticRegression(
        l1_ratio=0.0,
        class_weight="balanced",
        solver="liblinear",
        C=float(c_value),
        random_state=0,
    )
    model.fit(x_train, train["label_binary"].astype(int))
    return model.predict_proba(x_test)[:, 1]


def _safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _average_precision_fast(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    true_positive_cumulative = np.cumsum(sorted_y)
    ap = 0.0
    start = 0
    while start < len(sorted_scores):
        end = start
        while end + 1 < len(sorted_scores) and sorted_scores[end + 1] == sorted_scores[start]:
            end += 1
        group_positives = int(sorted_y[start : end + 1].sum())
        if group_positives:
            precision_at_threshold = float(true_positive_cumulative[end] / (end + 1))
            ap += float(group_positives / n_positive) * precision_at_threshold
        start = end + 1
    return ap


def _average_precision_columns(
    y_true: np.ndarray,
    score_matrix: np.ndarray,
    descending: bool,
) -> np.ndarray:
    """Compute sklearn-equivalent AP for columns, grouping equal thresholds.

    Average precision is a sum over *distinct score thresholds*.  Treating rows
    inside a tied score group as separate thresholds makes the result depend on
    their input order and can substantially inflate AP for discrete features.
    This implementation sorts each column once, then iterates over rows while
    updating every feature column vectorially at the end of each tied group.
    """
    y_true = np.asarray(y_true, dtype=int)
    score_matrix = np.asarray(score_matrix, dtype=float)
    if y_true.ndim != 1 or score_matrix.ndim != 2:
        raise ValueError("y_true must be one-dimensional and score_matrix two-dimensional")
    if score_matrix.shape[0] != len(y_true):
        raise ValueError("score_matrix row count must equal y_true length")
    n_positive = int(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        return np.full(score_matrix.shape[1], np.nan)
    output = np.full(score_matrix.shape[1], np.nan, dtype=float)
    finite_columns = np.all(np.isfinite(score_matrix), axis=0)
    if not finite_columns.any():
        return output

    finite_scores = score_matrix[:, finite_columns]
    sort_values = -finite_scores if descending else finite_scores
    order = np.argsort(sort_values, axis=0, kind="mergesort")
    sorted_scores = np.take_along_axis(finite_scores, order, axis=0)
    sorted_y = y_true[order]
    true_positive_cumulative = np.cumsum(sorted_y, axis=0)
    group_end = np.ones_like(sorted_scores, dtype=bool)
    if len(sorted_scores) > 1:
        group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]

    previous_group_tp = np.zeros(sorted_scores.shape[1], dtype=float)
    ap = np.zeros(sorted_scores.shape[1], dtype=float)
    for row_idx in range(sorted_scores.shape[0]):
        ending = group_end[row_idx]
        if not ending.any():
            continue
        cumulative_tp = true_positive_cumulative[row_idx].astype(float, copy=False)
        group_positives = cumulative_tp - previous_group_tp
        precision_at_threshold = cumulative_tp / float(row_idx + 1)
        ap[ending] += (
            group_positives[ending]
            * precision_at_threshold[ending]
            / float(n_positive)
        )
        previous_group_tp[ending] = cumulative_tp[ending]
    output[finite_columns] = ap
    return output


def score_features_inner_auprc(train: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    x = train[columns].to_numpy(dtype=float)
    y = train["label_binary"].to_numpy(dtype=int)
    patients = train["patient"].to_numpy()
    return score_features_inner_auprc_array(x, y, patients, columns)


def score_features_inner_auprc_array(
    x: np.ndarray,
    y: np.ndarray,
    patients: np.ndarray,
    columns: list[str],
) -> pd.DataFrame:
    """Score features with orientation learned only from inner-training patients."""
    score_sums = np.zeros(x.shape[1], dtype=float)
    score_counts = np.zeros(x.shape[1], dtype=int)
    for heldout in sorted(set(patients.tolist())):
        train_mask = patients != heldout
        test_mask = patients == heldout
        y_train = y[train_mask]
        y_test = y[test_mask]
        if len(np.unique(y_test)) < 2:
            continue
        x_train = x[train_mask]
        x_test = x[test_mask]
        with np.errstate(all="ignore"):
            fill_values = np.nanmedian(x_train, axis=0)
            train_min = np.nanmin(x_train, axis=0)
            train_max = np.nanmax(x_train, axis=0)
            positive_median = np.nanmedian(x_train[y_train == 1], axis=0)
            negative_median = np.nanmedian(x_train[y_train == 0], axis=0)
        valid_feature = (
            np.isfinite(fill_values)
            & np.isfinite(train_min)
            & np.isfinite(train_max)
            & (train_min != train_max)
            & np.isfinite(positive_median)
            & np.isfinite(negative_median)
        )
        test_matrix = np.where(
            np.isfinite(x_test), x_test, fill_values[np.newaxis, :]
        )
        direction = np.where(positive_median >= negative_median, 1.0, -1.0)
        oriented_test = test_matrix * direction[np.newaxis, :]
        scores = _average_precision_columns(
            y_test, oriented_test, descending=True
        )
        valid = np.isfinite(scores) & valid_feature
        score_sums[valid] += scores[valid]
        score_counts[valid] += 1
    rows = [
        {
            "feature": column,
            "inner_auprc": float(score_sums[idx] / score_counts[idx]) if score_counts[idx] else float("-inf"),
        }
        for idx, column in enumerate(columns)
    ]
    return pd.DataFrame.from_records(rows)


def select_features_and_c(
    train: pd.DataFrame,
    candidates: list[str],
    top_k_features: int,
    c_values: list[float],
    non_missing_threshold: float,
) -> tuple[list[str], float, pd.DataFrame]:
    x_all = train[candidates].to_numpy(dtype=float)
    non_missing = np.mean(np.isfinite(x_all), axis=0)
    with np.errstate(all="ignore"):
        feature_min = np.nanmin(x_all, axis=0)
        feature_max = np.nanmax(x_all, axis=0)
    eligible_mask = (non_missing >= float(non_missing_threshold)) & np.isfinite(feature_min) & np.isfinite(feature_max) & (feature_min != feature_max)
    eligible = [column for column, keep in zip(candidates, eligible_mask, strict=True) if keep]
    if not eligible:
        raise SystemExit("No eligible features after missingness and variance filters")
    eligible_indices = np.flatnonzero(eligible_mask)
    feature_score_frame = score_features_inner_auprc_array(
        x_all[:, eligible_indices],
        train["label_binary"].to_numpy(dtype=int),
        train["patient"].to_numpy(),
        eligible,
    ).sort_values(
        ["inner_auprc", "feature"],
        ascending=[False, True],
    )
    selected = feature_score_frame["feature"].head(top_k_features).tolist()

    c_rows: list[dict[str, object]] = []
    for c_value in sorted(float(c) for c in c_values):
        fold_scores: list[float] = []
        for heldout in sorted(train["patient"].unique()):
            inner_train = train[train["patient"] != heldout]
            inner_test = train[train["patient"] == heldout]
            if inner_train["label_binary"].nunique() < 2 or inner_test["label_binary"].nunique() < 2:
                continue
            pred = _fit_predict_scores(inner_train, inner_test, selected, c_value)
            score = _safe_auprc(inner_test["label_binary"].to_numpy(dtype=int), pred)
            if np.isfinite(score):
                fold_scores.append(score)
        c_rows.append(
            {
                "C": c_value,
                "inner_auprc": float(np.mean(fold_scores)) if fold_scores else float("-inf"),
            }
        )
    c_frame = pd.DataFrame(c_rows).sort_values(["inner_auprc", "C"], ascending=[False, True])
    return selected, float(c_frame.iloc[0]["C"]), feature_score_frame


def run_lopo_predictions(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    top_k_features: int = 5,
    c_values: Iterable[float] | None = None,
    random_seed: int = 20260706,
    non_missing_threshold: float = 0.70,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    del random_seed
    c_list = [0.01, 0.1, 1.0, 10.0] if c_values is None else [float(value) for value in c_values]
    data = join_features_and_labels(features, labels)
    candidates = feature_columns(data)
    return run_lopo_predictions_from_candidates(
        data,
        candidates,
        top_k_features=top_k_features,
        c_values=c_list,
        non_missing_threshold=non_missing_threshold,
    )


def run_lopo_predictions_from_candidates(
    data: pd.DataFrame,
    candidates: list[str],
    top_k_features: int,
    c_values: list[float],
    non_missing_threshold: float,
    frequency_band: str | None = None,
    model_scope: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    if not candidates:
        scope = f" for {frequency_band}" if frequency_band else ""
        raise SystemExit(f"No candidate features{scope}")
    predictions: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for heldout_patient in sorted(data["patient"].unique()):
        train = data[data["patient"] != heldout_patient].copy()
        test = data[data["patient"] == heldout_patient].copy()
        selected, selected_c, score_frame = select_features_and_c(
            train,
            candidates,
            top_k_features=top_k_features,
            c_values=c_values,
            non_missing_threshold=non_missing_threshold,
        )
        if frequency_band is not None:
            assert_selected_features_match_band(selected, frequency_band)
        scores = _fit_predict_scores(train, test, selected, selected_c)
        order = pd.Series(scores).rank(method="average", ascending=False).to_numpy(dtype=float)
        for row, score, rank in zip(test.to_dict("records"), scores, order, strict=True):
            prediction_row = {
                "heldout_patient": heldout_patient,
                "patient": row["patient"],
                "desikan_killiany_label": row["desikan_killiany_label"],
                "tissue_class": row["tissue_class"],
                "is_white_csf_ventricle_unknown": row["is_white_csf_ventricle_unknown"],
                "y_true": int(row["label_binary"]),
                "y_score": float(score),
                "rank_within_patient": float(rank),
                "selected_features": "|".join(selected),
                "selected_C": selected_c,
            }
            if frequency_band is not None:
                prediction_row["frequency_band"] = frequency_band
                prediction_row["model_scope"] = model_scope or "by_frequency_band"
            predictions.append(prediction_row)
        fold_row = {
            "heldout_patient": heldout_patient,
            "train_patients": "|".join(sorted(train["patient"].unique())),
            "selected_features": "|".join(selected),
            "selected_C": selected_c,
            "univariate_orientation_rule": UNIVARIATE_ORIENTATION_RULE,
            "average_precision_implementation": AVERAGE_PRECISION_IMPLEMENTATION,
            "feature_score_table": score_frame.to_json(orient="records"),
        }
        if frequency_band is not None:
            fold_row["frequency_band"] = frequency_band
            fold_row["model_scope"] = model_scope or "by_frequency_band"
        fold_rows.append(fold_row)
    return pd.DataFrame.from_records(predictions), fold_rows


def run_lopo_predictions_by_band(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    frequency_bands: Iterable[str],
    top_k_features: int = 5,
    c_values: Iterable[float] | None = None,
    random_seed: int = 20260706,
    non_missing_threshold: float = 0.70,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    del random_seed
    c_list = [0.01, 0.1, 1.0, 10.0] if c_values is None else [float(value) for value in c_values]
    data = join_features_and_labels(features, labels)
    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for band in [str(value) for value in frequency_bands]:
        candidates = feature_columns_for_band(data, band)
        predictions, folds = run_lopo_predictions_from_candidates(
            data,
            candidates,
            top_k_features=top_k_features,
            c_values=c_list,
            non_missing_threshold=non_missing_threshold,
            frequency_band=band,
            model_scope="by_frequency_band",
        )
        prediction_frames.append(predictions)
        fold_rows.extend(folds)
    if not prediction_frames:
        raise SystemExit("No frequency bands configured")
    return pd.concat(prediction_frames, ignore_index=True), fold_rows


def top_recall(predictions: pd.DataFrame, k: int | None = None, fraction: float | None = None) -> float:
    recalls: list[float] = []
    for _, patient_frame in predictions.groupby("patient"):
        positives = int(patient_frame["y_true"].sum())
        if positives == 0:
            continue
        if fraction is not None:
            n_top = max(1, int(np.ceil(fraction * len(patient_frame))))
        elif k is not None:
            n_top = min(k, len(patient_frame))
        else:
            raise ValueError("Provide either k or fraction")
        top = patient_frame.sort_values("y_score", ascending=False).head(n_top)
        recalls.append(float(top["y_true"].sum() / positives))
    return float(np.mean(recalls)) if recalls else float("nan")


def _validate_metric_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    required = {"patient", "y_true", "y_score"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table missing metric columns: {missing}")
    if predictions.empty:
        raise ValueError("Cannot compute region metrics from an empty prediction table")
    frame = predictions.loc[:, ["patient", "y_true", "y_score"]].copy()
    if frame["patient"].isna().any() or frame["patient"].astype(str).str.strip().eq("").any():
        raise ValueError("Region metric rows require a non-empty patient identifier")
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["y_score"] = pd.to_numeric(frame["y_score"], errors="raise").astype(float)
    if not set(frame["y_true"].unique()).issubset({0, 1}):
        raise ValueError("Region metric labels must be binary")
    if not np.isfinite(frame["y_score"].to_numpy(dtype=float)).all():
        raise ValueError("Region metric scores must be finite")
    return frame


def compute_patient_prediction_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute ranking metrics within each held-out patient."""
    frame = _validate_metric_predictions(predictions)
    rows: list[dict[str, object]] = []
    for patient, patient_frame in frame.groupby("patient", sort=False):
        y_true = patient_frame["y_true"].to_numpy(dtype=int)
        scores = patient_frame["y_score"].to_numpy(dtype=float)
        if len(np.unique(y_true)) != 2:
            raise ValueError(
                f"Patient {patient} does not contain both regional label classes; "
                "patient-macro AUPRC/AUROC is undefined"
            )
        prevalence = float(np.mean(y_true))
        auprc = float(average_precision_score(y_true, scores))
        normalized = float((auprc - prevalence) / (1.0 - prevalence))
        rows.append(
            {
                "patient": str(patient),
                "n_regions": len(patient_frame),
                "n_positive_regions": int(y_true.sum()),
                "prevalence": prevalence,
                "average_precision": auprc,
                "normalized_average_precision": normalized,
                "AUPRC": auprc,
                "normalized_AUPRC": normalized,
                "AUROC": float(roc_auc_score(y_true, scores)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _pooled_binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prevalence = float(np.mean(y_true))
    auprc = float(average_precision_score(y_true, scores))
    normalized = (
        float((auprc - prevalence) / (1.0 - prevalence))
        if prevalence < 1.0
        else float("nan")
    )
    auroc = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "average_precision": auprc,
        "normalized_average_precision": normalized,
        "AUPRC": auprc,
        "normalized_AUPRC": normalized,
        "AUROC": auroc,
        "prevalence": prevalence,
    }


def compute_prediction_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Compute the patient-macro primary metric plus two pooled audit estimands."""
    frame = _validate_metric_predictions(predictions)
    patient_metrics = compute_patient_prediction_metrics(frame)

    y_true = frame["y_true"].to_numpy(dtype=int)
    raw_scores = frame["y_score"].to_numpy(dtype=float)
    pooled_raw = _pooled_binary_metrics(y_true, raw_scores)

    rank_scores = frame.groupby("patient", sort=False)["y_score"].rank(
        method="average", pct=True
    )
    rank_pooled = _pooled_binary_metrics(y_true, rank_scores.to_numpy(dtype=float))

    return {
        # Public primary columns remain backward-compatible, but now contain
        # patient-equal macro metrics rather than raw-score pooled metrics.
        "average_precision": float(patient_metrics["average_precision"].mean()),
        "normalized_average_precision": float(
            patient_metrics["normalized_average_precision"].mean()
        ),
        "AUPRC": float(patient_metrics["AUPRC"].mean()),
        "normalized_AUPRC": float(patient_metrics["normalized_AUPRC"].mean()),
        "AUROC": float(patient_metrics["AUROC"].mean()),
        "prevalence": float(patient_metrics["prevalence"].mean()),
        "primary_metric": PRIMARY_REGION_METRIC,
        "metric_scope": PRIMARY_REGION_METRIC_SCOPE,
        "n_patients_metric": len(patient_metrics),
        "pooled_prevalence": pooled_raw["prevalence"],
        "pooled_raw_AUPRC": pooled_raw["AUPRC"],
        "pooled_raw_normalized_AUPRC": pooled_raw["normalized_AUPRC"],
        "pooled_raw_AUROC": pooled_raw["AUROC"],
        "within_patient_rank_pooled_AUPRC": rank_pooled["AUPRC"],
        "within_patient_rank_pooled_normalized_AUPRC": rank_pooled[
            "normalized_AUPRC"
        ],
        "within_patient_rank_pooled_AUROC": rank_pooled["AUROC"],
        "top1_recall": top_recall(predictions, k=1),
        "top2_recall": top_recall(predictions, k=2),
        "top3_recall": top_recall(predictions, k=3),
        "top20pct_recall": top_recall(predictions, fraction=0.20),
    }


def compute_patient_prediction_metrics_by_band(predictions: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for band, frame in predictions.groupby("frequency_band", sort=False):
        patient_metrics = compute_patient_prediction_metrics(frame)
        patient_metrics.insert(0, "metric_scope", "within_heldout_patient")
        patient_metrics.insert(0, "frequency_band", band)
        frames.append(patient_metrics)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compute_prediction_metrics_by_band(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for band, frame in predictions.groupby("frequency_band", sort=False):
        rows.append(
            {
                "frequency_band": band,
                "model_scope": "by_frequency_band",
                **compute_prediction_metrics(frame),
            }
        )
    return pd.DataFrame.from_records(rows)


def permute_labels_within_patient(labels: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    output = labels.copy()
    for patient, idx in output.groupby("patient").groups.items():
        primary_idx = output.loc[idx][output.loc[idx, "inclusion_status"] == "primary"].index
        values = output.loc[primary_idx, "label_binary"].to_numpy(dtype=float)
        output.loc[primary_idx, "label_binary"] = rng.permutation(values)
    return output


def run_permutation_test(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    observed_auprc: float,
    n_permutations: int,
    seed: int,
    top_k_features: int,
    c_values: list[float],
    non_missing_threshold: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for permutation_id in range(1, n_permutations + 1):
        permuted = permute_labels_within_patient(labels, rng)
        predictions, _ = run_lopo_predictions(
            features,
            permuted,
            top_k_features=top_k_features,
            c_values=c_values,
            random_seed=seed + permutation_id,
            non_missing_threshold=non_missing_threshold,
        )
        metrics = compute_prediction_metrics(predictions)
        rows.append(
            {
                "permutation_id": permutation_id,
                "seed": seed + permutation_id,
                "primary_metric": PRIMARY_REGION_METRIC,
                "metric_scope": PRIMARY_REGION_METRIC_SCOPE,
                "average_precision_implementation": AVERAGE_PRECISION_IMPLEMENTATION,
                "AUPRC": metrics["AUPRC"],
                "pooled_raw_AUPRC": metrics["pooled_raw_AUPRC"],
                "within_patient_rank_pooled_AUPRC": metrics[
                    "within_patient_rank_pooled_AUPRC"
                ],
                "null_ge_observed_AUPRC": bool(metrics["AUPRC"] >= observed_auprc),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_permutation_test_by_band(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    frequency_bands: Iterable[str],
    n_permutations: int,
    seed: int,
    top_k_features: int,
    c_values: list[float],
    non_missing_threshold: float,
) -> pd.DataFrame:
    observed = observed_metrics.set_index("frequency_band")["AUPRC"].to_dict()
    rows: list[dict[str, object]] = []
    for band_idx, band in enumerate([str(value) for value in frequency_bands]):
        observed_auprc = float(observed[band])
        for permutation_id in range(1, n_permutations + 1):
            permutation_seed = int(seed + (band_idx + 1) * 100000 + permutation_id)
            permuted = permute_labels_within_patient(labels, np.random.default_rng(permutation_seed))
            predictions, _ = run_lopo_predictions_by_band(
                features,
                permuted,
                frequency_bands=[band],
                top_k_features=top_k_features,
                c_values=c_values,
                random_seed=permutation_seed,
                non_missing_threshold=non_missing_threshold,
            )
            metrics = compute_prediction_metrics(predictions)
            rows.append(
                {
                    "frequency_band": band,
                    "model_scope": "by_frequency_band",
                    "permutation_id": permutation_id,
                    "seed": permutation_seed,
                    "primary_metric": PRIMARY_REGION_METRIC,
                    "metric_scope": PRIMARY_REGION_METRIC_SCOPE,
                    "average_precision_implementation": AVERAGE_PRECISION_IMPLEMENTATION,
                    "AUPRC": metrics["AUPRC"],
                    "pooled_raw_AUPRC": metrics["pooled_raw_AUPRC"],
                    "within_patient_rank_pooled_AUPRC": metrics[
                        "within_patient_rank_pooled_AUPRC"
                    ],
                    "null_ge_observed_AUPRC": bool(metrics["AUPRC"] >= observed_auprc),
                }
            )
    return pd.DataFrame.from_records(rows)


def run_patient_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patients = sorted(predictions["patient"].unique())
    rows: list[dict[str, object]] = []
    for bootstrap_id in range(1, n_bootstrap + 1):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        frames = []
        for sample_idx, patient in enumerate(sampled):
            frame = predictions[predictions["patient"] == patient].copy()
            frame["patient"] = f"{patient}__boot{sample_idx}"
            frames.append(frame)
        metrics = compute_prediction_metrics(pd.concat(frames, ignore_index=True))
        rows.append({"bootstrap_id": bootstrap_id, **metrics})
    return pd.DataFrame.from_records(rows)


def run_patient_bootstrap_by_band(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for band_idx, (band, frame) in enumerate(predictions.groupby("frequency_band", sort=False)):
        bootstrap = run_patient_bootstrap(
            frame,
            n_bootstrap=n_bootstrap,
            seed=int(seed + (band_idx + 1) * 100000),
        )
        bootstrap.insert(0, "model_scope", "by_frequency_band")
        bootstrap.insert(0, "frequency_band", band)
        frames.append(bootstrap)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ci_from_bootstrap(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric in [
        "AUPRC",
        "normalized_AUPRC",
        "AUROC",
        "pooled_raw_AUPRC",
        "pooled_raw_normalized_AUPRC",
        "pooled_raw_AUROC",
        "within_patient_rank_pooled_AUPRC",
        "within_patient_rank_pooled_normalized_AUPRC",
        "within_patient_rank_pooled_AUROC",
    ]:
        values = bootstrap[metric].dropna()
        rows.append(
            {
                "metric": metric,
                "primary_metric": PRIMARY_REGION_METRIC,
                "metric_scope": PRIMARY_REGION_METRIC_SCOPE,
                "ci_low": float(values.quantile(0.025)) if not values.empty else float("nan"),
                "ci_high": float(values.quantile(0.975)) if not values.empty else float("nan"),
            }
        )
    return pd.DataFrame.from_records(rows)


def ci_from_bootstrap_by_band(bootstrap: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for band, frame in bootstrap.groupby("frequency_band", sort=False):
        ci = ci_from_bootstrap(frame)
        ci.insert(0, "model_scope", "by_frequency_band")
        ci.insert(0, "frequency_band", band)
        frames.append(ci)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def add_permutation_p_values_by_band(metrics: pd.DataFrame, permutation: pd.DataFrame) -> pd.DataFrame:
    p_rows: list[dict[str, object]] = []
    for band, frame in permutation.groupby("frequency_band", sort=False):
        p_rows.append(
            {
                "frequency_band": band,
                "permutation_statistic": PRIMARY_REGION_METRIC,
                "permutation_p_AUPRC": float(
                    (int(frame["null_ge_observed_AUPRC"].sum()) + 1) / (len(frame) + 1)
                ),
            }
        )
    return metrics.merge(pd.DataFrame.from_records(p_rows), on="frequency_band", how="left")


def frames_equal_as_text(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns):
        return False
    left_text = left.fillna("").astype(str).reset_index(drop=True)
    right_text = right.fillna("").astype(str).reset_index(drop=True)
    return bool(left_text.equals(right_text))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_dataframe_atomic(frame: pd.DataFrame, path: Path) -> None:
    assert_allowed_output_root(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    assert_allowed_output_root(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}_", suffix=".tmp", dir=path.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def compute_bonferroni_correction(
    metrics: pd.DataFrame,
    permutation: pd.DataFrame,
    frequency_bands: Iterable[str],
) -> pd.DataFrame:
    bands = [str(value) for value in frequency_bands]
    if metrics["frequency_band"].tolist() != bands:
        raise SystemExit("Metrics do not contain the locked frequency-band order")
    rows: list[dict[str, object]] = []
    for row in metrics.itertuples(index=False):
        band = str(row.frequency_band)
        band_permutations = permutation.loc[permutation["frequency_band"].eq(band)]
        if band_permutations.empty:
            raise SystemExit(f"No permutation results for {band}")
        raw_p = float(
            (int(band_permutations["null_ge_observed_AUPRC"].sum()) + 1)
            / (len(band_permutations) + 1)
        )
        if not np.isclose(raw_p, float(row.permutation_p_AUPRC), atol=1e-15, rtol=0.0):
            raise SystemExit(f"Stored permutation P does not match recomputed P for {band}")
        rows.append(
            {
                "frequency_band": band,
                "observed_AUPRC": float(row.AUPRC),
                "primary_metric": PRIMARY_REGION_METRIC,
                "metric_scope": PRIMARY_REGION_METRIC_SCOPE,
                "n_permutations": len(band_permutations),
                "permutation_p_raw_AUPRC": raw_p,
                "multiple_comparison_method": "bonferroni_fwer",
                "n_frequency_tests": len(bands),
                "permutation_p_fwer_bonferroni_AUPRC": min(1.0, raw_p * len(bands)),
            }
        )
    return pd.DataFrame.from_records(rows)




def resolve_safe_worker_count(
    requested_n_jobs: int,
    parallel_config: dict[str, object],
    *,
    physical_cores: int | None = None,
    available_memory_gb: float | None = None,
) -> tuple[int, dict[str, object]]:
    if requested_n_jobs < 1:
        raise SystemExit("n_jobs must be positive")
    detected_physical = int(psutil.cpu_count(logical=False) or 1)
    physical = detected_physical if physical_cores is None else int(physical_cores)
    available = (
        float(psutil.virtual_memory().available / (1024**3))
        if available_memory_gb is None
        else float(available_memory_gb)
    )
    hard_max = int(parallel_config["hard_max_n_jobs"])
    reserve = float(parallel_config["memory_reserve_gb"])
    per_worker = float(parallel_config["estimated_worker_memory_gb"])
    minimum_start = float(parallel_config["minimum_available_memory_gb_to_start"])
    if hard_max < 1 or per_worker <= 0 or reserve < 0 or minimum_start <= 0:
        raise SystemExit("Invalid parallel safety configuration")
    if available < minimum_start:
        raise SystemExit(
            f"Insufficient available memory to start safely: {available:.2f} GB < "
            f"{minimum_start:.2f} GB"
        )
    memory_cap = max(1, int(np.floor(max(0.0, available - reserve) / per_worker)))
    safe_cap = max(1, min(physical, hard_max, memory_cap))
    actual = min(requested_n_jobs, safe_cap)
    diagnostics = {
        "requested_n_jobs": requested_n_jobs,
        "actual_n_jobs": actual,
        "detected_physical_cores": physical,
        "detected_logical_cores": int(psutil.cpu_count(logical=True) or physical),
        "available_memory_gb_at_start": available,
        "memory_reserve_gb": reserve,
        "estimated_worker_memory_gb": per_worker,
        "memory_worker_cap": memory_cap,
        "hard_max_n_jobs": hard_max,
        "blas_threads_per_worker": int(parallel_config["blas_threads_per_worker"]),
        "was_safety_clamped": actual != requested_n_jobs,
    }
    return actual, diagnostics


def _initialize_permutation_worker(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    observed: dict[str, float],
    seed: int,
    top_k_features: int,
    c_values: list[float],
    non_missing_threshold: float,
    blas_threads: int,
) -> None:
    global _PERMUTATION_WORKER_STATE
    _PERMUTATION_WORKER_STATE = {
        "features": features,
        "labels": labels,
        "observed": observed,
        "seed": seed,
        "top_k_features": top_k_features,
        "c_values": c_values,
        "non_missing_threshold": non_missing_threshold,
        "blas_threads": blas_threads,
    }


def _run_permutation_worker(task: tuple[int, str, int]) -> dict[str, object]:
    band_idx, band, permutation_id = task
    state = _PERMUTATION_WORKER_STATE
    if not state:
        raise RuntimeError("Permutation worker state was not initialized")
    permutation_seed = int(state["seed"]) + (band_idx + 1) * 100000 + permutation_id
    with threadpool_limits(limits=int(state["blas_threads"])):
        permuted = permute_labels_within_patient(
            state["labels"], np.random.default_rng(permutation_seed)
        )
        predictions, _ = run_lopo_predictions_by_band(
            state["features"],
            permuted,
            frequency_bands=[band],
            top_k_features=int(state["top_k_features"]),
            c_values=state["c_values"],
            random_seed=permutation_seed,
            non_missing_threshold=float(state["non_missing_threshold"]),
        )
        permutation_metrics = compute_prediction_metrics(predictions)
        permutation_auprc = permutation_metrics["AUPRC"]
    return {
        "frequency_band": band,
        "model_scope": "by_frequency_band",
        "univariate_orientation_rule": UNIVARIATE_ORIENTATION_RULE,
        "permutation_id": permutation_id,
        "seed": permutation_seed,
        "primary_metric": PRIMARY_REGION_METRIC,
        "metric_scope": PRIMARY_REGION_METRIC_SCOPE,
        "average_precision_implementation": AVERAGE_PRECISION_IMPLEMENTATION,
        "AUPRC": permutation_auprc,
        "pooled_raw_AUPRC": permutation_metrics["pooled_raw_AUPRC"],
        "within_patient_rank_pooled_AUPRC": permutation_metrics[
            "within_patient_rank_pooled_AUPRC"
        ],
        "null_ge_observed_AUPRC": bool(
            permutation_auprc >= float(state["observed"][band])
        ),
    }


def run_permutation_test_by_band_checkpointed(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    frequency_bands: Iterable[str],
    n_permutations: int,
    seed: int,
    top_k_features: int,
    c_values: list[float],
    non_missing_threshold: float,
    checkpoint_dir: Path,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
    n_jobs: int = 1,
    blas_threads_per_worker: int = 1,
    max_tasks_per_child: int = 500,
    minimum_free_memory_gb_during_run: float = 1.0,
) -> pd.DataFrame:
    if checkpoint_every < 1 or n_jobs < 1 or blas_threads_per_worker < 1:
        raise SystemExit("checkpoint_every, n_jobs and BLAS thread count must be positive")
    if max_tasks_per_child < 1 or minimum_free_memory_gb_during_run <= 0:
        raise SystemExit("Invalid process recycling or runtime memory threshold")
    assert_allowed_output_root(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    observed = observed_metrics.set_index("frequency_band")["AUPRC"].to_dict()
    bands = [str(value) for value in frequency_bands]
    records_by_band: dict[str, list[dict[str, object]]] = {}
    checkpoint_by_band: dict[str, Path] = {}
    pending: list[tuple[int, str, int]] = []
    for band_idx, band in enumerate(bands):
        checkpoint = checkpoint_dir / f"permutations_{band}.csv"
        checkpoint_by_band[band] = checkpoint
        if checkpoint.exists() and not (resume or overwrite):
            raise SystemExit(f"Permutation checkpoint exists; pass --resume or --overwrite: {checkpoint}")
        if checkpoint.exists() and resume:
            completed = pd.read_csv(checkpoint)
        else:
            completed = pd.DataFrame()
        if not completed.empty:
            required = {
                "frequency_band",
                "model_scope",
                "univariate_orientation_rule",
                "permutation_id",
                "seed",
                "primary_metric",
                "metric_scope",
                "average_precision_implementation",
                "AUPRC",
                "pooled_raw_AUPRC",
                "within_patient_rank_pooled_AUPRC",
                "null_ge_observed_AUPRC",
            }
            missing = sorted(required.difference(completed.columns))
            if missing:
                raise SystemExit(f"Checkpoint {checkpoint} missing columns: {missing}")
            if set(completed["frequency_band"].astype(str)) != {band}:
                raise SystemExit(f"Checkpoint contains the wrong band: {checkpoint}")
            if set(completed["univariate_orientation_rule"].astype(str)) != {
                UNIVARIATE_ORIENTATION_RULE
            }:
                raise SystemExit(
                    f"Checkpoint uses an incompatible orientation rule: {checkpoint}"
                )
            if set(completed["primary_metric"].astype(str)) != {PRIMARY_REGION_METRIC}:
                raise SystemExit(
                    f"Checkpoint uses an incompatible primary metric: {checkpoint}"
                )
            if set(completed["metric_scope"].astype(str)) != {
                PRIMARY_REGION_METRIC_SCOPE
            }:
                raise SystemExit(
                    f"Checkpoint uses an incompatible metric scope: {checkpoint}"
                )
            if set(completed["average_precision_implementation"].astype(str)) != {
                AVERAGE_PRECISION_IMPLEMENTATION
            }:
                raise SystemExit(
                    f"Checkpoint uses an incompatible AP implementation: {checkpoint}"
                )
            if completed["permutation_id"].duplicated().any():
                raise SystemExit(f"Checkpoint contains duplicate permutation IDs: {checkpoint}")
            ids = set(completed["permutation_id"].astype(int))
            if not ids.issubset(set(range(1, n_permutations + 1))):
                raise SystemExit(f"Checkpoint contains out-of-range permutation IDs: {checkpoint}")

        records = completed.to_dict("records")
        records_by_band[band] = records
        completed_ids = {int(row["permutation_id"]) for row in records}
        for permutation_id in range(1, n_permutations + 1):
            if permutation_id not in completed_ids:
                pending.append((band_idx, band, permutation_id))

    dirty_bands: set[str] = set()

    def flush_checkpoints() -> None:
        for dirty_band in sorted(dirty_bands):
            frame = pd.DataFrame.from_records(records_by_band[dirty_band]).sort_values(
                "permutation_id", kind="mergesort"
            )
            write_dataframe_atomic(frame, checkpoint_by_band[dirty_band])
        dirty_bands.clear()

    worker_args = (
        features,
        labels,
        observed,
        seed,
        top_k_features,
        c_values,
        non_missing_threshold,
        blas_threads_per_worker,
    )
    completed_new = 0
    if pending and n_jobs == 1:
        _initialize_permutation_worker(*worker_args)
        for task in pending:
            record = _run_permutation_worker(task)
            records_by_band[str(record["frequency_band"])].append(record)
            dirty_bands.add(str(record["frequency_band"]))
            completed_new += 1
            if completed_new % checkpoint_every == 0:
                flush_checkpoints()
                available_gb = psutil.virtual_memory().available / (1024**3)
                if available_gb < minimum_free_memory_gb_during_run:
                    raise SystemExit(
                        f"Available memory fell below the safety threshold after checkpoint: "
                        f"{available_gb:.2f} GB < {minimum_free_memory_gb_during_run:.2f} GB; "
                        "rerun with --resume"
                    )
    elif pending:
        executor = ProcessPoolExecutor(
            max_workers=min(n_jobs, len(pending)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_permutation_worker,
            initargs=worker_args,
            max_tasks_per_child=max_tasks_per_child,
        )
        futures = {executor.submit(_run_permutation_worker, task): task for task in pending}
        try:
            for future in as_completed(futures):
                task = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    flush_checkpoints()
                    for queued in futures:
                        queued.cancel()
                    raise SystemExit(
                        "Permutation worker failed after completed results were checkpointed; "
                        f"rerun with --resume. Failed task={task}: {error}"
                    ) from error
                records_by_band[str(record["frequency_band"])].append(record)
                dirty_bands.add(str(record["frequency_band"]))
                completed_new += 1
                if completed_new % checkpoint_every == 0:
                    flush_checkpoints()
                    available_gb = psutil.virtual_memory().available / (1024**3)
                    print(
                        f"permutation_progress={completed_new}/{len(pending)} "
                        f"workers={n_jobs} available_memory_gb={available_gb:.2f}"
                    )
                    if available_gb < minimum_free_memory_gb_during_run:
                        for queued in futures:
                            queued.cancel()
                        raise SystemExit(
                            f"Available memory fell below the safety threshold after checkpoint: "
                            f"{available_gb:.2f} GB < {minimum_free_memory_gb_during_run:.2f} GB; "
                            "rerun with --resume"
                        )
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
    flush_checkpoints()

    frames: list[pd.DataFrame] = []
    for band in bands:
        result = pd.DataFrame.from_records(records_by_band[band]).sort_values(
            "permutation_id", kind="mergesort"
        )
        if len(result) != n_permutations or set(result["permutation_id"].astype(int)) != set(
            range(1, n_permutations + 1)
        ):
            raise SystemExit(f"Incomplete permutation checkpoint for {band}")
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def write_region_labels(
    labels: pd.DataFrame,
    overwrite: bool,
    labels_file: Path | None = None,
) -> None:
    labels_file = FEATURE_ROOT / "region_labels.csv" if labels_file is None else Path(labels_file)
    assert_allowed_output_root(labels_file.parent)
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    if labels_file.exists():
        existing = pd.read_csv(labels_file)
        if frames_equal_as_text(existing, labels):
            return
        if not overwrite:
            raise SystemExit(f"Output already exists with different content; pass --overwrite: {labels_file}")
    labels.to_csv(labels_file, index=False, encoding="utf-8-sig")


def write_outputs_by_band(
    out_dir: Path,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    permutation: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    summary: dict[str, object],
    overwrite: bool,
    labels_file: Path | None = None,
) -> None:
    assert_allowed_output_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lopo_region_predictions_by_band.csv": predictions,
        "lopo_region_metrics_by_band.csv": metrics,
        "permutation_results_by_band.csv": permutation,
        "bootstrap_ci_by_band.csv": bootstrap_ci,
    }
    for filename in outputs:
        out_file = out_dir / filename
        if out_file.exists() and not overwrite:
            raise SystemExit(f"Output already exists; pass --overwrite: {out_file}")
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")
    summary_file = out_dir / "modeling_run_summary_by_band.json"
    if summary_file.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite: {summary_file}")
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_region_labels(labels, overwrite=overwrite, labels_file=labels_file)


def write_outputs(
    out_dir: Path,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: dict[str, float],
    permutation: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    summary: dict[str, object],
    overwrite: bool,
    labels_file: Path | None = None,
) -> None:
    assert_allowed_output_root(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "lopo_region_predictions.csv": predictions,
        "lopo_region_metrics.csv": pd.DataFrame([metrics]),
        "permutation_results.csv": permutation,
        "bootstrap_ci.csv": bootstrap_ci,
    }
    for filename in outputs:
        out_file = out_dir / filename
        if out_file.exists() and not overwrite:
            raise SystemExit(f"Output already exists; pass --overwrite: {out_file}")
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")
    summary_file = out_dir / "modeling_run_summary.json"
    if summary_file.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite: {summary_file}")
    summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    write_region_labels(labels, overwrite=overwrite, labels_file=labels_file)


def frames_equivalent_after_csv(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or left.shape != right.shape:
        return False
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError:
        return frames_equal_as_text(left, right)
    return True


def write_or_validate_frame(
    frame: pd.DataFrame,
    path: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    if path.exists() and resume:
        existing = pd.read_csv(path, keep_default_na=False, na_values=[""])
        if not frames_equivalent_after_csv(existing, frame):
            raise SystemExit(f"Existing output does not match deterministic recomputation: {path}")
        return
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --resume or --overwrite: {path}")
    write_dataframe_atomic(frame, path)


def snapshot_file_hashes(paths: Iterable[Path]) -> dict[str, str]:
    return {str(path): file_sha256(path) for path in paths if Path(path).is_file()}


def _stable_payload_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_checkpoint_manifest(
    *,
    args: argparse.Namespace,
    config: dict[str, object],
    frequency_bands: list[str],
    n_permutations: int,
    n_bootstrap: int,
) -> dict[str, object]:
    """Build the immutable fingerprint used to validate checkpoint resumption."""
    modeling = dict(config["modeling"])
    payload: dict[str, object] = {
        "run_id": str(config.get("run_id", "legacy_unspecified")),
        "average_precision_implementation": AVERAGE_PRECISION_IMPLEMENTATION,
        "average_precision_definition": AVERAGE_PRECISION_DEFINITION,
        "primary_metric": PRIMARY_REGION_METRIC,
        "primary_metric_scope": PRIMARY_REGION_METRIC_SCOPE,
        "univariate_orientation_rule": UNIVARIATE_ORIENTATION_RULE,
        "frequency_bands": frequency_bands,
        "top_k_features": int(modeling["top_k_features"]),
        "feature_non_missing_threshold": float(
            modeling["feature_non_missing_threshold"]
        ),
        "c_values": [float(value) for value in modeling["c_values"]],
        "random_seed": int(modeling["random_seed"]),
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
        "input_paths": {
            "feature_table": str(Path(args.features).resolve()),
            "audit_workbook": str(
                Path(str(config["audit_workbook"])).resolve()
            ),
            "config": str(Path(args.config).resolve()),
        },
        "input_sha256": {
            "feature_table": file_sha256(Path(args.features)),
            "audit_workbook": file_sha256(
                Path(str(config["audit_workbook"]))
            ),
            "config": file_sha256(Path(args.config)),
            "runner": file_sha256(Path(__file__)),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    return {**payload, "fingerprint_sha256": _stable_payload_sha256(payload)}


def write_or_validate_checkpoint_manifest(
    manifest: dict[str, object],
    path: Path,
    *,
    resume: bool,
) -> None:
    if path.exists():
        if not resume:
            raise SystemExit(
                f"Checkpoint manifest already exists; pass --resume: {path}"
            )
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint_sha256") != manifest.get(
            "fingerprint_sha256"
        ):
            raise SystemExit(
                "Checkpoint manifest fingerprint does not match the current "
                f"algorithm, inputs, configuration, or environment: {path}"
            )
        return
    if resume:
        raise SystemExit(f"Cannot resume without checkpoint manifest: {path}")
    write_json_atomic(manifest, path)


def run_ablation_rule_experiment(
    *,
    args: argparse.Namespace,
    config: dict[str, object],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    label_audit: pd.DataFrame,
    predictions: pd.DataFrame,
    fold_rows: list[dict[str, object]],
    observed_metrics: pd.DataFrame,
    candidate_coverage: dict[str, dict[str, int]],
    out_dir: Path,
    n_permutations: int,
    n_bootstrap: int,
    n_jobs: int,
    parallel_diagnostics: dict[str, object],
) -> None:
    assert_allowed_output_root(out_dir)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    if n_permutations < 1 or n_bootstrap < 1:
        raise SystemExit("The formal ablation-rule experiment requires positive permutation and bootstrap counts")

    labels_cfg = config["labels"]
    modeling = config["modeling"]
    configured_primary_metric = str(
        modeling.get("primary_metric", PRIMARY_REGION_METRIC)
    )
    if configured_primary_metric != PRIMARY_REGION_METRIC:
        raise SystemExit(
            "Unsupported regional primary metric: "
            f"{configured_primary_metric}; expected {PRIMARY_REGION_METRIC}"
        )
    frequency_bands = [str(value) for value in config["frequency_bands"]]
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = out_dir / "_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest = build_checkpoint_manifest(
        args=args,
        config=config,
        frequency_bands=frequency_bands,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
    )
    checkpoint_manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    write_or_validate_checkpoint_manifest(
        checkpoint_manifest,
        checkpoint_manifest_path,
        resume=args.resume,
    )
    observed_metrics_checkpoint = checkpoint_dir / "observed_metrics_by_band.csv"
    labels_file = out_dir / str(
        labels_cfg.get("output_filename", "region_labels_ablation_top4_rule.csv")
    )
    label_audit_file = out_dir / "label_derivation_audit.csv"
    prediction_file = out_dir / "lopo_region_predictions_by_band.csv"
    fold_file = out_dir / "lopo_region_fold_details_by_band.csv"
    patient_metrics_file = out_dir / "lopo_region_patient_metrics_by_band.csv"
    summary_file = out_dir / "modeling_run_summary_by_band.json"

    expected_existing_paths = [
        labels_file,
        label_audit_file,
        prediction_file,
        fold_file,
        patient_metrics_file,
        summary_file,
        observed_metrics_checkpoint,
        out_dir / "lopo_region_metrics_by_band.csv",
        out_dir / "permutation_results_by_band.csv",
        out_dir / "bootstrap_results_by_band.csv",
        out_dir / "bootstrap_ci_by_band.csv",
        out_dir / "region_band_multiple_comparison_correction.csv",
    ]
    if not (args.resume or args.overwrite):
        existing = [path for path in expected_existing_paths if path.exists()]
        existing.extend(checkpoint_dir.glob("permutations_*.csv") if checkpoint_dir.exists() else [])
        if existing:
            raise SystemExit(
                "New experiment outputs already exist; pass --resume or --overwrite: "
                f"{existing[0]}"
            )
    if args.overwrite and checkpoint_dir.exists():
        for checkpoint in checkpoint_dir.glob("permutations_*.csv"):
            checkpoint.unlink()

    fold_frame = pd.DataFrame.from_records(fold_rows)
    patient_metrics = compute_patient_prediction_metrics_by_band(predictions)
    write_or_validate_frame(labels, labels_file, args.resume, args.overwrite)
    write_or_validate_frame(label_audit, label_audit_file, args.resume, args.overwrite)
    write_or_validate_frame(predictions, prediction_file, args.resume, args.overwrite)
    write_or_validate_frame(fold_frame, fold_file, args.resume, args.overwrite)
    write_or_validate_frame(
        patient_metrics, patient_metrics_file, args.resume, args.overwrite
    )
    write_or_validate_frame(
        observed_metrics, observed_metrics_checkpoint, args.resume, args.overwrite
    )

    running_summary = {
        "status": "observed_complete_permutation_pending",
        "label_definition": "ablation_top4_rule",
        "config": str(args.config),
        "feature_table": str(args.features),
        "audit_workbook": str(config["audit_workbook"]),
        "input_sha256": {
            "feature_table": file_sha256(args.features),
            "audit_workbook": file_sha256(Path(str(config["audit_workbook"]))),
        },
        "n_label_rows": len(labels),
        "n_primary_rows": int(labels["inclusion_status"].eq("primary").sum()),
        "n_positive_rows": int(labels["label_binary"].fillna(0).sum()),
        "n_predictions": len(predictions),
        "n_folds": len(fold_rows),
        "primary_metric": PRIMARY_REGION_METRIC,
        "primary_metric_scope": PRIMARY_REGION_METRIC_SCOPE,
        "average_precision_implementation": AVERAGE_PRECISION_IMPLEMENTATION,
        "average_precision_definition": AVERAGE_PRECISION_DEFINITION,
        "metric_alias_note": "AUPRC columns use sklearn non-interpolated average precision.",
        "secondary_metric_scopes": [
            "within_patient_rank_then_pooled",
            "raw_score_pooled_across_outer_folds",
        ],
        "frequency_bands": frequency_bands,
        "network_methods": [str(value) for value in config["network_methods"]],
        "univariate_orientation_rule": UNIVARIATE_ORIENTATION_RULE,
        "candidate_feature_counts_by_band_and_method": candidate_coverage,
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "parallel": parallel_diagnostics,
        "label_rule": labels_cfg["rule"],
        "folds": fold_rows,
        "observed_metrics": observed_metrics.to_dict(orient="records"),
        "checkpoint_manifest": checkpoint_manifest,
        "command": [sys.executable, *sys.argv],
    }
    write_json_atomic(running_summary, summary_file)

    permutation = run_permutation_test_by_band_checkpointed(
        features,
        labels,
        observed_metrics=observed_metrics,
        frequency_bands=frequency_bands,
        n_permutations=n_permutations,
        seed=int(modeling["random_seed"]),
        top_k_features=int(modeling["top_k_features"]),
        c_values=[float(value) for value in modeling["c_values"]],
        non_missing_threshold=float(modeling["feature_non_missing_threshold"]),
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=int(modeling.get("permutation_checkpoint_every", 25)),
        resume=args.resume,
        overwrite=args.overwrite,
        n_jobs=n_jobs,
        blas_threads_per_worker=int(config["parallel"]["blas_threads_per_worker"]),
        max_tasks_per_child=int(config["parallel"]["max_tasks_per_child"]),
        minimum_free_memory_gb_during_run=float(
            config["parallel"]["minimum_free_memory_gb_during_run"]
        ),
    )
    metrics = add_permutation_p_values_by_band(observed_metrics, permutation)
    bootstrap = run_patient_bootstrap_by_band(
        predictions,
        n_bootstrap=n_bootstrap,
        seed=int(modeling["random_seed"]),
    )
    bootstrap_ci = ci_from_bootstrap_by_band(bootstrap)
    correction = compute_bonferroni_correction(metrics, permutation, frequency_bands)

    final_outputs = {
        "lopo_region_metrics_by_band.csv": metrics,
        "permutation_results_by_band.csv": permutation,
        "bootstrap_results_by_band.csv": bootstrap,
        "bootstrap_ci_by_band.csv": bootstrap_ci,
        "region_band_multiple_comparison_correction.csv": correction,
    }
    for filename, frame in final_outputs.items():
        write_dataframe_atomic(frame, out_dir / filename)

    final_summary = {
        **running_summary,
        "status": "complete",
        "permutation_test_status": "completed",
        "bootstrap_status": "completed",
        "multiple_comparison_method": "bonferroni_fwer_across_8_bands",
        "metrics": metrics.to_dict(orient="records"),
        "bootstrap_ci": bootstrap_ci.to_dict(orient="records"),
    }
    write_json_atomic(final_summary, summary_file)
    output_files = [path for path in out_dir.rglob("*") if path.is_file()]
    run_manifest = {
        **checkpoint_manifest,
        "status": "complete",
        "command": [sys.executable, *sys.argv],
        "output_sha256": snapshot_file_hashes(output_files),
    }
    write_json_atomic(run_manifest, out_dir / "run_manifest.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive")
    config = load_config(args.config)
    args.features = args.features or Path(str(config["region_features_root"])) / "region_patient_features_unlabeled.csv"
    modeling = config["modeling"]
    configured_orientation_rule = str(
        modeling.get("univariate_orientation_rule", UNIVARIATE_ORIENTATION_RULE)
    )
    if configured_orientation_rule != UNIVARIATE_ORIENTATION_RULE:
        raise SystemExit(
            "Unsupported univariate orientation rule: "
            f"{configured_orientation_rule}"
        )
    out_dir = (
        Path(args.out)
        if args.out is not None
        else Path(str(config.get("by_band_output_root", MODELING_BY_BAND_ROOT)))
    )
    if args.overwrite and _is_relative_to(_resolve(out_dir), _resolve(REGIONAL_ANALYSIS_ROOT)):
        raise SystemExit(
            "--overwrite is disabled under regional_analysis; use a new "
            "directory or --resume with an identical checkpoint fingerprint"
        )
    region_rows = load_region_summary_rows(Path(str(config["audit_workbook"])))
    labels_cfg = config["labels"]
    label_mode = str(labels_cfg.get("mode", "audit_status"))
    label_audit = pd.DataFrame()
    if label_mode == "audit_status":
        labels = build_label_table_from_region_rows(region_rows)
        labels_file = Path(str(config["region_features_root"])) / "region_labels.csv"
    elif label_mode == "ablation_top4_rule":
        labels, label_audit = build_ablation_top4_label_table(
            region_rows, dict(labels_cfg["rule"])
        )
        validate_ablation_rule_expectations(
            labels, label_audit, dict(labels_cfg["expected"])
        )
        labels_file = out_dir / str(
            labels_cfg.get("output_filename", "region_labels_ablation_top4_rule.csv")
        )
    else:
        raise SystemExit(f"Unsupported labels.mode: {label_mode}")
    if not args.features.exists():
        print(f"features_missing={args.features}")
        print(f"label_rows={len(labels)} primary_rows={(labels['inclusion_status'] == 'primary').sum()}")
        if not args.dry_run:
            raise SystemExit(f"Missing feature table: {args.features}")
        return
    features = pd.read_csv(args.features)
    frequency_bands = [str(value) for value in config["frequency_bands"]]
    network_methods = [str(value) for value in config["network_methods"]]
    features = select_feature_methods(features, network_methods)
    expected_per_method = config.get("expected_candidate_features_per_method_per_band")
    candidate_coverage = validate_feature_method_coverage(
        features,
        frequency_bands,
        network_methods,
        expected_per_method_per_band=None
        if expected_per_method is None
        else int(expected_per_method),
    )
    predictions, fold_rows = run_lopo_predictions_by_band(
        features,
        labels,
        frequency_bands=frequency_bands,
        top_k_features=int(modeling["top_k_features"]),
        c_values=[float(value) for value in modeling["c_values"]],
        random_seed=int(modeling["random_seed"]),
        non_missing_threshold=float(modeling["feature_non_missing_threshold"]),
    )
    metrics = compute_prediction_metrics_by_band(predictions)
    n_permutations = (
        int(modeling["n_permutations_default"]) if args.n_permutations is None else int(args.n_permutations)
    )
    n_bootstrap = int(modeling["n_bootstrap_default"]) if args.n_bootstrap is None else int(args.n_bootstrap)
    n_jobs = 1
    parallel_diagnostics: dict[str, object] = {}
    if label_mode == "ablation_top4_rule":
        parallel_config = dict(config["parallel"])
        requested_n_jobs = (
            int(parallel_config["default_n_jobs"])
            if args.n_jobs is None
            else int(args.n_jobs)
        )
        n_jobs, parallel_diagnostics = resolve_safe_worker_count(
            requested_n_jobs, parallel_config
        )
    print(f"label_rows={len(labels)} primary_rows={(labels['inclusion_status'] == 'primary').sum()}")
    print(f"frequency_bands={'|'.join(frequency_bands)}")
    print(f"prediction_rows={len(predictions)}")
    print(
        f"primary_metric={PRIMARY_REGION_METRIC} "
        f"metric_scope={PRIMARY_REGION_METRIC_SCOPE}"
    )
    print(metrics[["frequency_band", "AUPRC", "normalized_AUPRC", "AUROC"]].to_string(index=False))
    print(f"n_permutations={n_permutations} n_bootstrap={n_bootstrap}")
    print(f"label_mode={label_mode} out={out_dir}")
    if label_mode == "ablation_top4_rule":
        print(
            f"parallel_workers={n_jobs} requested={parallel_diagnostics['requested_n_jobs']} "
            f"physical_cores={parallel_diagnostics['detected_physical_cores']} "
            f"available_memory_gb={parallel_diagnostics['available_memory_gb_at_start']:.2f} "
            f"blas_threads_per_worker={parallel_diagnostics['blas_threads_per_worker']}"
        )
    if label_mode == "ablation_top4_rule":
        changed = label_audit.loc[label_audit["label_changed_from_audit_status"]]
        print(
            "derived_positive_rows="
            f"{int(labels['label_binary'].fillna(0).sum())} changed_rows={len(changed)}"
        )
        if not changed.empty:
            print(
                changed[
                    [
                        "patient",
                        "desikan_killiany_label",
                        "original_suspicious_region_status",
                        "derived_suspicious_region_status",
                    ]
                ].to_string(index=False)
            )
    if args.dry_run:
        return
    if label_mode == "ablation_top4_rule":
        run_ablation_rule_experiment(
            args=args,
            config=config,
            features=features,
            labels=labels,
            label_audit=label_audit,
            predictions=predictions,
            fold_rows=fold_rows,
            observed_metrics=metrics,
            candidate_coverage=candidate_coverage,
            out_dir=out_dir,
            n_permutations=n_permutations,
            n_bootstrap=n_bootstrap,
            n_jobs=n_jobs,
            parallel_diagnostics=parallel_diagnostics,
        )
        print(f"wrote={out_dir}")
        return
    if n_permutations > 0:
        permutation = run_permutation_test_by_band(
            features,
            labels,
            observed_metrics=metrics,
            frequency_bands=frequency_bands,
            n_permutations=n_permutations,
            seed=int(modeling["random_seed"]),
            top_k_features=int(modeling["top_k_features"]),
            c_values=[float(value) for value in modeling["c_values"]],
            non_missing_threshold=float(modeling["feature_non_missing_threshold"]),
        )
        metrics = add_permutation_p_values_by_band(metrics, permutation)
    else:
        permutation = pd.DataFrame(
            columns=[
                "frequency_band",
                "model_scope",
                "permutation_id",
                "seed",
                "primary_metric",
                "metric_scope",
                "AUPRC",
                "pooled_raw_AUPRC",
                "within_patient_rank_pooled_AUPRC",
                "null_ge_observed_AUPRC",
            ]
        )
        metrics["permutation_p_AUPRC"] = np.nan
    bootstrap = run_patient_bootstrap_by_band(
        predictions,
        n_bootstrap=n_bootstrap,
        seed=int(modeling["random_seed"]),
    )
    bootstrap_ci = ci_from_bootstrap_by_band(bootstrap)
    summary = {
        "config": str(args.config),
        "feature_table": str(args.features),
        "model_scope": "by_frequency_band",
        "frequency_bands": frequency_bands,
        "network_methods": network_methods,
        "primary_metric": PRIMARY_REGION_METRIC,
        "primary_metric_scope": PRIMARY_REGION_METRIC_SCOPE,
        "secondary_metric_scopes": [
            "within_patient_rank_then_pooled",
            "raw_score_pooled_across_outer_folds",
        ],
        "univariate_orientation_rule": UNIVARIATE_ORIENTATION_RULE,
        "candidate_feature_counts_by_band_and_method": candidate_coverage,
        "n_predictions": len(predictions),
        "n_folds": len(fold_rows),
        "n_permutations": n_permutations,
        "permutation_test_status": "not_run" if n_permutations == 0 else "completed",
        "folds": fold_rows,
        "metrics": metrics.to_dict(orient="records"),
    }
    write_outputs_by_band(
        out_dir,
        labels,
        predictions,
        metrics,
        permutation,
        bootstrap_ci,
        summary,
        args.overwrite,
        labels_file=labels_file,
    )
    print(f"wrote={out_dir}")


if __name__ == "__main__":
    main()
