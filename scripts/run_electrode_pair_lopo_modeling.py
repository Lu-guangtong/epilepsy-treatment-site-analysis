"""Run leakage-isolated, by-band LOPO classification of bipolar contacts."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import RobustScaler


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "electrode_pair_modeling.yaml"
FEATURE_ROOT = ROOT / "data_derived" / "electrode_pair_features"
MODELING_ROOT = ROOT / "data_derived" / "electrode_pair_modeling_by_band"
CHECK_ROOT = ROOT / "checks"
KEYS = ["patient", "bipolar_channel"]
NON_FEATURE_COLUMNS = {
    "patient",
    "bipolar_channel",
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
    "remove_as_bad_channel",
    "doctor_burned_and_evaluable",
    "doctor_unburned_and_evaluable",
    "label_binary",
    "inclusion_status",
}


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
    modeling_root = _resolve(MODELING_ROOT)
    feature_root = _resolve(FEATURE_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == modeling_root
        or _is_relative_to(out_dir, modeling_root)
        or out_dir == feature_root
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


def load_electrode_labels(audit_workbook: Path) -> pd.DataFrame:
    workbook = openpyxl.load_workbook(audit_workbook, read_only=True, data_only=True)
    rows: list[dict[str, object]] = []
    required = [
        "patient_id",
        "bipolar_channel",
        "is_white_csf_ventricle_unknown",
        "remove_as_bad_channel",
        "doctor_burned_and_evaluable",
        "doctor_unburned_and_evaluable",
    ]
    for sheet_name in workbook.sheetnames:
        if not sheet_name.endswith("_bipolar"):
            continue
        patient = sheet_name.removesuffix("_bipolar")
        ws = workbook[sheet_name]
        headers = header_map(ws)
        missing = [column for column in required if column not in headers]
        if missing:
            raise SystemExit(f"{sheet_name}: missing columns {missing}")
        for row_idx in range(2, ws.max_row + 1):
            channel = clean_text(ws.cell(row_idx, headers["bipolar_channel"]).value)
            if not channel:
                continue
            remove_bad = truthy(ws.cell(row_idx, headers["remove_as_bad_channel"]).value)
            is_wcu = truthy(
                ws.cell(row_idx, headers["is_white_csf_ventricle_unknown"]).value
            )
            burned = truthy(ws.cell(row_idx, headers["doctor_burned_and_evaluable"]).value)
            unburned = truthy(
                ws.cell(row_idx, headers["doctor_unburned_and_evaluable"]).value
            )
            if remove_bad:
                inclusion = "excluded_bad_channel"
                label = np.nan
            elif is_wcu:
                inclusion = "excluded_white_csf_ventricle_unknown"
                label = np.nan
            elif burned == unburned:
                raise SystemExit(
                    f"{sheet_name} row {row_idx}: primary label must be exactly one of burned/unburned"
                )
            else:
                inclusion = "primary"
                label = int(burned)
            rows.append(
                {
                    "patient": patient,
                    "bipolar_channel": channel,
                    "is_white_csf_ventricle_unknown": is_wcu,
                    "remove_as_bad_channel": remove_bad,
                    "doctor_burned_and_evaluable": burned,
                    "doctor_unburned_and_evaluable": unburned,
                    "label_binary": label,
                    "inclusion_status": inclusion,
                }
            )
    workbook.close()
    labels = pd.DataFrame.from_records(rows)
    if labels.duplicated(KEYS).any():
        raise SystemExit("Duplicate patient/bipolar_channel labels")
    return labels.sort_values(KEYS, kind="stable").reset_index(drop=True)


def feature_columns(features: pd.DataFrame) -> list[str]:
    return sorted(
        column
        for column in features.columns
        if column not in NON_FEATURE_COLUMNS
        and not column.startswith("qc__")
        and column.startswith("feature__")
        and pd.api.types.is_numeric_dtype(features[column])
    )


def feature_band(column: str) -> str | None:
    parts = str(column).split("__")
    return parts[1] if len(parts) >= 4 and parts[0] == "feature" else None


def feature_method(column: str) -> str | None:
    parts = str(column).split("__")
    return parts[2] if len(parts) >= 4 and parts[0] == "feature" else None


def feature_columns_for_band(features: pd.DataFrame, band: str) -> list[str]:
    return [column for column in feature_columns(features) if feature_band(column) == band]


def validate_candidate_coverage(
    features: pd.DataFrame,
    bands: Iterable[str],
    methods: Iterable[str],
    expected_per_method: int,
) -> dict[str, dict[str, int]]:
    expected_methods = [str(value) for value in methods]
    coverage: dict[str, dict[str, int]] = {}
    for band in [str(value) for value in bands]:
        counts = {method: 0 for method in expected_methods}
        for column in feature_columns_for_band(features, band):
            method = feature_method(column)
            if method not in counts:
                raise SystemExit(f"Unknown method feature for {band}: {column}")
            counts[str(method)] += 1
        if any(count != expected_per_method for count in counts.values()):
            raise SystemExit(f"Unexpected feature coverage for {band}: {counts}")
        coverage[band] = counts
    return coverage


def join_primary_features_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if features.duplicated(KEYS).any():
        raise SystemExit("Duplicate patient/bipolar_channel feature rows")
    primary = labels[labels["inclusion_status"] == "primary"].copy()
    merged = features.merge(primary, on=KEYS, how="inner", validate="one_to_one")
    if len(merged) != len(primary) or len(merged) != len(features):
        raise SystemExit(
            f"Feature/label primary rows do not align: features={len(features)} "
            f"labels={len(primary)} merged={len(merged)}"
        )
    return merged


def _safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(average_precision_score(y_true, scores))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(y_true, scores))
        if len(np.unique(y_true)) == 2
        else float("nan")
    )


def _average_precision_fast(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    n_positive = int(y_true.sum())
    if n_positive == 0 or n_positive == len(y_true):
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    cumulative_true = np.cumsum(sorted_y)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    group_starts = np.r_[0, group_ends[:-1] + 1]
    group_positives = np.add.reduceat(sorted_y, group_starts)
    precision = cumulative_true[group_ends] / (group_ends + 1)
    return float(np.sum((group_positives / n_positive) * precision))


def patient_macro_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    per_patient: list[dict[str, float]] = []
    for _, frame in predictions.groupby("patient", sort=True):
        y = frame["y_true"].to_numpy(dtype=int)
        score = frame["y_score"].to_numpy(dtype=float)
        prevalence = float(np.mean(y))
        ap = _safe_ap(y, score)
        normalized = (ap - prevalence) / (1.0 - prevalence) if prevalence < 1.0 else np.nan
        per_patient.append(
            {
                "AUPRC": ap,
                "normalized_AUPRC": float(normalized),
                "AUROC": _safe_auc(y, score),
            }
        )
    frame = pd.DataFrame.from_records(per_patient)
    return {
        "macro_AUPRC": float(frame["AUPRC"].mean()),
        "macro_normalized_AUPRC": float(frame["normalized_AUPRC"].mean()),
        "macro_AUROC": float(frame["AUROC"].mean()),
    }


def compute_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    y = predictions["y_true"].to_numpy(dtype=int)
    score = predictions["y_score"].to_numpy(dtype=float)
    pred = predictions["y_pred"].to_numpy(dtype=int)
    macro = patient_macro_metrics(predictions)
    prevalence = float(np.mean(y))
    pooled_ap = _safe_ap(y, score)
    pooled_normalized = (
        (pooled_ap - prevalence) / (1.0 - prevalence) if prevalence < 1.0 else np.nan
    )
    rank_rows: list[dict[str, float]] = []
    for _, frame in predictions.groupby("patient", sort=True):
        ordered = frame.sort_values("y_score", ascending=False, kind="stable")
        row: dict[str, float] = {"top1_hit": float(ordered.iloc[0]["y_true"])}
        for fraction in [0.10, 0.20]:
            n_top = max(1, int(np.ceil(len(ordered) * fraction)))
            top = ordered.head(n_top)
            row[f"precision_top{int(fraction * 100)}pct"] = float(top["y_true"].mean())
            row[f"recall_top{int(fraction * 100)}pct"] = float(
                top["y_true"].sum() / ordered["y_true"].sum()
            )
        rank_rows.append(row)
    ranking = pd.DataFrame.from_records(rank_rows).mean().to_dict()
    positives = y == 1
    negatives = y == 0
    return {
        **macro,
        "pooled_AUPRC": pooled_ap,
        "pooled_normalized_AUPRC": float(pooled_normalized),
        "pooled_AUROC": _safe_auc(y, score),
        "prevalence": prevalence,
        "F1": float(f1_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(np.mean(pred[positives] == 1)),
        "specificity": float(np.mean(pred[negatives] == 0)),
        **{key: float(value) for key, value in ranking.items()},
    }


def _fit_predict(
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
        max_iter=2000,
    )
    model.fit(x_train, train["label_binary"].astype(int))
    return model.predict_proba(x_test)[:, 1]


def score_features_inner_macro_auprc(
    train: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    score_sums = np.zeros(len(columns), dtype=np.float64)
    score_counts = np.zeros(len(columns), dtype=int)
    for heldout in sorted(train["patient"].unique()):
        inner_train = train[train["patient"] != heldout]
        inner_test = train[train["patient"] == heldout]
        y_train = inner_train["label_binary"].to_numpy(dtype=int)
        y_test = inner_test["label_binary"].to_numpy(dtype=int)
        x_train = inner_train[columns].to_numpy(dtype=float)
        x_test = inner_test[columns].to_numpy(dtype=float)
        with np.errstate(all="ignore"):
            fill = np.nanmedian(x_train, axis=0)
            positive_median = np.nanmedian(x_train[y_train == 1], axis=0)
            negative_median = np.nanmedian(x_train[y_train == 0], axis=0)
        valid = np.isfinite(fill) & np.isfinite(positive_median) & np.isfinite(negative_median)
        x_test = np.where(np.isfinite(x_test), x_test, fill[np.newaxis, :])
        direction = np.where(positive_median >= negative_median, 1.0, -1.0)
        oriented = x_test * direction[np.newaxis, :]
        for index in np.flatnonzero(valid):
            score = _average_precision_fast(y_test, oriented[:, index])
            if np.isfinite(score):
                score_sums[index] += score
                score_counts[index] += 1
    return pd.DataFrame.from_records(
        [
            {
                "feature": column,
                "inner_macro_AUPRC": (
                    float(score_sums[index] / score_counts[index])
                    if score_counts[index]
                    else float("-inf")
                ),
            }
            for index, column in enumerate(columns)
        ]
    ).sort_values(["inner_macro_AUPRC", "feature"], ascending=[False, True])


def select_uncorrelated_top_features(
    train: pd.DataFrame,
    score_table: pd.DataFrame,
    top_k: int,
    correlation_threshold: float,
) -> tuple[list[str], list[dict[str, object]]]:
    selected: list[str] = []
    dropped: list[dict[str, object]] = []
    for feature in score_table["feature"]:
        if len(selected) >= top_k:
            break
        candidate = train[feature].astype(float)
        correlated_with = None
        correlated_rho = np.nan
        for kept in selected:
            rho = candidate.corr(train[kept].astype(float), method="spearman")
            if np.isfinite(rho) and abs(rho) >= correlation_threshold:
                correlated_with = kept
                correlated_rho = float(rho)
                break
        if correlated_with is None:
            selected.append(str(feature))
        else:
            dropped.append(
                {
                    "feature": str(feature),
                    "correlated_with": correlated_with,
                    "spearman_rho": correlated_rho,
                }
            )
    if len(selected) < top_k:
        raise SystemExit(f"Only {len(selected)} uncorrelated eligible features; need {top_k}")
    return selected, dropped


def inner_oof_predictions(
    train: pd.DataFrame,
    columns: list[str],
    c_value: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for heldout in sorted(train["patient"].unique()):
        inner_train = train[train["patient"] != heldout]
        inner_test = train[train["patient"] == heldout]
        scores = _fit_predict(inner_train, inner_test, columns, c_value)
        for patient, y_true, score in zip(
            inner_test["patient"],
            inner_test["label_binary"].astype(int),
            scores,
            strict=True,
        ):
            rows.append(
                {
                    "patient": patient,
                    "y_true": int(y_true),
                    "y_score": float(score),
                }
            )
    output = pd.DataFrame.from_records(rows)
    output["y_pred"] = (output["y_score"] >= 0.5).astype(int)
    return output


def select_threshold_max_f1(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y_true[order]
    sorted_scores = scores[order]
    cumulative_true = np.cumsum(sorted_y)
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    true_positive = cumulative_true[group_ends]
    predicted_positive = group_ends + 1
    false_positive = predicted_positive - true_positive
    false_negative = int(y_true.sum()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1_values = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=float),
        where=denominator > 0,
    )
    best_f1 = np.max(f1_values)
    candidate_thresholds = sorted_scores[group_ends][f1_values == best_f1]
    return float(np.max(candidate_thresholds))


def select_features_c_threshold(
    train: pd.DataFrame,
    candidates: list[str],
    top_k: int,
    c_values: Iterable[float],
    non_missing_threshold: float,
    correlation_threshold: float,
) -> tuple[list[str], float, float, pd.DataFrame, list[dict[str, object]]]:
    matrix = train[candidates].to_numpy(dtype=float)
    non_missing = np.mean(np.isfinite(matrix), axis=0)
    with np.errstate(all="ignore"):
        minima = np.nanmin(matrix, axis=0)
        maxima = np.nanmax(matrix, axis=0)
    eligible = [
        column
        for column, valid, minimum, maximum in zip(
            candidates, non_missing, minima, maxima, strict=True
        )
        if valid >= non_missing_threshold
        and np.isfinite(minimum)
        and np.isfinite(maximum)
        and minimum != maximum
    ]
    if not eligible:
        raise SystemExit("No eligible features after missingness/variance filtering")
    score_table = score_features_inner_macro_auprc(train, eligible)
    selected, dropped = select_uncorrelated_top_features(
        train,
        score_table,
        top_k=top_k,
        correlation_threshold=correlation_threshold,
    )
    c_rows: list[dict[str, float]] = []
    oof_by_c: dict[float, pd.DataFrame] = {}
    for c_value in sorted(float(value) for value in c_values):
        oof = inner_oof_predictions(train, selected, c_value)
        oof_by_c[c_value] = oof
        c_rows.append(
            {
                "C": c_value,
                "inner_macro_AUPRC": patient_macro_metrics(oof)["macro_AUPRC"],
            }
        )
    c_table = pd.DataFrame.from_records(c_rows).sort_values(
        ["inner_macro_AUPRC", "C"], ascending=[False, True]
    )
    selected_c = float(c_table.iloc[0]["C"])
    oof = oof_by_c[selected_c]
    threshold = select_threshold_max_f1(
        oof["y_true"].to_numpy(dtype=int), oof["y_score"].to_numpy(dtype=float)
    )
    return selected, selected_c, threshold, score_table, dropped


def run_lopo_by_band(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    bands: Iterable[str],
    top_k: int,
    c_values: Iterable[float],
    non_missing_threshold: float,
    correlation_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = join_primary_features_labels(features, labels)
    predictions: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for band in [str(value) for value in bands]:
        candidates = feature_columns_for_band(data, band)
        if not candidates:
            raise SystemExit(f"No candidate features for {band}")
        for heldout in sorted(data["patient"].unique()):
            train = data[data["patient"] != heldout].copy()
            test = data[data["patient"] == heldout].copy()
            selected, c_value, threshold, score_table, dropped = select_features_c_threshold(
                train,
                candidates,
                top_k=top_k,
                c_values=c_values,
                non_missing_threshold=non_missing_threshold,
                correlation_threshold=correlation_threshold,
            )
            if any(feature_band(column) != band for column in selected):
                raise SystemExit(f"Cross-band feature selected for {band}: {selected}")
            scores = _fit_predict(train, test, selected, c_value)
            ranks = pd.Series(scores).rank(method="first", ascending=False).astype(int).to_numpy()
            for patient, channel, region, y_true, score, rank in zip(
                test["patient"],
                test["bipolar_channel"],
                test["desikan_killiany_label"],
                test["label_binary"].astype(int),
                scores,
                ranks,
                strict=True,
            ):
                predictions.append(
                    {
                        "frequency_band": band,
                        "heldout_patient": heldout,
                        "patient": patient,
                        "bipolar_channel": channel,
                        "desikan_killiany_label": region,
                        "y_true": int(y_true),
                        "y_score": float(score),
                        "y_pred": int(score >= threshold),
                        "rank_within_patient": int(rank),
                        "selected_features": "|".join(selected),
                        "selected_C": c_value,
                        "selected_threshold": threshold,
                    }
                )
            folds.append(
                {
                    "frequency_band": band,
                    "heldout_patient": heldout,
                    "train_patients": "|".join(sorted(train["patient"].unique())),
                    "selected_features": "|".join(selected),
                    "selected_C": c_value,
                    "selected_threshold": threshold,
                    "feature_score_table": score_table.to_json(orient="records"),
                    "correlation_drops": json.dumps(dropped, ensure_ascii=False),
                }
            )
    return pd.DataFrame.from_records(predictions), pd.DataFrame.from_records(folds)


def metrics_by_band(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [
            {"frequency_band": band, **compute_metrics(frame)}
            for band, frame in predictions.groupby("frequency_band", sort=False)
        ]
    )


def permute_labels_within_patient(
    labels: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    output = labels.copy()
    for _, indices in output.groupby("patient").groups.items():
        primary = output.loc[indices]
        primary_indices = primary[primary["inclusion_status"] == "primary"].index
        values = output.loc[primary_indices, "label_binary"].to_numpy(dtype=float)
        output.loc[primary_indices, "label_binary"] = rng.permutation(values)
    return output


def _run_permutation_chunk(
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
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for permutation_id in permutation_ids:
        permutation_seed = seed + permutation_id
        permuted = permute_labels_within_patient(
            labels, np.random.default_rng(permutation_seed)
        )
        predictions, _ = run_lopo_by_band(
            features,
            permuted,
            bands,
            top_k,
            c_values,
            non_missing_threshold,
            correlation_threshold,
        )
        metrics = metrics_by_band(predictions)
        null_max = float(metrics["macro_AUPRC"].max())
        for row in metrics.to_dict("records"):
            band = str(row["frequency_band"])
            value = float(row["macro_AUPRC"])
            rows.append(
                {
                    "permutation_id": permutation_id,
                    "seed": permutation_seed,
                    "frequency_band": band,
                    "macro_AUPRC": value,
                    "max_macro_AUPRC_across_bands": null_max,
                    "null_ge_observed": value >= float(observed[band]),
                    "max_null_ge_observed": null_max >= float(observed[band]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _validate_permutation_checkpoint(
    frame: pd.DataFrame,
    permutation_ids: list[int],
    bands: list[str],
) -> None:
    required = {
        "permutation_id",
        "seed",
        "frequency_band",
        "macro_AUPRC",
        "max_macro_AUPRC_across_bands",
        "null_ge_observed",
        "max_null_ge_observed",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise SystemExit(f"Permutation checkpoint missing columns: {sorted(missing)}")
    expected_rows = len(permutation_ids) * len(bands)
    if len(frame) != expected_rows:
        raise SystemExit(
            f"Permutation checkpoint row mismatch: expected={expected_rows} got={len(frame)}"
        )
    expected_pairs = {(int(pid), band) for pid in permutation_ids for band in bands}
    actual_pairs = {
        (int(row.permutation_id), str(row.frequency_band))
        for row in frame[["permutation_id", "frequency_band"]].itertuples(index=False)
    }
    if actual_pairs != expected_pairs:
        raise SystemExit("Permutation checkpoint task identities do not match its chunk")
    numeric = frame[["macro_AUPRC", "max_macro_AUPRC_across_bands"]].to_numpy(
        dtype=float
    )
    if not np.isfinite(numeric).all():
        raise SystemExit("Permutation checkpoint contains non-finite metrics")


def _run_permutation_chunk_to_checkpoint(
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
) -> str:
    frame = _run_permutation_chunk(
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
    )
    _validate_permutation_checkpoint(frame, permutation_ids, bands)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_file.with_suffix(".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(checkpoint_file)
    return str(checkpoint_file)


def run_permutations(
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
    n_jobs: int = 1,
    checkpoint_dir: Path | None = None,
    checkpoint_chunk_size: int = 10,
) -> pd.DataFrame:
    if n_permutations <= 0:
        return pd.DataFrame()
    if n_jobs < 1:
        raise SystemExit(f"n_jobs must be >= 1, got {n_jobs}")
    if checkpoint_chunk_size < 1:
        raise SystemExit(
            f"checkpoint_chunk_size must be >= 1, got {checkpoint_chunk_size}"
        )
    if checkpoint_dir is None:
        checkpoint_dir = CHECK_ROOT / "electrode_pair_permutation_checkpoints"
    assert_allowed_output_root(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    observed = observed_metrics.set_index("frequency_band")["macro_AUPRC"].to_dict()
    permutation_ids = list(range(1, n_permutations + 1))
    chunks = [
        permutation_ids[start : start + checkpoint_chunk_size]
        for start in range(0, len(permutation_ids), checkpoint_chunk_size)
    ]
    chunk_files = {
        tuple(chunk): checkpoint_dir
        / f"permutations_{chunk[0]:04d}_{chunk[-1]:04d}.csv"
        for chunk in chunks
    }
    pending: list[list[int]] = []
    for chunk in chunks:
        path = chunk_files[tuple(chunk)]
        if not path.exists():
            pending.append(chunk)
            continue
        _validate_permutation_checkpoint(pd.read_csv(path), chunk, bands)
    if pending and n_jobs == 1:
        for index, chunk in enumerate(pending, start=1):
            _run_permutation_chunk_to_checkpoint(
                chunk,
                chunk_files[tuple(chunk)],
                features,
                labels,
                bands,
                observed,
                seed,
                top_k,
                c_values,
                non_missing_threshold,
                correlation_threshold,
            )
            print(f"permutation_checkpoint={index}/{len(pending)}")
    elif pending:
        # Each process handles one short batch and exits. This avoids long-lived
        # Windows workers while preserving the exact full permutation procedure.
        with ProcessPoolExecutor(
            max_workers=min(n_jobs, len(pending)), max_tasks_per_child=1
        ) as executor:
            futures = {
                executor.submit(
                    _run_permutation_chunk_to_checkpoint,
                    chunk,
                    chunk_files[tuple(chunk)],
                    features,
                    labels,
                    bands,
                    observed,
                    seed,
                    top_k,
                    c_values,
                    non_missing_threshold,
                    correlation_threshold,
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
                        "Permutation worker failed after completed checkpoints were "
                        f"saved; rerun to resume. Failed chunk={chunk[0]}-{chunk[-1]}: "
                        f"{error}"
                    ) from error
                completed += 1
                print(f"permutation_checkpoint={completed}/{len(pending)}")
    frames: list[pd.DataFrame] = []
    for chunk in chunks:
        path = chunk_files[tuple(chunk)]
        if not path.exists():
            raise SystemExit(f"Missing completed permutation checkpoint: {path}")
        frame = pd.read_csv(path)
        _validate_permutation_checkpoint(frame, chunk, bands)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["permutation_id", "frequency_band"], kind="stable"
    ).reset_index(drop=True)


def add_permutation_p_values(
    metrics: pd.DataFrame,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for band, frame in permutation.groupby("frequency_band", sort=False):
        rows.append(
            {
                "frequency_band": band,
                "permutation_p_raw_macro_AUPRC": (
                    int(frame["null_ge_observed"].sum()) + 1
                )
                / (len(frame) + 1),
                "permutation_p_fwer_maxT_macro_AUPRC": (
                    int(frame["max_null_ge_observed"].sum()) + 1
                )
                / (len(frame) + 1),
            }
        )
    return metrics.merge(pd.DataFrame.from_records(rows), on="frequency_band", how="left")


def run_patient_bootstrap(
    predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    frames: list[dict[str, object]] = []
    for band_index, (band, band_frame) in enumerate(
        predictions.groupby("frequency_band", sort=False), start=1
    ):
        patients = sorted(band_frame["patient"].unique())
        band_rng = np.random.default_rng(seed + band_index * 100000)
        for bootstrap_id in range(1, n_bootstrap + 1):
            sampled = band_rng.choice(patients, size=len(patients), replace=True)
            samples: list[pd.DataFrame] = []
            for sample_index, patient in enumerate(sampled):
                frame = band_frame[band_frame["patient"] == patient].copy()
                frame["patient"] = f"{patient}__bootstrap{sample_index}"
                samples.append(frame)
            metrics = compute_metrics(pd.concat(samples, ignore_index=True))
            frames.append(
                {
                    "frequency_band": band,
                    "bootstrap_id": bootstrap_id,
                    **metrics,
                }
            )
    return pd.DataFrame.from_records(frames)


def bootstrap_ci(bootstrap: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ["macro_AUPRC", "macro_normalized_AUPRC", "macro_AUROC"]
    for band, frame in bootstrap.groupby("frequency_band", sort=False):
        for metric in metrics:
            values = frame[metric].dropna()
            rows.append(
                {
                    "frequency_band": band,
                    "metric": metric,
                    "ci_low": float(values.quantile(0.025)),
                    "ci_high": float(values.quantile(0.975)),
                }
            )
    return pd.DataFrame.from_records(rows)


def write_outputs(
    out_dir: Path,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    permutation: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bootstrap_intervals: pd.DataFrame,
    summary: dict[str, object],
    overwrite: bool,
) -> None:
    assert_allowed_output_root(out_dir)
    outputs = {
        "lopo_electrode_predictions_by_band.csv": predictions,
        "lopo_electrode_metrics_by_band.csv": metrics,
        "lopo_electrode_fold_details.csv": folds,
        "permutation_results_by_band.csv": permutation,
        "bootstrap_results_by_band.csv": bootstrap,
        "bootstrap_ci_by_band.csv": bootstrap_intervals,
    }
    resolved_out = _resolve(out_dir)
    resolved_tmp = _resolve(CHECK_ROOT)
    labels_file = (
        out_dir / "electrode_pair_labels.csv"
        if resolved_out == resolved_tmp or _is_relative_to(resolved_out, resolved_tmp)
        else FEATURE_ROOT / "electrode_pair_labels.csv"
    )
    assert_allowed_output_root(labels_file.parent)
    paths = [out_dir / filename for filename in outputs]
    paths.extend([out_dir / "electrode_modeling_run_summary.json", labels_file])
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite: {existing[0]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_file.parent.mkdir(parents=True, exist_ok=True)
    for filename, frame in outputs.items():
        frame.to_csv(out_dir / filename, index=False, encoding="utf-8-sig")
    labels.to_csv(labels_file, index=False, encoding="utf-8-sig")
    (out_dir / "electrode_modeling_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument(
        "--features",
        type=Path,
        default=FEATURE_ROOT / "electrode_patient_features_unlabeled.csv",
    )
    parser.add_argument("--out", type=Path, default=MODELING_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--n-permutations", type=int, default=None)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--permutation-checkpoint-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    assert_allowed_output_root(args.out)
    labels = load_electrode_labels(Path(str(config["audit_workbook"])))
    primary = labels[labels["inclusion_status"] == "primary"]
    expected = config["expected"]
    positive_count = int(primary["label_binary"].sum())
    negative_count = int((primary["label_binary"] == 0).sum())
    if (
        len(primary) != int(expected["primary_bipolar_pairs"])
        or positive_count != int(expected["positive_pairs"])
        or negative_count != int(expected["negative_pairs"])
    ):
        raise SystemExit(
            f"Unexpected primary labels: rows={len(primary)} positive={positive_count} "
            f"negative={negative_count}"
        )
    print(
        f"label_rows={len(labels)} primary={len(primary)} "
        f"positive={int(primary['label_binary'].sum())} negative={int((primary['label_binary'] == 0).sum())}"
    )
    if not args.features.exists():
        if args.dry_run:
            print(f"features_missing={args.features}")
            return
        raise SystemExit(f"Missing feature table: {args.features}")
    features = pd.read_csv(args.features)
    if len(features) != int(expected["patient_feature_rows"]):
        raise SystemExit(f"Unexpected feature rows: {len(features)}")
    bands = [str(value) for value in config["frequency_bands"]]
    methods = [str(value) for value in config["methods"]]
    expected_per_method = int(config["features"]["expected_candidates_per_method_per_band"])
    coverage = validate_candidate_coverage(features, bands, methods, expected_per_method)
    modeling = config["modeling"]
    predictions, folds = run_lopo_by_band(
        features,
        labels,
        bands,
        top_k=int(modeling["top_k_features"]),
        c_values=[float(value) for value in modeling["c_values"]],
        non_missing_threshold=float(modeling["feature_non_missing_threshold"]),
        correlation_threshold=float(modeling["correlation_pruning_spearman_abs"]),
    )
    metrics = metrics_by_band(predictions)
    print(metrics[["frequency_band", "macro_AUPRC", "macro_normalized_AUPRC", "macro_AUROC"]].to_string(index=False))
    print(f"prediction_rows={len(predictions)} fold_rows={len(folds)}")
    if args.dry_run:
        return
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
    if n_permutations > 0:
        permutation = run_permutations(
            features,
            labels,
            bands,
            metrics,
            n_permutations,
            int(modeling["random_seed"]),
            int(modeling["top_k_features"]),
            [float(value) for value in modeling["c_values"]],
            float(modeling["feature_non_missing_threshold"]),
            float(modeling["correlation_pruning_spearman_abs"]),
            n_jobs=n_jobs,
            checkpoint_dir=args.out / ".permutation_checkpoints",
            checkpoint_chunk_size=int(args.permutation_checkpoint_size),
        )
        metrics = add_permutation_p_values(metrics, permutation)
    else:
        permutation = pd.DataFrame()
        metrics["permutation_p_raw_macro_AUPRC"] = np.nan
        metrics["permutation_p_fwer_maxT_macro_AUPRC"] = np.nan
    bootstrap = run_patient_bootstrap(
        predictions,
        n_bootstrap=n_bootstrap,
        seed=int(modeling["random_seed"]),
    )
    intervals = bootstrap_ci(bootstrap) if not bootstrap.empty else pd.DataFrame()
    if len(predictions) != int(expected["prediction_rows"]):
        raise SystemExit(f"Unexpected prediction rows: {len(predictions)}")
    summary = {
        "config": str(args.config),
        "feature_table": str(args.features),
        "model_scope": "by_frequency_band_two_methods",
        "primary_metric": "patient_macro_AUPRC",
        "frequency_bands": bands,
        "methods": methods,
        "candidate_coverage": coverage,
        "n_primary_pairs": len(primary),
        "n_predictions": len(predictions),
        "n_outer_folds": len(folds),
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "permutation_n_jobs": n_jobs,
        "metrics": metrics.to_dict(orient="records"),
        "heldout_patient_used_in_training": False,
        "region_score_used": False,
    }
    write_outputs(
        args.out,
        labels,
        predictions,
        metrics,
        folds,
        permutation,
        bootstrap,
        intervals,
        summary,
        overwrite=args.overwrite,
    )
    print(f"wrote={args.out}")


if __name__ == "__main__":
    main()
