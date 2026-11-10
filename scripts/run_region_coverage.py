"""Compare regional connectivity with electrode-coverage baselines."""

from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Iterable, Sequence

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
import sklearn


ROOT = output_root()
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = config_root() / "regional_analysis" / "coverage_models.yaml"
ANALYSIS_ROOT = ROOT / "data_derived" / "regional_analysis"
TMP_ANALYSIS_ROOT = ROOT / "checks" / "regional_analysis"
REGION_KEYS = [
    "patient",
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
]
MODEL_ORDER = [
    "coverage_only",
    "connectivity_corrected",
    "coverage_plus_residual_connectivity",
]
METRIC_DEFINITION = "sklearn_average_precision_not_trapezoidal_pr_auc"
ALGORITHM_VERSION = "regional_coverage_v1"
AP_IMPLEMENTATION = "sklearn_equivalent_tie_grouped_v2"


@dataclass
class SplitArrays:
    train_idx: np.ndarray
    test_idx: np.ndarray
    raw_train: np.ndarray
    raw_test: np.ndarray
    residual_train: np.ndarray
    residual_test: np.ndarray


@dataclass
class OuterDesign:
    heldout_patient: str
    eligible: np.ndarray
    outer: SplitArrays
    inner: list[SplitArrays]


@dataclass
class BandDesign:
    band: str
    candidates: list[str]
    outers: list[OuterDesign]


_PERMUTATION_STATE: dict[str, object] = {}


def load_yaml(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = expand_config(yaml.safe_load(handle))
    if not isinstance(value, dict):
        raise SystemExit(f"Expected a YAML mapping: {path}")
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def assert_allowed_output_root(path: Path) -> None:
    assert_output_path(path)
    resolved = Path(path).resolve()
    allowed = [ANALYSIS_ROOT.resolve(), TMP_ANALYSIS_ROOT.resolve()]
    if not any(resolved == root or _is_relative_to(resolved, root) for root in allowed):
        raise SystemExit(f"Refusing to write outside analysis directories: {resolved}")
    forbidden = [
        source_root("data").resolve(),
        (ROOT / "data_derived" / "regional_analysis" / "region_models").resolve(),
        (ROOT / "data_derived" / "region_features").resolve(),
    ]
    if any(resolved == root or _is_relative_to(resolved, root) for root in forbidden):
        raise SystemExit(f"Refusing to write into a protected input directory: {resolved}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    if not path.exists():
        digest.update(b"MISSING")
        return digest.hexdigest()
    if path.is_file():
        return file_sha256(path)
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(child.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(child).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _atomic_path(path: Path) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    fd, tmp_name = _atomic_path(path)
    os.close(fd)
    try:
        frame.to_csv(tmp_name, index=False)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def atomic_write_json(value: object, path: Path) -> None:
    fd, tmp_name = _atomic_path(path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _feature_band(column: str) -> str | None:
    parts = str(column).split("__")
    if len(parts) < 3:
        return None
    if parts[0] == "trial" and len(parts) >= 4:
        return parts[2]
    if parts[0] == "contrast":
        return parts[1]
    return None


def connectivity_columns_for_band(frame: pd.DataFrame, band: str) -> list[str]:
    excluded = set(REGION_KEYS) | {
        "label_binary",
        "suspicious_region_status",
        "inclusion_status",
        "n_bipolar_pairs_after_bad_channel_removal",
    }
    columns = [
        column
        for column in frame.columns
        if column not in excluded
        and not column.startswith("qc__")
        and pd.api.types.is_numeric_dtype(frame[column])
        and _feature_band(column) == band
    ]
    return sorted(columns)


def add_coverage_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    output = frame.copy()
    node_columns = [
        column
        for column in output.columns
        if column.startswith("qc__") and column.endswith("__n_region_nodes_max")
    ]
    if not node_columns:
        raise ValueError("No qc__*__n_region_nodes_max columns were found")
    node_matrix = output[node_columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    for row_index, row in enumerate(node_matrix):
        values = row[np.isfinite(row)]
        if not len(values):
            raise ValueError(f"Row {row_index} has no finite region-node coverage")
        if len(np.unique(values)) != 1:
            raise ValueError(f"Row {row_index} has inconsistent region-node coverage fields")
    nodes = np.nanmax(node_matrix, axis=1)
    if np.any(nodes <= 0):
        raise ValueError("Primary patient-region rows require positive node coverage")
    output["coverage__n_region_nodes"] = nodes
    totals = output.groupby("patient", sort=False)["coverage__n_region_nodes"].transform("sum")
    output["coverage__fraction_patient_nodes"] = nodes / totals
    output["coverage__log1p_region_nodes"] = np.log1p(nodes)
    output["coverage__n_within_edges"] = nodes * (nodes - 1.0) / 2.0
    output["coverage__n_out_edges"] = nodes * (totals - nodes)
    total_pairs = totals * (totals - 1.0) / 2.0
    output["coverage__fraction_patient_pairs_within"] = (
        output["coverage__n_within_edges"] / total_pairs
    )
    output["coverage__fraction_patient_pairs_out"] = output["coverage__n_out_edges"] / total_pairs
    coverage_columns = sorted(column for column in output if column.startswith("coverage__"))
    if len(coverage_columns) != 7:
        raise ValueError(f"Expected seven coverage functions, found {len(coverage_columns)}")
    if not np.isfinite(output[coverage_columns].to_numpy(float)).all():
        raise ValueError("Coverage functions contain non-finite values")
    output["coverage_rank_percentile"] = output.groupby("patient", sort=False)[
        "coverage__n_region_nodes"
    ].rank(method="average", pct=True, ascending=True)
    patient_median = output.groupby("patient", sort=False)["coverage__n_region_nodes"].transform("median")
    output["coverage_layer"] = np.where(nodes <= patient_median, "low", "high")
    return output, coverage_columns


def average_precision_columns_exact(y_true: np.ndarray, score_matrix: np.ndarray) -> np.ndarray:
    """Vectorized sklearn-equivalent AP with tied scores settled as groups."""
    y = np.asarray(y_true, dtype=int)
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
    if len(y) != scores.shape[0]:
        raise ValueError("Label and score row counts differ")
    n_positive = int(y.sum())
    if n_positive == 0 or n_positive == len(y):
        return np.full(scores.shape[1], np.nan)
    if not np.isfinite(scores).all():
        raise ValueError("AP score matrix must be finite")
    order = np.argsort(-scores, axis=0, kind="stable")
    sorted_scores = np.take_along_axis(scores, order, axis=0)
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y, axis=0, dtype=float)
    boundary = np.ones(sorted_scores.shape, dtype=bool)
    if len(y) > 1:
        boundary[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    previous_boundary_tp = np.zeros_like(cumulative_tp)
    if len(y) > 1:
        previous_boundary_tp[1:] = np.where(boundary[:-1], cumulative_tp[:-1], 0.0)
    previous_boundary_tp = np.maximum.accumulate(previous_boundary_tp, axis=0)
    group_positive = cumulative_tp - previous_boundary_tp
    ranks = np.arange(1, len(y) + 1, dtype=float)[:, np.newaxis]
    precision = cumulative_tp / ranks
    contributions = np.where(boundary, (group_positive / n_positive) * precision, 0.0)
    return contributions.sum(axis=0)


def _median_iqr_transform(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.asarray(train, dtype=float)
    test = np.asarray(test, dtype=float)
    with np.errstate(all="ignore"):
        fill = np.nanmedian(train, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0)
    train_filled = np.where(np.isfinite(train), train, fill[np.newaxis, :])
    test_filled = np.where(np.isfinite(test), test, fill[np.newaxis, :])
    center = np.median(train_filled, axis=0)
    q25 = np.quantile(train_filled, 0.25, axis=0)
    q75 = np.quantile(train_filled, 0.75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    return (train_filled - center) / scale, (test_filled - center) / scale


def make_split_arrays(
    connectivity: np.ndarray,
    coverage: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> SplitArrays:
    raw_train, raw_test = _median_iqr_transform(connectivity[train_idx], connectivity[test_idx])
    cov_train, cov_test = _median_iqr_transform(coverage[train_idx], coverage[test_idx])
    target_train, target_test = _median_iqr_transform(connectivity[train_idx], connectivity[test_idx])
    ridge = Ridge(alpha=1.0, fit_intercept=True, solver="svd")
    ridge.fit(cov_train, target_train)
    residual_train = target_train - ridge.predict(cov_train)
    residual_test = target_test - ridge.predict(cov_test)
    residual_train, residual_test = _median_iqr_transform(residual_train, residual_test)
    return SplitArrays(
        train_idx=np.asarray(train_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        raw_train=np.asarray(raw_train, dtype=np.float32),
        raw_test=np.asarray(raw_test, dtype=np.float32),
        residual_train=np.asarray(residual_train, dtype=np.float32),
        residual_test=np.asarray(residual_test, dtype=np.float32),
    )


def build_band_designs(
    data: pd.DataFrame,
    bands: Sequence[str],
    coverage_columns: Sequence[str],
    non_missing_threshold: float,
    expected_candidates_per_band: int | None = None,
) -> dict[str, BandDesign]:
    patients = data["patient"].astype(str).to_numpy()
    coverage = data[list(coverage_columns)].to_numpy(float)
    designs: dict[str, BandDesign] = {}
    for band in bands:
        candidates = connectivity_columns_for_band(data, band)
        if not candidates:
            raise ValueError(f"No connectivity candidates for band {band}")
        if expected_candidates_per_band is not None and len(candidates) != int(expected_candidates_per_band):
            raise ValueError(
                f"{band}: expected {expected_candidates_per_band} candidates, found {len(candidates)}"
            )
        matrix = data[candidates].to_numpy(float)
        outers: list[OuterDesign] = []
        for heldout in sorted(set(patients.tolist())):
            train_idx = np.flatnonzero(patients != heldout)
            test_idx = np.flatnonzero(patients == heldout)
            train_matrix = matrix[train_idx]
            non_missing = np.mean(np.isfinite(train_matrix), axis=0)
            with np.errstate(all="ignore"):
                minimum = np.nanmin(train_matrix, axis=0)
                maximum = np.nanmax(train_matrix, axis=0)
            eligible = (
                (non_missing >= float(non_missing_threshold))
                & np.isfinite(minimum)
                & np.isfinite(maximum)
                & (minimum != maximum)
            )
            if int(eligible.sum()) == 0:
                raise ValueError(f"{band}/{heldout}: no eligible connectivity features")
            outer = make_split_arrays(matrix, coverage, train_idx, test_idx)
            inner: list[SplitArrays] = []
            for inner_heldout in sorted(set(patients[train_idx].tolist())):
                inner_train_idx = train_idx[patients[train_idx] != inner_heldout]
                inner_test_idx = train_idx[patients[train_idx] == inner_heldout]
                inner.append(make_split_arrays(matrix, coverage, inner_train_idx, inner_test_idx))
            outers.append(
                OuterDesign(
                    heldout_patient=str(heldout),
                    eligible=np.asarray(eligible, dtype=bool),
                    outer=outer,
                    inner=inner,
                )
            )
        designs[str(band)] = BandDesign(str(band), candidates, outers)
    return designs


def _feature_scores(outer: OuterDesign, y: np.ndarray, mode: str) -> np.ndarray:
    score_sum = np.zeros(len(outer.eligible), dtype=float)
    score_count = np.zeros(len(outer.eligible), dtype=int)
    for split in outer.inner:
        y_train = y[split.train_idx]
        y_test = y[split.test_idx]
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue
        train = split.raw_train if mode == "raw" else split.residual_train
        test = split.raw_test if mode == "raw" else split.residual_test
        positive_median = np.median(train[y_train == 1], axis=0)
        negative_median = np.median(train[y_train == 0], axis=0)
        direction = np.where(positive_median >= negative_median, 1.0, -1.0)
        valid = np.ptp(train, axis=0) > 0
        scores = average_precision_columns_exact(y_test, test * direction[np.newaxis, :])
        keep = np.isfinite(scores) & valid & outer.eligible
        score_sum[keep] += scores[keep]
        score_count[keep] += 1
    output = np.full(len(outer.eligible), -np.inf, dtype=float)
    valid = score_count > 0
    output[valid] = score_sum[valid] / score_count[valid]
    output[~outer.eligible] = -np.inf
    return output


def _select_feature_indices(
    outer: OuterDesign,
    y: np.ndarray,
    mode: str,
    candidates: Sequence[str],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _feature_scores(outer, y, mode)
    eligible_indices = np.flatnonzero(np.isfinite(scores))
    if not len(eligible_indices):
        raise ValueError(f"{outer.heldout_patient}: no feature received a finite inner AP")
    order = sorted(eligible_indices.tolist(), key=lambda idx: (-scores[idx], str(candidates[idx])))
    selected = np.asarray(order[: min(int(top_k), len(order))], dtype=int)
    return selected, scores


def _fit_logistic_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    c_value: float,
) -> np.ndarray:
    if len(np.unique(y_train)) < 2:
        raise ValueError("Logistic-regression training labels contain one class")
    model = LogisticRegression(
        l1_ratio=0.0,
        class_weight="balanced",
        solver="liblinear",
        C=float(c_value),
        random_state=0,
        max_iter=1000,
    )
    model.fit(np.asarray(x_train, dtype=float), y_train.astype(int))
    return model.predict_proba(np.asarray(x_test, dtype=float))[:, 1]


def average_percentile_rank(scores: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(scores, dtype=float)).rank(
        method="average", pct=True, ascending=True
    ).to_numpy(float)


def _safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def _choose_raw_c(
    outer: OuterDesign,
    selected: np.ndarray,
    y: np.ndarray,
    c_values: Sequence[float],
) -> tuple[float, float]:
    rows: list[tuple[float, float]] = []
    for c_value in sorted(float(value) for value in c_values):
        fold_scores: list[float] = []
        for split in outer.inner:
            y_train = y[split.train_idx]
            y_test = y[split.test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            prediction = _fit_logistic_predict(
                split.raw_train[:, selected], y_train, split.raw_test[:, selected], c_value
            )
            fold_scores.append(_safe_ap(y_test, prediction))
        rows.append((float(np.mean(fold_scores)) if fold_scores else -np.inf, c_value))
    best_score, best_c = sorted(rows, key=lambda row: (-row[0], row[1]))[0]
    return float(best_c), float(best_score)


def _choose_residual_lambda_c(
    outer: OuterDesign,
    selected: np.ndarray,
    y: np.ndarray,
    coverage_rank: np.ndarray,
    c_values: Sequence[float],
    lambdas: Sequence[float],
) -> tuple[float, float, float]:
    rows: list[tuple[float, float, float]] = []
    for c_value in sorted(float(value) for value in c_values):
        by_fold: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for split in outer.inner:
            y_train = y[split.train_idx]
            y_test = y[split.test_idx]
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue
            prediction = _fit_logistic_predict(
                split.residual_train[:, selected],
                y_train,
                split.residual_test[:, selected],
                c_value,
            )
            by_fold.append(
                (y_test, average_percentile_rank(prediction), coverage_rank[split.test_idx])
            )
        for lambda_value in sorted(float(value) for value in lambdas):
            fold_scores = [
                _safe_ap(y_test, (1.0 - lambda_value) * coverage + lambda_value * q)
                for y_test, q, coverage in by_fold
            ]
            rows.append(
                (
                    float(np.mean(fold_scores)) if fold_scores else -np.inf,
                    lambda_value,
                    c_value,
                )
            )
    best_score, best_lambda, best_c = sorted(rows, key=lambda row: (-row[0], row[1], row[2]))[0]
    return float(best_lambda), float(best_c), float(best_score)


def _prediction_rows(
    data: pd.DataFrame,
    indices: np.ndarray,
    y: np.ndarray,
    scores: np.ndarray,
    band: str,
    model: str,
) -> list[dict[str, object]]:
    rank = pd.Series(scores).rank(method="average", ascending=False).to_numpy(float)
    rows: list[dict[str, object]] = []
    for position, index in enumerate(indices):
        source = data.iloc[int(index)]
        rows.append(
            {
                "frequency_band": band,
                "model": model,
                "heldout_patient": str(source["patient"]),
                "patient": str(source["patient"]),
                "desikan_killiany_label": str(source["desikan_killiany_label"]),
                "tissue_class": str(source["tissue_class"]),
                "is_white_csf_ventricle_unknown": bool(source["is_white_csf_ventricle_unknown"]),
                "y_true": int(y[int(index)]),
                "y_score": float(scores[position]),
                "rank_within_patient": float(rank[position]),
            }
        )
    return rows


def run_models(
    data: pd.DataFrame,
    designs: dict[str, BandDesign],
    y: np.ndarray,
    bands: Sequence[str],
    top_k: int,
    c_values: Sequence[float],
    lambdas: Sequence[float],
    include_connectivity: bool = True,
    collect_folds: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coverage_rank = data["coverage_rank_percentile"].to_numpy(float)
    prediction_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for band in bands:
        design = designs[str(band)]
        for outer in design.outers:
            test_idx = outer.outer.test_idx
            coverage_score = coverage_rank[test_idx]
            prediction_rows.extend(
                _prediction_rows(data, test_idx, y, coverage_score, str(band), "coverage_only")
            )
            if include_connectivity:
                raw_selected, raw_scores = _select_feature_indices(
                    outer, y, "raw", design.candidates, top_k
                )
                raw_c, raw_inner_ap = _choose_raw_c(outer, raw_selected, y, c_values)
                raw_prediction = _fit_logistic_predict(
                    outer.outer.raw_train[:, raw_selected],
                    y[outer.outer.train_idx],
                    outer.outer.raw_test[:, raw_selected],
                    raw_c,
                )
                prediction_rows.extend(
                    _prediction_rows(
                        data,
                        test_idx,
                        y,
                        raw_prediction,
                        str(band),
                        "connectivity_corrected",
                    )
                )
                if collect_folds:
                    fold_rows.append(
                        {
                            "frequency_band": band,
                            "model": "connectivity_corrected",
                            "heldout_patient": outer.heldout_patient,
                            "selected_features": "|".join(design.candidates[i] for i in raw_selected),
                            "selected_feature_inner_average_precision": "|".join(
                                f"{raw_scores[i]:.17g}" for i in raw_selected
                            ),
                            "selected_C": raw_c,
                            "selected_lambda": np.nan,
                            "inner_average_precision": raw_inner_ap,
                        }
                    )
            residual_selected, residual_scores = _select_feature_indices(
                outer, y, "residual", design.candidates, top_k
            )
            selected_lambda, residual_c, residual_inner_ap = _choose_residual_lambda_c(
                outer,
                residual_selected,
                y,
                coverage_rank,
                c_values,
                lambdas,
            )
            residual_prediction = _fit_logistic_predict(
                outer.outer.residual_train[:, residual_selected],
                y[outer.outer.train_idx],
                outer.outer.residual_test[:, residual_selected],
                residual_c,
            )
            q_score = average_percentile_rank(residual_prediction)
            combined_score = (1.0 - selected_lambda) * coverage_score + selected_lambda * q_score
            prediction_rows.extend(
                _prediction_rows(
                    data,
                    test_idx,
                    y,
                    combined_score,
                    str(band),
                    "coverage_plus_residual_connectivity",
                )
            )
            if collect_folds:
                fold_rows.append(
                    {
                        "frequency_band": band,
                        "model": "coverage_plus_residual_connectivity",
                        "heldout_patient": outer.heldout_patient,
                        "selected_features": "|".join(
                            design.candidates[i] for i in residual_selected
                        ),
                        "selected_feature_inner_average_precision": "|".join(
                            f"{residual_scores[i]:.17g}" for i in residual_selected
                        ),
                        "selected_C": residual_c,
                        "selected_lambda": selected_lambda,
                        "inner_average_precision": residual_inner_ap,
                        "residualizer": "Ridge(alpha=1; fit_on_training_fold_only)",
                    }
                )
    return pd.DataFrame.from_records(prediction_rows), pd.DataFrame.from_records(fold_rows)


def patient_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_columns = [column for column in ["label_rule", "frequency_band", "model", "patient"] if column in predictions]
    for keys, frame in predictions.groupby(group_columns, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        identity = dict(zip(group_columns, keys, strict=True))
        y = frame["y_true"].to_numpy(int)
        score = frame["y_score"].to_numpy(float)
        if len(np.unique(y)) != 2:
            raise ValueError(f"Patient metric is undefined for {identity}: one label class")
        prevalence = float(y.mean())
        ap = float(average_precision_score(y, score))
        rows.append(
            {
                **identity,
                "n_regions": len(frame),
                "n_positive": int(y.sum()),
                "prevalence": prevalence,
                "average_precision": ap,
                "AUPRC": ap,
                "AUPRC_definition": METRIC_DEFINITION,
                "normalized_average_precision": float((ap - prevalence) / (1.0 - prevalence)),
                "AUROC": float(roc_auc_score(y, score)),
            }
        )
    return pd.DataFrame.from_records(rows)


def metric_table(patient_metrics: pd.DataFrame) -> pd.DataFrame:
    group_columns = [column for column in ["label_rule", "frequency_band", "model"] if column in patient_metrics]
    metrics = ["average_precision", "normalized_average_precision", "AUROC", "prevalence"]
    output = patient_metrics.groupby(group_columns, sort=False)[metrics].mean().reset_index()
    output["AUPRC"] = output["average_precision"]
    output["AUPRC_definition"] = METRIC_DEFINITION
    output["primary_metric"] = "patient_macro_average_precision"
    output["metric_scope"] = "within_patient_then_equal_macro_average"
    output["n_patients_metric"] = patient_metrics["patient"].nunique()
    return output


def paired_bootstrap(
    patient_metrics: pd.DataFrame,
    rule_names: Sequence[str],
    bands: Sequence[str],
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    patients = sorted(patient_metrics["patient"].astype(str).unique())
    models = MODEL_ORDER
    metric_names = ["average_precision", "normalized_average_precision", "AUROC"]
    tensor = np.full(
        (len(patients), len(rule_names), len(bands), len(models), len(metric_names)),
        np.nan,
        dtype=float,
    )
    patient_index = {value: idx for idx, value in enumerate(patients)}
    rule_index = {value: idx for idx, value in enumerate(rule_names)}
    band_index = {value: idx for idx, value in enumerate(bands)}
    model_index = {value: idx for idx, value in enumerate(models)}
    for row in patient_metrics.itertuples(index=False):
        pi = patient_index[str(row.patient)]
        ri = rule_index[str(row.label_rule)]
        bi = band_index[str(row.frequency_band)]
        mi = model_index[str(row.model)]
        tensor[pi, ri, bi, mi] = [
            float(row.average_precision),
            float(row.normalized_average_precision),
            float(row.AUROC),
        ]
    if not np.isfinite(tensor).all():
        raise ValueError("Incomplete patient-metric tensor for paired bootstrap")
    rng = np.random.default_rng(int(seed))
    rows: list[dict[str, object]] = []
    for bootstrap_id in range(1, int(n_bootstrap) + 1):
        sampled = rng.integers(0, len(patients), size=len(patients))
        mean = tensor[sampled].mean(axis=0)
        sample_hash = hashlib.sha256(sampled.astype(np.int16).tobytes()).hexdigest()
        for rule_idx, rule in enumerate(rule_names):
            for band_idx, band in enumerate(bands):
                values = mean[rule_idx, band_idx]
                record: dict[str, object] = {
                    "bootstrap_id": bootstrap_id,
                    "sample_index_sha256": sample_hash,
                    "label_rule": rule,
                    "frequency_band": band,
                }
                for model_idx, model in enumerate(models):
                    for metric_idx, metric in enumerate(metric_names):
                        record[f"{model}__{metric}"] = float(values[model_idx, metric_idx])
                record["delta_average_precision"] = float(
                    values[2, 0] - values[0, 0]
                )
                rows.append(record)
    return pd.DataFrame.from_records(rows)


def bootstrap_summary(bootstrap: pd.DataFrame) -> pd.DataFrame:
    if bootstrap.empty:
        return pd.DataFrame()
    value_columns = [
        column
        for column in bootstrap.columns
        if column.endswith("__average_precision") or column == "delta_average_precision"
    ]
    rows: list[dict[str, object]] = []
    for (rule, band), frame in bootstrap.groupby(["label_rule", "frequency_band"], sort=False):
        record: dict[str, object] = {
            "label_rule": rule,
            "frequency_band": band,
            "n_bootstrap": int(frame["bootstrap_id"].nunique()),
        }
        for column in value_columns:
            record[f"{column}__median"] = float(frame[column].median())
            record[f"{column}__ci_low"] = float(frame[column].quantile(0.025))
            record[f"{column}__ci_high"] = float(frame[column].quantile(0.975))
        rows.append(record)
    return pd.DataFrame.from_records(rows)


def conditioned_permute_labels(
    y: np.ndarray,
    patients: np.ndarray,
    layers: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.asarray(y, dtype=int).copy()
    group_keys = sorted(set(zip(patients.astype(str), layers.astype(str), strict=True)))
    for patient, layer in group_keys:
        indices = np.flatnonzero((patients.astype(str) == patient) & (layers.astype(str) == layer))
        if len(indices) > 1:
            output[indices] = rng.permutation(output[indices])
        if int(output[indices].sum()) != int(np.asarray(y, dtype=int)[indices].sum()):
            raise RuntimeError("Conditional permutation changed a patient-by-coverage-layer positive count")
        if len(np.unique(np.asarray(y, dtype=int)[indices])) <= 1:
            if not np.array_equal(output[indices], np.asarray(y, dtype=int)[indices]):
                raise RuntimeError("A non-exchangeable conditional stratum changed")
    return output


def _permutation_metric_rows(
    permutation_id: int,
    data: pd.DataFrame,
    designs: dict[str, BandDesign],
    y: np.ndarray,
    bands: Sequence[str],
    top_k: int,
    c_values: Sequence[float],
    lambdas: Sequence[float],
    seed: int,
) -> list[dict[str, object]]:
    permutation_seed = int(seed + permutation_id * 1_000_003)
    permuted = conditioned_permute_labels(
        y,
        data["patient"].astype(str).to_numpy(),
        data["coverage_layer"].astype(str).to_numpy(),
        np.random.default_rng(permutation_seed),
    )
    predictions, _ = run_models(
        data,
        designs,
        permuted,
        bands,
        top_k,
        c_values,
        lambdas,
        include_connectivity=False,
        collect_folds=False,
    )
    patient_metrics = patient_metric_table(predictions)
    metrics = metric_table(patient_metrics)
    pivot = metrics.pivot(index="frequency_band", columns="model", values="average_precision")
    rows: list[dict[str, object]] = []
    for band in bands:
        coverage_ap = float(pivot.loc[band, "coverage_only"])
        combined_ap = float(pivot.loc[band, "coverage_plus_residual_connectivity"])
        rows.append(
            {
                "permutation_id": int(permutation_id),
                "permutation_seed": permutation_seed,
                "frequency_band": str(band),
                "coverage_only_average_precision": coverage_ap,
                "coverage_plus_residual_connectivity_average_precision": combined_ap,
                "delta_average_precision": combined_ap - coverage_ap,
            }
        )
    return rows


def _permutation_initializer(state: dict[str, object]) -> None:
    global _PERMUTATION_STATE
    _PERMUTATION_STATE = state


def _permutation_worker(permutation_id: int) -> list[dict[str, object]]:
    state = _PERMUTATION_STATE
    return _permutation_metric_rows(
        permutation_id,
        state["data"],
        state["designs"],
        state["y"],
        state["bands"],
        state["top_k"],
        state["c_values"],
        state["lambdas"],
        state["seed"],
    )


def run_conditional_permutations(
    data: pd.DataFrame,
    designs: dict[str, BandDesign],
    y: np.ndarray,
    bands: Sequence[str],
    top_k: int,
    c_values: Sequence[float],
    lambdas: Sequence[float],
    seed: int,
    n_permutations: int,
    n_jobs: int,
    checkpoint_every: int,
    checkpoint_path: Path,
    resume: bool,
) -> pd.DataFrame:
    if resume and checkpoint_path.exists():
        checkpoint = pd.read_csv(checkpoint_path)
    else:
        checkpoint = pd.DataFrame()
    complete: set[int] = set()
    if not checkpoint.empty:
        required = {
            "permutation_id",
            "permutation_seed",
            "frequency_band",
            "coverage_only_average_precision",
            "coverage_plus_residual_connectivity_average_precision",
            "delta_average_precision",
        }
        missing = sorted(required.difference(checkpoint.columns))
        if missing:
            raise SystemExit(f"Permutation checkpoint is missing columns: {missing}")
        expected_bands = set(str(value) for value in bands)
        for permutation_id, frame in checkpoint.groupby("permutation_id", sort=False):
            actual_bands = set(frame["frequency_band"].astype(str))
            if actual_bands == expected_bands and len(frame) == len(expected_bands):
                expected_seed = int(seed + int(permutation_id) * 1_000_003)
                if set(frame["permutation_seed"].astype(int)) != {expected_seed}:
                    raise SystemExit(f"Permutation {permutation_id} checkpoint seed mismatch")
                complete.add(int(permutation_id))
        checkpoint = checkpoint.loc[checkpoint["permutation_id"].isin(complete)].copy()
    pending = [value for value in range(1, int(n_permutations) + 1) if value not in complete]
    state = {
        "data": data,
        "designs": designs,
        "y": np.asarray(y, dtype=int),
        "bands": list(bands),
        "top_k": int(top_k),
        "c_values": list(c_values),
        "lambdas": list(lambdas),
        "seed": int(seed),
    }
    accumulated = checkpoint.to_dict("records") if not checkpoint.empty else []
    if int(n_jobs) <= 1:
        iterator: Iterable[list[dict[str, object]]] = (
            _permutation_metric_rows(
                permutation_id,
                data,
                designs,
                y,
                bands,
                top_k,
                c_values,
                lambdas,
                seed,
            )
            for permutation_id in pending
        )
        executor = None
    else:
        executor = ProcessPoolExecutor(
            max_workers=int(n_jobs),
            initializer=_permutation_initializer,
            initargs=(state,),
        )
        iterator = executor.map(_permutation_worker, pending, chunksize=1)
    try:
        for completed_count, rows in enumerate(iterator, start=1):
            accumulated.extend(rows)
            if completed_count % int(checkpoint_every) == 0:
                atomic_write_csv(
                    pd.DataFrame.from_records(accumulated).sort_values(
                        ["permutation_id", "frequency_band"], kind="stable"
                    ),
                    checkpoint_path,
                )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    output = pd.DataFrame.from_records(accumulated)
    if int(n_permutations) == 0:
        return output
    output = output.drop_duplicates(["permutation_id", "frequency_band"], keep="last")
    output = output.sort_values(["permutation_id", "frequency_band"], kind="stable")
    expected_rows = int(n_permutations) * len(bands)
    if len(output) != expected_rows:
        raise RuntimeError(f"Conditional permutation table has {len(output)} rows, expected {expected_rows}")
    atomic_write_csv(output, checkpoint_path)
    return output


def conditional_inference(
    permutation: pd.DataFrame,
    observed_metrics: pd.DataFrame,
    bands: Sequence[str],
) -> pd.DataFrame:
    primary = observed_metrics.loc[observed_metrics["label_rule"].eq("primary")]
    pivot = primary.pivot(index="frequency_band", columns="model", values="average_precision")
    observed = {
        band: float(
            pivot.loc[band, "coverage_plus_residual_connectivity"]
            - pivot.loc[band, "coverage_only"]
        )
        for band in bands
    }
    if permutation.empty:
        return pd.DataFrame.from_records(
            [
                {
                    "frequency_band": band,
                    "observed_delta_average_precision": observed[band],
                    "n_permutations": 0,
                    "raw_one_sided_P": np.nan,
                    "maxT_FWER_one_sided_P": np.nan,
                }
                for band in bands
            ]
        )
    maxima = permutation.groupby("permutation_id")["delta_average_precision"].max()
    rows: list[dict[str, object]] = []
    for band in bands:
        null = permutation.loc[
            permutation["frequency_band"].eq(band), "delta_average_precision"
        ].to_numpy(float)
        raw_p = float((1 + int(np.sum(null >= observed[band]))) / (len(null) + 1))
        max_p = float((1 + int(np.sum(maxima.to_numpy(float) >= observed[band]))) / (len(maxima) + 1))
        if max_p + 1e-15 < raw_p:
            raise RuntimeError("maxT-FWER P value is smaller than its raw P value")
        rows.append(
            {
                "frequency_band": band,
                "observed_delta_average_precision": observed[band],
                "n_permutations": len(null),
                "raw_one_sided_P": raw_p,
                "maxT_FWER_one_sided_P": max_p,
            }
        )
    return pd.DataFrame.from_records(rows)


def default_label_rules(base_rule: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    definitions = [
        ("primary", 4, 0.50, 5),
        ("top3", 3, 0.50, 5),
        ("top5", 5, 0.50, 5),
        ("no_topk_limit", 999, 0.50, 5),
        ("rate_gt_0p40", 4, 0.40, 5),
        ("rate_gt_0p60", 4, 0.60, 5),
        ("count_gt_3", 4, 0.50, 3),
        ("count_gt_7", 4, 0.50, 7),
    ]
    for name, top_k, rate, count in definitions:
        rule = dict(base_rule)
        rule.update(
            {
                "name": name,
                "top_k": top_k,
                "burned_rate_strictly_greater_than": rate,
                "burned_count_strictly_greater_than": count,
                "threshold_logic": "or",
            }
        )
        rows.append(rule)
    return rows


def load_region_core() -> object:
    scripts = str(REPO_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import run_region_lopo_modeling as region_core

    return region_core


def prepare_data_and_labels(
    features_path: Path,
    audit_workbook: Path,
    base_rule: dict[str, object],
    selected_rules: Sequence[str],
) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame, pd.DataFrame, list[str]]:
    features = pd.read_csv(features_path)
    missing_keys = sorted(set(REGION_KEYS).difference(features.columns))
    if missing_keys:
        raise ValueError(f"Feature table is missing region keys: {missing_keys}")
    core = load_region_core()
    region_rows = core.load_region_summary_rows(audit_workbook)
    definitions = default_label_rules(base_rule)
    definition_by_name = {str(item["name"]): item for item in definitions}
    unknown = sorted(set(selected_rules).difference(definition_by_name))
    if unknown:
        raise ValueError(f"Unknown label rules: {unknown}")
    labels_by_rule: dict[str, pd.DataFrame] = {}
    audit_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    for rule_name in selected_rules:
        definition = dict(definition_by_name[rule_name])
        definition.pop("name")
        labels, audit = core.build_ablation_top4_label_table(region_rows, definition)
        labels = labels.copy()
        labels.insert(0, "label_rule", rule_name)
        audit = audit.copy()
        audit.insert(0, "label_rule", rule_name)
        labels_by_rule[rule_name] = labels
        label_frames.append(labels)
        audit_frames.append(audit)
    if "primary" not in labels_by_rule:
        raise ValueError("The primary label rule is required")
    primary = labels_by_rule["primary"]
    data = features.merge(primary.drop(columns="label_rule"), on=REGION_KEYS, how="inner", validate="one_to_one")
    data = data.loc[data["inclusion_status"].eq("primary")].copy()
    data = data.sort_values(["patient", "desikan_killiany_label"], kind="stable").reset_index(drop=True)
    data, coverage_columns = add_coverage_features(data)
    if len(data) != 84 or int(data["label_binary"].sum()) != 22:
        raise ValueError(
            f"Primary label invariant failed: rows={len(data)}, positives={int(data['label_binary'].sum())}"
        )
    y_by_rule: dict[str, np.ndarray] = {}
    data_keys = pd.MultiIndex.from_frame(data[REGION_KEYS])
    for rule_name, labels in labels_by_rule.items():
        primary_rule = labels.loc[labels["inclusion_status"].eq("primary")].copy()
        mapping = primary_rule.set_index(REGION_KEYS)["label_binary"]
        aligned = mapping.reindex(data_keys)
        if aligned.isna().any():
            raise ValueError(f"{rule_name}: primary modeled row set changed across OFAT rule")
        y = aligned.to_numpy(dtype=int)
        for patient, indices in data.groupby("patient", sort=False).groups.items():
            if len(np.unique(y[np.asarray(list(indices), dtype=int)])) != 2:
                raise ValueError(f"{rule_name}/{patient}: both label classes are required")
        y_by_rule[rule_name] = y
    return (
        data,
        y_by_rule,
        pd.concat(label_frames, ignore_index=True),
        pd.concat(audit_frames, ignore_index=True),
        coverage_columns,
    )


def label_jaccard(labels: pd.DataFrame) -> pd.DataFrame:
    sets: dict[str, set[tuple[str, str]]] = {}
    for rule, frame in labels.groupby("label_rule", sort=False):
        positive = frame.loc[frame["label_binary"].eq(1)]
        sets[str(rule)] = set(
            zip(
                positive["patient"].astype(str),
                positive["desikan_killiany_label"].astype(str),
                strict=True,
            )
        )
    primary = sets["primary"]
    rows = []
    for rule, values in sets.items():
        union = primary | values
        rows.append(
            {
                "label_rule": rule,
                "n_positive": len(values),
                "n_overlap_with_primary": len(primary & values),
                "jaccard_with_primary": float(len(primary & values) / len(union)),
            }
        )
    return pd.DataFrame.from_records(rows)


def rule_stability_summary(
    metrics: pd.DataFrame,
    patient_metrics: pd.DataFrame,
    bands: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_pivot = metrics.pivot(
        index=["label_rule", "frequency_band"], columns="model", values="average_precision"
    ).reset_index()
    metric_pivot["delta_average_precision"] = (
        metric_pivot["coverage_plus_residual_connectivity"] - metric_pivot["coverage_only"]
    )
    patient_pivot = patient_metrics.pivot(
        index=["label_rule", "frequency_band", "patient"],
        columns="model",
        values="average_precision",
    ).reset_index()
    patient_pivot["delta_average_precision"] = (
        patient_pivot["coverage_plus_residual_connectivity"] - patient_pivot["coverage_only"]
    )
    counts = (
        patient_pivot.assign(delta_positive=patient_pivot["delta_average_precision"] > 0)
        .groupby(["label_rule", "frequency_band"], sort=False)["delta_positive"]
        .sum()
        .rename("n_patients_delta_positive")
        .reset_index()
    )
    metric_pivot = metric_pivot.merge(counts, on=["label_rule", "frequency_band"], how="left")
    summary_rows: list[dict[str, object]] = []
    for band in bands:
        values = metric_pivot.loc[metric_pivot["frequency_band"].eq(band), "delta_average_precision"]
        positive = int((values > 0).sum())
        summary_rows.append(
            {
                "frequency_band": band,
                "n_rules": len(values),
                "delta_average_precision_min": float(values.min()),
                "delta_average_precision_median": float(values.median()),
                "delta_average_precision_max": float(values.max()),
                "n_rules_delta_positive": positive,
                "rule_robust_at_least_6_of_8": bool(len(values) == 8 and positive >= 6),
            }
        )
    return metric_pivot, pd.DataFrame.from_records(summary_rows)


def run_complete_delete_patient_refits(
    data: pd.DataFrame,
    y: np.ndarray,
    bands: Sequence[str],
    coverage_columns: Sequence[str],
    non_missing_threshold: float,
    expected_candidates: int | None,
    top_k: int,
    c_values: Sequence[float],
    lambdas: Sequence[float],
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for omitted in sorted(data["patient"].astype(str).unique()):
        keep = ~data["patient"].astype(str).eq(omitted).to_numpy()
        subset = data.loc[keep].copy().reset_index(drop=True)
        subset_y = np.asarray(y)[keep]
        designs = build_band_designs(
            subset,
            bands,
            coverage_columns,
            non_missing_threshold,
            expected_candidates,
        )
        predictions, _ = run_models(
            subset,
            designs,
            subset_y,
            bands,
            top_k,
            c_values,
            lambdas,
            include_connectivity=False,
            collect_folds=False,
        )
        patient_metrics = patient_metric_table(predictions)
        metrics = metric_table(patient_metrics)
        pivot = metrics.pivot(index="frequency_band", columns="model", values="average_precision").reset_index()
        pivot.insert(0, "omitted_patient", omitted)
        pivot["delta_average_precision"] = (
            pivot["coverage_plus_residual_connectivity"] - pivot["coverage_only"]
        )
        rows.append(pivot)
    return pd.concat(rows, ignore_index=True)


def protected_snapshots(paths: Sequence[str]) -> dict[str, str]:
    return {str(Path(path).resolve()): directory_sha256(Path(path)) for path in paths}


def build_manifest_payload(
    config_path: Path,
    config: dict[str, object],
    features_path: Path,
    audit_workbook: Path,
    bands: Sequence[str],
    rules: Sequence[str],
    n_permutations: int,
    n_bootstrap: int,
    n_rule_bootstrap: int,
    n_jobs: int,
) -> tuple[dict[str, object], str]:
    core_path = REPO_ROOT / "scripts" / "run_region_lopo_modeling.py"
    identity = {
        "algorithm_version": ALGORITHM_VERSION,
        "average_precision_implementation": AP_IMPLEMENTATION,
        "config_sha256": file_sha256(config_path),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "region_core_sha256": file_sha256(core_path),
        "features_sha256": file_sha256(features_path),
        "audit_workbook_sha256": file_sha256(audit_workbook),
        "bands": list(bands),
        "label_rules": list(rules),
        "n_permutations": int(n_permutations),
        "n_bootstrap": int(n_bootstrap),
        "n_rule_bootstrap": int(n_rule_bootstrap),
        "n_jobs": int(n_jobs),
        "config": config,
    }
    fingerprint = canonical_hash(identity)
    manifest = {
        **identity,
        "analysis_fingerprint": fingerprint,
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "pyyaml": yaml.__version__,
        },
        "command": [str(value) for value in sys.argv],
    }
    return manifest, fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-permutations", type=int)
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--n-rule-bootstrap", type=int)
    parser.add_argument("--n-jobs", type=int)
    parser.add_argument("--bands", help="Comma-separated subset for an explicitly non-formal smoke run")
    parser.add_argument("--label-rules", help="Comma-separated subset; primary is always required")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    output_root = args.out or Path(str(config["output_root"]))
    assert_allowed_output_root(output_root)
    inputs = dict(config["inputs"])
    features_path = Path(str(inputs["region_patient_features"])).resolve()
    audit_workbook = Path(str(inputs["audit_workbook"])).resolve()
    for required in [features_path, audit_workbook, SCRIPT_PATH]:
        if not required.exists():
            raise SystemExit(f"Missing required input: {required}")
    modeling = dict(config["modeling"])
    inference = dict(config["inference"])
    execution = dict(config["execution"])
    bands = (
        [item.strip() for item in args.bands.split(",") if item.strip()]
        if args.bands
        else [str(value) for value in config["frequency_bands"]]
    )
    selected_rules = (
        [item.strip() for item in args.label_rules.split(",") if item.strip()]
        if args.label_rules
        else [str(value) for value in config["label_rules"]]
    )
    if "primary" not in selected_rules:
        raise SystemExit("--label-rules must include primary")
    n_permutations = int(
        args.n_permutations if args.n_permutations is not None else inference["n_conditional_permutations"]
    )
    n_bootstrap = int(args.n_bootstrap if args.n_bootstrap is not None else inference["n_primary_bootstrap"])
    n_rule_bootstrap = int(
        args.n_rule_bootstrap
        if args.n_rule_bootstrap is not None
        else inference["n_label_rule_bootstrap"]
    )
    n_jobs = int(args.n_jobs if args.n_jobs is not None else execution["n_jobs"])
    if min(n_permutations, n_bootstrap, n_rule_bootstrap) < 0 or n_jobs < 1:
        raise SystemExit("Iteration counts must be non-negative and n_jobs must be positive")
    data, y_by_rule, labels, label_audit, coverage_columns = prepare_data_and_labels(
        features_path,
        audit_workbook,
        dict(config["base_label_rule"]),
        selected_rules,
    )
    expected_candidates = modeling.get("expected_candidates_per_band")
    expected_candidates = None if expected_candidates is None else int(expected_candidates)
    expected = dict(config.get("expected", {}))
    if expected:
        primary_negative = int(len(data) - y_by_rule["primary"].sum())
        actual_core = {
            "primary_rows": len(data),
            "primary_positive": int(y_by_rule["primary"].sum()),
            "primary_negative": primary_negative,
            "patients": int(data["patient"].nunique()),
        }
        for key, actual in actual_core.items():
            if key in expected and actual != int(expected[key]):
                raise SystemExit(f"Expected invariant failed for {key}: {actual} != {expected[key]}")
    for band in bands:
        candidates = connectivity_columns_for_band(data, band)
        if expected_candidates is not None and len(candidates) != expected_candidates:
            raise SystemExit(f"{band}: expected {expected_candidates} candidates, found {len(candidates)}")
    print(f"run_id={config['run_id']}")
    print(f"features={features_path}")
    print(f"output_root={output_root}")
    print(f"rows={len(data)} positives={int(y_by_rule['primary'].sum())} patients={data['patient'].nunique()}")
    print(f"bands={'|'.join(bands)} rules={'|'.join(selected_rules)}")
    print(
        f"conditional_permutations={n_permutations} primary_bootstrap={n_bootstrap} "
        f"rule_bootstrap={n_rule_bootstrap} n_jobs={n_jobs}"
    )
    if bool(config.get("legacy_input_smoke_only", False)):
        print("legacy_input_smoke_only=true; outputs are limited to development checks")
    if args.dry_run:
        print("dry_run=true; no files written")
        return
    manifest, fingerprint = build_manifest_payload(
        args.config.resolve(),
        config,
        features_path,
        audit_workbook,
        bands,
        selected_rules,
        n_permutations,
        n_bootstrap,
        n_rule_bootstrap,
        n_jobs,
    )
    manifest_path = output_root / "run_manifest.json"
    if output_root.exists() and not args.resume:
        raise SystemExit(f"Output already exists; use a new directory or --resume: {output_root}")
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("--resume requires an existing run_manifest.json")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("analysis_fingerprint") != fingerprint:
            raise SystemExit("Resume fingerprint mismatch; refusing to mix analyses")
        protected_before = dict(previous["protected_sha256_before"])
        protected_now = protected_snapshots([str(value) for value in config.get("protected_paths", [])])
        if protected_now != protected_before:
            raise SystemExit("A protected legacy/formal path changed since this run started")
        if previous.get("status") == "complete":
            print("status=complete; matching run already finished")
            return
    else:
        output_root.mkdir(parents=True, exist_ok=False)
        protected_before = protected_snapshots([str(value) for value in config.get("protected_paths", [])])
    started = time.time()
    manifest.update(
        {
            "status": "running",
            "started_unix": started,
            "protected_sha256_before": protected_before,
        }
    )
    atomic_write_json(manifest, manifest_path)
    atomic_write_csv(labels, output_root / "label_rule_labels.csv")
    atomic_write_csv(label_audit, output_root / "label_rule_derivation_audit.csv")
    atomic_write_csv(label_jaccard(labels), output_root / "label_rule_jaccard.csv")
    designs = build_band_designs(
        data,
        bands,
        coverage_columns,
        float(modeling["feature_non_missing_threshold"]),
        expected_candidates,
    )
    prediction_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    for rule_name in selected_rules:
        predictions, folds = run_models(
            data,
            designs,
            y_by_rule[rule_name],
            bands,
            int(modeling["top_k_features"]),
            [float(value) for value in modeling["c_values"]],
            [float(value) for value in modeling["lambda_values"]],
            include_connectivity=True,
            collect_folds=True,
        )
        predictions.insert(0, "label_rule", rule_name)
        folds.insert(0, "label_rule", rule_name)
        prediction_frames.append(predictions)
        fold_frames.append(folds)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    expected_prediction_rows = len(data) * len(bands) * len(selected_rules) * len(MODEL_ORDER)
    expected_fold_rows = data["patient"].nunique() * len(bands) * len(selected_rules) * 2
    if len(predictions) != expected_prediction_rows:
        raise RuntimeError(
            f"Observed prediction invariant failed: {len(predictions)} != {expected_prediction_rows}"
        )
    if len(folds) != expected_fold_rows:
        raise RuntimeError(f"Observed fold invariant failed: {len(folds)} != {expected_fold_rows}")
    if expected and len(bands) == int(expected.get("frequency_bands", len(bands))):
        primary_predictions = predictions.loc[predictions["label_rule"].eq("primary")]
        for model in MODEL_ORDER:
            actual = int(primary_predictions["model"].eq(model).sum())
            target = int(expected.get("predictions_per_model", len(data) * len(bands)))
            if actual != target:
                raise RuntimeError(f"{model}: prediction invariant failed: {actual} != {target}")
        for model in ["connectivity_corrected", "coverage_plus_residual_connectivity"]:
            actual = int(
                (folds["label_rule"].eq("primary") & folds["model"].eq(model)).sum()
            )
            target = int(
                expected.get("folds_per_learned_model", data["patient"].nunique() * len(bands))
            )
            if actual != target:
                raise RuntimeError(f"{model}: fold invariant failed: {actual} != {target}")
    patient_metrics = patient_metric_table(predictions)
    metrics = metric_table(patient_metrics)
    atomic_write_csv(predictions, output_root / "observed_predictions_all_label_rules.csv")
    atomic_write_csv(folds, output_root / "observed_fold_details_all_label_rules.csv")
    atomic_write_csv(patient_metrics, output_root / "observed_patient_metrics_all_label_rules.csv")
    atomic_write_csv(metrics, output_root / "observed_metrics_all_label_rules.csv")
    rule_detail, rule_summary = rule_stability_summary(metrics, patient_metrics, bands)
    atomic_write_csv(rule_detail, output_root / "label_rule_increment_detail.csv")
    atomic_write_csv(rule_summary, output_root / "label_rule_stability_summary.csv")
    rule_bootstrap = paired_bootstrap(
        patient_metrics,
        selected_rules,
        bands,
        n_rule_bootstrap,
        int(inference["bootstrap_seed"]),
    )
    atomic_write_csv(rule_bootstrap, output_root / "label_rule_paired_patient_bootstrap.csv")
    atomic_write_csv(
        bootstrap_summary(rule_bootstrap),
        output_root / "label_rule_bootstrap_summary.csv",
    )
    primary_patient_metrics = patient_metrics.loc[patient_metrics["label_rule"].eq("primary")]
    primary_bootstrap = paired_bootstrap(
        primary_patient_metrics,
        ["primary"],
        bands,
        n_bootstrap,
        int(inference["bootstrap_seed"]),
    )
    atomic_write_csv(primary_bootstrap, output_root / "primary_paired_patient_bootstrap.csv")
    atomic_write_csv(
        bootstrap_summary(primary_bootstrap),
        output_root / "primary_bootstrap_summary.csv",
    )
    if bool(execution.get("run_complete_delete_patient_refits", True)):
        jackknife = run_complete_delete_patient_refits(
            data,
            y_by_rule["primary"],
            bands,
            coverage_columns,
            float(modeling["feature_non_missing_threshold"]),
            expected_candidates,
            int(modeling["top_k_features"]),
            [float(value) for value in modeling["c_values"]],
            [float(value) for value in modeling["lambda_values"]],
        )
    else:
        jackknife = pd.DataFrame()
    atomic_write_csv(jackknife, output_root / "primary_complete_delete_patient_refits.csv")
    permutation = run_conditional_permutations(
        data,
        designs,
        y_by_rule["primary"],
        bands,
        int(modeling["top_k_features"]),
        [float(value) for value in modeling["c_values"]],
        [float(value) for value in modeling["lambda_values"]],
        int(inference["permutation_seed"]),
        n_permutations,
        n_jobs,
        int(execution["checkpoint_every"]),
        output_root / "conditional_permutation_checkpoint.csv",
        args.resume,
    )
    inference_frame = conditional_inference(permutation, metrics, bands)
    atomic_write_csv(inference_frame, output_root / "conditional_permutation_inference.csv")
    decision = inference_frame.copy()
    if n_bootstrap:
        ci = (
            primary_bootstrap.groupby("frequency_band")["delta_average_precision"]
            .quantile([0.025, 0.975])
            .unstack()
            .rename(columns={0.025: "delta_ci_low", 0.975: "delta_ci_high"})
            .reset_index()
        )
        decision = decision.merge(ci, on="frequency_band", how="left")
    else:
        decision["delta_ci_low"] = np.nan
        decision["delta_ci_high"] = np.nan
    if not jackknife.empty:
        jackknife_status = (
            jackknife.groupby("frequency_band")["delta_average_precision"]
            .agg(delete_patient_delta_min="min", all_delete_patient_delta_positive=lambda s: bool((s > 0).all()))
            .reset_index()
        )
        decision = decision.merge(jackknife_status, on="frequency_band", how="left")
    else:
        decision["delete_patient_delta_min"] = np.nan
        decision["all_delete_patient_delta_positive"] = False
    decision["strong_increment_claim_retained"] = (
        decision["delta_ci_low"].gt(0)
        & decision["maxT_FWER_one_sided_P"].lt(0.05)
        & decision["all_delete_patient_delta_positive"].eq(True)
    )
    atomic_write_csv(decision, output_root / "primary_incremental_decision.csv")
    protected_after = protected_snapshots([str(value) for value in config.get("protected_paths", [])])
    if protected_after != protected_before:
        raise RuntimeError("A protected input path changed during the analysis")
    output_hashes = {
        path.relative_to(output_root).as_posix(): file_sha256(path)
        for path in sorted(output_root.glob("*.csv"))
    }
    manifest.update(
        {
            "status": "complete",
            "completed_unix": time.time(),
            "elapsed_seconds": time.time() - started,
            "protected_sha256_after": protected_after,
            "output_sha256": output_hashes,
            "n_primary_rows": len(data),
            "n_primary_positive": int(y_by_rule["primary"].sum()),
            "n_patients": int(data["patient"].nunique()),
            "n_bands": len(bands),
            "n_label_rules": len(selected_rules),
            "n_conditional_permutations": n_permutations,
        }
    )
    atomic_write_json(manifest, manifest_path)
    print(f"status=complete output_root={output_root}")


if __name__ == "__main__":
    main()
