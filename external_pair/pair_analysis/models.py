"""Patient-level feature selection, logistic models and ranking metrics."""

from __future__ import annotations


from pathlib import Path
from typing import Any, Iterable, Sequence
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from .utils import safe_average_precision, safe_auroc


_FEATURE_TABLE_LABEL_EXACT = {
    "label",
    "labels",
    "label_binary",
    "primary_status",
    "target",
    "targets",
    "target_y",
    "y_true",
    "y_pred",
    "annotation",
    "annotations",
    "clinical_label",
    "status_description",
    "resect",
    "resection",
    "soz",
    "therapy",
    "treatment",
    "outcome",
    "engel",
    "surgery",
}


_FEATURE_TABLE_LABEL_PREFIXES = (
    "label_",
    "target_",
    "clinical_",
    "annotation_",
    "resect_",
    "soz_",
    "therapy_",
    "treatment_",
    "outcome_",
    "engel_",
    "surgery_",
)


_FEATURE_TABLE_TECHNICAL_ALLOWLIST = {
    # These names are emitted by rate/sampling bookkeeping and are not target
    # annotations.  They remain admissible in a label-blind feature table.
    "target_hz",
    "target_sampling_hz",
    "target_sampling_frequency_hz",
    "sampling_frequency_target_hz",
    "target_annotation_fields_read",
}


def _assert_feature_table_label_blind(path: Path, columns: Sequence[Any]) -> None:
    """Reject clinical/target columns before materialising a feature table.

    Source labels are joined from the separate reviewed workbook, and target
    labels are joined only in the explicit ``--join-labels`` branch.  Reading
    a feature CSV wholesale before checking its header would make a tainted
    artifact indistinguishable from a clean one in the provenance ledger.
    """

    bad: list[str] = []
    for raw in columns:
        name = str(raw).strip().lower()
        if name in _FEATURE_TABLE_TECHNICAL_ALLOWLIST:
            continue
        if name in _FEATURE_TABLE_LABEL_EXACT or any(
            name.startswith(prefix) for prefix in _FEATURE_TABLE_LABEL_PREFIXES
        ):
            bad.append(str(raw))
    if bad:
        raise ValueError(
            f"clinical/target columns in label-blind feature table {path}: {bad}"
        )


def feature_columns(frame: pd.DataFrame, band: str | None = None) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        name = str(column)
        if not name.startswith("feature__") or not pd.api.types.is_numeric_dtype(
            frame[column]
        ):
            continue
        parts = name.split("__")
        if len(parts) < 4:
            continue
        if band is not None and parts[1] != str(band):
            continue
        columns.append(name)
    return sorted(columns)


def feature_method(column: str) -> str:
    parts = str(column).split("__")
    return parts[2] if len(parts) >= 3 else ""


def _ap_fast(y: np.ndarray, score: np.ndarray) -> float:
    """Average precision with deterministic ties and no sklearn warnings."""
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    if y.size == 0 or y.sum() == 0 or y.sum() == y.size:
        return float("nan")
    order = np.argsort(-score, kind="mergesort")
    sy = y[order]
    ss = score[order]
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    starts = np.r_[0, ends[:-1] + 1]
    cumulative = np.cumsum(sy)
    positives = np.add.reduceat(sy, starts)
    precision = cumulative[ends] / (ends + 1.0)
    return float(np.sum(positives * precision) / y.sum())


def patient_macro_metrics(
    predictions: pd.DataFrame,
    patient_col: str = "patient",
    y_col: str = "y_true",
    score_col: str = "y_score",
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for patient, group in predictions.groupby(patient_col, sort=True):
        y = pd.to_numeric(group[y_col], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(group[score_col], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(y) & np.isfinite(s)
        y = y[keep].astype(int)
        s = s[keep]
        if y.size == 0 or len(np.unique(y)) < 2:
            continue
        ap = safe_average_precision(y, s)
        prevalence = float(y.mean())
        norm = (ap - prevalence) / (1.0 - prevalence) if prevalence < 1.0 else np.nan
        rows.append(
            {
                "patient": str(patient),
                "AUPRC": ap,
                "normalized_AUPRC": norm,
                "AUROC": safe_auroc(y, s),
                "n": float(y.size),
            }
        )
    if not rows:
        return {
            "patient_macro_AUPRC": float("nan"),
            "patient_macro_normalized_AUPRC": float("nan"),
            "patient_macro_AUROC": float("nan"),
            "n_metric_patients": 0,
        }
    table = pd.DataFrame(rows)
    return {
        "patient_macro_AUPRC": float(table["AUPRC"].mean()),
        "patient_macro_normalized_AUPRC": float(table["normalized_AUPRC"].mean()),
        "patient_macro_AUROC": float(table["AUROC"].mean()),
        "n_metric_patients": int(len(table)),
    }


def _fit_logistic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    c_value: float,
    *,
    directions: Sequence[float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit the formal imputer/robust-scaler/logistic stack.

    ``directions`` is learned from the training fold only and is applied
    before imputation/scaling.  Keeping it in the serialized preparation
    record makes the source deployment and every target outer fold exactly
    reproducible.
    """
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
    if train_x.ndim != 2 or test_x.ndim != 2 or train_x.shape[1] != test_x.shape[1]:
        raise ValueError(
            "train_x and test_x must be two-dimensional with matching columns"
        )
    if directions is not None:
        orient = np.asarray(directions, dtype=float).reshape(-1)
        if (
            len(orient) != train_x.shape[1]
            or not np.isfinite(orient).all()
            or not np.isin(orient, [-1.0, 1.0]).all()
        ):
            raise ValueError(
                "directions must contain one finite +/-1 value per feature"
            )
        train_x = train_x * orient[None, :]
        test_x = test_x * orient[None, :]
    else:
        orient = np.ones(train_x.shape[1], dtype=float)
    med = np.nanmedian(train_x, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    train_imp = np.where(np.isfinite(train_x), train_x, med[None, :])
    test_imp = np.where(np.isfinite(test_x), test_x, med[None, :])
    # RobustScaler is reimplemented explicitly so the deployment JSON is
    # self-contained and independent of sklearn serialization.
    center = np.nanmedian(train_imp, axis=0)
    q25 = np.nanpercentile(train_imp, 25, axis=0)
    q75 = np.nanpercentile(train_imp, 75, axis=0)
    scale = q75 - q25
    scale = np.where(np.isfinite(scale) & (scale > 1e-12), scale, 1.0)
    xtr = (train_imp - center[None, :]) / scale[None, :]
    xte = (test_imp - center[None, :]) / scale[None, :]
    if len(np.unique(train_y)) < 2:
        # A constant training fold has no discriminative model.  Returning a
        # prevalence score is preferable to silently dropping the fold.
        p = float(np.mean(train_y)) if len(train_y) else 0.5
        return np.full(len(test_x), p, dtype=float), {
            "imputer": med.tolist(),
            "center": center.tolist(),
            "scale": scale.tolist(),
            "coefficients": [0.0] * train_x.shape[1],
            "intercept": float(
                np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
            ),
            "constant_training": True,
            "directions": orient.tolist(),
        }
    model = LogisticRegression(
        penalty="l2",
        class_weight="balanced",
        solver="liblinear",
        C=float(c_value),
        max_iter=2000,
        random_state=0,
    )
    model.fit(xtr, train_y.astype(int))
    scores = model.predict_proba(xte)[:, 1] if len(xte) else np.empty(0, dtype=float)
    return scores, {
        "imputer": med.tolist(),
        "center": center.tolist(),
        "scale": scale.tolist(),
        "coefficients": model.coef_[0].astype(float).tolist(),
        "intercept": float(model.intercept_[0]),
        "constant_training": False,
        "directions": orient.tolist(),
    }


def _deployment_missing_stats(
    frame: pd.DataFrame, features: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-row missing feature count and fraction for deployment.

    Missingness is reported before model imputation.  This keeps the frozen
    model score reproducible while making it possible to distinguish a true
    score from a row that was largely imputed.  The helper deliberately does
    not inspect any clinical/target columns.
    """

    names = [str(item) for item in features]
    if not names:
        n = len(frame)
        return np.zeros(n, dtype=int), np.zeros(n, dtype=float)
    values = (
        frame.reindex(columns=names)
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )
    missing = (~np.isfinite(values)).sum(axis=1).astype(int)
    return missing, missing.astype(float) / float(len(names))


def _learn_directions(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    y_col: str = "label_binary",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Learn deterministic +/-1 feature directions from one training fold.

    The diagnostic table records positive/negative medians and finite counts;
    it is useful for auditing source deployment and every outer fold.  When a
    class has no finite values for a feature, direction ``+1`` is retained as
    a neutral deterministic fallback and the row is marked invalid.
    """

    names = [str(column) for column in columns]
    x = frame.reindex(columns=names).to_numpy(dtype=float)
    y = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(dtype=float)
    directions = np.ones(len(names), dtype=float)
    rows: list[dict[str, Any]] = []
    for j, column in enumerate(names):
        values = (
            x[:, j] if x.ndim == 2 and j < x.shape[1] else np.array([], dtype=float)
        )
        pos = values[(y == 1) & np.isfinite(values)]
        neg = values[(y == 0) & np.isfinite(values)]
        pos_median = float(np.median(pos)) if len(pos) else np.nan
        neg_median = float(np.median(neg)) if len(neg) else np.nan
        valid = bool(np.isfinite(pos_median) and np.isfinite(neg_median))
        if valid and pos_median < neg_median:
            directions[j] = -1.0
        rows.append(
            {
                "feature": column,
                "positive_median": pos_median,
                "negative_median": neg_median,
                "direction": float(directions[j]),
                "positive_n_finite": int(len(pos)),
                "negative_n_finite": int(len(neg)),
                "direction_valid": int(valid),
            }
        )
    return directions, pd.DataFrame(rows)


def select_threshold_max_f1(
    y_true: Sequence[int], scores: Sequence[float], *, fallback: float = 0.5
) -> float:
    """Select a deterministic threshold from out-of-fold scores.

    Candidate thresholds are unique observed scores.  The largest threshold
    among ties is selected, matching the formal implementation's conservative
    tie-break (fewer predicted positives).  If scores are empty/non-finite or
    contain only one class, ``fallback`` is returned and the caller records
    the degenerate selection in its fold manifest.
    """

    y_raw = pd.to_numeric(pd.Series(list(y_true)), errors="coerce").to_numpy(
        dtype=float
    )
    s = pd.to_numeric(pd.Series(list(scores)), errors="coerce").to_numpy(dtype=float)
    n = min(len(y_raw), len(s))
    y_raw, s = y_raw[:n], s[:n]
    keep = np.isfinite(s) & np.isfinite(y_raw)
    y, s = y_raw[keep].astype(int), s[keep]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float(fallback)
    order = np.argsort(-s, kind="mergesort")
    sy = y[order]
    ss = s[order]
    ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
    tp = np.cumsum(sy)[ends].astype(float)
    predicted_positive = (ends + 1).astype(float)
    fp = predicted_positive - tp
    fn = float(y.sum()) - tp
    denominator = 2.0 * tp + fp + fn
    f1 = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    if len(f1) == 0 or not np.isfinite(f1).any():
        return float(fallback)
    best = float(np.nanmax(f1))
    candidate = ss[ends][np.isclose(f1, best, rtol=0.0, atol=1e-12)]
    return float(np.max(candidate)) if len(candidate) else float(fallback)


def _inner_feature_scores(
    train: pd.DataFrame,
    candidates: list[str],
    patient_col: str = "patient",
    y_col: str = "label_binary",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    patients = sorted(train[patient_col].dropna().astype(str).unique())
    for column in candidates:
        values: list[float] = []
        for heldout in patients:
            tr = train[train[patient_col].astype(str) != heldout]
            te = train[train[patient_col].astype(str) == heldout]
            if tr.empty or te.empty:
                continue
            ytr_raw = pd.to_numeric(tr[y_col], errors="coerce").to_numpy(dtype=float)
            yte_raw = pd.to_numeric(te[y_col], errors="coerce").to_numpy(dtype=float)
            valid_tr = np.isfinite(ytr_raw)
            valid_te = np.isfinite(yte_raw)
            if not valid_tr.any() or not valid_te.any():
                continue
            ytr = np.where(valid_tr, ytr_raw, -1).astype(int)
            yte = yte_raw[valid_te].astype(int)
            pos = (
                np.nanmedian(tr.loc[ytr == 1, column].to_numpy(dtype=float))
                if np.any(ytr == 1)
                else np.nan
            )
            neg = (
                np.nanmedian(tr.loc[ytr == 0, column].to_numpy(dtype=float))
                if np.any(ytr == 0)
                else np.nan
            )
            if not (np.isfinite(pos) and np.isfinite(neg)):
                continue
            direction = (
                1.0 if np.isfinite(pos) and np.isfinite(neg) and pos >= neg else -1.0
            )
            score = te[column].to_numpy(dtype=float) * direction
            # ``score`` is already oriented so larger values indicate the
            # positive class.  The training-fold median used for missing test
            # values must be transformed by the same direction; otherwise a
            # negatively oriented feature receives an imputation on the wrong
            # side of its decision scale and its inner-LOPO AUPRC is biased.
            fill = np.nanmedian(tr[column].to_numpy(dtype=float))
            if not np.isfinite(fill):
                continue
            score = np.where(np.isfinite(score), score, fill * direction)[valid_te]
            ap = _ap_fast(yte, score)
            if np.isfinite(ap):
                values.append(ap)
        rows.append(
            {
                "feature": column,
                "inner_macro_AUPRC": float(np.mean(values))
                if values
                else float("-inf"),
                "n_inner": len(values),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["inner_macro_AUPRC", "feature"], ascending=[False, True], kind="stable"
        )
        .reset_index(drop=True)
    )


def _eligible_feature_names(
    train: pd.DataFrame, candidates: Sequence[str], threshold: float
) -> list[str]:
    """Apply training-fold finite/variance gates to a candidate list."""

    names = [str(c) for c in candidates if str(c) in train.columns]
    if not names:
        return []
    matrix = train[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    eligible: list[str] = []
    for idx, name in enumerate(names):
        values = matrix[:, idx]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            continue
        if float(len(finite)) / max(len(values), 1) < float(threshold):
            continue
        if float(np.min(finite)) == float(np.max(finite)):
            continue
        eligible.append(name)
    return eligible


def select_features(
    train: pd.DataFrame,
    candidates: list[str],
    top_k: int = 5,
    corr_threshold: float = 0.9,
    patient_col: str = "patient",
    y_col: str = "label_binary",
    *,
    non_missing_threshold: float = 0.70,
    strict_top_k: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    """Rank eligible features, prune correlated candidates and select up to top_k.

    Missingness and variance gates use training observations. When
    strict_top_k is false, ranked eligible candidates backfill a subset
    reduced by correlation pruning."""

    if not candidates:
        return [], pd.DataFrame(columns=["feature", "inner_macro_AUPRC", "n_inner"])
    if not 0.0 <= float(non_missing_threshold) <= 1.0:
        raise ValueError("non_missing_threshold must lie in [0, 1]")
    top_k = max(1, int(top_k))
    eligible = _eligible_feature_names(train, candidates, float(non_missing_threshold))
    if not eligible:
        return [], pd.DataFrame(columns=["feature", "inner_macro_AUPRC", "n_inner"])
    scores = _inner_feature_scores(
        train, eligible, patient_col=patient_col, y_col=y_col
    )
    selected: list[str] = []
    for column in scores["feature"].tolist():
        if len(selected) >= int(top_k):
            break
        good = True
        for kept in selected:
            rho = (
                train[column]
                .astype(float)
                .corr(train[kept].astype(float), method="spearman")
            )
            if np.isfinite(rho) and abs(float(rho)) >= float(corr_threshold):
                good = False
                break
        if good:
            selected.append(str(column))
    # Strict mode rejects a subset reduced by correlation pruning.
    # Otherwise, remaining eligible candidates backfill the ranked subset.
    if strict_top_k and len(selected) < min(int(top_k), len(scores)):
        return [], scores.assign(selection_status="insufficient_uncorrelated_features")
    if len(selected) < min(int(top_k), len(scores)):
        for column in scores["feature"].tolist():
            if column not in selected:
                selected.append(str(column))
            if len(selected) >= int(top_k):
                break
    scores = scores.copy()
    scores["selection_status"] = "eligible"
    return selected[: int(top_k)], scores


def _inner_oof_predictions(
    train: pd.DataFrame,
    selected: Sequence[str],
    c_value: float,
    *,
    patient_col: str = "patient",
    y_col: str = "label_binary",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate patient-isolated inner-OOF scores and direction diagnostics."""

    rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    patients = sorted(train[patient_col].dropna().astype(str).unique())
    for heldout in patients:
        inner_train = train[train[patient_col].astype(str) != heldout].copy()
        inner_test = train[train[patient_col].astype(str) == heldout].copy()
        if inner_train.empty or inner_test.empty:
            continue
        directions, diagnostics = _learn_directions(inner_train, selected, y_col=y_col)
        diagnostics.insert(0, "heldout_patient", heldout)
        direction_rows.extend(diagnostics.to_dict("records"))
        y_train_series = pd.to_numeric(inner_train[y_col], errors="coerce")
        y_test_series = pd.to_numeric(inner_test[y_col], errors="coerce")
        valid_train = y_train_series.notna().to_numpy()
        valid_test = y_test_series.notna().to_numpy()
        if not valid_train.any() or not valid_test.any():
            continue
        inner_train = inner_train.loc[valid_train].copy()
        inner_test = inner_test.loc[valid_test].copy()
        y_train = y_train_series.loc[valid_train].to_numpy(dtype=int)
        y_test = y_test_series.loc[valid_test].to_numpy(dtype=int)
        scores, _ = _fit_logistic(
            inner_train[list(selected)].to_numpy(dtype=float),
            y_train,
            inner_test[list(selected)].to_numpy(dtype=float),
            float(c_value),
            directions=directions,
        )
        for (_, item), score in zip(inner_test.iterrows(), scores):
            rows.append(
                {
                    "patient": str(item[patient_col]),
                    "heldout_patient": heldout,
                    "y_true": int(item[y_col]),
                    "y_score": float(score),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(direction_rows)


def _choose_c_threshold(
    train: pd.DataFrame,
    selected: Sequence[str],
    c_values: Iterable[float],
    *,
    patient_col: str = "patient",
    y_col: str = "label_binary",
    threshold_selection: str = "inner_oof_maximize_f1",
) -> tuple[float, float, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune C and the hard-decision threshold using inner patient-OOF only."""

    values_set: set[float] = set()
    iterable_c = c_values if c_values is not None else ()
    for value in iterable_c:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(parsed) and parsed > 0:
            values_set.add(parsed)
    values = sorted(values_set) or [1.0]
    if not selected:
        return values[0], 0.5, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    c_rows: list[dict[str, Any]] = []
    oof_by_c: dict[float, pd.DataFrame] = {}
    direction_by_c: dict[float, pd.DataFrame] = {}
    for c_value in values:
        oof, directions = _inner_oof_predictions(
            train, selected, c_value, patient_col=patient_col, y_col=y_col
        )
        oof_by_c[c_value] = oof
        direction_by_c[c_value] = directions
        metric = (
            patient_macro_metrics(oof.rename(columns={patient_col: "patient"}))
            if not oof.empty
            else {}
        )
        score = float(metric.get("patient_macro_AUPRC", np.nan))
        c_rows.append(
            {
                "C": float(c_value),
                "inner_macro_AUPRC": score,
                "n_oof": int(len(oof)),
                "n_metric_patients": int(metric.get("n_metric_patients", 0)),
            }
        )
    c_table = pd.DataFrame(c_rows)
    if (
        c_table.empty
        or not np.isfinite(
            pd.to_numeric(c_table["inner_macro_AUPRC"], errors="coerce")
        ).any()
    ):
        selected_c = values[0]
    else:
        c_table["_score"] = pd.to_numeric(
            c_table["inner_macro_AUPRC"], errors="coerce"
        ).fillna(-np.inf)
        selected_c = float(
            c_table.sort_values(
                ["_score", "C"], ascending=[False, True], kind="stable"
            ).iloc[0]["C"]
        )
        c_table = c_table.drop(columns=["_score"])
    selected_oof = oof_by_c.get(selected_c, pd.DataFrame())
    if str(threshold_selection).lower() in {"fixed", "fixed_0.5"}:
        threshold = 0.5
    else:
        threshold = select_threshold_max_f1(
            selected_oof.get("y_true", pd.Series(dtype=int)).to_numpy(dtype=int),
            selected_oof.get("y_score", pd.Series(dtype=float)).to_numpy(dtype=float),
        )
    return (
        selected_c,
        float(threshold),
        c_table,
        selected_oof,
        direction_by_c.get(selected_c, pd.DataFrame()),
    )
