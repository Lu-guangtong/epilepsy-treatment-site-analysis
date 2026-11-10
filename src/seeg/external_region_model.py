"""Patient-isolated regional classification and patient-level inference."""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import RobustScaler

from .external_region_contract import (
    KEYS, candidate_features, load_analysis_module, read_json, sha256, write_json,
)


def join_model_data(features: pd.DataFrame, labels: pd.DataFrame,
                    band: str, patients: list[str]) -> pd.DataFrame:
    """Join labels at the model boundary with exact patient-region coverage."""
    columns = candidate_features(band)
    if set(KEYS + columns) - set(features.columns):
        raise ValueError("Patient feature table is missing required columns")
    extras = [c for c in features if c not in KEYS + columns and not c.startswith("qc__")]
    if extras:
        raise ValueError(f"Unexpected columns in unlabeled features: {extras}")
    if set(features.patient) != set(patients) or features.duplicated(KEYS).any():
        raise ValueError("Unexpected or duplicate feature patients/regions")
    targets = labels[labels.patient.isin(patients)][KEYS + ["label_binary"]].copy()
    gray = features[features.tissue_class == "gray"][KEYS + columns].copy()
    if targets.duplicated(KEYS).any():
        raise ValueError("Duplicate model label keys")
    data = gray.merge(targets, on=KEYS, how="outer", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all():
        raise ValueError("Gray feature and label region keys do not match exactly")
    data = data.drop(columns="_merge").sort_values(KEYS).reset_index(drop=True)
    if np.isinf(data[columns].to_numpy(float)).any():
        raise ValueError("Infinite predictive feature values")
    for patient, frame in data.groupby("patient"):
        if set(frame.label_binary) != {0, 1}:
            raise ValueError(f"Patient lacks two classes: {patient}")
    return data


def rank_features(train: pd.DataFrame, candidates: list[str], top_k: int,
                  non_missing: float) -> tuple[list[str], pd.DataFrame]:
    """All filtering and univariate orientation are fitted on training patients."""
    x = train[candidates].to_numpy(float)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        mins, maxs = np.nanmin(x, axis=0), np.nanmax(x, axis=0)
    valid = ((np.isfinite(x).mean(axis=0) >= non_missing)
             & np.isfinite(mins) & np.isfinite(maxs) & (mins != maxs))
    eligible = [c for c, ok in zip(candidates, valid, strict=True) if ok]
    if not eligible or train.patient.nunique() < 2:
        raise ValueError("Feature selection needs >=2 training patients and nonconstant features")
    formal = load_analysis_module("run_region_lopo_modeling")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
        ranking = formal.score_features_inner_auprc_array(
            x[:, valid], train.label_binary.to_numpy(int), train.patient.to_numpy(), eligible)
    ranking = ranking[np.isfinite(ranking.inner_auprc)].sort_values(
        ["inner_auprc", "feature"], ascending=[False, True], kind="stable")
    selected = ranking.feature.head(top_k).tolist()
    if not selected:
        raise ValueError("No feature has a valid training-only univariate score")
    return selected, ranking


def fit_scores(train: pd.DataFrame, test: pd.DataFrame, columns: list[str],
               c_value: float, seed: int) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(quantile_range=(25, 75))
    x_train = scaler.fit_transform(imputer.fit_transform(train[columns]))
    x_test = scaler.transform(imputer.transform(test[columns]))
    model = LogisticRegression(C=c_value, l1_ratio=0.0, solver="liblinear",
                               class_weight="balanced", random_state=seed, max_iter=2000)
    model.fit(x_train, train.label_binary.to_numpy(int))
    scores = model.predict_proba(x_test)[:, 1]
    if not np.isfinite(scores).all():
        raise ValueError("Classifier produced nonfinite predictions")
    return scores


def fit_outer_fold(data: pd.DataFrame, band: str, heldout: str, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Re-select features inside EACH C-validation training split.

    The final feature subset is selected again on all outer-training patients.
    This avoids reusing a subset informed by an inner validation patient's label.
    """
    train, test = data[data.patient != heldout], data[data.patient == heldout]
    if train.patient.nunique() < 3 or test.empty:
        raise ValueError("Nested regional LOPO requires at least four patients")
    candidates = candidate_features(band)
    c_values = sorted(cfg["c_values"])
    scores_by_c = {c: [] for c in c_values}
    inner_audit = []
    for validation_patient in sorted(train.patient.unique()):
        inner_train = train[train.patient != validation_patient]
        inner_test = train[train.patient == validation_patient]
        selected, _ = rank_features(inner_train, candidates, cfg["top_k_features"],
                                    cfg["non_missing_threshold"])
        values = {}
        for c_value in c_values:
            prediction = fit_scores(inner_train, inner_test, selected, c_value, cfg["seed"])
            value = float(average_precision_score(inner_test.label_binary, prediction))
            scores_by_c[c_value].append(value)
            values[str(c_value)] = value
        inner_audit.append({"validation_patient": validation_patient,
                            "training_patients": sorted(inner_train.patient.unique()),
                            "selected_features": selected, "average_precision_by_C": values})
    c_scores = {c: float(np.mean(values)) for c, values in scores_by_c.items()}
    selected_c = min(c_values, key=lambda c: (-c_scores[c], c))
    selected, ranking = rank_features(train, candidates, cfg["top_k_features"],
                                      cfg["non_missing_threshold"])
    scores = fit_scores(train, test, selected, selected_c, cfg["seed"])
    result = test[KEYS].copy()
    result["frequency_band"] = band
    result["heldout_patient"] = heldout
    result["y_true"] = test.label_binary.to_numpy(int)
    result["y_score"] = scores
    result["rank_within_patient"] = pd.Series(scores).rank(method="average", ascending=False).to_numpy()
    result["selected_C"] = selected_c
    result["selected_features"] = "|".join(selected)
    audit = {"heldout_patient": heldout, "training_patients": sorted(train.patient.unique()),
             "selected_C": selected_c, "inner_patient_macro_AP_by_C": c_scores,
             "selected_features": selected, "inner_folds": inner_audit,
             "final_training_feature_ranking": ranking.to_dict("records")}
    return result, audit


def run_lopo(data: pd.DataFrame, band: str, cfg: dict, progress: bool = True) -> tuple[pd.DataFrame, list[dict]]:
    predictions, audits = [], []
    for heldout in sorted(data.patient.unique()):
        if progress:
            print(f"model {band} heldout={heldout}", flush=True)
        prediction, audit = fit_outer_fold(data, band, heldout, cfg)
        predictions.append(prediction)
        audits.append(audit)
    return pd.concat(predictions, ignore_index=True), audits


def patient_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient, group in predictions.groupby("patient", sort=True):
        y, score = group.y_true.to_numpy(int), group.y_score.to_numpy(float)
        if set(y) != {0, 1} or not np.isfinite(score).all():
            raise ValueError(f"Invalid within-patient predictions: {patient}")
        ap, prevalence = float(average_precision_score(y, score)), float(y.mean())
        rows.append({"patient": patient, "n_regions": len(y), "n_positive_regions": int(y.sum()),
                     "prevalence": prevalence, "AUPRC": ap,
                     "AUROC": float(roc_auc_score(y, score)),
                     "normalized_AUPRC": (ap - prevalence) / (1 - prevalence)})
    return pd.DataFrame(rows)


def bootstrap_metrics(metrics: pd.DataFrame, n_bootstrap: int, seed: int) -> dict:
    """Resample whole patients with replacement, giving each equal weight."""
    columns = ["AUPRC", "AUROC", "normalized_AUPRC"]
    values = metrics[columns].to_numpy(float)
    result = {c: float(metrics[c].mean()) for c in columns}
    result.update({"n_patients": len(metrics), "n_regions": int(metrics.n_regions.sum()),
                   "n_positive_regions": int(metrics.n_positive_regions.sum()),
                   "primary_metric": "patient_macro_AUPRC", "n_bootstrap": n_bootstrap})
    if n_bootstrap:
        rng = np.random.default_rng(seed)
        sampled = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
        means = values[sampled].mean(axis=1)
        for i, column in enumerate(columns):
            result[f"{column}_ci_low"] = float(np.quantile(means[:, i], 0.025))
            result[f"{column}_ci_high"] = float(np.quantile(means[:, i], 0.975))
    return result


def permute_labels(data: pd.DataFrame, seed: int, permutation: int) -> pd.DataFrame:
    rng = np.random.default_rng(np.random.SeedSequence([seed, permutation]))
    result = data.copy()
    for patient in sorted(result.patient.unique()):
        mask = result.patient == patient
        result.loc[mask, "label_binary"] = rng.permutation(result.loc[mask, "label_binary"].to_numpy())
    return result


def permutation_test(data: pd.DataFrame, band: str, cfg: dict, observed: float,
                     n_permutations: int, path: Path, fingerprint: str, resume: bool) -> dict:
    nulls = []
    if path.exists():
        saved = read_json(path)
        if (not resume or saved["fingerprint"] != fingerprint
                or saved["frequency_band"] != band or saved["observed_AUPRC"] != observed
                or saved["target_permutations"] != n_permutations):
            raise ValueError("Permutation checkpoint mismatch")
        nulls = saved["null_AUPRC"]
        if len(nulls) > n_permutations or not np.isfinite(nulls).all():
            raise ValueError("Invalid permutation checkpoint")
    for i in range(len(nulls) + 1, n_permutations + 1):
        print(f"permutation {band} starting {i}/{n_permutations}", flush=True)
        shuffled = permute_labels(data, cfg["seed"], i)
        predictions, _ = run_lopo(shuffled, band, cfg, progress=False)
        nulls.append(float(patient_metrics(predictions).AUPRC.mean()))
        if i % 25 == 0 or i == n_permutations:
            write_json(path, {"fingerprint": fingerprint, "frequency_band": band,
                "observed_AUPRC": observed, "target_permutations": n_permutations,
                "null_AUPRC": nulls})
            print(f"permutation {band} {i}/{n_permutations}", flush=True)
    return {"n_permutations": n_permutations,
            "permutation_p_AUPRC": (1 + int(np.sum(np.asarray(nulls) >= observed))) /
                                   (n_permutations + 1) if n_permutations else None}


def model_band(feature_dir: Path, labels: pd.DataFrame, band: str, cfg: dict,
               out: Path, feature_fingerprint: str, model_fingerprint: str,
               resume: bool, n_permutations: int, n_bootstrap: int) -> dict:
    manifest = read_json(feature_dir / "feature_manifest.json")
    if manifest["status"] != "complete" or manifest["fingerprint"] != feature_fingerprint:
        raise ValueError("Incomplete or mismatched feature manifest")
    entry = manifest["files"]["region_patient_features_unlabeled"]
    path = Path(entry["path"])
    if sha256(path) != entry["sha256"]:
        raise ValueError("Patient feature SHA256 mismatch")
    features = pd.read_csv(path)
    data = join_model_data(features, labels, band, manifest["patients"])
    out.mkdir(parents=True, exist_ok=True)
    predictions, folds = run_lopo(data, band, cfg)
    by_patient = patient_metrics(predictions)
    by_patient["native_sampling_hz"] = by_patient.patient.map(manifest["patient_native_sampling_hz"])
    summary = bootstrap_metrics(by_patient, n_bootstrap, cfg["seed"])
    summary.update(permutation_test(data, band, cfg, summary["AUPRC"], n_permutations,
                                   out / "permutation_checkpoint.json", model_fingerprint, resume))
    predictions.to_csv(out / "region_predictions.csv", index=False, encoding="utf-8-sig")
    by_patient.to_csv(out / "patient_metrics.csv", index=False, encoding="utf-8-sig")
    strata = [{"native_sampling_hz": int(fs), **bootstrap_metrics(group, n_bootstrap, cfg["seed"])}
              for fs, group in by_patient.groupby("native_sampling_hz")]
    pd.DataFrame(strata).to_csv(out / "sampling_strata_metrics.csv", index=False, encoding="utf-8-sig")
    data[KEYS + ["label_binary"]].to_csv(out / "model_labels.csv", index=False, encoding="utf-8-sig")
    write_json(out / "fold_audit.json", folds)
    write_json(out / "metrics.json", {"frequency_band": band, **summary})
    return {"frequency_band": band, **summary}
