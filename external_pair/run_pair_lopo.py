"""Fit patient-held-out models from external bipolar-pair feature tables."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# sklearn 1.9 emits a deprecation warning for the explicit ``penalty='l2'``
# argument used by the logistic model implementation.  It is unrelated to
# the scientific result and would otherwise flood long external runs.
warnings.filterwarnings(
    "ignore", category=FutureWarning, module=r"sklearn\\.linear_model"
)

# Shared model primitives are imported from the local analysis package.
PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pair_analysis.models import (  # noqa: E402
    _choose_c_threshold,
    _deployment_missing_stats,
    _fit_logistic,
    _learn_directions,
    feature_columns,
    feature_method,
    patient_macro_metrics,
    select_features,
)

PATIENTS = [
    "sub-HUP130",
    "sub-HUP134",
    "sub-HUP138",
    "sub-HUP139",
    "sub-HUP140",
    "sub-HUP146",
    "sub-HUP157",
    "sub-HUP160",
    "sub-HUP162",
    "sub-HUP163",
    "sub-HUP164",
    "sub-HUP166",
    "sub-HUP190",
]
COMMON_BANDS = ("delta", "theta", "alpha", "beta", "low_gamma", "high_gamma")
ALL_BANDS = COMMON_BANDS + ("ripple", "fast_ripple")
METHODS = ("line_length_rate", "absolute_band_power_multitaper")


def read_selected_feature_table(
    path: Path, band: str, patients: list[str]
) -> pd.DataFrame:
    """Read only keys and columns belonging to one band."""
    header = pd.read_csv(path, nrows=0)
    cols = list(header.columns)
    key = "patient" if "patient" in cols else "subject"
    pair = "bipolar_channel" if "bipolar_channel" in cols else "bipolar_pair"
    wanted = [key, pair]
    wanted += [c for c in cols if str(c).startswith(f"feature__{band}__")]
    if len(wanted) <= 2:
        raise ValueError(f"No feature columns found for band={band} in {path}")
    frame = pd.read_csv(path, usecols=wanted, low_memory=False)
    if key != "patient":
        frame = frame.rename(columns={key: "patient"})
    if pair != "bipolar_channel":
        frame = frame.rename(columns={pair: "bipolar_channel"})
    frame["patient"] = frame["patient"].astype(str).str.strip()
    frame["bipolar_channel"] = frame["bipolar_channel"].astype(str).str.strip()
    frame = frame[frame["patient"].isin(patients)].copy()
    for c in frame.columns:
        if c not in {"patient", "bipolar_channel"}:
            frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame.drop_duplicates(["patient", "bipolar_channel"], keep="first")


def load_labels(path: Path, patients: list[str]) -> pd.DataFrame:
    labels = pd.read_csv(path, low_memory=False)
    if "patient" not in labels.columns and "subject" in labels.columns:
        labels = labels.rename(columns={"subject": "patient"})
    if "bipolar_channel" not in labels.columns and "bipolar_pair" in labels.columns:
        labels = labels.rename(columns={"bipolar_pair": "bipolar_channel"})
    required = {"patient", "bipolar_channel", "label_binary", "primary_status"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label table missing columns: {sorted(missing)}")
    labels["patient"] = labels["patient"].astype(str).str.strip()
    labels["bipolar_channel"] = labels["bipolar_channel"].astype(str).str.strip()
    labels = labels[labels["patient"].isin(patients)].copy()
    labels = labels[
        labels["primary_status"].astype(str).str.lower().eq("primary")
    ].copy()
    labels["label_binary"] = pd.to_numeric(labels["label_binary"], errors="coerce")
    labels = labels[labels["label_binary"].isin([0, 1])].copy()
    return labels[["patient", "bipolar_channel", "label_binary"]].drop_duplicates(
        ["patient", "bipolar_channel"], keep="first"
    )


def run_band(
    frame: pd.DataFrame, band: str, controls: dict[str, Any]
) -> dict[str, pd.DataFrame | dict[str, Any]]:
    labels = controls["labels"]
    data = frame.merge(
        labels, on=["patient", "bipolar_channel"], how="inner", validate="one_to_one"
    )
    if data.empty:
        raise ValueError(f"No labelled rows after merge for {band}")
    candidates = [
        c for c in feature_columns(data, band) if feature_method(c) in METHODS
    ]
    if not candidates:
        raise ValueError(f"No model candidates for {band}")
    pred_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    c_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    for heldout in sorted(data["patient"].unique()):
        tr = data[data["patient"] != heldout].copy()
        te = data[data["patient"] == heldout].copy()
        base = {
            "frequency_band": band,
            "heldout_patient": heldout,
            "n_candidates": len(candidates),
            "status": "",
        }
        if tr["label_binary"].nunique() < 2:
            fold_rows.append(
                {
                    **base,
                    "status": "SKIP_CONSTANT_TRAINING",
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_selected": 0,
                    "selected_features": "",
                }
            )
            continue
        selected, scores = select_features(
            tr,
            candidates,
            top_k=controls["top_k"],
            corr_threshold=controls["corr_threshold"],
            patient_col="patient",
            y_col="label_binary",
            non_missing_threshold=controls["missing_threshold"],
            strict_top_k=False,
        )
        for item in scores.to_dict("records"):
            score_rows.append(
                {"frequency_band": band, "heldout_patient": heldout, **item}
            )
        if not selected:
            fold_rows.append(
                {
                    **base,
                    "status": "SKIP_NO_ELIGIBLE_FEATURES",
                    "n_train": len(tr),
                    "n_test": len(te),
                    "n_selected": 0,
                    "selected_features": "",
                }
            )
            continue
        c, threshold, ctab, selected_oof, _ = _choose_c_threshold(
            tr,
            selected,
            controls["c_values"],
            patient_col="patient",
            y_col="label_binary",
            threshold_selection=controls["threshold_selection"],
        )
        for item in ctab.to_dict("records"):
            c_rows.append({"frequency_band": band, "heldout_patient": heldout, **item})
        for item in selected_oof.to_dict("records"):
            inner_rows.append(
                {"frequency_band": band, "heldout_patient": heldout, **item}
            )
        directions, diagnostics = _learn_directions(tr, selected, y_col="label_binary")
        for item in diagnostics.to_dict("records"):
            direction_rows.append(
                {"frequency_band": band, "heldout_patient": heldout, **item}
            )
        scores_test, prep = _fit_logistic(
            tr[selected].to_numpy(float),
            tr.label_binary.to_numpy(int),
            te[selected].to_numpy(float),
            c,
            directions=directions,
        )
        miss_n, miss_frac = _deployment_missing_stats(te, selected)
        for idx, ((_, row), score) in enumerate(zip(te.iterrows(), scores_test)):
            pred_rows.append(
                {
                    "patient": row.patient,
                    "bipolar_channel": row.bipolar_channel,
                    "frequency_band": band,
                    "heldout_patient": heldout,
                    "y_true": int(row.label_binary),
                    "y_score": float(score),
                    "selected_threshold": float(threshold),
                    "y_pred": int(float(score) >= float(threshold)),
                    "feature_missing_count": int(miss_n[idx]),
                    "feature_missing_fraction": float(miss_frac[idx]),
                    "n_selected": len(selected),
                    "selected_C": float(c),
                }
            )
        fold_rows.append(
            {
                **base,
                "status": "COMPLETED",
                "n_train": len(tr),
                "n_test": len(te),
                "n_selected": len(selected),
                "selected_features": "|".join(selected),
                "selected_C": float(c),
                "selected_threshold": float(threshold),
                "n_inner_oof": len(selected_oof),
                "constant_training": bool(prep.get("constant_training", False)),
            }
        )
    pred = pd.DataFrame(pred_rows)
    folds = pd.DataFrame(fold_rows)
    metric = (
        patient_macro_metrics(pred)
        if not pred.empty
        else {
            "patient_macro_AUPRC": np.nan,
            "patient_macro_normalized_AUPRC": np.nan,
            "patient_macro_AUROC": np.nan,
            "n_metric_patients": 0,
        }
    )
    metrics = pd.DataFrame([{"frequency_band": band, **metric}])
    status = {
        "frequency_band": band,
        "status": "COMPLETED"
        if len(fold_rows) == len(data.patient.unique())
        and all(f.get("status") == "COMPLETED" for f in fold_rows)
        else "COMPLETED_PARTIAL",
        "n_patients": int(data.patient.nunique()),
        "n_labelled_pairs": int(len(data)),
        "n_positive": int(data.label_binary.sum()),
        "n_predictions": int(len(pred)),
        "n_completed_folds": int((folds.status == "COMPLETED").sum()),
        "n_failed_folds": int((folds.status != "COMPLETED").sum()),
        "n_candidates": int(len(candidates)),
        "modeling_controls": {k: v for k, v in controls.items() if k != "labels"},
    }
    return {
        "pred": pred,
        "folds": folds,
        "metrics": metrics,
        "feature_scores": pd.DataFrame(score_rows),
        "c_tuning": pd.DataFrame(c_rows),
        "directions": pd.DataFrame(direction_rows),
        "inner_oof": pd.DataFrame(inner_rows),
        "status": status,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Cohort root containing labels/external_pair_labels_13_1024hz.csv",
    )
    ap.add_argument(
        "--feature-root",
        type=Path,
        required=True,
        help="Reviewed final feature-table directory; raw extraction is separate",
    )
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument(
        "--bands", nargs="+", default=list(ALL_BANDS), choices=list(ALL_BANDS)
    )
    ap.add_argument("--patients", nargs="+", default=PATIENTS, choices=PATIENTS)
    args = ap.parse_args()
    data_root = args.data_root.resolve()
    run_features = args.feature_root.resolve()
    from pair_analysis.utils import assert_safe_output_root

    out = assert_safe_output_root(args.output_root, [data_root, run_features])
    labels_path = data_root / "labels" / "external_pair_labels_13_1024hz.csv"
    labels = load_labels(labels_path, args.patients)
    sources = {
        b: (
            run_features / "patient_features_label_blind.csv"
            if b in COMMON_BANDS
            else run_features / f"native_{b}_1024hz_patient_features_label_blind.csv"
        )
        for b in args.bands
    }
    controls = {
        "labels": labels,
        "top_k": 5,
        "missing_threshold": 0.70,
        "corr_threshold": 0.90,
        "c_values": [0.01, 0.1, 1.0, 10.0],
        "threshold_selection": "inner_oof_maximize_f1",
    }
    all_metrics = []
    all_status = []
    for band in args.bands:
        source = sources[band]
        if not source.is_file():
            raise FileNotFoundError(source)
        frame = read_selected_feature_table(source, band, args.patients)
        result = run_band(frame, band, controls)
        for name, df in [
            ("predictions", result["pred"]),
            ("folds", result["folds"]),
            ("metrics", result["metrics"]),
            ("feature_scores", result["feature_scores"]),
            ("c_tuning", result["c_tuning"]),
            ("directions", result["directions"]),
            ("inner_oof", result["inner_oof"]),
        ]:
            df.to_csv(out / f"{name}_{band}.csv", index=False)
        (out / f"status_{band}.json").write_text(
            json.dumps(result["status"], indent=2), encoding="utf-8"
        )
        all_metrics.append(result["metrics"])
        all_status.append(result["status"])
        print(
            f"{band}: {result['status']['status']} predictions={result['status']['n_predictions']} AUPRC={result['metrics'].iloc[0]['patient_macro_AUPRC']:.6f}",
            flush=True,
        )
    metrics = (
        pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    )
    metrics.to_csv(out / "target_nested_lopo_metrics_all_bands.csv", index=False)
    status = {
        "status": "COMPLETED"
        if all(s["status"] == "COMPLETED" for s in all_status)
        else "COMPLETED_PARTIAL",
        "patients": args.patients,
        "bands": args.bands,
        "band_status": all_status,
        "labels_path": str(labels_path),
        "label_rows": int(len(labels)),
        "source_feature_root": str(run_features),
    }
    (out / "target_nested_lopo_status_all_bands.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    (out / "run_parameters.json").write_text(
        json.dumps(
            {
                "patients": args.patients,
                "bands": args.bands,
                "controls": {k: v for k, v in controls.items() if k != "labels"},
                "source_feature_root": str(run_features),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
