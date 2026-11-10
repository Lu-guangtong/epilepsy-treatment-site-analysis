"""Link held-out region rankings to residual errors from the contact model."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "region_contact_residual_burden.yaml"
OUTPUT_ROOT = ROOT / "data_derived" / "region_contact_residual_burden"
CHECK_ROOT = ROOT / "checks"

REGION_KEYS = ["patient", "desikan_killiany_label"]
REGION_BAND_KEYS = ["frequency_band", *REGION_KEYS]
CONTACT_BAND_KEYS = ["frequency_band", "patient", "bipolar_channel"]


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(path: Path) -> None:
    assert_output_path(path)
    resolved = _resolve(path)
    formal_root = _resolve(OUTPUT_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if resolved == AUX_ROOT or _is_relative_to(resolved, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {resolved}")
    if not (
        resolved == formal_root
        or _is_relative_to(resolved, formal_root)
        or resolved == tmp_root
        or _is_relative_to(resolved, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = expand_config(yaml.safe_load(handle))
    if not isinstance(config, dict):
        raise SystemExit(f"Invalid configuration: {path}")
    return config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_columns(frame: pd.DataFrame, required: set[str], source: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"{source}: missing columns {missing}")


def _finite_probability(values: pd.Series, field: str, source: Path) -> None:
    numeric = values.to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or np.any((numeric < 0.0) | (numeric > 1.0)):
        raise SystemExit(f"{source}: {field} must contain finite values in [0, 1]")


def _filter_requested(
    frame: pd.DataFrame,
    *,
    bands: Sequence[str],
    patients: Sequence[str] | None,
) -> pd.DataFrame:
    selected = frame[frame["frequency_band"].astype(str).isin(bands)].copy()
    if patients is not None:
        patient_set = {str(value) for value in patients}
        selected = selected[selected["patient"].astype(str).isin(patient_set)].copy()
    if selected.empty:
        raise SystemExit("No rows remain after applying patient/frequency-band filters")
    return selected


def load_region_labels(path: Path, patients: Sequence[str] | None = None) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing region label table: {path}")
    frame = pd.read_csv(path)
    required = {
        *REGION_KEYS,
        "suspicious_region_status",
        "label_binary",
        "inclusion_status",
        "n_bipolar_pairs_after_bad_channel_removal",
    }
    _require_columns(frame, required, path)
    frame = frame[frame["inclusion_status"].astype(str).eq("primary")].copy()
    if patients is not None:
        frame = frame[frame["patient"].astype(str).isin({str(p) for p in patients})].copy()
    if frame.empty:
        raise SystemExit("No primary region labels remain")
    if frame.duplicated(REGION_KEYS).any():
        raise SystemExit("Primary region labels contain duplicate patient-region rows")
    if not frame["suspicious_region_status"].isin(["yes", "no"]).all():
        raise SystemExit("Primary region labels must contain only yes/no status")
    if not frame["label_binary"].isin([0, 1]).all():
        raise SystemExit("Primary region labels must contain binary labels")
    expected_binary = frame["suspicious_region_status"].map({"no": 0, "yes": 1})
    if not np.array_equal(
        expected_binary.to_numpy(dtype=int), frame["label_binary"].to_numpy(dtype=int)
    ):
        raise SystemExit("Region status and binary labels disagree")
    counts = frame["n_bipolar_pairs_after_bad_channel_removal"].to_numpy(dtype=int)
    if np.any(counts <= 0):
        raise SystemExit("Primary region labels must have positive signal coverage")
    return frame.sort_values(REGION_KEYS).reset_index(drop=True)


def load_region_predictions(
    path: Path,
    bands: Sequence[str],
    patients: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing region prediction table: {path}")
    frame = pd.read_csv(path)
    required = {
        *REGION_BAND_KEYS,
        "heldout_patient",
        "y_true",
        "y_score",
        "rank_within_patient",
    }
    _require_columns(frame, required, path)
    frame = _filter_requested(frame, bands=bands, patients=patients)
    if frame.duplicated(REGION_BAND_KEYS).any():
        raise SystemExit("Region predictions contain duplicate patient-region-band rows")
    if not frame["patient"].astype(str).eq(frame["heldout_patient"].astype(str)).all():
        raise SystemExit("Region prediction table contains non-heldout patient rows")
    if not frame["y_true"].isin([0, 1]).all():
        raise SystemExit("Region prediction labels must be binary")
    _finite_probability(frame["y_score"], "y_score", path)
    if not np.isfinite(frame["rank_within_patient"].to_numpy(dtype=np.float64)).all():
        raise SystemExit("Region ranks must be finite")
    return frame.sort_values(REGION_BAND_KEYS).reset_index(drop=True)


def load_contact_predictions(
    path: Path,
    bands: Sequence[str],
    patients: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing contact prediction table: {path}")
    frame = pd.read_csv(path)
    required = {
        *CONTACT_BAND_KEYS,
        "heldout_patient",
        "desikan_killiany_label",
        "y_true",
        "y_score",
        "y_pred",
    }
    _require_columns(frame, required, path)
    frame = _filter_requested(frame, bands=bands, patients=patients)
    if frame.duplicated(CONTACT_BAND_KEYS).any():
        raise SystemExit("Contact predictions contain duplicate patient-contact-band rows")
    if not frame["patient"].astype(str).eq(frame["heldout_patient"].astype(str)).all():
        raise SystemExit("Contact prediction table contains non-heldout patient rows")
    if not frame["y_true"].isin([0, 1]).all() or not frame["y_pred"].isin([0, 1]).all():
        raise SystemExit("Contact y_true/y_pred values must be binary")
    _finite_probability(frame["y_score"], "y_score", path)
    identity_columns = ["patient", "bipolar_channel", "desikan_killiany_label", "y_true"]
    identity_counts = frame.groupby(["patient", "bipolar_channel"], sort=False)[
        identity_columns[2:]
    ].nunique(dropna=False)
    if (identity_counts > 1).any().any():
        raise SystemExit("Contact identity, region, or label changes across frequency bands")
    return frame.sort_values(CONTACT_BAND_KEYS).reset_index(drop=True)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    num = numerator.to_numpy(dtype=np.float64)
    den = denominator.to_numpy(dtype=np.float64)
    values = np.full(len(num), np.nan, dtype=np.float64)
    np.divide(num, den, out=values, where=den > 0)
    return pd.Series(values, index=numerator.index)


def build_joint_tables(
    region_predictions: pd.DataFrame,
    contact_predictions: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    label_columns = [
        *REGION_KEYS,
        "suspicious_region_status",
        "label_binary",
        "n_bipolar_pairs_after_bad_channel_removal",
    ]
    region = region_predictions.merge(
        labels[label_columns],
        on=REGION_KEYS,
        how="left",
        validate="many_to_one",
    )
    if region["suspicious_region_status"].isna().any():
        missing = region.loc[region["suspicious_region_status"].isna(), REGION_KEYS]
        raise SystemExit(f"Region predictions missing locked labels: {missing.head().to_dict('records')}")
    if not np.array_equal(
        region["y_true"].to_numpy(dtype=int), region["label_binary"].to_numpy(dtype=int)
    ):
        raise SystemExit("Region held-out labels disagree with locked ablation-top4 labels")
    region = region.rename(
        columns={
            "y_true": "region_y_true",
            "y_score": "region_y_score",
            "rank_within_patient": "region_rank_within_patient",
        }
    )

    contact = contact_predictions.merge(
        labels[label_columns],
        on=REGION_KEYS,
        how="left",
        validate="many_to_one",
    )
    if contact["suspicious_region_status"].isna().any():
        missing = contact.loc[
            contact["suspicious_region_status"].isna(),
            ["patient", "bipolar_channel", "desikan_killiany_label"],
        ]
        raise SystemExit(f"Contacts missing primary-region mapping: {missing.head().to_dict('records')}")
    region_scores = region[
        [
            *REGION_BAND_KEYS,
            "region_y_true",
            "region_y_score",
            "region_rank_within_patient",
        ]
    ]
    contact = contact.merge(
        region_scores,
        on=REGION_BAND_KEYS,
        how="left",
        validate="many_to_one",
    )
    if contact["region_y_score"].isna().any():
        raise SystemExit("Contacts are missing held-out region scores")
    if not np.array_equal(
        contact["region_y_true"].to_numpy(dtype=int),
        contact["label_binary"].to_numpy(dtype=int),
    ):
        raise SystemExit("Contact region mapping disagrees with locked region labels")

    contact["classification_state"] = np.select(
        [
            contact["y_true"].eq(1) & contact["y_pred"].eq(1),
            contact["y_true"].eq(1) & contact["y_pred"].eq(0),
            contact["y_true"].eq(0) & contact["y_pred"].eq(1),
            contact["y_true"].eq(0) & contact["y_pred"].eq(0),
        ],
        ["TP", "FN", "FP", "TN"],
        default="ERROR",
    )
    if contact["classification_state"].eq("ERROR").any():
        raise SystemExit("Could not assign a contact classification state")

    for state in ["TP", "FN", "FP", "TN"]:
        contact[f"is_{state.lower()}"] = contact["classification_state"].eq(state).astype(int)
    grouped = contact.groupby(REGION_BAND_KEYS, sort=True, as_index=False).agg(
        n_contacts=("bipolar_channel", "size"),
        n_doctor_ablated=("y_true", "sum"),
        n_model_predicted_ablated=("y_pred", "sum"),
        n_tp=("is_tp", "sum"),
        n_fn=("is_fn", "sum"),
        n_fp=("is_fp", "sum"),
        n_tn=("is_tn", "sum"),
        contact_score_mean=("y_score", "mean"),
        contact_score_median=("y_score", "median"),
    )
    region = region.merge(grouped, on=REGION_BAND_KEYS, how="left", validate="one_to_one")
    count_columns = [
        "n_contacts",
        "n_doctor_ablated",
        "n_model_predicted_ablated",
        "n_tp",
        "n_fn",
        "n_fp",
        "n_tn",
    ]
    if region[count_columns].isna().any().any():
        raise SystemExit("Some primary region predictions have no mapped contacts")
    for column in count_columns:
        region[column] = region[column].astype(int)
    if not np.array_equal(
        region["n_contacts"].to_numpy(dtype=int),
        region["n_bipolar_pairs_after_bad_channel_removal"].to_numpy(dtype=int),
    ):
        raise SystemExit("Region contact totals disagree with the locked label audit")
    if not np.array_equal(
        region["n_contacts"].to_numpy(dtype=int),
        (region["n_tp"] + region["n_fn"] + region["n_fp"] + region["n_tn"]).to_numpy(
            dtype=int
        ),
    ):
        raise SystemExit("Region confusion counts do not sum to the number of contacts")
    region["contact_sensitivity"] = _safe_ratio(region["n_tp"], region["n_doctor_ablated"])
    region["residual_fn_rate"] = _safe_ratio(region["n_fn"], region["n_doctor_ablated"])
    region["contact_false_positive_rate"] = _safe_ratio(
        region["n_fp"], region["n_fp"] + region["n_tn"]
    )
    region["has_residual_fn"] = region["n_fn"].gt(0)
    return (
        contact.sort_values(CONTACT_BAND_KEYS).reset_index(drop=True),
        region.sort_values(REGION_BAND_KEYS).reset_index(drop=True),
    )


def summarize_suspicious_burden(
    region_table: pd.DataFrame,
    suspicious_status: str = "yes",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = region_table[
        region_table["suspicious_region_status"].astype(str).eq(suspicious_status)
    ].copy()
    if selected.empty:
        raise SystemExit("No suspicious regions remain for residual-burden analysis")
    patient = selected.groupby(["frequency_band", "patient"], sort=True, as_index=False).agg(
        n_suspicious_regions=("desikan_killiany_label", "size"),
        n_suspicious_regions_with_fn=("has_residual_fn", "sum"),
        n_contacts=("n_contacts", "sum"),
        n_doctor_ablated=("n_doctor_ablated", "sum"),
        n_tp=("n_tp", "sum"),
        n_fn=("n_fn", "sum"),
    )
    patient["residual_fn_rate"] = _safe_ratio(patient["n_fn"], patient["n_doctor_ablated"])
    patient["has_residual_fn"] = patient["n_fn"].gt(0)

    band = selected.groupby("frequency_band", sort=True, as_index=False).agg(
        n_patients=("patient", "nunique"),
        n_suspicious_regions=("desikan_killiany_label", "size"),
        n_suspicious_regions_with_fn=("has_residual_fn", "sum"),
        n_contacts=("n_contacts", "sum"),
        n_doctor_ablated=("n_doctor_ablated", "sum"),
        n_tp=("n_tp", "sum"),
        n_fn=("n_fn", "sum"),
    )
    patients_with_fn = patient.groupby("frequency_band", sort=True)["has_residual_fn"].sum()
    macro_fnr = patient.groupby("frequency_band", sort=True)["residual_fn_rate"].mean()
    band["n_patients_with_fn"] = band["frequency_band"].map(patients_with_fn).astype(int)
    band["pooled_residual_fn_rate"] = _safe_ratio(band["n_fn"], band["n_doctor_ablated"])
    band["patient_macro_residual_fn_rate"] = band["frequency_band"].map(macro_fnr)
    return patient, band


def compare_region_status_fn_rates(
    region_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_status = region_table.groupby(
        ["frequency_band", "patient", "suspicious_region_status"],
        sort=True,
        as_index=False,
    ).agg(
        n_regions=("desikan_killiany_label", "size"),
        n_contacts=("n_contacts", "sum"),
        n_doctor_ablated=("n_doctor_ablated", "sum"),
        n_tp=("n_tp", "sum"),
        n_fn=("n_fn", "sum"),
    )
    patient_status["residual_fn_rate"] = _safe_ratio(
        patient_status["n_fn"], patient_status["n_doctor_ablated"]
    )

    paired = patient_status.pivot(
        index=["frequency_band", "patient"],
        columns="suspicious_region_status",
        values=["n_doctor_ablated", "n_fn", "residual_fn_rate"],
    )
    paired.columns = [f"{field}_{status}" for field, status in paired.columns]
    paired = paired.reset_index()
    required = ["residual_fn_rate_yes", "residual_fn_rate_no"]
    for column in required:
        if column not in paired.columns:
            paired[column] = np.nan
    paired["paired_comparison_evaluable"] = paired[required].notna().all(axis=1)
    paired["suspicious_minus_non_suspicious_fn_rate"] = (
        paired["residual_fn_rate_yes"] - paired["residual_fn_rate_no"]
    )

    records: list[dict[str, object]] = []
    for band, band_status in patient_status.groupby("frequency_band", sort=True):
        band_paired = paired[
            paired["frequency_band"].astype(str).eq(str(band))
            & paired["paired_comparison_evaluable"]
        ]
        status_lookup = {
            status: frame
            for status, frame in band_status.groupby("suspicious_region_status", sort=True)
        }
        yes = status_lookup.get("yes", pd.DataFrame())
        no = status_lookup.get("no", pd.DataFrame())
        records.append(
            {
                "frequency_band": band,
                "n_paired_patients": len(band_paired),
                "n_suspicious_doctor_ablated": int(yes["n_doctor_ablated"].sum()),
                "n_suspicious_fn": int(yes["n_fn"].sum()),
                "n_non_suspicious_doctor_ablated": int(no["n_doctor_ablated"].sum()),
                "n_non_suspicious_fn": int(no["n_fn"].sum()),
                "pooled_suspicious_fn_rate": float(yes["n_fn"].sum() / yes["n_doctor_ablated"].sum()),
                "pooled_non_suspicious_fn_rate": float(no["n_fn"].sum() / no["n_doctor_ablated"].sum()),
                "paired_patient_macro_suspicious_fn_rate": float(
                    band_paired["residual_fn_rate_yes"].mean()
                ),
                "paired_patient_macro_non_suspicious_fn_rate": float(
                    band_paired["residual_fn_rate_no"].mean()
                ),
                "paired_patient_macro_fn_rate_difference": float(
                    band_paired["suspicious_minus_non_suspicious_fn_rate"].mean()
                ),
            }
        )
    return patient_status, paired, pd.DataFrame.from_records(records)


def _weighted_linear_association(
    frame: pd.DataFrame,
    patient_draw: Sequence[str] | None = None,
) -> dict[str, float | int]:
    eligible = frame[
        frame["n_doctor_ablated"].gt(0)
        & frame["residual_fn_rate"].notna()
        & frame["region_y_score"].notna()
    ].copy()
    if patient_draw is None:
        patient_draw = sorted(eligible["patient"].astype(str).unique())
    parts: list[pd.DataFrame] = []
    for copy_index, patient in enumerate(patient_draw):
        part = eligible[eligible["patient"].astype(str).eq(str(patient))].copy()
        if part.empty:
            continue
        total = float(part["n_doctor_ablated"].sum())
        if total <= 0:
            continue
        part["analysis_weight"] = part["n_doctor_ablated"].astype(float) / total
        part["bootstrap_patient_copy"] = copy_index
        parts.append(part)
    if not parts:
        return {
            "n_patients": 0,
            "n_regions": 0,
            "n_doctor_ablated": 0,
            "weighted_slope": np.nan,
            "weighted_correlation": np.nan,
        }
    data = pd.concat(parts, ignore_index=True)
    x = data["region_y_score"].to_numpy(dtype=np.float64)
    y = data["residual_fn_rate"].to_numpy(dtype=np.float64)
    w = data["analysis_weight"].to_numpy(dtype=np.float64)
    weight_sum = float(w.sum())
    x_mean = float(np.sum(w * x) / weight_sum)
    y_mean = float(np.sum(w * y) / weight_sum)
    x_centered = x - x_mean
    y_centered = y - y_mean
    x_var = float(np.sum(w * x_centered**2) / weight_sum)
    y_var = float(np.sum(w * y_centered**2) / weight_sum)
    covariance = float(np.sum(w * x_centered * y_centered) / weight_sum)
    slope = covariance / x_var if x_var > 0 else np.nan
    correlation = covariance / np.sqrt(x_var * y_var) if x_var > 0 and y_var > 0 else np.nan
    return {
        "n_patients": len(parts),
        "n_regions": len(data),
        "n_doctor_ablated": int(data["n_doctor_ablated"].sum()),
        "weighted_slope": float(slope),
        "weighted_correlation": float(correlation),
    }


def region_score_associations(region_table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    scopes = {
        "all_primary_regions_with_ablation": pd.Series(True, index=region_table.index),
        "suspicious_regions_with_ablation": region_table["suspicious_region_status"].eq("yes"),
    }
    for band, band_frame in region_table.groupby("frequency_band", sort=True):
        for scope, mask in scopes.items():
            selected = band_frame.loc[mask.loc[band_frame.index]]
            records.append(
                {
                    "frequency_band": band,
                    "analysis_scope": scope,
                    **_weighted_linear_association(selected),
                }
            )
    return pd.DataFrame.from_records(records)


def _ordered_capture_curve(
    frame: pd.DataFrame,
    order: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    ordered = frame.iloc[order]
    total_contacts = int(ordered["n_contacts"].sum())
    total_fn = int(ordered["n_fn"].sum())
    if total_contacts <= 0 or total_fn <= 0:
        raise ValueError("Capture curves require positive contact and FN totals")
    x = np.r_[0.0, ordered["n_contacts"].cumsum().to_numpy(dtype=float) / total_contacts]
    y = np.r_[0.0, ordered["n_fn"].cumsum().to_numpy(dtype=float) / total_fn]
    return x, y, float(np.trapezoid(y, x))


def _stable_seed(base_seed: int, *values: object) -> int:
    text = "|".join(str(value) for value in values)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32)


def build_capture_analysis(
    region_table: pd.DataFrame,
    *,
    n_random_rankings: int,
    grid_points: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_random_rankings < 1:
        raise SystemExit("random_region_rankings must be positive")
    if grid_points < 2:
        raise SystemExit("capture_curve_grid_points must be at least 2")
    metric_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []
    grid = np.linspace(0.0, 1.0, grid_points)
    for (band, patient), frame in region_table.groupby(
        ["frequency_band", "patient"], sort=True
    ):
        data = frame.sort_values("desikan_killiany_label").reset_index(drop=True)
        n_fn = int(data["n_fn"].sum())
        if n_fn == 0:
            metric_records.append(
                {
                    "frequency_band": band,
                    "patient": patient,
                    "n_regions": len(data),
                    "n_contacts": int(data["n_contacts"].sum()),
                    "n_residual_fn": 0,
                    "qc_status": "not_computable_no_residual_fn",
                    "network_capture_auc": np.nan,
                    "coverage_capture_auc": np.nan,
                    "random_capture_auc_mean": np.nan,
                    "network_minus_random_auc": np.nan,
                    "network_minus_coverage_auc": np.nan,
                }
            )
            continue
        network_order = np.lexsort(
            (data["desikan_killiany_label"].astype(str), -data["region_y_score"].to_numpy())
        )
        coverage_order = np.lexsort(
            (data["desikan_killiany_label"].astype(str), -data["n_contacts"].to_numpy())
        )
        network_x, network_y, network_auc = _ordered_capture_curve(data, network_order)
        coverage_x, coverage_y, coverage_auc = _ordered_capture_curve(data, coverage_order)

        rng = np.random.default_rng(_stable_seed(seed, band, patient))
        random_aucs = np.empty(n_random_rankings, dtype=np.float64)
        random_curves = np.empty((n_random_rankings, grid_points), dtype=np.float64)
        for iteration in range(n_random_rankings):
            x_random, y_random, random_aucs[iteration] = _ordered_capture_curve(
                data, rng.permutation(len(data))
            )
            random_curves[iteration] = np.interp(grid, x_random, y_random)
        random_auc = float(random_aucs.mean())
        metric_records.append(
            {
                "frequency_band": band,
                "patient": patient,
                "n_regions": len(data),
                "n_contacts": int(data["n_contacts"].sum()),
                "n_residual_fn": n_fn,
                "qc_status": "computed",
                "network_capture_auc": network_auc,
                "coverage_capture_auc": coverage_auc,
                "random_capture_auc_mean": random_auc,
                "network_minus_random_auc": network_auc - random_auc,
                "network_minus_coverage_auc": network_auc - coverage_auc,
            }
        )
        for method, x_values, y_values in [
            ("region_network_score", network_x, network_y),
            ("coverage_only", coverage_x, coverage_y),
            ("random_region_order", grid, random_curves.mean(axis=0)),
        ]:
            for point_index, (x_value, y_value) in enumerate(zip(x_values, y_values, strict=True)):
                curve_records.append(
                    {
                        "frequency_band": band,
                        "patient": patient,
                        "ranking_method": method,
                        "curve_point": point_index,
                        "cumulative_contact_fraction": float(x_value),
                        "cumulative_residual_fn_fraction": float(y_value),
                    }
                )
    return pd.DataFrame.from_records(metric_records), pd.DataFrame.from_records(curve_records)


def summarize_capture(capture_patient: pd.DataFrame) -> pd.DataFrame:
    computed = capture_patient[capture_patient["qc_status"].eq("computed")]
    records: list[dict[str, object]] = []
    for band, frame in computed.groupby("frequency_band", sort=True):
        records.append(
            {
                "frequency_band": band,
                "n_evaluable_patients": frame["patient"].nunique(),
                "n_residual_fn": int(frame["n_residual_fn"].sum()),
                "network_capture_auc_macro": float(frame["network_capture_auc"].mean()),
                "coverage_capture_auc_macro": float(frame["coverage_capture_auc"].mean()),
                "random_capture_auc_macro": float(frame["random_capture_auc_mean"].mean()),
                "network_minus_random_auc_macro": float(
                    frame["network_minus_random_auc"].mean()
                ),
                "network_minus_coverage_auc_macro": float(
                    frame["network_minus_coverage_auc"].mean()
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def _draw_patient_mean(
    frame: pd.DataFrame,
    field: str,
    patient_draw: Sequence[str],
) -> float:
    by_patient = frame.set_index("patient")[field]
    values = [float(by_patient.loc[patient]) for patient in patient_draw if patient in by_patient.index]
    values = [value for value in values if np.isfinite(value)]
    return float(np.mean(values)) if values else np.nan


def build_observed_metrics(
    patient_burden: pd.DataFrame,
    patient_status_comparison: pd.DataFrame,
    associations: pd.DataFrame,
    capture_patient: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    bands = sorted(patient_burden["frequency_band"].astype(str).unique())
    for band in bands:
        burden = patient_burden[patient_burden["frequency_band"].astype(str).eq(band)]
        status_comparison = patient_status_comparison[
            patient_status_comparison["frequency_band"].astype(str).eq(band)
            & patient_status_comparison["paired_comparison_evaluable"]
        ]
        capture = capture_patient[
            capture_patient["frequency_band"].astype(str).eq(band)
            & capture_patient["qc_status"].eq("computed")
        ]
        association = associations[associations["frequency_band"].astype(str).eq(band)]
        values = {
            "suspicious_region_macro_residual_fn_rate": float(
                burden["residual_fn_rate"].mean()
            ),
            "suspicious_minus_non_suspicious_residual_fn_rate": float(
                status_comparison["suspicious_minus_non_suspicious_fn_rate"].mean()
            ),
            "region_score_residual_slope_all": float(
                association.loc[
                    association["analysis_scope"].eq("all_primary_regions_with_ablation"),
                    "weighted_slope",
                ].iloc[0]
            ),
            "region_score_residual_slope_suspicious": float(
                association.loc[
                    association["analysis_scope"].eq("suspicious_regions_with_ablation"),
                    "weighted_slope",
                ].iloc[0]
            ),
            "network_capture_auc": float(capture["network_capture_auc"].mean()),
            "coverage_capture_auc": float(capture["coverage_capture_auc"].mean()),
            "random_capture_auc": float(capture["random_capture_auc_mean"].mean()),
            "network_minus_random_capture_auc": float(
                capture["network_minus_random_auc"].mean()
            ),
            "network_minus_coverage_capture_auc": float(
                capture["network_minus_coverage_auc"].mean()
            ),
        }
        records.extend(
            {"frequency_band": band, "statistic": statistic, "estimate": estimate}
            for statistic, estimate in values.items()
        )
    return pd.DataFrame.from_records(records)


def patient_bootstrap(
    region_table: pd.DataFrame,
    patient_burden: pd.DataFrame,
    patient_status_comparison: pd.DataFrame,
    capture_patient: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    n_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations < 1:
        raise SystemExit("patient_bootstrap_iterations must be positive")
    patients = sorted(region_table["patient"].astype(str).unique())
    bands = sorted(region_table["frequency_band"].astype(str).unique())
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for iteration in range(1, n_iterations + 1):
        draw = rng.choice(patients, size=len(patients), replace=True).tolist()
        for band in bands:
            burden = patient_burden[
                patient_burden["frequency_band"].astype(str).eq(band)
            ]
            status_comparison = patient_status_comparison[
                patient_status_comparison["frequency_band"].astype(str).eq(band)
                & patient_status_comparison["paired_comparison_evaluable"]
            ]
            capture = capture_patient[
                capture_patient["frequency_band"].astype(str).eq(band)
                & capture_patient["qc_status"].eq("computed")
            ]
            band_regions = region_table[
                region_table["frequency_band"].astype(str).eq(band)
            ]
            association_all = _weighted_linear_association(band_regions, draw)
            association_suspicious = _weighted_linear_association(
                band_regions[band_regions["suspicious_region_status"].eq("yes")], draw
            )
            values = {
                "suspicious_region_macro_residual_fn_rate": _draw_patient_mean(
                    burden, "residual_fn_rate", draw
                ),
                "suspicious_minus_non_suspicious_residual_fn_rate": _draw_patient_mean(
                    status_comparison,
                    "suspicious_minus_non_suspicious_fn_rate",
                    draw,
                ),
                "region_score_residual_slope_all": association_all["weighted_slope"],
                "region_score_residual_slope_suspicious": association_suspicious[
                    "weighted_slope"
                ],
                "network_capture_auc": _draw_patient_mean(
                    capture, "network_capture_auc", draw
                ),
                "coverage_capture_auc": _draw_patient_mean(
                    capture, "coverage_capture_auc", draw
                ),
                "random_capture_auc": _draw_patient_mean(
                    capture, "random_capture_auc_mean", draw
                ),
                "network_minus_random_capture_auc": _draw_patient_mean(
                    capture, "network_minus_random_auc", draw
                ),
                "network_minus_coverage_capture_auc": _draw_patient_mean(
                    capture, "network_minus_coverage_auc", draw
                ),
            }
            records.extend(
                {
                    "bootstrap_iteration": iteration,
                    "frequency_band": band,
                    "statistic": statistic,
                    "value": float(value),
                }
                for statistic, value in values.items()
            )
    bootstrap = pd.DataFrame.from_records(records)
    ci_records: list[dict[str, object]] = []
    for (band, statistic), frame in bootstrap.groupby(
        ["frequency_band", "statistic"], sort=True
    ):
        values = frame["value"].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        estimate = observed.loc[
            observed["frequency_band"].astype(str).eq(str(band))
            & observed["statistic"].astype(str).eq(str(statistic)),
            "estimate",
        ]
        ci_records.append(
            {
                "frequency_band": band,
                "statistic": statistic,
                "estimate": float(estimate.iloc[0]),
                "bootstrap_valid_iterations": len(values),
                "ci_lower_95": float(np.quantile(values, 0.025)) if len(values) else np.nan,
                "ci_upper_95": float(np.quantile(values, 0.975)) if len(values) else np.nan,
            }
        )
    return bootstrap, pd.DataFrame.from_records(ci_records)


def validate_full_run(
    contact_table: pd.DataFrame,
    region_table: pd.DataFrame,
    labels: pd.DataFrame,
    bands: Sequence[str],
    expected: dict[str, object],
) -> None:
    unique_contacts = contact_table[["patient", "bipolar_channel"]].drop_duplicates()
    unique_contact_labels = contact_table[
        ["patient", "bipolar_channel", "y_true", "suspicious_region_status"]
    ].drop_duplicates()
    unique_regions = region_table[REGION_KEYS].drop_duplicates()
    actual = {
        "patients": labels["patient"].nunique(),
        "primary_regions": len(unique_regions),
        "suspicious_regions": int(labels["suspicious_region_status"].eq("yes").sum()),
        "non_suspicious_regions": int(labels["suspicious_region_status"].eq("no").sum()),
        "primary_contacts": len(unique_contacts),
        "doctor_ablated_contacts": int(unique_contact_labels["y_true"].sum()),
        "suspicious_region_doctor_ablated_contacts": int(
            unique_contact_labels.loc[
                unique_contact_labels["suspicious_region_status"].eq("yes"), "y_true"
            ].sum()
        ),
        "contact_prediction_rows": len(contact_table),
        "region_prediction_rows": len(region_table),
    }
    mismatches = {
        key: (actual[key], int(expected[key]))
        for key in actual
        if actual[key] != int(expected[key])
    }
    if mismatches:
        raise SystemExit(f"Formal input expectations failed: {mismatches}")
    if set(contact_table["frequency_band"].astype(str)) != set(bands):
        raise SystemExit("Contact table does not contain every configured frequency band")
    if set(region_table["frequency_band"].astype(str)) != set(bands):
        raise SystemExit("Region table does not contain every configured frequency band")


def _check_output_targets(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "Output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing[:5])
        )


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8-sig", newline="", suffix=".csv", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        frame.to_csv(handle, index=False)
    temp_path.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    temp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patients", nargs="+")
    parser.add_argument("--bands", nargs="+")
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--n-random-rankings", type=int)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true", help="Write reviewed outputs")
    mode.add_argument("--dry-run", action="store_true", help="Validate and compute without writing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    inputs = config["inputs"]
    analysis = config["analysis"]
    expected = config["expected"]
    if not isinstance(inputs, dict) or not isinstance(analysis, dict) or not isinstance(expected, dict):
        raise SystemExit("Configuration inputs/analysis/expected sections must be mappings")
    configured_bands = [str(value) for value in config["frequency_bands"]]
    bands = [str(value) for value in (args.bands or configured_bands)]
    unknown_bands = sorted(set(bands).difference(configured_bands))
    if unknown_bands:
        raise SystemExit(f"Unknown frequency bands: {unknown_bands}")
    output_root = args.out or Path(str(config["output_root"]))
    assert_allowed_output_root(output_root)

    region_path = Path(str(inputs["region_predictions"]))
    label_path = Path(str(inputs["region_labels"]))
    contact_path = Path(str(inputs["contact_predictions"]))
    labels = load_region_labels(label_path, args.patients)
    region_predictions = load_region_predictions(region_path, bands, args.patients)
    contact_predictions = load_contact_predictions(contact_path, bands, args.patients)
    contact_table, region_table = build_joint_tables(
        region_predictions, contact_predictions, labels
    )

    full_run = args.patients is None and args.bands is None
    if full_run:
        validate_full_run(contact_table, region_table, labels, bands, expected)

    patient_burden, band_burden = summarize_suspicious_burden(
        region_table, str(analysis["suspicious_region_status"])
    )
    patient_status, patient_status_comparison, status_comparison = (
        compare_region_status_fn_rates(region_table)
    )
    associations = region_score_associations(region_table)
    n_random = int(
        analysis["random_region_rankings"]
        if args.n_random_rankings is None
        else args.n_random_rankings
    )
    capture_patient, capture_curves = build_capture_analysis(
        region_table,
        n_random_rankings=n_random,
        grid_points=int(analysis["capture_curve_grid_points"]),
        seed=int(analysis["random_seed"]),
    )
    capture_band = summarize_capture(capture_patient)
    observed = build_observed_metrics(
        patient_burden,
        patient_status_comparison,
        associations,
        capture_patient,
    )
    n_bootstrap = int(
        analysis["patient_bootstrap_iterations"]
        if args.n_bootstrap is None
        else args.n_bootstrap
    )
    bootstrap, bootstrap_ci = patient_bootstrap(
        region_table,
        patient_burden,
        patient_status_comparison,
        capture_patient,
        observed,
        n_iterations=n_bootstrap,
        seed=int(analysis["random_seed"]),
    )

    output_frames = {
        "region_contact_predictions_by_band.csv": contact_table,
        "region_contact_residuals_by_band.csv": region_table,
        "suspicious_region_patient_band_burden.csv": patient_burden,
        "suspicious_region_burden_by_band.csv": band_burden,
        "region_status_patient_band_burden.csv": patient_status,
        "region_status_patient_band_fn_rate_comparison.csv": patient_status_comparison,
        "region_status_fn_rate_comparison_by_band.csv": status_comparison,
        "region_score_residual_association_by_band.csv": associations,
        "residual_fn_capture_patient_band.csv": capture_patient,
        "residual_fn_capture_curves.csv": capture_curves,
        "residual_fn_capture_by_band.csv": capture_band,
        "bootstrap_results.csv": bootstrap,
        "bootstrap_ci.csv": bootstrap_ci,
    }
    output_paths = [output_root / name for name in output_frames]
    summary_path = output_root / "region_contact_residual_run_summary.json"
    output_paths.append(summary_path)
    if args.apply:
        _check_output_targets(output_paths, args.overwrite)

    summary = {
        "status": "computed_not_written" if not args.apply else "complete",
        "analysis_scope": analysis["interpretation"],
        "config": str(args.config),
        "inputs": {
            "region_predictions": str(region_path),
            "region_predictions_sha256": sha256_file(region_path),
            "region_labels": str(label_path),
            "region_labels_sha256": sha256_file(label_path),
            "contact_predictions": str(contact_path),
            "contact_predictions_sha256": sha256_file(contact_path),
        },
        "patients": sorted(contact_table["patient"].astype(str).unique()),
        "frequency_bands": bands,
        "n_contact_prediction_rows": len(contact_table),
        "n_region_prediction_rows": len(region_table),
        "n_primary_contacts": int(
            contact_table[["patient", "bipolar_channel"]].drop_duplicates().shape[0]
        ),
        "n_primary_regions": int(region_table[REGION_KEYS].drop_duplicates().shape[0]),
        "n_suspicious_regions": int(
            labels["suspicious_region_status"].eq("yes").sum()
        ),
        "n_non_suspicious_regions": int(
            labels["suspicious_region_status"].eq("no").sum()
        ),
        "n_bootstrap": n_bootstrap,
        "n_random_region_rankings": n_random,
        "region_grouping_rule": "locked_ablation_top4_rule_only",
        "additional_region_score_cutoff": False,
    }
    if args.apply:
        for name, frame in output_frames.items():
            _atomic_csv(frame, output_root / name)
        _atomic_json(summary, summary_path)

    print(
        json.dumps(
            {
                "mode": "apply" if args.apply else "dry-run",
                "output_root": str(output_root),
                "contact_rows": len(contact_table),
                "region_rows": len(region_table),
                "suspicious_regions": summary["n_suspicious_regions"],
                "bootstrap_iterations": n_bootstrap,
                "random_rankings": n_random,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
