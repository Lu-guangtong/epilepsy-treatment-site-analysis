"""Measure the regional distribution of bipolar-pair false negatives."""
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
CONFIG_PATH = config_root() / "regional_false_negatives.yaml"
OUTPUT_ROOT = ROOT / "data_derived" / "fn_suspicious_region_concentration"
CHECK_ROOT = ROOT / "checks"

BAND_PATIENT_KEYS = ["frequency_band", "patient"]
CONTACT_KEYS = ["frequency_band", "patient", "bipolar_channel"]
IDENTITY_KEYS = ["patient", "bipolar_channel"]
COUNT_COLUMNS = [
    "n_contacts",
    "n_contacts_suspicious",
    "n_contacts_non_suspicious",
    "n_ablated",
    "n_ablated_suspicious",
    "n_ablated_non_suspicious",
    "n_fn",
    "n_fn_suspicious",
    "n_fn_non_suspicious",
]
BOOTSTRAP_STATISTICS = [
    "fn_share_in_suspicious",
    "contact_share_in_suspicious",
    "ablated_share_in_suspicious",
    "fn_enrichment_vs_contact_coverage",
    "fn_enrichment_vs_ablated_distribution",
    "suspicious_fn_rate",
    "non_suspicious_fn_rate",
    "fn_rate_difference",
    "patient_macro_fn_rate_difference",
]


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


def _safe_ratio(numerator: float | int, denominator: float | int) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def load_predictions(
    path: Path,
    bands: Sequence[str],
    patients: Sequence[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing contact prediction table: {path}")
    frame = pd.read_csv(path)
    required = {
        *CONTACT_KEYS,
        "heldout_patient",
        "desikan_killiany_label",
        "suspicious_region_status",
        "y_true",
        "y_pred",
    }
    _require_columns(frame, required, path)
    frame = frame[frame["frequency_band"].astype(str).isin(bands)].copy()
    if patients is not None:
        requested = {str(value) for value in patients}
        frame = frame[frame["patient"].astype(str).isin(requested)].copy()
    if frame.empty:
        raise SystemExit("No contact predictions remain after filtering")
    if frame.duplicated(CONTACT_KEYS).any():
        raise SystemExit("Contact predictions contain duplicate patient-contact-band rows")
    if not frame["patient"].astype(str).eq(frame["heldout_patient"].astype(str)).all():
        raise SystemExit("Contact predictions contain non-heldout patient rows")
    if not frame["y_true"].isin([0, 1]).all() or not frame["y_pred"].isin([0, 1]).all():
        raise SystemExit("Contact labels and predictions must be binary")
    if not frame["suspicious_region_status"].isin(["yes", "no"]).all():
        raise SystemExit("Suspicious-region status must contain only yes/no")
    identity = frame.groupby(IDENTITY_KEYS, sort=False).agg(
        n_regions=("desikan_killiany_label", "nunique"),
        n_status=("suspicious_region_status", "nunique"),
        n_labels=("y_true", "nunique"),
        n_bands=("frequency_band", "nunique"),
    )
    if (identity[["n_regions", "n_status", "n_labels"]] != 1).any().any():
        raise SystemExit("Contact identity, region status, or clinician label changes across bands")
    expected_band_count = len(set(bands))
    if not identity["n_bands"].eq(expected_band_count).all():
        raise SystemExit("At least one contact is missing a requested frequency band")
    frame["is_fn"] = frame["y_true"].eq(1) & frame["y_pred"].eq(0)
    if "classification_state" in frame.columns:
        expected_state = np.select(
            [
                frame["y_true"].eq(1) & frame["y_pred"].eq(1),
                frame["is_fn"],
                frame["y_true"].eq(0) & frame["y_pred"].eq(1),
                frame["y_true"].eq(0) & frame["y_pred"].eq(0),
            ],
            ["TP", "FN", "FP", "TN"],
            default="ERROR",
        )
        if not np.array_equal(expected_state, frame["classification_state"].astype(str)):
            raise SystemExit("Stored classification states disagree with y_true/y_pred")
    return frame.sort_values(CONTACT_KEYS).reset_index(drop=True)


def _derive_metrics(counts: dict[str, int | float]) -> dict[str, float]:
    contact_share = _safe_ratio(counts["n_contacts_suspicious"], counts["n_contacts"])
    ablated_share = _safe_ratio(counts["n_ablated_suspicious"], counts["n_ablated"])
    fn_share = _safe_ratio(counts["n_fn_suspicious"], counts["n_fn"])
    suspicious_fn_rate = _safe_ratio(
        counts["n_fn_suspicious"], counts["n_ablated_suspicious"]
    )
    non_suspicious_fn_rate = _safe_ratio(
        counts["n_fn_non_suspicious"], counts["n_ablated_non_suspicious"]
    )
    return {
        "contact_share_in_suspicious": contact_share,
        "ablated_share_in_suspicious": ablated_share,
        "fn_share_in_suspicious": fn_share,
        "fn_enrichment_vs_contact_coverage": _safe_ratio(fn_share, contact_share),
        "fn_enrichment_vs_ablated_distribution": _safe_ratio(fn_share, ablated_share),
        "suspicious_fn_rate": suspicious_fn_rate,
        "non_suspicious_fn_rate": non_suspicious_fn_rate,
        "fn_rate_difference": suspicious_fn_rate - non_suspicious_fn_rate,
        "fn_rate_ratio": _safe_ratio(suspicious_fn_rate, non_suspicious_fn_rate),
    }


def _count_group(frame: pd.DataFrame) -> dict[str, int]:
    suspicious = frame["suspicious_region_status"].eq("yes")
    ablated = frame["y_true"].eq(1)
    fn = ablated & frame["y_pred"].eq(0)
    return {
        "n_contacts": int(len(frame)),
        "n_contacts_suspicious": int(suspicious.sum()),
        "n_contacts_non_suspicious": int((~suspicious).sum()),
        "n_ablated": int(ablated.sum()),
        "n_ablated_suspicious": int((ablated & suspicious).sum()),
        "n_ablated_non_suspicious": int((ablated & ~suspicious).sum()),
        "n_fn": int(fn.sum()),
        "n_fn_suspicious": int((fn & suspicious).sum()),
        "n_fn_non_suspicious": int((fn & ~suspicious).sum()),
    }


def build_patient_band_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (band, patient), frame in predictions.groupby(BAND_PATIENT_KEYS, sort=True):
        counts = _count_group(frame)
        records.append(
            {
                "frequency_band": str(band),
                "patient": str(patient),
                **counts,
                **_derive_metrics(counts),
                "fn_share_evaluable": counts["n_fn"] > 0,
                "paired_fn_rate_evaluable": (
                    counts["n_ablated_suspicious"] > 0
                    and counts["n_ablated_non_suspicious"] > 0
                ),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(BAND_PATIENT_KEYS).reset_index(drop=True)


def _sum_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {column: int(frame[column].sum()) for column in COUNT_COLUMNS}


def build_band_summary(patient_table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for band, frame in patient_table.groupby("frequency_band", sort=True):
        counts = _sum_counts(frame)
        metrics = _derive_metrics(counts)
        valid_share = frame["fn_share_in_suspicious"].dropna()
        valid_difference = frame["fn_rate_difference"].dropna()
        records.append(
            {
                "frequency_band": str(band),
                **counts,
                **metrics,
                "n_patients_with_fn": int(frame["n_fn"].gt(0).sum()),
                "n_paired_patients": int(frame["paired_fn_rate_evaluable"].sum()),
                "patient_macro_fn_share_in_suspicious": float(valid_share.mean()),
                "patient_macro_fn_rate_difference": float(valid_difference.mean()),
            }
        )
    return pd.DataFrame.from_records(records).sort_values("frequency_band").reset_index(drop=True)


def patient_bootstrap(
    patient_table: pd.DataFrame,
    *,
    bands: Sequence[str],
    n_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations <= 0:
        raise SystemExit("Bootstrap iterations must be positive")
    patients = sorted(patient_table["patient"].astype(str).unique())
    lookup = {
        (str(row.frequency_band), str(row.patient)): row
        for row in patient_table.itertuples(index=False)
    }
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for iteration in range(n_iterations):
        draw = rng.choice(patients, size=len(patients), replace=True)
        for band in bands:
            rows = [lookup[(str(band), str(patient))] for patient in draw]
            counts = {
                column: int(sum(getattr(row, column) for row in rows))
                for column in COUNT_COLUMNS
            }
            metrics = _derive_metrics(counts)
            valid_differences = np.asarray(
                [row.fn_rate_difference for row in rows], dtype=np.float64
            )
            metrics["patient_macro_fn_rate_difference"] = float(
                np.nanmean(valid_differences)
            )
            records.append(
                {
                    "bootstrap_iteration": iteration,
                    "frequency_band": str(band),
                    **metrics,
                }
            )
    bootstrap = pd.DataFrame.from_records(records)
    observed = build_band_summary(patient_table).set_index("frequency_band")
    ci_records: list[dict[str, object]] = []
    for band in bands:
        band_bootstrap = bootstrap[bootstrap["frequency_band"].eq(str(band))]
        for statistic in BOOTSTRAP_STATISTICS:
            values = band_bootstrap[statistic].dropna().to_numpy(dtype=np.float64)
            ci_records.append(
                {
                    "frequency_band": str(band),
                    "statistic": statistic,
                    "estimate": float(observed.loc[str(band), statistic]),
                    "bootstrap_valid_iterations": int(len(values)),
                    "ci_lower_95": float(np.quantile(values, 0.025)),
                    "ci_upper_95": float(np.quantile(values, 0.975)),
                }
            )
    return bootstrap, pd.DataFrame.from_records(ci_records)


def within_patient_permutation(
    predictions: pd.DataFrame,
    observed: pd.DataFrame,
    *,
    bands: Sequence[str],
    n_permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_permutations <= 0:
        raise SystemExit("Permutation count must be positive")
    groups: dict[tuple[str, str], tuple[np.ndarray, int]] = {}
    for (band, patient), frame in predictions.groupby(BAND_PATIENT_KEYS, sort=True):
        burned = frame[frame["y_true"].eq(1)]
        groups[(str(band), str(patient))] = (
            burned["suspicious_region_status"].eq("yes").to_numpy(dtype=np.int8),
            int(burned["y_pred"].eq(0).sum()),
        )
    patients = sorted(predictions["patient"].astype(str).unique())
    observed_indexed = observed.set_index("frequency_band")
    records: list[dict[str, object]] = []
    for permutation_id in range(n_permutations):
        permutation_seed = int(seed + permutation_id)
        rng = np.random.default_rng(permutation_seed)
        for band in bands:
            n_fn_suspicious = 0
            n_fn = 0
            for patient in patients:
                suspicious_flags, n_patient_fn = groups[(str(band), str(patient))]
                if n_patient_fn:
                    shuffled = rng.permutation(suspicious_flags)
                    n_fn_suspicious += int(shuffled[:n_patient_fn].sum())
                n_fn += n_patient_fn
            n_fn_non_suspicious = n_fn - n_fn_suspicious
            row = observed_indexed.loc[str(band)]
            counts = {
                "n_contacts": int(row.n_contacts),
                "n_contacts_suspicious": int(row.n_contacts_suspicious),
                "n_contacts_non_suspicious": int(row.n_contacts_non_suspicious),
                "n_ablated": int(row.n_ablated),
                "n_ablated_suspicious": int(row.n_ablated_suspicious),
                "n_ablated_non_suspicious": int(row.n_ablated_non_suspicious),
                "n_fn": n_fn,
                "n_fn_suspicious": n_fn_suspicious,
                "n_fn_non_suspicious": n_fn_non_suspicious,
            }
            metrics = _derive_metrics(counts)
            records.append(
                {
                    "permutation_id": permutation_id,
                    "permutation_seed": permutation_seed,
                    "frequency_band": str(band),
                    "permuted_fn_suspicious": n_fn_suspicious,
                    "permuted_fn_non_suspicious": n_fn_non_suspicious,
                    "fn_share_in_suspicious": metrics["fn_share_in_suspicious"],
                    "fn_enrichment_vs_ablated_distribution": metrics[
                        "fn_enrichment_vs_ablated_distribution"
                    ],
                    "fn_rate_difference": metrics["fn_rate_difference"],
                }
            )
    permutation = pd.DataFrame.from_records(records)
    null_stats: dict[str, tuple[float, float]] = {}
    for band in bands:
        values = permutation.loc[
            permutation["frequency_band"].eq(str(band)),
            "fn_enrichment_vs_ablated_distribution",
        ].to_numpy(dtype=np.float64)
        null_stats[str(band)] = (float(values.mean()), float(values.std(ddof=1)))
    permutation["standardized_enrichment"] = [
        (value - null_stats[str(band)][0]) / null_stats[str(band)][1]
        for band, value in zip(
            permutation["frequency_band"],
            permutation["fn_enrichment_vs_ablated_distribution"],
            strict=True,
        )
    ]
    max_t = permutation.groupby("permutation_id", sort=True)[
        "standardized_enrichment"
    ].max()
    summary_records: list[dict[str, object]] = []
    for band in bands:
        band_null = permutation[permutation["frequency_band"].eq(str(band))]
        observed_stat = float(
            observed_indexed.loc[str(band), "fn_enrichment_vs_ablated_distribution"]
        )
        null_mean, null_sd = null_stats[str(band)]
        observed_z = (observed_stat - null_mean) / null_sd
        raw_p = (1 + int((band_null["fn_enrichment_vs_ablated_distribution"] >= observed_stat).sum())) / (
            n_permutations + 1
        )
        max_t_p = (1 + int((max_t >= observed_z).sum())) / (n_permutations + 1)
        summary_records.append(
            {
                "frequency_band": str(band),
                "observed_fn_enrichment_vs_ablated_distribution": observed_stat,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "observed_standardized_enrichment": observed_z,
                "one_sided_permutation_p": raw_p,
                "maxT_FWER_p": max_t_p,
                "n_permutations": n_permutations,
            }
        )
    summary = pd.DataFrame.from_records(summary_records)
    if (summary["maxT_FWER_p"] + 1e-15 < summary["one_sided_permutation_p"]).any():
        raise SystemExit("maxT-FWER p-values must not be smaller than raw p-values")
    return permutation, summary


def validate_full_run(
    predictions: pd.DataFrame,
    patient_table: pd.DataFrame,
    band_table: pd.DataFrame,
    bands: Sequence[str],
    expected: dict[str, object],
) -> None:
    checks = {
        "prediction_rows": len(predictions),
        "patients": predictions["patient"].nunique(),
        "primary_contacts": predictions[IDENTITY_KEYS].drop_duplicates().shape[0],
        "primary_regions": predictions[["patient", "desikan_killiany_label"]].drop_duplicates().shape[0],
    }
    unique_contacts = predictions.drop_duplicates(IDENTITY_KEYS)
    checks.update(
        {
            "suspicious_regions": unique_contacts.loc[
                unique_contacts["suspicious_region_status"].eq("yes"),
                ["patient", "desikan_killiany_label"],
            ].drop_duplicates().shape[0],
            "non_suspicious_regions": unique_contacts.loc[
                unique_contacts["suspicious_region_status"].eq("no"),
                ["patient", "desikan_killiany_label"],
            ].drop_duplicates().shape[0],
            "suspicious_contacts": int(
                unique_contacts["suspicious_region_status"].eq("yes").sum()
            ),
            "clinician_ablated_contacts": int(unique_contacts["y_true"].sum()),
            "suspicious_clinician_ablated_contacts": int(
                (
                    unique_contacts["y_true"].eq(1)
                    & unique_contacts["suspicious_region_status"].eq("yes")
                ).sum()
            ),
        }
    )
    mismatches = {
        key: (value, int(expected[key]))
        for key, value in checks.items()
        if value != int(expected[key])
    }
    if mismatches:
        raise SystemExit(f"Formal input count validation failed: {mismatches}")
    if set(predictions["frequency_band"].astype(str)) != set(bands):
        raise SystemExit("Formal predictions do not contain every configured band")
    if len(patient_table) != int(expected["patients"]) * len(bands):
        raise SystemExit("Patient-band table has an unexpected number of rows")
    if len(band_table) != len(bands):
        raise SystemExit("Band summary has an unexpected number of rows")
    if not (band_table["n_contacts"] == int(expected["primary_contacts"])).all():
        raise SystemExit("Contact totals are not constant across bands")
    if not (band_table["n_ablated"] == int(expected["clinician_ablated_contacts"])).all():
        raise SystemExit("Clinician-ablation totals are not constant across bands")


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
        mode="w",
        encoding="utf-8-sig",
        newline="",
        suffix=".csv",
        dir=path.parent,
        delete=False,
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
    parser.add_argument("--n-permutations", type=int)
    parser.add_argument("--overwrite", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
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
    unknown = sorted(set(bands).difference(configured_bands))
    if unknown:
        raise SystemExit(f"Unknown frequency bands: {unknown}")
    output_root = args.out or Path(str(config["output_root"]))
    assert_allowed_output_root(output_root)
    prediction_path = Path(str(inputs["contact_predictions"]))
    predictions = load_predictions(prediction_path, bands, args.patients)
    patient_table = build_patient_band_summary(predictions)
    band_table = build_band_summary(patient_table)
    full_run = args.patients is None and args.bands is None
    if full_run:
        validate_full_run(predictions, patient_table, band_table, bands, expected)
    n_bootstrap = int(
        analysis["patient_bootstrap_iterations"]
        if args.n_bootstrap is None
        else args.n_bootstrap
    )
    n_permutations = int(
        analysis["within_patient_permutations"]
        if args.n_permutations is None
        else args.n_permutations
    )
    bootstrap, bootstrap_ci = patient_bootstrap(
        patient_table,
        bands=bands,
        n_iterations=n_bootstrap,
        seed=int(analysis["random_seed"]),
    )
    permutation, permutation_summary = within_patient_permutation(
        predictions,
        band_table,
        bands=bands,
        n_permutations=n_permutations,
        seed=int(analysis["random_seed"]),
    )
    output_frames = {
        "fn_concentration_patient_band.csv": patient_table,
        "fn_concentration_by_band.csv": band_table,
        "fn_concentration_bootstrap_results.csv": bootstrap,
        "fn_concentration_bootstrap_ci.csv": bootstrap_ci,
        "fn_concentration_permutation_results.csv": permutation,
        "fn_concentration_permutation_summary.csv": permutation_summary,
    }
    summary_path = output_root / "fn_concentration_run_summary.json"
    output_paths = [output_root / name for name in output_frames]
    output_paths.append(summary_path)
    if args.apply:
        _check_output_targets(output_paths, args.overwrite)
    summary = {
        "status": "complete" if args.apply else "computed_not_written",
        "analysis_scope": analysis["interpretation"],
        "config": str(args.config),
        "input": str(prediction_path),
        "input_sha256": sha256_file(prediction_path),
        "patients": sorted(predictions["patient"].astype(str).unique()),
        "frequency_bands": bands,
        "n_prediction_rows": len(predictions),
        "n_unique_contacts": predictions[IDENTITY_KEYS].drop_duplicates().shape[0],
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
        "permutation_unit": "within_patient_and_band_among_clinician_ablated_contacts",
        "primary_permutation_statistic": analysis["primary_permutation_statistic"],
        "region_label_rule": "locked_ablation_derived_rule_from_formal_region_outputs",
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
                "prediction_rows": len(predictions),
                "patient_band_rows": len(patient_table),
                "band_rows": len(band_table),
                "bootstrap_iterations": n_bootstrap,
                "permutations": n_permutations,
            },
            ensure_ascii=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
