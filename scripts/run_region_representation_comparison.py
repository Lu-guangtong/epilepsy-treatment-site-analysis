"""Compare local and connectivity representations at the regional target."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

import run_cross_scale_models as cross_model
import run_region_lopo_modeling as region_model


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = (
    config_root() / "region_representation_comparison.yaml"
)
OUTPUT_ROOT = (
    ROOT
    / "data_derived"
    / "cross_scale_models"
    / "ablation_top4_rule"
)
CHECK_ROOT = ROOT / "checks"
DRIVER_PATH = Path(__file__).resolve()
REGION_MODEL_PATH = Path(region_model.__file__).resolve()
REGION_KEYS = [
    "frequency_band",
    "patient",
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
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
    resolved = _resolve(out_dir)
    output_root = _resolve(OUTPUT_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if resolved == AUX_ROOT or _is_relative_to(resolved, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {resolved}")
    if not (
        resolved == output_root
        or _is_relative_to(resolved, output_root)
        or resolved == tmp_root
        or _is_relative_to(resolved, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _preflight_outputs(paths: Iterable[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite: {existing[0]}")


def validate_orientation_contract(config: dict[str, object]) -> dict[str, str]:
    modeling = config["modeling"]
    configured_rule = str(modeling.get("univariate_orientation_rule", ""))
    configured_scope = str(modeling.get("univariate_orientation_scope", ""))
    required_rule = region_model.UNIVARIATE_ORIENTATION_RULE
    if configured_rule != required_rule:
        raise SystemExit(
            "Cross-level configuration does not lock the reviewed univariate "
            f"orientation rule: {configured_rule!r} != {required_rule!r}"
        )
    if configured_scope != "inner_training_patients_only":
        raise SystemExit(
            "Univariate feature orientation must be decided from inner-training "
            "patients only"
        )
    return {
        "rule": required_rule,
        "scope": configured_scope,
        "decision": (
            "positive_median_ge_negative_median_keeps_native_else_sign_reverse"
        ),
        "validation_use": (
            "fixed_orientation_applied_to_inner_validation_patient"
        ),
    }




def output_paths(out_dir: Path) -> dict[str, Path]:
    return {
        "predictions": out_dir / "electrode_to_region_predictions_by_band.csv",
        "metrics": out_dir / "electrode_to_region_metrics_by_band.csv",
        "folds": out_dir / "electrode_to_region_fold_details_by_band.csv",
        "bootstrap": out_dir / "electrode_to_region_bootstrap_results.csv",
        "bootstrap_ci": out_dir / "electrode_to_region_bootstrap_ci.csv",
        "comparison": out_dir / "current_cross_level_method_comparison.csv",
        "summary": out_dir / "cross_level_run_summary.json",
    }


def load_locked_region_labels(
    path: Path,
    expected: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if not path.exists():
        raise SystemExit(f"Locked region label file is missing: {path}")
    labels = pd.read_csv(path)
    required = {
        "patient",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
        "label_binary",
        "inclusion_status",
    }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise SystemExit(f"Locked region labels are missing columns: {missing}")
    key_columns = REGION_KEYS[1:]
    if labels.duplicated(key_columns).any():
        raise SystemExit("Locked region labels contain duplicate patient-region keys")
    primary = labels[labels["inclusion_status"].eq("primary")].copy()
    if primary["label_binary"].isna().any():
        raise SystemExit("Primary region labels contain missing label_binary values")
    observed_values = set(primary["label_binary"].astype(int).unique())
    if not observed_values.issubset({0, 1}):
        raise SystemExit(f"Primary labels must be binary, got {observed_values}")
    counts = {
        "label_rows": len(labels),
        "primary_region_rows": len(primary),
        "positive_region_rows": int(primary["label_binary"].astype(int).sum()),
        "negative_region_rows": int((primary["label_binary"].astype(int) == 0).sum()),
    }
    for key, value in counts.items():
        expected_value = int(expected[key])
        if value != expected_value:
            raise SystemExit(
                f"Locked region label count mismatch for {key}: "
                f"{value} != {expected_value}"
            )
    return labels, counts


def validate_locked_labels_against_audit(
    labels: pd.DataFrame,
    audit_workbook: Path,
    rule: dict[str, object],
) -> dict[str, object]:
    if not audit_workbook.exists():
        raise SystemExit(f"Audit workbook is missing: {audit_workbook}")
    region_rows = region_model.load_region_summary_rows(audit_workbook)
    derived, derivation_audit = region_model.build_ablation_top4_label_table(
        region_rows, rule
    )
    compare_columns = [
        *REGION_KEYS[1:],
        "label_binary",
        "inclusion_status",
        "n_bipolar_pairs_after_bad_channel_removal",
    ]

    def normalized(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame[compare_columns].copy()
        output["patient"] = output["patient"].astype(str)
        output["desikan_killiany_label"] = output[
            "desikan_killiany_label"
        ].map(region_model.normalize_region_label)
        output["tissue_class"] = output["tissue_class"].astype(str)
        output["is_white_csf_ventricle_unknown"] = output[
            "is_white_csf_ventricle_unknown"
        ].map(region_model.truthy)
        output["label_binary"] = pd.to_numeric(
            output["label_binary"], errors="coerce"
        ).fillna(-1.0)
        output["n_bipolar_pairs_after_bad_channel_removal"] = pd.to_numeric(
            output["n_bipolar_pairs_after_bad_channel_removal"], errors="raise"
        ).astype(int)
        return output.sort_values(REGION_KEYS[1:], kind="mergesort").reset_index(
            drop=True
        )

    locked_normalized = normalized(labels)
    derived_normalized = normalized(derived)
    try:
        pd.testing.assert_frame_equal(
            locked_normalized,
            derived_normalized,
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError as error:
        raise SystemExit(
            "Locked region label CSV does not match labels re-derived from the "
            f"current audit workbook: {error}"
        ) from error
    return {
        "audit_workbook": str(audit_workbook),
        "audit_sha256": _sha256(audit_workbook),
        "derived_label_rows": len(derived),
        "derived_primary_rows": int(
            derived["inclusion_status"].eq("primary").sum()
        ),
        "changed_from_current_audit_status": int(
            derivation_audit["label_changed_from_audit_status"].sum()
        ),
        "locked_csv_matches_rederived_labels": True,
    }


def _assert_selected_features_match_band(
    frame: pd.DataFrame,
    band_column: str = "frequency_band",
) -> None:
    for row in frame.itertuples(index=False):
        band = str(getattr(row, band_column))
        selected = str(getattr(row, "selected_features"))
        features = [value for value in selected.split("|") if value]
        if len(features) != 5:
            raise SystemExit(
                f"Expected five selected features for {band}, got {len(features)}"
            )
        invalid = [
            feature
            for feature in features
            if f"__{band}__" not in feature
        ]
        if invalid:
            raise SystemExit(
                f"Selected feature crosses frequency bands for {band}: {invalid[0]}"
            )


def validate_reused_network_branch(
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    source = config["reused_network_to_electrode"]
    paths = {key: Path(str(value)) for key, value in source.items()}
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise SystemExit(f"Missing reused network-to-electrode result: {missing[0]}")
    predictions = pd.read_csv(paths["observed_predictions"])
    metrics = pd.read_csv(paths["observed_metrics"])
    folds = pd.read_csv(paths["observed_folds"])
    comparison = pd.read_csv(paths["observed_comparison"])
    predictions = predictions[
        predictions["direction"].eq("network_to_electrode")
    ].copy()
    metrics = metrics[metrics["direction"].eq("network_to_electrode")].copy()
    folds = folds[folds["direction"].eq("network_to_electrode")].copy()
    comparison = comparison[
        comparison["direction"].eq("network_to_electrode")
    ].copy()
    expected = config["expected"]
    bands = {str(value) for value in config["frequency_bands"]}
    checks = {
        "predictions": (
            len(predictions),
            int(expected["reused_network_to_electrode_prediction_rows"]),
        ),
        "folds": (
            len(folds),
            int(expected["reused_network_to_electrode_fold_rows"]),
        ),
        "metrics": (len(metrics), int(expected["metric_rows_per_direction"])),
        "comparison": (
            len(comparison),
            int(expected["metric_rows_per_direction"]),
        ),
    }
    for name, (actual, required) in checks.items():
        if actual != required:
            raise SystemExit(
                f"Reused network branch {name} count mismatch: {actual} != {required}"
            )
    for name, frame in {
        "predictions": predictions,
        "folds": folds,
        "metrics": metrics,
        "comparison": comparison,
    }.items():
        if set(frame["frequency_band"].astype(str)) != bands:
            raise SystemExit(f"Reused network branch has incomplete bands in {name}")
    if predictions.duplicated(cross_model.ELECTRODE_KEYS).any():
        raise SystemExit("Reused network predictions contain duplicate identities")
    _assert_selected_features_match_band(folds)

    native = pd.read_csv(Path(str(config["native_electrode_predictions"])))
    native = native[native["frequency_band"].astype(str).isin(bands)].copy()
    cross_model._validate_prediction_alignment(
        predictions,
        native,
        cross_model.ELECTRODE_KEYS,
    )
    audit = {
        "reuse_status": "reused_verified_not_retrained",
        "prediction_rows": len(predictions),
        "fold_rows": len(folds),
        "metric_rows": len(metrics),
        "source_files": {
            key: {"path": str(path), "sha256": _sha256(path)}
            for key, path in paths.items()
        },
        "native_electrode_predictions": {
            "path": str(config["native_electrode_predictions"]),
            "sha256": _sha256(Path(str(config["native_electrode_predictions"]))),
        },
    }
    return predictions, metrics, comparison, audit


def validate_region_inputs(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, object],
) -> dict[str, object]:
    expected = config["expected"]
    if len(features) != int(expected["region_feature_rows"]):
        raise SystemExit(
            f"Region feature row mismatch: {len(features)} != "
            f"{expected['region_feature_rows']}"
        )
    joined = region_model.join_features_and_labels(features, labels)
    if len(joined) != int(expected["primary_region_rows"]):
        raise SystemExit(
            f"Primary joined region count mismatch: {len(joined)} != "
            f"{expected['primary_region_rows']}"
        )
    bands = [str(value) for value in config["frequency_bands"]]
    direction = config["electrode_to_region"]
    variants = [
        f"{method}_{aggregation}"
        for method in direction["methods"]
        for aggregation in direction["contact_aggregations"]
    ]
    coverage = region_model.validate_feature_method_coverage(
        features,
        bands,
        variants,
        expected_per_method_per_band=int(
            direction["expected_candidates_per_method_aggregation_per_band"]
        ),
    )
    return {
        "region_feature_rows": len(features),
        "primary_region_rows": len(joined),
        "candidate_coverage": coverage,
    }


def run_electrode_to_region(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling = config["modeling"]
    bands = [str(value) for value in config["frequency_bands"]]
    predictions, fold_records = region_model.run_lopo_predictions_by_band(
        features,
        labels,
        bands,
        top_k_features=int(modeling["top_k_features"]),
        c_values=[float(value) for value in modeling["c_values"]],
        random_seed=int(modeling["random_seed"]),
        non_missing_threshold=float(modeling["feature_non_missing_threshold"]),
    )
    metrics = region_model.compute_prediction_metrics_by_band(predictions)
    folds = pd.DataFrame.from_records(fold_records)
    predictions.insert(0, "direction", "electrode_to_region")
    predictions.insert(1, "target_level", "brain_region")
    predictions["label_rule"] = "ablation_top4_rule"
    predictions["univariate_orientation_rule"] = (
        region_model.UNIVARIATE_ORIENTATION_RULE
    )
    metrics.insert(0, "direction", "electrode_to_region")
    metrics.insert(1, "target_level", "brain_region")
    metrics["label_rule"] = "ablation_top4_rule"
    metrics["univariate_orientation_rule"] = (
        region_model.UNIVARIATE_ORIENTATION_RULE
    )
    folds.insert(0, "direction", "electrode_to_region")
    folds.insert(1, "target_level", "brain_region")
    folds["label_rule"] = "ablation_top4_rule"
    return predictions, metrics, folds


def validate_region_outputs(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    config: dict[str, object],
) -> None:
    expected = config["expected"]
    checks = {
        "predictions": (
            len(predictions),
            int(expected["electrode_to_region_prediction_rows"]),
        ),
        "folds": (len(folds), int(expected["electrode_to_region_fold_rows"])),
        "metrics": (len(metrics), int(expected["metric_rows_per_direction"])),
    }
    for name, (actual, required) in checks.items():
        if actual != required:
            raise SystemExit(
                f"Electrode-to-region {name} count mismatch: {actual} != {required}"
            )
    bands = {str(value) for value in config["frequency_bands"]}
    for name, frame in {
        "predictions": predictions,
        "folds": folds,
        "metrics": metrics,
    }.items():
        if set(frame["frequency_band"].astype(str)) != bands:
            raise SystemExit(f"Electrode-to-region has incomplete bands in {name}")
        if "univariate_orientation_rule" not in frame.columns:
            raise SystemExit(f"Electrode-to-region {name} lacks orientation provenance")
        rules = set(frame["univariate_orientation_rule"].astype(str))
        if rules != {region_model.UNIVARIATE_ORIENTATION_RULE}:
            raise SystemExit(
                f"Electrode-to-region {name} uses an unreviewed orientation rule: "
                f"{sorted(rules)}"
            )
    if predictions.duplicated(REGION_KEYS).any():
        raise SystemExit("Electrode-to-region predictions contain duplicate identities")
    _assert_selected_features_match_band(folds)
    expected_patients = {str(value) for value in config["patients"]}
    for row in folds.itertuples(index=False):
        heldout = str(row.heldout_patient)
        train = {value for value in str(row.train_patients).split("|") if value}
        if heldout in train or train != expected_patients - {heldout}:
            raise SystemExit(
                f"LOPO training-patient leakage or omission for heldout={heldout}"
            )


def build_region_comparison(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    config: dict[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    native_predictions_path = Path(str(config["native_region_predictions"]))
    native_metrics_path = Path(str(config["native_region_metrics"]))
    native_correction_path = Path(
        str(config["native_region_multiple_comparison"])
    )
    for path in [
        native_predictions_path,
        native_metrics_path,
        native_correction_path,
    ]:
        if not path.exists():
            raise SystemExit(f"Missing native top-4 region result: {path}")
    bands = {str(value) for value in config["frequency_bands"]}
    native_predictions = pd.read_csv(native_predictions_path)
    native_predictions = native_predictions[
        native_predictions["frequency_band"].astype(str).isin(bands)
    ].copy()
    native_metrics = pd.read_csv(native_metrics_path)
    native_metrics = native_metrics[
        native_metrics["frequency_band"].astype(str).isin(bands)
    ].copy()
    native_correction = pd.read_csv(native_correction_path)
    native_correction = native_correction[
        native_correction["frequency_band"].astype(str).isin(bands)
    ].copy()
    if len(native_correction) != len(bands):
        raise SystemExit("Native top-4 multiple-comparison table is incomplete")
    native_metrics = native_metrics.merge(
        native_correction[
            [
                "frequency_band",
                "permutation_p_raw_AUPRC",
                "permutation_p_fwer_bonferroni_AUPRC",
            ]
        ],
        on="frequency_band",
        how="left",
        validate="one_to_one",
    )
    cross_model._validate_prediction_alignment(
        predictions,
        native_predictions,
        REGION_KEYS,
    )
    native_by_band = native_metrics.set_index("frequency_band")
    rows: list[dict[str, object]] = []
    for row in metrics.to_dict("records"):
        band = str(row["frequency_band"])
        native = native_by_band.loc[band]
        record: dict[str, object] = {
            "direction": "electrode_to_region",
            "target_level": "brain_region",
            "frequency_band": band,
            "label_rule": "ablation_top4_rule",
            "execution_status": (
                "rerun_clean_training_only_orientation_top4_labels"
            ),
            "univariate_orientation_rule": (
                region_model.UNIVARIATE_ORIENTATION_RULE
            ),
            "native_method": "wpli2_plus_imcoh_to_region",
            "cross_method": "line_length_plus_band_power_to_region",
        }
        for metric in ["AUPRC", "normalized_AUPRC", "AUROC"]:
            native_value = float(native[metric])
            cross_value = float(row[metric])
            record[f"native_{metric}"] = native_value
            record[f"cross_{metric}"] = cross_value
            record[f"delta_cross_minus_native_{metric}"] = (
                cross_value - native_value
            )
        record["native_permutation_p_raw"] = native.get(
            "permutation_p_raw_AUPRC", np.nan
        )
        record["native_permutation_p_fwer_bonferroni"] = native.get(
            "permutation_p_fwer_bonferroni_AUPRC", np.nan
        )
        rows.append(record)
    return pd.DataFrame.from_records(rows), native_predictions


def bootstrap_region_comparison(
    predictions: pd.DataFrame,
    native_predictions: pd.DataFrame,
    n_bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    bootstrap = cross_model._bootstrap_direction_metrics(
        "electrode_to_region",
        predictions,
        native_predictions,
        n_bootstrap,
        seed,
    )
    ci_rows: list[dict[str, object]] = []
    for band, frame in bootstrap.groupby("frequency_band", sort=True):
        for metric in ["AUPRC", "normalized_AUPRC", "AUROC"]:
            for family in ["cross", "native", "delta_cross_minus_native"]:
                column = f"{family}_{metric}"
                values = frame[column].dropna()
                ci_rows.append(
                    {
                        "direction": "electrode_to_region",
                        "frequency_band": band,
                        "estimate": column,
                        "ci_low": (
                            float(values.quantile(0.025))
                            if not values.empty
                            else np.nan
                        ),
                        "ci_high": (
                            float(values.quantile(0.975))
                            if not values.empty
                            else np.nan
                        ),
                    }
                )
    return bootstrap, pd.DataFrame.from_records(ci_rows)


def build_current_comparison(
    reused_comparison: pd.DataFrame,
    region_comparison: pd.DataFrame,
) -> pd.DataFrame:
    network = reused_comparison.copy()
    network["label_rule"] = "clinician_ablated_evaluable_bipolar_points"
    network["execution_status"] = "reused_verified_not_retrained"
    columns = list(region_comparison.columns)
    for column in columns:
        if column not in network.columns:
            network[column] = np.nan
    for column in network.columns:
        if column not in region_comparison.columns:
            region_comparison[column] = np.nan
    return pd.concat(
        [network[region_comparison.columns], region_comparison],
        ignore_index=True,
    )


def write_outputs(
    out_dir: Path,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    folds: pd.DataFrame,
    bootstrap: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    comparison: pd.DataFrame,
    config_path: Path,
    validation: dict[str, object],
    reuse_audit: dict[str, object],
    orientation_contract: dict[str, str],
    n_bootstrap: int,
    overwrite: bool,
) -> dict[str, object]:
    paths = output_paths(out_dir)
    _preflight_outputs(paths.values(), overwrite)
    frames = {
        "predictions": predictions,
        "metrics": metrics,
        "folds": folds,
        "bootstrap": bootstrap,
        "bootstrap_ci": bootstrap_ci,
        "comparison": comparison,
    }
    for key, frame in frames.items():
        _atomic_write_csv(frame, paths[key])
    summary: dict[str, object] = {
        "status": "complete",
        "experiment": "cross_scale_models_ablation_top4_rule",
        "rerun_direction": "electrode_to_region",
        "reused_direction": "network_to_electrode",
        "permutation_tests_run": False,
        "n_bootstrap": n_bootstrap,
        "univariate_orientation": orientation_contract,
        "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        "implementation_provenance": {
            "driver": {
                "path": str(DRIVER_PATH),
                "sha256": _sha256(DRIVER_PATH),
            },
            "region_model": {
                "path": str(REGION_MODEL_PATH),
                "sha256": _sha256(REGION_MODEL_PATH),
            },
        },
        "validation": validation,
        "reused_network_to_electrode": reuse_audit,
        "files": {
            key: {
                "path": str(paths[key]),
                "rows": len(frame),
                "sha256": _sha256(paths[key]),
            }
            for key, frame in frames.items()
        },
    }
    _atomic_write_json(summary, paths["summary"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--n-bootstrap", type=int, default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.apply and args.dry_run:
        raise SystemExit("--apply and --dry-run are mutually exclusive")
    if args.overwrite and not args.apply:
        raise SystemExit("--overwrite requires --apply")
    assert_allowed_output_root(args.out)
    config = load_config(args.config)
    orientation_contract = validate_orientation_contract(config)
    configured_output = _resolve(Path(str(config["output_root"])))
    if _resolve(args.out) != configured_output and _resolve(args.out) != _resolve(OUTPUT_ROOT):
        if not _is_relative_to(_resolve(args.out), _resolve(CHECK_ROOT)):
            raise SystemExit("--out must match the configured formal output or checks")
    modeling = config["modeling"]
    n_bootstrap = (
        int(modeling["n_bootstrap_default"])
        if args.n_bootstrap is None
        else int(args.n_bootstrap)
    )
    if n_bootstrap <= 0:
        raise SystemExit("--n-bootstrap must be positive")

    feature_path = Path(str(config["region_feature_file"]))
    label_path = Path(str(config["region_label_file"]))
    if not feature_path.exists():
        raise SystemExit(f"Missing cross-level region feature file: {feature_path}")
    features = pd.read_csv(feature_path)
    labels, label_counts = load_locked_region_labels(
        label_path, config["expected"]
    )
    audit_validation = validate_locked_labels_against_audit(
        labels,
        Path(str(config["audit_workbook"])),
        config["region_label_rule"],
    )
    region_validation = validate_region_inputs(features, labels, config)
    _, _, reused_comparison, reuse_audit = validate_reused_network_branch(config)
    validation = {
        "label_counts": label_counts,
        "audit_label_crosscheck": audit_validation,
        "region_inputs": region_validation,
        "region_label_file": {
            "path": str(label_path),
            "sha256": _sha256(label_path),
        },
        "region_feature_file": {
            "path": str(feature_path),
            "sha256": _sha256(feature_path),
        },
        "native_region_sources": {
            key: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for key, path in {
                "predictions": Path(str(config["native_region_predictions"])),
                "metrics": Path(str(config["native_region_metrics"])),
                "multiple_comparison": Path(
                    str(config["native_region_multiple_comparison"])
                ),
            }.items()
            if path.exists()
        },
    }
    if len(validation["native_region_sources"]) != 3:
        raise SystemExit("One or more native top-4 region source files are missing")
    if args.dry_run or not args.apply:
        print(
            json.dumps(
                {
                    "mode": "dry_run",
                    "formal_write_requested": False,
                    "rerun_direction": "electrode_to_region",
                    "reused_direction": "network_to_electrode",
                    "frequency_bands": config["frequency_bands"],
                    "n_bootstrap": n_bootstrap,
                    "univariate_orientation": orientation_contract,
                    "output_root": str(args.out),
                    "validation": validation,
                    "reuse_audit": reuse_audit,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    _preflight_outputs(output_paths(args.out).values(), args.overwrite)

    predictions, metrics, folds = run_electrode_to_region(
        features, labels, config
    )
    validate_region_outputs(predictions, metrics, folds, config)
    region_comparison, native_region_predictions = build_region_comparison(
        predictions, metrics, config
    )
    bootstrap, bootstrap_ci = bootstrap_region_comparison(
        predictions,
        native_region_predictions,
        n_bootstrap,
        int(modeling["random_seed"]),
    )
    comparison = build_current_comparison(reused_comparison, region_comparison)
    summary = write_outputs(
        args.out,
        predictions,
        metrics,
        folds,
        bootstrap,
        bootstrap_ci,
        comparison,
        args.config,
        validation,
        reuse_audit,
        orientation_contract,
        n_bootstrap,
        args.overwrite,
    )
    print(
        region_comparison[
            [
                "frequency_band",
                "native_AUPRC",
                "cross_AUPRC",
                "delta_cross_minus_native_AUPRC",
            ]
        ].to_string(index=False)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
