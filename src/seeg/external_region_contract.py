"""Frozen ds004100 DK region cohort, label and signal contracts.

Treatment is used only to audit the prespecified cohort and construct targets.
The network functions receive signal/anatomy records, never these targets.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))
from pipeline_config import expand_config
RULE_ID = "dk_gray_both_end_treated_ge1_20260914_v1"
KEYS = ["patient", "desikan_killiany_label", "tissue_class",
        "is_white_csf_ventricle_unknown"]
METHODS = ("wpli2_debiased", "imcoh")
# Same boundaries, epoch lengths and overlap as the original formal experiment.
BANDS = {
    "delta": (1.0, 4.0, 5.0), "theta": (4.0, 8.0, 5.0),
    "alpha": (8.0, 13.0, 5.0), "beta": (13.0, 30.0, 5.0),
    "low_gamma": (30.0, 80.0, 2.0), "high_gamma": (80.0, 150.0, 2.0),
    "ripple": (80.0, 250.0, 2.0), "fast_ripple": (250.0, 500.0, 2.0),
}
WINDOWS = {"baseline": (0.0, 35.0), "early_ictal": (40.0, 50.0)}
BASE_FEATURES = ("within_mean", "within_q90", "out_mean", "out_q90",
                 "node_strength_mean_normalized", "node_strength_q90_normalized")
AGGREGATIONS = ("trial_mean", "trial_median", "trial_iqr", "trial_q90",
                "trial_top25_fraction_within_patient")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode("utf-8")).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=None)
def load_analysis_module(name: str):
    """Import reviewed pure functions without invoking the original CLI."""
    allowed = {"extract_region_network_features", "run_region_lopo_modeling"}
    if name not in allowed:
        raise ValueError(f"Unsupported formal dependency: {name}")
    spec = importlib.util.spec_from_file_location(
        f"external_dependency_{name}", PROJECT_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> dict:
    cfg = expand_config(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
    cfg["config_path"] = str(Path(path).resolve())
    for name, value in cfg["inputs"].items():
        source = Path(value)
        cfg["inputs"][name] = str(source if source.is_absolute()
                                   else (PROJECT_ROOT / source).resolve())
    bands = cfg["primary_bands"] + cfg["supplementary_bands"]
    if not bands or len(set(bands)) != len(bands) or set(bands) - BANDS.keys():
        raise ValueError("Configured bands must be unique canonical frequency bands")
    if cfg["label_rule_id"] != RULE_ID:
        raise ValueError("Unexpected external region label rule")
    return cfg


def validate_rule(rule: dict) -> None:
    checks = [
        rule.get("rule_id") == RULE_ID, rule.get("status") == "FROZEN_BY_USER",
        rule["patient_inclusion"]["tissue_class"] == "gray",
        rule["patient_inclusion"]["minimum_total_both_end_treated_pairs"] == 1,
        rule["patient_inclusion"]["only_mixed_treatment_does_not_qualify"] is True,
        rule["positive_region"] == {"tissue_class": "gray",
            "n_both_end_treated_pairs_greater_than_or_equal": 1, "label": 1},
        rule["negative_region"] == {"tissue_class": "gray",
            "n_both_end_treated_pairs_equal": 0, "label": 0,
            "mixed_only_regions_are_negative": True},
        rule.get("top_k_restriction") is None,
        rule.get("minimum_implanted_pair_restriction") is None,
        rule.get("fraction_threshold") is None,
        set(rule["excluded_tissues"]) == {"white", "csf", "ventricle", "unknown", "non_neural"},
    ]
    if not all(checks):
        raise ValueError("Frozen both-end-treated >=1 rule has changed")


def derive_labels(counts: list[dict]) -> tuple[pd.DataFrame, list[str]]:
    """No top-k, rate or minimum-coverage threshold; mixed-only means zero."""
    rows, seen = [], set()
    for row in counts:
        key = (row["subject"], row["atlas_region"], row["tissue_class"])
        if key in seen:
            raise ValueError(f"Duplicate region count: {key}")
        seen.add(key)
        n, b, m, neither = [row[c] for c in (
            "n_bipolar_pairs", "n_both_end_treated_pairs",
            "n_mixed_treatment_pairs", "n_neither_end_treated_pairs")]
        if any(type(v) is not int or v < 0 for v in (n, b, m, neither)):
            raise ValueError(f"Invalid treatment counts: {key}")
        if n != b + m + neither or n != len(set(row["bipolar_pairs"])) or n == 0:
            raise ValueError(f"Inconsistent treatment counts: {key}")
        if row["tissue_class"] != "gray":
            continue
        if row["region_model_eligible_tissue"] is not True:
            raise ValueError(f"Gray eligibility conflict: {key}")
        rows.append({"patient": row["subject"],
                     "desikan_killiany_label": row["atlas_region"],
                     "tissue_class": "gray", "is_white_csf_ventricle_unknown": False,
                     "label_binary": int(b >= 1), "n_bipolar_pairs": n,
                     "n_both_end_treated_pairs": b, "n_mixed_treatment_pairs": m})
    labels = pd.DataFrame(rows)
    if labels.empty:
        raise ValueError("No clear-gray region available for classification")
    totals = labels.groupby("patient")["n_both_end_treated_pairs"].sum()
    excluded = sorted({r["subject"] for r in counts} - set(totals[totals >= 1].index))
    labels = labels[~labels.patient.isin(excluded)].sort_values(KEYS).reset_index(drop=True)
    for patient, group in labels.groupby("patient"):
        if group.label_binary.nunique() != 2:
            raise ValueError(f"Patient lacks both regional classes: {patient}")
    return labels, excluded


@dataclass
class Inputs:
    config: dict
    signals: list[dict]
    anatomy: list[dict]
    labels: pd.DataFrame
    cohort: dict
    hashes: dict[str, str]


def load_inputs(cfg: dict, verify_fif_hashes: bool = False) -> Inputs:
    paths = {k: Path(v) for k, v in cfg["inputs"].items()}
    hashes = {k: sha256(p) for k, p in paths.items()}
    rule, cohort = read_json(paths["rule"]), read_json(paths["cohort"])
    validate_rule(rule)
    for key, field in [("counts", "source_count_sha256"),
                       ("pair_audit", "source_pair_audit_sha256")]:
        if hashes[key] != rule[field]:
            raise ValueError(f"Frozen source hash mismatch: {key}")
    labels, excluded = derive_labels(read_json(paths["counts"]))
    patients = sorted(labels.patient.unique())
    if cohort["rule_id"] != RULE_ID or patients != sorted(cohort["included_subjects"]):
        raise ValueError("Active cohort differs from treatment-derived eligible patients")
    if excluded != sorted(r["subject"] for r in cohort["excluded_subjects"]):
        raise ValueError("Excluded patient list differs from the frozen rule")
    frozen = read_json(paths["labels"])
    expected = {(r.patient, r.desikan_killiany_label): r for r in labels.itertuples()}
    actual = {(r["subject"], r["atlas_region"]): r for r in frozen}
    if len(actual) != len(frozen) or set(actual) != set(expected):
        raise ValueError("Frozen label table has duplicate or missing region keys")
    for key, target in expected.items():
        row = actual[key]
        if (row["label"] != target.label_binary
                or row["suspected_region_label"] != target.label_binary
                or row["tissue_class"] != "gray" or row["label_rule_id"] != RULE_ID
                or row["n_bipolar_pairs"] != target.n_bipolar_pairs
                or row["n_both_end_treated_pairs"] != target.n_both_end_treated_pairs
                or row["n_mixed_treatment_pairs"] != target.n_mixed_treatment_pairs):
            raise ValueError(f"Frozen label mismatch: {key}")
    signals = read_json(paths["signals"])
    anatomy = [r for r in read_json(paths["anatomy"]) if r["subject"] in patients]
    anatomy_keys = {(r["subject"], r["bipolar_pair"]) for r in anatomy}
    if len(anatomy_keys) != len(anatomy):
        raise ValueError("Duplicate anatomy pair")
    for key, row in actual.items():
        pairs = {r["bipolar_pair"] for r in anatomy
                 if (r["subject"], r["atlas_region"]) == key and r["tissue_class"] == "gray"}
        if pairs != set(row["bipolar_pairs"]):
            raise ValueError(f"Anatomy and frozen region membership differ: {key}")
    expected_runs = {(r["subject"], run) for r in cohort["patients"] for run in r["included_runs"]}
    if {(r["patient"], r["trial"]) for r in signals} != expected_runs or len(signals) != len(expected_runs):
        raise ValueError("Active signal manifest has missing, extra or duplicate seizures")
    by_patient = {p: [r for r in signals if r["patient"] == p] for p in patients}
    for p, runs in by_patient.items():
        if len(runs) < 2 or len({r["sampling_hz"] for r in runs}) != 1:
            raise ValueError(f"Need >=2 seizures and stable native sampling: {p}")
        pairs = {r["bipolar_pair"] for r in anatomy if r["subject"] == p}
        for row in runs:
            validate_signal_record(row)
            if set(row["bipolar_channel_names"]) != pairs:
                raise ValueError(f"Signal/anatomy channel mismatch: {p}")
            if row["bipolar_channel_names"] != runs[0]["bipolar_channel_names"]:
                raise ValueError(f"Cross-seizure channel order differs: {p}")
            if verify_fif_hashes and sha256(Path(row["signal_path"])) != row["output_sha256"]:
                raise ValueError(f"Signal SHA256 mismatch: {row['signal_path']}")
    if any(r["network_eligible"] is not True for r in anatomy):
        raise ValueError("Anatomy contains non-network-eligible pairs")
    summary = {"patients": len(patients), "analysis_runs": len(signals),
               "common_pairs": len(anatomy), "gray_regions": len(labels),
               "positive_regions": int(labels.label_binary.sum()),
               "negative_regions": int((labels.label_binary == 0).sum())}
    for key, value in summary.items():
        if value != cfg["expected"][key] or value != cohort["summary"][key]:
            raise ValueError(f"Cohort control mismatch: {key}={value}")
    return Inputs(cfg, signals, anatomy, labels, cohort, hashes)


def validate_signal_record(row: dict) -> None:
    fs = row["sampling_hz"]
    names = row["bipolar_channel_names"]
    if (fs not in (512, 1024) or row["native_sampling_hz"] != fs
            or row["resampling_applied"] or row["qc_status"] != "PASS"
            or row["first_samp"] != 0 or row["first_time_s"] != 0
            or row["onset_sec"] != 40 or row["n_samples"] != 50 * fs
            or len(names) != row["n_channels"] or len(set(names)) != len(names)
            or not Path(row["signal_path"]).is_file()):
        raise ValueError(f"Invalid native signal contract: {row['patient']} run {row['trial']}")
    for field, limits in [("baseline_samples_half_open", [0, 35 * fs]),
                          ("early_samples_half_open", [40 * fs, 50 * fs])]:
        if row[field] != limits:
            raise ValueError(f"Window contract mismatch: {field}")


def band_eligibility(sfreq: float, band: str) -> tuple[bool, str]:
    """Computational eligibility only; does not certify acquisition passband."""
    _, high, duration = BANDS[band]
    half_bandwidth = (8.0 / duration) / 2.0
    if high + half_bandwidth >= sfreq / 2.0:
        return False, "upper_band_plus_multitaper_half_bandwidth_reaches_Nyquist"
    return True, "numerically_supported_hardware_passband_not_verified"


def selected_signals(inputs: Inputs, bands: list[str], smoke: bool) -> list[dict]:
    if not bands or set(bands) - set(BANDS):
        raise ValueError("Unknown or empty band selection")
    signals = inputs.signals
    if smoke:
        eligible = sorted({r["patient"] for r in signals
                           if all(band_eligibility(r["sampling_hz"], b)[0] for b in bands)})
        # Include native 1024 Hz in the ordinary 512 Hz smoke sample as well.
        chosen = eligible[:4]
        if len(chosen) < 4:
            raise ValueError("Smoke requires four band-eligible patients")
        signals = [r for p in chosen for r in sorted(
            [x for x in signals if x["patient"] == p], key=lambda x: x["trial"])[:2]]
    return signals


def make_plan(signals: list[dict], bands: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, availability = [], []
    for signal in signals:
        for band in bands:
            eligible, reason = band_eligibility(signal["sampling_hz"], band)
            availability.append({"patient": signal["patient"], "trial": signal["trial"],
                                 "frequency_band": band, "sampling_hz": signal["sampling_hz"],
                                 "eligible": eligible, "reason": reason})
            if not eligible:
                continue
            low, high, duration = BANDS[band]
            for window, (start, end) in WINDOWS.items():
                for method in METHODS:
                    rows.append({"patient": signal["patient"], "trial": signal["trial"],
                        "frequency_band": band, "window": window, "method": method,
                        "sampling_hz": signal["sampling_hz"], "fmin_hz": low, "fmax_hz": high,
                        "window_start_sec": start, "window_end_sec": end,
                        "sub_epoch_length_sec": duration, "overlap_fraction": 0.5,
                        "mt_bandwidth": 8.0 / duration,
                        "n_sub_epochs": int((end - start - duration) / (duration / 2)) + 1,
                        "n_channels": signal["n_channels"]})
    return pd.DataFrame(rows), pd.DataFrame(availability)


def candidate_features(band: str) -> list[str]:
    """Explicit allowlist: treatment, identity and QC can never become features."""
    if band not in BANDS:
        raise ValueError(f"Unknown band: {band}")
    ids = [f"trial__{w}__{band}__{m}__{f}"
           for w in WINDOWS for m in METHODS for f in BASE_FEATURES]
    ids += [f"contrast__{band}__{m}__{f}__{contrast}"
            for m in METHODS for f in BASE_FEATURES for contrast in ("delta", "log_ratio")]
    return sorted(f"{fid}__{agg}" for fid in ids for agg in AGGREGATIONS)


def input_summary(inputs: Inputs, signals: list[dict], bands: list[str]) -> dict:
    plan, availability = make_plan(signals, bands)
    return {"label_rule_id": RULE_ID, "frozen_cohort": inputs.cohort["summary"],
            "selected_patients": sorted({r["patient"] for r in signals}),
            "selected_runs": len(signals), "matrix_count": len(plan),
            "run_sampling_hz_counts": dict(Counter(str(r["sampling_hz"]) for r in signals)),
            "bands": bands, "eligible_patients_by_band": {
                b: sorted(availability.loc[(availability.frequency_band == b)
                    & availability.eligible, "patient"].unique()) for b in bands},
            "source_hashes": inputs.hashes, "hardware_passband_verified": False,
            "experiment_type": "external_method_replication_nested_LOPO_not_locked_model_transfer"}
