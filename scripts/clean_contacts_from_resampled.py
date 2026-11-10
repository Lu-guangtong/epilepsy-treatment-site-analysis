"""Create clean monopolar-contact FIF files from 2 kHz raw FIF segments.

This is an atomic channel-removal step only. It removes auxiliary channels and
Brainstorm monopolar contacts whose Implantation ChannelFlag is -1, then
reorders kept contacts to the official Implantation channel order. It does not
filter, notch, build a bipolar montage, or read outcome/ablation labels.

Inspect the processing plan with --dry-run before execution:
  python scripts/clean_contacts_from_resampled.py --dry-run

Full cleaning requires review of the processing plan before execution.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import re
from pathlib import Path
from typing import Iterable, NamedTuple

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
RESAMPLED_ROOT = ROOT / "data_derived" / "resampled"
CLEAN_ROOT = ROOT / "data_derived" / "clean_contacts"
CHECK_ROOT = ROOT / "checks"
BST_DATA_ROOT = (source_root("brainstorm") / "data")
TARGET_SFREQ_HZ = 2000.0
CONVERSION_MODE = "drop_aux_and_bad_monopolar_only"
PATIENT_TO_BST_FOLDER = private_value("cohort", "patient_to_bst_folder")


class PatientChannelRules(NamedTuple):
    patient: str
    source_folder: str
    official_order: list[str]
    bad_contacts: set[str]
    keep_order: list[str]


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(out_dir: Path) -> None:
    """Reject writes outside reviewed formal output roots."""
    assert_output_path(out_dir)
    out_dir = _resolve(out_dir)
    clean_root = _resolve(CLEAN_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == clean_root
        or _is_relative_to(out_dir, clean_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def assert_output_available(out_file: Path, overwrite: bool) -> None:
    if out_file.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite to replace it: {out_file}")


def assert_zero_timebase(raw: mne.io.BaseRaw, source: Path) -> None:
    if int(raw.first_samp) != 0 or float(raw.first_time) != 0.0:
        raise SystemExit(
            f"{source}: source FIF is not an independent segment "
            f"(first_samp={raw.first_samp}, first_time={raw.first_time}); "
            "run normalize_segment_timebase.py before contact cleaning"
        )


def _normalize_patients(patients: Iterable[str] | None) -> set[str] | None:
    if patients is None:
        return None
    return {p.strip().lower() for p in patients if p.strip()}


def _normalize_trials(trials: Iterable[int] | None) -> set[int] | None:
    if trials is None:
        return None
    return {int(t) for t in trials}


def brainstorm_folder_for_patient(patient: str) -> str:
    patient_key = patient.strip().lower()
    try:
        return PATIENT_TO_BST_FOLDER[patient_key]
    except KeyError as exc:
        raise SystemExit(f"No Brainstorm folder mapping for patient: {patient}") from exc


def _brainstorm_implantation_folder(patient: str) -> Path:
    return BST_DATA_ROOT / brainstorm_folder_for_patient(patient) / "Implantation"


def _mat_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item()).strip()
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in arr.ravel()).strip()
    return str(value).strip()


def read_channel_mat_names(implantation_folder: Path) -> list[str]:
    channel_path = Path(implantation_folder) / "channel.mat"
    if not channel_path.exists():
        raise SystemExit(f"Missing Brainstorm channel.mat: {channel_path}")
    mat = loadmat(channel_path, squeeze_me=True, struct_as_record=False, variable_names=["Channel"])
    if "Channel" not in mat:
        raise SystemExit(f"Brainstorm channel.mat has no Channel variable: {channel_path}")
    channels = np.atleast_1d(mat["Channel"])
    names = [_mat_string(getattr(channel, "Name", "")) for channel in channels]
    names = [name for name in names if name]
    if not names:
        raise SystemExit(f"No channel names found in Brainstorm channel.mat: {channel_path}")
    if len(names) != len(set(names)):
        raise SystemExit(f"Duplicate channel names in Brainstorm channel.mat: {channel_path}")
    return names


def _read_channelflag_bad_contacts(data_path: Path, official_order: list[str]) -> set[str]:
    mat = loadmat(data_path, squeeze_me=True, struct_as_record=False, variable_names=["ChannelFlag"])
    if "ChannelFlag" not in mat:
        raise SystemExit(f"Brainstorm data file has no ChannelFlag variable: {data_path}")
    flags = np.asarray(mat["ChannelFlag"]).reshape(-1)
    if flags.size != len(official_order):
        raise SystemExit(
            f"ChannelFlag length mismatch in {data_path}: "
            f"{flags.size} flags for {len(official_order)} official contacts"
        )
    return {official_order[idx] for idx, flag in enumerate(flags) if float(flag) == -1.0}


def assert_consistent_bad_sets(bad_sets: list[set[str]], source_names: list[str]) -> set[str]:
    if not bad_sets:
        raise SystemExit("No Brainstorm data_*.mat files found for ChannelFlag consensus")
    first = set(bad_sets[0])
    first_name = source_names[0] if source_names else "<first>"
    for bad_set, source_name in zip(bad_sets[1:], source_names[1:], strict=False):
        if set(bad_set) != first:
            added = sorted(set(bad_set) - first)
            removed = sorted(first - set(bad_set))
            raise SystemExit(
                "ChannelFlag bad-contact mismatch across Brainstorm data files: "
                f"{first_name} vs {source_name}; added={added}; removed={removed}"
            )
    return first


def read_consensus_bad_contacts(implantation_folder: Path, official_order: list[str]) -> set[str]:
    data_files = sorted(Path(implantation_folder).glob("data_*.mat"))
    bad_sets: list[set[str]] = []
    source_names: list[str] = []
    for data_file in data_files:
        bad_sets.append(_read_channelflag_bad_contacts(data_file, official_order))
        source_names.append(data_file.name)
    return assert_consistent_bad_sets(bad_sets, source_names)


def build_patient_channel_rules(patient: str) -> PatientChannelRules:
    patient_key = patient.strip().lower()
    source_folder = brainstorm_folder_for_patient(patient_key)
    implantation_folder = _brainstorm_implantation_folder(patient_key)
    if not implantation_folder.exists():
        raise SystemExit(f"Missing Brainstorm Implantation folder: {implantation_folder}")
    official_order = read_channel_mat_names(implantation_folder)
    bad_contacts = read_consensus_bad_contacts(implantation_folder, official_order)
    missing_bad = sorted(bad_contacts - set(official_order))
    if missing_bad:
        raise SystemExit(f"Bad contacts not present in official channel order for {patient_key}: {missing_bad}")
    keep_order = [channel for channel in official_order if channel not in bad_contacts]
    return PatientChannelRules(
        patient=patient_key,
        source_folder=source_folder,
        official_order=official_order,
        bad_contacts=bad_contacts,
        keep_order=keep_order,
    )


def classify_aux_reason(channel: str) -> str:
    upper = channel.strip().upper()
    if upper.startswith("ECG"):
        return "aux_ecg"
    if re.fullmatch(r"BP\s*\d+", upper):
        return "aux_device(BP)"
    return "aux_not_in_montage"


def validate_official_subset_present(patient: str, official_order: Iterable[str], fif_channels: Iterable[str]) -> None:
    fif_set = set(fif_channels)
    missing = [channel for channel in official_order if channel not in fif_set]
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f" ... ({len(missing)} total)"
        raise SystemExit(f"{patient}: official channels not present in FIF: {preview}{suffix}")


def _read_fif_channel_names(fif_path: Path) -> list[str]:
    if not fif_path.exists():
        raise SystemExit(f"Missing 2 kHz FIF source file: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    return list(raw.ch_names)


def build_trial_plan(
    manifest_path: Path = RESAMPLED_ROOT / "resample_manifest.csv",
    out_root: Path = CLEAN_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build one row per 2 kHz FIF segment selected for contact cleaning."""
    assert_allowed_output_root(out_root)
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"Missing resample manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"patient", "trial", "out_2k_fif", "target_sfreq_hz", "n_channels"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Resample manifest missing required columns: {', '.join(missing)}")

    selected = df.copy()
    patient_filter = _normalize_patients(patients)
    if patient_filter is not None:
        selected = selected[selected["patient"].astype(str).str.lower().isin(patient_filter)].copy()
    trial_filter = _normalize_trials(trials)
    if trial_filter is not None:
        selected = selected[selected["trial"].astype(int).isin(trial_filter)].copy()

    duplicate = selected.duplicated(["patient", "trial"], keep=False)
    if duplicate.any():
        dupes = selected.loc[duplicate, ["patient", "trial"]].to_dict("records")
        raise SystemExit(f"Duplicate patient/trial rows in resample manifest: {dupes}")

    bad_sfreq = selected[selected["target_sfreq_hz"].astype(float) != TARGET_SFREQ_HZ]
    if not bad_sfreq.empty:
        rows = bad_sfreq[["patient", "trial", "target_sfreq_hz"]].to_dict("records")
        raise SystemExit(f"Unexpected target sampling rate in resample manifest: {rows}")

    records: list[dict[str, object]] = []
    out_root = Path(out_root)
    for _, row in selected.sort_values(["patient", "trial"]).iterrows():
        patient = str(row["patient"]).lower()
        trial = int(row["trial"])
        in_file = Path(str(row["out_2k_fif"]))
        records.append(
            {
                "patient": patient,
                "trial": trial,
                "source_folder": brainstorm_folder_for_patient(patient),
                "in_file": str(in_file),
                "out_file": str(out_root / patient / f"{patient}_trial{trial:02d}_clean_raw.fif"),
                "sfreq_hz": float(row["target_sfreq_hz"]),
                "n_channels_manifest": int(row["n_channels"]),
                "conversion_mode": CONVERSION_MODE,
            }
        )
    return pd.DataFrame.from_records(records)


def validate_trial_sources(trial_plan: pd.DataFrame) -> None:
    missing = [Path(str(path)) for path in trial_plan["in_file"] if not Path(str(path)).exists()]
    if missing:
        raise SystemExit("Missing 2 kHz FIF source files:\n" + "\n".join(str(path) for path in missing))


def _assert_unique_channel_names(patient: str, trial: int, channel_names: list[str]) -> None:
    if len(channel_names) != len(set(channel_names)):
        raise SystemExit(f"{patient} trial {trial}: duplicate FIF channel names detected")


def _channel_manifest_rows_for_trial(
    trial_record: pd.Series,
    rules: PatientChannelRules,
    fif_channels: list[str],
) -> list[dict[str, object]]:
    patient = str(trial_record["patient"])
    trial = int(trial_record["trial"])
    _assert_unique_channel_names(patient, trial, fif_channels)
    validate_official_subset_present(f"{patient} trial {trial}", rules.official_order, fif_channels)

    official_set = set(rules.official_order)
    aux_channels = [channel for channel in fif_channels if channel not in official_set]
    bad_channels = [channel for channel in rules.official_order if channel in rules.bad_contacts]
    n_in = len(fif_channels)
    n_aux = len(aux_channels)
    n_bad = len(bad_channels)
    n_kept = len(rules.keep_order)
    if n_kept + n_bad + n_aux != n_in:
        raise SystemExit(
            f"{patient} trial {trial}: channel accounting mismatch "
            f"n_in={n_in}, n_kept={n_kept}, n_bad={n_bad}, n_aux={n_aux}"
        )

    base = {
        "patient": patient,
        "trial": trial,
        "source_folder": rules.source_folder,
        "in_file": str(trial_record["in_file"]),
        "out_file": str(trial_record["out_file"]),
        "n_in": n_in,
        "n_kept": n_kept,
        "n_aux": n_aux,
        "n_bad": n_bad,
    }
    rows: list[dict[str, object]] = []
    for channel in rules.keep_order:
        rows.append({**base, "channel": channel, "role": "kept_official", "reason": "official_good"})
    for channel in bad_channels:
        rows.append(
            {**base, "channel": channel, "role": "removed_bad", "reason": "monopolar_channelflag=-1"}
        )
    for channel in aux_channels:
        rows.append({**base, "channel": channel, "role": "removed_aux", "reason": classify_aux_reason(channel)})
    return rows


def build_clean_manifest(trial_plan: pd.DataFrame) -> pd.DataFrame:
    """Build the per-input-channel manifest without loading FIF data samples."""
    if trial_plan.empty:
        return pd.DataFrame(
            columns=[
                "patient",
                "trial",
                "channel",
                "role",
                "reason",
                "source_folder",
                "in_file",
                "out_file",
                "n_in",
                "n_kept",
                "n_aux",
                "n_bad",
            ]
        )

    rules_cache: dict[str, PatientChannelRules] = {}
    rows: list[dict[str, object]] = []
    for _, trial_record in trial_plan.sort_values(["patient", "trial"]).iterrows():
        patient = str(trial_record["patient"])
        if patient not in rules_cache:
            rules_cache[patient] = build_patient_channel_rules(patient)
        fif_channels = _read_fif_channel_names(Path(str(trial_record["in_file"])))
        rows.extend(_channel_manifest_rows_for_trial(trial_record, rules_cache[patient], fif_channels))
    columns = [
        "patient",
        "trial",
        "channel",
        "role",
        "reason",
        "source_folder",
        "in_file",
        "out_file",
        "n_in",
        "n_kept",
        "n_aux",
        "n_bad",
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


def summarize_channel_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(columns=["patient", "n_trials", "n_in", "n_kept", "n_aux", "n_bad"])
    trial_counts = manifest[
        ["patient", "trial", "n_in", "n_kept", "n_aux", "n_bad"]
    ].drop_duplicates()
    inconsistent: list[str] = []
    for patient, patient_counts in trial_counts.groupby("patient", sort=True):
        unique_counts = patient_counts[["n_in", "n_kept", "n_aux", "n_bad"]].drop_duplicates()
        if len(unique_counts) != 1:
            inconsistent.append(str(patient))
    if inconsistent:
        raise SystemExit(f"Per-trial channel counts are not consistent for patients: {', '.join(inconsistent)}")
    rows: list[dict[str, object]] = []
    for patient, patient_counts in trial_counts.groupby("patient", sort=True):
        counts = patient_counts.iloc[0]
        rows.append(
            {
                "patient": patient,
                "n_trials": int(patient_counts["trial"].nunique()),
                "n_in": int(counts["n_in"]),
                "n_kept": int(counts["n_kept"]),
                "n_aux": int(counts["n_aux"]),
                "n_bad": int(counts["n_bad"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def _manifest_for_trial(manifest: pd.DataFrame, patient: str, trial: int) -> pd.DataFrame:
    selected = manifest[(manifest["patient"] == patient) & (manifest["trial"].astype(int) == int(trial))]
    if selected.empty:
        raise SystemExit(f"Missing channel manifest rows for {patient} trial {trial}")
    return selected


def clean_one_trial(trial_record: pd.Series, manifest: pd.DataFrame, overwrite: bool) -> None:
    patient = str(trial_record["patient"])
    trial = int(trial_record["trial"])
    in_file = Path(str(trial_record["in_file"]))
    out_file = Path(str(trial_record["out_file"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite)
    if not in_file.exists():
        raise SystemExit(f"Missing 2 kHz FIF source file: {in_file}")

    trial_manifest = _manifest_for_trial(manifest, patient, trial)
    keep_order = trial_manifest[trial_manifest["role"] == "kept_official"]["channel"].tolist()
    drop_channels = trial_manifest[trial_manifest["role"].isin(["removed_bad", "removed_aux"])]["channel"].tolist()

    raw = mne.io.read_raw_fif(in_file, preload=True, verbose="ERROR")
    assert_zero_timebase(raw, in_file)
    sfreq = float(raw.info["sfreq"])
    if sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected sampling rate in {in_file}: {sfreq}")
    missing_drop = [channel for channel in drop_channels if channel not in raw.ch_names]
    missing_keep = [channel for channel in keep_order if channel not in raw.ch_names]
    if missing_drop:
        raise SystemExit(f"{patient} trial {trial}: drop channels not present in raw FIF: {missing_drop}")
    if missing_keep:
        raise SystemExit(f"{patient} trial {trial}: keep channels not present in raw FIF: {missing_keep}")

    raw.drop_channels(drop_channels)
    raw.reorder_channels(keep_order)
    if list(raw.ch_names) != keep_order:
        raise SystemExit(f"{patient} trial {trial}: clean channel order did not match official order")
    if int(raw.info["nchan"]) != len(keep_order):
        raise SystemExit(f"{patient} trial {trial}: clean channel count mismatch")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_file, overwrite=overwrite, verbose="ERROR")


def validate_outputs_available(trial_plan: pd.DataFrame, overwrite: bool) -> None:
    for out_file in trial_plan["out_file"].map(Path):
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite)


def write_manifests(manifest: pd.DataFrame, out_root: Path) -> None:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for patient, patient_df in manifest.groupby("patient", sort=True):
        patient_dir = out_root / str(patient)
        patient_dir.mkdir(parents=True, exist_ok=True)
        patient_df.to_csv(patient_dir / f"{patient}_clean_contacts_manifest.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(out_root / "clean_contacts_manifest.csv", index=False, encoding="utf-8-sig")


def print_plan(trial_plan: pd.DataFrame, channel_manifest: pd.DataFrame, out_root: Path, mode: str) -> None:
    cols = ["patient", "trial", "sfreq_hz", "n_channels_manifest", "in_file", "out_file"]
    print(trial_plan[cols].to_string(index=False))
    print(f"\nmode={mode}  n_segments={len(trial_plan)}  out={out_root}")
    if not trial_plan.empty:
        print("\npatient segment counts:")
        print(trial_plan.groupby("patient").size().to_string())
    if not channel_manifest.empty:
        print("\nchannel accounting per trial:")
        print(summarize_channel_manifest(channel_manifest).to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Drop auxiliary and bad monopolar contacts from 2 kHz raw FIF files")
    ap.add_argument("--dry-run", action="store_true", help="print cleaning plan and write nothing")
    ap.add_argument("--patients", nargs="+", default=None, help="optional patient filter, e.g. --patients demo001 demo002")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional trial-id filter")
    ap.add_argument("--overwrite", action="store_true", help="replace existing clean FIF outputs")
    ap.add_argument("--out", default=None, help="override output directory under reviewed output roots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out) if args.out else CLEAN_ROOT
    assert_allowed_output_root(out_root)

    trial_plan = build_trial_plan(
        RESAMPLED_ROOT / "resample_manifest.csv",
        out_root,
        patients=args.patients,
        trials=args.trials,
    )
    if trial_plan.empty:
        raise SystemExit("No 2 kHz FIF segments selected for clean-contact generation.")
    validate_trial_sources(trial_plan)
    channel_manifest = build_clean_manifest(trial_plan)
    mode = "dry-run" if args.dry_run else "full"
    print_plan(trial_plan, channel_manifest, out_root, mode)

    if args.dry_run:
        return

    validate_outputs_available(trial_plan, overwrite=args.overwrite)
    for _, trial_record in trial_plan.iterrows():
        clean_one_trial(trial_record, channel_manifest, overwrite=args.overwrite)
    write_manifests(channel_manifest, out_root)
    print(f"\nmanifest: {out_root / 'clean_contacts_manifest.csv'}")


if __name__ == "__main__":
    main()
