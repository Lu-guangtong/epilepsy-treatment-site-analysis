"""Build adjacent bipolar raw FIF files from clean monopolar-contact FIF files.

This is an atomic montage step only. It computes adjacent same-shaft bipolar
signals from clean monopolar contacts, applies the approved bipolar-level manual
exclusions, and joins nonclinical official anatomy columns. It does not filter,
notch, model, or read ablation/outcome labels.

Inspect the processing plan with --dry-run before execution:
  python scripts/build_bipolar_from_clean_contacts.py --dry-run

Full bipolar generation requires review of the processing plan before execution.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import re
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CLEAN_ROOT = ROOT / "data_derived" / "clean_contacts"
BIPOLAR_ROOT = ROOT / "data_derived" / "bipolar"
CHECK_ROOT = ROOT / "checks"
FINAL_AUDIT = source_root("data") / "annotations" / "clinical_annotations.xlsx"
BST_DATA_ROOT = (source_root("brainstorm") / "data")
TARGET_SFREQ_HZ = 2000.0
MONTAGE_SOURCE = "generated_adjacent_clean_contacts"
ANATOMY_SOURCE = "clinical_annotations.patient_bipolar_nonclinical_columns"
OFFICIAL_CROSSCHECK_SOURCE = "Implantation_bipolar_2/channel.mat"
CONTACT_RE = re.compile(r"^([A-Za-z]'?)(\d+)$")
ANATOMY_COLUMNS = [
    "patient_id",
    "bipolar_channel",
    "desikan_killiany_label",
    "tissue_class",
    "is_white_csf_ventricle_unknown",
    "is_bad_bipolar",
    "remove_as_bad_channel",
]
MANUAL_EXCLUDE_BIPOLARS = {patient: set(channels) for patient, channels in private_value("cohort", "manual_exclude_bipolars").items()}
PATIENT_TO_BST_FOLDER = private_value("cohort", "patient_to_bst_folder")


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
    bipolar_root = _resolve(BIPOLAR_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == bipolar_root
        or _is_relative_to(out_dir, bipolar_root)
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
            "run normalize_segment_timebase.py before building bipolar montage"
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


def _mat_string(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item()).strip()
    if arr.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in arr.ravel()).strip()
    return str(value).strip()


def _read_channel_mat_names(channel_path: Path) -> list[str]:
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


def read_official_bipolar_names(patient: str) -> set[str]:
    folder = brainstorm_folder_for_patient(patient)
    channel_path = BST_DATA_ROOT / folder / "Implantation_bipolar_2" / "channel.mat"
    return set(_read_channel_mat_names(channel_path))


def _anode_cathode_from_bipolar_name(name: str) -> tuple[str, str]:
    parts = str(name).split("-")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise SystemExit(f"Cannot parse bipolar channel name into anode/cathode: {name}")
    return parts[0], parts[1]


def load_anatomy_table(audit_path: Path = FINAL_AUDIT) -> pd.DataFrame:
    audit_path = Path(audit_path)
    if not audit_path.exists():
        raise SystemExit(f"Missing final audit workbook: {audit_path}")
    excel = pd.ExcelFile(audit_path)
    bipolar_sheets = [name for name in excel.sheet_names if name.endswith("_bipolar")]
    if not bipolar_sheets:
        raise SystemExit(f"Final audit workbook has no patient *_bipolar worksheets: {audit_path}")
    df = pd.concat(
        [
            pd.read_excel(excel, sheet_name=sheet, usecols=ANATOMY_COLUMNS)
            for sheet in bipolar_sheets
        ],
        ignore_index=True,
    )
    missing = sorted(set(ANATOMY_COLUMNS) - set(df.columns))
    if missing:
        raise SystemExit(f"final audit bipolar worksheets missing required columns: {', '.join(missing)}")
    duplicate = df.duplicated(["patient_id", "bipolar_channel"], keep=False)
    if duplicate.any():
        dupes = df.loc[duplicate, ["patient_id", "bipolar_channel"]].to_dict("records")
        raise SystemExit(f"Duplicate anatomy rows: {dupes}")
    df = df[ANATOMY_COLUMNS].copy()
    parsed = df["bipolar_channel"].map(_anode_cathode_from_bipolar_name)
    df["anode"] = [item[0] for item in parsed]
    df["cathode"] = [item[1] for item in parsed]
    return df[
        [
            "patient_id",
            "bipolar_channel",
            "anode",
            "cathode",
            "desikan_killiany_label",
            "tissue_class",
            "is_white_csf_ventricle_unknown",
            "is_bad_bipolar",
            "remove_as_bad_channel",
        ]
    ].copy()


def build_trial_plan(
    clean_manifest_path: Path = CLEAN_ROOT / "clean_contacts_manifest.csv",
    out_root: Path = BIPOLAR_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    assert_allowed_output_root(out_root)
    clean_manifest_path = Path(clean_manifest_path)
    if not clean_manifest_path.exists():
        raise SystemExit(f"Missing clean contacts manifest: {clean_manifest_path}")
    df = pd.read_csv(clean_manifest_path, encoding="utf-8-sig")
    required = {"patient", "trial", "role", "out_file", "n_kept"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Clean contacts manifest missing required columns: {', '.join(missing)}")

    kept = df[df["role"] == "kept_official"].copy()
    selected = kept[["patient", "trial", "out_file", "n_kept"]].drop_duplicates().copy()
    selected["patient"] = selected["patient"].astype(str).str.lower()
    selected["trial"] = selected["trial"].astype(int)

    patient_filter = _normalize_patients(patients)
    if patient_filter is not None:
        selected = selected[selected["patient"].isin(patient_filter)].copy()
    trial_filter = _normalize_trials(trials)
    if trial_filter is not None:
        selected = selected[selected["trial"].isin(trial_filter)].copy()

    duplicate = selected.duplicated(["patient", "trial"], keep=False)
    if duplicate.any():
        dupes = selected.loc[duplicate, ["patient", "trial"]].to_dict("records")
        raise SystemExit(f"Duplicate patient/trial rows in clean contacts manifest: {dupes}")

    records: list[dict[str, object]] = []
    out_root = Path(out_root)
    for _, row in selected.sort_values(["patient", "trial"]).iterrows():
        patient = str(row["patient"])
        trial = int(row["trial"])
        records.append(
            {
                "patient": patient,
                "trial": trial,
                "source_clean_fif": str(Path(str(row["out_file"]))),
                "out_file": str(out_root / patient / f"{patient}_trial{trial:02d}_bipolar_raw.fif"),
                "n_clean_contacts": int(row["n_kept"]),
            }
        )
    return pd.DataFrame.from_records(records)


def validate_trial_sources(trial_plan: pd.DataFrame) -> None:
    missing = [Path(str(path)) for path in trial_plan["source_clean_fif"] if not Path(str(path)).exists()]
    if missing:
        raise SystemExit("Missing clean-contact FIF source files:\n" + "\n".join(str(path) for path in missing))


def _read_fif_channel_names(fif_path: Path) -> list[str]:
    if not fif_path.exists():
        raise SystemExit(f"Missing clean-contact FIF source file: {fif_path}")
    raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected source sampling rate in {fif_path}: {sfreq}")
    return list(raw.ch_names)


def generate_adjacent_pairs(channels: Iterable[str]) -> list[tuple[str, str, str]]:
    by_shaft: dict[str, dict[int, str]] = {}
    for channel in channels:
        match = CONTACT_RE.fullmatch(str(channel))
        if match is None:
            continue
        shaft = match.group(1)
        number = int(match.group(2))
        by_shaft.setdefault(shaft, {})[number] = str(channel)

    pairs: list[tuple[str, str, str]] = []
    for shaft in sorted(by_shaft):
        numbered = by_shaft[shaft]
        for number in sorted(numbered):
            if number + 1 in numbered:
                anode = numbered[number]
                cathode = numbered[number + 1]
                pairs.append((f"{anode}-{cathode}", anode, cathode))
    return pairs


def _expected_pairs_from_anatomy(patient: str, anatomy: pd.DataFrame) -> set[str]:
    return set(anatomy[anatomy["patient_id"].astype(str).str.lower() == patient]["bipolar_channel"])


def _anatomy_lookup(anatomy: pd.DataFrame) -> pd.DataFrame:
    indexed = anatomy.assign(patient_id=anatomy["patient_id"].astype(str).str.lower()).set_index(
        ["patient_id", "bipolar_channel"]
    )
    if not indexed.index.is_unique:
        duplicated = indexed.index[indexed.index.duplicated()].tolist()
        raise SystemExit(f"Duplicate anatomy rows: {duplicated}")
    return indexed


def _rows_for_trial(
    trial_record: pd.Series,
    anatomy: pd.DataFrame,
    official_cache: dict[str, set[str]],
) -> list[dict[str, object]]:
    patient = str(trial_record["patient"])
    trial = int(trial_record["trial"])
    channels = _read_fif_channel_names(Path(str(trial_record["source_clean_fif"])))
    generated_pairs = generate_adjacent_pairs(channels)
    if not generated_pairs:
        raise SystemExit(f"{patient} trial {trial}: no adjacent bipolar pairs generated")

    official = official_cache.setdefault(patient, read_official_bipolar_names(patient))
    generated_names = {name for name, _, _ in generated_pairs}
    not_official = sorted(generated_names - official)
    if not_official:
        raise SystemExit(f"{patient} trial {trial}: generated bipolars absent from official Brainstorm set: {not_official}")

    anatomy_indexed = _anatomy_lookup(anatomy)
    expected_pairs = _expected_pairs_from_anatomy(patient, anatomy)
    missing_anatomy = sorted(generated_names - expected_pairs)
    if missing_anatomy:
        raise SystemExit(f"{patient} trial {trial}: generated bipolars absent from anatomy table: {missing_anatomy}")

    manual_exclusions = MANUAL_EXCLUDE_BIPOLARS.get(patient, set())
    rows: list[dict[str, object]] = []
    for bipolar_channel, anode, cathode in generated_pairs:
        anatomy_row = anatomy_indexed.loc[(patient, bipolar_channel)]
        manual_exclusion = bipolar_channel in manual_exclusions
        rows.append(
            {
                "patient": patient,
                "trial": trial,
                "bipolar_channel": bipolar_channel,
                "anode": anode,
                "cathode": cathode,
                "desikan_killiany_label": anatomy_row["desikan_killiany_label"],
                "tissue_class": anatomy_row["tissue_class"],
                "is_white_csf_ventricle_unknown": anatomy_row["is_white_csf_ventricle_unknown"],
                "is_bad_bipolar": anatomy_row["is_bad_bipolar"],
                "remove_as_bad_channel": anatomy_row["remove_as_bad_channel"],
                "source_clean_fif": str(trial_record["source_clean_fif"]),
                "out_file": str(trial_record["out_file"]),
                "montage_source": MONTAGE_SOURCE,
                "anatomy_source": ANATOMY_SOURCE,
                "official_crosscheck_source": OFFICIAL_CROSSCHECK_SOURCE,
                "manual_exclusion_applied": bool(manual_exclusion),
                "include_bipolar": not manual_exclusion,
                "n_bipolar_generated_before_manual": len(generated_pairs),
                "n_bipolar_after_manual": len(generated_pairs) - len(generated_names & manual_exclusions),
            }
        )
    return rows


def build_bipolar_manifest(trial_plan: pd.DataFrame, anatomy: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "patient",
        "trial",
        "bipolar_channel",
        "anode",
        "cathode",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
        "is_bad_bipolar",
        "remove_as_bad_channel",
        "source_clean_fif",
        "out_file",
        "montage_source",
        "anatomy_source",
        "official_crosscheck_source",
        "manual_exclusion_applied",
        "include_bipolar",
        "n_bipolar_generated_before_manual",
        "n_bipolar_after_manual",
    ]
    if trial_plan.empty:
        return pd.DataFrame(columns=columns)

    official_cache: dict[str, set[str]] = {}
    rows: list[dict[str, object]] = []
    for _, trial_record in trial_plan.sort_values(["patient", "trial"]).iterrows():
        rows.extend(_rows_for_trial(trial_record, anatomy, official_cache))
    return pd.DataFrame.from_records(rows, columns=columns)


def summarize_bipolar_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return pd.DataFrame(
            columns=["patient", "n_trials", "n_bipolar_generated_before_manual", "n_bipolar_after_manual"]
        )
    trial_counts = manifest[
        [
            "patient",
            "trial",
            "n_bipolar_generated_before_manual",
            "n_bipolar_after_manual",
        ]
    ].drop_duplicates()
    inconsistent: list[str] = []
    for patient, patient_counts in trial_counts.groupby("patient", sort=True):
        unique_counts = patient_counts[
            ["n_bipolar_generated_before_manual", "n_bipolar_after_manual"]
        ].drop_duplicates()
        if len(unique_counts) != 1:
            inconsistent.append(str(patient))
    if inconsistent:
        raise SystemExit(f"Per-trial bipolar counts are not consistent for patients: {', '.join(inconsistent)}")
    rows: list[dict[str, object]] = []
    for patient, patient_counts in trial_counts.groupby("patient", sort=True):
        counts = patient_counts.iloc[0]
        rows.append(
            {
                "patient": patient,
                "n_trials": int(patient_counts["trial"].nunique()),
                "n_bipolar_generated_before_manual": int(counts["n_bipolar_generated_before_manual"]),
                "n_bipolar_after_manual": int(counts["n_bipolar_after_manual"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def compute_bipolar_data(raw: mne.io.BaseRaw, pairs: list[tuple[str, str, str]]) -> np.ndarray:
    name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}
    missing = sorted({name for _, anode, cathode in pairs for name in (anode, cathode)} - set(name_to_index))
    if missing:
        raise SystemExit(f"Cannot compute bipolar data; missing monopolar channels: {missing}")
    data = raw.get_data()
    out = np.empty((len(pairs), raw.n_times), dtype=data.dtype)
    for idx, (_, anode, cathode) in enumerate(pairs):
        out[idx] = data[name_to_index[anode]] - data[name_to_index[cathode]]
    return out


def _manifest_for_trial(manifest: pd.DataFrame, patient: str, trial: int) -> pd.DataFrame:
    selected = manifest[
        (manifest["patient"] == patient)
        & (manifest["trial"].astype(int) == int(trial))
        & (manifest["include_bipolar"] == True)  # noqa: E712
    ]
    if selected.empty:
        raise SystemExit(f"Missing included bipolar manifest rows for {patient} trial {trial}")
    return selected


def build_one_trial(trial_record: pd.Series, manifest: pd.DataFrame, overwrite: bool) -> None:
    patient = str(trial_record["patient"])
    trial = int(trial_record["trial"])
    source = Path(str(trial_record["source_clean_fif"]))
    out_file = Path(str(trial_record["out_file"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite)
    if not source.exists():
        raise SystemExit(f"Missing clean-contact FIF source file: {source}")

    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    assert_zero_timebase(raw, source)
    sfreq = float(raw.info["sfreq"])
    if sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected source sampling rate in {source}: {sfreq}")

    trial_manifest = _manifest_for_trial(manifest, patient, trial)
    pairs = [
        (str(row["bipolar_channel"]), str(row["anode"]), str(row["cathode"]))
        for _, row in trial_manifest.iterrows()
    ]
    bipolar_data = compute_bipolar_data(raw, pairs)
    info = mne.create_info(
        [name for name, _, _ in pairs],
        sfreq=raw.info["sfreq"],
        ch_types="seeg",
    )
    bipolar_raw = mne.io.RawArray(bipolar_data, info, first_samp=0, verbose="ERROR")
    bipolar_raw.set_meas_date(raw.info.get("meas_date"))
    bipolar_raw.set_annotations(raw.annotations.copy())
    if bipolar_raw.n_times != raw.n_times:
        raise SystemExit(f"{patient} trial {trial}: bipolar n_times changed")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    bipolar_raw.save(out_file, overwrite=overwrite, verbose="ERROR")


def validate_outputs_available(trial_plan: pd.DataFrame, overwrite: bool) -> None:
    for out_file in trial_plan["out_file"].map(Path):
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite)


def write_manifests(manifest: pd.DataFrame, out_root: Path) -> None:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    included = manifest[manifest["include_bipolar"] == True].copy()  # noqa: E712
    for patient, patient_df in included.groupby("patient", sort=True):
        patient_dir = out_root / str(patient)
        patient_dir.mkdir(parents=True, exist_ok=True)
        patient_df.to_csv(patient_dir / f"{patient}_bipolar_manifest.csv", index=False, encoding="utf-8-sig")
    included.to_csv(out_root / "bipolar_manifest.csv", index=False, encoding="utf-8-sig")


def print_plan(trial_plan: pd.DataFrame, manifest: pd.DataFrame, out_root: Path, mode: str) -> None:
    cols = ["patient", "trial", "n_clean_contacts", "source_clean_fif", "out_file"]
    print(trial_plan[cols].to_string(index=False))
    print(f"\nmode={mode}  n_segments={len(trial_plan)}  out={out_root}")
    if not trial_plan.empty:
        print("\npatient segment counts:")
        print(trial_plan.groupby("patient").size().to_string())
    if not manifest.empty:
        print("\nbipolar accounting per trial:")
        print(summarize_bipolar_manifest(manifest).to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build adjacent bipolar raw FIF files from clean monopolar FIF files")
    ap.add_argument("--dry-run", action="store_true", help="print bipolar montage plan and write nothing")
    ap.add_argument("--patients", nargs="+", default=None, help="optional patient filter, e.g. --patients demo001 demo002")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional trial-id filter")
    ap.add_argument("--overwrite", action="store_true", help="replace existing bipolar FIF outputs")
    ap.add_argument("--out", default=None, help="override output directory under reviewed output roots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out) if args.out else BIPOLAR_ROOT
    assert_allowed_output_root(out_root)

    trial_plan = build_trial_plan(
        CLEAN_ROOT / "clean_contacts_manifest.csv",
        out_root,
        patients=args.patients,
        trials=args.trials,
    )
    if trial_plan.empty:
        raise SystemExit("No clean-contact FIF segments selected for bipolar montage generation.")
    validate_trial_sources(trial_plan)
    anatomy = load_anatomy_table(FINAL_AUDIT)
    manifest = build_bipolar_manifest(trial_plan, anatomy)
    mode = "dry-run" if args.dry_run else "full"
    print_plan(trial_plan, manifest, out_root, mode)

    if args.dry_run:
        return

    validate_outputs_available(trial_plan, overwrite=args.overwrite)
    for _, trial_record in trial_plan.iterrows():
        build_one_trial(trial_record, manifest, overwrite=args.overwrite)
    write_manifests(manifest, out_root)
    print(f"\nmanifest: {out_root / 'bipolar_manifest.csv'}")


if __name__ == "__main__":
    main()
