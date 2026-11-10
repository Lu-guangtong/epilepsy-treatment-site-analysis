"""Convert single-trial EDF files to raw FIF files.

This is a format-conversion step only. Non-continuous EDF files are already single
trial recordings, so this script saves the whole EDF as MNE raw FIF without
cropping, filtering, downsampling, bad-channel handling, bipolar montage, or
label access.

Run with the formal environment:
  python scripts/convert_single_trial_edf_to_fif.py --dry-run
  python scripts/convert_single_trial_edf_to_fif.py

Full conversion requires review of the processing plan before execution.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path
CONTINUOUS_PATIENT = str(private_value("continuous_recording", "patient"))
SINGLE_TRIAL_MANIFEST = str(private_value("continuous_recording", "single_trial_manifest_filename"))

import argparse
from pathlib import Path
from typing import Iterable

import mne
import pandas as pd


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
FINAL_AUDIT = source_root("data") / "annotations" / "clinical_annotations.xlsx"
SEGMENTS_ROOT = ROOT / "data_derived" / "segments"
SMOKE_ROOT = ROOT / "checks" / "single_trial_segment_smoke"
EXPECTED_SFREQ_HZ = 10000.0
CONVERSION_MODE = "full_edf_no_crop"


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(out_dir: Path) -> None:
    """Reject writes outside the reviewed formal output roots."""
    assert_output_path(out_dir)
    out_dir = _resolve(out_dir)
    data_root = _resolve(SEGMENTS_ROOT)
    smoke_root = _resolve(ROOT / "checks")
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == data_root
        or _is_relative_to(out_dir, data_root)
        or out_dir == smoke_root
        or _is_relative_to(out_dir, smoke_root)
    ):
        raise SystemExit(f"Refusing to write outside formal output roots: {out_dir}")


def assert_output_available(out_file: Path, overwrite: bool) -> None:
    if out_file.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite to replace it: {out_file}")


def _normalize_patients(patients: Iterable[str] | None) -> set[str] | None:
    if patients is None:
        return None
    return {p.strip().lower() for p in patients if p.strip()}


def _normalize_trials(trials: Iterable[int] | None) -> set[int] | None:
    if trials is None:
        return None
    return {int(t) for t in trials}


def _format_meas_date(raw: mne.io.BaseRaw) -> str:
    meas_date = raw.info.get("meas_date")
    return "" if meas_date is None else meas_date.isoformat()


def build_conversion_plan(
    audit_path: Path = FINAL_AUDIT,
    out_root: Path = SEGMENTS_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build the single-trial EDF-to-FIF conversion plan from the final audit workbook."""
    assert_allowed_output_root(out_root)
    audit_path = Path(audit_path)
    if not audit_path.exists():
        raise SystemExit(f"Audit workbook not found: {audit_path}")

    excel = pd.ExcelFile(audit_path)
    data_sheets = [name for name in excel.sheet_names if name.endswith("_data")]
    if not data_sheets:
        raise SystemExit(f"Final audit workbook has no patient *_data worksheets: {audit_path}")
    df = pd.concat(
        [pd.read_excel(excel, sheet_name=sheet) for sheet in data_sheets],
        ignore_index=True,
    )
    required = {
        "patient_id",
        "trial_id",
        "edf_path",
        "record_duration_sec",
        "sampling_rate_hz",
        "n_total_channels",
        "onset_sec_from_edf_start",
        "analysis_status",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Final audit workbook missing required *_data columns: {', '.join(missing)}")

    selected = df[
        (df["analysis_status"] == "included_for_computation")
        & (df["patient_id"].astype(str).str.lower() != CONTINUOUS_PATIENT)
    ].copy()

    patient_filter = _normalize_patients(patients)
    if patient_filter is not None:
        selected = selected[selected["patient_id"].astype(str).str.lower().isin(patient_filter)].copy()

    trial_filter = _normalize_trials(trials)
    if trial_filter is not None:
        selected = selected[selected["trial_id"].astype(int).isin(trial_filter)].copy()

    duplicate = selected.duplicated(["patient_id", "trial_id"], keep=False)
    if duplicate.any():
        dupes = selected.loc[duplicate, ["patient_id", "trial_id"]].to_dict("records")
        raise SystemExit(f"Duplicate patient/trial rows in audit selection: {dupes}")

    bad_sfreq = selected[selected["sampling_rate_hz"].astype(float) != EXPECTED_SFREQ_HZ]
    if not bad_sfreq.empty:
        rows = bad_sfreq[["patient_id", "trial_id", "sampling_rate_hz"]].to_dict("records")
        raise SystemExit(f"Unexpected sampling rate in audit selection: {rows}")

    records: list[dict[str, object]] = []
    out_root = Path(out_root)
    for _, row in selected.sort_values(["patient_id", "trial_id"]).iterrows():
        patient = str(row["patient_id"])
        trial = int(row["trial_id"])
        source = Path(str(row["edf_path"]))
        out_file = out_root / patient / f"{patient}_trial{trial:02d}_raw.fif"
        records.append(
            {
                "patient": patient,
                "trial": trial,
                "source_edf": str(source),
                "out_file": str(out_file),
                "edf_duration_sec": float(row["record_duration_sec"]),
                "sfreq_hz": float(row["sampling_rate_hz"]),
                "n_channels": int(row["n_total_channels"]),
                "n_samples": pd.NA,
                "meas_date": "",
                "onset_sec_from_raw_edf_start": float(row["onset_sec_from_edf_start"]),
                "conversion_mode": CONVERSION_MODE,
            }
        )

    return pd.DataFrame.from_records(records)


def validate_plan_sources(plan: pd.DataFrame) -> None:
    missing = [p for p in plan["source_edf"].map(Path) if not p.exists()]
    if missing:
        raise SystemExit("Missing EDF source files:\n" + "\n".join(str(p) for p in missing))


def convert_one(record: pd.Series, overwrite: bool) -> dict[str, object]:
    source = Path(str(record["source_edf"]))
    out_file = Path(str(record["out_file"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite)
    if not source.exists():
        raise SystemExit(f"EDF source file not found: {source}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw = mne.io.read_raw_edf(source, preload=True, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    if sfreq != EXPECTED_SFREQ_HZ:
        raise SystemExit(f"Unexpected sampling rate in {source}: {sfreq}")

    raw.save(out_file, overwrite=overwrite, verbose="ERROR")
    result = record.to_dict()
    result["sfreq_hz"] = sfreq
    result["n_channels"] = int(raw.info["nchan"])
    result["n_samples"] = int(raw.n_times)
    result["meas_date"] = _format_meas_date(raw)
    return result


def write_manifests(manifest: pd.DataFrame, out_root: Path) -> None:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    for patient, patient_df in manifest.groupby("patient", sort=True):
        patient_dir = out_root / str(patient)
        patient_dir.mkdir(parents=True, exist_ok=True)
        patient_df.to_csv(patient_dir / f"{patient}_convert_manifest.csv", index=False, encoding="utf-8-sig")
    manifest.to_csv(out_root / SINGLE_TRIAL_MANIFEST, index=False, encoding="utf-8-sig")


def print_plan(plan: pd.DataFrame, out_root: Path, mode: str) -> None:
    cols = ["patient", "trial", "edf_duration_sec", "sfreq_hz", "n_channels", "source_edf", "out_file"]
    print(plan[cols].to_string(index=False))
    print(f"\nmode={mode}  n_trials={len(plan)}  out={out_root}")
    if not plan.empty:
        print("\npatient counts:")
        print(plan.groupby("patient").size().to_string())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert single-trial single-trial EDF files to raw FIF")
    ap.add_argument("--dry-run", action="store_true", help="print conversion plan and write nothing")
    ap.add_argument("--smoke-test", action="store_true", help="convert only the first selected trial to checks")
    ap.add_argument("--patients", nargs="+", default=None, help="optional patient filter, e.g. --patients demo001 demo002")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional trial-id filter")
    ap.add_argument("--overwrite", action="store_true", help="replace existing FIF outputs")
    ap.add_argument("--out", default=None, help="override output directory under formal output roots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out) if args.out else (SMOKE_ROOT if args.smoke_test else SEGMENTS_ROOT)
    assert_allowed_output_root(out_root)

    plan = build_conversion_plan(FINAL_AUDIT, out_root, patients=args.patients, trials=args.trials)
    if args.smoke_test:
        plan = plan.head(1).reset_index(drop=True)
    if plan.empty:
        raise SystemExit("No EDF rows selected for conversion.")

    validate_plan_sources(plan)
    mode = "dry-run" if args.dry_run else ("smoke-test" if args.smoke_test else "full")
    print_plan(plan, out_root, mode)

    if args.dry_run:
        return

    converted = [convert_one(row, overwrite=args.overwrite) for _, row in plan.iterrows()]
    manifest = pd.DataFrame.from_records(converted)
    write_manifests(manifest, out_root)
    print(f"\nmanifest: {out_root / SINGLE_TRIAL_MANIFEST}")


if __name__ == "__main__":
    main()
