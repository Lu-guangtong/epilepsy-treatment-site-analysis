"""Normalize derived continuous-recording segments to per-trial time axes."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path
CONTINUOUS_PATIENT = str(private_value("continuous_recording", "patient"))
SINGLE_TRIAL_MANIFEST = str(private_value("continuous_recording", "single_trial_manifest_filename"))

import argparse
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
SEGMENTS_ROOT = ROOT / "data_derived" / "segments"
RESAMPLED_ROOT = ROOT / "data_derived" / "resampled"
CLEAN_ROOT = ROOT / "data_derived" / "clean_contacts"
BIPOLAR_ROOT = ROOT / "data_derived" / "bipolar"
TIMEBASE_FIX_ROOT = ROOT / "data_derived" / "timebase_fix"
CHECK_ROOT = ROOT / "checks"
FINAL_AUDIT = source_root("data") / "annotations" / "clinical_annotations.xlsx"
PATIENT = private_value("continuous_recording", "patient")
TRIALS = private_value("continuous_recording", "trials")
SEGMENT_ONSET_SEC = 60.0


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
    timebase_root = _resolve(TIMEBASE_FIX_ROOT)
    data_derived_root = _resolve(ROOT / "data_derived")
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == timebase_root
        or _is_relative_to(out_dir, timebase_root)
        or out_dir == data_derived_root
        or _is_relative_to(out_dir, data_derived_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def assert_output_available(out_file: Path, overwrite: bool) -> None:
    if out_file.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite to replace it: {out_file}")


def _format_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_datetime(value: object) -> datetime:
    if pd.isna(value):
        raise SystemExit("Missing datetime value in continuous crop manifest source fields")
    if isinstance(value, pd.Timestamp):
        dt = value.to_pydatetime()
    elif isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _load_continuous_audit_timing() -> pd.DataFrame:
    audit = FINAL_AUDIT
    if not audit.exists():
        raise SystemExit(f"Missing final audit workbook: {audit}")
    df = pd.read_excel(audit, sheet_name=f"{PATIENT}_data")
    required = {
        "patient_id",
        "trial_id",
        "t0_time",
        "onset_sec_from_edf_start",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"final audit {PATIENT}_data missing required columns: {', '.join(missing)}")
    selected = df[(df["patient_id"] == PATIENT) & (df["trial_id"].isin(TRIALS))].copy()
    if len(selected) != len(TRIALS):
        raise SystemExit(f"Expected {len(TRIALS)} continuous audit rows, found {len(selected)}")
    selected["trial"] = selected["trial_id"].astype(int)
    selected["raw_edf_t0_time"] = selected["t0_time"].map(_parse_datetime)
    selected["onset_sec_from_raw_edf_start"] = selected["onset_sec_from_edf_start"].astype(float)
    return selected[["trial", "raw_edf_t0_time", "onset_sec_from_raw_edf_start"]]


def build_continuous_crop_manifest_updates(manifest_path: Path = SEGMENTS_ROOT / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv") -> pd.DataFrame:
    """Return the continuous crop manifest with explicit segment and long-EDF time fields."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"Missing continuous crop manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"patient", "trial", "onset_sec", "actual_pre_s", "actual_post_s"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"continuous crop manifest missing required columns: {', '.join(missing)}")
    if "crop_start_sec_from_raw_edf_start" not in df.columns and "crop_start_sec" not in df.columns:
        raise SystemExit("continuous crop manifest missing crop start field")
    if "crop_end_sec_from_raw_edf_start" not in df.columns and "crop_end_sec" not in df.columns:
        raise SystemExit("continuous crop manifest missing crop end field")
    df = df.copy()
    df["trial"] = df["trial"].astype(int)
    selected = df[(df["patient"].astype(str).str.lower() == PATIENT) & (df["trial"].isin(TRIALS))].copy()
    if len(selected) != len(TRIALS):
        raise SystemExit(f"Expected {len(TRIALS)} continuous crop rows, found {len(selected)}")

    audit = _load_continuous_audit_timing()
    audit_timing = {
        int(row["trial"]): {
            "raw_edf_t0_time": row["raw_edf_t0_time"],
            "onset_sec_from_raw_edf_start": float(row["onset_sec_from_raw_edf_start"]),
        }
        for _, row in audit.iterrows()
    }
    merged = selected.copy()
    merged["raw_edf_t0_time"] = [audit_timing[int(trial)]["raw_edf_t0_time"] for trial in merged["trial"]]
    merged["onset_sec_from_raw_edf_start"] = [
        audit_timing[int(trial)]["onset_sec_from_raw_edf_start"] for trial in merged["trial"]
    ]
    if merged["raw_edf_t0_time"].isna().any():
        raise SystemExit("Missing continuous audit timing after merging crop manifest")

    if "crop_start_sec_from_raw_edf_start" not in merged.columns:
        merged = merged.rename(columns={"crop_start_sec": "crop_start_sec_from_raw_edf_start"})
    if "crop_end_sec_from_raw_edf_start" not in merged.columns:
        merged = merged.rename(columns={"crop_end_sec": "crop_end_sec_from_raw_edf_start"})
    merged["onset_sec_from_raw_edf_start"] = merged["onset_sec_from_raw_edf_start"].astype(float)
    merged["crop_start_sec_from_raw_edf_start"] = merged["crop_start_sec_from_raw_edf_start"].astype(float)
    merged["crop_end_sec_from_raw_edf_start"] = merged["crop_end_sec_from_raw_edf_start"].astype(float)
    merged["onset_sec"] = merged["onset_sec_from_raw_edf_start"] - merged["crop_start_sec_from_raw_edf_start"]
    if not np.allclose(merged["onset_sec"].to_numpy(float), SEGMENT_ONSET_SEC):
        rows = merged[["trial", "onset_sec"]].to_dict("records")
        raise SystemExit(f"Unexpected continuous segment-relative onset seconds: {rows}")

    merged["raw_edf_t0_time"] = merged["raw_edf_t0_time"].map(_format_iso)
    merged["segment_t0_time"] = [
        _format_iso(_parse_datetime(raw_t0) + timedelta(seconds=float(crop_start)))
        for raw_t0, crop_start in zip(
            merged["raw_edf_t0_time"],
            merged["crop_start_sec_from_raw_edf_start"],
            strict=True,
        )
    ]

    front = [
        "patient",
        "trial",
        "onset_sec",
        "onset_sec_from_raw_edf_start",
        "crop_start_sec_from_raw_edf_start",
        "crop_end_sec_from_raw_edf_start",
        "segment_t0_time",
        "raw_edf_t0_time",
    ]
    rest = [col for col in merged.columns if col not in front]
    return merged[front + rest].sort_values("trial").reset_index(drop=True)


def _continuous_timing_by_trial(crop_updates: pd.DataFrame) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for _, row in crop_updates.iterrows():
        trial = int(row["trial"])
        rows[trial] = {
            "onset_sec": float(row["onset_sec"]),
            "onset_sec_from_raw_edf_start": float(row["onset_sec_from_raw_edf_start"]),
            "crop_start_sec_from_raw_edf_start": float(row["crop_start_sec_from_raw_edf_start"]),
            "crop_end_sec_from_raw_edf_start": float(row["crop_end_sec_from_raw_edf_start"]),
            "segment_t0_time": str(row["segment_t0_time"]),
            "raw_edf_t0_time": str(row["raw_edf_t0_time"]),
        }
    return rows


def build_resample_manifest_updates(
    manifest_path: Path = RESAMPLED_ROOT / "resample_manifest.csv",
    crop_manifest_path: Path = SEGMENTS_ROOT / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv",
) -> pd.DataFrame:
    """Return resample manifest with continuous onset_sec changed to segment-relative seconds."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise SystemExit(f"Missing resample manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"patient", "trial", "onset_sec"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"resample manifest missing required columns: {', '.join(missing)}")

    updated = df.copy()
    updated["trial"] = updated["trial"].astype(int)
    timing = _continuous_timing_by_trial(build_continuous_crop_manifest_updates(crop_manifest_path))
    for trial, fields in timing.items():
        mask = (updated["patient"].astype(str).str.lower() == PATIENT) & (updated["trial"] == trial)
        if int(mask.sum()) != 1:
            raise SystemExit(f"Expected one continuous trial {trial} row in resample manifest, found {int(mask.sum())}")
        for field, value in fields.items():
            updated.loc[mask, field] = value
    return updated


def _stage_specs(
    segments_root: Path,
    resampled_root: Path,
    clean_root: Path,
    bipolar_root: Path,
) -> list[tuple[str, Path, str]]:
    return [
        ("segments", Path(segments_root) / CONTINUOUS_PATIENT, f"{CONTINUOUS_PATIENT}_trial{{trial:02d}}_raw.fif"),
        ("resampled", Path(resampled_root) / CONTINUOUS_PATIENT, f"{CONTINUOUS_PATIENT}_trial{{trial:02d}}_2k_raw.fif"),
        ("clean_contacts", Path(clean_root) / CONTINUOUS_PATIENT, f"{CONTINUOUS_PATIENT}_trial{{trial:02d}}_clean_raw.fif"),
        ("bipolar", Path(bipolar_root) / CONTINUOUS_PATIENT, f"{CONTINUOUS_PATIENT}_trial{{trial:02d}}_bipolar_raw.fif"),
    ]


def build_timebase_plan(
    segments_root: Path = SEGMENTS_ROOT,
    resampled_root: Path = RESAMPLED_ROOT,
    clean_root: Path = CLEAN_ROOT,
    bipolar_root: Path = BIPOLAR_ROOT,
    out_root: Path = TIMEBASE_FIX_ROOT,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build the configured trial-by-stage plan for continuous FIF timebase normalization."""
    assert_allowed_output_root(out_root)
    selected_trials = TRIALS if trials is None else [int(trial) for trial in trials]
    timing = _continuous_timing_by_trial(build_continuous_crop_manifest_updates(Path(segments_root) / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv"))
    missing_trials = sorted(set(selected_trials) - set(timing))
    if missing_trials:
        raise SystemExit(f"Unknown continuous trials for timebase normalization: {missing_trials}")

    rows: list[dict[str, object]] = []
    for stage, folder, pattern in _stage_specs(segments_root, resampled_root, clean_root, bipolar_root):
        for trial in selected_trials:
            source = folder / pattern.format(trial=trial)
            if not source.exists():
                raise SystemExit(f"Missing continuous {stage} FIF: {source}")
            rows.append(
                {
                    "patient": PATIENT,
                    "trial": trial,
                    "stage": stage,
                    "fif_file": str(source),
                    "segment_t0_time": timing[trial]["segment_t0_time"],
                    "raw_edf_t0_time": timing[trial]["raw_edf_t0_time"],
                    "onset_sec": timing[trial]["onset_sec"],
                    "onset_sec_from_raw_edf_start": timing[trial]["onset_sec_from_raw_edf_start"],
                    "crop_start_sec_from_raw_edf_start": timing[trial]["crop_start_sec_from_raw_edf_start"],
                    "crop_end_sec_from_raw_edf_start": timing[trial]["crop_end_sec_from_raw_edf_start"],
                }
            )
    return pd.DataFrame.from_records(rows).sort_values(["stage", "trial"]).reset_index(drop=True)


def normalize_raw_to_segment_timebase(raw: mne.io.BaseRaw, segment_t0_time: datetime) -> mne.io.BaseRaw:
    """Return a Raw object with the same data/channels but first_samp reset to zero."""
    data = raw._data  # noqa: SLF001 - avoid an extra full copy of large preloaded FIF data.
    info = raw.info.copy()
    fixed = mne.io.RawArray(data, info, first_samp=0, copy="info", verbose="ERROR")
    fixed.set_meas_date(segment_t0_time)
    if len(raw.annotations) == 0:
        fixed.set_annotations(mne.Annotations(onset=[], duration=[], description=[], orig_time=segment_t0_time))
    else:
        original_orig_time = raw.annotations.orig_time or raw.info.get("meas_date")
        if original_orig_time is None:
            raise SystemExit("Cannot safely remap non-empty annotations without an absolute orig_time")
        original_orig_time = _parse_datetime(original_orig_time)
        new_onsets = [
            (original_orig_time + timedelta(seconds=float(onset)) - segment_t0_time).total_seconds()
            for onset in raw.annotations.onset
        ]
        fixed.set_annotations(
            mne.Annotations(
                onset=new_onsets,
                duration=raw.annotations.duration.copy(),
                description=raw.annotations.description.copy(),
                orig_time=segment_t0_time,
            )
        )
    return fixed


def _raw_summary(raw: mne.io.BaseRaw, prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_first_samp": int(raw.first_samp),
        f"{prefix}_first_time": float(raw.first_time),
        f"{prefix}_meas_date": _format_iso(raw.info["meas_date"]) if raw.info.get("meas_date") else "",
        f"{prefix}_sfreq_hz": float(raw.info["sfreq"]),
        f"{prefix}_n_channels": int(raw.info["nchan"]),
        f"{prefix}_n_times": int(raw.n_times),
    }


def _validate_fixed_raw(original: mne.io.BaseRaw, fixed: mne.io.BaseRaw, fif_file: Path) -> None:
    if fixed.first_samp != 0 or fixed.first_time != 0.0:
        raise SystemExit(f"{fif_file}: fixed raw does not start at sample/time zero")
    if original.ch_names != fixed.ch_names:
        raise SystemExit(f"{fif_file}: channel names changed during timebase normalization")
    if original.get_channel_types() != fixed.get_channel_types():
        raise SystemExit(f"{fif_file}: channel types changed during timebase normalization")
    if original.info["sfreq"] != fixed.info["sfreq"]:
        raise SystemExit(f"{fif_file}: sampling rate changed during timebase normalization")
    if original.n_times != fixed.n_times:
        raise SystemExit(f"{fif_file}: n_times changed during timebase normalization")
    np.testing.assert_allclose(original._data, fixed._data, rtol=0.0, atol=0.0)  # noqa: SLF001


def normalize_one_file(row: pd.Series, overwrite: bool) -> dict[str, object]:
    fif_file = Path(str(row["fif_file"]))
    assert_output_available(fif_file, overwrite=overwrite)
    segment_t0 = _parse_datetime(row["segment_t0_time"])
    raw = mne.io.read_raw_fif(fif_file, preload=True, verbose="ERROR")
    before = _raw_summary(raw, "before")
    fixed = normalize_raw_to_segment_timebase(raw, segment_t0)
    _validate_fixed_raw(raw, fixed, fif_file)
    after = _raw_summary(fixed, "after")

    tmp_file = fif_file.with_name(f"{fif_file.stem}.timebase_tmp{fif_file.suffix}")
    if tmp_file.exists():
        tmp_file.unlink()
    try:
        fixed.save(tmp_file, overwrite=True, verbose="ERROR")
        reread = mne.io.read_raw_fif(tmp_file, preload=False, verbose="ERROR")
        if reread.first_samp != 0 or reread.first_time != 0.0:
            raise SystemExit(f"{tmp_file}: saved fixed raw did not preserve zero timebase")
        os.replace(tmp_file, fif_file)
    finally:
        if tmp_file.exists():
            tmp_file.unlink()

    return {
        **row.to_dict(),
        **before,
        **after,
        "timebase_fix_status": "fixed",
    }


def inspect_one_file(row: pd.Series) -> dict[str, object]:
    fif_file = Path(str(row["fif_file"]))
    raw = mne.io.read_raw_fif(fif_file, preload=False, verbose="ERROR")
    return {
        **row.to_dict(),
        **_raw_summary(raw, "before"),
        "after_first_samp": 0,
        "after_first_time": 0.0,
        "after_meas_date": str(row["segment_t0_time"]),
        "after_sfreq_hz": float(raw.info["sfreq"]),
        "after_n_channels": int(raw.info["nchan"]),
        "after_n_times": int(raw.n_times),
        "timebase_fix_status": "planned",
    }


def write_crop_manifest(updates: pd.DataFrame, manifest_path: Path) -> None:
    manifest_path = Path(manifest_path)
    assert_allowed_output_root(manifest_path.parent)
    updates.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def write_resample_manifest(updates: pd.DataFrame, manifest_path: Path) -> None:
    manifest_path = Path(manifest_path)
    assert_allowed_output_root(manifest_path.parent)
    updates.to_csv(manifest_path, index=False, encoding="utf-8-sig")


def write_timebase_fix_manifest(records: list[dict[str, object]], out_root: Path) -> Path:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / f"{CONTINUOUS_PATIENT}_timebase_fix_manifest.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_plan(plan: pd.DataFrame, mode: str, out_root: Path) -> None:
    cols = ["stage", "trial", "fif_file", "onset_sec", "segment_t0_time"]
    print(plan[cols].to_string(index=False))
    print(f"\nmode={mode}  n_files={len(plan)}  out={out_root}")
    print("\nstage counts:")
    print(plan.groupby("stage").size().to_string())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Normalize continuous FIF first_samp/timebase to independent trial segments")
    ap.add_argument("--dry-run", action="store_true", help="print plan and inspect headers without writing")
    ap.add_argument("--in-place", action="store_true", help="rewrite formal continuous FIFs and manifests in place")
    ap.add_argument("--overwrite", action="store_true", help="required with --in-place for formal FIF replacement")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional continuous trial filter, e.g. --trials 1")
    ap.add_argument("--out", default=None, help="audit output directory under data_derived/timebase_fix or checks")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out) if args.out else TIMEBASE_FIX_ROOT
    assert_allowed_output_root(out_root)
    if args.in_place and not args.overwrite:
        raise SystemExit("--in-place requires --overwrite for explicit formal data replacement")
    if args.dry_run and args.in_place:
        raise SystemExit("Choose either --dry-run or --in-place, not both")
    mode = "in-place" if args.in_place else "dry-run"

    crop_updates = build_continuous_crop_manifest_updates(SEGMENTS_ROOT / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv")
    resample_updates = build_resample_manifest_updates(
        RESAMPLED_ROOT / "resample_manifest.csv",
        SEGMENTS_ROOT / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv",
    )
    plan = build_timebase_plan(
        segments_root=SEGMENTS_ROOT,
        resampled_root=RESAMPLED_ROOT,
        clean_root=CLEAN_ROOT,
        bipolar_root=BIPOLAR_ROOT,
        out_root=out_root,
        trials=args.trials,
    )
    if plan.empty:
        raise SystemExit("No continuous FIF files selected for timebase normalization.")
    print_plan(plan, mode, out_root)

    records: list[dict[str, object]] = []
    if args.in_place:
        for _, row in plan.iterrows():
            records.append(normalize_one_file(row, overwrite=args.overwrite))
        write_crop_manifest(crop_updates, SEGMENTS_ROOT / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv")
        write_resample_manifest(resample_updates, RESAMPLED_ROOT / "resample_manifest.csv")
    else:
        for _, row in plan.iterrows():
            records.append(inspect_one_file(row))

    manifest_path = write_timebase_fix_manifest(records, out_root)
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
