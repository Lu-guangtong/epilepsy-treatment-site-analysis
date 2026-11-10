"""Resample 10 kHz raw FIF segments to 2 kHz raw FIF segments.

This is an atomic anti-aliasing + resampling step only. It does not notch,
band-pass, remove channels, apply bad-channel rules, build a bipolar montage,
or read clinician ablation labels.

Inspect the processing plan with --dry-run before execution:
  python scripts/resample_segments_10k_to_2k.py --dry-run

Full resampling requires review of the processing plan before execution.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path
CONTINUOUS_PATIENT = str(private_value("continuous_recording", "patient"))
SINGLE_TRIAL_MANIFEST = str(private_value("continuous_recording", "single_trial_manifest_filename"))

import argparse
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
SEGMENTS_ROOT = ROOT / "data_derived" / "segments"
RESAMPLED_ROOT = ROOT / "data_derived" / "resampled"
SMOKE_ROOT = ROOT / "checks" / "resample_10k_to_2k_smoke"
SOURCE_SFREQ_HZ = 10000.0
TARGET_SFREQ_HZ = 2000.0
MAX_DURATION_DIFF_SAMPLES = 2
MAX_DURATION_DIFF_SEC = MAX_DURATION_DIFF_SAMPLES / TARGET_SFREQ_HZ
FLOAT_TOLERANCE_SEC = 1e-9
RESAMPLE_METHOD = "polyphase"
RESAMPLE_WINDOW = ("kaiser", 5.0)
RESAMPLE_WINDOW_LABEL = "kaiser_5.0"
RESAMPLE_PAD = "reflect"
CONVERSION_MODE = "anti_alias_resample_only"


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
    resampled_root = _resolve(RESAMPLED_ROOT)
    smoke_root = _resolve(ROOT / "checks")
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == resampled_root
        or _is_relative_to(out_dir, resampled_root)
        or out_dir == smoke_root
        or _is_relative_to(out_dir, smoke_root)
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
            "run normalize_segment_timebase.py before resampling"
        )


def _normalize_patients(patients: Iterable[str] | None) -> set[str] | None:
    if patients is None:
        return None
    return {p.strip().lower() for p in patients if p.strip()}


def _normalize_trials(trials: Iterable[int] | None) -> set[int] | None:
    if trials is None:
        return None
    return {int(t) for t in trials}


def _duration_sec(n_times: int, sfreq: float) -> float:
    if n_times <= 0:
        return 0.0
    return (int(n_times) - 1) / float(sfreq)


def _continuous_manifest_value(row: pd.Series, new_name: str, old_name: str) -> float:
    if new_name in row.index and not pd.isna(row[new_name]):
        return float(row[new_name])
    return float(row[old_name])


def _continuous_onset_fields(row: pd.Series) -> dict[str, object]:
    if "onset_sec_from_raw_edf_start" in row.index and not pd.isna(row["onset_sec_from_raw_edf_start"]):
        onset_abs = float(row["onset_sec_from_raw_edf_start"])
        onset_segment = float(row["onset_sec"])
    else:
        onset_abs = float(row["onset_sec"])
        crop_start = _continuous_manifest_value(row, "crop_start_sec_from_raw_edf_start", "crop_start_sec")
        onset_segment = onset_abs - crop_start
    return {
        "onset_sec": onset_segment,
        "onset_sec_from_raw_edf_start": onset_abs,
    }


def _load_continuous_rows(segments_root: Path, out_root: Path) -> list[dict[str, object]]:
    manifest_path = segments_root / CONTINUOUS_PATIENT / f"{CONTINUOUS_PATIENT}_crop_manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing continuous crop manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {"patient", "trial", "out_file", "onset_sec", "sfreq_hz", "n_channels", "n_samples"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"continuous crop manifest missing required columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        patient = str(row["patient"])
        trial = int(row["trial"])
        onset_fields = _continuous_onset_fields(row)
        rows.append(
            {
                "patient": patient,
                "trial": trial,
                "source_10k_fif": str(Path(str(row["out_file"]))),
                "out_2k_fif": str(out_root / patient / f"{patient}_trial{trial:02d}_2k_raw.fif"),
                "source_sfreq_hz": float(row["sfreq_hz"]),
                "target_sfreq_hz": TARGET_SFREQ_HZ,
                "n_channels": int(row["n_channels"]),
                "source_n_samples": int(row["n_samples"]),
                "resampled_n_samples": pd.NA,
                "source_duration_sec": _duration_sec(int(row["n_samples"]), float(row["sfreq_hz"])),
                "resampled_duration_sec": pd.NA,
                "duration_diff_sec": pd.NA,
                "onset_sec": float(onset_fields["onset_sec"]),
                "onset_sec_from_raw_edf_start": float(onset_fields["onset_sec_from_raw_edf_start"]),
                "crop_start_sec_from_raw_edf_start": _continuous_manifest_value(
                    row,
                    "crop_start_sec_from_raw_edf_start",
                    "crop_start_sec",
                ),
                "crop_end_sec_from_raw_edf_start": _continuous_manifest_value(
                    row,
                    "crop_end_sec_from_raw_edf_start",
                    "crop_end_sec",
                ),
                "segment_t0_time": row["segment_t0_time"] if "segment_t0_time" in row.index else "",
                "raw_edf_t0_time": row["raw_edf_t0_time"] if "raw_edf_t0_time" in row.index else "",
            }
        )
    return rows


def _load_single_trial_rows(segments_root: Path, out_root: Path) -> list[dict[str, object]]:
    manifest_path = segments_root / SINGLE_TRIAL_MANIFEST
    if not manifest_path.exists():
        raise SystemExit(f"Missing single-trial convert manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, encoding="utf-8-sig")
    required = {
        "patient",
        "trial",
        "out_file",
        "sfreq_hz",
        "n_channels",
        "n_samples",
        "onset_sec_from_raw_edf_start",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"single-trial convert manifest missing required columns: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        patient = str(row["patient"])
        trial = int(row["trial"])
        rows.append(
            {
                "patient": patient,
                "trial": trial,
                "source_10k_fif": str(Path(str(row["out_file"]))),
                "out_2k_fif": str(out_root / patient / f"{patient}_trial{trial:02d}_2k_raw.fif"),
                "source_sfreq_hz": float(row["sfreq_hz"]),
                "target_sfreq_hz": TARGET_SFREQ_HZ,
                "n_channels": int(row["n_channels"]),
                "source_n_samples": int(row["n_samples"]),
                "resampled_n_samples": pd.NA,
                "source_duration_sec": _duration_sec(int(row["n_samples"]), float(row["sfreq_hz"])),
                "resampled_duration_sec": pd.NA,
                "duration_diff_sec": pd.NA,
                "onset_sec": float(row["onset_sec_from_raw_edf_start"]),
                "onset_sec_from_raw_edf_start": float(row["onset_sec_from_raw_edf_start"]),
                "crop_start_sec_from_raw_edf_start": pd.NA,
                "crop_end_sec_from_raw_edf_start": pd.NA,
                "segment_t0_time": "",
                "raw_edf_t0_time": "",
            }
        )
    return rows


def _add_static_metadata(rows: list[dict[str, object]]) -> pd.DataFrame:
    for row in rows:
        row["mne_version"] = mne.__version__
        row["resample_method"] = RESAMPLE_METHOD
        row["resample_window"] = RESAMPLE_WINDOW_LABEL
        row["resample_pad"] = RESAMPLE_PAD
        row["conversion_mode"] = CONVERSION_MODE
        row["qc_status"] = "planned"
        row["qc_notes"] = ""
    return pd.DataFrame.from_records(rows)


def build_resample_plan(
    segments_root: Path = SEGMENTS_ROOT,
    out_root: Path = RESAMPLED_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build a 61-row 10 kHz FIF to 2 kHz FIF resampling plan."""
    segments_root = Path(segments_root)
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)

    rows = _load_continuous_rows(segments_root, out_root) + _load_single_trial_rows(segments_root, out_root)
    plan = _add_static_metadata(rows)
    plan = plan.sort_values(["patient", "trial"]).reset_index(drop=True)

    patient_filter = _normalize_patients(patients)
    if patient_filter is not None:
        plan = plan[plan["patient"].astype(str).str.lower().isin(patient_filter)].copy()

    trial_filter = _normalize_trials(trials)
    if trial_filter is not None:
        plan = plan[plan["trial"].astype(int).isin(trial_filter)].copy()

    duplicate = plan.duplicated(["patient", "trial"], keep=False)
    if duplicate.any():
        dupes = plan.loc[duplicate, ["patient", "trial"]].to_dict("records")
        raise SystemExit(f"Duplicate patient/trial rows in resample plan: {dupes}")

    bad_sfreq = plan[plan["source_sfreq_hz"].astype(float) != SOURCE_SFREQ_HZ]
    if not bad_sfreq.empty:
        rows_bad = bad_sfreq[["patient", "trial", "source_sfreq_hz"]].to_dict("records")
        raise SystemExit(f"Unexpected source sampling rate in plan: {rows_bad}")

    return plan.reset_index(drop=True)


def validate_plan_sources(plan: pd.DataFrame) -> None:
    missing = [p for p in plan["source_10k_fif"].map(Path) if not p.exists()]
    if missing:
        raise SystemExit("Missing 10 kHz FIF source files:\n" + "\n".join(str(p) for p in missing))


def _assert_raw_has_no_nan_inf(raw: mne.io.BaseRaw) -> None:
    for start in range(0, raw.n_times, int(raw.info["sfreq"]) * 10):
        stop = min(raw.n_times, start + int(raw.info["sfreq"]) * 10)
        block = raw.get_data(start=start, stop=stop)
        if not np.isfinite(block).all():
            raise SystemExit(f"Found NaN/Inf in resampled data near samples {start}:{stop}")


def _qc_status_and_notes(source_duration: float, resampled_duration: float) -> tuple[str, str, float]:
    diff = abs(float(resampled_duration) - float(source_duration))
    if diff <= MAX_DURATION_DIFF_SEC + FLOAT_TOLERANCE_SEC:
        return "pass", "", diff
    return (
        "fail",
        f"duration_diff_sec exceeds {MAX_DURATION_DIFF_SAMPLES} target samples "
        f"({MAX_DURATION_DIFF_SEC} sec): {diff}",
        diff,
    )


def _qc_plot_source_summary(raw: mne.io.BaseRaw) -> dict[str, np.ndarray]:
    pick = 0
    duration = min(5.0, raw.n_times / raw.info["sfreq"])
    data = raw.get_data(picks=[pick], start=0, stop=max(1, int(duration * raw.info["sfreq"])))[0]
    freqs, psd = welch(data, fs=float(raw.info["sfreq"]), nperseg=min(data.size, int(raw.info["sfreq"])))
    return {
        "time": np.arange(data.size) / raw.info["sfreq"],
        "data": data,
        "freqs": freqs,
        "psd": psd,
    }


def save_qc_plot(source_summary: dict[str, np.ndarray], resampled_raw: mne.io.BaseRaw, out_file: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qc_path = out_file.with_suffix(".qc.png")
    pick = 0
    duration = min(5.0, resampled_raw.n_times / resampled_raw.info["sfreq"])
    source_data = source_summary["data"]
    source_t = source_summary["time"]
    source_freqs = source_summary["freqs"]
    source_psd = source_summary["psd"]
    resampled_data = resampled_raw.get_data(picks=[pick], start=0, stop=max(1, int(duration * resampled_raw.info["sfreq"])))[0]
    resampled_t = np.arange(resampled_data.size) / resampled_raw.info["sfreq"]
    resampled_freqs, resampled_psd = welch(
        resampled_data,
        fs=float(resampled_raw.info["sfreq"]),
        nperseg=min(resampled_data.size, int(resampled_raw.info["sfreq"])),
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].semilogy(source_freqs, source_psd, lw=0.7, label="10 kHz")
    axes[0].semilogy(resampled_freqs, resampled_psd, lw=0.7, label="2 kHz")
    axes[0].set_xlim(0, 1500)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title(f"PSD before/after resample: {out_file.stem}")
    axes[0].legend(loc="upper right")
    axes[1].plot(source_t, source_data / (np.nanstd(source_data) + 1e-12), lw=0.5, label="10 kHz")
    axes[1].plot(resampled_t, resampled_data / (np.nanstd(resampled_data) + 1e-12), lw=0.5, label="2 kHz")
    axes[1].set_title("Waveform overlay, first channel, first 5 s")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(loc="upper right")
    fig.savefig(qc_path, dpi=120)
    plt.close(fig)


def resample_one(record: pd.Series, overwrite: bool, plot_qc: bool) -> dict[str, object]:
    source = Path(str(record["source_10k_fif"]))
    out_file = Path(str(record["out_2k_fif"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite)
    if not source.exists():
        raise SystemExit(f"10 kHz source FIF not found: {source}")

    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    assert_zero_timebase(raw, source)
    source_sfreq = float(raw.info["sfreq"])
    if source_sfreq != SOURCE_SFREQ_HZ:
        raise SystemExit(f"Unexpected source sampling rate in {source}: {source_sfreq}")
    source_n_channels = int(raw.info["nchan"])
    source_n_samples = int(raw.n_times)
    source_duration = _duration_sec(source_n_samples, source_sfreq)
    source_summary = _qc_plot_source_summary(raw) if plot_qc else None

    raw.resample(
        TARGET_SFREQ_HZ,
        method=RESAMPLE_METHOD,
        window=RESAMPLE_WINDOW,
        pad=RESAMPLE_PAD,
        verbose="ERROR",
    )
    target_sfreq = float(raw.info["sfreq"])
    if target_sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected target sampling rate in {out_file}: {target_sfreq}")
    if int(raw.info["nchan"]) != source_n_channels:
        raise SystemExit(f"Channel count changed for {out_file}")
    _assert_raw_has_no_nan_inf(raw)

    resampled_n_samples = int(raw.n_times)
    resampled_duration = _duration_sec(resampled_n_samples, target_sfreq)
    qc_status, qc_notes, duration_diff = _qc_status_and_notes(source_duration, resampled_duration)
    if qc_status != "pass":
        raise SystemExit(f"QC failed for {out_file}: {qc_notes}")

    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_file, overwrite=overwrite, verbose="ERROR")
    if plot_qc:
        if source_summary is None:
            raise SystemExit("Internal error: missing source QC summary")
        save_qc_plot(source_summary, raw, out_file)

    result = record.to_dict()
    result["source_sfreq_hz"] = source_sfreq
    result["target_sfreq_hz"] = target_sfreq
    result["n_channels"] = source_n_channels
    result["source_n_samples"] = source_n_samples
    result["resampled_n_samples"] = resampled_n_samples
    result["source_duration_sec"] = source_duration
    result["resampled_duration_sec"] = resampled_duration
    result["duration_diff_sec"] = duration_diff
    result["mne_version"] = mne.__version__
    result["qc_status"] = qc_status
    result["qc_notes"] = qc_notes
    return result


def write_manifest(manifest: pd.DataFrame, out_root: Path) -> Path:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "resample_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_plan(plan: pd.DataFrame, out_root: Path, mode: str) -> None:
    cols = [
        "patient",
        "trial",
        "source_sfreq_hz",
        "target_sfreq_hz",
        "n_channels",
        "source_n_samples",
        "source_10k_fif",
        "out_2k_fif",
    ]
    print(plan[cols].to_string(index=False))
    print(f"\nmode={mode}  n_segments={len(plan)}  out={out_root}")
    if not plan.empty:
        print("\npatient counts:")
        print(plan.groupby("patient").size().to_string())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Resample 10 kHz raw FIF segments to 2 kHz raw FIF segments")
    ap.add_argument("--dry-run", action="store_true", help="print resampling plan and write nothing")
    ap.add_argument("--patients", nargs="+", default=None, help="optional patient filter, e.g. --patients demo001 demo002")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional trial-id filter")
    ap.add_argument("--overwrite", action="store_true", help="replace existing 2 kHz FIF outputs")
    ap.add_argument("--plot-qc", action="store_true", help="write one PSD/waveform QC PNG per resampled segment")
    ap.add_argument("--out", default=None, help="override output directory under reviewed output roots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out) if args.out else RESAMPLED_ROOT
    assert_allowed_output_root(out_root)
    plan = build_resample_plan(SEGMENTS_ROOT, out_root, patients=args.patients, trials=args.trials)
    if plan.empty:
        raise SystemExit("No 10 kHz FIF segments selected for resampling.")
    validate_plan_sources(plan)
    mode = "dry-run" if args.dry_run else "full"
    print_plan(plan, out_root, mode)

    if args.dry_run:
        return

    results = [resample_one(row, overwrite=args.overwrite, plot_qc=args.plot_qc) for _, row in plan.iterrows()]
    manifest = pd.DataFrame.from_records(results)
    manifest_path = write_manifest(manifest, out_root)
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
