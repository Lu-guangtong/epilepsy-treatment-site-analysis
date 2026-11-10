"""Apply line-noise notch filtering to bipolar raw FIF segments.

This is an atomic notch step only. It reads the 2 kHz bipolar FIF files,
applies fixed 50 Hz harmonic FIR notch filters, and writes a new notched
FIF copy. It does not band-pass, high-pass, low-pass, resample, crop,
change channels, or read label tables.

Inspect the processing plan with --dry-run before execution:
  python scripts/notch_bipolar_line_noise.py --dry-run

Full notch filtering requires review of the processing plan before execution.
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
from scipy.signal import welch


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
BIPOLAR_ROOT = ROOT / "data_derived" / "bipolar"
NOTCHED_ROOT = ROOT / "data_derived" / "notched"
CHECK_ROOT = ROOT / "checks"
TARGET_SFREQ_HZ = 2000.0
NOTCH_FREQS_HZ = np.arange(50.0, 1000.0, 50.0)
NOTCH_FREQS_LABEL = ",".join(str(int(freq)) for freq in NOTCH_FREQS_HZ)
NOTCH_WIDTHS_HZ = 2.0
TRANS_BANDWIDTH_HZ = 1.0
NOTCH_METHOD = "fir"
NOTCH_PHASE = "zero"
FIR_WINDOW = "hamming"
FIR_DESIGN = "firwin"
FILTER_PAD = "reflect_limited"
CONVERSION_MODE = "line_noise_notch_only"
FIF_RE = re.compile(r"^(?P<patient>[a-z0-9]+)_trial(?P<trial>\d+)_bipolar_raw\.fif$", re.IGNORECASE)
EXPECTED_FULL_COUNTS = private_value("cohort", "expected_full_counts")


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
    notched_root = _resolve(NOTCHED_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == notched_root
        or _is_relative_to(out_dir, notched_root)
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
            "run normalize_segment_timebase.py before notch filtering"
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


def _meas_date_to_text(meas_date: object) -> str:
    if meas_date is None:
        return ""
    if hasattr(meas_date, "isoformat"):
        return meas_date.isoformat()
    return str(meas_date)


def _parse_bipolar_fif(path: Path) -> tuple[str, int] | None:
    match = FIF_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group("patient").lower(), int(match.group("trial"))


def _empty_plan() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient",
            "trial",
            "source_fif",
            "out_fif",
            "target_sfreq_hz",
            "notch_freqs_hz",
            "notch_widths_hz",
            "trans_bandwidth_hz",
            "method",
            "phase",
            "fir_window",
            "fir_design",
            "pad",
            "mne_version",
            "conversion_mode",
            "qc_status",
            "qc_notes",
        ]
    )


def build_notch_plan(
    bipolar_root: Path = BIPOLAR_ROOT,
    out_root: Path = NOTCHED_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build the bipolar-FIF to notched-FIF plan from existing files."""
    bipolar_root = Path(bipolar_root)
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    if not bipolar_root.exists():
        raise SystemExit(f"Missing bipolar input root: {bipolar_root}")

    patient_filter = _normalize_patients(patients)
    trial_filter = _normalize_trials(trials)
    rows: list[dict[str, object]] = []
    for source in sorted(bipolar_root.glob("*/*_bipolar_raw.fif")):
        parsed = _parse_bipolar_fif(source)
        if parsed is None:
            continue
        patient, trial = parsed
        if patient_filter is not None and patient not in patient_filter:
            continue
        if trial_filter is not None and trial not in trial_filter:
            continue
        rows.append(
            {
                "patient": patient,
                "trial": trial,
                "source_fif": str(source),
                "out_fif": str(out_root / patient / f"{patient}_trial{trial:02d}_notch_raw.fif"),
                "target_sfreq_hz": TARGET_SFREQ_HZ,
                "notch_freqs_hz": NOTCH_FREQS_LABEL,
                "notch_widths_hz": NOTCH_WIDTHS_HZ,
                "trans_bandwidth_hz": TRANS_BANDWIDTH_HZ,
                "method": NOTCH_METHOD,
                "phase": NOTCH_PHASE,
                "fir_window": FIR_WINDOW,
                "fir_design": FIR_DESIGN,
                "pad": FILTER_PAD,
                "mne_version": mne.__version__,
                "conversion_mode": CONVERSION_MODE,
                "qc_status": "planned",
                "qc_notes": "",
            }
        )
    if not rows:
        return _empty_plan()

    plan = pd.DataFrame.from_records(rows).sort_values(["patient", "trial"]).reset_index(drop=True)
    duplicate = plan.duplicated(["patient", "trial"], keep=False)
    if duplicate.any():
        dupes = plan.loc[duplicate, ["patient", "trial", "source_fif"]].to_dict("records")
        raise SystemExit(f"Duplicate bipolar FIF sources in notch plan: {dupes}")

    if patient_filter is None and trial_filter is None:
        counts = plan.groupby("patient").size().to_dict()
        if counts != EXPECTED_FULL_COUNTS:
            raise SystemExit(f"Unexpected full bipolar segment counts: {counts}")
    return plan


def validate_plan_sources(plan: pd.DataFrame) -> None:
    missing = [Path(str(path)) for path in plan["source_fif"] if not Path(str(path)).exists()]
    if missing:
        raise SystemExit("Missing bipolar FIF source files:\n" + "\n".join(str(path) for path in missing))


def validate_outputs_available(plan: pd.DataFrame, overwrite: bool) -> None:
    for out_file in plan["out_fif"].map(Path):
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite)


def capture_raw_state(raw: mne.io.BaseRaw) -> dict[str, object]:
    return {
        "sfreq": float(raw.info["sfreq"]),
        "n_channels": int(raw.info["nchan"]),
        "n_times": int(raw.n_times),
        "ch_names": list(raw.ch_names),
        "first_samp": int(raw.first_samp),
        "first_time": float(raw.first_time),
        "meas_date": raw.info.get("meas_date"),
    }


def _assert_raw_has_no_nan_inf(raw: mne.io.BaseRaw) -> None:
    block_size = max(1, int(float(raw.info["sfreq"]) * 10))
    for start in range(0, raw.n_times, block_size):
        stop = min(raw.n_times, start + block_size)
        block = raw.get_data(start=start, stop=stop)
        if not np.isfinite(block).all():
            raise SystemExit(f"Found NaN/Inf in notched data near samples {start}:{stop}")


def qc_notched_raw(raw: mne.io.BaseRaw, before: dict[str, object]) -> tuple[str, str]:
    notes: list[str] = []
    if float(raw.info["sfreq"]) != TARGET_SFREQ_HZ:
        notes.append(f"sfreq_after={float(raw.info['sfreq'])}, expected {TARGET_SFREQ_HZ}")
    if float(before["sfreq"]) != TARGET_SFREQ_HZ:
        notes.append(f"sfreq_before={float(before['sfreq'])}, expected {TARGET_SFREQ_HZ}")
    if int(raw.info["nchan"]) != int(before["n_channels"]):
        notes.append("channel count changed")
    if list(raw.ch_names) != list(before["ch_names"]):
        notes.append("channel names changed")
    if int(raw.n_times) != int(before["n_times"]):
        notes.append("n_times changed")
    if int(raw.first_samp) != int(before["first_samp"]) or float(raw.first_time) != float(before["first_time"]):
        notes.append("timebase changed")
    if int(raw.first_samp) != 0 or float(raw.first_time) != 0.0:
        notes.append("source is not an independent segment")
    if notes:
        return "fail", "; ".join(notes)
    return "pass", ""


def apply_notch(raw: mne.io.BaseRaw) -> None:
    raw.notch_filter(
        freqs=NOTCH_FREQS_HZ,
        notch_widths=NOTCH_WIDTHS_HZ,
        trans_bandwidth=TRANS_BANDWIDTH_HZ,
        method=NOTCH_METHOD,
        phase=NOTCH_PHASE,
        fir_window=FIR_WINDOW,
        fir_design=FIR_DESIGN,
        pad=FILTER_PAD,
        verbose="ERROR",
    )


def _pick_qc_channel(raw: mne.io.BaseRaw) -> int:
    variances = np.var(raw.get_data(start=0, stop=min(raw.n_times, int(raw.info["sfreq"]) * 10)), axis=1)
    if not np.isfinite(variances).all():
        return 0
    return int(np.nanargmax(variances))


def save_qc_plot(before_data: np.ndarray, after_raw: mne.io.BaseRaw, channel_index: int, out_file: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sfreq = float(after_raw.info["sfreq"])
    channel = after_raw.ch_names[channel_index]
    after_data = after_raw.get_data(picks=[channel_index])[0]
    nperseg = min(int(sfreq * 4), before_data.size, after_data.size)
    freqs_before, psd_before = welch(before_data, fs=sfreq, nperseg=nperseg)
    freqs_after, psd_after = welch(after_data, fs=sfreq, nperseg=nperseg)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].semilogy(freqs_before, psd_before, lw=0.7, label="before")
    axes[0].semilogy(freqs_after, psd_after, lw=0.7, label="after")
    axes[0].set_xlim(0, 1000)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title(f"Notch QC PSD: {out_file.stem} / {channel}")
    axes[0].legend(loc="upper right")

    duration = min(2.0, after_data.size / sfreq)
    stop = max(1, int(duration * sfreq))
    t = np.arange(stop) / sfreq
    before_std = np.nanstd(before_data[:stop]) + 1e-12
    after_std = np.nanstd(after_data[:stop]) + 1e-12
    axes[1].plot(t, before_data[:stop] / before_std, lw=0.5, label="before")
    axes[1].plot(t, after_data[:stop] / after_std, lw=0.5, label="after")
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("z-score")
    axes[1].set_title("Waveform overlay, first 2 s")
    axes[1].legend(loc="upper right")

    qc_path = out_file.with_suffix(".qc.png")
    fig.savefig(qc_path, dpi=120)
    plt.close(fig)


def notch_one(record: pd.Series, overwrite: bool, plot_qc: bool) -> dict[str, object]:
    source = Path(str(record["source_fif"]))
    out_file = Path(str(record["out_fif"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite)
    if not source.exists():
        raise SystemExit(f"Bipolar FIF source not found: {source}")

    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    assert_zero_timebase(raw, source)
    before = capture_raw_state(raw)
    if float(before["sfreq"]) != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected source sampling rate in {source}: {before['sfreq']}")

    qc_channel = _pick_qc_channel(raw) if plot_qc else None
    before_qc_data = raw.get_data(picks=[qc_channel])[0].copy() if qc_channel is not None else None

    apply_notch(raw)
    qc_status, qc_notes = qc_notched_raw(raw, before)
    if qc_status != "pass":
        raise SystemExit(f"QC failed for {out_file}: {qc_notes}")
    _assert_raw_has_no_nan_inf(raw)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_file, overwrite=overwrite, verbose="ERROR")
    if plot_qc:
        if qc_channel is None or before_qc_data is None:
            raise SystemExit("Internal error: missing QC data")
        save_qc_plot(before_qc_data, raw, qc_channel, out_file)

    result = record.to_dict()
    result.update(
        {
            "sfreq_before": float(before["sfreq"]),
            "sfreq_after": float(raw.info["sfreq"]),
            "n_channels_before": int(before["n_channels"]),
            "n_channels_after": int(raw.info["nchan"]),
            "n_times_before": int(before["n_times"]),
            "n_times_after": int(raw.n_times),
            "duration_sec_before": _duration_sec(int(before["n_times"]), float(before["sfreq"])),
            "duration_sec_after": _duration_sec(int(raw.n_times), float(raw.info["sfreq"])),
            "first_samp": int(raw.first_samp),
            "first_time": float(raw.first_time),
            "meas_date": _meas_date_to_text(raw.info.get("meas_date")),
            "mne_version": mne.__version__,
            "notch_freqs_hz": NOTCH_FREQS_LABEL,
            "notch_widths_hz": NOTCH_WIDTHS_HZ,
            "trans_bandwidth_hz": TRANS_BANDWIDTH_HZ,
            "method": NOTCH_METHOD,
            "phase": NOTCH_PHASE,
            "fir_window": FIR_WINDOW,
            "fir_design": FIR_DESIGN,
            "pad": FILTER_PAD,
            "conversion_mode": CONVERSION_MODE,
            "qc_status": qc_status,
            "qc_notes": qc_notes,
        }
    )
    return result


def write_manifest(manifest: pd.DataFrame, out_root: Path) -> Path:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "notch_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_plan(plan: pd.DataFrame, out_root: Path, mode: str) -> None:
    cols = [
        "patient",
        "trial",
        "target_sfreq_hz",
        "notch_freqs_hz",
        "notch_widths_hz",
        "source_fif",
        "out_fif",
    ]
    print(plan[cols].to_string(index=False))
    print(f"\nmode={mode}  n_segments={len(plan)}  out={out_root}")
    if not plan.empty:
        print("\npatient counts:")
        print(plan.groupby("patient").size().to_string())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply fixed line-noise notch filtering to bipolar raw FIF files")
    ap.add_argument("--dry-run", action="store_true", help="print notch plan and write nothing")
    ap.add_argument("--patients", nargs="+", default=None, help="optional patient filter, e.g. --patients demo001 demo002")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional trial-id filter")
    ap.add_argument("--overwrite", action="store_true", help="replace existing notched FIF outputs")
    ap.add_argument("--plot-qc", action="store_true", help="write one PSD/waveform QC PNG per notched segment")
    ap.add_argument("--out", default=None, help="override output directory under reviewed output roots")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out) if args.out else NOTCHED_ROOT
    assert_allowed_output_root(out_root)
    plan = build_notch_plan(BIPOLAR_ROOT, out_root, patients=args.patients, trials=args.trials)
    if plan.empty:
        raise SystemExit("No bipolar FIF segments selected for notch filtering.")
    validate_plan_sources(plan)
    mode = "dry-run" if args.dry_run else "full"
    print_plan(plan, out_root, mode)

    if args.dry_run:
        return

    validate_outputs_available(plan, overwrite=args.overwrite)
    results = [notch_one(row, overwrite=args.overwrite, plot_qc=args.plot_qc) for _, row in plan.iterrows()]
    manifest_path = write_manifest(pd.DataFrame.from_records(results), out_root)
    print(f"\nmanifest: {manifest_path}")


if __name__ == "__main__":
    main()
