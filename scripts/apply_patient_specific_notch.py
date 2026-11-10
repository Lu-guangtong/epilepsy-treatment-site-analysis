"""Apply patient-specific narrowband notch filters to already line-notched FIF files.

This is an optional, reviewed cleanup step for residual narrowband PSD artifacts
identified per patient. It reads from ${SEEG_OUTPUT_ROOT}/data_derived/notched and writes
new FIF files under ${SEEG_OUTPUT_ROOT}/data_derived/custom_notched by default.

Inspect the processing plan with --dry-run before execution:
  python scripts/apply_patient_specific_notch.py --dry-run

Full patient-specific filtering should be run manually by the user after review:
  python scripts/apply_patient_specific_notch.py --apply --plot-qc
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "patient_specific_notch.yaml"
NOTCHED_ROOT = ROOT / "data_derived" / "notched"
CUSTOM_NOTCHED_ROOT = ROOT / "data_derived" / "custom_notched"
CHECK_ROOT = ROOT / "checks"
CONVERSION_MODE = "patient_specific_custom_notch"
FIF_RE = re.compile(r"^(?P<patient>[a-z0-9]+)_trial(?P<trial>\d+)_notch_raw\.fif$", re.IGNORECASE)


@dataclass(frozen=True)
class PatientSpecificNotchConfig:
    input_root: Path
    output_root: Path
    patient_freqs_hz: dict[str, list[float]]
    notch_widths_hz: float
    trans_bandwidth_hz: float
    method: str
    phase: str
    fir_window: str
    fir_design: str
    pad: str
    conversion_mode: str = CONVERSION_MODE


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
    aux_root = _resolve(AUX_ROOT)
    notched_root = _resolve(NOTCHED_ROOT)
    custom_root = _resolve(CUSTOM_NOTCHED_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == aux_root or _is_relative_to(out_dir, aux_root):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if out_dir == notched_root or _is_relative_to(out_dir, notched_root):
        raise SystemExit(f"Refusing to write into original notched root: {out_dir}")
    if not (
        out_dir == custom_root
        or _is_relative_to(out_dir, custom_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def assert_output_available(out_file: Path, overwrite: bool) -> None:
    if out_file.exists() and not overwrite:
        raise SystemExit(f"Output already exists; pass --overwrite to replace it: {out_file}")


def load_notch_config(path: Path = CONFIG_PATH) -> PatientSpecificNotchConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = expand_config(yaml.safe_load(f))
    filt = raw["filter"]
    patient_freqs = {
        str(patient).lower(): [float(freq) for freq in freqs]
        for patient, freqs in raw["patient_freqs_hz"].items()
    }
    return PatientSpecificNotchConfig(
        input_root=Path(raw.get("input_root", NOTCHED_ROOT)),
        output_root=Path(raw.get("output_root", CUSTOM_NOTCHED_ROOT)),
        patient_freqs_hz=patient_freqs,
        notch_widths_hz=float(filt["notch_widths_hz"]),
        trans_bandwidth_hz=float(filt["trans_bandwidth_hz"]),
        method=str(filt["method"]),
        phase=str(filt["phase"]),
        fir_window=str(filt["fir_window"]),
        fir_design=str(filt["fir_design"]),
        pad=str(filt["pad"]),
        conversion_mode=str(raw.get("conversion_mode", CONVERSION_MODE)),
    )


def _normalize_patients(patients: Iterable[str] | None) -> set[str] | None:
    if patients is None:
        return None
    return {patient.strip().lower() for patient in patients if patient.strip()}


def _normalize_trials(trials: Iterable[int] | None) -> set[int] | None:
    if trials is None:
        return None
    return {int(trial) for trial in trials}


def _parse_notched_fif(path: Path) -> tuple[str, int] | None:
    match = FIF_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group("patient").lower(), int(match.group("trial"))


def _freqs_label(freqs_hz: Iterable[float]) -> str:
    return ",".join(f"{float(freq):g}" for freq in freqs_hz)


def _empty_plan() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "patient",
            "trial",
            "source_fif",
            "out_fif",
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


def build_patient_specific_notch_plan(
    config: PatientSpecificNotchConfig,
    input_root: Path | None = None,
    out_root: Path | None = None,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> pd.DataFrame:
    input_root = Path(input_root) if input_root is not None else config.input_root
    out_root = Path(out_root) if out_root is not None else config.output_root
    assert_allowed_output_root(out_root)
    if not input_root.exists():
        raise SystemExit(f"Missing notched input root: {input_root}")

    patient_filter = _normalize_patients(patients)
    trial_filter = _normalize_trials(trials)
    rows: list[dict[str, object]] = []
    for source in sorted(input_root.glob("*/*_notch_raw.fif")):
        parsed = _parse_notched_fif(source)
        if parsed is None:
            continue
        patient, trial = parsed
        if patient not in config.patient_freqs_hz:
            continue
        if patient_filter is not None and patient not in patient_filter:
            continue
        if trial_filter is not None and trial not in trial_filter:
            continue
        freqs = config.patient_freqs_hz[patient]
        rows.append(
            {
                "patient": patient,
                "trial": trial,
                "source_fif": str(source),
                "out_fif": str(out_root / patient / f"{patient}_trial{trial:02d}_custom_notch_raw.fif"),
                "notch_freqs_hz": _freqs_label(freqs),
                "notch_widths_hz": config.notch_widths_hz,
                "trans_bandwidth_hz": config.trans_bandwidth_hz,
                "method": config.method,
                "phase": config.phase,
                "fir_window": config.fir_window,
                "fir_design": config.fir_design,
                "pad": config.pad,
                "mne_version": mne.__version__,
                "conversion_mode": config.conversion_mode,
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
        raise SystemExit(f"Duplicate notched FIF sources in custom notch plan: {dupes}")
    return plan


def validate_plan_sources(plan: pd.DataFrame) -> None:
    missing = [Path(str(path)) for path in plan["source_fif"] if not Path(str(path)).exists()]
    if missing:
        raise SystemExit("Missing notched FIF source files:\n" + "\n".join(str(path) for path in missing))


def validate_outputs_available(plan: pd.DataFrame, overwrite: bool) -> None:
    for out_file in plan["out_fif"].map(Path):
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite)


def assert_zero_timebase(raw: mne.io.BaseRaw, source: Path) -> None:
    if int(raw.first_samp) != 0 or float(raw.first_time) != 0.0:
        raise SystemExit(
            f"{source}: source FIF is not an independent segment "
            f"(first_samp={raw.first_samp}, first_time={raw.first_time})"
        )


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


def qc_notched_raw(raw: mne.io.BaseRaw, before: dict[str, object]) -> tuple[str, str]:
    notes: list[str] = []
    if float(raw.info["sfreq"]) != float(before["sfreq"]):
        notes.append("sampling rate changed")
    if int(raw.info["nchan"]) != int(before["n_channels"]):
        notes.append("channel count changed")
    if list(raw.ch_names) != list(before["ch_names"]):
        notes.append("channel names changed")
    if int(raw.n_times) != int(before["n_times"]):
        notes.append("n_times changed")
    if int(raw.first_samp) != int(before["first_samp"]) or float(raw.first_time) != float(before["first_time"]):
        notes.append("timebase changed")
    if notes:
        return "fail", "; ".join(notes)
    return "pass", ""


def _assert_raw_has_no_nan_inf(raw: mne.io.BaseRaw) -> None:
    block_size = max(1, int(float(raw.info["sfreq"]) * 10))
    for start in range(0, raw.n_times, block_size):
        stop = min(raw.n_times, start + block_size)
        block = raw.get_data(start=start, stop=stop)
        if not np.isfinite(block).all():
            raise SystemExit(f"Found NaN/Inf in custom-notched data near samples {start}:{stop}")


def apply_patient_specific_notch(
    raw: mne.io.BaseRaw,
    freqs_hz: Iterable[float],
    notch_widths_hz: float = 2.0,
    trans_bandwidth_hz: float = 1.0,
    method: str = "fir",
    phase: str = "zero",
    fir_window: str = "hamming",
    fir_design: str = "firwin",
    pad: str = "reflect_limited",
) -> None:
    freqs = np.asarray([float(freq) for freq in freqs_hz], dtype=float)
    raw.notch_filter(
        freqs=freqs,
        notch_widths=float(notch_widths_hz),
        trans_bandwidth=float(trans_bandwidth_hz),
        method=method,
        phase=phase,
        fir_window=fir_window,
        fir_design=fir_design,
        pad=pad,
        verbose="ERROR",
    )


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


def _pick_qc_channel(raw: mne.io.BaseRaw) -> int:
    stop = min(raw.n_times, int(float(raw.info["sfreq"]) * 10))
    variances = np.var(raw.get_data(start=0, stop=stop), axis=1)
    if not np.isfinite(variances).all():
        return 0
    return int(np.nanargmax(variances))


def save_qc_plot(
    before_data: np.ndarray,
    after_raw: mne.io.BaseRaw,
    channel_index: int,
    freqs_hz: Iterable[float],
    out_file: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch

    sfreq = float(after_raw.info["sfreq"])
    channel = after_raw.ch_names[channel_index]
    after_data = after_raw.get_data(picks=[channel_index])[0]
    nperseg = min(int(sfreq * 4), before_data.size, after_data.size)
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs_before, psd_before = welch(before_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    freqs_after, psd_after = welch(after_data, fs=sfreq, nperseg=nperseg, noverlap=noverlap)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].semilogy(freqs_before, psd_before, lw=0.7, label="before")
    axes[0].semilogy(freqs_after, psd_after, lw=0.7, label="after")
    for freq in freqs_hz:
        axes[0].axvline(float(freq), color="#b23a2e", lw=0.8, alpha=0.8)
    axes[0].set_xlim(0, 200)
    axes[0].set_xlabel("frequency (Hz)")
    axes[0].set_ylabel("PSD")
    axes[0].set_title(f"Patient-specific notch QC: {out_file.stem} / {channel}")
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


def custom_notch_one(record: pd.Series, overwrite: bool, plot_qc: bool) -> dict[str, object]:
    source = Path(str(record["source_fif"]))
    out_file = Path(str(record["out_fif"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite)
    if not source.exists():
        raise SystemExit(f"Notched FIF source not found: {source}")

    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    assert_zero_timebase(raw, source)
    before = capture_raw_state(raw)
    freqs = [float(freq) for freq in str(record["notch_freqs_hz"]).split(",") if freq]

    qc_channel = _pick_qc_channel(raw) if plot_qc else None
    before_qc_data = raw.get_data(picks=[qc_channel])[0].copy() if qc_channel is not None else None

    apply_patient_specific_notch(
        raw,
        freqs_hz=freqs,
        notch_widths_hz=float(record["notch_widths_hz"]),
        trans_bandwidth_hz=float(record["trans_bandwidth_hz"]),
        method=str(record["method"]),
        phase=str(record["phase"]),
        fir_window=str(record["fir_window"]),
        fir_design=str(record["fir_design"]),
        pad=str(record["pad"]),
    )
    qc_status, qc_notes = qc_notched_raw(raw, before)
    if qc_status != "pass":
        raise SystemExit(f"QC failed for {out_file}: {qc_notes}")
    _assert_raw_has_no_nan_inf(raw)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    raw.save(out_file, overwrite=overwrite, verbose="ERROR")
    if plot_qc:
        if qc_channel is None or before_qc_data is None:
            raise SystemExit("Internal error: missing QC data")
        save_qc_plot(before_qc_data, raw, qc_channel, freqs, out_file)

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
            "qc_status": qc_status,
            "qc_notes": qc_notes,
        }
    )
    return result


def write_manifest(manifest: pd.DataFrame, out_root: Path) -> Path:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    path = out_root / "custom_notch_manifest.csv"
    manifest.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def print_plan(plan: pd.DataFrame, out_root: Path, mode: str) -> None:
    cols = [
        "patient",
        "trial",
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


def run_patient_specific_notch(
    config_path: Path = CONFIG_PATH,
    out_root: Path | None = None,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
    apply: bool = False,
    overwrite: bool = False,
    plot_qc: bool = False,
) -> pd.DataFrame:
    config = load_notch_config(config_path)
    final_out = Path(out_root) if out_root is not None else config.output_root
    plan = build_patient_specific_notch_plan(config, config.input_root, final_out, patients=patients, trials=trials)
    if plan.empty:
        raise SystemExit("No notched FIF segments selected for patient-specific notch filtering.")
    validate_plan_sources(plan)
    mode = "apply" if apply else "dry-run"
    print_plan(plan, final_out, mode)

    if not apply:
        return plan

    validate_outputs_available(plan, overwrite=overwrite)
    results = [custom_notch_one(row, overwrite=overwrite, plot_qc=plot_qc) for _, row in plan.iterrows()]
    manifest = pd.DataFrame.from_records(results)
    manifest_path = write_manifest(manifest, final_out)
    print(f"\nmanifest: {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Apply reviewed patient-specific custom notch filters to notched FIF files"
    )
    ap.add_argument("--config", type=Path, default=CONFIG_PATH, help="patient-specific notch YAML config")
    ap.add_argument("--dry-run", action="store_true", help="print plan and write nothing")
    ap.add_argument("--apply", action="store_true", help="write custom-notched FIF files")
    ap.add_argument("--patients", nargs="+", default=None, help="optional patient filter, e.g. --patients demo001 demo002")
    ap.add_argument("--trials", nargs="+", type=int, default=None, help="optional trial number filter")
    ap.add_argument("--out", type=Path, default=None, help="override output root under custom_notched or checks")
    ap.add_argument("--overwrite", action="store_true", help="replace existing outputs under allowed output roots")
    ap.add_argument("--plot-qc", action="store_true", help="write one PSD/waveform QC PNG per processed FIF")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.apply and args.dry_run:
        raise SystemExit("Use either --dry-run or --apply, not both.")
    run_patient_specific_notch(
        config_path=args.config,
        out_root=args.out,
        patients=args.patients,
        trials=args.trials,
        apply=args.apply,
        overwrite=args.overwrite,
        plot_qc=args.plot_qc,
    )


if __name__ == "__main__":
    main()
