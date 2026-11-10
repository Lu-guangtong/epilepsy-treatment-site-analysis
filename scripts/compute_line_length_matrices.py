"""Compute raw subwindow-level line-length matrices for bipolar SEEG.

Each output matrix has shape ``channels x subwindows``. The script performs
band-pass filtering and the line-length calculation only; it does not aggregate
subwindows or trials and never reads clinical or anatomical labels.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import csv
import gc
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import mne
import numpy as np
import pandas as pd
import scipy


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
FINAL_SIGNAL_ROOT = ROOT / "data_derived" / "custom_notched"
ONSET_MANIFEST = ROOT / "data_derived" / "resampled" / "resample_manifest.csv"
LINE_LENGTH_ROOT = ROOT / "data_derived" / "electrode_metrics_raw" / "line_length"
CHECK_ROOT = ROOT / "checks"
TARGET_SFREQ_HZ = 2000.0
METHOD = "line_length_rate"
MATRIX_MODE = "all_bipolar_channels_by_subwindow_no_aggregation"
VALUE_FORMULA = "sum(abs(diff(bandpassed_signal))) / ((n_samples - 1) / sfreq_hz)"
VALUE_UNIT = "V_per_s"
EXPECTED_FULL_COUNTS = private_value("cohort", "expected_full_counts")
FINAL_FIF_RE = re.compile(
    r"^(?P<patient>[a-z0-9]+)_trial(?P<trial>\d+)_custom_notch_raw\.fif$",
    re.IGNORECASE,
)

WINDOWS = {
    "baseline": (-40.0, -5.0),
    "early_ictal": (0.0, 10.0),
}
FREQUENCY_BANDS = {
    "delta": (1.0, 4.0, 5.0, 0.5),
    "theta": (4.0, 8.0, 5.0, 0.5),
    "alpha": (8.0, 13.0, 5.0, 0.5),
    "beta": (13.0, 30.0, 5.0, 0.5),
    "low_gamma": (30.0, 80.0, 2.0, 0.5),
    "high_gamma": (80.0, 150.0, 2.0, 0.5),
    "ripple": (80.0, 250.0, 2.0, 0.5),
    "fast_ripple": (250.0, 500.0, 2.0, 0.5),
    "broadband_1_500": (1.0, 500.0, 2.0, 0.5),
}

FILTER_METHOD = "fir"
FILTER_PHASE = "zero"
FIR_WINDOW = "hamming"
FIR_DESIGN = "firwin"
FILTER_LENGTH = "auto"
L_TRANS_BANDWIDTH = "auto"
H_TRANS_BANDWIDTH = "auto"
FILTER_PAD = "reflect_limited"
FILTER_N_JOBS = 1

QC_VERSION = "subwindow_artifact_qc_v1"
QC_FLAG_FLAT = np.uint8(1)
QC_FLAG_CLIPPING = np.uint8(2)
QC_FLAG_JUMP = np.uint8(4)
QC_FLAG_LINE_LENGTH_OUTLIER = np.uint8(8)
FLAT_ZERO_DIFF_FRACTION_THRESHOLD = 0.99
CLIPPING_EXTREME_FRACTION_THRESHOLD = 0.01
JUMP_ROBUST_Z_THRESHOLD = 20.0
LINE_LENGTH_OUTLIER_ROBUST_Z_THRESHOLD = 8.0


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
    formal_root = _resolve(LINE_LENGTH_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == formal_root
        or _is_relative_to(out_dir, formal_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def assert_output_available(out_file: Path, overwrite: bool, resume: bool) -> None:
    if out_file.exists() and not (overwrite or resume):
        raise SystemExit(f"Output already exists; pass --overwrite or --resume: {out_file}")


def _normalize_strings(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized = {str(value).strip().lower() for value in values if str(value).strip()}
    return normalized or None


def _normalize_ints(values: Iterable[int] | None) -> set[int] | None:
    return None if values is None else {int(value) for value in values}


def _normalize_names(
    values: Iterable[str] | None,
    allowed: Iterable[str],
    label: str,
) -> set[str] | None:
    normalized = _normalize_strings(values)
    if normalized is None:
        return None
    unknown = sorted(normalized - set(allowed))
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}; allowed={sorted(allowed)}")
    return normalized


def parse_final_fif(path: Path) -> tuple[str, int] | None:
    match = FINAL_FIF_RE.fullmatch(path.name)
    if match is None:
        return None
    return match.group("patient").lower(), int(match.group("trial"))


def load_onsets(onset_manifest: Path = ONSET_MANIFEST) -> dict[tuple[str, int], float]:
    required = {"patient", "trial", "onset_sec"}
    onsets: dict[tuple[str, int], float] = {}
    with Path(onset_manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"{onset_manifest}: missing required onset columns")
        for row in reader:
            key = (str(row["patient"]).strip().lower(), int(row["trial"]))
            if key in onsets:
                raise SystemExit(f"{onset_manifest}: duplicate onset for {key}")
            onset = float(row["onset_sec"])
            if not np.isfinite(onset):
                raise SystemExit(f"{onset_manifest}: non-finite onset for {key}")
            onsets[key] = onset
    return onsets


def discover_signal_files(
    signals_root: Path = FINAL_SIGNAL_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> list[dict[str, object]]:
    patient_filter = _normalize_strings(patients)
    trial_filter = _normalize_ints(trials)
    rows: list[dict[str, object]] = []
    for path in sorted(Path(signals_root).glob("*/*.fif")):
        parsed = parse_final_fif(path)
        if parsed is None:
            continue
        patient, trial = parsed
        if patient_filter is not None and patient not in patient_filter:
            continue
        if trial_filter is not None and trial not in trial_filter:
            continue
        rows.append({"patient": patient, "trial": trial, "source_fif": str(path)})
    return rows


def expected_subwindow_count(
    window_duration_sec: float,
    subwindow_length_sec: float,
    overlap_fraction: float,
) -> int:
    step = subwindow_length_sec * (1.0 - overlap_fraction)
    return int(np.floor((window_duration_sec - subwindow_length_sec) / step + 1.0 + 1e-9))


def build_line_length_plan(
    signals_root: Path = FINAL_SIGNAL_ROOT,
    onset_manifest: Path = ONSET_MANIFEST,
    out_root: Path = LINE_LENGTH_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
    bands: Iterable[str] | None = None,
    windows: Iterable[str] | None = None,
) -> pd.DataFrame:
    assert_allowed_output_root(Path(out_root))
    onsets = load_onsets(onset_manifest)
    signals = discover_signal_files(signals_root, patients=patients, trials=trials)
    if not signals:
        raise SystemExit(f"No final FIF sources found under {signals_root}")

    band_filter = _normalize_names(bands, FREQUENCY_BANDS, "frequency bands")
    window_filter = _normalize_names(windows, WINDOWS, "windows")
    selected_bands = [name for name in FREQUENCY_BANDS if band_filter is None or name in band_filter]
    selected_windows = [name for name in WINDOWS if window_filter is None or name in window_filter]

    keys = [(str(row["patient"]), int(row["trial"])) for row in signals]
    if len(keys) != len(set(keys)):
        raise SystemExit("Duplicate final FIF source for one or more trials")
    if patients is None and trials is None:
        counts = pd.DataFrame(signals).groupby("patient").size().to_dict()
        if counts != EXPECTED_FULL_COUNTS:
            raise SystemExit(f"Unexpected final FIF counts: {counts}")

    records: list[dict[str, object]] = []
    for signal in sorted(signals, key=lambda row: (str(row["patient"]), int(row["trial"]))):
        patient = str(signal["patient"])
        trial = int(signal["trial"])
        onset = onsets.get((patient, trial))
        if onset is None:
            raise SystemExit(f"Missing onset for {patient} trial {trial:02d}")
        for window_name in selected_windows:
            rel_start, rel_end = WINDOWS[window_name]
            window_start = onset + rel_start
            window_end = onset + rel_end
            if window_start < 0.0 or window_end <= window_start:
                raise SystemExit(f"Invalid {window_name} window for {patient} trial {trial:02d}")
            for band_name in selected_bands:
                fmin, fmax, length, overlap = FREQUENCY_BANDS[band_name]
                n_subwindows = expected_subwindow_count(
                    window_end - window_start,
                    length,
                    overlap,
                )
                out_file = (
                    Path(out_root)
                    / patient
                    / f"{patient}_trial{trial:02d}_{window_name}_{band_name}_line_length_raw.npz"
                )
                records.append(
                    {
                        "patient": patient,
                        "trial": trial,
                        "source_fif": str(signal["source_fif"]),
                        "out_file": str(out_file),
                        "window": window_name,
                        "window_start_sec": float(window_start),
                        "window_end_sec": float(window_end),
                        "window_start_sec_from_onset": float(rel_start),
                        "window_end_sec_from_onset": float(rel_end),
                        "frequency_band": band_name,
                        "fmin_hz": fmin,
                        "fmax_hz": fmax,
                        "subwindow_length_sec": length,
                        "overlap_fraction": overlap,
                        "expected_n_subwindows": n_subwindows,
                        "method": METHOD,
                        "onset_sec": float(onset),
                        "value_formula": VALUE_FORMULA,
                        "value_unit": VALUE_UNIT,
                        "filter_method": FILTER_METHOD,
                        "filter_phase": FILTER_PHASE,
                        "fir_window": FIR_WINDOW,
                        "fir_design": FIR_DESIGN,
                        "filter_length": FILTER_LENGTH,
                        "l_trans_bandwidth": L_TRANS_BANDWIDTH,
                        "h_trans_bandwidth": H_TRANS_BANDWIDTH,
                        "filter_pad": FILTER_PAD,
                        "matrix_mode": MATRIX_MODE,
                        "mne_version": mne.__version__,
                        "numpy_version": np.__version__,
                        "scipy_version": scipy.__version__,
                    }
                )
    return pd.DataFrame.from_records(records)


def subwindow_sample_slices(
    window_start_sec: float,
    window_end_sec: float,
    sfreq: float,
    subwindow_length_sec: float,
    overlap_fraction: float,
) -> list[tuple[int, int]]:
    if not 0.0 <= overlap_fraction < 1.0:
        raise SystemExit(f"Invalid overlap fraction: {overlap_fraction}")
    if window_end_sec <= window_start_sec or subwindow_length_sec <= 0.0 or sfreq <= 0.0:
        raise SystemExit("Invalid subwindow geometry")
    n_samples = int(round(subwindow_length_sec * sfreq))
    step_samples = int(round(subwindow_length_sec * (1.0 - overlap_fraction) * sfreq))
    first_start = int(round(window_start_sec * sfreq))
    last_start = int(round((window_end_sec - subwindow_length_sec) * sfreq))
    starts = range(first_start, last_start + 1, step_samples)
    return [(start, start + n_samples) for start in starts]


def filter_full_trial(data: np.ndarray, sfreq: float, fmin_hz: float, fmax_hz: float) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 2:
        raise SystemExit(f"Expected channels x samples data, got {data.shape}")
    if sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected sampling rate: {sfreq}")
    if not np.isfinite(data).all():
        raise SystemExit("Input signal contains NaN or Inf")
    if not 0.0 < fmin_hz < fmax_hz < sfreq / 2.0:
        raise SystemExit(f"Invalid passband {fmin_hz}-{fmax_hz} Hz")
    filtered = mne.filter.filter_data(
        data,
        sfreq=sfreq,
        l_freq=fmin_hz,
        h_freq=fmax_hz,
        method=FILTER_METHOD,
        phase=FILTER_PHASE,
        fir_window=FIR_WINDOW,
        fir_design=FIR_DESIGN,
        filter_length=FILTER_LENGTH,
        l_trans_bandwidth=L_TRANS_BANDWIDTH,
        h_trans_bandwidth=H_TRANS_BANDWIDTH,
        pad=FILTER_PAD,
        n_jobs=FILTER_N_JOBS,
        copy=True,
        verbose="ERROR",
    )
    if filtered.shape != data.shape or not np.isfinite(filtered).all():
        raise SystemExit("Band-pass output is non-finite or has an unexpected shape")
    return np.asarray(filtered, dtype=np.float64)


def compute_line_length_matrix(
    filtered_full_trial: np.ndarray,
    slices: list[tuple[int, int]],
    sfreq: float,
) -> np.ndarray:
    filtered = np.asarray(filtered_full_trial, dtype=np.float64)
    if filtered.ndim != 2 or not slices:
        raise SystemExit("Line length requires channels x samples data and at least one subwindow")
    matrix = np.empty((filtered.shape[0], len(slices)), dtype=np.float64)
    for column, (start, stop) in enumerate(slices):
        if start < 0 or stop > filtered.shape[1] or stop - start < 2:
            raise SystemExit(f"Invalid line-length slice {start}:{stop}")
        segment = filtered[:, start:stop]
        duration_sec = (segment.shape[1] - 1) / sfreq
        matrix[:, column] = np.sum(np.abs(np.diff(segment, axis=1)), axis=1) / duration_sec
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise SystemExit("Line-length matrix contains invalid values")
    if not np.any(matrix > 0.0):
        raise SystemExit("Line-length matrix is entirely zero")
    return matrix


def _robust_positive_z(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=axis, keepdims=True)
    mad = np.median(np.abs(values - median), axis=axis, keepdims=True)
    scale = 1.4826 * mad
    max_abs = np.max(np.abs(values), axis=axis, keepdims=True)
    scale_floor = np.finfo(np.float64).eps * np.maximum(max_abs, np.finfo(np.float64).tiny)
    return np.maximum((values - median) / np.maximum(scale, scale_floor), 0.0)


def compute_subwindow_artifact_qc(
    raw_full_trial: np.ndarray,
    line_length_matrix: np.ndarray,
    slices: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    """Return audit-only artifact metrics without changing or excluding samples."""
    raw_data = np.asarray(raw_full_trial, dtype=np.float64)
    line_length = np.asarray(line_length_matrix, dtype=np.float64)
    expected_shape = (raw_data.shape[0], len(slices))
    if raw_data.ndim != 2 or line_length.shape != expected_shape or not slices:
        raise SystemExit("Artifact QC inputs have incompatible shapes")
    if not np.isfinite(raw_data).all() or not np.isfinite(line_length).all():
        raise SystemExit("Artifact QC inputs contain NaN or Inf")

    zero_diff_fraction = np.empty(expected_shape, dtype=np.float64)
    peak_to_peak_v = np.empty(expected_shape, dtype=np.float64)
    extreme_value_fraction = np.empty(expected_shape, dtype=np.float64)
    max_abs_step_v = np.empty(expected_shape, dtype=np.float64)
    step_robust_z = np.empty(expected_shape, dtype=np.float64)

    for column, (start, stop) in enumerate(slices):
        if start < 0 or stop > raw_data.shape[1] or stop - start < 2:
            raise SystemExit(f"Invalid artifact-QC slice {start}:{stop}")
        segment = raw_data[:, start:stop]
        abs_steps = np.abs(np.diff(segment, axis=1))
        segment_min = np.min(segment, axis=1, keepdims=True)
        segment_max = np.max(segment, axis=1, keepdims=True)

        zero_diff_fraction[:, column] = np.mean(abs_steps == 0.0, axis=1)
        peak_to_peak_v[:, column] = np.ptp(segment, axis=1)
        extreme_value_fraction[:, column] = np.mean(
            (segment == segment_min) | (segment == segment_max), axis=1
        )
        max_abs_step_v[:, column] = np.max(abs_steps, axis=1)
        step_robust_z[:, column] = _robust_positive_z(abs_steps, axis=1).max(axis=1)

    line_length_robust_z = _robust_positive_z(line_length, axis=0)
    flag_mask = np.zeros(expected_shape, dtype=np.uint8)
    flat = (zero_diff_fraction >= FLAT_ZERO_DIFF_FRACTION_THRESHOLD) | (
        peak_to_peak_v == 0.0
    )
    clipping = extreme_value_fraction >= CLIPPING_EXTREME_FRACTION_THRESHOLD
    jump = step_robust_z >= JUMP_ROBUST_Z_THRESHOLD
    line_length_outlier = (
        line_length_robust_z >= LINE_LENGTH_OUTLIER_ROBUST_Z_THRESHOLD
    )
    flag_mask[flat] |= QC_FLAG_FLAT
    flag_mask[clipping] |= QC_FLAG_CLIPPING
    flag_mask[jump] |= QC_FLAG_JUMP
    flag_mask[line_length_outlier] |= QC_FLAG_LINE_LENGTH_OUTLIER

    metrics = {
        "qc_raw_zero_diff_fraction": zero_diff_fraction,
        "qc_raw_peak_to_peak_v": peak_to_peak_v,
        "qc_raw_extreme_value_fraction": extreme_value_fraction,
        "qc_raw_max_abs_step_v": max_abs_step_v,
        "qc_raw_step_robust_z": step_robust_z,
        "qc_line_length_robust_z": line_length_robust_z,
        "qc_flag_mask": flag_mask,
    }
    if any(array.shape != expected_shape for array in metrics.values()):
        raise SystemExit("Artifact QC produced an unexpected matrix shape")
    if any(not np.isfinite(array).all() for key, array in metrics.items() if key != "qc_flag_mask"):
        raise SystemExit("Artifact QC produced NaN or Inf")
    return metrics


def _run_with_heartbeat(fn: Callable[[], np.ndarray], label: str, heartbeat_sec: int) -> np.ndarray:
    if heartbeat_sec <= 0:
        return fn()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        while not future.done():
            try:
                return future.result(timeout=heartbeat_sec)
            except TimeoutError:
                print(
                    f"  still running {label}; elapsed={time.perf_counter() - started:.1f}s",
                    flush=True,
                )
        return future.result()


def _np_scalar(value: object) -> np.ndarray:
    return np.asarray(value)


def save_matrix_npz(
    out_file: Path,
    matrix: np.ndarray,
    artifact_qc: dict[str, np.ndarray],
    channel_names: list[str],
    slices: list[tuple[int, int]],
    record: dict[str, object],
    sfreq: float,
) -> None:
    starts_segment = np.asarray([start / sfreq for start, _ in slices], dtype=np.float64)
    ends_segment = np.asarray([stop / sfreq for _, stop in slices], dtype=np.float64)
    onset = float(record["onset_sec"])
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_name(out_file.name + ".tmp")
    with tmp_file.open("wb") as handle:
        np.savez_compressed(
            handle,
            matrix=matrix,
            **artifact_qc,
            channel_names=np.asarray(channel_names, dtype=str),
            subwindow_start_sec_from_segment=starts_segment,
            subwindow_end_sec_from_segment=ends_segment,
            subwindow_start_sec_from_onset=starts_segment - onset,
            subwindow_end_sec_from_onset=ends_segment - onset,
            patient=_np_scalar(record["patient"]),
            trial=_np_scalar(int(record["trial"])),
            window=_np_scalar(record["window"]),
            frequency_band=_np_scalar(record["frequency_band"]),
            method=_np_scalar(METHOD),
            sfreq_hz=_np_scalar(sfreq),
            onset_sec=_np_scalar(onset),
            fmin_hz=_np_scalar(float(record["fmin_hz"])),
            fmax_hz=_np_scalar(float(record["fmax_hz"])),
            subwindow_length_sec=_np_scalar(float(record["subwindow_length_sec"])),
            subwindow_overlap_fraction=_np_scalar(float(record["overlap_fraction"])),
            value_formula=_np_scalar(VALUE_FORMULA),
            value_unit=_np_scalar(VALUE_UNIT),
            qc_version=_np_scalar(QC_VERSION),
            qc_flags_are_audit_only=_np_scalar(True),
            qc_flat_zero_diff_fraction_threshold=_np_scalar(
                FLAT_ZERO_DIFF_FRACTION_THRESHOLD
            ),
            qc_clipping_extreme_fraction_threshold=_np_scalar(
                CLIPPING_EXTREME_FRACTION_THRESHOLD
            ),
            qc_jump_robust_z_threshold=_np_scalar(JUMP_ROBUST_Z_THRESHOLD),
            qc_line_length_outlier_robust_z_threshold=_np_scalar(
                LINE_LENGTH_OUTLIER_ROBUST_Z_THRESHOLD
            ),
            filter_method=_np_scalar(FILTER_METHOD),
            filter_phase=_np_scalar(FILTER_PHASE),
            fir_window=_np_scalar(FIR_WINDOW),
            fir_design=_np_scalar(FIR_DESIGN),
            filter_length=_np_scalar(FILTER_LENGTH),
            l_trans_bandwidth=_np_scalar(L_TRANS_BANDWIDTH),
            h_trans_bandwidth=_np_scalar(H_TRANS_BANDWIDTH),
            filter_pad=_np_scalar(FILTER_PAD),
            matrix_mode=_np_scalar(MATRIX_MODE),
            source_fif=_np_scalar(record["source_fif"]),
            mne_version=_np_scalar(mne.__version__),
            numpy_version=_np_scalar(np.__version__),
            scipy_version=_np_scalar(scipy.__version__),
        )
    tmp_file.replace(out_file)


def _npz_matches_record(out_file: Path, record: dict[str, object]) -> bool:
    try:
        with np.load(out_file, allow_pickle=False) as saved:
            return all(
                [
                    str(saved["patient"].item()) == str(record["patient"]),
                    int(saved["trial"].item()) == int(record["trial"]),
                    str(saved["window"].item()) == str(record["window"]),
                    str(saved["frequency_band"].item()) == str(record["frequency_band"]),
                    str(saved["method"].item()) == METHOD,
                    str(saved["value_formula"].item()) == VALUE_FORMULA,
                    str(saved["qc_version"].item()) == QC_VERSION,
                    float(saved["fmin_hz"].item()) == float(record["fmin_hz"]),
                    float(saved["fmax_hz"].item()) == float(record["fmax_hz"]),
                    float(saved["subwindow_length_sec"].item())
                    == float(record["subwindow_length_sec"]),
                    int(saved["matrix"].shape[1]) == int(record["expected_n_subwindows"]),
                ]
            )
    except Exception:
        return False


def _validate_source(raw: mne.io.BaseRaw, source: Path, records: list[dict[str, object]]) -> None:
    if int(raw.first_samp) != 0 or float(raw.first_time) != 0.0:
        raise SystemExit(f"{source}: source FIF does not have an independent zero timebase")
    sfreq = float(raw.info["sfreq"])
    if sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected sampling rate in {source}: {sfreq}")
    if len(raw.ch_names) != len(set(raw.ch_names)) or not raw.ch_names:
        raise SystemExit(f"Missing or duplicate channel names in {source}")
    max_stop = max(float(record["window_end_sec"]) for record in records)
    if int(round(max_stop * sfreq)) > raw.n_times:
        raise SystemExit(f"Analysis window exceeds source duration: {source}")


def _existing_result(record: dict[str, object], out_file: Path) -> dict[str, object]:
    with np.load(out_file, allow_pickle=False) as saved:
        matrix = np.asarray(saved["matrix"], dtype=np.float64)
        flag_mask = np.asarray(saved["qc_flag_mask"], dtype=np.uint8)
    if matrix.shape != flag_mask.shape or not np.isfinite(matrix).all():
        raise SystemExit(f"Existing output contains invalid matrix or QC data: {out_file}")
    n_flat = int(np.count_nonzero(flag_mask & QC_FLAG_FLAT))
    n_clipping = int(np.count_nonzero(flag_mask & QC_FLAG_CLIPPING))
    n_jump = int(np.count_nonzero(flag_mask & QC_FLAG_JUMP))
    n_line_length_outlier = int(
        np.count_nonzero(flag_mask & QC_FLAG_LINE_LENGTH_OUTLIER)
    )
    n_any = int(np.count_nonzero(flag_mask))
    return {
        **record,
        "qc_status": "computed",
        "qc_notes": (
            "existing output validated during resume; artifact flags retained; "
            "no samples or subwindows excluded"
        ),
        "n_channels": matrix.shape[0],
        "n_subwindows": matrix.shape[1],
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "matrix_min": float(matrix.min()),
        "matrix_max": float(matrix.max()),
        "n_flat_flags": n_flat,
        "n_clipping_flags": n_clipping,
        "n_jump_flags": n_jump,
        "n_line_length_outlier_flags": n_line_length_outlier,
        "n_any_artifact_flags": n_any,
        "artifact_flag_fraction": n_any / flag_mask.size,
    }


def compute_one_trial(
    trial_plan: pd.DataFrame,
    overwrite: bool = False,
    resume: bool = False,
    heartbeat_sec: int = 60,
) -> list[dict[str, object]]:
    records = [row.to_dict() for _, row in trial_plan.iterrows()]
    if not records:
        return []
    sources = {str(record["source_fif"]) for record in records}
    if len(sources) != 1:
        raise SystemExit("Trial plan contains multiple source FIF files")
    source = Path(next(iter(sources)))
    if not source.exists():
        raise SystemExit(f"Missing source FIF: {source}")

    results: list[dict[str, object]] = []
    pending_by_band: dict[str, list[dict[str, object]]] = {}
    for record in records:
        out_file = Path(str(record["out_file"]))
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite=overwrite, resume=resume)
        if out_file.exists() and resume and not overwrite:
            if not _npz_matches_record(out_file, record):
                raise SystemExit(f"Existing output metadata mismatch: {out_file}")
            results.append(_existing_result(record, out_file))
        else:
            pending_by_band.setdefault(str(record["frequency_band"]), []).append(record)
    if not pending_by_band:
        return results

    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    _validate_source(raw, source, records)
    sfreq = float(raw.info["sfreq"])
    channel_names = list(raw.ch_names)
    data = raw.get_data().astype(np.float64, copy=False)
    del raw
    gc.collect()
    if not np.isfinite(data).all():
        raise SystemExit(f"Source contains NaN or Inf: {source}")

    patient = str(records[0]["patient"])
    trial = int(records[0]["trial"])
    for band_name in FREQUENCY_BANDS:
        band_records = pending_by_band.get(band_name)
        if not band_records:
            continue
        first = band_records[0]
        filtered = _run_with_heartbeat(
            lambda: filter_full_trial(
                data,
                sfreq,
                float(first["fmin_hz"]),
                float(first["fmax_hz"]),
            ),
            f"{patient} trial{trial:02d} {band_name} full-trial FIR",
            heartbeat_sec,
        )
        for record in sorted(band_records, key=lambda item: str(item["window"])):
            slices = subwindow_sample_slices(
                float(record["window_start_sec"]),
                float(record["window_end_sec"]),
                sfreq,
                float(record["subwindow_length_sec"]),
                float(record["overlap_fraction"]),
            )
            if len(slices) != int(record["expected_n_subwindows"]):
                raise SystemExit(f"Unexpected subwindow count for {record['out_file']}")
            matrix = compute_line_length_matrix(filtered, slices, sfreq)
            artifact_qc = compute_subwindow_artifact_qc(data, matrix, slices)
            expected_shape = (len(channel_names), len(slices))
            if matrix.shape != expected_shape:
                raise SystemExit(f"Output matrix shape mismatch: {matrix.shape} != {expected_shape}")
            out_file = Path(str(record["out_file"]))
            save_matrix_npz(
                out_file,
                matrix,
                artifact_qc,
                channel_names,
                slices,
                record,
                sfreq,
            )
            flag_mask = artifact_qc["qc_flag_mask"]
            n_flat = int(np.count_nonzero(flag_mask & QC_FLAG_FLAT))
            n_clipping = int(np.count_nonzero(flag_mask & QC_FLAG_CLIPPING))
            n_jump = int(np.count_nonzero(flag_mask & QC_FLAG_JUMP))
            n_line_length_outlier = int(
                np.count_nonzero(flag_mask & QC_FLAG_LINE_LENGTH_OUTLIER)
            )
            n_any = int(np.count_nonzero(flag_mask))
            results.append(
                {
                    **record,
                    "qc_status": "computed",
                    "qc_notes": "artifact flags retained; no samples or subwindows excluded",
                    "n_channels": len(channel_names),
                    "n_subwindows": len(slices),
                    "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                    "matrix_min": float(matrix.min()),
                    "matrix_max": float(matrix.max()),
                    "n_flat_flags": n_flat,
                    "n_clipping_flags": n_clipping,
                    "n_jump_flags": n_jump,
                    "n_line_length_outlier_flags": n_line_length_outlier,
                    "n_any_artifact_flags": n_any,
                    "artifact_flag_fraction": n_any / flag_mask.size,
                }
            )
        del filtered
        gc.collect()
    return results


def write_run_outputs(results: list[dict[str, object]], out_root: Path, started_at: float) -> None:
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame.from_records(results)
    if not manifest.empty:
        manifest = manifest.sort_values(
            ["patient", "trial", "window", "frequency_band"], kind="stable"
        ).reset_index(drop=True)
    manifest.to_csv(
        out_root / "line_length_matrix_manifest.csv", index=False, encoding="utf-8-sig"
    )
    summary = {
        "n_results": len(results),
        "qc_status_counts": manifest["qc_status"].value_counts().to_dict()
        if not manifest.empty
        else {},
        "elapsed_sec": time.perf_counter() - started_at,
        "method": METHOD,
        "value_formula": VALUE_FORMULA,
        "value_unit": VALUE_UNIT,
        "matrix_mode": MATRIX_MODE,
        "artifact_qc": {
            "version": QC_VERSION,
            "flags_are_audit_only": True,
            "no_samples_or_subwindows_excluded": True,
            "flag_bits": {
                "flat": int(QC_FLAG_FLAT),
                "clipping_or_saturation": int(QC_FLAG_CLIPPING),
                "isolated_jump": int(QC_FLAG_JUMP),
                "line_length_outlier": int(QC_FLAG_LINE_LENGTH_OUTLIER),
            },
            "flat_zero_diff_fraction_threshold": FLAT_ZERO_DIFF_FRACTION_THRESHOLD,
            "clipping_extreme_fraction_threshold": CLIPPING_EXTREME_FRACTION_THRESHOLD,
            "jump_robust_z_threshold": JUMP_ROBUST_Z_THRESHOLD,
            "line_length_outlier_robust_z_threshold": (
                LINE_LENGTH_OUTLIER_ROBUST_Z_THRESHOLD
            ),
        },
        "windows": WINDOWS,
        "frequency_bands": FREQUENCY_BANDS,
        "filter": {
            "method": FILTER_METHOD,
            "phase": FILTER_PHASE,
            "fir_window": FIR_WINDOW,
            "fir_design": FIR_DESIGN,
            "filter_length": FILTER_LENGTH,
            "l_trans_bandwidth": L_TRANS_BANDWIDTH,
            "h_trans_bandwidth": H_TRANS_BANDWIDTH,
            "pad": FILTER_PAD,
        },
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    (out_root / "line_length_matrix_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def print_plan(plan: pd.DataFrame, mode: str, out_root: Path) -> None:
    print(f"mode={mode} out={out_root}")
    print(f"n_matrix_tasks={len(plan)}")
    print("patients:")
    print(plan.groupby("patient")["trial"].nunique().to_string())
    print("\nwindows:")
    print(plan.groupby("window").size().to_string())
    print("\nbands:")
    print(plan.groupby("frequency_band").size().to_string())
    print("\npreview:")
    print(
        plan[
            [
                "patient",
                "trial",
                "window",
                "frequency_band",
                "expected_n_subwindows",
                "out_file",
            ]
        ]
        .head(18)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute raw bipolar SEEG line-length matrices")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--trials", nargs="*", type=int, default=None)
    parser.add_argument("--bands", nargs="*", default=None)
    parser.add_argument("--windows", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=LINE_LENGTH_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--heartbeat-sec", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise SystemExit("Use either --overwrite or --resume, not both")
    out_root = Path(args.out)
    plan = build_line_length_plan(
        FINAL_SIGNAL_ROOT,
        ONSET_MANIFEST,
        out_root,
        patients=args.patients,
        trials=args.trials,
        bands=args.bands,
        windows=args.windows,
    )
    print_plan(plan, "apply" if args.apply else "dry-run", out_root)
    if not args.apply:
        return
    for out_file in plan["out_file"].map(Path):
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite=args.overwrite, resume=args.resume)

    started_at = time.perf_counter()
    results: list[dict[str, object]] = []
    trial_groups = list(plan.groupby(["patient", "trial"], sort=True))
    for index, ((patient, trial), trial_plan) in enumerate(trial_groups, start=1):
        trial_started = time.perf_counter()
        print(
            f"START trial {index}/{len(trial_groups)} {patient} trial{int(trial):02d} "
            f"tasks={len(trial_plan)}",
            flush=True,
        )
        trial_results = compute_one_trial(
            trial_plan,
            overwrite=args.overwrite,
            resume=args.resume,
            heartbeat_sec=args.heartbeat_sec,
        )
        results.extend(trial_results)
        print(
            f"DONE  trial {index}/{len(trial_groups)} {patient} trial{int(trial):02d} "
            f"results={len(trial_results)} elapsed={time.perf_counter() - trial_started:.1f}s",
            flush=True,
        )
    write_run_outputs(results, out_root, started_at)
    print(f"manifest: {out_root / 'line_length_matrix_manifest.csv'}")
    print(f"summary: {out_root / 'line_length_matrix_run_summary.json'}")


if __name__ == "__main__":
    main()
