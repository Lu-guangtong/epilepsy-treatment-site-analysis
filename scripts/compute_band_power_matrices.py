"""Compute raw subwindow-level absolute band-power matrices for bipolar SEEG.

Every value is obtained from a DPSS multitaper PSD integrated over one target
frequency band. Outputs retain every channel and subwindow; no logarithm,
relative normalization, baseline contrast, trial aggregation, or labels are
used in this stage.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import csv
import gc
import io
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
from mne.time_frequency import psd_array_multitaper


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
FINAL_SIGNAL_ROOT = ROOT / "data_derived" / "custom_notched"
ONSET_MANIFEST = ROOT / "data_derived" / "resampled" / "resample_manifest.csv"
BAND_POWER_ROOT = ROOT / "data_derived" / "electrode_metrics_raw" / "band_power"
CHECK_ROOT = ROOT / "checks"
TARGET_SFREQ_HZ = 2000.0
NPZ_WRITE_MAX_ATTEMPTS = 3
NPZ_WRITE_RETRY_SEC = 5.0
METHOD = "absolute_band_power_multitaper"
MATRIX_MODE = "all_bipolar_channels_by_subwindow_no_aggregation"
VALUE_FORMULA = "trapezoidal_integral(multitaper_PSD, fmin_hz, fmax_hz)"
VALUE_UNIT = "V2"
PSD_UNIT = "V2_per_Hz"
MT_ADAPTIVE = False
MT_LOW_BIAS = True
MT_NORMALIZATION = "full"
MT_REMOVE_DC = True
MT_OUTPUT = "power"
MT_N_JOBS = 1
MT_MAX_ITER = 150
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
    formal_root = _resolve(BAND_POWER_ROOT)
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


def multitaper_bandwidth(subwindow_length_sec: float) -> float:
    if subwindow_length_sec <= 0.0:
        raise SystemExit(f"Invalid subwindow duration: {subwindow_length_sec}")
    return 8.0 / subwindow_length_sec


def expected_subwindow_count(
    window_duration_sec: float,
    subwindow_length_sec: float,
    overlap_fraction: float,
) -> int:
    step = subwindow_length_sec * (1.0 - overlap_fraction)
    return int(np.floor((window_duration_sec - subwindow_length_sec) / step + 1.0 + 1e-9))


def build_band_power_plan(
    signals_root: Path = FINAL_SIGNAL_ROOT,
    onset_manifest: Path = ONSET_MANIFEST,
    out_root: Path = BAND_POWER_ROOT,
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
                    / f"{patient}_trial{trial:02d}_{window_name}_{band_name}_absolute_power_raw.npz"
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
                        "mt_bandwidth_hz": multitaper_bandwidth(length),
                        "mt_adaptive": MT_ADAPTIVE,
                        "mt_low_bias": MT_LOW_BIAS,
                        "mt_normalization": MT_NORMALIZATION,
                        "mt_remove_dc": MT_REMOVE_DC,
                        "mt_output": MT_OUTPUT,
                        "value_formula": VALUE_FORMULA,
                        "value_unit": VALUE_UNIT,
                        "psd_unit": PSD_UNIT,
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


def compute_absolute_power_matrix(
    full_trial_data: np.ndarray,
    slices: list[tuple[int, int]],
    sfreq: float,
    fmin_hz: float,
    fmax_hz: float,
    bandwidth_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(full_trial_data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < 1 or data.shape[1] < 2 or not slices:
        raise SystemExit("Band power requires channels x samples data and at least one subwindow")
    if not np.isfinite(data).all():
        raise SystemExit("Input signal contains NaN or Inf")
    if not 0.0 < fmin_hz < fmax_hz < sfreq / 2.0 or bandwidth_hz <= 0.0:
        raise SystemExit("Invalid multitaper frequency parameters")

    matrix = np.empty((data.shape[0], len(slices)), dtype=np.float64)
    reference_freqs: np.ndarray | None = None
    for column, (start, stop) in enumerate(slices):
        if start < 0 or stop > data.shape[1] or stop - start < 2:
            raise SystemExit(f"Invalid band-power slice {start}:{stop}")
        segment = data[:, start:stop]
        psd, freqs = psd_array_multitaper(
            segment,
            sfreq=sfreq,
            fmin=fmin_hz,
            fmax=fmax_hz,
            bandwidth=bandwidth_hz,
            adaptive=MT_ADAPTIVE,
            low_bias=MT_LOW_BIAS,
            normalization=MT_NORMALIZATION,
            remove_dc=MT_REMOVE_DC,
            output=MT_OUTPUT,
            n_jobs=MT_N_JOBS,
            max_iter=MT_MAX_ITER,
            verbose="ERROR",
        )
        psd = np.asarray(psd, dtype=np.float64)
        freqs = np.asarray(freqs, dtype=np.float64)
        if psd.shape != (data.shape[0], freqs.size) or freqs.size < 2:
            raise SystemExit(f"Unexpected multitaper PSD shape: {psd.shape}, freqs={freqs.size}")
        if not np.isfinite(psd).all() or np.any(psd < 0.0) or not np.isfinite(freqs).all():
            raise SystemExit("Multitaper PSD contains invalid values")
        if reference_freqs is None:
            reference_freqs = freqs
        elif not np.array_equal(freqs, reference_freqs):
            raise SystemExit("Multitaper frequency grid changed across equal-length subwindows")
        matrix[:, column] = np.trapezoid(psd, freqs, axis=-1)

    if reference_freqs is None:
        raise SystemExit("No multitaper frequency grid was generated")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise SystemExit("Absolute-power matrix contains invalid values")
    if not np.any(matrix > 0.0):
        raise SystemExit("Absolute-power matrix is entirely zero")
    return matrix, reference_freqs


def _run_with_heartbeat(
    fn: Callable[[], tuple[np.ndarray, np.ndarray]],
    label: str,
    heartbeat_sec: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    frequency_bins_hz: np.ndarray,
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
    buffer = io.BytesIO()
    with buffer:
        np.savez_compressed(
            buffer,
            matrix=matrix,
            frequency_bins_hz=np.asarray(frequency_bins_hz, dtype=np.float64),
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
            mt_bandwidth_hz=_np_scalar(float(record["mt_bandwidth_hz"])),
            mt_adaptive=_np_scalar(MT_ADAPTIVE),
            mt_low_bias=_np_scalar(MT_LOW_BIAS),
            mt_normalization=_np_scalar(MT_NORMALIZATION),
            mt_remove_dc=_np_scalar(MT_REMOVE_DC),
            mt_output=_np_scalar(MT_OUTPUT),
            mt_n_jobs=_np_scalar(MT_N_JOBS),
            mt_max_iter=_np_scalar(MT_MAX_ITER),
            value_formula=_np_scalar(VALUE_FORMULA),
            value_unit=_np_scalar(VALUE_UNIT),
            psd_unit=_np_scalar(PSD_UNIT),
            matrix_mode=_np_scalar(MATRIX_MODE),
            source_fif=_np_scalar(record["source_fif"]),
            mne_version=_np_scalar(mne.__version__),
            numpy_version=_np_scalar(np.__version__),
            scipy_version=_np_scalar(scipy.__version__),
        )
        compressed = buffer.getvalue()

    for attempt in range(1, NPZ_WRITE_MAX_ATTEMPTS + 1):
        try:
            tmp_file.write_bytes(compressed)
            tmp_file.replace(out_file)
            return
        except OSError:
            tmp_file.unlink(missing_ok=True)
            if attempt == NPZ_WRITE_MAX_ATTEMPTS:
                raise
            time.sleep(NPZ_WRITE_RETRY_SEC)


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
                    float(saved["fmin_hz"].item()) == float(record["fmin_hz"]),
                    float(saved["fmax_hz"].item()) == float(record["fmax_hz"]),
                    float(saved["mt_bandwidth_hz"].item())
                    == float(record["mt_bandwidth_hz"]),
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
        frequency_bins_hz = np.asarray(saved["frequency_bins_hz"], dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise SystemExit(f"Existing output contains invalid power data: {out_file}")
    if frequency_bins_hz.ndim != 1 or not np.isfinite(frequency_bins_hz).all():
        raise SystemExit(f"Existing output contains invalid frequency bins: {out_file}")
    return {
        **record,
        "qc_status": "computed",
        "qc_notes": "existing output validated during resume",
        "n_channels": matrix.shape[0],
        "n_subwindows": matrix.shape[1],
        "n_frequency_bins": frequency_bins_hz.size,
        "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        "matrix_min": float(matrix.min()),
        "matrix_max": float(matrix.max()),
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
    pending: list[dict[str, object]] = []
    for record in records:
        out_file = Path(str(record["out_file"]))
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite=overwrite, resume=resume)
        if out_file.exists() and resume and not overwrite:
            if not _npz_matches_record(out_file, record):
                raise SystemExit(f"Existing output metadata mismatch: {out_file}")
            results.append(_existing_result(record, out_file))
        else:
            pending.append(record)
    if not pending:
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
    for record in sorted(pending, key=lambda item: (str(item["frequency_band"]), str(item["window"]))):
        slices = subwindow_sample_slices(
            float(record["window_start_sec"]),
            float(record["window_end_sec"]),
            sfreq,
            float(record["subwindow_length_sec"]),
            float(record["overlap_fraction"]),
        )
        if len(slices) != int(record["expected_n_subwindows"]):
            raise SystemExit(f"Unexpected subwindow count for {record['out_file']}")
        matrix, freqs = _run_with_heartbeat(
            lambda current=record, current_slices=slices: compute_absolute_power_matrix(
                data,
                current_slices,
                sfreq,
                float(current["fmin_hz"]),
                float(current["fmax_hz"]),
                float(current["mt_bandwidth_hz"]),
            ),
            f"{patient} trial{trial:02d} {record['window']} {record['frequency_band']} multitaper",
            heartbeat_sec,
        )
        expected_shape = (len(channel_names), len(slices))
        if matrix.shape != expected_shape:
            raise SystemExit(f"Output matrix shape mismatch: {matrix.shape} != {expected_shape}")
        out_file = Path(str(record["out_file"]))
        save_matrix_npz(out_file, matrix, freqs, channel_names, slices, record, sfreq)
        results.append(
            {
                **record,
                "qc_status": "computed",
                "qc_notes": "",
                "n_channels": len(channel_names),
                "n_subwindows": len(slices),
                "n_frequency_bins": freqs.size,
                "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
                "matrix_min": float(matrix.min()),
                "matrix_max": float(matrix.max()),
            }
        )
        del matrix, freqs
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
        out_root / "band_power_matrix_manifest.csv", index=False, encoding="utf-8-sig"
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
        "psd_unit": PSD_UNIT,
        "matrix_mode": MATRIX_MODE,
        "windows": WINDOWS,
        "frequency_bands": FREQUENCY_BANDS,
        "multitaper": {
            "bandwidth_formula": "8 / subwindow_duration_sec",
            "adaptive": MT_ADAPTIVE,
            "low_bias": MT_LOW_BIAS,
            "normalization": MT_NORMALIZATION,
            "remove_dc": MT_REMOVE_DC,
            "output": MT_OUTPUT,
            "n_jobs": MT_N_JOBS,
            "max_iter": MT_MAX_ITER,
        },
        "mne_version": mne.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
    }
    (out_root / "band_power_matrix_run_summary.json").write_text(
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
                "mt_bandwidth_hz",
                "out_file",
            ]
        ]
        .head(18)
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute raw bipolar SEEG multitaper absolute-power matrices"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--trials", nargs="*", type=int, default=None)
    parser.add_argument("--bands", nargs="*", default=None)
    parser.add_argument("--windows", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=BAND_POWER_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--heartbeat-sec", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise SystemExit("Use either --overwrite or --resume, not both")
    out_root = Path(args.out)
    plan = build_band_power_plan(
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
    print(f"manifest: {out_root / 'band_power_matrix_manifest.csv'}")
    print(f"summary: {out_root / 'band_power_matrix_run_summary.json'}")


if __name__ == "__main__":
    main()
