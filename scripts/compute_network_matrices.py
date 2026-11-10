"""Compute formal SEEG connectivity matrices from final bipolar FIF segments.

This is the S5 network-matrix step only. It reads final preprocessed bipolar
FIF segments, uses segment-relative onset times to cut fixed windows, computes
predefined spectral connectivity matrices, and writes one matrix per
patient/trial/window/frequency/method. It does not read ablation labels, does
not aggregate by brain region, and does not model outcomes.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import mne
import mne_connectivity
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
FINAL_SIGNAL_ROOT = ROOT / "data_derived" / "custom_notched"
ONSET_MANIFEST = ROOT / "data_derived" / "resampled" / "resample_manifest.csv"
NETWORK_ROOT = ROOT / "data_derived" / "network_matrices"
CHECK_ROOT = ROOT / "checks"
TARGET_SFREQ_HZ = 2000.0
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
    "delta": {"fmin_hz": 1.0, "fmax_hz": 4.0, "sub_epoch_length_sec": 5.0, "overlap_fraction": 0.5},
    "theta": {"fmin_hz": 4.0, "fmax_hz": 8.0, "sub_epoch_length_sec": 5.0, "overlap_fraction": 0.5},
    "alpha": {"fmin_hz": 8.0, "fmax_hz": 13.0, "sub_epoch_length_sec": 5.0, "overlap_fraction": 0.5},
    "beta": {"fmin_hz": 13.0, "fmax_hz": 30.0, "sub_epoch_length_sec": 5.0, "overlap_fraction": 0.5},
    "low_gamma": {"fmin_hz": 30.0, "fmax_hz": 80.0, "sub_epoch_length_sec": 2.0, "overlap_fraction": 0.5},
    "high_gamma": {"fmin_hz": 80.0, "fmax_hz": 150.0, "sub_epoch_length_sec": 2.0, "overlap_fraction": 0.5},
    "ripple": {"fmin_hz": 80.0, "fmax_hz": 250.0, "sub_epoch_length_sec": 2.0, "overlap_fraction": 0.5},
    "fast_ripple": {"fmin_hz": 250.0, "fmax_hz": 500.0, "sub_epoch_length_sec": 2.0, "overlap_fraction": 0.5},
}
METHODS = ("wpli2_debiased", "imcoh")
CONNECTIVITY_MODE = "multitaper"
FAVERAGE = True
MT_ADAPTIVE = False
MT_LOW_BIAS = True
N_JOBS = 1
BLOCK_SIZE = 1000
MATRIX_MODE = "all_final_bipolar_channels_no_region_aggregation"


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
    network_root = _resolve(NETWORK_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if out_dir == AUX_ROOT or _is_relative_to(out_dir, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {out_dir}")
    if not (
        out_dir == network_root
        or _is_relative_to(out_dir, network_root)
        or out_dir == tmp_root
        or _is_relative_to(out_dir, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {out_dir}")


def assert_output_available(out_file: Path, overwrite: bool, resume: bool) -> None:
    if out_file.exists() and not (overwrite or resume):
        raise SystemExit(f"Output already exists; pass --overwrite or --resume: {out_file}")


def _normalize_patients(patients: Iterable[str] | None) -> set[str] | None:
    if patients is None:
        return None
    normalized = {patient.strip().lower() for patient in patients if patient.strip()}
    return normalized or None


def _normalize_ints(values: Iterable[int] | None) -> set[int] | None:
    if values is None:
        return None
    return {int(value) for value in values}


def _normalize_names(values: Iterable[str] | None, allowed: Iterable[str], label: str) -> set[str] | None:
    if values is None:
        return None
    allowed_set = set(allowed)
    normalized = {value.strip() for value in values if value.strip()}
    unknown = sorted(normalized - allowed_set)
    if unknown:
        raise SystemExit(f"Unknown {label}: {unknown}; allowed={sorted(allowed_set)}")
    return normalized or None


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
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"{onset_manifest}: missing required onset columns: {sorted(required)}")
        for row in reader:
            key = (str(row["patient"]).strip().lower(), int(row["trial"]))
            if key in onsets:
                raise SystemExit(f"{onset_manifest}: duplicate onset row for {key[0]} trial {key[1]:02d}")
            onsets[key] = float(row["onset_sec"])
    return onsets


def discover_signal_files(
    signals_root: Path = FINAL_SIGNAL_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
) -> list[dict[str, object]]:
    patient_filter = _normalize_patients(patients)
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


def mt_bandwidth_for_epoch(sub_epoch_length_sec: float) -> float:
    return 8.0 / float(sub_epoch_length_sec)


def _expected_subepoch_count(window_duration_sec: float, sub_epoch_length_sec: float, overlap_fraction: float) -> int:
    step = sub_epoch_length_sec * (1.0 - overlap_fraction)
    return int(np.floor((window_duration_sec - sub_epoch_length_sec) / step + 1.0 + 1e-9))


def build_matrix_plan(
    signals_root: Path = FINAL_SIGNAL_ROOT,
    onset_manifest: Path = ONSET_MANIFEST,
    out_root: Path = NETWORK_ROOT,
    patients: Iterable[str] | None = None,
    trials: Iterable[int] | None = None,
    bands: Iterable[str] | None = None,
    methods: Iterable[str] | None = None,
    windows: Iterable[str] | None = None,
) -> pd.DataFrame:
    assert_allowed_output_root(Path(out_root))
    onsets = load_onsets(onset_manifest)
    signal_rows = discover_signal_files(signals_root, patients=patients, trials=trials)
    if not signal_rows:
        raise SystemExit(f"No final FIF sources found under {signals_root}")

    patient_filter = _normalize_patients(patients)
    trial_filter = _normalize_ints(trials)
    band_filter = _normalize_names(bands, FREQUENCY_BANDS, "frequency bands")
    method_filter = _normalize_names(methods, METHODS, "methods")
    window_filter = _normalize_names(windows, WINDOWS, "windows")
    selected_bands = [name for name in FREQUENCY_BANDS if band_filter is None or name in band_filter]
    selected_methods = [name for name in METHODS if method_filter is None or name in method_filter]
    selected_windows = [name for name in WINDOWS if window_filter is None or name in window_filter]

    duplicate_keys = [
        (row["patient"], row["trial"])
        for row in signal_rows
        if sum(1 for item in signal_rows if item["patient"] == row["patient"] and item["trial"] == row["trial"]) > 1
    ]
    if duplicate_keys:
        raise SystemExit(f"Duplicate final FIF sources: {sorted(set(duplicate_keys))}")

    if patient_filter is None and trial_filter is None:
        counts: dict[str, int] = {}
        for row in signal_rows:
            counts[str(row["patient"])] = counts.get(str(row["patient"]), 0) + 1
        if counts != EXPECTED_FULL_COUNTS:
            raise SystemExit(f"Unexpected final FIF counts: {counts}")

    records: list[dict[str, object]] = []
    out_root = Path(out_root)
    for signal_row in sorted(signal_rows, key=lambda item: (str(item["patient"]), int(item["trial"]))):
        patient = str(signal_row["patient"])
        trial = int(signal_row["trial"])
        onset = onsets.get((patient, trial))
        if onset is None:
            raise SystemExit(f"Missing onset for {patient} trial {trial:02d}")
        for window_name in selected_windows:
            start_rel, end_rel = WINDOWS[window_name]
            window_start = float(onset) + start_rel
            window_end = float(onset) + end_rel
            if window_start < 0 or window_end <= window_start:
                raise SystemExit(
                    f"{patient} trial {trial:02d} invalid window {window_name}: "
                    f"{window_start} to {window_end}"
                )
            window_duration = window_end - window_start
            for band_name in selected_bands:
                band = FREQUENCY_BANDS[band_name]
                n_sub_epochs = _expected_subepoch_count(
                    window_duration,
                    float(band["sub_epoch_length_sec"]),
                    float(band["overlap_fraction"]),
                )
                for method in selected_methods:
                    out_file = (
                        out_root
                        / patient
                        / f"{patient}_trial{trial:02d}_{window_name}_{band_name}_{method}.npz"
                    )
                    records.append(
                        {
                            "patient": patient,
                            "trial": trial,
                            "source_fif": str(signal_row["source_fif"]),
                            "out_file": str(out_file),
                            "window": window_name,
                            "window_start_sec": window_start,
                            "window_end_sec": window_end,
                            "frequency_band": band_name,
                            "fmin_hz": float(band["fmin_hz"]),
                            "fmax_hz": float(band["fmax_hz"]),
                            "sub_epoch_length_sec": float(band["sub_epoch_length_sec"]),
                            "overlap_fraction": float(band["overlap_fraction"]),
                            "expected_n_sub_epochs": int(n_sub_epochs),
                            "method": method,
                            "onset_sec": float(onset),
                            "mt_bandwidth": mt_bandwidth_for_epoch(float(band["sub_epoch_length_sec"])),
                            "mode": CONNECTIVITY_MODE,
                            "faverage": FAVERAGE,
                            "mt_adaptive": MT_ADAPTIVE,
                            "mt_low_bias": MT_LOW_BIAS,
                            "n_jobs": N_JOBS,
                            "block_size": BLOCK_SIZE,
                            "matrix_mode": MATRIX_MODE,
                            "mne_version": mne.__version__,
                            "mne_connectivity_version": mne_connectivity.__version__,
                        }
                    )
    return pd.DataFrame.from_records(records)


def subepoch_sample_slices(
    window_start_sec: float,
    window_end_sec: float,
    sfreq: float,
    sub_epoch_length_sec: float,
    overlap_fraction: float,
) -> list[tuple[int, int]]:
    if not 0.0 <= overlap_fraction < 1.0:
        raise SystemExit(f"Invalid overlap_fraction: {overlap_fraction}")
    if window_end_sec <= window_start_sec:
        raise SystemExit(f"Invalid window: {window_start_sec} to {window_end_sec}")
    if sub_epoch_length_sec <= 0:
        raise SystemExit(f"Invalid sub_epoch_length_sec: {sub_epoch_length_sec}")

    step_sec = sub_epoch_length_sec * (1.0 - overlap_fraction)
    n_samples = int(round(sub_epoch_length_sec * sfreq))
    step_samples = int(round(step_sec * sfreq))
    first_start = int(round(window_start_sec * sfreq))
    last_start = int(round((window_end_sec - sub_epoch_length_sec) * sfreq))
    starts = list(range(first_start, last_start + 1, step_samples))
    return [(start, start + n_samples) for start in starts]


def extract_demeaned_subepochs(raw: mne.io.BaseRaw, slices: list[tuple[int, int]]) -> np.ndarray:
    if not slices:
        raise SystemExit("No sub-epochs available for connectivity estimation")
    epochs = np.empty((len(slices), len(raw.ch_names), slices[0][1] - slices[0][0]), dtype=np.float64)
    for idx, (start, stop) in enumerate(slices):
        if start < 0 or stop > raw.n_times or stop <= start:
            raise SystemExit(f"Invalid sub-epoch sample slice: {start}:{stop}, n_times={raw.n_times}")
        data = raw.get_data(start=start, stop=stop).astype(np.float64, copy=False)
        epochs[idx] = data - data.mean(axis=1, keepdims=True)
    return epochs


def reconstruct_connectivity_matrix(mne_dense: np.ndarray, method: str) -> np.ndarray:
    dense = np.asarray(mne_dense, dtype=np.float64)
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise SystemExit(f"Connectivity output is not a square matrix: {dense.shape}")
    lower = np.tril(dense, k=-1)
    if method == "wpli2_debiased":
        matrix = lower + lower.T
    elif method == "imcoh":
        matrix = lower - lower.T
    else:
        raise SystemExit(f"Unsupported method for matrix reconstruction: {method}")
    np.fill_diagonal(matrix, np.nan)
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if not np.isfinite(off_diagonal).all():
        raise SystemExit(f"Non-finite off-diagonal connectivity values for method={method}")
    return matrix


def _compute_connectivity_dense(epochs: np.ndarray, record: dict[str, object]) -> np.ndarray:
    con = spectral_connectivity_epochs(
        epochs,
        method=str(record["method"]),
        sfreq=TARGET_SFREQ_HZ,
        mode=CONNECTIVITY_MODE,
        fmin=float(record["fmin_hz"]),
        fmax=float(record["fmax_hz"]),
        faverage=FAVERAGE,
        mt_bandwidth=float(record.get("mt_bandwidth", mt_bandwidth_for_epoch(float(record["sub_epoch_length_sec"])))),
        mt_adaptive=MT_ADAPTIVE,
        mt_low_bias=MT_LOW_BIAS,
        n_jobs=N_JOBS,
        block_size=BLOCK_SIZE,
        verbose="ERROR",
    )
    dense = con.get_data(output="dense")
    if dense.ndim != 3 or dense.shape[2] != 1:
        raise SystemExit(f"Unexpected MNE connectivity output shape: {dense.shape}")
    return dense[:, :, 0]


def _run_with_heartbeat(
    fn: Callable[[], np.ndarray],
    label: str,
    heartbeat_sec: int,
) -> np.ndarray:
    if heartbeat_sec <= 0:
        return fn()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(fn)
        while not future.done():
            try:
                return future.result(timeout=heartbeat_sec)
            except TimeoutError:
                elapsed = time.perf_counter() - start
                print(f"  still running {label}; elapsed={elapsed:.1f}s", flush=True)
        return future.result()


def assert_zero_timebase(raw: mne.io.BaseRaw, source: Path) -> None:
    if int(raw.first_samp) != 0 or float(raw.first_time) != 0.0:
        raise SystemExit(
            f"{source}: source FIF is not an independent segment "
            f"(first_samp={raw.first_samp}, first_time={raw.first_time})"
        )


def _np_scalar(value: object) -> np.ndarray:
    return np.asarray(value)


def save_matrix_npz(
    out_file: Path,
    matrix: np.ndarray,
    channel_names: list[str],
    record: dict[str, object],
    raw: mne.io.BaseRaw,
    n_sub_epochs: int,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = out_file.with_name(out_file.name + ".tmp")
    with tmp_file.open("wb") as handle:
        np.savez_compressed(
            handle,
            matrix=matrix,
            channel_names=np.asarray(channel_names, dtype=str),
            patient=_np_scalar(record["patient"]),
            trial=_np_scalar(int(record["trial"])),
            window=_np_scalar(record["window"]),
            frequency_band=_np_scalar(record["frequency_band"]),
            method=_np_scalar(record["method"]),
            sfreq_hz=_np_scalar(float(raw.info["sfreq"])),
            onset_sec=_np_scalar(float(record["onset_sec"])),
            window_start_sec=_np_scalar(float(record["window_start_sec"])),
            window_end_sec=_np_scalar(float(record["window_end_sec"])),
            sub_epoch_length_sec=_np_scalar(float(record["sub_epoch_length_sec"])),
            sub_epoch_overlap_fraction=_np_scalar(float(record["overlap_fraction"])),
            n_sub_epochs=_np_scalar(int(n_sub_epochs)),
            fmin_hz=_np_scalar(float(record["fmin_hz"])),
            fmax_hz=_np_scalar(float(record["fmax_hz"])),
            mode=_np_scalar(CONNECTIVITY_MODE),
            mt_bandwidth=_np_scalar(
                float(record.get("mt_bandwidth", mt_bandwidth_for_epoch(float(record["sub_epoch_length_sec"]))))
            ),
            mt_adaptive=_np_scalar(MT_ADAPTIVE),
            mt_low_bias=_np_scalar(MT_LOW_BIAS),
            faverage=_np_scalar(FAVERAGE),
            mne_version=_np_scalar(mne.__version__),
            mne_connectivity_version=_np_scalar(mne_connectivity.__version__),
            source_fif=_np_scalar(record["source_fif"]),
            matrix_mode=_np_scalar(MATRIX_MODE),
        )
    tmp_file.replace(out_file)


def _npz_matches_record(out_file: Path, record: dict[str, object]) -> bool:
    try:
        saved = np.load(out_file, allow_pickle=False)
        checks = [
            str(saved["patient"].item()) == str(record["patient"]),
            int(saved["trial"].item()) == int(record["trial"]),
            str(saved["window"].item()) == str(record["window"]),
            str(saved["frequency_band"].item()) == str(record["frequency_band"]),
            str(saved["method"].item()) == str(record["method"]),
            float(saved["fmin_hz"].item()) == float(record["fmin_hz"]),
            float(saved["fmax_hz"].item()) == float(record["fmax_hz"]),
            float(saved["sub_epoch_length_sec"].item()) == float(record["sub_epoch_length_sec"]),
            float(saved["sub_epoch_overlap_fraction"].item()) == float(record["overlap_fraction"]),
        ]
        return all(checks)
    except Exception:
        return False


def compute_one_matrix(
    record: dict[str, object] | pd.Series,
    overwrite: bool = False,
    resume: bool = False,
    heartbeat_sec: int = 60,
) -> dict[str, object]:
    if isinstance(record, pd.Series):
        record = record.to_dict()
    source = Path(str(record["source_fif"]))
    out_file = Path(str(record["out_file"]))
    assert_allowed_output_root(out_file.parent)
    assert_output_available(out_file, overwrite=overwrite, resume=resume)
    if out_file.exists() and resume and not overwrite:
        if _npz_matches_record(out_file, record):
            result = record.copy()
            result.update({"qc_status": "skipped_existing", "qc_notes": "resume metadata matched"})
            return result
        raise SystemExit(f"Existing output does not match planned metadata: {out_file}")
    if not source.exists():
        raise SystemExit(f"Missing source FIF: {source}")

    raw = mne.io.read_raw_fif(source, preload=True, verbose="ERROR")
    assert_zero_timebase(raw, source)
    sfreq = float(raw.info["sfreq"])
    if sfreq != TARGET_SFREQ_HZ:
        raise SystemExit(f"Unexpected sampling rate in {source}: {sfreq}")

    slices = subepoch_sample_slices(
        float(record["window_start_sec"]),
        float(record["window_end_sec"]),
        sfreq,
        float(record["sub_epoch_length_sec"]),
        float(record["overlap_fraction"]),
    )
    expected = int(record["expected_n_sub_epochs"])
    if len(slices) != expected:
        raise SystemExit(f"Unexpected sub-epoch count: {len(slices)} != {expected} for {out_file}")

    epochs = extract_demeaned_subepochs(raw, slices)
    label = (
        f"{record['patient']} trial{int(record['trial']):02d} "
        f"{record['window']} {record['frequency_band']} {record['method']}"
    )
    dense = _run_with_heartbeat(lambda: _compute_connectivity_dense(epochs, record), label, heartbeat_sec)
    matrix = reconstruct_connectivity_matrix(dense, str(record["method"]))
    if matrix.shape != (len(raw.ch_names), len(raw.ch_names)):
        raise SystemExit(f"Output matrix shape mismatch for {out_file}: {matrix.shape}")
    save_matrix_npz(out_file, matrix, raw.ch_names, record, raw, len(slices))
    result = record.copy()
    result.update(
        {
            "qc_status": "computed",
            "qc_notes": "",
            "n_channels": len(raw.ch_names),
            "n_sub_epochs": len(slices),
            "matrix_shape": f"{matrix.shape[0]}x{matrix.shape[1]}",
        }
    )
    return result


def validate_plan_outputs(plan: pd.DataFrame, overwrite: bool, resume: bool) -> None:
    for out_file in plan["out_file"].map(Path):
        assert_allowed_output_root(out_file.parent)
        assert_output_available(out_file, overwrite=overwrite, resume=resume)


def write_run_outputs(results: list[dict[str, object]], out_root: Path, started_at: float) -> None:
    out_root = Path(out_root)
    assert_allowed_output_root(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame.from_records(results)
    manifest.to_csv(out_root / "network_matrix_manifest.csv", index=False, encoding="utf-8-sig")
    summary = {
        "n_results": len(results),
        "qc_status_counts": manifest["qc_status"].value_counts().to_dict() if not manifest.empty else {},
        "elapsed_sec": time.perf_counter() - started_at,
        "mne_version": mne.__version__,
        "mne_connectivity_version": mne_connectivity.__version__,
        "methods": list(METHODS),
        "windows": WINDOWS,
        "frequency_bands": FREQUENCY_BANDS,
        "connectivity_mode": CONNECTIVITY_MODE,
        "mt_adaptive": MT_ADAPTIVE,
        "mt_low_bias": MT_LOW_BIAS,
        "faverage": FAVERAGE,
        "n_jobs": N_JOBS,
        "block_size": BLOCK_SIZE,
    }
    (out_root / "network_matrix_run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
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
    print("\nmethods:")
    print(plan.groupby("method").size().to_string())
    preview_cols = ["patient", "trial", "window", "frequency_band", "method", "expected_n_sub_epochs", "out_file"]
    print("\npreview:")
    print(plan[preview_cols].head(16).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute formal SEEG spectral connectivity matrices")
    parser.add_argument("--apply", action="store_true", help="write network matrix NPZ files")
    parser.add_argument("--patients", nargs="*", default=None)
    parser.add_argument("--trials", nargs="*", type=int, default=None)
    parser.add_argument("--bands", nargs="*", default=None)
    parser.add_argument("--methods", nargs="*", default=None)
    parser.add_argument("--windows", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=NETWORK_ROOT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--heartbeat-sec", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.overwrite and args.resume:
        raise SystemExit("Use either --overwrite or --resume, not both")
    out_root = Path(args.out)
    plan = build_matrix_plan(
        FINAL_SIGNAL_ROOT,
        ONSET_MANIFEST,
        out_root,
        patients=args.patients,
        trials=args.trials,
        bands=args.bands,
        methods=args.methods,
        windows=args.windows,
    )
    print_plan(plan, "apply" if args.apply else "dry-run", out_root)
    if not args.apply:
        return

    validate_plan_outputs(plan, overwrite=args.overwrite, resume=args.resume)
    started_at = time.perf_counter()
    results: list[dict[str, object]] = []
    total = len(plan)
    for idx, (_, row) in enumerate(plan.iterrows(), start=1):
        label = (
            f"{row['patient']} trial{int(row['trial']):02d} "
            f"{row['window']} {row['frequency_band']} {row['method']}"
        )
        matrix_started = time.perf_counter()
        print(f"START {idx}/{total} {label}", flush=True)
        result = compute_one_matrix(row, overwrite=args.overwrite, resume=args.resume, heartbeat_sec=args.heartbeat_sec)
        elapsed = time.perf_counter() - matrix_started
        print(
            f"DONE  {idx}/{total} {label} status={result['qc_status']} elapsed={elapsed:.1f}s",
            flush=True,
        )
        results.append(result)
    write_run_outputs(results, out_root, started_at)
    print(f"manifest: {out_root / 'network_matrix_manifest.csv'}")
    print(f"summary: {out_root / 'network_matrix_run_summary.json'}")


if __name__ == "__main__":
    main()
