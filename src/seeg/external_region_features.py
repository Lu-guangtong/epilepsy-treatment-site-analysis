"""Native-rate connectivity and label-free cross-seizure regional features."""
from __future__ import annotations

import json
from pathlib import Path

import mne
from mne_connectivity import spectral_connectivity_epochs
import numpy as np
import pandas as pd

from .external_region_contract import (
    BANDS, KEYS, METHODS, WINDOWS, band_eligibility, candidate_features,
    load_analysis_module, sha256, write_json,
)


def load_signal(record: dict) -> mne.io.BaseRaw:
    path = Path(record["signal_path"])
    if sha256(path) != record["output_sha256"]:
        raise ValueError(f"Signal SHA256 mismatch: {path}")
    raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
    if (raw.first_samp != 0 or raw.first_time != 0
            or raw.ch_names != record["bipolar_channel_names"]
            or raw.n_times != record["n_samples"]
            or raw.info["sfreq"] != record["sampling_hz"]
            or raw.info["bads"]):
        raise ValueError(f"FIF header differs from frozen signal manifest: {path}")
    return raw


def subepochs(raw: mne.io.BaseRaw, window: str, band: str) -> np.ndarray:
    fs = float(raw.info["sfreq"])
    if not band_eligibility(fs, band)[0]:
        raise ValueError(f"Frequency band {band} is unsupported at {fs} Hz")
    start, stop = WINDOWS[window]
    duration = BANDS[band][2]
    width, step = int(round(fs * duration)), int(round(fs * duration / 2))
    first, last = int(round(start * fs)), int(round(stop * fs))
    if first < 0 or last > raw.n_times:
        raise ValueError("Required half-open analysis window exceeds the signal")
    data = raw.get_data(start=first, stop=last)
    if not np.isfinite(data).all() or np.any(np.ptp(data, axis=1) == 0):
        raise ValueError("Nonfinite or constant channel in a required window")
    epochs = np.stack([data[:, i:i + width] for i in range(0, last - first - width + 1, step)])
    return epochs - epochs.mean(axis=2, keepdims=True)


def reconstruct(dense: np.ndarray, method: str) -> np.ndarray:
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise ValueError("Connectivity matrix must be square")
    lower = np.tril(dense, -1)
    matrix = lower + lower.T if method == "wpli2_debiased" else lower - lower.T
    np.fill_diagonal(matrix, np.nan)
    validate_matrix(matrix, method)
    return matrix


def validate_matrix(matrix: np.ndarray, method: str) -> None:
    if method not in METHODS or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Invalid connectivity method or shape")
    mask = ~np.eye(len(matrix), dtype=bool)
    if not np.isfinite(matrix[mask]).all() or not np.isnan(np.diag(matrix)).all():
        raise ValueError("Invalid connectivity diagonal/off-diagonal values")
    transpose = matrix.T if method == "wpli2_debiased" else -matrix.T
    if not np.allclose(matrix[mask], transpose[mask], atol=1e-12):
        raise ValueError("Connectivity symmetry/antisymmetry is invalid")


def connectivity(epochs: np.ndarray, sampling_hz: float, band: str) -> dict[str, np.ndarray]:
    if not band_eligibility(sampling_hz, band)[0]:
        raise ValueError("Unsupported frequency band; clipping or resampling is forbidden")
    low, high, duration = BANDS[band]
    results = spectral_connectivity_epochs(
        epochs, method=list(METHODS), sfreq=sampling_hz, mode="multitaper",
        fmin=low, fmax=high, faverage=True, mt_bandwidth=8.0 / duration,
        mt_adaptive=False, mt_low_bias=True, n_jobs=1, block_size=1000,
        verbose="ERROR",
    )
    matrices = {}
    for method, con in zip(METHODS, results, strict=True):
        dense = con.get_data(output="dense")
        if dense.shape != (epochs.shape[1], epochs.shape[1], 1):
            raise ValueError(f"Unexpected connectivity output: {dense.shape}")
        matrices[method] = reconstruct(dense[:, :, 0], method)
    return matrices


def region_index(record: dict, anatomy: list[dict]) -> dict[tuple, list[int]]:
    by_pair = {r["bipolar_pair"]: r for r in anatomy if r["subject"] == record["patient"]}
    if set(by_pair) != set(record["bipolar_channel_names"]):
        raise ValueError("Anatomy must cover every network node exactly")
    regions = {}
    for index, name in enumerate(record["bipolar_channel_names"]):
        row = by_pair[name]
        key = (row["atlas_region"], row["tissue_class"], row["tissue_class"] != "gray")
        regions.setdefault(key, []).append(index)
    return regions


def extract_band(
    signals: list[dict], anatomy: list[dict], band: str, out: Path,
    fingerprint: str, resume: bool = False,
) -> dict:
    """All technically valid nodes contribute edges, including mixed/white pairs."""
    extractor = load_analysis_module("extract_region_network_features")
    eligible = [s for s in signals if band_eligibility(s["sampling_hz"], band)[0]]
    if not eligible:
        raise ValueError(f"No eligible signals for {band}")
    for patient in {r["patient"] for r in eligible}:
        if sum(r["patient"] == patient for r in eligible) < 2:
            raise ValueError(f"At least two qualified seizures required: {patient} {band}")
    out.mkdir(parents=True, exist_ok=True)
    rows, matrix_manifest = [], []
    for sequence, record in enumerate(eligible, 1):
        print(f"features {band} {record['patient']} run-{record['trial']:02d} "
              f"({sequence}/{len(eligible)})", flush=True)
        raw = None
        indices = region_index(record, anatomy)
        for window in WINDOWS:
            metadata = {"patient": record["patient"], "trial": record["trial"],
                        "frequency_band": band, "window": window}
            shared = {**metadata, "fingerprint": fingerprint,
                      "signal_sha256": record["output_sha256"],
                      "sampling_hz": record["sampling_hz"],
                      "fmin_hz": BANDS[band][0], "fmax_hz": BANDS[band][1]}
            paths = {m: out / "matrices" / record["patient"] /
                     f"run-{record['trial']:02d}_{window}_{m}.npz" for m in METHODS}
            computed = None
            for method in METHODS:
                path, expected = paths[method], {**shared, "method": method}
                if path.exists():
                    if not resume:
                        raise FileExistsError(f"Matrix already exists: {path}")
                    checksum = path.with_suffix(".sha256")
                    if not checksum.is_file() or checksum.read_text().strip() != sha256(path):
                        raise ValueError(f"Matrix resume SHA256 mismatch: {path}")
                    with np.load(path, allow_pickle=False) as saved:
                        if (json.loads(str(saved["metadata"].item())) != expected
                                or saved["channel_names"].tolist() != record["bipolar_channel_names"]):
                            raise ValueError(f"Matrix resume metadata mismatch: {path}")
                        matrix = saved["matrix"]
                    validate_matrix(matrix, method)
                else:
                    if computed is None:
                        if raw is None:
                            raw = load_signal(record)
                        computed = connectivity(subepochs(raw, window, band), record["sampling_hz"], band)
                    matrix = computed[method]
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temporary = path.with_name(path.name + ".tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, matrix=matrix,
                            channel_names=np.asarray(record["bipolar_channel_names"], dtype=str),
                            metadata=np.asarray(json.dumps(expected, sort_keys=True)))
                    temporary.replace(path)
                    path.with_suffix(".sha256").write_text(sha256(path), encoding="ascii")
                if matrix.shape != (record["n_channels"], record["n_channels"]):
                    raise ValueError("Network node count changed")
                matrix_manifest.append({**expected, "path": str(path.resolve()),
                                        "sha256": sha256(path)})
                transformed = extractor.transform_matrix_values(matrix, method)
                rows.extend(extractor.compute_region_feature_rows(
                    transformed, indices, {**metadata, "method": method}))
        if raw is not None:
            raw.close()
    trial = pd.DataFrame(rows)
    trial_keys = ["patient", "trial", *KEYS[1:], "window", "method", "feature_name"]
    if trial.duplicated(trial_keys).any():
        raise ValueError("Duplicate seizure-level feature rows")
    contrast = extractor.build_contrast_features(trial)
    patient = extractor.aggregate_patient_region_features(trial, contrast).sort_values(KEYS)
    candidates = candidate_features(band)
    if set(candidates) - set(patient.columns):
        raise ValueError("Incomplete formal candidate feature grid")
    allowed = set(KEYS + candidates)
    unknown = [c for c in patient if c not in allowed and not c.startswith("qc__")]
    if unknown or patient.duplicated(KEYS).any():
        raise ValueError(f"Unexpected or duplicate patient feature columns: {unknown}")
    files = {}
    for name, frame in [("region_trial_features_long", trial),
                        ("region_trial_contrast_features_long", contrast),
                        ("region_patient_features_unlabeled", patient)]:
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        files[name] = {"path": str(path.resolve()), "sha256": sha256(path), "rows": len(frame)}
    manifest = {"status": "complete", "frequency_band": band, "fingerprint": fingerprint,
                "patients": sorted(patient.patient.unique()), "n_signals": len(eligible),
                "patient_native_sampling_hz": {r["patient"]: r["sampling_hz"] for r in eligible},
                "signal_keys": [[r["patient"], r["trial"]] for r in eligible],
                "n_candidate_features": len(candidates), "files": files,
                "matrices": matrix_manifest, "treatment_labels_read": False,
                "network_nodes": "all_technically_valid_common_bipolar_pairs",
                "region_rank_scope": "all_mapped_regions_as_in_formal_feature_extractor"}
    write_json(out / "feature_manifest.json", manifest)
    return manifest
