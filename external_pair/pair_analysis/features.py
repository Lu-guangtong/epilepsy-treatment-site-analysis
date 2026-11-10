"""Recording-level local features and within-patient seizure aggregation."""

from __future__ import annotations


from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
from .utils import trapezoid
from .signal import (
    _apply_notch_filter,
    _import_mne,
    _resample_poly,
    compatible_bands,
    starts_for_window,
)


_MISSING_PATH_VALUES = frozenset(
    {"", ".", "..", "nan", "none", "null", "n/a", "na", "<na>"}
)


def _path_text(value: Any) -> str:
    """Normalise manifest path tokens without resolving sentinels to CWD."""

    text = "" if value is None else str(value).strip()
    return "" if text.lower() in _MISSING_PATH_VALUES else text


def _record_path(value: Any) -> Path | None:
    """Resolve a record path while keeping CSV missing sentinels missing."""

    text = _path_text(value)
    if not text:
        return None
    return Path(text).expanduser().resolve()


WINDOW_FEATURES = [
    "log_mean",
    "log_median",
    "log_q75",
    "log_q90",
    "log_iqr",
    "log_mad",
    "log_peak",
    "log_peak_prominence",
    "log_theil_sen_slope_per_sec",
]


CONTRAST_FEATURES = [
    "median_log10_change",
    "q90_log10_change",
    "peak_log10_change",
    "robust_median_shift",
    "early_exceed_baseline_q95_fraction",
    "early_positive_excess_over_baseline_q95_mean",
    "log_iqr_change",
    "log_mad_change",
    "log_slope_change",
    "early_peak_latency_fraction",
]


RANK_SOURCES = [
    "early__log_median",
    "early__log_q90",
    "contrast__median_log10_change",
    "contrast__robust_median_shift",
    "contrast__early_positive_excess_over_baseline_q95_mean",
]


DIRECTIONAL_CONTRASTS = [
    "median_log10_change",
    "q90_log10_change",
    "peak_log10_change",
    "robust_median_shift",
    "log_iqr_change",
    "log_mad_change",
    "log_slope_change",
]


AGGREGATIONS = ["trial_mean", "trial_median", "trial_q90", "trial_iqr"]


METHODS = ["line_length_rate", "absolute_band_power_multitaper"]


TRIAL_COLUMNS = (
    "subject",
    "run",
    "bipolar_pair",
    "sampling_frequency_target_hz",
    "frequency_band",
    "method",
)


PATIENT_COLUMNS = (
    "patient",
    "bipolar_pair",
    "bipolar_channel",
    "n_successful_runs",
)


DICTIONARY_COLUMNS = (
    "feature",
    "frequency_band",
    "method",
    "trial_feature",
    "aggregation",
)


def multitaper_bandwidth(window_duration_s: float) -> float:
    """Return the formal DPSS bandwidth for one subwindow.

    The manuscript specifies a time-bandwidth setting of ``8 / duration``.
    Keeping this calculation in one helper ensures that target extraction and
    the source-only 500-Hz re-extraction use exactly the same value (1.6 Hz
    for 5-s windows and 4 Hz for 2-s windows).
    """

    try:
        duration = float(window_duration_s)
    except (TypeError, ValueError) as exc:
        raise ValueError("window duration must be a finite positive number") from exc
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("window duration must be a finite positive number")
    return 8.0 / duration


def _subwindow_duration_for_band(fmax_hz: float) -> float:
    """Use the locked 5-s low-frequency/2-s high-frequency windows."""

    value = float(fmax_hz)
    if not np.isfinite(value):
        raise ValueError("band upper frequency must be finite")
    return 5.0 if value <= 30.0 else 2.0


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _strict_log10(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("metric matrix must be finite and strictly positive")
    return np.log10(values)


def _theil_sen(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if values.shape[1] < 2:
        return np.zeros(values.shape[0], dtype=float)
    left, right = np.triu_indices(values.shape[1], k=1)
    dt = times[right] - times[left]
    return np.median((values[:, right] - values[:, left]) / dt[None, :], axis=1)


def _window_features(
    log_values: np.ndarray, centers: np.ndarray
) -> dict[str, np.ndarray]:
    med = np.median(log_values, axis=1)
    q25 = np.quantile(log_values, 0.25, axis=1)
    q75 = np.quantile(log_values, 0.75, axis=1)
    q90 = np.quantile(log_values, 0.90, axis=1)
    peak = np.max(log_values, axis=1)
    return {
        "log_mean": np.mean(log_values, axis=1),
        "log_median": med,
        "log_q75": q75,
        "log_q90": q90,
        "log_iqr": q75 - q25,
        "log_mad": np.median(np.abs(log_values - med[:, None]), axis=1),
        "log_peak": peak,
        "log_peak_prominence": peak - med,
        "log_theil_sen_slope_per_sec": _theil_sen(log_values, centers),
    }


def _contrast(
    base: np.ndarray,
    early: np.ndarray,
    bf: dict[str, np.ndarray],
    ef: dict[str, np.ndarray],
    early_centers: np.ndarray,
    window_duration_s: float | None = None,
) -> dict[str, np.ndarray]:
    q95 = np.quantile(base, 0.95, axis=1)
    median_change = ef["log_median"] - bf["log_median"]
    denom = 1.4826 * bf["log_mad"]
    robust = np.divide(
        median_change, denom, out=np.full_like(median_change, np.nan), where=denom > 0
    )
    excess = np.maximum(early - q95[:, None], 0.0)
    if window_duration_s is None:
        # Infer the locked duration from equally spaced centers for backwards
        # compatibility with direct callers.  A single window defaults to 5 s
        # (the historical low-frequency contract).
        if len(early_centers) > 1:
            step = float(np.median(np.diff(early_centers)))
            window_duration_s = step / 0.5 if step > 0 else 5.0
        else:
            window_duration_s = 5.0
    duration = float(window_duration_s)
    latency = early_centers[np.argmax(early, axis=1)] / max(
        float(early_centers[-1] - early_centers[0] + duration), 1e-9
    )
    return {
        "median_log10_change": median_change,
        "q90_log10_change": ef["log_q90"] - bf["log_q90"],
        "peak_log10_change": ef["log_peak"] - bf["log_peak"],
        "robust_median_shift": robust,
        "early_exceed_baseline_q95_fraction": np.mean(early > q95[:, None], axis=1),
        "early_positive_excess_over_baseline_q95_mean": np.mean(excess, axis=1),
        "log_iqr_change": ef["log_iqr"] - bf["log_iqr"],
        "log_mad_change": ef["log_mad"] - bf["log_mad"],
        "log_slope_change": ef["log_theil_sen_slope_per_sec"]
        - bf["log_theil_sen_slope_per_sec"],
        "early_peak_latency_fraction": np.clip(latency, 0.0, 1.0),
    }


def _rank(values: np.ndarray) -> np.ndarray:
    out = np.full(len(values), np.nan)
    finite = np.isfinite(values)
    n = int(finite.sum())
    if n == 0:
        return out
    if n == 1:
        out[finite] = 0.5
        return out
    ranks = pd.Series(values[finite]).rank(method="average").to_numpy(float)
    out[finite] = (ranks - 1) / (n - 1)
    return out


def _aggregate(values: Sequence[float], op: str) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    if op == "trial_mean":
        return float(np.mean(a))
    if op == "trial_median":
        return float(np.median(a))
    if op == "trial_q90":
        return float(np.quantile(a, 0.9))
    if op == "trial_iqr":
        return (
            float(np.quantile(a, 0.75) - np.quantile(a, 0.25))
            if a.size >= 2
            else np.nan
        )
    raise ValueError(op)


def _bandpass(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> np.ndarray:
    mne = _import_mne()
    return np.asarray(
        mne.filter.filter_data(
            data,
            sfreq=sfreq,
            l_freq=fmin,
            h_freq=fmax,
            method="fir",
            phase="zero",
            fir_window="hamming",
            fir_design="firwin",
            filter_length="auto",
            l_trans_bandwidth="auto",
            h_trans_bandwidth="auto",
            pad="reflect_limited",
            n_jobs=1,
            copy=True,
            verbose="ERROR",
        ),
        dtype=float,
    )


def _line_matrix(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    starts: list[float],
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    filtered = _bandpass(data, sfreq, fmin, fmax)
    slices = [(int(round(s * sfreq)), int(round((s + length) * sfreq))) for s in starts]
    values = np.empty((data.shape[0], len(slices)), dtype=float)
    for j, (a, b) in enumerate(slices):
        seg = filtered[:, a:b]
        if seg.shape[1] < 2:
            raise ValueError("line-length window outside extracted interval")
        values[:, j] = np.sum(np.abs(np.diff(seg, axis=1)), axis=1) / (
            (seg.shape[1] - 1) / sfreq
        )
    centers = np.asarray([s + length / 2 for s in starts], dtype=float)
    return values, centers


def _power_matrix(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    starts: list[float],
    length: float,
) -> tuple[np.ndarray, np.ndarray]:
    from mne.time_frequency import psd_array_multitaper

    n = int(round(length * sfreq))
    out = np.empty((data.shape[0], len(starts)), dtype=float)
    for j, s in enumerate(starts):
        a = int(round(s * sfreq))
        seg = data[:, a : a + n]
        if seg.shape[1] != n:
            raise ValueError("power window outside extracted interval")
        psd, freqs = psd_array_multitaper(
            seg,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            bandwidth=multitaper_bandwidth(length),
            adaptive=False,
            low_bias=True,
            normalization="full",
            remove_dc=True,
            output="power",
            n_jobs=1,
            max_iter=150,
            verbose="ERROR",
        )
        integ = trapezoid(psd, freqs, axis=-1)
        if not np.isfinite(integ).all() or np.any(integ <= 0):
            raise ValueError("invalid multitaper power")
        out[:, j] = integ
    return out, np.asarray([s + length / 2 for s in starts], dtype=float)


def compute_feature_arrays(
    base_values: np.ndarray,
    early_values: np.ndarray,
    base_centers: np.ndarray,
    early_centers: np.ndarray,
    window_duration_s: float | None = None,
) -> dict[str, np.ndarray]:
    base_log = _strict_log10(base_values)
    early_log = _strict_log10(early_values)
    bf = _window_features(base_log, base_centers)
    ef = _window_features(early_log, early_centers)
    arrays: dict[str, np.ndarray] = {}
    arrays.update({f"baseline__{k}": v for k, v in bf.items()})
    arrays.update({f"early__{k}": v for k, v in ef.items()})
    arrays.update(
        {
            f"contrast__{k}": v
            for k, v in _contrast(
                base_log, early_log, bf, ef, early_centers, window_duration_s
            ).items()
        }
    )
    for source in RANK_SOURCES:
        rank = _rank(arrays[source])
        arrays[f"rank_percentile__{source}"] = rank
        arrays[f"rank_top25__{source}"] = np.where(
            np.isfinite(rank), (rank >= 0.75).astype(float), np.nan
        )
    return arrays


def extract_recording_features(
    record: Mapping[str, Any],
    pairs: pd.DataFrame,
    cfg: Mapping[str, Any],
    *,
    residual_peaks_hz: Sequence[float] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mne = _import_mne()
    subject = str(record["subject"])
    run = int(record["run"])
    edf = _record_path(record.get("edf_path", record.get("raw_path", "")))
    native_hz = _safe_float(record.get("sampling_frequency_hz"))
    onset = _safe_float(record.get("seizure_onset_s", record.get("onset_s")))
    if native_hz is None or onset is None or edf is None or not edf.is_file():
        raise ValueError("missing sfreq/onset")
    target_hz = float((cfg.get("sampling") or {}).get("target_hz", 500))
    if native_hz < target_hz and not bool(
        (cfg.get("sampling") or {}).get("allow_upsampling", False)
    ):
        raise ValueError("NOT_HARMONIZABLE_NO_UPSAMPLING")
    subset = pairs[pairs.subject.astype(str).eq(subject)].copy()
    if subset.empty:
        raise ValueError("no pairs for subject")
    endpoints = sorted(
        set(subset.contact_1.astype(str)) | set(subset.contact_2.astype(str))
    )
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    missing = sorted(set(endpoints) - set(raw.ch_names))
    if missing:
        raise ValueError(f"missing endpoints: {missing[:4]}")
    guard = 2.0
    first = int(round((onset - 40.0 - guard) * native_hz))
    stop = int(round((onset + 10.0 + guard) * native_hz))
    if first < 0:
        raise ValueError("window before recording")
    data = raw.get_data(picks=endpoints, start=first, stop=stop)
    endpoint_index = {name: i for i, name in enumerate(endpoints)}
    bipolar = np.vstack(
        [
            data[endpoint_index[str(row.contact_1)]]
            - data[endpoint_index[str(row.contact_2)]]
            for row in subset.itertuples(index=False)
        ]
    )
    bipolar = _apply_notch_filter(
        bipolar,
        native_hz,
        cfg,
        power_line_hz=_safe_float(record.get("power_line_frequency_hz")),
        residual_peaks_hz=residual_peaks_hz,
    )
    if native_hz != target_hz:
        bipolar = _resample_poly(bipolar, native_hz, target_hz)
    trim = int(round(guard * target_hz))
    bipolar = bipolar[:, trim:-trim]
    bands = cfg.get("bands", {}) or {}
    rows = [
        {
            "subject": subject,
            "run": run,
            "bipolar_pair": str(pair),
            "sampling_frequency_target_hz": target_hz,
        }
        for pair in subset.bipolar_pair.astype(str)
    ]
    available, unavailable = compatible_bands(bands, target_hz)
    for band in available:
        fmin, fmax = map(float, bands[band])
        length = _subwindow_duration_for_band(fmax)
        bs = starts_for_window(-40.0, -5.0, length)
        es = starts_for_window(0.0, 10.0, length)
        # Convert relative-to-onset starts into the extracted interval's
        # onset-40 coordinate system.
        bstarts = [s + 40.0 for s in bs]
        estarts = [s + 40.0 for s in es]
        for method in METHODS:
            if method == "line_length_rate":
                bv, bc = _line_matrix(bipolar, target_hz, fmin, fmax, bstarts, length)
                ev, ec = _line_matrix(bipolar, target_hz, fmin, fmax, estarts, length)
            else:
                bv, bc = _power_matrix(bipolar, target_hz, fmin, fmax, bstarts, length)
                ev, ec = _power_matrix(bipolar, target_hz, fmin, fmax, estarts, length)
            arrays = compute_feature_arrays(
                bv, ev, np.asarray(bs) + length / 2, np.asarray(es) + length / 2, length
            )
            for i in range(len(rows)):
                for k, v in arrays.items():
                    rows[i][f"{band}__{method}__{k}"] = (
                        float(v[i]) if np.isfinite(v[i]) else np.nan
                    )
                # Keep row-level QC technical only.
                rows[i][f"{band}__{method}__baseline_n_subwindows"] = len(bs)
                rows[i][f"{band}__{method}__early_n_subwindows"] = len(es)
    for band in unavailable:
        for method in METHODS:
            for i in range(len(rows)):
                rows[i][f"{band}__{method}__status"] = "NOT_HARMONIZABLE"
    return rows, {
        "subject": subject,
        "run": run,
        "native_sampling_frequency_hz": native_hz,
        "target_sampling_frequency_hz": target_hz,
        "n_pairs": len(rows),
        "available_bands": "|".join(available),
        "unavailable_bands": "|".join(sorted(unavailable)),
        "status": "success",
    }


def aggregate_patient_features(
    trial: pd.DataFrame, bands: Mapping[str, Sequence[float]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trial.empty:
        # Preserve a parseable schema for blocked/empty runs.  Do not add
        # candidate feature columns here: an unsupported band is represented
        # in ``feature_band_status_label_blind.csv`` rather than as all-NaN
        # scientific features.
        return (
            pd.DataFrame(columns=list(PATIENT_COLUMNS)),
            pd.DataFrame(columns=list(DICTIONARY_COLUMNS)),
        )
    records: dict[tuple[str, str], dict[str, Any]] = {}
    dictionary: list[dict[str, Any]] = []
    for (patient, pair, band, method), group in trial.groupby(
        ["subject", "bipolar_pair", "frequency_band", "method"], sort=True
    ):
        row = records.setdefault(
            (str(patient), str(pair)),
            {"patient": str(patient), "bipolar_pair": str(pair)},
        )
        prefix = f"feature__{band}__{method}"
        base_names = (
            [f"baseline__{x}" for x in WINDOW_FEATURES]
            + [f"early__{x}" for x in WINDOW_FEATURES]
            + [f"contrast__{x}" for x in CONTRAST_FEATURES]
            + [f"rank_percentile__{x}" for x in RANK_SOURCES]
        )
        for name in base_names:
            # ``group`` retains the union of columns from every band/method.
            # A status-only, Nyquist-incompatible band therefore appears to
            # contain the feature column names even though every value is
            # absent.  Require at least one finite value in this group before
            # materialising a patient feature; otherwise unsupported bands
            # would masquerade as 144 valid all-NaN candidates downstream.
            if name not in group:
                continue
            values = pd.to_numeric(group[name], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).any():
                continue
            for op in AGGREGATIONS:
                col = f"{prefix}__{name}__{op}"
                row[col] = _aggregate(group[name], op)
                dictionary.append(
                    {
                        "feature": col,
                        "frequency_band": band,
                        "method": method,
                        "trial_feature": name,
                        "aggregation": op,
                    }
                )
        for name in RANK_SOURCES:
            source = f"rank_top25__{name}"
            if source in group:
                values = pd.to_numeric(group[source], errors="coerce")
                if not np.isfinite(values.to_numpy(dtype=float)).any():
                    continue
                col = f"{prefix}__rank_top25__{name}__trial_top25_fraction"
                row[col] = _aggregate(group[source], "trial_mean")
                dictionary.append(
                    {
                        "feature": col,
                        "frequency_band": band,
                        "method": method,
                        "trial_feature": source,
                        "aggregation": "trial_top25_fraction",
                    }
                )
        for name in DIRECTIONAL_CONTRASTS:
            source = f"contrast__{name}"
            if source in group:
                vals = pd.to_numeric(group[source], errors="coerce")
                finite = vals[np.isfinite(vals)]
                if len(finite) == 0:
                    continue
                col = f"{prefix}__contrast__{name}__trial_positive_fraction"
                row[col] = float((finite > 0).mean())
                dictionary.append(
                    {
                        "feature": col,
                        "frequency_band": band,
                        "method": method,
                        "trial_feature": source,
                        "aggregation": "trial_positive_fraction",
                    }
                )
    # ``dict.values()`` is a dynamic view.  Pandas 3.0 no longer accepts the
    # view directly in ``from_records`` (it attempts positional indexing), so
    # materialise it explicitly.  This keeps deterministic insertion order
    # while remaining compatible with pandas 2.x and 3.x.
    patient = pd.DataFrame.from_records(list(records.values()))
    counts = (
        trial.groupby(["subject", "bipolar_pair"], sort=True)["run"]
        .nunique()
        .rename("n_successful_runs")
        .reset_index()
        .rename(columns={"subject": "patient"})
    )
    patient = counts.merge(
        patient, on=["patient", "bipolar_pair"], how="right", validate="one_to_one"
    )
    patient["bipolar_channel"] = patient["bipolar_pair"]
    dictionary_frame = pd.DataFrame.from_records(
        dictionary, columns=list(DICTIONARY_COLUMNS)
    ).drop_duplicates("feature", keep="first")
    return patient, dictionary_frame


_FEATURE_VALUE_PREFIXES = (
    "baseline__",
    "early__",
    "contrast__",
    "rank_percentile__",
    "rank_top25__",
)


def stack_feature_rows(
    raw: pd.DataFrame, bands: Mapping[str, Sequence[float]]
) -> pd.DataFrame:
    """Convert wide per-pair rows into the compact formal trial schema.

    Unsupported bands are represented by ``*_status`` columns in ``raw``.
    They are intentionally omitted from the trial table; their explicit
    status is written by :func:`_band_status_table` instead.  This guarantees
    that no all-NaN candidate columns are manufactured during aggregation.
    """

    if raw.empty:
        return pd.DataFrame(columns=list(TRIAL_COLUMNS))
    id_cols = ["subject", "run", "bipolar_pair", "sampling_frequency_target_hz"]
    records: list[dict[str, Any]] = []
    for _, record in raw.iterrows():
        common = {key: record[key] for key in id_cols if key in record}
        for band in bands:
            for method in METHODS:
                prefix = f"{band}__{method}__"
                keys = [
                    key
                    for key in raw.columns
                    if key.startswith(prefix)
                    and key[len(prefix) :].startswith(_FEATURE_VALUE_PREFIXES)
                ]
                if not keys:
                    # Status-only (e.g. 80--250 Hz at a 500-Hz target) block.
                    continue
                row = {**common, "frequency_band": band, "method": method}
                for key in keys:
                    row[key[len(prefix) :]] = record[key]
                records.append(row)
    return pd.DataFrame(records)
