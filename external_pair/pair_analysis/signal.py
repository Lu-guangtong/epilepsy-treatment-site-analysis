"""Signal filtering, resampling and spectral-window definitions."""

from __future__ import annotations


import math
from typing import Any, Mapping, Sequence
import numpy as np
from .utils import bool_value


DEFAULT_NYQUIST_GUARD_HZ = 1.0


WINDOW_LENGTH_SECONDS = 5.0


WINDOW_OVERLAP = 0.5


NOTCH_FREQUENCY_HZ = 60.0


NOTCH_WIDTH_HZ = 2.0


NOTCH_TRANSITION_HZ = 1.0


NOTCH_NYQUIST_GUARD_HZ = 2.0


DEFAULT_HARMONIC_MAX_HZ = 950.0


DEFAULT_RESIDUAL_PEAK_CLUSTER_HZ = 1.2


def _notch_frequency_plan(
    native_hz: float,
    config: Mapping[str, Any] | None = None,
    *,
    power_line_hz: float | None = None,
    residual_peaks_hz: Sequence[float] | None = None,
) -> list[float]:
    """Return the deterministic zero-phase FIR notch frequencies.

    The formal preprocessing removes the declared mains frequency and its
    harmonics before any downsampling.  The external recordings have mixed
    native rates, therefore frequencies that would touch the native Nyquist
    are omitted rather than clipped.  Optional residual peaks are supplied by
    a label-blind detector/configuration and are merged after a 1.2-Hz
    clustering rule.  This helper is pure and is unit-testable without MNE.
    """

    cfg = dict(config or {})
    filtering = (
        cfg.get("filtering", {})
        if isinstance(cfg.get("filtering", {}), Mapping)
        else {}
    )
    line = power_line_hz
    if line is None:
        try:
            line = float(filtering.get("power_line_frequency_hz", NOTCH_FREQUENCY_HZ))
        except (TypeError, ValueError):
            line = NOTCH_FREQUENCY_HZ
    if not np.isfinite(line) or line <= 0:
        line = NOTCH_FREQUENCY_HZ
    try:
        max_hz = float(filtering.get("harmonic_max_hz", DEFAULT_HARMONIC_MAX_HZ))
    except (TypeError, ValueError):
        max_hz = DEFAULT_HARMONIC_MAX_HZ
    include_harmonics = bool_value(filtering.get("include_harmonics", True))
    nyquist = float(native_hz) / 2.0
    upper = min(max_hz, nyquist - NOTCH_NYQUIST_GUARD_HZ)
    frequencies: list[float] = []
    if include_harmonics and upper > 0:
        harmonic = line
        while harmonic <= upper + 1e-9:
            frequencies.append(float(harmonic))
            harmonic += line
    elif upper >= line:
        frequencies.append(float(line))
    # A configured residual list is preferred to silently inventing peaks.
    # Values are still checked against Nyquist and merged deterministically.
    configured_residual = filtering.get("residual_peaks_hz", [])
    if residual_peaks_hz is None:
        residual_peaks_hz = (
            configured_residual
            if isinstance(configured_residual, Sequence)
            and not isinstance(configured_residual, (str, bytes))
            else []
        )
    for value in residual_peaks_hz:
        try:
            frequency = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(frequency) and frequency > 0 and frequency <= upper + 1e-9:
            frequencies.append(frequency)
    if not frequencies:
        return []
    # Cluster within the prespecified 1.2-Hz tolerance and retain the
    # frequency with the largest deterministic priority (the harmonic centre
    # wins ties).  MNE accepts a sorted unique list.
    cluster_hz = float(
        filtering.get("residual_peak_cluster_hz", DEFAULT_RESIDUAL_PEAK_CLUSTER_HZ)
    )
    cluster_hz = (
        cluster_hz
        if np.isfinite(cluster_hz) and cluster_hz > 0
        else DEFAULT_RESIDUAL_PEAK_CLUSTER_HZ
    )
    ordered = sorted(float(x) for x in frequencies)
    clusters: list[list[float]] = []
    for frequency in ordered:
        if not clusters or frequency - clusters[-1][-1] > cluster_hz:
            clusters.append([frequency])
        else:
            clusters[-1].append(frequency)
    return [float(np.mean(cluster)) for cluster in clusters]


def _apply_notch_filter(
    data: np.ndarray,
    native_hz: float,
    config: Mapping[str, Any] | None = None,
    *,
    power_line_hz: float | None = None,
    residual_peaks_hz: Sequence[float] | None = None,
) -> np.ndarray:
    """Apply the formal zero-phase FIR notch plan and return finite data."""

    mne = _import_mne()
    frequencies = _notch_frequency_plan(
        native_hz,
        config,
        power_line_hz=power_line_hz,
        residual_peaks_hz=residual_peaks_hz,
    )
    if not frequencies:
        return np.asarray(data, dtype=float)
    filtered = mne.filter.notch_filter(
        np.asarray(data, dtype=float),
        Fs=float(native_hz),
        freqs=np.asarray(frequencies, dtype=float),
        notch_widths=NOTCH_WIDTH_HZ,
        trans_bandwidth=NOTCH_TRANSITION_HZ,
        method="fir",
        phase="zero",
        fir_window="hamming",
        fir_design="firwin",
        pad="reflect_limited",
        n_jobs=1,
        verbose="ERROR",
    )
    if not np.isfinite(filtered).all():
        raise ValueError("notch filter produced non-finite samples")
    return np.asarray(filtered, dtype=float)


def starts_for_window(
    start: float, stop: float, length: float = WINDOW_LENGTH_SECONDS
) -> list[float]:
    """Return deterministic half-overlapping window starts in seconds."""

    if stop <= start or length <= 0:
        return []
    step = length * (1.0 - WINDOW_OVERLAP)
    count = int(math.floor((stop - start - length) / step + 1.0 + 1e-9))
    return [float(start + i * step) for i in range(max(0, count))]


def compatible_bands(
    bands: Mapping[str, Sequence[float]],
    sampling_frequency: float,
    nyquist_guard_hz: float = DEFAULT_NYQUIST_GUARD_HZ,
) -> tuple[list[str], dict[str, str]]:
    """Partition bands into Nyquist-compatible and unavailable bands.

    ``fmax`` must leave a small guard below Nyquist.  This prevents calling a
    250--500-Hz band valid after resampling a 500-Hz recording.
    """

    nyquist = float(sampling_frequency) / 2.0
    available: list[str] = []
    unavailable: dict[str, str] = {}
    for name, bounds in bands.items():
        if len(bounds) != 2:
            unavailable[str(name)] = "band_bounds_must_have_two_values"
            continue
        fmin, fmax = float(bounds[0]), float(bounds[1])
        if fmin <= 0 or fmax <= fmin:
            unavailable[str(name)] = "invalid_band_bounds"
        elif fmax >= nyquist - float(nyquist_guard_hz):
            unavailable[str(name)] = (
                f"fmax_{fmax:g}_not_below_nyquist_{nyquist:g}_guard"
            )
        else:
            available.append(str(name))
    return available, unavailable


def _import_mne() -> Any:
    try:
        import mne  # type: ignore

        return mne
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "MNE-Python is required for run_features on EDF inputs"
        ) from exc


def _resample_poly(data: np.ndarray, source_hz: float, target_hz: float) -> np.ndarray:
    if abs(source_hz - target_hz) < 1e-9:
        return np.asarray(data, dtype=float)
    try:
        from scipy.signal import resample_poly
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("SciPy is required for deterministic downsampling") from exc
    source_int = int(round(source_hz))
    target_int = int(round(target_hz))
    divisor = math.gcd(source_int, target_int)
    return np.asarray(
        resample_poly(data, target_int // divisor, source_int // divisor, axis=-1),
        dtype=float,
    )
