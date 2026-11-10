"""Synthetic regressions for external signal and feature calculations."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pair_analysis.features import aggregate_patient_features, compute_feature_arrays
from pair_analysis.signal import _resample_poly, compatible_bands


@pytest.mark.parametrize(
    "sampling_hz, expected",
    [(500, ["beta"]), (1024, ["beta", "ripple", "fast_ripple"])],
)
def test_band_eligibility(sampling_hz, expected):
    bands = {"beta": (13, 30), "ripple": (80, 250), "fast_ripple": (250, 500)}
    available, unavailable = compatible_bands(bands, sampling_hz)
    assert available == expected
    assert set(available) | set(unavailable) == set(bands)


def test_patient_aggregation_produces_complete_feature_sets():
    rng = np.random.default_rng(318)
    rows = []
    for run in (1, 2, 3):
        arrays = compute_feature_arrays(
            rng.lognormal(size=(2, 13)),
            rng.lognormal(size=(2, 9)),
            np.arange(13) * 2.5 + 2.5,
            np.arange(9) + 1.0,
            2.0,
        )
        for method in ("line_length_rate", "absolute_band_power_multitaper"):
            for index, pair in enumerate(("A1-A2", "A2-A3")):
                rows.append(
                    {
                        "subject": "synthetic",
                        "run": run,
                        "bipolar_pair": pair,
                        "frequency_band": "ripple",
                        "method": method,
                        **{name: values[index] for name, values in arrays.items()},
                    }
                )
    patient, dictionary = aggregate_patient_features(
        pd.DataFrame(rows), {"ripple": (80, 250)}
    )
    assert len(patient) == 2
    assert patient.n_successful_runs.eq(3).all()
    assert patient.bipolar_channel.equals(patient.bipolar_pair)
    assert len(dictionary) == 288
    assert dictionary.groupby("method").size().eq(144).all()
    assert set(dictionary.feature).issubset(patient.columns)


def test_polyphase_resampling_preserves_duration():
    data = np.sin(2 * np.pi * 20 * np.arange(2048) / 1024)[None, :]
    result = _resample_poly(data, 1024, 500)
    assert result.shape == (1, 1000)
    assert np.isfinite(result).all()
