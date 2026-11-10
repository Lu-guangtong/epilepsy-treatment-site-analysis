"""Synthetic numerical, permutation and checkpoint regression tests."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import permutation as core  # noqa: E402
import run_permutation_tests as runner  # noqa: E402
import run_pair_lopo as reference  # noqa: E402
from pair_analysis.models import _ap_fast  # noqa: E402


def fixture_data():
    rng = np.random.default_rng(439)
    codes = np.repeat(np.arange(4), 12)
    y = np.tile([0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0], 4)
    x = rng.normal(size=(48, 9))
    x[:, 0] = np.round(x[:, 0])  # tied scores
    x[:, 1] = -y + rng.normal(scale=0.2, size=48)  # negative direction
    x[::5, 1] = np.nan
    x[:, 2] = 7  # ineligible constant
    x[:32, 3] = np.nan  # outer-fold-specific eligibility
    x[:, 4] = x[:, 0]  # exact duplicate/Spearman pruning
    names = [f"feature__ripple__line_length_rate__test_{j}" for j in range(9)]
    return x, y, codes, names


@pytest.mark.parametrize("sign", [1, -1])
def test_ap_ties_exact(sign):
    rng = np.random.default_rng(52)
    scores = sign * rng.integers(-3, 4, size=(50, 12)).astype(float)
    plan = core.APPlan(scores)
    for _ in range(4):
        y = rng.integers(0, 2, 50)
        np.testing.assert_array_equal(
            plan.evaluate(y), [_ap_fast(y, s) for s in scores.T]
        )


def test_permutation_preserves_counts_and_seed():
    _, y, codes, _ = fixture_data()
    a = core.shuffled_labels(y, codes, 11)
    b = core.shuffled_labels(y, codes, 11)
    np.testing.assert_array_equal(a, b)
    for p in np.unique(codes):
        assert a[codes == p].sum() == y[codes == p].sum()
    assert not np.array_equal(a, core.shuffled_labels(y, codes, 12))


def test_spearman_exact_pruning_boundary_matches_reference():
    x = pd.Series([0.0, 1.0, 2.0, 3.0, 4.0])
    y = pd.Series([1.0, 0.0, 2.0, 3.0, 4.0])
    ref = x.corr(y, method="spearman")
    got = core.reference_correlations(np.column_stack([x, y]))[0, 1]
    assert got == ref
    assert (abs(got) >= 0.9) == (abs(ref) >= 0.9)


@pytest.mark.parametrize("permutation_id", [0, 2])
def test_complete_reference_equivalence(permutation_id):
    x, y, codes, names = fixture_data()
    if permutation_id:
        y = core.shuffled_labels(y, codes, permutation_id)
    frame = pd.DataFrame(x, columns=names)
    frame.insert(0, "bipolar_channel", [f"pair_{i}" for i in range(len(y))])
    frame.insert(0, "patient", [f"P{p}" for p in codes])
    labels = frame[["patient", "bipolar_channel"]].assign(label_binary=y)
    controls = {
        "labels": labels,
        "top_k": 5,
        "missing_threshold": 0.7,
        "corr_threshold": 0.9,
        "c_values": list(core.C_VALUES),
        "threshold_selection": "inner_oof_maximize_f1",
    }
    slow = reference.run_band(frame, "ripple", controls)
    engine = core.BandEngine(x, codes, names)
    fast = engine.run(y, audit=True)
    np.testing.assert_allclose(
        fast["prediction"], slow["pred"].y_score, atol=1e-12, rtol=1e-11
    )
    np.testing.assert_allclose(
        fast["thresholds"], slow["pred"].selected_threshold, atol=1e-12
    )
    for j, f in enumerate(fast["folds"]):
        row = slow["folds"].iloc[j]
        assert f["selected_features"] == row.selected_features.split("|")
        assert f["selected_C"] == row.selected_C


def test_outer_labels_cannot_change_outer_predictions():
    x, y, codes, names = fixture_data()
    engine = core.BandEngine(x, codes, names)
    first = engine.run(y, audit=True)
    changed = y.copy()
    changed[codes == 0] = np.roll(changed[codes == 0], 1)
    second = engine.run(changed, audit=True)
    np.testing.assert_array_equal(
        first["prediction"][codes == 0], second["prediction"][codes == 0]
    )
    assert (
        first["folds"][0]["selected_features"]
        == second["folds"][0]["selected_features"]
    )


def test_maxT_manual_and_incomplete_rejected():
    null = np.arange(3 * 2 * 2 * 1, dtype=float).reshape(3, 2, 2, 1) / 12
    observed = np.array([[[0.45], [0.55]], [[0.65], [0.75]]])
    raw, track, joint = core.permutation_summary(observed, null)
    for b in range(2):
        for t in range(2):
            assert raw[b, t, 0] == (1 + sum(null[:, b, t, 0] >= observed[b, t, 0])) / 4
            assert (
                track[b, t, 0]
                == (1 + sum(null[:, :, t, 0].max(axis=1) >= observed[b, t, 0])) / 4
            )
            assert joint[b, t, 0] >= track[b, t, 0] >= raw[b, t, 0]
    null[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        core.permutation_summary(observed, null)


def test_exact_sign_test_includes_identity():
    d = np.ones((3, 2, 1))
    obs, p, corrected = core.exact_paired_test(d)
    np.testing.assert_array_equal(obs, 1)
    np.testing.assert_array_equal(p, 1 / 8)
    np.testing.assert_array_equal(corrected, 1 / 8)
    _, p0, _ = core.exact_paired_test(np.zeros_like(d))
    np.testing.assert_array_equal(p0, 1)


def test_checkpoint_resume_duplicate_and_tamper(tmp_path):
    db = runner.open_db(tmp_path, "fingerprint-A")
    item = {
        "band": "ripple",
        "permutation_id": 4,
        "label_hash": "abc",
        "raw": [0.3, 0.6, 0.1],
        "smoothed": [0.4, 0.7, 0.2],
        "seconds": 2,
    }
    runner.record(db, item)
    runner.record(db, {**item, "seconds": 9})
    assert len(runner.read_records(db)) == 1
    with pytest.raises(ValueError):
        runner.record(db, {**item, "raw": [0.9, 0.6, 0.1]})
    db.close()
    with pytest.raises(ValueError):
        runner.open_db(tmp_path, "fingerprint-B")
    db = runner.open_db(tmp_path, "fingerprint-A")
    with db:
        db.execute(
            "UPDATE results SET payload=?",
            (json.dumps({**item, "raw": [0.99, 0.6, 0.1]}),),
        )
    with pytest.raises(ValueError):
        runner.read_records(db)
    db.close()


def test_paths_reject_sources_and_output_roots(tmp_path, monkeypatch):
    base = tmp_path / "outputs"
    monkeypatch.setattr(runner, "OUTPUT_BASE", base)
    for path in (tmp_path / "inputs", base, runner.CODE / "outputs"):
        with pytest.raises(ValueError):
            runner.safe_output(path)
    assert runner.safe_output(base / "run_1") == (base / "run_1").resolve()
    monkeypatch.setattr(runner, "FEATURE_ROOT", base / "source_features")
    with pytest.raises(ValueError):
        runner.safe_output(base / "source_features" / "run")


def test_real_spawn_parallel_and_resume(tmp_path):
    x, y, codes, names = fixture_data()
    cache = tmp_path / "cache"
    cache.mkdir()
    np.save(cache / "features_ripple.npy", x)
    np.save(cache / "labels.npy", y)
    np.save(cache / "patient_codes.npy", codes)
    for j in range(4):
        np.save(cache / f"graph_{j}.npy", np.zeros((12, 12)))
    runner.dump(
        cache / "cache.json",
        {"patients": list(range(4)), "features": {"ripple": names}},
    )
    runner.dump(tmp_path / "manifest.json", {"config": {"seed": 53}})
    first = []
    runner.schedule(
        tmp_path,
        [("ripple", 1), ("ripple", 2)],
        1,
        first.append,
        reserve=0.5,
        critical=0.2,
    )
    second = []
    runner.schedule(
        tmp_path,
        [("ripple", 2), ("ripple", 1)],
        2,
        second.append,
        reserve=0.5,
        critical=0.2,
        ramp=False,
    )
    first.sort(key=lambda r: r["permutation_id"])
    second.sort(key=lambda r: r["permutation_id"])
    for a, b in zip(first, second):
        assert a["label_hash"] == b["label_hash"]
        np.testing.assert_array_equal(
            [a["raw"], a["smoothed"]], [b["raw"], b["smoothed"]]
        )
    db = runner.open_db(tmp_path, "synthetic-run")
    runner.record(db, first[0])
    db.close()
    db = runner.open_db(tmp_path, "synthetic-run")
    runner.record(db, second[0])
    runner.record(db, second[1])
    assert len(runner.read_records(db)) == 2
    db.close()


def test_stop_and_bad_worker_arguments(tmp_path):
    with pytest.raises(ValueError):
        runner.schedule(tmp_path, [], 0, lambda r: None)
    runner.dump(tmp_path / "manifest.json", {"config": {"seed": 53}})
    (tmp_path / "STOP").touch()
    r = runner.schedule(tmp_path, [("ripple", 1)], 1, lambda r: None)
    assert r["interrupted"] and r["completed"] == 0


def test_source_and_cache_changes_reject_continuation(tmp_path):
    source, cache = tmp_path / "source.txt", tmp_path / "cache.bin"
    source.write_text("original source", encoding="utf-8")
    cache.write_bytes(b"original cache")
    manifest = {
        "config": {},
        "software": runner.software(),
        "source_hashes": {str(source): runner.sha(source)},
        "cache_hashes": {"cache.bin": runner.sha(cache)},
    }
    manifest["fingerprint"] = runner.digest(manifest)
    runner.dump(tmp_path / "manifest.json", manifest)
    runner.verify(tmp_path)
    source.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="Source changed"):
        runner.verify(tmp_path)
    source.write_text("original source", encoding="utf-8")
    cache.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Cache changed"):
        runner.verify(tmp_path)


def test_exclusive_run_lock(tmp_path):
    with runner.RunLock(tmp_path):
        with pytest.raises(RuntimeError, match="Another process"):
            with runner.RunLock(tmp_path):
                pass
    with runner.RunLock(tmp_path):
        pass


def test_low_memory_refuses_start(tmp_path, monkeypatch):
    from types import SimpleNamespace

    runner.dump(tmp_path / "manifest.json", {"config": {"seed": 53}})
    monkeypatch.setattr(
        runner.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=0.3 * runner.GIB),
    )
    with pytest.raises(MemoryError, match="Insufficient"):
        runner.schedule(tmp_path, [("ripple", 1)], 2, lambda r: None)
