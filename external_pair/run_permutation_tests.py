"""Run synchronized full-refit permutations for external bipolar-pair models."""

from __future__ import annotations
# ruff: noqa: E402 -- thread limits must precede NumPy/BLAS imports on spawn.

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_key] = "1"

import argparse
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import queue
import sqlite3
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from threadpoolctl import threadpool_limits

import permutation as core
import run_pair_lopo as reference
import run_graph_diffusion as graph_reference

CODE = Path(__file__).resolve().parent
WORK_ROOT = (
    Path(os.environ.get("SEEG_OUTPUT_ROOT", str(CODE.parent / "work")))
    .expanduser()
    .resolve()
)
OUTPUT_BASE = WORK_ROOT / "external_permutations"
REFERENCE_RUN = None
GRAPH_RUN = None
LABELS = None
FEATURE_ROOT = None
RUN_PURPOSE = "formal"
GIB = 1024**3


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=True).encode()
    ).hexdigest()


def software():
    return {
        "python": sys.version,
        **{
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "pandas",
                "scipy",
                "scikit-learn",
                "threadpoolctl",
                "psutil",
            )
        },
    }


def safe_output(path):
    """Require an isolated child of the output base and protect all inputs."""
    path = Path(path).resolve()
    base = Path(OUTPUT_BASE).resolve()
    if path == base or not path.is_relative_to(base):
        raise ValueError(
            "Output must be a child of --output-base (not the base itself)"
        )
    inputs = [CODE, FEATURE_ROOT, REFERENCE_RUN, GRAPH_RUN]
    if LABELS is not None:
        inputs.append(Path(LABELS).resolve().parent)
    for item in inputs:
        if item is None:
            continue
        source = Path(item).resolve()
        if path.is_relative_to(source) or source.is_relative_to(path):
            raise ValueError(f"Output overlaps a read-only input: {source}")
    return path


class RunLock:
    """OS-held lock releases on crash, with no unsafe stale-lock deletion."""

    def __init__(self, root):
        self.path = Path(root) / "run.lock"

    def __enter__(self):
        self.handle = self.path.open("a+b")
        if self.path.stat().st_size == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            raise RuntimeError("Another process owns this run directory") from None
        return self

    def __exit__(self, *args):
        self.handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def prepare(root, target, seed):
    if target < 1 or seed < 0:
        raise ValueError("Invalid permutation count/seed")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(
            "Prepare requires an empty directory; use existing run commands to resume"
        )
    root.mkdir(parents=True, exist_ok=True)
    cache = root / "cache"
    cache.mkdir()
    source_paths = [
        LABELS,
        CODE / "run_pair_lopo.py",
        CODE / "run_graph_diffusion.py",
        CODE / "permutation.py",
        Path(__file__).resolve(),
    ]
    source_paths.extend(sorted((core.PACKAGE_ROOT / "pair_analysis").rglob("*.py")))
    labels = pd.read_csv(LABELS)
    if labels.duplicated(["patient", "bipolar_channel"]).any():
        raise ValueError("Duplicate label keys")
    labels = labels[labels.primary_status.eq("primary")].copy()
    if (
        len(labels) != 1398
        or labels.label_binary.sum() != 87
        or set(labels.patient) != set(reference.PATIENTS)
        or not labels.label_binary.isin([0, 1]).all()
    ):
        raise ValueError("The locked 13-patient label contract changed")
    canonical = None
    feature_names = {}
    reference_scores, reference_thresholds, smooth_scores = [], [], []
    graph_path = GRAPH_RUN / "graph_smoothed_predictions_all_bands.csv"
    graph_predictions = pd.read_csv(graph_path)
    source_paths.append(graph_path)
    for band in core.BANDS:
        path = FEATURE_ROOT / (
            "patient_features_label_blind.csv"
            if band in reference.COMMON_BANDS
            else f"native_{band}_1024hz_patient_features_label_blind.csv"
        )
        source_paths.append(path)
        header = pd.read_csv(path, nrows=0)
        from pair_analysis.models import _assert_feature_table_label_blind

        _assert_feature_table_label_blind(path, header.columns)
        key = "patient" if "patient" in header else "subject"
        pair = "bipolar_channel" if "bipolar_channel" in header else "bipolar_pair"
        keys = pd.read_csv(path, usecols=[key, pair])
        keys = keys[keys[key].isin(reference.PATIENTS)]
        if keys.duplicated([key, pair]).any():
            raise ValueError(f"Duplicate feature keys in {band}")
        frame = reference.read_selected_feature_table(path, band, reference.PATIENTS)
        data = frame.merge(
            labels[["patient", "bipolar_channel", "label_binary"]],
            on=["patient", "bipolar_channel"],
            validate="one_to_one",
        )
        if len(data) != 1398:
            raise ValueError(f"Missing feature rows in {band}")
        names = [
            c
            for c in reference.feature_columns(data, band)
            if reference.feature_method(c) in reference.METHODS
        ]
        if len(names) != 288:
            raise ValueError(f"Unexpected feature count for {band}")
        this_keys = pd.MultiIndex.from_frame(data[["patient", "bipolar_channel"]])
        if canonical is None:
            canonical = this_keys
            cache_labels = data[["patient", "bipolar_channel", "label_binary"]].copy()
        elif not this_keys.equals(canonical):
            # Changing training row order can perturb tie-breaking or solver
            # results. Fail rather than silently reordering a reference run.
            raise ValueError(f"Feature row ordering differs in {band}")
        np.save(cache / f"features_{band}.npy", data[names].to_numpy(float))
        feature_names[band] = names
        pred_path = REFERENCE_RUN / f"predictions_{band}.csv"
        pred = pd.read_csv(pred_path).set_index(["patient", "bipolar_channel"])
        if not pred.index.is_unique or set(pred.index) != set(canonical):
            raise ValueError("Reference prediction population mismatch")
        pred = pred.loc[canonical]
        if not np.array_equal(
            pred.y_true.to_numpy(), cache_labels.label_binary.to_numpy()
        ):
            raise ValueError("Reference labels differ")
        reference_scores.append(pred.y_score.to_numpy())
        reference_thresholds.append(pred.selected_threshold.to_numpy())
        smooth = graph_predictions[graph_predictions.frequency_band.eq(band)].set_index(
            ["patient", "bipolar_channel"]
        )
        if not smooth.index.is_unique or set(smooth.index) != set(canonical):
            raise ValueError("Smoothed prediction population mismatch")
        smooth_scores.append(smooth.loc[canonical].y_score_smoothed.to_numpy())
        source_paths.extend(
            [
                pred_path,
                REFERENCE_RUN / f"folds_{band}.csv",
                REFERENCE_RUN / f"feature_scores_{band}.csv",
                REFERENCE_RUN / f"c_tuning_{band}.csv",
                REFERENCE_RUN / f"directions_{band}.csv",
            ]
        )
    patients = sorted(cache_labels.patient.unique())
    codes = cache_labels.patient.map(
        dict(zip(patients, range(len(patients))))
    ).to_numpy(int)
    y = cache_labels.label_binary.to_numpy(np.int8)
    cache_labels.to_csv(cache / "keys.csv", index=False)
    np.save(cache / "labels.npy", y)
    np.save(cache / "patient_codes.npy", codes)
    np.save(cache / "observed_raw.npy", np.array(reference_scores))
    np.save(cache / "observed_thresholds.npy", np.array(reference_thresholds))
    np.save(cache / "observed_smoothed.npy", np.array(smooth_scores))
    # Rebuild geometry from coordinate/key columns only.
    geometry = labels[
        [
            "patient",
            "bipolar_channel",
            "shaft",
            "midpoint_x_raw_units",
            "midpoint_y_raw_units",
            "midpoint_z_raw_units",
        ]
    ]
    # The reference graph omits edges for missing coordinates and retains
    # those nodes as isolated (original score unchanged). Preserve this rule.
    missing_coordinates = int(
        (~np.isfinite(geometry.iloc[:, 3:].to_numpy(float)).all(axis=1)).sum()
    )
    points = graph_reference.make_points(geometry)
    edges = graph_reference.build_edges(points, 10.0)
    groups = []
    for i, patient in enumerate(patients):
        idx = np.flatnonzero(codes == i)
        groups.append(idx)
        order = cache_labels.iloc[idx].bipolar_channel.tolist()
        weights, _ = graph_reference.adjacency_matrix(
            len(idx), edges[edges.subject.eq(patient)], node_order=order
        )
        np.save(cache / f"graph_{i}.npy", weights)
    all_metrics = np.array(
        [
            [
                core.patient_metrics(y, scores, groups)
                for scores in (reference_scores[b], smooth_scores[b])
            ]
            for b in range(8)
        ]
    )  # band, track, patient, metric
    np.save(cache / "observed_patient_metrics.npy", all_metrics)
    dump(cache / "cache.json", {"patients": patients, "features": feature_names})
    config = {
        "n_permutations": target,
        "seed": seed,
        "bands": list(core.BANDS),
        "primary_band": "ripple",
        "primary_metric": "patient_macro_AUPRC",
        "permutation": "within_patient_bipolar_labels_synchronized_across_all_bands",
        "n_patients": 13,
        "n_pairs": 1398,
        "n_positive": 87,
        "run_purpose": RUN_PURPOSE,
        "full_refit": True,
        "graph": {
            "k": 3,
            "radius_mm": 10,
            "sigma_mm": 5,
            "alpha": 0.8,
            "missing_coordinate_isolates": missing_coordinates,
        },
    }
    payload = {
        "config": config,
        "software": software(),
        "source_hashes": {str(p): sha(p) for p in sorted(set(source_paths))},
        "cache_hashes": {
            str(p.relative_to(root)): sha(p) for p in sorted(cache.iterdir())
        },
    }
    payload["fingerprint"] = digest(payload)
    dump(root / "manifest.json", payload)
    dump(root / "status.json", {"status": "PREPARED", "target_tasks": target * 8})
    print(
        json.dumps(
            {
                "status": "PREPARED",
                "root": str(root),
                "fingerprint": payload["fingerprint"],
            }
        ),
        flush=True,
    )


def verify(root, require_validation=False):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest.pop("fingerprint")
    if digest(manifest) != expected:
        raise ValueError("Manifest fingerprint mismatch")
    manifest["fingerprint"] = expected
    if manifest["software"] != software():
        raise ValueError("Software environment changed")
    for path, expected_hash in manifest["source_hashes"].items():
        if sha(path) != expected_hash:
            raise ValueError(f"Source changed: {path}")
    for path, expected_hash in manifest["cache_hashes"].items():
        if sha(root / path) != expected_hash:
            raise ValueError(f"Cache changed: {path}")
    if require_validation:
        gate = json.loads((root / "validation.json").read_text(encoding="utf-8"))
        if gate.get("status") != "PASS" or gate.get("fingerprint") != expected:
            raise ValueError("Passed reference-equivalence validation required")
    return manifest


def validate(root):
    manifest = verify(root)
    cache = root / "cache"
    y, codes = np.load(cache / "labels.npy"), np.load(cache / "patient_codes.npy")
    metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    ref_raw, ref_smooth = (
        np.load(cache / "observed_raw.npy"),
        np.load(cache / "observed_smoothed.npy"),
    )
    ref_threshold = np.load(cache / "observed_thresholds.npy")
    keys = pd.read_csv(cache / "keys.csv")
    results = []
    for b, band in enumerate(core.BANDS):
        started = time.perf_counter()
        engine = core.load_engine(cache, band)
        got = engine.run(y, audit=True)
        np.testing.assert_allclose(
            got["prediction"], ref_raw[b], rtol=1e-10, atol=1e-11
        )
        np.testing.assert_allclose(
            got["smoothed_prediction"], ref_smooth[b], rtol=1e-10, atol=1e-11
        )
        np.testing.assert_allclose(
            got["thresholds"], ref_threshold[b], rtol=1e-10, atol=1e-11
        )
        folds = pd.read_csv(REFERENCE_RUN / f"folds_{band}.csv").set_index(
            "heldout_patient"
        )
        fs = pd.read_csv(REFERENCE_RUN / f"feature_scores_{band}.csv")
        directions = pd.read_csv(REFERENCE_RUN / f"directions_{band}.csv")
        for fold in got["folds"]:
            patient = metadata["patients"][fold["outer"]]
            row = folds.loc[patient]
            if (
                fold["selected_features"] != row.selected_features.split("|")
                or fold["selected_C"] != row.selected_C
            ):
                raise AssertionError(f"Selection mismatch: {band}/{patient}")
            dd = directions[directions.heldout_patient.eq(patient)].set_index("feature")
            np.testing.assert_array_equal(
                fold["directions"], dd.loc[fold["selected_features"], "direction"]
            )
            expected = (
                fs[fs.heldout_patient.eq(patient)]
                .set_index("feature")
                .inner_macro_AUPRC
            )
            for name, score in fold["feature_scores"].items():
                np.testing.assert_allclose(score, expected[name], rtol=0, atol=1e-14)
        entry = {
            "band": band,
            "observed_prediction_max_abs_error": float(
                np.max(np.abs(got["prediction"] - ref_raw[b]))
            ),
            "seconds": time.perf_counter() - started,
            "raw": got["raw"],
            "smoothed": got["smoothed"],
        }
        results.append(entry)
        dump(
            root / "validation_progress.json",
            {"status": "RUNNING", "observed": results},
        )
        print(json.dumps(entry), flush=True)
        if band == "ripple":
            perm_y = core.shuffled_labels(y, codes, 1, manifest["config"]["seed"])
            t_fast = time.perf_counter()
            fast_null = engine.run(perm_y, audit=True)
            fast_seconds = time.perf_counter() - t_fast
            frame = keys[["patient", "bipolar_channel"]].copy()
            frame = pd.concat(
                [frame, pd.DataFrame(engine.x, columns=engine.names)], axis=1
            )
            labels = keys[["patient", "bipolar_channel"]].assign(label_binary=perm_y)
            controls = {
                "labels": labels,
                "top_k": 5,
                "missing_threshold": 0.7,
                "corr_threshold": 0.9,
                "c_values": list(core.C_VALUES),
                "threshold_selection": "inner_oof_maximize_f1",
            }
            print("reference null replay: ripple permutation 1", flush=True)
            t_reference = time.perf_counter()
            with threadpool_limits(limits=1), warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message="'penalty' was deprecated.*",
                )
                slow = reference.run_band(frame, band, controls)
            reference_seconds = time.perf_counter() - t_reference
            pred = (
                slow["pred"]
                .set_index(["patient", "bipolar_channel"])
                .loc[pd.MultiIndex.from_frame(keys[["patient", "bipolar_channel"]])]
            )
            np.testing.assert_allclose(
                fast_null["prediction"], pred.y_score, rtol=1e-10, atol=1e-11
            )
            for fold in fast_null["folds"]:
                patient = metadata["patients"][fold["outer"]]
                old = slow["folds"].set_index("heldout_patient").loc[patient]
                if (
                    fold["selected_features"] != old.selected_features.split("|")
                    or fold["selected_C"] != old.selected_C
                ):
                    raise AssertionError("Null feature/C mismatch")
                np.testing.assert_allclose(
                    fold["selected_threshold"],
                    old.selected_threshold,
                    atol=1e-11,
                    rtol=1e-10,
                )
            dump(
                root / "reference_null_validation.json",
                {
                    "status": "PASS",
                    "permutation_id": 1,
                    "band": band,
                    "cached_fast_seconds": fast_seconds,
                    "reference_seconds": reference_seconds,
                    "prediction_max_abs_error": float(
                        np.max(np.abs(fast_null["prediction"] - pred.y_score))
                    ),
                },
            )
        del engine
    verify(root)
    dump(
        root / "validation.json",
        {
            "status": "PASS",
            "fingerprint": manifest["fingerprint"],
            "observed": results,
            "reference_null_band": "ripple",
            "reference_null_id": 1,
        },
    )
    print("VALIDATION PASS", flush=True)


def open_db(root, fingerprint):
    db = sqlite3.connect(root / "checkpoint.sqlite")
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS results (band TEXT, permutation_id INTEGER, payload TEXT NOT NULL, checksum TEXT NOT NULL, PRIMARY KEY(band,permutation_id))"
    )
    saved = db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
    if saved and saved[0] != fingerprint:
        db.close()
        raise ValueError("Checkpoint fingerprint mismatch")
    with db:
        db.execute(
            "INSERT OR IGNORE INTO metadata VALUES ('fingerprint',?)", (fingerprint,)
        )
    return db


def record(db, result):
    metrics = {
        k: result[k]
        for k in ("band", "permutation_id", "label_hash", "raw", "smoothed")
    }
    checksum = digest(metrics)
    existing = db.execute(
        "SELECT checksum FROM results WHERE band=? AND permutation_id=?",
        (result["band"], result["permutation_id"]),
    ).fetchone()
    if existing:
        if existing[0] != checksum:
            raise ValueError("Conflicting duplicate task result")
        return
    with db:
        db.execute(
            "INSERT INTO results VALUES (?,?,?,?)",
            (
                result["band"],
                result["permutation_id"],
                json.dumps(result, allow_nan=False),
                checksum,
            ),
        )


def read_records(db):
    output = []
    for band, pid, payload, checksum in db.execute(
        "SELECT band,permutation_id,payload,checksum FROM results"
    ):
        item = json.loads(payload)
        if item["band"] != band or item["permutation_id"] != pid:
            raise ValueError("Checkpoint key mismatch")
        if (
            digest(
                {
                    k: item[k]
                    for k in ("band", "permutation_id", "label_hash", "raw", "smoothed")
                }
            )
            != checksum
        ):
            raise ValueError("Checkpoint payload checksum mismatch")
        if not np.isfinite([item["raw"], item["smoothed"]]).all():
            raise ValueError("Nonfinite checkpoint metric")
        output.append(item)
    return output


def worker(cache, seed, inbox, outbox):
    try:
        if os.name == "nt":
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        cache = Path(cache)
        y, codes = np.load(cache / "labels.npy"), np.load(cache / "patient_codes.npy")
        current, engine = None, None
        with threadpool_limits(limits=1):
            while True:
                task = inbox.get()
                if task is None:
                    return
                band, pid = task
                start = time.perf_counter()
                if current != band:
                    engine = None
                    engine = core.load_engine(cache, band)
                    current = band
                labels = core.shuffled_labels(y, codes, pid, seed)
                result = engine.run(labels)
                mem = psutil.Process().memory_info()
                result.update(
                    band=band,
                    permutation_id=pid,
                    label_hash=hashlib.sha256(labels.tobytes()).hexdigest(),
                    seconds=time.perf_counter() - start,
                    worker_peak_gib=getattr(mem, "peak_wset", mem.rss) / GIB,
                )
                outbox.put((os.getpid(), "OK", result))
    except BaseException:
        outbox.put((os.getpid(), "ERROR", traceback.format_exc()))


def schedule(root, tasks, workers, on_result, reserve=2.5, critical=1.5, ramp=True):
    """Bounded, one-task-per-worker scheduling; no nested or unbounded queues.

    Cache locality: idle workers preferentially take another task in their band.
    At soft pressure retire an idle process; critical pressure terminates only
    this scheduler's children. Their uncommitted tasks remain resumable.
    """
    if not 1 <= workers <= 10 or not 0 < critical < reserve:
        raise ValueError("Invalid worker count or memory limits")
    if any(b not in core.BANDS or p < 1 for b, p in tasks) or len(set(tasks)) != len(
        tasks
    ):
        raise ValueError("Invalid/duplicate scheduler tasks")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    ctx = mp.get_context("spawn")
    outbox = ctx.Queue()
    active = {}
    pending = list(tasks)
    completed, samples = 0, []
    start, last_report = time.monotonic(), 0.0
    peak_worker = (
        0.65  # Conservative initial estimate, replaced upwards by measurement.
    )
    error = None
    interrupted = False
    hardware_cap = max(1, min(10, (psutil.cpu_count(logical=True) or 2) - 2))
    target_workers = min(workers, hardware_cap)
    critical_since = None
    try:
        while pending or any(w["task"] is not None for w in active.values()):
            available = psutil.virtual_memory().available / GIB
            if (root / "STOP").exists():
                interrupted = True
            if available < critical:
                critical_since = critical_since or time.monotonic()
            else:
                critical_since = None
            if critical_since and time.monotonic() - critical_since >= 5:
                raise MemoryError(
                    "Critical system memory pressure; completed results are safe; resume with fewer workers"
                )
            # Initial ramp has at most two workers; spawn additional workers
            # after first results demonstrate actual memory usage.
            desired = (
                target_workers if completed or not ramp else min(2, target_workers)
            )
            if (
                pending
                and not interrupted
                and len(active) < desired
                and available >= reserve + peak_worker * 1.3
            ):
                inbox = ctx.Queue(maxsize=1)
                process = ctx.Process(
                    target=worker,
                    args=(
                        str(root / "cache"),
                        manifest["config"]["seed"],
                        inbox,
                        outbox,
                    ),
                )
                process.start()
                active[process.pid] = {
                    "process": process,
                    "inbox": inbox,
                    "task": None,
                    "band": None,
                }
            if (
                pending
                and not active
                and not interrupted
                and available < reserve + peak_worker * 1.3
            ):
                raise MemoryError("Insufficient available RAM to start safely")
            for pid, w in list(active.items()):
                if not w["process"].is_alive():
                    raise RuntimeError(
                        f"Worker {pid} exited unexpectedly (task {w['task']})"
                    )
                if w["task"] is None:
                    if available < reserve or interrupted:
                        w["inbox"].put(None)
                        w["process"].join(timeout=5)
                        if w["process"].is_alive():
                            w["process"].terminate()
                            w["process"].join()
                        w["inbox"].close()
                        del active[pid]
                        continue
                    if pending:
                        same = next(
                            (i for i, t in enumerate(pending) if t[0] == w["band"]), 0
                        )
                        task = pending.pop(same)
                        w["inbox"].put(task)
                        w["task"], w["band"] = task, task[0]
            try:
                pid, status, result = outbox.get(timeout=1)
                if status != "OK":
                    raise RuntimeError(result)
                expected_task = active[pid]["task"]
                if (result["band"], result["permutation_id"]) != expected_task:
                    raise RuntimeError("Worker result/task mismatch")
                on_result(result)
                completed += 1
                samples.append(result)
                peak_worker = max(peak_worker, result["worker_peak_gib"])
                active[pid]["task"] = None
            except queue.Empty:
                pass
            if time.monotonic() - last_report > 10:
                progress = {
                    "status": "DRAINING" if interrupted else "RUNNING",
                    "completed_this_call": completed,
                    "pending_this_call": len(pending),
                    "active_workers": len(active),
                    "busy_workers": sum(w["task"] is not None for w in active.values()),
                    "available_ram_gib": available,
                    "worker_peak_gib": peak_worker,
                    "elapsed_seconds": time.monotonic() - start,
                }
                dump(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_report = time.monotonic()
            if interrupted and not any(w["task"] is not None for w in active.values()):
                break
    except BaseException as exc:
        error = exc
    finally:
        for w in active.values():
            if w["task"] is None and w["process"].is_alive():
                w["inbox"].put(None)
        for w in active.values():
            w["process"].join(timeout=2)
            if w["process"].is_alive():
                w["process"].terminate()
                w["process"].join()
            w["inbox"].close()
        outbox.close()
    if error:
        raise error
    dump(
        root / "progress.json",
        {
            "status": "STOPPED" if interrupted else "IDLE",
            "completed_this_call": completed,
            "active_workers": 0,
            "elapsed_seconds": time.monotonic() - start,
        },
    )
    return {
        "seconds": time.monotonic() - start,
        "completed": completed,
        "interrupted": interrupted,
        "samples": samples,
        "peak_worker_gib": max([s["worker_peak_gib"] for s in samples], default=0.0),
    }


def benchmark(root, candidates):
    verify(root, require_validation=True)
    if not candidates or any(not 1 <= c <= 10 for c in candidates):
        raise ValueError("Benchmark worker counts must be 1..10")
    if (root / "STOP").exists():
        raise ValueError("Remove STOP explicitly before benchmarking")
    report = []
    expected = {}
    # Same eight full band tasks for every candidate; calibration is not
    # counted repeatedly as inferential evidence.
    tasks = [(b, 1) for b in core.BANDS]
    for count in candidates:
        available = psutil.virtual_memory().available / GIB
        worker_estimate = max([r["peak_worker_gib"] for r in report], default=0.65)
        if available < 2.5 + count * worker_estimate * 1.3:
            print(f"skip {count} workers: insufficient memory headroom", flush=True)
            continue

        def accept(item):
            key = item["band"]
            if key in expected:
                np.testing.assert_array_equal(
                    [item["raw"], item["smoothed"]], expected[key]
                )
            else:
                expected[key] = [item["raw"], item["smoothed"]]

        result = schedule(root, tasks, count, accept, ramp=False)
        report.append(
            {
                "requested_workers": count,
                "seconds": result["seconds"],
                "completed": result["completed"],
                "peak_worker_gib": result["peak_worker_gib"],
                "tasks_per_minute": 60 * result["completed"] / result["seconds"],
            }
        )
        dump(root / "benchmark.json", {"status": "RUNNING", "profiles": report})
        print(json.dumps(report[-1]), flush=True)
    if not report:
        raise MemoryError("No safe benchmark configuration")
    best = max(report, key=lambda r: (r["tasks_per_minute"], -r["requested_workers"]))
    dump(
        root / "benchmark.json",
        {
            "status": "COMPLETE",
            "profiles": report,
            "recommended_workers": best["requested_workers"],
            "note": "Small calibration including process/cache startup; full-run ETA uses live throughput.",
        },
    )


def run(root, workers, limit):
    manifest = verify(root, require_validation=True)
    if (root / "STOP").exists():
        raise ValueError("STOP exists; remove it explicitly to resume")
    if workers == "auto":
        profile = root / "benchmark.json"
        workers = (
            json.loads(profile.read_text(encoding="utf-8"))["recommended_workers"]
            if profile.exists()
            else 2
        )
    workers = int(workers)
    if not 1 <= workers <= 10:
        raise ValueError("workers must be auto or 1..10")
    db = open_db(root, manifest["fingerprint"])
    try:
        saved = read_records(db)
        done = {(r["band"], r["permutation_id"]) for r in saved}
        target = manifest["config"]["n_permutations"]
        all_tasks = [(b, pid) for pid in range(1, target + 1) for b in core.BANDS]
        if not done.issubset(set(all_tasks)):
            raise ValueError("Out-of-range checkpoint tasks")
        pending = [task for task in all_tasks if task not in done]
        if limit is not None:
            if limit < 1:
                raise ValueError("max-tasks must be positive")
            pending = pending[:limit]
        dump(
            root / "status.json",
            {
                "status": "RUNNING",
                "completed_tasks": len(done),
                "target_tasks": target * 8,
            },
        )
        result = schedule(root, pending, workers, lambda r: record(db, r))
        n = db.execute("SELECT COUNT(*) FROM results").fetchone()[0]
        state = "COMPLETE" if n == target * 8 else "PARTIAL"
        dump(
            root / "status.json",
            {
                "status": state,
                "completed_tasks": n,
                "target_tasks": target * 8,
                "last_call_seconds": result["seconds"],
                "workers_requested": workers,
            },
        )
    except BaseException as exc:
        dump(
            root / "status.json",
            {
                "status": "INTERRUPTED",
                "error": str(exc),
                "completed_tasks": db.execute(
                    "SELECT COUNT(*) FROM results"
                ).fetchone()[0],
            },
        )
        raise
    finally:
        db.close()
    summarize(root)


def summarize(root):
    manifest = verify(root, require_validation=True)
    db = open_db(root, manifest["fingerprint"])
    try:
        records = read_records(db)
    finally:
        db.close()
    target = manifest["config"]["n_permutations"]
    grouped = {}
    for r in records:
        if r["band"] not in core.BANDS or not 1 <= r["permutation_id"] <= target:
            raise ValueError("Out-of-range checkpoint task")
        grouped.setdefault(r["permutation_id"], {})[r["band"]] = r
    complete = sorted(
        pid for pid, rows in grouped.items() if set(rows) == set(core.BANDS)
    )
    for pid in complete:
        if len({r["label_hash"] for r in grouped[pid].values()}) != 1:
            raise ValueError("Cross-band label synchronization failure")
    patient_values = np.load(root / "cache/observed_patient_metrics.npy")
    observed = patient_values.mean(axis=2)
    paired = core.exact_paired_test(
        np.transpose(patient_values[:, 1] - patient_values[:, 0], (1, 0, 2))
    )
    pair_rows = []
    for b, band in enumerate(core.BANDS):
        for m, metric in enumerate(core.METRICS):
            pair_rows.append(
                {
                    "band": band,
                    "metric": metric,
                    "mean_delta": paired[0][b, m],
                    "p_one_sided_exact": paired[1][b, m],
                    "p_maxT_8bands": paired[2][b, m],
                    "n_assignments": 8192,
                    "scope": "conditional_fixed_OOF_patient_sign_exchangeability",
                }
            )
    pd.DataFrame(pair_rows).to_csv(root / "paired_graph_exact.csv", index=False)
    dump(
        root / "completion.json",
        {
            "completed_band_tasks": len(records),
            "complete_synchronized_permutations": len(complete),
            "target_permutations": target,
            "status": "COMPLETE" if len(complete) == target else "PARTIAL",
        },
    )
    # Do not expose provisional p-values as formal results during optional stopping.
    if len(complete) != target or len(records) != target * 8:
        print(
            f"PARTIAL: {len(records)}/{target * 8} tasks; {len(complete)}/{target} complete synchronized IDs",
            flush=True,
        )
        return
    null = np.array(
        [
            [
                [grouped[pid][band][t] for t in ("raw", "smoothed")]
                for band in core.BANDS
            ]
            for pid in complete
        ]
    )
    p, adjusted, joint = core.permutation_summary(observed, null)
    rows = []
    for b, band in enumerate(core.BANDS):
        for t, track in enumerate(("raw", "smoothed")):
            for m, metric in enumerate(core.METRICS):
                rows.append(
                    {
                        "band": band,
                        "track": track,
                        "metric": metric,
                        "observed": observed[b, t, m],
                        "p_raw_one_sided": p[b, t, m],
                        "p_maxT_8bands_same_track": adjusted[b, t, m],
                        "p_maxT_16_band_track": joint[b, t, m],
                        "n_permutations": target,
                        "n_patients": 13,
                        "run_purpose": manifest["config"]["run_purpose"],
                        "inferential_role": "primary"
                        if metric == "AUPRC" and track == "raw"
                        else "secondary",
                    }
                )
    pd.DataFrame(rows).to_csv(root / "permutation_summary.csv", index=False)
    np.save(root / "null_metrics.npy", null)
    print("COMPLETE: permutation_summary.csv", flush=True)


def main():
    global OUTPUT_BASE, LABELS, FEATURE_ROOT, REFERENCE_RUN, GRAPH_RUN, RUN_PURPOSE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "validate", "benchmark", "run", "summarize", "status"),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-base",
        type=Path,
        default=OUTPUT_BASE,
        help="Run directory must be a strict child",
    )
    parser.add_argument(
        "--labels", type=Path, help="Required for prepare: final cohort labels CSV"
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        help="Required for prepare: final reviewed feature tables",
    )
    parser.add_argument(
        "--reference-run", type=Path, help="Required for prepare: native LOPO outputs"
    )
    parser.add_argument(
        "--graph-run", type=Path, help="Required for prepare: graph smoothing outputs"
    )
    parser.add_argument("--run-purpose", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=core.SEED)
    parser.add_argument("--workers", default="auto")
    parser.add_argument(
        "--benchmark-workers", type=int, nargs="+", default=[2, 4, 6, 8, 10]
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        help="Bounded smoke run; omitted only for a deliberate full run",
    )
    args = parser.parse_args()
    OUTPUT_BASE = args.output_base.resolve()
    LABELS, FEATURE_ROOT = args.labels, args.feature_root
    REFERENCE_RUN, GRAPH_RUN = args.reference_run, args.graph_run
    RUN_PURPOSE = args.run_purpose
    if args.command == "prepare" and any(
        x is None for x in (LABELS, FEATURE_ROOT, REFERENCE_RUN, GRAPH_RUN)
    ):
        parser.error(
            "prepare requires --labels, --feature-root, --reference-run and --graph-run"
        )
    root = safe_output(args.run_dir)
    if args.command == "prepare":
        prepare(root, args.permutations, args.seed)
        return
    if args.command == "status":
        for name in ("status.json", "completion.json", "progress.json"):
            if (root / name).exists():
                print(name, (root / name).read_text(encoding="utf-8"))
        return
    with RunLock(root):
        if args.command == "validate":
            validate(root)
        elif args.command == "benchmark":
            benchmark(root, args.benchmark_workers)
        elif args.command == "run":
            run(root, args.workers, args.max_tasks)
        else:
            summarize(root)


if __name__ == "__main__":
    mp.freeze_support()
    main()
