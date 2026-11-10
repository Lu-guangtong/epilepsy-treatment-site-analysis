"""Memory-limited scheduling of the unchanged external-region permutations.

Only the parent writes checkpoints. Permutation IDs, seeds and the complete
nested LOPO estimator are identical to external_region_model.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import time

import numpy as np
import psutil
from threadpoolctl import threadpool_limits

from . import external_region_model as model
from .external_region_contract import read_json, write_json

GIB = 1024 ** 3
_WORKER = {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ResourcePolicy:
    max_workers: int = 8
    reserve_gib: float = 1.5
    critical_gib: float = .75
    estimated_worker_gib: float = .40
    initial_workers: int = 2
    tasks_per_worker: int = 4

    def __post_init__(self):
        if (self.max_workers < 1 or self.initial_workers < 1 or self.tasks_per_worker < 1
                or not 0 < self.critical_gib < self.reserve_gib
                or self.estimated_worker_gib <= 0):
            raise ValueError("Invalid parallel resource policy")


def safe_worker_count(available_gib: float, logical_cpus: int,
                      worker_gib: float, policy: ResourcePolicy) -> int:
    """Zero means wait for memory; never force a worker below the reserve."""
    memory_cap = max(0, math.floor((available_gib - policy.reserve_gib) / worker_gib))
    cpu_cap = max(1, logical_cpus - 2)
    return min(policy.max_workers, cpu_cap, memory_cap)


def _valid_score(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


class PermutationCheckpoint:
    """Read serial prefixes and durably retain out-of-order parallel results."""

    def __init__(self, path: Path, band: str, observed: float,
                 target: int, fingerprint: str, resume: bool):
        self.path = Path(path)
        self.header = {"fingerprint": fingerprint, "frequency_band": band,
                       "observed_AUPRC": observed, "target_permutations": target}
        self.completed: dict[int, float] = {}
        if target < 0 or not _valid_score(observed):
            raise ValueError("Invalid permutation target or observed score")
        if not self.path.exists():
            return
        saved = read_json(self.path)
        if not resume or any(saved.get(k) != v for k, v in self.header.items()):
            raise ValueError("Permutation checkpoint mismatch")
        prefix = saved["null_AUPRC"]
        if not isinstance(prefix, list) or len(prefix) > target:
            raise ValueError("Invalid permutation prefix")
        for index, score in enumerate(prefix, 1):
            self._insert(index, score)
        pending = saved.get("completed_by_id", {})
        if not isinstance(pending, dict):
            raise ValueError("Invalid out-of-order permutation results")
        for key, score in pending.items():
            index = int(key)
            if str(index) != key:
                raise ValueError("Noncanonical permutation ID")
            self._insert(index, score)

    def _insert(self, index: int, score: float):
        if not 1 <= index <= self.header["target_permutations"] or not _valid_score(score):
            raise ValueError("Invalid permutation ID or score")
        if index in self.completed and self.completed[index] != score:
            raise ValueError("Conflicting permutation result")
        self.completed[index] = float(score)

    def add(self, index: int, score: float):
        self._insert(index, score)
        self.save()

    def save(self):
        prefix = []
        while len(prefix) + 1 in self.completed:
            prefix.append(self.completed[len(prefix) + 1])
        gaps = {str(i): self.completed[i] for i in sorted(self.completed) if i > len(prefix)}
        write_json(self.path, {**self.header, "null_AUPRC": prefix,
                              "completed_by_id": gaps, "checkpoint_schema": 2})

    def summary(self) -> dict:
        target = self.header["target_permutations"]
        if len(self.completed) != target:
            raise ValueError("Cannot summarize incomplete permutations")
        nulls = np.asarray([self.completed[i] for i in range(1, target + 1)])
        observed = self.header["observed_AUPRC"]
        return {"n_permutations": target,
                "permutation_p_AUPRC": (1 + int(np.sum(nulls >= observed))) / (target + 1)
                if target else None}


def _initialize_worker(data, band: str, cfg: dict, registry):
    process = psutil.Process()
    registry.put((process.pid, process.create_time()))
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[variable] = "1"
    limiter = threadpool_limits(limits=1)
    if os.name == "nt":
        process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    _WORKER.update(data=data, band=band, cfg=cfg, limiter=limiter)


def _run_one(index: int) -> dict:
    started = time.perf_counter()
    shuffled = model.permute_labels(_WORKER["data"], _WORKER["cfg"]["seed"], index)
    predictions, _ = model.run_lopo(shuffled, _WORKER["band"], _WORKER["cfg"], progress=False)
    score = float(model.patient_metrics(predictions).AUPRC.mean())
    memory = psutil.Process().memory_info()
    return {"permutation_id": index, "AUPRC": score, "pid": os.getpid(),
            "seconds": time.perf_counter() - started,
            "peak_worker_gib": max(memory.rss, getattr(memory, "peak_wset", memory.rss),
                                   getattr(memory, "private", memory.rss)) / GIB}


class ParallelPermutations:
    """Use short, bounded pools so RAM limits can shrink as well as grow.

    By default start with at most two workers, then grow by one per batch. Below
    the soft reserve, drain and recycle the pool; below the critical threshold,
    terminate only this pool and retry unfinished IDs after memory recovers.
    """

    def __init__(self, policy: ResourcePolicy, progress_path: Path | None = None):
        self.policy = policy
        self.progress_path = progress_path
        self.worker_gib = policy.estimated_worker_gib
        self.measured_peak_gib = 0.0
        self.next_limit = policy.initial_workers
        self.current = {}

    def report(self, **fields):
        self.current.update(fields, updated_utc=utc_now(), scheduler_pid=os.getpid())
        if self.progress_path:
            write_json(self.progress_path, self.current)

    def run_ids(self, data, band: str, cfg: dict, ids: list[int], on_result):
        pending = sorted(ids)
        if len(set(pending)) != len(pending) or any(i < 1 for i in pending):
            raise ValueError("Invalid pending permutation IDs")
        # Spawned imports must also see single-thread limits, before initialization.
        for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
            os.environ[variable] = "1"
        context = mp.get_context("spawn")
        while pending:
            available = psutil.virtual_memory().available / GIB
            capacity = safe_worker_count(available, psutil.cpu_count() or 1,
                                         self.worker_gib, self.policy)
            workers = min(capacity, self.next_limit, len(pending))
            if workers == 0:
                self.report(status="waiting_for_memory", frequency_band=band, workers=0,
                            available_gib=available, remaining=len(pending))
                time.sleep(5)
                continue
            batch = pending[:workers * self.policy.tasks_per_worker]
            print(f"parallel {band}: workers={workers} available={available:.2f} GiB "
                  f"remaining={len(pending)}", flush=True)
            self.report(status="running", frequency_band=band, workers=workers,
                        available_gib=available, estimated_worker_gib=self.worker_gib,
                        remaining=len(pending), in_flight=[])
            queued = list(batch)
            active = {}
            done_ids = set()
            drain = critical = False
            batch_started = time.perf_counter()
            registry = context.Queue()
            worker_processes = {}
            # Pool.terminate is a public API and is scoped to these own workers.
            with context.Pool(workers, initializer=_initialize_worker,
                              initargs=(data, band, cfg, registry)) as pool:
                while queued or active:
                    while True:
                        try:
                            pid, created = registry.get_nowait()
                            worker_processes[pid] = created
                        except Empty:
                            break
                    # Pool normally replaces crashed workers but can strand their
                    # AsyncResult; stop visibly instead of waiting indefinitely.
                    for pid, created in worker_processes.items():
                        try:
                            alive = psutil.Process(pid).create_time() == created
                        except psutil.NoSuchProcess:
                            alive = False
                        if not alive:
                            raise RuntimeError(f"Permutation worker exited unexpectedly: PID {pid}")
                    available = psutil.virtual_memory().available / GIB
                    if available < self.policy.critical_gib:
                        critical = True
                        print(f"parallel {band}: critical memory; recycling own workers", flush=True)
                        break
                    if available < self.policy.reserve_gib:
                        drain = True
                    while queued and len(active) < workers and not drain:
                        index = queued.pop(0)
                        active[index] = pool.apply_async(_run_one, (index,))
                    for index, future in list(active.items()):
                        if future.ready():
                            record = future.get()  # Exceptions stop; never fabricate a null score.
                            if record["permutation_id"] != index or not _valid_score(record["AUPRC"]):
                                raise ValueError("Invalid permutation worker result")
                            on_result(record)  # Atomically checkpoint BEFORE counting this ID done.
                            done_ids.add(index)
                            del active[index]
                            self.measured_peak_gib = max(self.measured_peak_gib, record["peak_worker_gib"])
                            self.worker_gib = max(.30, 1.30 * self.measured_peak_gib)
                            self.report(last_result=record)
                    self.report(status="draining_for_memory" if drain else "running",
                                available_gib=available, in_flight=sorted(active),
                                remaining=len(pending) - len(done_ids), worker_pids=sorted(worker_processes))
                    if drain and not active:
                        break
                    if active:
                        time.sleep(1)
            registry.close()
            pending = [i for i in pending if i not in done_ids]
            elapsed = time.perf_counter() - batch_started
            self.next_limit = max(1, workers - 1) if drain or critical else workers + 1
            self.report(status="recycling", workers=0, in_flight=[],
                        last_batch_completed=len(done_ids), last_batch_seconds=elapsed,
                        last_batch_workers=workers,
                        estimated_worker_gib=self.worker_gib, remaining=len(pending))
            print(f"parallel {band}: batch saved={len(done_ids)} elapsed={elapsed:.1f}s", flush=True)
            if critical:
                time.sleep(5)

    def __call__(self, data, band, cfg, observed, n_permutations, path, fingerprint, resume):
        checkpoint = PermutationCheckpoint(path, band, observed, n_permutations, fingerprint, resume)
        pending = [i for i in range(1, n_permutations + 1) if i not in checkpoint.completed]
        self.report(completed=len(checkpoint.completed), target=n_permutations,
                    fingerprint=fingerprint, frequency_band=band)

        def accept(record):
            checkpoint.add(record["permutation_id"], record["AUPRC"])
            self.report(completed=len(checkpoint.completed))
            print(f"permutation {band} saved {len(checkpoint.completed)}/{n_permutations} "
                  f"id={record['permutation_id']} seconds={record['seconds']:.1f}", flush=True)

        self.run_ids(data, band, cfg, pending, accept)
        self.report(status="band_complete", workers=0, in_flight=[])
        return checkpoint.summary()
