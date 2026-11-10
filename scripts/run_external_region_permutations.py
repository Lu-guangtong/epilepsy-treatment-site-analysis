"""Run resumable external regional permutations with bounded parallel workers."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import importlib.metadata
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = output_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import psutil  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from seeg import external_region_model as model  # noqa: E402
from seeg.external_region_contract import (  # noqa: E402
    digest_json, load_config, load_inputs, read_json, sha256, write_json,
)
from seeg.external_region_parallel import (  # noqa: E402
    ParallelPermutations, PermutationCheckpoint, ResourcePolicy, utc_now,
)


def verify_run(run: Path, cfg: dict):
    manifest = read_json(run / "run_manifest.json")
    feature = manifest["feature_provenance"]
    estimator = manifest["model_provenance"]
    if feature["smoke"] or manifest["stage"] != "all":
        raise ValueError("Continuation requires the full-data all-stage feature run")
    if (digest_json(feature) != manifest["feature_fingerprint"]
            or digest_json(estimator) != manifest["fingerprint"]
            or estimator["feature_fingerprint"] != manifest["feature_fingerprint"]):
        raise ValueError("Analysis provenance fingerprint mismatch")
    bands = cfg["primary_bands"] + cfg["supplementary_bands"]
    if (bands != feature["bands"] or bands != estimator["configured_bands"]
            or cfg["modeling"] != estimator["modeling"]
            or sha256(Path(cfg["config_path"])) != estimator["config_sha256"]):
        raise ValueError("Analysis configuration changed")
    code_hashes = {**feature["code_hashes"],
                  "src/seeg/external_region_model.py": estimator["code_sha256"],
                  "scripts/run_region_lopo_modeling.py": estimator["formal_model_sha256"]}
    for relative, expected in code_hashes.items():
        if sha256(REPO_ROOT / relative) != expected:
            raise ValueError(f"Original statistical/feature code changed: {relative}")
    for package, expected in {**feature["versions"], "scikit-learn": estimator["sklearn"]}.items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Package version changed: {package}")
    inputs = load_inputs(cfg)
    if inputs.hashes != feature["input_hashes"]:
        raise ValueError("Frozen input hashes changed")
    patients = manifest["summary"]["selected_patients"]
    verified_features = {}
    matrix_count = 0
    for band in bands:
        path = run / "features" / band / "feature_manifest.json"
        item = read_json(path)
        if (item["status"] != "complete" or item["frequency_band"] != band
                or item["fingerprint"] != manifest["feature_fingerprint"]
                or item["patients"] != patients or item["signal_keys"] != feature["signal_keys"]):
            raise ValueError(f"Incomplete or different feature scope: {band}")
        for entry in [*item["files"].values(), *item["matrices"]]:
            source = Path(entry["path"]).resolve()
            if not source.is_relative_to(path.parent.resolve()) or sha256(source) != entry["sha256"]:
                raise ValueError(f"Feature artifact hash/path mismatch: {source}")
        matrix_count += len(item["matrices"])
        verified_features[str(path.relative_to(run))] = sha256(path)
    if matrix_count != manifest["summary"]["matrix_count"]:
        raise ValueError("Matrix count changed")
    return manifest, inputs, bands, verified_features


def read_model_data(run, band, labels):
    feature = read_json(run / "features" / band / "feature_manifest.json")
    entry = feature["files"]["region_patient_features_unlabeled"]
    return model.join_model_data(pd.read_csv(entry["path"]), labels, band, feature["patients"])


def complete_band(run, band, cfg, fingerprint):
    directory = run / "model" / band
    if not (directory / "metrics.json").exists():
        return None
    result = read_json(directory / "metrics.json")
    if (result["frequency_band"] != band or result["n_bootstrap"] != cfg["n_bootstrap"]
            or result["n_permutations"] != cfg["n_permutations"]):
        raise ValueError("Completed model configuration mismatch")
    checkpoint = PermutationCheckpoint(directory / "permutation_checkpoint.json", band,
        result["AUPRC"], cfg["n_permutations"], fingerprint, True)
    if checkpoint.summary()["permutation_p_AUPRC"] != result["permutation_p_AUPRC"]:
        raise ValueError("Completed model p-value/checkpoint mismatch")
    predictions = pd.read_csv(directory / "region_predictions.csv")
    metrics = model.patient_metrics(predictions)
    for key in ("AUPRC", "AUROC", "normalized_AUPRC"):
        if abs(float(metrics[key].mean()) - result[key]) > 1e-12:
            raise ValueError("Completed predictions/metrics mismatch")
    for name in ("fold_audit.json", "model_labels.csv", "patient_metrics.csv", "sampling_strata_metrics.csv"):
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete model output: {name}")
    return result


@contextmanager
def single_writer(run):
    """An OS lock is released even if the parent is killed; the small file stays."""
    if os.name != "nt":
        raise RuntimeError("This continuation lock is intended for the Windows host")
    import msvcrt
    path = run / "parallel_writer.lock"
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            # The legacy CLI does not use our lock, so explicitly refuse it.
            for process in psutil.process_iter(["pid", "name"]):
                if "python" not in (process.info["name"] or "").lower():
                    continue
                try:
                    args = process.cmdline()
                    legacy = any(Path(x).name == "run_external_region_classification.py" for x in args)
                    if legacy and "--out" in args and Path(args[args.index("--out") + 1]).resolve() == run:
                        raise RuntimeError(f"Legacy serial writer is still alive: PID {process.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def check_equivalence(args, manifest, inputs, cfg, controller):
    out = args.qa_out.resolve()
    if out == ROOT / "checks" or not out.is_relative_to((ROOT / "checks").resolve()):
        raise ValueError("Equivalence check outputs must be under checks")
    band = "delta"
    saved = PermutationCheckpoint(args.run / "model" / band / "permutation_checkpoint.json",
        band, read_json(args.run / "model" / band / "metrics.json")["AUPRC"],
        cfg["modeling"]["n_permutations"], manifest["fingerprint"], True)
    if any(i not in saved.completed for i in args.check_ids):
        raise ValueError("Requested serial reference permutation is not saved")
    out.mkdir(parents=True, exist_ok=True)
    controller.progress_path = out / "progress.json"
    records = []
    started = time.perf_counter()

    def accept(record):
        expected = saved.completed[record["permutation_id"]]
        record.update(serial_AUPRC=expected, exactly_equal=record["AUPRC"] == expected)
        records.append(record)
        write_json(out / "equivalence_partial.json", records)
        if not record["exactly_equal"]:
            raise ValueError("Serial/parallel permutation results differ")

    controller.run_ids(read_model_data(args.run, band, inputs.labels), band,
                       cfg["modeling"], args.check_ids, accept)
    report = {"status": "PASS", "fingerprint": manifest["fingerprint"],
              "feature_fingerprint": manifest["feature_fingerprint"],
              "patients": cfg["expected"]["patients"], "policy": asdict(controller.policy),
              "elapsed_seconds": time.perf_counter() - started, "records": records,
              "verified_utc": utc_now()}
    write_json(out / "equivalence_result.json", report)
    print(f"Exact serial/parallel agreement: {len(records)} full-cohort permutations", flush=True)


def execute(args, manifest, inputs, cfg, bands, feature_hashes, controller):
    run = args.run
    record_path = run / "parallel_execution.json"
    provenance = {"started_utc": utc_now(), "pid": os.getpid(),
        "create_time": psutil.Process().create_time(), "command": [sys.executable, *sys.argv],
        "fingerprint": manifest["fingerprint"], "feature_fingerprint": manifest["feature_fingerprint"],
        "execution_code_sha256": {
            "src/seeg/external_region_parallel.py": sha256(REPO_ROOT / "src/seeg/external_region_parallel.py"),
            "scripts/run_external_region_permutations.py": sha256(Path(__file__))},
        "policy": asdict(controller.policy), "verified_feature_manifests": feature_hashes,
        "statistical_algorithm": "unchanged original functions; only permutation scheduling differs",
        "preserved_complete_files": {}, "initial_checkpoints": {}}
    with single_writer(run):
        if record_path.exists():
            previous = read_json(record_path)
            if previous["fingerprint"] != manifest["fingerprint"]:
                raise ValueError("Previous execution fingerprint mismatch")
            for relative, expected in previous.get("preserved_complete_files", {}).items():
                if sha256(run / relative) != expected:
                    raise ValueError("Previously preserved complete result changed")
            provenance["prior_execution"] = {k: previous[k] for k in ("started_utc", "pid", "status")}
        completed = {}
        for band in bands:
            result = complete_band(run, band, cfg["modeling"], manifest["fingerprint"])
            if result is not None:
                completed[band] = result
                for path in (run / "model" / band).iterdir():
                    if path.is_file():
                        provenance["preserved_complete_files"][str(path.relative_to(run))] = sha256(path)
            path = run / "model" / band / "permutation_checkpoint.json"
            if path.exists():
                saved = read_json(path)
                cp = PermutationCheckpoint(path, band, saved["observed_AUPRC"],
                    cfg["modeling"]["n_permutations"], manifest["fingerprint"], True)
                provenance["initial_checkpoints"][band] = {"completed": len(cp.completed), "sha256": sha256(path)}
        provenance["status"] = "running"
        write_json(record_path, provenance)
        manifest["status"] = "parallel_model_running"
        manifest["execution_continuation"] = str(record_path)
        write_json(run / "run_manifest.json", manifest)
        original = model.permutation_test
        try:
            # This changes only the current parent's call target. Spawned children
            # import the unmodified estimator and execute its pure functions.
            model.permutation_test = controller
            results = []
            for band in bands:
                if band in completed:
                    print(f"preserved completed band: {band}", flush=True)
                    result = dict(completed[band])
                else:
                    controller.report(status="observed_model", frequency_band=band, workers=0)
                    result = model.model_band(run / "features" / band, inputs.labels, band,
                        cfg["modeling"], run / "model" / band, manifest["feature_fingerprint"],
                        manifest["fingerprint"], True,
                        cfg["modeling"]["n_permutations"], cfg["modeling"]["n_bootstrap"])
                result["analysis_tier"] = "primary" if band in cfg["primary_bands"] else "supplementary"
                result["permutation_p_bonferroni_configured_bands"] = min(1.0, len(bands) * result["permutation_p_AUPRC"])
                results.append(result)
                provenance["completed_bands"] = [r["frequency_band"] for r in results]
                write_json(record_path, provenance)
            for relative, expected in provenance["preserved_complete_files"].items():
                if sha256(run / relative) != expected:
                    raise ValueError("Preserved completed result was modified")
            pd.DataFrame(results).to_csv(run / "region_metrics_by_band.csv", index=False, encoding="utf-8-sig")
            provenance.update(status="complete", completed_utc=utc_now())
            write_json(record_path, provenance)
            controller.report(status="complete", workers=0, in_flight=[])
            manifest.update(status="complete", results_are_smoke_only=False)
            manifest["output_hashes"] = {str(p.relative_to(run)): sha256(p)
                for p in sorted(run.rglob("*")) if p.is_file()
                and p.name not in ("run_manifest.json", "parallel_writer.lock")}
            write_json(run / "run_manifest.json", manifest)
        except BaseException as exc:
            provenance.update(status="failed", failure=f"{type(exc).__name__}: {exc}")
            write_json(record_path, provenance)
            controller.report(status="failed", failure=provenance["failure"], workers=0)
            manifest.update(status="parallel_failed", failure=provenance["failure"])
            write_json(run / "run_manifest.json", manifest)
            raise
        finally:
            model.permutation_test = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=config_root() / "external_region_ds004100.yaml")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-equivalence", action="store_true")
    parser.add_argument("--qa-out", type=Path)
    parser.add_argument("--check-ids", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--initial-workers", type=int, default=2,
                        help="Starting worker cap, still limited by available CPU and memory")
    args = parser.parse_args(argv)
    args.run = args.run.resolve()
    if not args.run.is_relative_to((ROOT / "external").resolve()):
        raise ValueError("Run must be inside the external run directory")
    if args.check_equivalence and args.qa_out is None:
        raise ValueError("--check-equivalence requires --qa-out under checks")
    policy = ResourcePolicy(max_workers=args.max_workers, initial_workers=args.initial_workers)
    cfg = load_config(args.config)
    manifest, inputs, bands, feature_hashes = verify_run(args.run, cfg)
    print("Verified original code, config, input hashes and all 1988 feature matrices", flush=True)
    controller = ParallelPermutations(policy, args.run / "parallel_progress.json")
    with threadpool_limits(limits=1):
        if args.check_equivalence:
            check_equivalence(args, manifest, inputs, cfg, controller)
        elif args.execute:
            execute(args, manifest, inputs, cfg, bands, feature_hashes, controller)
        else:
            print("Audit complete; no execution requested", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
