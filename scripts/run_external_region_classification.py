"""Audit, extract and classify external ds004100 DK brain regions.

Default: read-only planning. --smoke-test executes a four-patient/two-seizure
test under checks. --execute explicitly runs the selected full-data stage;
the project's external code-review requirement applies before that invocation.
"""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
ROOT = output_root()
sys.path.insert(0, str(REPO_ROOT / "src"))

import mne  # noqa: E402
import mne_connectivity  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from seeg.external_region_contract import (  # noqa: E402
    BANDS, RULE_ID, digest_json, input_summary, load_config, load_inputs,
    make_plan, read_json, selected_signals, sha256, write_json,
)
from seeg.external_region_features import extract_band  # noqa: E402
from seeg.external_region_model import model_band  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_root() / "external_region_ds004100.yaml")
    parser.add_argument("--stage", choices=["audit", "features", "model", "all"], default="all")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--smoke-test", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--feature-run", type=Path, help="Completed feature run for --stage model")
    parser.add_argument("--bands", nargs="+", choices=list(BANDS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verify-signal-hashes", action="store_true")
    return parser.parse_args(argv)


def validate_output(path: Path, smoke: bool) -> Path:
    assert_output_path(path)
    path = path.resolve()
    allowed = [ROOT / "checks"] if smoke else [ROOT / "checks", ROOT / "external"]
    if not any(path.is_relative_to(p.resolve()) and path != p.resolve() for p in allowed):
        raise ValueError(f"Output is outside the permitted run directories: {path}")
    return path


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    configured_bands = cfg["primary_bands"] + cfg["supplementary_bands"]
    bands = args.bands or (["delta"] if args.smoke_test else configured_bands)
    if len(set(bands)) != len(bands):
        raise ValueError("Duplicate requested frequency bands")
    if set(bands) - set(configured_bands):
        raise ValueError("Requested band is not configured; amend/review the config explicitly")
    execute = (args.execute or args.smoke_test) and args.stage != "audit"
    inputs = load_inputs(cfg, verify_fif_hashes=args.verify_signal_hashes or execute)
    signals = selected_signals(inputs, bands, args.smoke_test)
    summary = input_summary(inputs, signals, bands)
    summary["smoke_test"] = args.smoke_test
    summary["fif_hashes_verified"] = bool(args.verify_signal_hashes or execute)
    if not execute:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.stage == "model" and args.feature_run is None:
        raise ValueError("--stage model requires --feature-run")
    if args.stage != "model" and args.feature_run is not None:
        raise ValueError("--feature-run applies only to --stage model")
    default_out = (ROOT / "checks/external_region_classification_smoke"
                   if args.smoke_test else ROOT / cfg["default_output"])
    out = validate_output(args.out or default_out, args.smoke_test)
    # Protect every source subtree even if the caller supplies a custom output.
    for path in cfg["inputs"].values():
        if Path(path).resolve().is_relative_to(out):
            raise ValueError("Output would overlap an input directory")
    code_paths = [REPO_ROOT / "src/seeg/external_region_contract.py",
                  REPO_ROOT / "src/seeg/external_region_features.py",
                  REPO_ROOT / "scripts/extract_region_network_features.py",
                  Path(__file__).resolve()]
    feature_payload = {
        "schema_version": 1, "input_hashes": inputs.hashes,
        "signal_keys": [[r["patient"], r["trial"]] for r in signals],
        "bands": bands, "smoke": args.smoke_test,
        "code_hashes": {str(p.relative_to(REPO_ROOT)): sha256(p) for p in code_paths},
        "versions": {"mne": mne.__version__, "mne_connectivity": mne_connectivity.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    feature_fingerprint = digest_json(feature_payload)
    model_payload = {"feature_fingerprint": feature_fingerprint, "modeling": cfg["modeling"],
                     "configured_bands": configured_bands,
                     "config_sha256": sha256(args.config),
                     "sklearn": sklearn.__version__,
                     "code_sha256": sha256(REPO_ROOT / "src/seeg/external_region_model.py"),
                     "formal_model_sha256": sha256(REPO_ROOT / "scripts/run_region_lopo_modeling.py")}
    fingerprint = digest_json(model_payload)
    if args.stage == "model":
        source = read_json(args.feature_run / "run_manifest.json")
        if (source["status"] not in ("features_complete", "complete")
                or source["feature_fingerprint"] != feature_fingerprint):
            raise ValueError("Feature run differs in sources, code, bands or smoke/full scope")
        for relative_path, expected_hash in source.get("output_hashes", {}).items():
            if sha256(args.feature_run / relative_path) != expected_hash:
                raise ValueError(f"Feature-run output hash mismatch: {relative_path}")
        if not source.get("output_hashes"):
            raise ValueError("Feature run lacks a completed output hash manifest")
        feature_root = args.feature_run / "features"
    else:
        feature_root = out / "features"
    if out.exists():
        manifest_path = out / "run_manifest.json"
        if not args.resume or not manifest_path.is_file():
            raise FileExistsError("Use a new run directory or --resume with matching provenance")
        previous = read_json(manifest_path)
        if previous["fingerprint"] != fingerprint or previous["stage"] != args.stage:
            raise ValueError("Run resume fingerprint or stage mismatch")
    else:
        out.mkdir(parents=True)
    manifest = {"status": "running", "stage": args.stage, "label_rule_id": RULE_ID,
                "fingerprint": fingerprint, "feature_fingerprint": feature_fingerprint,
                "feature_provenance": feature_payload, "model_provenance": model_payload,
                "summary": summary, "command": [sys.executable, *sys.argv],
                "formal_code_review": "external_review_required_before_full_run"}
    write_json(out / "run_manifest.json", manifest)
    plan, availability = make_plan(signals, bands)
    plan.to_csv(out / "matrix_plan.csv", index=False, encoding="utf-8-sig")
    availability.to_csv(out / "frequency_availability.csv", index=False, encoding="utf-8-sig")
    n_permutations = 0 if args.smoke_test else cfg["modeling"]["n_permutations"]
    n_bootstrap = 50 if args.smoke_test else cfg["modeling"]["n_bootstrap"]
    try:
        with threadpool_limits(limits=1):
            if args.stage in ("features", "all"):
                for band in bands:
                    extract_band(signals, inputs.anatomy, band, feature_root / band,
                                 feature_fingerprint, resume=args.resume)
                manifest["status"] = "features_complete"
                write_json(out / "run_manifest.json", manifest)
            if args.stage in ("model", "all"):
                results = []
                for band in bands:
                    result = model_band(feature_root / band, inputs.labels, band, cfg["modeling"],
                        out / "model" / band, feature_fingerprint, fingerprint,
                        args.resume, n_permutations, n_bootstrap)
                    result["analysis_tier"] = "primary" if band in cfg["primary_bands"] else "supplementary"
                    p_value = result["permutation_p_AUPRC"]
                    result["permutation_p_bonferroni_configured_bands"] = (
                        min(1.0, len(configured_bands) * p_value) if p_value is not None else None)
                    results.append(result)
                pd.DataFrame(results).to_csv(out / "region_metrics_by_band.csv", index=False, encoding="utf-8-sig")
                manifest["status"] = "complete"
                manifest["results_are_smoke_only"] = args.smoke_test
        manifest["output_hashes"] = {str(p.relative_to(out)): sha256(p)
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "run_manifest.json"}
        write_json(out / "run_manifest.json", manifest)
        print(json.dumps({"status": manifest["status"], "output": str(out),
                          "smoke_only": args.smoke_test}, ensure_ascii=False))
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = f"{type(exc).__name__}: {exc}"
        write_json(out / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
