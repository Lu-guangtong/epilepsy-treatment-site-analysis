"""Build regional connectivity features with tie-aware within-patient ranks."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd


ROOT = output_root()
ANALYSIS_ROOT = ROOT / "data_derived" / "regional_analysis"
DEFAULT_TRIAL = (
    ROOT / "data_derived" / "region_features" / "region_trial_features_long.csv"
)
DEFAULT_CONTRAST = (
    ROOT
    / "data_derived"
    / "region_features"
    / "region_trial_contrast_features_long.csv"
)
DEFAULT_OUT = ANALYSIS_ROOT / "region_features"
EXTRACTOR = REPO_ROOT / "scripts" / "extract_region_network_features.py"
RUN_ID = "regional_analysis"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def assert_analysis_output(out_dir: Path) -> None:
    assert_output_path(out_dir)
    if not _is_relative_to(out_dir, ANALYSIS_ROOT):
        raise SystemExit(f"Output must remain under {ANALYSIS_ROOT}: {out_dir}")


def load_extractor():
    spec = importlib.util.spec_from_file_location(
        "region_feature_extractor", EXTRACTOR
    )
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Cannot load {EXTRACTOR}")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temporary = Path(handle.name)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", type=Path, default=DEFAULT_TRIAL)
    parser.add_argument("--contrast", type=Path, default=DEFAULT_CONTRAST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_analysis_output(args.out)
    for path in (args.trial, args.contrast, EXTRACTOR):
        if not path.is_file():
            raise SystemExit(f"Missing required input: {path}")

    extractor = load_extractor()
    fingerprint_payload: dict[str, object] = {
        "run_id": RUN_ID,
        "stage": "region_features",
        "rank_tie_method": extractor.RANK_TIE_METHOD,
        "top_fraction_threshold": extractor.TOP_FRACTION_THRESHOLD,
        "inputs": {
            str(args.trial.resolve()): file_sha256(args.trial),
            str(args.contrast.resolve()): file_sha256(args.contrast),
            str(EXTRACTOR.resolve()): file_sha256(EXTRACTOR),
            str(Path(__file__).resolve()): file_sha256(Path(__file__)),
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    encoded = json.dumps(
        fingerprint_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    manifest = {
        **fingerprint_payload,
        "fingerprint_sha256": hashlib.sha256(encoded).hexdigest(),
        "command": [sys.executable, *sys.argv],
    }
    manifest_path = args.out / "run_manifest.json"
    output_path = args.out / "region_patient_features_unlabeled.csv"

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.out.exists() and not args.resume:
        raise SystemExit(f"Output directory already exists; use --resume: {args.out}")
    if args.resume:
        if not manifest_path.is_file() or not output_path.is_file():
            raise SystemExit("Cannot resume without both manifest and patient feature table")
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint_sha256") != manifest["fingerprint_sha256"]:
            raise SystemExit("Resume fingerprint mismatch")
        print(f"validated={output_path}")
        return

    trial = pd.read_csv(args.trial)
    contrast = pd.read_csv(args.contrast)
    patient = extractor.aggregate_patient_region_features(trial, contrast)
    patient = patient.sort_values(
        [
            "patient",
            "desikan_killiany_label",
            "tissue_class",
            "is_white_csf_ventricle_unknown",
        ],
        kind="mergesort",
    ).reset_index(drop=True)
    args.out.mkdir(parents=True, exist_ok=False)
    patient.to_csv(output_path, index=False, encoding="utf-8-sig")
    manifest.update(
        {
            "status": "complete",
            "n_trial_rows": len(trial),
            "n_contrast_rows": len(contrast),
            "n_patient_region_rows": len(patient),
            "n_columns": len(patient.columns),
            "output_sha256": {str(output_path): file_sha256(output_path)},
        }
    )
    write_json_atomic(manifest, manifest_path)
    print(
        f"wrote={output_path} rows={len(patient)} columns={len(patient.columns)}"
    )


if __name__ == "__main__":
    main()
