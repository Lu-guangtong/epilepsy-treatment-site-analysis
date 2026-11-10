"""Apply maximum-statistic correction across eight bipolar-pair frequency bands."""

from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd


ROOT = output_root()
MODEL_ROOT = ROOT / "data_derived" / "electrode_pair_modeling_by_band"
DEFAULT_PERMUTATIONS = MODEL_ROOT / "permutation_results_by_band.csv"
DEFAULT_METRICS = MODEL_ROOT / "lopo_electrode_metrics_by_band.csv"
DEFAULT_OUTPUT = MODEL_ROOT / "electrode_primary_8band_multiple_comparison_correction.csv"
ALLOWED_OUTPUT_ROOTS = (MODEL_ROOT.resolve(), (ROOT / "checks").resolve())
PRIMARY_BANDS = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "low_gamma",
    "high_gamma",
    "ripple",
    "fast_ripple",
)


def assert_allowed_output(path: Path) -> None:
    assert_output_path(path)
    resolved = path.resolve()
    if not any(resolved == root or root in resolved.parents for root in ALLOWED_OUTPUT_ROOTS):
        raise SystemExit(f"Output is outside allowed roots: {resolved}")
    if source_root("aux").resolve() == resolved or source_root("aux").resolve() in resolved.parents:
        raise SystemExit("Writing to the configured auxiliary input root is forbidden")


def compute_primary_correction(
    permutations: pd.DataFrame,
    metrics: pd.DataFrame,
    bands: tuple[str, ...] = PRIMARY_BANDS,
) -> pd.DataFrame:
    permutation_required = {"permutation_id", "frequency_band", "macro_AUPRC"}
    metric_required = {"frequency_band", "macro_AUPRC"}
    if missing := permutation_required.difference(permutations.columns):
        raise SystemExit(f"Permutation table missing columns: {sorted(missing)}")
    if missing := metric_required.difference(metrics.columns):
        raise SystemExit(f"Metric table missing columns: {sorted(missing)}")

    selected = permutations.loc[permutations["frequency_band"].isin(bands)].copy()
    if set(selected["frequency_band"]) != set(bands):
        raise SystemExit("Permutation table does not contain all eight primary bands")
    counts = selected.groupby("permutation_id")["frequency_band"].nunique()
    if counts.empty or not counts.eq(len(bands)).all():
        raise SystemExit("Each permutation must contain every primary band exactly once")
    if selected.duplicated(["permutation_id", "frequency_band"]).any():
        raise SystemExit("Duplicate permutation-band rows detected")
    n_permutations = int(counts.size)
    if n_permutations != 1000:
        raise SystemExit(f"Expected 1000 permutations, found {n_permutations}")
    if not np.isfinite(selected["macro_AUPRC"].to_numpy(dtype=float)).all():
        raise SystemExit("Permutation statistics contain non-finite values")

    observed = metrics.set_index("frequency_band")["macro_AUPRC"]
    if not set(bands).issubset(observed.index):
        raise SystemExit("Observed metrics do not contain all eight primary bands")
    null_max = selected.groupby("permutation_id")["macro_AUPRC"].max()
    rows: list[dict[str, object]] = []
    for band in bands:
        band_null = selected.loc[selected["frequency_band"].eq(band), "macro_AUPRC"]
        observed_value = float(observed.loc[band])
        raw_p = float((int((band_null >= observed_value).sum()) + 1) / (n_permutations + 1))
        max_t_p = float((int((null_max >= observed_value).sum()) + 1) / (n_permutations + 1))
        rows.append(
            {
                "frequency_band": band,
                "observed_macro_AUPRC": observed_value,
                "n_permutations": n_permutations,
                "permutation_p_raw_macro_AUPRC": raw_p,
                "permutation_p_fwer_maxT_macro_AUPRC": max_t_p,
                "multiple_comparison_scope": "eight_primary_frequency_bands",
                "null_source": str(DEFAULT_PERMUTATIONS),
                "model_refit": False,
                "new_permutations_generated": False,
            }
        )
    return pd.DataFrame.from_records(rows)


def write_atomic(frame: pd.DataFrame, output: Path, overwrite: bool) -> None:
    assert_allowed_output(output)
    if output.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace it: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}_", suffix=".tmp", dir=output.parent
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=Path, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    correction = compute_primary_correction(
        pd.read_csv(args.permutations), pd.read_csv(args.metrics)
    )
    print(correction.to_string(index=False))
    if args.apply:
        write_atomic(correction, args.out, args.overwrite)
        print(f"wrote={args.out}")
    else:
        print("dry_run=true")


if __name__ == "__main__":
    main()
