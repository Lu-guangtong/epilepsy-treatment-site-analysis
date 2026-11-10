"""Build a verified, label-free coordinate manifest for formal bipolar contacts."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
import ast
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "electrode_graph_smoothing.yaml"
COORDINATE_ROOT = ROOT / "data_derived" / "electrode_coordinates"
CHECK_ROOT = ROOT / "checks"
REQUIRED_TSV_COLUMNS = {"Channel", "SCS", "MNI", "World"}
COORDINATE_SPACES = ("SCS", "MNI", "World")
MIDPOINT_TOLERANCE_MM = 0.0011


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_path(path: Path) -> None:
    assert_output_path(path)
    resolved = _resolve(path)
    coordinate_root = _resolve(COORDINATE_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if resolved == AUX_ROOT or _is_relative_to(resolved, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {resolved}")
    if not (
        _is_relative_to(resolved, coordinate_root)
        or _is_relative_to(resolved, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def truthy(value: object) -> bool:
    if value is True:
        return True
    if value is False or value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "y", "1"}


def parse_vector(value: object, *, source: str) -> np.ndarray:
    try:
        vector = np.asarray(ast.literal_eval(str(value)), dtype=np.float64)
    except (ValueError, SyntaxError) as exc:
        raise SystemExit(f"{source}: invalid coordinate vector {value!r}") from exc
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise SystemExit(f"{source}: expected three finite coordinates, got {value!r}")
    return vector


def split_bipolar_name(name: str) -> tuple[str, str]:
    parts = str(name).split("-", maxsplit=1)
    if len(parts) != 2 or not all(parts):
        raise SystemExit(f"Invalid bipolar channel name: {name!r}")
    return parts[0], parts[1]


def read_coordinate_tsv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing coordinate TSV: {path}")
    frame = pd.read_csv(path, sep="\t")
    missing = sorted(REQUIRED_TSV_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if frame["Channel"].astype(str).duplicated().any():
        duplicates = frame.loc[
            frame["Channel"].astype(str).duplicated(keep=False), "Channel"
        ].astype(str)
        raise SystemExit(f"{path}: duplicate channels {sorted(set(duplicates))[:10]}")
    return frame


def load_brainstorm_scs_mm(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise SystemExit(f"Missing Brainstorm channel file: {path}")
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    channels = np.atleast_1d(mat.get("Channel"))
    output: dict[str, np.ndarray] = {}
    for channel in channels:
        name = str(channel.Name)
        loc = np.asarray(channel.Loc, dtype=np.float64).reshape(-1)
        if loc.shape != (3,) or not np.isfinite(loc).all():
            raise SystemExit(f"{path}: invalid Loc for {name}")
        if name in output:
            raise SystemExit(f"{path}: duplicate Brainstorm channel {name}")
        output[name] = loc * 1000.0
    return output


def load_formal_bipolars(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing formal bipolar manifest: {path}")
    frame = pd.read_csv(path)
    required = {
        "patient",
        "bipolar_channel",
        "anode",
        "cathode",
        "aseg_label",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
        "include_bipolar",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    frame = frame[frame["include_bipolar"].map(truthy)].copy()
    identity_columns = [
        "patient",
        "bipolar_channel",
        "anode",
        "cathode",
        "aseg_label",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
    ]
    conflicts = (
        frame[identity_columns]
        .drop_duplicates()
        .groupby(["patient", "bipolar_channel"], dropna=False)
        .size()
    )
    conflicts = conflicts[conflicts != 1]
    if not conflicts.empty:
        raise SystemExit(f"Conflicting formal bipolar metadata: {conflicts.index[:5].tolist()}")
    return (
        frame[identity_columns]
        .drop_duplicates(["patient", "bipolar_channel"])
        .sort_values(["patient", "bipolar_channel"])
        .reset_index(drop=True)
    )


def verify_midpoint(
    bipolar_vector: np.ndarray,
    anode_vector: np.ndarray,
    cathode_vector: np.ndarray,
    *,
    source: str,
) -> float:
    expected = (anode_vector + cathode_vector) / 2.0
    error = float(np.max(np.abs(bipolar_vector - expected)))
    if error > MIDPOINT_TOLERANCE_MM:
        raise SystemExit(
            f"{source}: bipolar coordinate is not the contact midpoint; "
            f"max_error_mm={error:.6g}"
        )
    return error


def build_patient_rows(
    patient: str,
    source_folder: str,
    formal: pd.DataFrame,
    sources: dict[str, Path],
) -> list[dict[str, object]]:
    single_path = sources["single_contact_tsv_root"] / f"{source_folder}.tsv"
    bipolar_path = sources["bipolar_tsv_root"] / f"{source_folder}.tsv"
    channel_path = (
        sources["brainstorm_data_root"]
        / source_folder
        / "Implantation_bipolar_2"
        / "channel.mat"
    )
    single = read_coordinate_tsv(single_path).set_index("Channel", drop=False)
    bipolar = read_coordinate_tsv(bipolar_path).set_index("Channel", drop=False)
    brainstorm = load_brainstorm_scs_mm(channel_path)
    patient_formal = formal[formal["patient"] == patient]
    rows: list[dict[str, object]] = []
    for item in patient_formal.itertuples(index=False):
        channel = str(item.bipolar_channel)
        anode, cathode = split_bipolar_name(channel)
        if anode != str(item.anode) or cathode != str(item.cathode):
            raise SystemExit(
                f"{patient} {channel}: formal anode/cathode do not match channel name"
            )
        if channel not in bipolar.index:
            raise SystemExit(f"{patient}: {channel} missing from {bipolar_path}")
        if anode not in single.index or cathode not in single.index:
            raise SystemExit(f"{patient} {channel}: endpoint missing from {single_path}")
        if channel not in brainstorm:
            raise SystemExit(f"{patient}: {channel} missing from {channel_path}")

        bipolar_row = bipolar.loc[channel]
        coordinate_values: dict[str, float] = {}
        midpoint_errors: dict[str, float] = {}
        for space in COORDINATE_SPACES:
            bipolar_vector = parse_vector(
                bipolar_row[space], source=f"{bipolar_path}:{channel}:{space}"
            )
            anode_vector = parse_vector(
                single.loc[anode, space], source=f"{single_path}:{anode}:{space}"
            )
            cathode_vector = parse_vector(
                single.loc[cathode, space], source=f"{single_path}:{cathode}:{space}"
            )
            midpoint_errors[space] = verify_midpoint(
                bipolar_vector,
                anode_vector,
                cathode_vector,
                source=f"{patient} {channel} {space}",
            )
            for axis, value in zip(("x", "y", "z"), bipolar_vector, strict=True):
                coordinate_values[f"{space.lower()}_{axis}_mm"] = float(value)

        brainstorm_error = float(
            np.max(
                np.abs(
                    brainstorm[channel]
                    - np.asarray(
                        [
                            coordinate_values["scs_x_mm"],
                            coordinate_values["scs_y_mm"],
                            coordinate_values["scs_z_mm"],
                        ]
                    )
                )
            )
        )
        if brainstorm_error > MIDPOINT_TOLERANCE_MM:
            raise SystemExit(
                f"{patient} {channel}: Brainstorm/TSV SCS mismatch "
                f"max_error_mm={brainstorm_error:.6g}"
            )
        rows.append(
            {
                "patient": patient,
                "source_folder": source_folder,
                "bipolar_channel": channel,
                "anode": anode,
                "cathode": cathode,
                **coordinate_values,
                "aseg_label": item.aseg_label,
                "desikan_killiany_label": item.desikan_killiany_label,
                "tissue_class": item.tissue_class,
                "is_white_csf_ventricle_unknown": truthy(
                    item.is_white_csf_ventricle_unknown
                ),
                "coordinate_definition": "midpoint_of_anode_and_cathode",
                "coordinate_unit": "mm",
                "source_bipolar_tsv": str(bipolar_path),
                "source_single_contact_tsv": str(single_path),
                "source_brainstorm_channel_mat": str(channel_path),
                "max_midpoint_error_mm": max(midpoint_errors.values()),
                "brainstorm_scs_error_mm": brainstorm_error,
                "qc_status": "pass",
            }
        )
    return rows


def build_coordinate_manifest(
    config: dict[str, object],
    patients: Iterable[str] | None = None,
) -> pd.DataFrame:
    source_config = {
        key: Path(value)
        for key, value in dict(config["coordinate_sources"]).items()
        if key.endswith("_root")
    }
    source_config["brainstorm_data_root"] = Path(
        dict(config["coordinate_sources"])["brainstorm_data_root"]
    )
    formal = load_formal_bipolars(
        Path(dict(config["coordinate_sources"])["formal_bipolar_manifest"])
    )
    folder_map = {
        str(patient): str(folder)
        for patient, folder in dict(config["patient_folder_map"]).items()
    }
    selected = set(folder_map if patients is None else [str(value) for value in patients])
    unknown = sorted(selected - set(folder_map))
    if unknown:
        raise SystemExit(f"Unknown patients: {unknown}")
    rows: list[dict[str, object]] = []
    for patient in folder_map:
        if patient not in selected:
            continue
        rows.extend(
            build_patient_rows(
                patient,
                folder_map[patient],
                formal,
                source_config,
            )
        )
    output = pd.DataFrame.from_records(rows)
    if output.duplicated(["patient", "bipolar_channel"]).any():
        raise SystemExit("Duplicate patient/bipolar rows in coordinate manifest")
    return output.sort_values(["patient", "bipolar_channel"]).reset_index(drop=True)


def atomic_write_csv(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
    assert_allowed_output_path(path)
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8-sig",
        newline="",
        suffix=".csv",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patients", nargs="+")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output = build_coordinate_manifest(config, args.patients)
    expected = int(dict(config["expected"])["formal_bipolar_coordinates"])
    if args.patients is None and len(output) != expected:
        raise SystemExit(f"Expected {expected} formal bipolar coordinates, got {len(output)}")
    counts = output.groupby("patient").size().to_dict()
    print(f"coordinate_rows={len(output)} patient_counts={counts}")
    print(
        "max_midpoint_error_mm="
        f"{output['max_midpoint_error_mm'].max():.6g} "
        "max_brainstorm_scs_error_mm="
        f"{output['brainstorm_scs_error_mm'].max():.6g}"
    )
    if not args.apply:
        print("dry_run=true; no files written")
        return
    output_path = args.out or Path(
        dict(config["coordinate_sources"])["output_manifest"]
    )
    atomic_write_csv(output, output_path, overwrite=args.overwrite)
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
