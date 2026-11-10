"""Evaluate graph diffusion and shaft-conditioned spatial controls."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from dataclasses import dataclass
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml


ROOT = output_root()
DEFAULT_CONFIG = config_root() / "regional_analysis" / "shaft_graph_controls.yaml"
DEFAULT_CORE_SCRIPT = REPO_ROOT / "scripts" / "run_electrode_graph_smoothing.py"
ANALYSIS_ROOT = ROOT / "data_derived" / "regional_analysis"
SMOKE_ROOT = ROOT / "checks" / "regional_analysis"
ALGORITHM_VERSION = "shaft_aware_graph_v1"
SCOPES = ("all", "same_shaft", "cross_shaft")
NULL_METHODS = ("within_shaft_shuffle", "circular_shift_reverse")


@dataclass(frozen=True)
class ScopeGraph:
    weights: np.ndarray
    transition: np.ndarray
    n_edges: int
    n_isolated: int


def _resolve(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(path: Path) -> None:
    """Restrict output to the analysis directory or a dedicated test directory."""
    assert_output_path(path)
    resolved = _resolve(path)
    allowed_parents = (_resolve(ANALYSIS_ROOT), _resolve(SMOKE_ROOT))
    if not any(_is_relative_to(resolved, parent) for parent in allowed_parents):
        raise SystemExit(f"Refusing to write outside analysis directories: {resolved}")
    if resolved in allowed_parents:
        raise SystemExit(f"Choose a script-specific child output directory: {resolved}")


def load_yaml(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = expand_config(yaml.safe_load(handle))
    if not isinstance(payload, dict):
        raise SystemExit(f"Configuration must be a mapping: {path}")
    return payload


def load_core_module(path: Path = DEFAULT_CORE_SCRIPT):
    spec = importlib.util.spec_from_file_location("shaft_graph_core", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot import graph core: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    """Hash one file or a directory tree, including relative names."""
    target = _resolve(path)
    if target.is_file():
        return file_sha256(target)
    if not target.is_dir():
        raise SystemExit(f"Protected/input path does not exist: {target}")
    digest = hashlib.sha256()
    files = sorted(item for item in target.rglob("*") if item.is_file())
    for item in files:
        relative = item.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
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


def shaft_contact_order(channel: str) -> tuple[float, str]:
    match = re.match(r"^([A-Za-z]+'?)(\d+)-([A-Za-z]+'?)(\d+)$", str(channel))
    if match is None or match.group(1) != match.group(3):
        raise SystemExit(f"Cannot parse within-shaft contact order from {channel!r}")
    midpoint = (int(match.group(2)) + int(match.group(4))) / 2.0
    return midpoint, str(channel)


def transition_from_weights(weights: np.ndarray) -> tuple[np.ndarray, int]:
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise SystemExit("Graph weights must be square")
    if not np.allclose(matrix, matrix.T, atol=1e-12) or np.any(matrix < 0.0):
        raise SystemExit("Graph weights must be symmetric and non-negative")
    degree = matrix.sum(axis=1)
    transition = np.zeros_like(matrix)
    connected = degree > 0.0
    transition[connected] = matrix[connected] / degree[connected, np.newaxis]
    transition[~connected, ~connected] = 1.0
    if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-12):
        raise SystemExit("Masked graph transition is not row stochastic")
    return transition, int(np.sum(~connected))


def mask_graph_by_scope(graph, scope: str, shaft_parser) -> ScopeGraph:
    if scope not in SCOPES:
        raise SystemExit(f"Unknown edge scope: {scope}")
    weights = np.asarray(graph.weights, dtype=np.float64).copy()
    shafts = np.asarray([shaft_parser(channel) for channel in graph.channels], dtype=object)
    same = shafts[:, np.newaxis] == shafts[np.newaxis, :]
    if scope == "same_shaft":
        weights[~same] = 0.0
    elif scope == "cross_shaft":
        weights[same] = 0.0
    np.fill_diagonal(weights, 0.0)
    transition, n_isolated = transition_from_weights(weights)
    n_edges = int(np.sum(np.triu(weights, k=1) > 0.0))
    return ScopeGraph(
        weights=weights,
        transition=transition,
        n_edges=n_edges,
        n_isolated=n_isolated,
    )


def prepare_inputs(
    config: dict[str, object],
    *,
    bands: Iterable[str] | None = None,
    patients: Iterable[str] | None = None,
):
    inputs = dict(config["inputs"])
    core = load_core_module(Path(str(inputs["core_graph_script"])))
    selected_bands = list(bands or config["frequency_bands"])
    predictions = core.load_predictions(
        Path(str(inputs["predictions"])), selected_bands, patients
    )
    coordinates = core.load_coordinates(Path(str(inputs["coordinates"])))
    aligned = core.align_coordinates(predictions, coordinates)
    label_counts = aligned.groupby(["patient", "bipolar_channel"])["y_true"].nunique()
    if not label_counts.eq(1).all():
        raise SystemExit("Contact labels differ across frequency bands")
    return core, aligned, selected_bands


def reference_frames(
    aligned: pd.DataFrame, first_band: str, shaft_parser
) -> dict[str, pd.DataFrame]:
    subset = aligned[aligned["frequency_band"].astype(str) == str(first_band)].copy()
    frames: dict[str, pd.DataFrame] = {}
    for patient, frame in subset.groupby("patient", sort=True):
        ordered = frame.sort_values("bipolar_channel").copy()
        ordered["shaft"] = [shaft_parser(value) for value in ordered["bipolar_channel"]]
        ordered["contact_order"] = [
            shaft_contact_order(value)[0] for value in ordered["bipolar_channel"]
        ]
        frames[str(patient)] = ordered.reset_index(drop=True)
    return frames


def build_graphs(
    frames: dict[str, pd.DataFrame],
    *,
    radii_mm: Iterable[float],
    n_neighbors: int,
    gaussian_sigma_mm: float,
    core,
) -> tuple[dict[float, dict[str, object]], dict[float, dict[str, dict[str, ScopeGraph]]], pd.DataFrame]:
    graphs: dict[float, dict[str, object]] = {}
    scoped: dict[float, dict[str, dict[str, ScopeGraph]]] = {}
    audit_rows: list[dict[str, object]] = []
    for radius in radii_mm:
        radius_value = float(radius)
        graphs[radius_value] = {}
        scoped[radius_value] = {}
        for patient, frame in frames.items():
            graph = core.build_mutual_knn_graph(
                frame["bipolar_channel"].astype(str),
                frame[["scs_x_mm", "scs_y_mm", "scs_z_mm"]].to_numpy(),
                n_neighbors=n_neighbors,
                max_distance_mm=radius_value,
                gaussian_sigma_mm=gaussian_sigma_mm,
            )
            graphs[radius_value][patient] = graph
            scope_graphs = {
                scope: mask_graph_by_scope(graph, scope, core.shaft_name)
                for scope in SCOPES
            }
            if (
                scope_graphs["same_shaft"].n_edges
                + scope_graphs["cross_shaft"].n_edges
                != scope_graphs["all"].n_edges
            ):
                raise SystemExit("Same- and cross-shaft edge masks do not partition all edges")
            scoped[radius_value][patient] = scope_graphs
            shaft_groups = frame.groupby("shaft", sort=False)["y_true"]
            mixed = int(sum(group.nunique() == 2 for _, group in shaft_groups))
            for scope, scope_graph in scope_graphs.items():
                audit_rows.append(
                    {
                        "radius_mm": radius_value,
                        "patient": patient,
                        "scope": scope,
                        "n_nodes": len(graph.channels),
                        "n_shafts": int(frame["shaft"].nunique()),
                        "n_mixed_label_shafts": mixed,
                        "n_edges": scope_graph.n_edges,
                        "n_isolated": scope_graph.n_isolated,
                        "n_neighbors": n_neighbors,
                        "gaussian_sigma_mm": gaussian_sigma_mm,
                    }
                )
    return graphs, scoped, pd.DataFrame.from_records(audit_rows)


def assert_formal_graph_counts(
    frames: dict[str, pd.DataFrame],
    audit: pd.DataFrame,
    expected: dict[str, object],
) -> None:
    n_contacts = int(sum(len(frame) for frame in frames.values()))
    n_shafts = int(sum(frame["shaft"].nunique() for frame in frames.values()))
    n_mixed = int(
        sum(
            group.nunique() == 2
            for frame in frames.values()
            for _, group in frame.groupby("shaft")["y_true"]
        )
    )
    radius = float(expected["primary_radius_mm"])
    primary = audit[np.isclose(audit["radius_mm"], radius)]
    edge_counts = primary.groupby("scope")["n_edges"].sum().to_dict()
    observed = {
        "contacts": n_contacts,
        "patient_shafts": n_shafts,
        "mixed_label_shafts": n_mixed,
        "all_edges": int(edge_counts.get("all", -1)),
        "same_shaft_edges": int(edge_counts.get("same_shaft", -1)),
        "cross_shaft_edges": int(edge_counts.get("cross_shaft", -1)),
    }
    wanted = {
        key: int(expected[key])
        for key in (
            "contacts",
            "patient_shafts",
            "mixed_label_shafts",
            "all_edges",
            "same_shaft_edges",
            "cross_shaft_edges",
        )
    }
    if observed != wanted:
        raise SystemExit(f"Formal graph anchor mismatch: expected={wanted}, observed={observed}")
    if observed["all_edges"] != (
        observed["same_shaft_edges"] + observed["cross_shaft_edges"]
    ):
        raise SystemExit("Formal same/cross edge counts do not sum to all edges")


def shaft_null_design(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enumerate each shaft's exact null state space and its joint log10 size."""
    rows: list[dict[str, object]] = []
    for patient, frame in frames.items():
        for shaft, group in frame.groupby("shaft", sort=True):
            ordered = group.sort_values(["contact_order", "bipolar_channel"])
            labels = tuple(ordered["y_true"].astype(int))
            n_nodes = len(labels)
            n_positive = int(sum(labels))
            circular_states = {
                tuple(np.roll(values, shift))
                for values in (np.asarray(labels), np.asarray(labels)[::-1])
                for shift in range(n_nodes)
            }
            rows.append(
                {
                    "patient": patient,
                    "shaft": shaft,
                    "n_nodes": n_nodes,
                    "n_positive": n_positive,
                    "is_mixed_label": 0 < n_positive < n_nodes,
                    "within_shaft_shuffle_unique_states": math.comb(
                        n_nodes, n_positive
                    ),
                    "circular_shift_reverse_unique_states": len(circular_states),
                }
            )
    audit = pd.DataFrame.from_records(rows)
    summaries: list[dict[str, object]] = []
    for method, column in (
        ("within_shaft_shuffle", "within_shaft_shuffle_unique_states"),
        ("circular_shift_reverse", "circular_shift_reverse_unique_states"),
    ):
        states = audit[column].to_numpy(dtype=np.int64)
        summaries.append(
            {
                "null_method": method,
                "n_patient_shafts": len(audit),
                "n_mixed_label_shafts": int(audit["is_mixed_label"].sum()),
                "n_movable_shafts": int(np.sum(states > 1)),
                "joint_unique_state_space_log10": float(np.log10(states).sum()),
                "minimum_unique_states_per_shaft": int(states.min()),
                "maximum_unique_states_per_shaft": int(states.max()),
            }
        )
    return audit, pd.DataFrame.from_records(summaries)


def permute_labels_within_shafts(
    labels: np.ndarray,
    shafts: np.ndarray,
    *,
    method: str,
    contact_order: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    y = np.asarray(labels, dtype=int)
    shaft_values = np.asarray(shafts, dtype=object)
    order_values = np.asarray(contact_order, dtype=np.float64)
    if y.ndim != 1 or shaft_values.shape != y.shape or order_values.shape != y.shape:
        raise SystemExit("Label, shaft, and contact-order vectors must align")
    if method not in NULL_METHODS:
        raise SystemExit(f"Unknown shaft null method: {method}")
    output = y.copy()
    for shaft in sorted(set(shaft_values.astype(str))):
        indices = np.flatnonzero(shaft_values.astype(str) == shaft)
        if method == "within_shaft_shuffle":
            output[indices] = y[rng.permutation(indices)]
        else:
            ordered = indices[np.lexsort((indices, order_values[indices]))]
            values = y[ordered].copy()
            if len(values) > 1 and int(rng.integers(0, 2)) == 1:
                values = values[::-1]
            shift = int(rng.integers(0, len(values))) if len(values) else 0
            output[ordered] = np.roll(values, shift)
    original_counts = pd.Series(y).groupby(shaft_values).sum().sort_index()
    permuted_counts = pd.Series(output).groupby(shaft_values).sum().sort_index()
    if not original_counts.equals(permuted_counts):
        raise SystemExit("Shaft-null permutation changed a shaft positive count")
    return output


def weighted_label_roughness(labels: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    upper_i, upper_j = np.where(np.triu(weights, k=1) > 0.0)
    edge_weights = weights[upper_i, upper_j]
    if len(edge_weights) == 0 or float(edge_weights.sum()) == 0.0:
        return float("nan")
    return float(
        np.sum(edge_weights * (y[upper_i] - y[upper_j]) ** 2)
        / edge_weights.sum()
    )


def observed_roughness(
    frames: dict[str, pd.DataFrame],
    scoped: dict[float, dict[str, dict[str, ScopeGraph]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for radius, patient_graphs in scoped.items():
        for patient, scope_graphs in patient_graphs.items():
            labels = frames[patient]["y_true"].to_numpy(dtype=int)
            for scope, graph in scope_graphs.items():
                rows.append(
                    {
                        "radius_mm": radius,
                        "patient": patient,
                        "scope": scope,
                        "observed_roughness": weighted_label_roughness(
                            labels, graph.weights
                        ),
                        "n_edges": graph.n_edges,
                    }
                )
    patient = pd.DataFrame.from_records(rows)
    summary = (
        patient.groupby(["radius_mm", "scope"], sort=True)
        .agg(
            observed_mean_roughness=("observed_roughness", "mean"),
            n_patients_with_edges=("observed_roughness", "count"),
            total_edges=("n_edges", "sum"),
        )
        .reset_index()
    )
    return patient, summary


def null_iteration_rows(
    frames: dict[str, pd.DataFrame],
    scoped: dict[float, dict[str, dict[str, ScopeGraph]]],
    *,
    method: str,
    seed: int,
    iteration: int,
) -> list[dict[str, object]]:
    method_code = NULL_METHODS.index(method) + 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, method_code, iteration]))
    permuted: dict[str, np.ndarray] = {}
    for patient, frame in frames.items():
        permuted[patient] = permute_labels_within_shafts(
            frame["y_true"].to_numpy(dtype=int),
            frame["shaft"].to_numpy(dtype=object),
            method=method,
            contact_order=frame["contact_order"].to_numpy(dtype=np.float64),
            rng=rng,
        )
    rows: list[dict[str, object]] = []
    for radius, patient_graphs in scoped.items():
        for scope in SCOPES:
            values = [
                weighted_label_roughness(
                    permuted[patient], patient_graphs[patient][scope].weights
                )
                for patient in sorted(patient_graphs)
            ]
            rows.append(
                {
                    "null_method": method,
                    "permutation_iteration": iteration,
                    "radius_mm": radius,
                    "scope": scope,
                    "null_mean_roughness": float(np.nanmean(values)),
                    "n_patients_with_edges": int(np.sum(np.isfinite(values))),
                }
            )
    return rows


def run_null_method(
    frames: dict[str, pd.DataFrame],
    scoped: dict[float, dict[str, dict[str, ScopeGraph]]],
    *,
    method: str,
    n_iterations: int,
    seed: int,
    checkpoint_path: Path,
    checkpoint_interval: int,
    resume: bool,
) -> pd.DataFrame:
    columns = [
        "null_method",
        "permutation_iteration",
        "radius_mm",
        "scope",
        "null_mean_roughness",
        "n_patients_with_edges",
    ]
    if n_iterations < 1:
        return pd.DataFrame(columns=columns)
    if resume and checkpoint_path.is_file():
        existing = pd.read_csv(checkpoint_path)
        if list(existing.columns) != columns or not existing["null_method"].eq(method).all():
            raise SystemExit(f"Invalid null checkpoint: {checkpoint_path}")
        completed = sorted(existing["permutation_iteration"].astype(int).unique())
        if completed != list(range(1, (completed[-1] if completed else 0) + 1)):
            raise SystemExit(f"Non-contiguous null checkpoint: {checkpoint_path}")
        rows = existing.to_dict("records")
        start = (completed[-1] + 1) if completed else 1
    else:
        rows = []
        start = 1
    for iteration in range(start, n_iterations + 1):
        rows.extend(
            null_iteration_rows(
                frames,
                scoped,
                method=method,
                seed=seed,
                iteration=iteration,
            )
        )
        if iteration % checkpoint_interval == 0 or iteration == n_iterations:
            atomic_write_csv(pd.DataFrame.from_records(rows, columns=columns), checkpoint_path)
    output = pd.DataFrame.from_records(rows, columns=columns)
    return output[output["permutation_iteration"].astype(int) <= n_iterations].copy()


def summarize_nulls(
    nulls: pd.DataFrame, observed_summary: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed_lookup = observed_summary.set_index(["radius_mm", "scope"])
    for (method, radius, scope), frame in nulls.groupby(
        ["null_method", "radius_mm", "scope"], sort=True
    ):
        observed = float(
            observed_lookup.loc[(float(radius), str(scope)), "observed_mean_roughness"]
        )
        values = frame["null_mean_roughness"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "null_method": method,
                "radius_mm": radius,
                "scope": scope,
                "n_permutations": len(values),
                "observed_mean_roughness": observed,
                "null_mean_roughness": float(np.mean(values)),
                "null_sd_roughness": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "p_lower": float((np.sum(values <= observed) + 1) / (len(values) + 1)),
            }
        )
    return pd.DataFrame.from_records(rows)


def _safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )


def ppr_scope_analysis(
    aligned: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    scoped: dict[float, dict[str, dict[str, ScopeGraph]]],
    *,
    alpha: float,
    core,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    for radius, patient_graphs in scoped.items():
        for (band, patient), frame in aligned.groupby(
            ["frequency_band", "patient"], sort=True
        ):
            patient_key = str(patient)
            ordered = frame.sort_values("bipolar_channel").copy()
            expected_channels = tuple(frames[patient_key]["bipolar_channel"].astype(str))
            if tuple(ordered["bipolar_channel"].astype(str)) != expected_channels:
                raise SystemExit(f"{patient} {band}: prediction/graph channels differ")
            labels = ordered["y_true"].to_numpy(dtype=int)
            original = ordered["y_score"].to_numpy(dtype=np.float64)
            for scope in SCOPES:
                transition = patient_graphs[patient_key][scope].transition
                scores = core.ppr_diffuse_scores(original, transition, alpha=alpha)
                temporary = ordered[["bipolar_channel", "y_score", "y_pred"]].copy()
                temporary["smoothed_score"] = scores
                predicted = core.count_preserving_predictions(temporary)
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "radius_mm": radius,
                            "scope": scope,
                            "frequency_band": str(band),
                            "patient": patient_key,
                            "bipolar_channel": ordered["bipolar_channel"].astype(str),
                            "y_true": labels,
                            "original_score": original,
                            "ppr_score": scores,
                            "original_pred": ordered["y_pred"].to_numpy(dtype=int),
                            "ppr_pred": predicted,
                        }
                    )
                )
                original_ap = float(average_precision_score(labels, original))
                ppr_ap = float(average_precision_score(labels, scores))
                original_auc = _safe_auroc(labels, original)
                ppr_auc = _safe_auroc(labels, scores)
                metric_rows.append(
                    {
                        "radius_mm": radius,
                        "scope": scope,
                        "frequency_band": str(band),
                        "patient": patient_key,
                        "n_contacts": len(labels),
                        "n_positive_labels": int(labels.sum()),
                        "original_average_precision": original_ap,
                        "ppr_average_precision": ppr_ap,
                        "delta_average_precision": ppr_ap - original_ap,
                        "original_auroc": original_auc,
                        "ppr_auroc": ppr_auc,
                        "delta_auroc": ppr_auc - original_auc,
                    }
                )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    patient_metrics = pd.DataFrame.from_records(metric_rows)
    summary = (
        patient_metrics.groupby(["radius_mm", "scope", "frequency_band"], sort=True)
        .agg(
            n_patients=("patient", "nunique"),
            macro_original_average_precision=("original_average_precision", "mean"),
            macro_ppr_average_precision=("ppr_average_precision", "mean"),
            delta_macro_average_precision=("delta_average_precision", "mean"),
            macro_original_auroc=("original_auroc", "mean"),
            macro_ppr_auroc=("ppr_auroc", "mean"),
            delta_macro_auroc=("delta_auroc", "mean"),
        )
        .reset_index()
    )
    return predictions, patient_metrics, summary


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "scikit-learn", "PyYAML"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def input_fingerprint(
    *,
    config_path: Path,
    config: dict[str, object],
    bands: list[str],
    patients: list[str] | None,
    n_permutations: int,
    output_root: Path,
) -> tuple[str, dict[str, object]]:
    inputs = dict(config["inputs"])
    paths = {
        "config": Path(config_path),
        "runner": Path(__file__),
        "core_graph_script": Path(str(inputs["core_graph_script"])),
        "predictions": Path(str(inputs["predictions"])),
        "coordinates": Path(str(inputs["coordinates"])),
    }
    payload: dict[str, object] = {
        "algorithm_version": ALGORITHM_VERSION,
        "input_sha256": {name: path_sha256(path) for name, path in paths.items()},
        "bands": bands,
        "patients": sorted(patients) if patients is not None else None,
        "n_permutations": n_permutations,
        "output_root": str(_resolve(output_root)),
        "python_version": sys.version,
        "package_versions": package_versions(),
        "graph": config["graph"],
        "nulls": config["nulls"],
        "expected": config["expected"],
        "artifact_policy": config.get("artifact_policy"),
        "window_variant": config.get("window_variant"),
        "seizure_sampling": config.get("seizure_sampling"),
    }
    return canonical_hash(payload), payload


def validate_complete_resume(output_root: Path, fingerprint: str) -> bool:
    manifest_path = output_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Resume requested but manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("fingerprint") != fingerprint:
        raise SystemExit("Resume fingerprint mismatch; use a new output directory")
    if manifest.get("status") != "complete":
        return False
    for relative, expected_hash in dict(manifest.get("output_sha256", {})).items():
        path = output_root / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            raise SystemExit(f"Completed-resume output hash mismatch: {path}")
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bands", nargs="+")
    parser.add_argument("--patients", nargs="+")
    parser.add_argument("--n-permutations", type=int)
    parser.add_argument("--resume", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.resume and args.dry_run:
        raise SystemExit("--resume cannot be combined with --dry-run")
    config = load_yaml(args.config)
    output_root = args.out or Path(str(config["output_root"]))
    assert_allowed_output_root(output_root)
    selected_bands = [str(value) for value in (args.bands or config["frequency_bands"])]
    selected_patients = [str(value) for value in args.patients] if args.patients else None
    graph_config = dict(config["graph"])
    null_config = dict(config["nulls"])
    radii = [float(value) for value in graph_config["radii_mm"]]
    if radii != [5.0, 7.5, 10.0, 12.5, 15.0]:
        raise SystemExit(f"Unexpected prespecified radius grid: {radii}")
    if tuple(graph_config["edge_scopes"]) != SCOPES:
        raise SystemExit(f"Unexpected edge scopes: {graph_config['edge_scopes']}")
    if tuple(null_config["methods"]) != NULL_METHODS:
        raise SystemExit(f"Unexpected shaft nulls: {null_config['methods']}")
    n_permutations = (
        int(args.n_permutations)
        if args.n_permutations is not None
        else int(null_config["n_permutations"])
    )
    if n_permutations < 1:
        raise SystemExit("n_permutations must be positive")

    core, aligned, selected_bands = prepare_inputs(
        config, bands=selected_bands, patients=selected_patients
    )
    frames = reference_frames(aligned, selected_bands[0], core.shaft_name)
    _, scoped, graph_audit = build_graphs(
        frames,
        radii_mm=radii,
        n_neighbors=int(graph_config["n_neighbors"]),
        gaussian_sigma_mm=float(graph_config["gaussian_sigma_mm"]),
        core=core,
    )
    full_run = selected_patients is None and selected_bands == [
        str(value) for value in config["frequency_bands"]
    ]
    if full_run:
        assert_formal_graph_counts(frames, graph_audit, dict(config["expected"]))
    ten = graph_audit[np.isclose(graph_audit["radius_mm"], 10.0)]
    edges = ten.groupby("scope")["n_edges"].sum().to_dict()
    print(
        f"patients={len(frames)} contacts={sum(len(frame) for frame in frames.values())} "
        f"patient_shafts={sum(frame['shaft'].nunique() for frame in frames.values())}"
    )
    print(
        f"radius_10mm_edges=all:{int(edges.get('all', 0))},"
        f"same:{int(edges.get('same_shaft', 0))},cross:{int(edges.get('cross_shaft', 0))}"
    )
    print(f"bands={len(selected_bands)} permutations_per_null={n_permutations}")
    if not args.apply and not args.resume:
        print("dry_run=true; no files written")
        return

    fingerprint, fingerprint_payload = input_fingerprint(
        config_path=args.config,
        config=config,
        bands=selected_bands,
        patients=selected_patients,
        n_permutations=n_permutations,
        output_root=output_root,
    )
    if output_root.exists():
        if not args.resume:
            raise SystemExit(f"Output root exists; pass --resume or choose a new root: {output_root}")
        if validate_complete_resume(output_root, fingerprint):
            print(f"resume_complete=true output_root={output_root}")
            return
    else:
        output_root.mkdir(parents=True)

    protected_paths = [Path(str(value)) for value in config["protected_paths"]]
    protected_before = {str(path): path_sha256(path) for path in protected_paths}
    manifest_path = output_root / "run_manifest.json"
    if args.resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("fingerprint") != fingerprint:
            raise SystemExit("Resume fingerprint mismatch; use a new output directory")
        if dict(prior.get("protected_sha256_before", {})) != protected_before:
            raise SystemExit("Protected legacy inputs changed since the interrupted run")
    running_manifest: dict[str, object] = {
        "run_id": config["run_id"],
        "status": "running",
        "algorithm_version": ALGORITHM_VERSION,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "command": [sys.executable, str(Path(__file__)), *sys.argv[1:]],
        "python_version": sys.version,
        "package_versions": package_versions(),
        "protected_sha256_before": protected_before,
        "n_permutations_per_null": n_permutations,
    }
    atomic_write_json(running_manifest, manifest_path)

    observed_patient, observed_summary = observed_roughness(frames, scoped)
    null_design, null_design_summary = shaft_null_design(frames)
    checkpoint_root = output_root / "_checkpoints"
    null_parts = []
    for method in NULL_METHODS:
        null_parts.append(
            run_null_method(
                frames,
                scoped,
                method=method,
                n_iterations=n_permutations,
                seed=int(null_config["random_seed"]),
                checkpoint_path=checkpoint_root / f"roughness_{method}.csv",
                checkpoint_interval=int(null_config["checkpoint_interval"]),
                resume=args.resume,
            )
        )
    nulls = pd.concat(null_parts, ignore_index=True)
    null_summary = summarize_nulls(nulls, observed_summary)
    ppr_predictions, ppr_patient_metrics, ppr_summary = ppr_scope_analysis(
        aligned,
        frames,
        scoped,
        alpha=float(graph_config["ppr_alpha"]),
        core=core,
    )

    outputs = {
        "graph_structure_by_radius_patient_scope.csv": graph_audit,
        "shaft_null_design_by_patient_shaft.csv": null_design,
        "shaft_null_design_summary.csv": null_design_summary,
        "roughness_observed_by_radius_patient_scope.csv": observed_patient,
        "roughness_observed_summary.csv": observed_summary,
        "roughness_null_iterations.csv": nulls,
        "roughness_null_summary.csv": null_summary,
        "ppr_predictions_by_radius_scope.csv": ppr_predictions,
        "ppr_patient_metrics_by_radius_scope_band.csv": ppr_patient_metrics,
        "ppr_metrics_by_radius_scope_band.csv": ppr_summary,
    }
    for name, frame in outputs.items():
        atomic_write_csv(frame, output_root / name)

    protected_after = {str(path): path_sha256(path) for path in protected_paths}
    if protected_after != protected_before:
        raise SystemExit("Protected legacy files changed during the shaft-graph run")
    output_hashes = {
        name: file_sha256(output_root / name) for name in sorted(outputs)
    }
    complete_manifest = {
        **running_manifest,
        "status": "complete",
        "protected_sha256_after": protected_after,
        "output_sha256": output_hashes,
        "counts": {
            "patients": len(frames),
            "contacts": int(sum(len(frame) for frame in frames.values())),
            "patient_shafts": int(sum(frame["shaft"].nunique() for frame in frames.values())),
            "bands": len(selected_bands),
            "radii": len(radii),
            "edge_scopes": len(SCOPES),
            "null_methods": len(NULL_METHODS),
        },
    }
    atomic_write_json(complete_manifest, manifest_path)
    print(f"wrote={output_root}")


if __name__ == "__main__":
    main()
