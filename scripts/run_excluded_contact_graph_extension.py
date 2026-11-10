"""Propagate formal electrode scores to excluded tissue contacts using geometry only."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import openpyxl
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "excluded_contact_graph_extension.yaml"
OUTPUT_ROOT = ROOT / "data_derived" / "electrode_graph_smoothing_excluded"
RADIUS_SENSITIVITY_ROOT = (
    ROOT / "data_derived" / "electrode_graph_smoothing_radius_sensitivity"
)
CHECK_ROOT = ROOT / "checks"
PATIENTS = tuple(private_value("cohort", "patients"))
SCORE_SOURCES = {
    "smoothed_anchor": "propagated_score_from_smoothed_anchor",
    "raw_anchor": "propagated_score_from_raw_anchor",
}
PRIMARY_SCORE_SOURCE = "smoothed_anchor"


@dataclass(frozen=True)
class FullGraph:
    channels: tuple[str, ...]
    weights: np.ndarray
    distances_mm: np.ndarray
    component_id: np.ndarray
    reachable_from_anchor: np.ndarray
    hops_to_anchor: np.ndarray
    n_edges: int
    n_isolated: int


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def assert_allowed_output_root(path: Path) -> None:
    assert_output_path(path)
    resolved = _resolve(path)
    formal_root = _resolve(OUTPUT_ROOT)
    sensitivity_root = _resolve(RADIUS_SENSITIVITY_ROOT)
    review_root = _resolve(CHECK_ROOT)
    if resolved == AUX_ROOT or _is_relative_to(resolved, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {resolved}")
    if not (
        resolved == formal_root
        or _is_relative_to(resolved, formal_root)
        or resolved == sensitivity_root
        or _is_relative_to(resolved, sensitivity_root)
        or resolved == review_root
        or _is_relative_to(resolved, review_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def _as_bool(value: object, *, field: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    raise SystemExit(f"{field}: expected a complete boolean value, got {value!r}")


def load_coordinates(
    path: Path,
    patients: Iterable[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing coordinate manifest: {path}")
    frame = pd.read_csv(path)
    required = {
        "patient",
        "bipolar_channel",
        "scs_x_mm",
        "scs_y_mm",
        "scs_z_mm",
        "desikan_killiany_label",
        "tissue_class",
        "is_white_csf_ventricle_unknown",
        "qc_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    selected = {str(value) for value in patients}
    frame = frame[frame["patient"].astype(str).isin(selected)].copy()
    if frame.empty:
        raise SystemExit("No coordinate rows remain after patient filtering")
    if frame.duplicated(["patient", "bipolar_channel"]).any():
        raise SystemExit("Coordinate manifest contains duplicate patient/channel rows")
    xyz = frame[["scs_x_mm", "scs_y_mm", "scs_z_mm"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise SystemExit("Coordinate manifest contains non-finite SCS coordinates")
    if not frame["qc_status"].astype(str).eq("pass").all():
        raise SystemExit("Coordinate manifest contains rows without qc_status=pass")
    frame["is_excluded_tissue"] = frame[
        "is_white_csf_ventricle_unknown"
    ].map(lambda value: _as_bool(value, field="is_white_csf_ventricle_unknown"))
    return frame.sort_values(["patient", "bipolar_channel"]).reset_index(drop=True)


def load_anchor_predictions(
    path: Path,
    bands: Iterable[str],
    patients: Iterable[str],
) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing anchor prediction table: {path}")
    frame = pd.read_csv(path)
    required = {
        "frequency_band",
        "patient",
        "heldout_patient",
        "bipolar_channel",
        "y_score",
        "smoothed_score",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    selected_bands = {str(value) for value in bands}
    selected_patients = {str(value) for value in patients}
    frame = frame[
        frame["frequency_band"].astype(str).isin(selected_bands)
        & frame["patient"].astype(str).isin(selected_patients)
    ].copy()
    if frame.empty:
        raise SystemExit("No anchor scores remain after filtering")
    keys = ["frequency_band", "patient", "bipolar_channel"]
    if frame.duplicated(keys).any():
        raise SystemExit("Anchor prediction table contains duplicate band/patient/channel rows")
    if not (
        frame["patient"].astype(str) == frame["heldout_patient"].astype(str)
    ).all():
        raise SystemExit("Anchor predictions contain non-heldout patient rows")
    for column in ("y_score", "smoothed_score"):
        values = frame[column].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise SystemExit(f"{column} must contain finite values in [0, 1]")
    actual_bands = set(frame["frequency_band"].astype(str))
    if actual_bands != selected_bands:
        raise SystemExit(
            f"Anchor band mismatch: expected={sorted(selected_bands)} "
            f"actual={sorted(actual_bands)}"
        )
    return frame[
        [
            "frequency_band",
            "patient",
            "bipolar_channel",
            "y_score",
            "smoothed_score",
        ]
    ].sort_values(keys).reset_index(drop=True)


def build_mutual_knn_weights(
    xyz_mm: np.ndarray,
    *,
    n_neighbors: int,
    max_distance_mm: float,
    gaussian_sigma_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise SystemExit("Graph coordinates must be finite n_contacts x 3 values")
    if n_neighbors < 1 or max_distance_mm <= 0.0 or gaussian_sigma_mm <= 0.0:
        raise SystemExit("Invalid graph parameters")
    delta = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    n_nodes = len(xyz)
    directed = np.zeros((n_nodes, n_nodes), dtype=bool)
    indices = np.arange(n_nodes)
    for node in range(n_nodes):
        candidates = indices[
            (indices != node)
            & np.isfinite(distances[node])
            & (distances[node] <= max_distance_mm)
        ]
        if not len(candidates):
            continue
        order = np.lexsort((candidates, distances[node, candidates]))
        directed[node, candidates[order[:n_neighbors]]] = True
    mutual = directed & directed.T
    weights = np.zeros_like(distances)
    weights[mutual] = np.exp(
        -(distances[mutual] ** 2) / (2.0 * gaussian_sigma_mm**2)
    )
    weights = np.maximum(weights, weights.T)
    return weights, distances


def graph_components_and_hops(
    weights: np.ndarray,
    anchor_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(weights, dtype=np.float64)
    anchors = np.asarray(anchor_mask, dtype=bool)
    if matrix.shape != (len(anchors), len(anchors)):
        raise SystemExit("Graph/anchor shape mismatch")
    adjacency = matrix > 0.0
    component_id = np.full(len(anchors), -1, dtype=int)
    reachable = np.zeros(len(anchors), dtype=bool)
    component = 0
    for start in range(len(anchors)):
        if component_id[start] >= 0:
            continue
        stack = [start]
        component_id[start] = component
        nodes: list[int] = []
        while stack:
            node = stack.pop()
            nodes.append(node)
            for neighbor in np.flatnonzero(adjacency[node]):
                if component_id[neighbor] < 0:
                    component_id[neighbor] = component
                    stack.append(int(neighbor))
        if anchors[nodes].any():
            reachable[nodes] = True
        component += 1

    hops = np.full(len(anchors), -1, dtype=int)
    queue: deque[int] = deque()
    for anchor in np.flatnonzero(anchors):
        hops[anchor] = 0
        queue.append(int(anchor))
    while queue:
        node = queue.popleft()
        for neighbor in np.flatnonzero(adjacency[node]):
            if hops[neighbor] < 0:
                hops[neighbor] = hops[node] + 1
                queue.append(int(neighbor))
    return component_id, reachable, hops


def build_full_graph(
    channels: Iterable[str],
    xyz_mm: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    n_neighbors: int,
    max_distance_mm: float,
    gaussian_sigma_mm: float,
) -> FullGraph:
    channel_tuple = tuple(str(value) for value in channels)
    if len(set(channel_tuple)) != len(channel_tuple):
        raise SystemExit("Graph channel names must be unique")
    weights, distances = build_mutual_knn_weights(
        xyz_mm,
        n_neighbors=n_neighbors,
        max_distance_mm=max_distance_mm,
        gaussian_sigma_mm=gaussian_sigma_mm,
    )
    component_id, reachable, hops = graph_components_and_hops(weights, anchor_mask)
    return FullGraph(
        channels=channel_tuple,
        weights=weights,
        distances_mm=distances,
        component_id=component_id,
        reachable_from_anchor=reachable,
        hops_to_anchor=hops,
        n_edges=int(np.sum(np.triu(weights, k=1) > 0.0)),
        n_isolated=int(np.sum(weights.sum(axis=1) == 0.0)),
    )


def clamped_harmonic_extension(
    weights: np.ndarray,
    anchor_mask: np.ndarray,
    anchor_values: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(weights, dtype=np.float64)
    anchors = np.asarray(anchor_mask, dtype=bool)
    values = np.asarray(anchor_values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, np.newaxis]
    if matrix.shape != (len(anchors), len(anchors)) or values.shape[0] != len(anchors):
        raise SystemExit("Harmonic extension shape mismatch")
    if not np.isfinite(values[anchors]).all():
        raise SystemExit("Anchor values must be finite")
    if np.any((values[anchors] < 0.0) | (values[anchors] > 1.0)):
        raise SystemExit("Anchor values must be in [0, 1]")

    component_id, reachable, _ = graph_components_and_hops(matrix, anchors)
    output = np.full_like(values, np.nan, dtype=np.float64)
    output[anchors] = values[anchors]
    for component in np.unique(component_id):
        nodes = np.flatnonzero(component_id == component)
        known = nodes[anchors[nodes]]
        unknown = nodes[~anchors[nodes]]
        if not len(unknown) or not len(known):
            continue
        degree = matrix[nodes].sum(axis=1)
        laplacian = np.diag(degree) - matrix[np.ix_(nodes, nodes)]
        local_unknown = np.flatnonzero(~anchors[nodes])
        local_known = np.flatnonzero(anchors[nodes])
        lhs = laplacian[np.ix_(local_unknown, local_unknown)]
        rhs = -laplacian[np.ix_(local_unknown, local_known)] @ values[known]
        try:
            solved = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError as exc:
            raise SystemExit(
                f"Anchored graph component {component} is singular"
            ) from exc
        output[unknown] = solved
    finite = np.isfinite(output)
    if np.any((output[finite] < -1e-10) | (output[finite] > 1.0 + 1e-10)):
        raise SystemExit("Harmonic extension produced scores outside [0, 1]")
    output[finite] = np.clip(output[finite], 0.0, 1.0)
    if not np.array_equal(np.isfinite(output).any(axis=1), reachable):
        raise SystemExit("Reachability and finite harmonic scores disagree")
    if not np.allclose(output[anchors], values[anchors], atol=0.0, rtol=0.0):
        raise SystemExit("Clamped harmonic extension changed anchor values")
    return output


def _nearest_anchor_distance(distances: np.ndarray, anchor_mask: np.ndarray) -> np.ndarray:
    anchors = np.flatnonzero(anchor_mask)
    if not len(anchors):
        raise SystemExit("Patient graph has no eligible anchors")
    return distances[:, anchors].min(axis=1)


def extend_scores_to_excluded(
    coordinates: pd.DataFrame,
    anchor_predictions: pd.DataFrame,
    *,
    n_neighbors: int,
    max_distance_mm: float,
    gaussian_sigma_mm: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prediction_parts: list[pd.DataFrame] = []
    node_parts: list[pd.DataFrame] = []
    graph_rows: list[dict[str, object]] = []
    bands = list(dict.fromkeys(anchor_predictions["frequency_band"].astype(str)))

    for patient, coordinate_frame in coordinates.groupby("patient", sort=True):
        nodes = coordinate_frame.sort_values("bipolar_channel").reset_index(drop=True)
        anchor_mask = ~nodes["is_excluded_tissue"].to_numpy(dtype=bool)
        graph = build_full_graph(
            nodes["bipolar_channel"].astype(str),
            nodes[["scs_x_mm", "scs_y_mm", "scs_z_mm"]].to_numpy(dtype=np.float64),
            anchor_mask,
            n_neighbors=n_neighbors,
            max_distance_mm=max_distance_mm,
            gaussian_sigma_mm=gaussian_sigma_mm,
        )
        direct_anchor = np.zeros(len(nodes), dtype=bool)
        direct_anchor[~anchor_mask] = (
            graph.weights[np.ix_(~anchor_mask, anchor_mask)] > 0.0
        ).any(axis=1)
        nearest_distance = _nearest_anchor_distance(graph.distances_mm, anchor_mask)
        node_audit = nodes.copy()
        node_audit["node_role"] = np.where(anchor_mask, "eligible_anchor", "excluded_target")
        node_audit["graph_component_id"] = graph.component_id
        node_audit["reachable_from_eligible_anchor"] = graph.reachable_from_anchor
        node_audit["graph_hops_to_anchor"] = graph.hops_to_anchor
        node_audit["direct_anchor_neighbor"] = direct_anchor
        node_audit["nearest_anchor_distance_mm"] = nearest_distance
        node_audit["graph_degree"] = np.sum(graph.weights > 0.0, axis=1)
        node_audit["graph_weighted_degree"] = graph.weights.sum(axis=1)
        node_audit["coverage_status"] = np.select(
            [
                anchor_mask,
                (~anchor_mask) & direct_anchor,
                (~anchor_mask) & graph.reachable_from_anchor,
            ],
            [
                "eligible_anchor",
                "reachable_direct",
                "reachable_multihop",
            ],
            default="unreachable_no_eligible_anchor",
        )
        node_parts.append(node_audit)
        graph_rows.append(
            {
                "patient": patient,
                "n_nodes": len(nodes),
                "n_eligible_anchors": int(anchor_mask.sum()),
                "n_excluded_targets": int((~anchor_mask).sum()),
                "n_excluded_direct": int(((~anchor_mask) & direct_anchor).sum()),
                "n_excluded_reachable": int(
                    ((~anchor_mask) & graph.reachable_from_anchor).sum()
                ),
                "n_excluded_unreachable": int(
                    ((~anchor_mask) & ~graph.reachable_from_anchor).sum()
                ),
                "n_edges": graph.n_edges,
                "n_isolated": graph.n_isolated,
                "n_neighbors": n_neighbors,
                "max_distance_mm": max_distance_mm,
                "gaussian_sigma_mm": gaussian_sigma_mm,
                "neighbor_rule": "mutual_knn_with_radius_cap",
            }
        )

        expected_anchor_channels = set(nodes.loc[anchor_mask, "bipolar_channel"].astype(str))
        for band in bands:
            frame = anchor_predictions[
                (anchor_predictions["patient"].astype(str) == str(patient))
                & (anchor_predictions["frequency_band"].astype(str) == band)
            ].sort_values("bipolar_channel")
            actual_anchor_channels = set(frame["bipolar_channel"].astype(str))
            if actual_anchor_channels != expected_anchor_channels:
                missing = sorted(expected_anchor_channels - actual_anchor_channels)
                extra = sorted(actual_anchor_channels - expected_anchor_channels)
                raise SystemExit(
                    f"{patient} {band}: anchor/eligible mismatch "
                    f"missing={missing[:5]} extra={extra[:5]}"
                )
            lookup = frame.set_index("bipolar_channel")
            anchor_values = np.full((len(nodes), 2), np.nan, dtype=np.float64)
            anchor_channels = nodes.loc[anchor_mask, "bipolar_channel"].astype(str)
            anchor_values[anchor_mask, 0] = lookup.loc[
                anchor_channels, "smoothed_score"
            ].to_numpy(dtype=np.float64)
            anchor_values[anchor_mask, 1] = lookup.loc[
                anchor_channels, "y_score"
            ].to_numpy(dtype=np.float64)
            extended = clamped_harmonic_extension(
                graph.weights,
                anchor_mask,
                anchor_values,
            )
            target = node_audit.loc[~anchor_mask].copy()
            target.insert(0, "frequency_band", band)
            target["propagated_score_from_smoothed_anchor"] = extended[~anchor_mask, 0]
            target["propagated_score_from_raw_anchor"] = extended[~anchor_mask, 1]
            target["extension_method"] = "clamped_harmonic_extension"
            target["primary_anchor_score"] = "smoothed_score"
            target["analysis_scope"] = "posthoc_exploratory_spatial_extension"
            prediction_parts.append(target)

    predictions = pd.concat(prediction_parts, ignore_index=True)
    node_audit = pd.concat(node_parts, ignore_index=True)
    graph_audit = pd.DataFrame.from_records(graph_rows)
    return (
        predictions.sort_values(
            ["frequency_band", "patient", "bipolar_channel"]
        ).reset_index(drop=True),
        node_audit.sort_values(["patient", "bipolar_channel"]).reset_index(drop=True),
        graph_audit.sort_values("patient").reset_index(drop=True),
    )


def load_audit_labels(
    workbook_path: Path,
    patients: Iterable[str],
) -> pd.DataFrame:
    if not workbook_path.is_file():
        raise SystemExit(f"Missing audit workbook: {workbook_path}")
    workbook = openpyxl.load_workbook(
        workbook_path,
        read_only=True,
        data_only=True,
    )
    rows: list[dict[str, object]] = []
    for patient in patients:
        sheet_name = f"{patient}_bipolar"
        if sheet_name not in workbook.sheetnames:
            raise SystemExit(f"Missing audit worksheet: {sheet_name}")
        values = list(workbook[sheet_name].iter_rows(values_only=True))
        if not values:
            raise SystemExit(f"Empty audit worksheet: {sheet_name}")
        columns = [str(value) for value in values[0]]
        required = {
            "patient_id",
            "bipolar_channel",
            "is_white_csf_ventricle_unknown",
            "remove_as_bad_channel",
            "doctor_ablation_label",
        }
        missing = sorted(required - set(columns))
        if missing:
            raise SystemExit(f"{sheet_name}: missing columns {missing}")
        for values_row in values[1:]:
            record = dict(zip(columns, values_row, strict=True))
            rows.append(
                {
                    "patient": str(record["patient_id"]),
                    "bipolar_channel": str(record["bipolar_channel"]),
                    "audit_is_excluded_tissue": _as_bool(
                        record["is_white_csf_ventricle_unknown"],
                        field=f"{sheet_name}.is_white_csf_ventricle_unknown",
                    ),
                    "audit_remove_as_bad_channel": _as_bool(
                        record["remove_as_bad_channel"],
                        field=f"{sheet_name}.remove_as_bad_channel",
                    ),
                    "doctor_ablation_label": int(
                        _as_bool(
                            record["doctor_ablation_label"],
                            field=f"{sheet_name}.doctor_ablation_label",
                        )
                    ),
                }
            )
    output = pd.DataFrame.from_records(rows)
    if output.duplicated(["patient", "bipolar_channel"]).any():
        raise SystemExit("Audit workbook contains duplicate patient/channel rows")
    return output


def attach_excluded_labels(
    predictions: pd.DataFrame,
    coordinates: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    coordinate_keys = coordinates[["patient", "bipolar_channel", "is_excluded_tissue"]]
    audited = coordinate_keys.merge(
        labels,
        on=["patient", "bipolar_channel"],
        how="left",
        validate="one_to_one",
    )
    if audited["doctor_ablation_label"].isna().any():
        raise SystemExit("Formal coordinate rows are missing doctor ablation labels")
    if audited["audit_is_excluded_tissue"].isna().any():
        raise SystemExit("Formal coordinate rows are missing audit tissue flags")
    if not (
        audited["is_excluded_tissue"].astype(bool)
        == audited["audit_is_excluded_tissue"].astype(bool)
    ).all():
        raise SystemExit("Coordinate and audit tissue-exclusion flags disagree")
    if audited["audit_remove_as_bad_channel"].astype(bool).any():
        rows = audited.loc[
            audited["audit_remove_as_bad_channel"].astype(bool),
            ["patient", "bipolar_channel"],
        ]
        raise SystemExit(
            "Coordinate manifest unexpectedly contains bad/manual-excluded rows: "
            f"{rows.head(5).to_dict('records')}"
        )
    excluded_labels = audited[audited["is_excluded_tissue"].astype(bool)][
        ["patient", "bipolar_channel", "doctor_ablation_label"]
    ]
    output = predictions.merge(
        excluded_labels,
        on=["patient", "bipolar_channel"],
        how="left",
        validate="many_to_one",
    )
    if output["doctor_ablation_label"].isna().any():
        raise SystemExit("Excluded prediction rows are missing doctor labels")
    output["y_true"] = output["doctor_ablation_label"].astype(int)
    return output.drop(columns=["doctor_ablation_label"])


def safe_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if int(np.sum(y_true == 1)) == 0:
        return 0.0
    return float(average_precision_score(y_true, scores))


def normalized_auprc(y_true: np.ndarray, scores: np.ndarray) -> float:
    prevalence = float(np.mean(y_true))
    auprc = safe_auprc(y_true, scores)
    if prevalence >= 1.0:
        return float("nan")
    return float((auprc - prevalence) / (1.0 - prevalence))


def safe_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def binary_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    positives = y_true == 1
    negatives = y_true == 0
    return {
        "F1": float(f1_score(y_true, predicted, zero_division=0)),
        "sensitivity": (
            float(np.mean(predicted[positives] == 1))
            if positives.any()
            else float("nan")
        ),
        "specificity": (
            float(np.mean(predicted[negatives] == 0))
            if negatives.any()
            else float("nan")
        ),
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(scores, dtype=np.float64))
    if not len(values) or not np.isfinite(values).all():
        raise SystemExit("Threshold selection requires finite scores")
    if len(values) == 1:
        return np.asarray(
            [np.nextafter(values[0], -np.inf), np.nextafter(values[0], np.inf)]
        )
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        (
            [np.nextafter(values[0], -np.inf)],
            midpoints,
            [np.nextafter(values[-1], np.inf)],
        )
    )


def select_macro_f1_threshold(
    training: pd.DataFrame,
    score_column: str,
) -> tuple[float, float]:
    patients = sorted(training["patient"].astype(str).unique())
    if len(patients) < 1:
        raise SystemExit("Threshold selection has no training patients")
    candidates = candidate_thresholds(training[score_column].to_numpy(dtype=np.float64))
    patient_f1_rows: list[np.ndarray] = []
    for _, frame in training.groupby("patient", sort=True):
        scores = frame[score_column].to_numpy(dtype=np.float64)
        y_true = frame["y_true"].to_numpy(dtype=int)
        predicted = scores[:, np.newaxis] >= candidates[np.newaxis, :]
        positives = y_true[:, np.newaxis] == 1
        tp = np.sum(predicted & positives, axis=0)
        fp = np.sum(predicted & ~positives, axis=0)
        fn = np.sum(~predicted & positives, axis=0)
        denominator = 2 * tp + fp + fn
        patient_f1_rows.append(
            np.divide(
                2 * tp,
                denominator,
                out=np.zeros_like(denominator, dtype=np.float64),
                where=denominator > 0,
            )
        )
    macro_f1 = np.mean(np.vstack(patient_f1_rows), axis=0)
    best_f1 = float(np.max(macro_f1))
    tied_indices = np.flatnonzero(
        np.isclose(macro_f1, best_f1, atol=1e-12, rtol=0.0)
    )
    best_index = min(
        tied_indices,
        key=lambda index: (abs(candidates[index] - 0.5), candidates[index]),
    )
    return float(candidates[best_index]), best_f1


def apply_lopo_thresholds(
    reachable_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = reachable_predictions.copy()
    threshold_rows: list[dict[str, object]] = []
    for source_name, score_column in SCORE_SOURCES.items():
        prediction_column = f"predicted_burned_{source_name}"
        threshold_column = f"threshold_{source_name}"
        output[prediction_column] = -1
        output[threshold_column] = np.nan
        for band, band_frame in output.groupby("frequency_band", sort=False):
            patients = sorted(band_frame["patient"].astype(str).unique())
            if len(patients) < 2:
                raise SystemExit("LOPO threshold evaluation requires at least two patients")
            for heldout in patients:
                train = band_frame[band_frame["patient"].astype(str) != heldout]
                test_index = band_frame[
                    band_frame["patient"].astype(str) == heldout
                ].index
                threshold, training_macro_f1 = select_macro_f1_threshold(
                    train,
                    score_column,
                )
                predicted = (
                    output.loc[test_index, score_column].to_numpy(dtype=np.float64)
                    >= threshold
                ).astype(int)
                output.loc[test_index, prediction_column] = predicted
                output.loc[test_index, threshold_column] = threshold
                threshold_rows.append(
                    {
                        "frequency_band": band,
                        "heldout_patient": heldout,
                        "score_source": source_name,
                        "score_column": score_column,
                        "threshold": threshold,
                        "training_macro_F1": training_macro_f1,
                        "n_training_patients": len(patients) - 1,
                        "n_training_contacts": len(train),
                        "n_heldout_contacts": len(test_index),
                        "threshold_rule": "leave_one_patient_out_macro_f1",
                    }
                )
        if (output[prediction_column] < 0).any():
            raise SystemExit(f"Missing predictions for score source {source_name}")
        output[f"rank_within_patient_{source_name}"] = (
            output.groupby(["frequency_band", "patient"], sort=False)[score_column]
            .rank(method="first", ascending=False)
            .astype(int)
        )
        output[f"classification_state_{source_name}"] = np.select(
            [
                (output["y_true"] == 1) & (output[prediction_column] == 1),
                (output["y_true"] == 0) & (output[prediction_column] == 1),
                (output["y_true"] == 1) & (output[prediction_column] == 0),
            ],
            ["TP", "FP", "FN"],
            default="TN",
        )
    return output, pd.DataFrame.from_records(threshold_rows)


def patient_band_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_name, score_column in SCORE_SOURCES.items():
        prediction_column = f"predicted_burned_{source_name}"
        for (band, patient), frame in predictions.groupby(
            ["frequency_band", "patient"],
            sort=False,
        ):
            y_true = frame["y_true"].to_numpy(dtype=int)
            scores = frame[score_column].to_numpy(dtype=np.float64)
            predicted = frame[prediction_column].to_numpy(dtype=int)
            binary = binary_metrics(y_true, predicted)
            rows.append(
                {
                    "frequency_band": band,
                    "patient": patient,
                    "score_source": source_name,
                    "n_contacts": len(frame),
                    "n_burned": int(y_true.sum()),
                    "n_unburned": int((y_true == 0).sum()),
                    "prevalence": float(y_true.mean()),
                    "AUPRC": safe_auprc(y_true, scores),
                    "normalized_AUPRC": normalized_auprc(y_true, scores),
                    "AUROC": safe_auroc(y_true, scores),
                    **binary,
                    "TP": int(np.sum((y_true == 1) & (predicted == 1))),
                    "FP": int(np.sum((y_true == 0) & (predicted == 1))),
                    "FN": int(np.sum((y_true == 1) & (predicted == 0))),
                    "TN": int(np.sum((y_true == 0) & (predicted == 0))),
                }
            )
    return pd.DataFrame.from_records(rows)


def band_metrics(
    predictions: pd.DataFrame,
    patient_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = ("AUPRC", "normalized_AUPRC", "AUROC", "F1", "sensitivity", "specificity")
    for (band, source_name), patient_frame in patient_metrics.groupby(
        ["frequency_band", "score_source"],
        sort=False,
    ):
        score_column = SCORE_SOURCES[str(source_name)]
        prediction_column = f"predicted_burned_{source_name}"
        subset = predictions[predictions["frequency_band"].astype(str) == str(band)]
        y_true = subset["y_true"].to_numpy(dtype=int)
        scores = subset[score_column].to_numpy(dtype=np.float64)
        predicted = subset[prediction_column].to_numpy(dtype=int)
        pooled_binary = binary_metrics(y_true, predicted)
        row: dict[str, object] = {
            "frequency_band": band,
            "score_source": source_name,
            "n_contacts": len(subset),
            "n_patients": subset["patient"].nunique(),
            "n_burned": int(y_true.sum()),
            "n_unburned": int((y_true == 0).sum()),
            "prevalence": float(y_true.mean()),
            "pooled_AUPRC": safe_auprc(y_true, scores),
            "pooled_normalized_AUPRC": normalized_auprc(y_true, scores),
            "pooled_AUROC": safe_auroc(y_true, scores),
            **{f"pooled_{key}": value for key, value in pooled_binary.items()},
        }
        for metric in metrics:
            row[f"macro_{metric}"] = float(patient_frame[metric].mean())
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def descriptive_subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    group_masks: list[tuple[str, str, pd.Series]] = [
        (
            "coverage",
            "all_reachable",
            pd.Series(True, index=predictions.index),
        ),
        (
            "coverage",
            "reachable_direct",
            predictions["coverage_status"].astype(str).eq("reachable_direct"),
        ),
        (
            "coverage",
            "reachable_multihop",
            predictions["coverage_status"].astype(str).eq("reachable_multihop"),
        ),
    ]
    for tissue_class in ("white", "cortex", "ventricle", "unknown"):
        group_masks.append(
            (
                "tissue_class",
                tissue_class,
                predictions["tissue_class"].astype(str).eq(tissue_class),
            )
        )
    rows: list[dict[str, object]] = []
    for group_type, group_value, mask in group_masks:
        selected = predictions[mask]
        for source_name, score_column in SCORE_SOURCES.items():
            prediction_column = f"predicted_burned_{source_name}"
            for band, frame in selected.groupby("frequency_band", sort=False):
                y_true = frame["y_true"].to_numpy(dtype=int)
                scores = frame[score_column].to_numpy(dtype=np.float64)
                predicted = frame[prediction_column].to_numpy(dtype=int)
                binary = binary_metrics(y_true, predicted)
                rows.append(
                    {
                        "frequency_band": band,
                        "score_source": source_name,
                        "group_type": group_type,
                        "group_value": group_value,
                        "n_contacts": len(frame),
                        "n_patients": frame["patient"].nunique(),
                        "n_burned": int(y_true.sum()),
                        "n_unburned": int((y_true == 0).sum()),
                        "prevalence": float(y_true.mean()),
                        "pooled_AUPRC": safe_auprc(y_true, scores),
                        "pooled_normalized_AUPRC": normalized_auprc(
                            y_true,
                            scores,
                        ),
                        "pooled_AUROC": safe_auroc(y_true, scores),
                        **{f"pooled_{key}": value for key, value in binary.items()},
                    }
                )
    return pd.DataFrame.from_records(rows)


def patient_bootstrap(
    patient_metrics_frame: pd.DataFrame,
    *,
    n_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations < 1:
        return pd.DataFrame(), pd.DataFrame()
    patients = sorted(patient_metrics_frame["patient"].astype(str).unique())
    metrics = ("AUPRC", "normalized_AUPRC", "AUROC", "F1", "sensitivity", "specificity")
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    indexed = patient_metrics_frame.set_index(
        ["patient", "frequency_band", "score_source"]
    )
    bands = list(dict.fromkeys(patient_metrics_frame["frequency_band"].astype(str)))
    for iteration in range(1, n_iterations + 1):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        for band in bands:
            for source_name in SCORE_SOURCES:
                for metric in metrics:
                    values = np.asarray(
                        [
                            indexed.loc[(patient, band, source_name), metric]
                            for patient in sampled
                        ],
                        dtype=np.float64,
                    )
                    records.append(
                        {
                            "bootstrap_iteration": iteration,
                            "frequency_band": band,
                            "score_source": source_name,
                            "metric": metric,
                            "estimate": float(np.nanmean(values)),
                        }
                    )
    results = pd.DataFrame.from_records(records)
    ci = (
        results.groupby(
            ["frequency_band", "score_source", "metric"],
            sort=False,
        )["estimate"]
        .agg(
            bootstrap_mean="mean",
            ci_lower=lambda value: float(np.quantile(value, 0.025)),
            ci_upper=lambda value: float(np.quantile(value, 0.975)),
        )
        .reset_index()
    )
    return results, ci


def permutation_test(
    predictions: pd.DataFrame,
    observed_band_metrics: pd.DataFrame,
    *,
    metrics: Iterable[str],
    n_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations < 1:
        return pd.DataFrame(), pd.DataFrame()
    metric_names = tuple(str(value) for value in metrics)
    allowed = {"AUPRC", "normalized_AUPRC", "AUROC"}
    if not set(metric_names).issubset(allowed):
        raise SystemExit(f"Unsupported permutation metrics: {metric_names}")
    rng = np.random.default_rng(seed)
    base = predictions[
        [
            "frequency_band",
            "patient",
            "bipolar_channel",
            "y_true",
            *SCORE_SOURCES.values(),
        ]
    ].copy()
    records: list[dict[str, object]] = []
    for iteration in range(1, n_iterations + 1):
        permuted = base.copy()
        for (_, patient), index in permuted.groupby(
            ["frequency_band", "patient"],
            sort=False,
        ).groups.items():
            del patient
            values = permuted.loc[index, "y_true"].to_numpy(dtype=int)
            permuted.loc[index, "y_true"] = rng.permutation(values)
        for source_name, score_column in SCORE_SOURCES.items():
            for band, band_frame in permuted.groupby("frequency_band", sort=False):
                patient_values = {metric: [] for metric in metric_names}
                for _, frame in band_frame.groupby("patient", sort=True):
                    y_true = frame["y_true"].to_numpy(dtype=int)
                    scores = frame[score_column].to_numpy(dtype=np.float64)
                    values = {
                        "AUPRC": safe_auprc(y_true, scores),
                        "normalized_AUPRC": normalized_auprc(y_true, scores),
                        "AUROC": safe_auroc(y_true, scores),
                    }
                    for metric in metric_names:
                        patient_values[metric].append(values[metric])
                for metric in metric_names:
                    records.append(
                        {
                            "permutation_iteration": iteration,
                            "frequency_band": band,
                            "score_source": source_name,
                            "metric": metric,
                            "null_macro_metric": float(
                                np.nanmean(patient_values[metric])
                            ),
                        }
                    )
    results = pd.DataFrame.from_records(records)
    observed_lookup = observed_band_metrics.set_index(
        ["frequency_band", "score_source"]
    )
    summary_rows: list[dict[str, object]] = []
    for source_name in SCORE_SOURCES:
        for metric in metric_names:
            source_metric = results[
                (results["score_source"] == source_name)
                & (results["metric"] == metric)
            ]
            max_null = source_metric.groupby("permutation_iteration")[
                "null_macro_metric"
            ].max()
            for band in source_metric["frequency_band"].astype(str).unique():
                observed = float(
                    observed_lookup.loc[(band, source_name), f"macro_{metric}"]
                )
                null = source_metric[
                    source_metric["frequency_band"].astype(str) == band
                ]["null_macro_metric"]
                summary_rows.append(
                    {
                        "frequency_band": band,
                        "score_source": source_name,
                        "metric": metric,
                        "observed_macro_metric": observed,
                        "null_mean": float(null.mean()),
                        "permutation_p_raw": float(
                            (np.sum(null >= observed) + 1) / (len(null) + 1)
                        ),
                        "permutation_p_maxT_FWER": float(
                            (np.sum(max_null >= observed) + 1)
                            / (len(max_null) + 1)
                        ),
                    }
                )
    return results, pd.DataFrame.from_records(summary_rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_csv(frame: pd.DataFrame, path: Path, *, overwrite: bool) -> None:
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


def atomic_write_json(payload: dict[str, object], path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output exists; pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".json",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    try:
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--anchor-predictions", type=Path)
    parser.add_argument("--audit-workbook", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patients", nargs="+")
    parser.add_argument("--bands", nargs="+")
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--n-permutations", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _validate_config(config: dict[str, object]) -> None:
    graph = dict(config["graph"])
    extension = dict(config["extension"])
    if graph["coordinate_space"] != "SCS" or graph["coordinate_unit"] != "mm":
        raise SystemExit("Graph extension must use patient-native SCS millimeter coordinates")
    if graph["neighbor_rule"] != "mutual_knn_with_radius_cap":
        raise SystemExit("Unsupported graph neighbor rule")
    if extension["method"] != "clamped_harmonic_extension":
        raise SystemExit("Unsupported graph extension method")
    if extension["primary_anchor_score"] != "smoothed_score":
        raise SystemExit("Primary anchor score must remain smoothed_score")
    if extension["sensitivity_anchor_score"] != "y_score":
        raise SystemExit("Sensitivity anchor score must remain y_score")
    if extension["unanchored_component_rule"] != "leave_nan":
        raise SystemExit("Unanchored components must remain unscored")


def _check_full_run_expectations(
    *,
    coordinates: pd.DataFrame,
    predictions: pd.DataFrame,
    reachable: pd.DataFrame,
    graph_audit: pd.DataFrame,
    expected: dict[str, object],
) -> None:
    checks = {
        "all_formal_contacts": len(coordinates),
        "eligible_anchor_contacts": int((~coordinates["is_excluded_tissue"]).sum()),
        "excluded_contacts": int(coordinates["is_excluded_tissue"].sum()),
        "excluded_burned": int(
            predictions.drop_duplicates(["patient", "bipolar_channel"])["y_true"].sum()
        ),
        "excluded_unburned": int(
            (
                predictions.drop_duplicates(["patient", "bipolar_channel"])["y_true"]
                == 0
            ).sum()
        ),
        "reachable_excluded_contacts": int(
            predictions.drop_duplicates(["patient", "bipolar_channel"])[
                "reachable_from_eligible_anchor"
            ].sum()
        ),
        "unreachable_excluded_contacts": int(
            (
                ~predictions.drop_duplicates(["patient", "bipolar_channel"])[
                    "reachable_from_eligible_anchor"
                ]
            ).sum()
        ),
        "direct_anchor_excluded_contacts": int(
            predictions.drop_duplicates(["patient", "bipolar_channel"])[
                "direct_anchor_neighbor"
            ].sum()
        ),
        "full_graph_edges": int(graph_audit["n_edges"].sum()),
        "full_graph_isolated_nodes": int(graph_audit["n_isolated"].sum()),
        "full_prediction_rows": len(predictions),
        "reachable_prediction_rows": len(reachable),
    }
    mismatches = {
        key: (int(expected[key]), value)
        for key, value in checks.items()
        if int(expected[key]) != value
    }
    if mismatches:
        raise SystemExit(f"Full-run expectation mismatch: {mismatches}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    _validate_config(config)
    inputs = dict(config["inputs"])
    graph_config = dict(config["graph"])
    stats_config = dict(config["statistics"])
    expected = dict(config["expected"])
    patients = args.patients or [str(value) for value in config["patients"]]
    bands = args.bands or [str(value) for value in config["frequency_bands"]]
    unknown_patients = sorted(set(patients) - set(PATIENTS))
    if unknown_patients:
        raise SystemExit(f"Unknown patients: {unknown_patients}")

    coordinate_path = args.coordinates or Path(inputs["coordinates"])
    anchor_path = args.anchor_predictions or Path(inputs["anchor_predictions"])
    audit_path = args.audit_workbook or Path(inputs["audit_workbook"])
    coordinates = load_coordinates(coordinate_path, patients)
    anchors = load_anchor_predictions(anchor_path, bands, patients)
    extended, node_audit, graph_audit = extend_scores_to_excluded(
        coordinates,
        anchors,
        n_neighbors=int(graph_config["n_neighbors"]),
        max_distance_mm=float(graph_config["max_distance_mm"]),
        gaussian_sigma_mm=float(graph_config["gaussian_sigma_mm"]),
    )

    labels = load_audit_labels(audit_path, patients)
    extended = attach_excluded_labels(extended, coordinates, labels)
    reachable = extended[
        extended["reachable_from_eligible_anchor"].astype(bool)
    ].copy()
    if reachable[list(SCORE_SOURCES.values())].isna().any().any():
        raise SystemExit("Reachable excluded contacts contain missing propagated scores")
    unreachable = extended[
        ~extended["reachable_from_eligible_anchor"].astype(bool)
    ].copy()
    if not unreachable[list(SCORE_SOURCES.values())].isna().all().all():
        raise SystemExit("Unreachable excluded contacts unexpectedly received scores")

    predicted, thresholds = apply_lopo_thresholds(reachable)
    patient_metrics_frame = patient_band_metrics(predicted)
    band_metrics_frame = band_metrics(predicted, patient_metrics_frame)
    subgroup_metrics_frame = descriptive_subgroup_metrics(predicted)

    full_run = args.patients is None and args.bands is None
    if full_run:
        _check_full_run_expectations(
            coordinates=coordinates,
            predictions=extended,
            reachable=reachable,
            graph_audit=graph_audit,
            expected=expected,
        )

    unique_targets = extended.drop_duplicates(["patient", "bipolar_channel"])
    print(
        f"all_contacts={len(coordinates)} anchors={int((~coordinates['is_excluded_tissue']).sum())} "
        f"excluded={len(unique_targets)} reachable={int(unique_targets['reachable_from_eligible_anchor'].sum())} "
        f"unreachable={int((~unique_targets['reachable_from_eligible_anchor']).sum())}"
    )
    primary_metrics = band_metrics_frame[
        band_metrics_frame["score_source"] == PRIMARY_SCORE_SOURCE
    ]
    print(
        "primary_macro="
        + ", ".join(
            f"{row.frequency_band}:AUPRC={row.macro_AUPRC:.6f},"
            f"AUROC={row.macro_AUROC:.6f},F1={row.macro_F1:.6f}"
            for row in primary_metrics.itertuples()
        )
    )
    if not args.apply:
        print("dry_run=true; no files written")
        return

    output_root = args.out or Path(inputs["output_root"])
    assert_allowed_output_root(output_root)
    observed_outputs = {
        "full_graph_node_audit.csv": node_audit,
        "full_graph_structure_audit.csv": graph_audit,
        "excluded_contact_predictions_by_band.csv": predicted,
        "excluded_contact_unreachable_by_band.csv": unreachable,
        "excluded_contact_thresholds.csv": thresholds,
        "excluded_contact_patient_band_metrics.csv": patient_metrics_frame,
        "excluded_contact_metrics_by_band.csv": band_metrics_frame,
        "excluded_contact_descriptive_subgroup_metrics.csv": subgroup_metrics_frame,
    }
    status_path = output_root / "observed_complete.json"
    prospective = [output_root / name for name in observed_outputs]
    prospective.append(status_path)
    existing = [path for path in prospective if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Observed outputs exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing[:5])
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in observed_outputs.items():
        atomic_write_csv(frame, output_root / name, overwrite=args.overwrite)
    atomic_write_json(
        {
            "status": "observed_complete",
            "n_prediction_rows": len(predicted),
            "n_unreachable_rows": len(unreachable),
            "frequency_bands": bands,
            "patients": patients,
        },
        status_path,
        overwrite=args.overwrite,
    )

    n_bootstrap = (
        args.n_bootstrap
        if args.n_bootstrap is not None
        else int(stats_config["patient_bootstrap_iterations"])
    )
    n_permutations = (
        args.n_permutations
        if args.n_permutations is not None
        else int(stats_config["within_patient_label_permutations"])
    )
    bootstrap, bootstrap_ci = patient_bootstrap(
        patient_metrics_frame,
        n_iterations=n_bootstrap,
        seed=int(stats_config["random_seed"]),
    )
    permutation, permutation_summary = permutation_test(
        predicted,
        band_metrics_frame,
        metrics=stats_config["permutation_metrics"],
        n_iterations=n_permutations,
        seed=int(stats_config["random_seed"]),
    )
    statistical_outputs = {
        "excluded_contact_bootstrap_results.csv": bootstrap,
        "excluded_contact_bootstrap_ci.csv": bootstrap_ci,
    }
    if n_permutations > 0:
        statistical_outputs["excluded_contact_permutation_results.csv"] = permutation
        statistical_outputs["excluded_contact_permutation_summary.csv"] = (
            permutation_summary
        )
    for name, frame in statistical_outputs.items():
        atomic_write_csv(frame, output_root / name, overwrite=args.overwrite)

    summary = {
        "status": "complete",
        "analysis_scope": "posthoc_exploratory_spatial_extension",
        "primary_score_source": PRIMARY_SCORE_SOURCE,
        "coordinate_manifest": str(coordinate_path),
        "coordinate_manifest_sha256": file_sha256(coordinate_path),
        "anchor_predictions": str(anchor_path),
        "anchor_predictions_sha256": file_sha256(anchor_path),
        "audit_workbook": str(audit_path),
        "audit_workbook_sha256": file_sha256(audit_path),
        "patients": patients,
        "frequency_bands": bands,
        "n_all_contacts": len(coordinates),
        "n_eligible_anchors": int((~coordinates["is_excluded_tissue"]).sum()),
        "n_excluded_contacts": len(unique_targets),
        "n_reachable_excluded_contacts": int(
            unique_targets["reachable_from_eligible_anchor"].sum()
        ),
        "n_unreachable_excluded_contacts": int(
            (~unique_targets["reachable_from_eligible_anchor"]).sum()
        ),
        "n_direct_anchor_excluded_contacts": int(
            unique_targets["direct_anchor_neighbor"].sum()
        ),
        "n_excluded_burned": int(unique_targets["y_true"].sum()),
        "n_excluded_unburned": int((unique_targets["y_true"] == 0).sum()),
        "graph": graph_config,
        "extension": dict(config["extension"]),
        "n_bootstrap": n_bootstrap,
        "n_permutations": n_permutations,
    }
    atomic_write_json(
        summary,
        output_root / "excluded_contact_run_summary.json",
        overwrite=args.overwrite,
    )
    print(f"wrote={output_root}")


if __name__ == "__main__":
    main()
