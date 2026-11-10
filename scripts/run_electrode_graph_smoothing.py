"""Apply fixed, geometry-only graph diffusion to electrode LOPO predictions."""
from __future__ import annotations

from pipeline_config import REPO_ROOT, output_root, source_root, config_root, private_value, expand_config, assert_output_path

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import yaml


ROOT = output_root()
AUX_ROOT = source_root("aux").resolve()
CONFIG_PATH = config_root() / "electrode_graph_smoothing.yaml"
COORDINATE_ROOT = ROOT / "data_derived" / "electrode_coordinates"
OUTPUT_ROOT = ROOT / "data_derived" / "electrode_graph_smoothing"
RADIUS_SENSITIVITY_ROOT = (
    ROOT / "data_derived" / "electrode_graph_smoothing_radius_sensitivity"
)
CHECK_ROOT = ROOT / "checks"
KEYS = ["frequency_band", "patient", "bipolar_channel"]


@dataclass(frozen=True)
class SpatialGraph:
    channels: tuple[str, ...]
    transition: np.ndarray
    weights: np.ndarray
    distances_mm: np.ndarray
    n_edges: int
    n_isolated: int
    n_same_shaft_edges: int
    n_cross_shaft_edges: int


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
    coordinate_root = _resolve(COORDINATE_ROOT)
    output_root = _resolve(OUTPUT_ROOT)
    sensitivity_root = _resolve(RADIUS_SENSITIVITY_ROOT)
    tmp_root = _resolve(CHECK_ROOT)
    if resolved == AUX_ROOT or _is_relative_to(resolved, AUX_ROOT):
        raise SystemExit(f"Refusing to write under read-only source root: {resolved}")
    if not (
        resolved == coordinate_root
        or _is_relative_to(resolved, coordinate_root)
        or resolved == output_root
        or _is_relative_to(resolved, output_root)
        or resolved == sensitivity_root
        or _is_relative_to(resolved, sensitivity_root)
        or resolved == tmp_root
        or _is_relative_to(resolved, tmp_root)
    ):
        raise SystemExit(f"Refusing to write outside reviewed output roots: {resolved}")


def load_config(path: Path = CONFIG_PATH) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return expand_config(yaml.safe_load(handle))


def shaft_name(channel: str) -> str:
    match = re.match(r"^([A-Za-z]+'?)\d+-", str(channel))
    if match is None:
        raise SystemExit(f"Cannot parse bipolar shaft from {channel!r}")
    return match.group(1)


def load_predictions(
    path: Path,
    bands: Iterable[str],
    patients: Iterable[str] | None = None,
) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(f"Missing electrode prediction table: {path}")
    frame = pd.read_csv(path)
    required = {
        "frequency_band",
        "heldout_patient",
        "patient",
        "bipolar_channel",
        "y_true",
        "y_score",
        "y_pred",
        "rank_within_patient",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    selected_bands = [str(value) for value in bands]
    frame = frame[frame["frequency_band"].astype(str).isin(selected_bands)].copy()
    if patients is not None:
        selected_patients = {str(value) for value in patients}
        frame = frame[frame["patient"].astype(str).isin(selected_patients)].copy()
    if frame.empty:
        raise SystemExit("No prediction rows remain after filtering")
    if frame.duplicated(KEYS).any():
        duplicates = frame.loc[frame.duplicated(KEYS, keep=False), KEYS].head(5)
        raise SystemExit(f"Duplicate prediction rows: {duplicates.to_dict('records')}")
    if not (frame["patient"].astype(str) == frame["heldout_patient"].astype(str)).all():
        raise SystemExit("Prediction table contains non-heldout patient rows")
    if not frame["y_true"].isin([0, 1]).all() or not frame["y_pred"].isin([0, 1]).all():
        raise SystemExit("y_true and y_pred must be binary")
    scores = frame["y_score"].to_numpy(dtype=np.float64)
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise SystemExit("Prediction scores must be finite probabilities")
    expected_band_set = set(selected_bands)
    actual_band_set = set(frame["frequency_band"].astype(str))
    if actual_band_set != expected_band_set:
        raise SystemExit(
            f"Frequency-band mismatch: expected={sorted(expected_band_set)} "
            f"actual={sorted(actual_band_set)}"
        )
    return frame.sort_values(KEYS).reset_index(drop=True)


def load_coordinates(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise SystemExit(
            f"Missing formal coordinate manifest: {path}. "
            "Run build_bipolar_coordinate_manifest.py first."
        )
    frame = pd.read_csv(path)
    required = {
        "patient",
        "bipolar_channel",
        "scs_x_mm",
        "scs_y_mm",
        "scs_z_mm",
        "qc_status",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    if frame.duplicated(["patient", "bipolar_channel"]).any():
        raise SystemExit(f"{path}: duplicate patient/bipolar coordinates")
    xyz = frame[["scs_x_mm", "scs_y_mm", "scs_z_mm"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise SystemExit(f"{path}: non-finite SCS coordinates")
    if not frame["qc_status"].astype(str).eq("pass").all():
        raise SystemExit(f"{path}: coordinate rows without qc_status=pass")
    return frame


def align_coordinates(
    predictions: pd.DataFrame,
    coordinates: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        "patient",
        "bipolar_channel",
        "scs_x_mm",
        "scs_y_mm",
        "scs_z_mm",
    ]
    output = predictions.merge(
        coordinates[columns],
        on=["patient", "bipolar_channel"],
        how="left",
        validate="many_to_one",
    )
    missing = output["scs_x_mm"].isna()
    if missing.any():
        rows = output.loc[missing, ["patient", "bipolar_channel"]].drop_duplicates()
        raise SystemExit(f"Predictions missing coordinates: {rows.head(10).to_dict('records')}")
    for patient, frame in output.groupby("patient", sort=True):
        channel_sets = {
            tuple(group["bipolar_channel"].astype(str))
            for _, group in frame.groupby("frequency_band", sort=False)
        }
        if len(channel_sets) != 1:
            raise SystemExit(f"{patient}: channel order/set differs across frequency bands")
    return output


def build_mutual_knn_graph(
    channels: Iterable[str],
    xyz_mm: np.ndarray,
    *,
    n_neighbors: int,
    max_distance_mm: float,
    gaussian_sigma_mm: float,
) -> SpatialGraph:
    channel_tuple = tuple(str(value) for value in channels)
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.shape != (len(channel_tuple), 3) or not np.isfinite(xyz).all():
        raise SystemExit("Graph coordinates must be finite n_channels x 3 values")
    if len(set(channel_tuple)) != len(channel_tuple):
        raise SystemExit("Graph channel names must be unique")
    if n_neighbors < 1 or max_distance_mm <= 0 or gaussian_sigma_mm <= 0:
        raise SystemExit("Invalid graph parameters")

    delta = xyz[:, np.newaxis, :] - xyz[np.newaxis, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    n_nodes = len(channel_tuple)
    directed = np.zeros((n_nodes, n_nodes), dtype=bool)
    indices = np.arange(n_nodes)
    for node in range(n_nodes):
        eligible = (
            (indices != node)
            & np.isfinite(distances[node])
            & (distances[node] <= max_distance_mm)
        )
        candidates = indices[eligible]
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
    degree = weights.sum(axis=1)
    transition = np.zeros_like(weights)
    connected = degree > 0
    transition[connected] = weights[connected] / degree[connected, np.newaxis]
    transition[~connected, ~connected] = 1.0

    upper_i, upper_j = np.where(np.triu(weights, k=1) > 0)
    shafts = np.asarray([shaft_name(value) for value in channel_tuple], dtype=object)
    same_shaft = shafts[upper_i] == shafts[upper_j]
    return SpatialGraph(
        channels=channel_tuple,
        transition=transition,
        weights=weights,
        distances_mm=distances,
        n_edges=len(upper_i),
        n_isolated=int(np.sum(~connected)),
        n_same_shaft_edges=int(np.sum(same_shaft)),
        n_cross_shaft_edges=int(np.sum(~same_shaft)),
    )


def ppr_diffuse_scores(
    scores: np.ndarray,
    transition: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    matrix = np.asarray(transition, dtype=np.float64)
    if values.ndim != 1 or matrix.shape != (len(values), len(values)):
        raise SystemExit("Score/transition shape mismatch")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise SystemExit("PPR input scores must be finite probabilities")
    if not 0.0 <= alpha < 1.0:
        raise SystemExit("PPR alpha must be in [0, 1)")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
        raise SystemExit("PPR transition matrix must be row stochastic")
    smoothed = np.linalg.solve(
        np.eye(len(values), dtype=np.float64) - alpha * matrix,
        (1.0 - alpha) * values,
    )
    if not np.isfinite(smoothed).all():
        raise SystemExit("PPR produced non-finite scores")
    if np.any((smoothed < -1e-10) | (smoothed > 1.0 + 1e-10)):
        raise SystemExit("PPR produced scores outside probability bounds")
    return np.clip(smoothed, 0.0, 1.0)


def count_preserving_predictions(frame: pd.DataFrame) -> np.ndarray:
    required = {"smoothed_score", "y_score", "y_pred", "bipolar_channel"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Count-preserving prediction missing columns {missing}")
    n_positive = int(frame["y_pred"].sum())
    ordered = frame.sort_values(
        ["smoothed_score", "y_score", "bipolar_channel"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    predicted = pd.Series(0, index=frame.index, dtype=int)
    predicted.loc[ordered.index[:n_positive]] = 1
    if int(predicted.sum()) != n_positive:
        raise SystemExit("Count-preserving classification changed positive count")
    return predicted.loc[frame.index].to_numpy(dtype=int)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def binary_metrics(y_true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    positives = y_true == 1
    negatives = y_true == 0
    return {
        "F1": float(f1_score(y_true, predicted)),
        "sensitivity": float(np.mean(predicted[positives] == 1)),
        "specificity": float(np.mean(predicted[negatives] == 0)),
    }


def patient_band_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (band, patient), frame in predictions.groupby(
        ["frequency_band", "patient"], sort=False
    ):
        y_true = frame["y_true"].to_numpy(dtype=int)
        original_score = frame["y_score"].to_numpy(dtype=np.float64)
        smoothed_score = frame["smoothed_score"].to_numpy(dtype=np.float64)
        original_pred = frame["y_pred"].to_numpy(dtype=int)
        smoothed_pred = frame["smoothed_pred"].to_numpy(dtype=int)
        original_binary = binary_metrics(y_true, original_pred)
        smoothed_binary = binary_metrics(y_true, smoothed_pred)
        rows.append(
            {
                "frequency_band": band,
                "patient": patient,
                "n_contacts": len(frame),
                "n_positive_labels": int(y_true.sum()),
                "n_predicted_positive": int(original_pred.sum()),
                "original_AUPRC": float(
                    average_precision_score(y_true, original_score)
                ),
                "smoothed_AUPRC": float(
                    average_precision_score(y_true, smoothed_score)
                ),
                "original_AUROC": _safe_auc(y_true, original_score),
                "smoothed_AUROC": _safe_auc(y_true, smoothed_score),
                **{f"original_{key}": value for key, value in original_binary.items()},
                **{f"smoothed_{key}": value for key, value in smoothed_binary.items()},
            }
        )
    output = pd.DataFrame.from_records(rows)
    for metric in ("AUPRC", "AUROC", "F1", "sensitivity", "specificity"):
        output[f"delta_{metric}"] = (
            output[f"smoothed_{metric}"] - output[f"original_{metric}"]
        )
    return output


def metrics_by_band(
    predictions: pd.DataFrame,
    patient_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for band, frame in predictions.groupby("frequency_band", sort=False):
        patient_frame = patient_metrics[patient_metrics["frequency_band"] == band]
        y_true = frame["y_true"].to_numpy(dtype=int)
        original_pred = frame["y_pred"].to_numpy(dtype=int)
        smoothed_pred = frame["smoothed_pred"].to_numpy(dtype=int)
        original_binary = binary_metrics(y_true, original_pred)
        smoothed_binary = binary_metrics(y_true, smoothed_pred)
        row: dict[str, object] = {
            "frequency_band": band,
            "n_contacts": len(frame),
            "n_patients": frame["patient"].nunique(),
            "macro_original_AUPRC": float(patient_frame["original_AUPRC"].mean()),
            "macro_smoothed_AUPRC": float(patient_frame["smoothed_AUPRC"].mean()),
            "macro_original_AUROC": float(patient_frame["original_AUROC"].mean()),
            "macro_smoothed_AUROC": float(patient_frame["smoothed_AUROC"].mean()),
            "macro_original_F1": float(patient_frame["original_F1"].mean()),
            "macro_smoothed_F1": float(patient_frame["smoothed_F1"].mean()),
            "macro_original_sensitivity": float(
                patient_frame["original_sensitivity"].mean()
            ),
            "macro_smoothed_sensitivity": float(
                patient_frame["smoothed_sensitivity"].mean()
            ),
            "macro_original_specificity": float(
                patient_frame["original_specificity"].mean()
            ),
            "macro_smoothed_specificity": float(
                patient_frame["smoothed_specificity"].mean()
            ),
            **{f"original_{key}": value for key, value in original_binary.items()},
            **{f"smoothed_{key}": value for key, value in smoothed_binary.items()},
        }
        for metric in ("AUPRC", "AUROC", "F1", "sensitivity", "specificity"):
            row[f"delta_macro_{metric}"] = (
                row[f"macro_smoothed_{metric}"]
                - row[f"macro_original_{metric}"]
            )
        for metric in ("F1", "sensitivity", "specificity"):
            row[f"delta_{metric}"] = (
                row[f"smoothed_{metric}"] - row[f"original_{metric}"]
            )
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def apply_graph_smoothing(
    predictions: pd.DataFrame,
    *,
    n_neighbors: int,
    max_distance_mm: float,
    gaussian_sigma_mm: float,
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_parts: list[pd.DataFrame] = []
    graph_rows: list[dict[str, object]] = []
    graph_cache: dict[str, SpatialGraph] = {}
    for patient, patient_frame in predictions.groupby("patient", sort=True):
        first_band = str(patient_frame["frequency_band"].iloc[0])
        reference = patient_frame[
            patient_frame["frequency_band"].astype(str) == first_band
        ].sort_values("bipolar_channel")
        graph = build_mutual_knn_graph(
            reference["bipolar_channel"].astype(str),
            reference[["scs_x_mm", "scs_y_mm", "scs_z_mm"]].to_numpy(),
            n_neighbors=n_neighbors,
            max_distance_mm=max_distance_mm,
            gaussian_sigma_mm=gaussian_sigma_mm,
        )
        graph_cache[str(patient)] = graph
        graph_rows.append(
            {
                "patient": patient,
                "n_nodes": len(graph.channels),
                "n_edges": graph.n_edges,
                "n_isolated": graph.n_isolated,
                "n_same_shaft_edges": graph.n_same_shaft_edges,
                "n_cross_shaft_edges": graph.n_cross_shaft_edges,
                "n_neighbors": n_neighbors,
                "max_distance_mm": max_distance_mm,
                "gaussian_sigma_mm": gaussian_sigma_mm,
                "neighbor_rule": "mutual_knn_with_radius_cap",
            }
        )

    for (band, patient), frame in predictions.groupby(
        ["frequency_band", "patient"], sort=False
    ):
        ordered = frame.sort_values("bipolar_channel").copy()
        graph = graph_cache[str(patient)]
        if tuple(ordered["bipolar_channel"].astype(str)) != graph.channels:
            raise SystemExit(f"{patient} {band}: graph/prediction channel mismatch")
        ordered["smoothed_score"] = ppr_diffuse_scores(
            ordered["y_score"].to_numpy(dtype=np.float64),
            graph.transition,
            alpha=alpha,
        )
        ordered["smoothed_pred"] = count_preserving_predictions(ordered)
        ordered["original_rank"] = ordered["y_score"].rank(
            method="first", ascending=False
        ).astype(int)
        ordered["smoothed_rank"] = ordered["smoothed_score"].rank(
            method="first", ascending=False
        ).astype(int)
        ordered["score_delta"] = ordered["smoothed_score"] - ordered["y_score"]
        ordered["rank_delta"] = ordered["original_rank"] - ordered["smoothed_rank"]
        ordered["flip_direction"] = np.select(
            [
                (ordered["y_pred"] == 0) & (ordered["smoothed_pred"] == 1),
                (ordered["y_pred"] == 1) & (ordered["smoothed_pred"] == 0),
            ],
            ["0_to_1", "1_to_0"],
            default="unchanged",
        )
        state_before = np.where(
            ordered["y_true"] == 1,
            np.where(ordered["y_pred"] == 1, "TP", "FN"),
            np.where(ordered["y_pred"] == 1, "FP", "TN"),
        )
        state_after = np.where(
            ordered["y_true"] == 1,
            np.where(ordered["smoothed_pred"] == 1, "TP", "FN"),
            np.where(ordered["smoothed_pred"] == 1, "FP", "TN"),
        )
        ordered["classification_transition"] = [
            before if before == after else f"{before}_to_{after}"
            for before, after in zip(state_before, state_after, strict=True)
        ]
        neighbor_names: list[str] = []
        neighbor_distances: list[str] = []
        weighted_degrees: list[float] = []
        for index in range(len(ordered)):
            neighbors = np.flatnonzero(graph.weights[index] > 0)
            neighbor_names.append("|".join(graph.channels[value] for value in neighbors))
            neighbor_distances.append(
                "|".join(f"{graph.distances_mm[index, value]:.6g}" for value in neighbors)
            )
            weighted_degrees.append(float(graph.weights[index].sum()))
        ordered["graph_neighbors"] = neighbor_names
        ordered["graph_neighbor_distances_mm"] = neighbor_distances
        ordered["graph_weighted_degree"] = weighted_degrees
        ordered["graph_alpha"] = alpha
        ordered["graph_direct_self_evidence"] = 1.0 - alpha
        ordered["smoothing_method"] = "personalized_pagerank_restart_diffusion"
        ordered["score_domain"] = "probability"
        ordered["hard_classification_rule"] = (
            "preserve_original_positive_count_within_patient_band"
        )
        ordered["analysis_scope"] = "posthoc_exploratory"
        output_parts.append(ordered)
    output = pd.concat(output_parts, ignore_index=True).sort_values(KEYS)
    for _, frame in output.groupby(["frequency_band", "patient"]):
        if int(frame["y_pred"].sum()) != int(frame["smoothed_pred"].sum()):
            raise SystemExit("Positive-count preservation failed")
    return output.reset_index(drop=True), pd.DataFrame.from_records(graph_rows)


def paired_patient_bootstrap(
    patient_metrics: pd.DataFrame,
    *,
    n_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations < 1:
        return pd.DataFrame(), pd.DataFrame()
    patients = sorted(patient_metrics["patient"].astype(str).unique())
    bands = list(dict.fromkeys(patient_metrics["frequency_band"].astype(str)))
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    metrics = ("AUPRC", "AUROC", "F1")
    indexed = patient_metrics.set_index(["patient", "frequency_band"])
    for iteration in range(1, n_iterations + 1):
        sampled = rng.choice(patients, size=len(patients), replace=True)
        for metric in metrics:
            band_values: list[float] = []
            for band in bands:
                values = np.asarray(
                    [indexed.loc[(patient, band), f"delta_{metric}"] for patient in sampled],
                    dtype=np.float64,
                )
                estimate = float(np.nanmean(values))
                band_values.append(estimate)
                records.append(
                    {
                        "bootstrap_iteration": iteration,
                        "scope": band,
                        "metric": metric,
                        "delta": estimate,
                    }
                )
            records.append(
                {
                    "bootstrap_iteration": iteration,
                    "scope": "mean_across_bands",
                    "metric": metric,
                    "delta": float(np.nanmean(band_values)),
                }
            )
    results = pd.DataFrame.from_records(records)
    ci = (
        results.groupby(["scope", "metric"], sort=False)["delta"]
        .agg(
            estimate="mean",
            ci_lower=lambda value: float(np.quantile(value, 0.025)),
            ci_upper=lambda value: float(np.quantile(value, 0.975)),
        )
        .reset_index()
    )
    return results, ci


def weighted_label_roughness(
    labels: np.ndarray,
    weights: np.ndarray,
) -> float:
    y = np.asarray(labels, dtype=np.float64)
    upper_i, upper_j = np.where(np.triu(weights, k=1) > 0)
    edge_weights = weights[upper_i, upper_j]
    if not len(edge_weights) or float(edge_weights.sum()) == 0.0:
        return float("nan")
    return float(
        np.sum(edge_weights * (y[upper_i] - y[upper_j]) ** 2)
        / edge_weights.sum()
    )


def spatial_label_permutation(
    predictions: pd.DataFrame,
    graphs: dict[str, SpatialGraph],
    *,
    n_iterations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    if n_iterations < 1:
        return pd.DataFrame(), {}
    first_band = str(predictions["frequency_band"].iloc[0])
    frames = {
        str(patient): frame.sort_values("bipolar_channel")
        for patient, frame in predictions[
            predictions["frequency_band"].astype(str) == first_band
        ].groupby("patient", sort=True)
    }
    observed_values = [
        weighted_label_roughness(
            frame["y_true"].to_numpy(dtype=int),
            graphs[patient].weights,
        )
        for patient, frame in frames.items()
    ]
    observed = float(np.nanmean(observed_values))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for iteration in range(1, n_iterations + 1):
        values = [
            weighted_label_roughness(
                rng.permutation(frame["y_true"].to_numpy(dtype=int)),
                graphs[patient].weights,
            )
            for patient, frame in frames.items()
        ]
        rows.append(
            {
                "permutation_iteration": iteration,
                "null_mean_roughness": float(np.nanmean(values)),
                "observed_mean_roughness": observed,
            }
        )
    output = pd.DataFrame.from_records(rows)
    p_value = float(
        (np.sum(output["null_mean_roughness"] <= observed) + 1)
        / (len(output) + 1)
    )
    summary = {
        "observed_mean_roughness": observed,
        "null_mean_roughness": float(output["null_mean_roughness"].mean()),
        "spatial_clustering_p_lower": p_value,
    }
    return output, summary


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
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--patients", nargs="+")
    parser.add_argument("--bands", nargs="+")
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--n-spatial-permutations", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_config = dict(config["modeling"])
    graph_config = dict(config["graph"])
    smoothing_config = dict(config["smoothing"])
    stats_config = dict(config["statistics"])
    expected = dict(config["expected"])
    bands = args.bands or [str(value) for value in model_config["frequency_bands"]]
    if graph_config["coordinate_space"] != "SCS":
        raise SystemExit("Formal graph distances must use patient-native SCS coordinates")
    if graph_config["neighbor_rule"] != "mutual_knn_with_radius_cap":
        raise SystemExit("Unsupported graph neighbor rule")
    if smoothing_config["method"] != "personalized_pagerank_restart_diffusion":
        raise SystemExit("Unsupported smoothing method")
    if smoothing_config["score_domain"] != "probability":
        raise SystemExit("Formal graph smoothing must operate in probability space")
    if smoothing_config["hard_classification"] != (
        "preserve_original_positive_count_within_patient_band"
    ):
        raise SystemExit("Unsupported hard-classification rule")
    alpha = float(smoothing_config["alpha"])
    if not np.isclose(
        float(smoothing_config["direct_self_evidence"]),
        1.0 - alpha,
        atol=1e-12,
    ):
        raise SystemExit("direct_self_evidence must equal 1 - alpha")
    prediction_path = args.predictions or Path(model_config["predictions"])
    coordinate_path = args.coordinates or Path(
        dict(config["coordinate_sources"])["output_manifest"]
    )
    predictions = load_predictions(prediction_path, bands, args.patients)
    predictions = align_coordinates(predictions, load_coordinates(coordinate_path))
    smoothed, graph_audit = apply_graph_smoothing(
        predictions,
        n_neighbors=int(graph_config["n_neighbors"]),
        max_distance_mm=float(graph_config["max_distance_mm"]),
        gaussian_sigma_mm=float(graph_config["gaussian_sigma_mm"]),
        alpha=alpha,
    )
    patient_metrics = patient_band_metrics(smoothed)
    band_metrics = metrics_by_band(smoothed, patient_metrics)

    n_bootstrap = (
        args.n_bootstrap
        if args.n_bootstrap is not None
        else int(stats_config["patient_bootstrap_iterations"])
    )
    bootstrap, bootstrap_ci = paired_patient_bootstrap(
        patient_metrics,
        n_iterations=n_bootstrap,
        seed=int(stats_config["random_seed"]),
    )
    graph_lookup: dict[str, SpatialGraph] = {}
    for patient, frame in predictions.groupby("patient", sort=True):
        reference = frame[
            frame["frequency_band"].astype(str) == str(bands[0])
        ].sort_values("bipolar_channel")
        graph_lookup[str(patient)] = build_mutual_knn_graph(
            reference["bipolar_channel"].astype(str),
            reference[["scs_x_mm", "scs_y_mm", "scs_z_mm"]].to_numpy(),
            n_neighbors=int(graph_config["n_neighbors"]),
            max_distance_mm=float(graph_config["max_distance_mm"]),
            gaussian_sigma_mm=float(graph_config["gaussian_sigma_mm"]),
        )
    n_spatial = (
        args.n_spatial_permutations
        if args.n_spatial_permutations is not None
        else int(stats_config["spatial_label_permutations"])
    )
    spatial_permutation, spatial_summary = spatial_label_permutation(
        predictions,
        graph_lookup,
        n_iterations=n_spatial,
        seed=int(stats_config["random_seed"]),
    )

    full_run = args.patients is None and args.bands is None
    if full_run:
        if len(smoothed) != int(expected["prediction_rows"]):
            raise SystemExit(
                f"Expected {expected['prediction_rows']} predictions, got {len(smoothed)}"
            )
        if smoothed["patient"].nunique() != 8:
            raise SystemExit("Expected eight patients")
        if smoothed["bipolar_channel"].groupby(smoothed["patient"]).nunique().sum() != int(
            expected["primary_modeling_contacts"]
        ):
            raise SystemExit("Unexpected unique primary contact count")
        if int(graph_audit["n_edges"].sum()) != int(expected["graph_edges"]):
            raise SystemExit("Unexpected graph edge count")
        if int(graph_audit["n_isolated"].sum()) != int(expected["isolated_nodes"]):
            raise SystemExit("Unexpected isolated-node count")
    mean_original_auprc = float(band_metrics["macro_original_AUPRC"].mean())
    mean_smoothed_auprc = float(band_metrics["macro_smoothed_AUPRC"].mean())
    mean_original_auroc = float(band_metrics["macro_original_AUROC"].mean())
    mean_smoothed_auroc = float(band_metrics["macro_smoothed_AUROC"].mean())
    mean_original_f1 = float(band_metrics["macro_original_F1"].mean())
    mean_smoothed_f1 = float(band_metrics["macro_smoothed_F1"].mean())
    print(
        f"prediction_rows={len(smoothed)} bands={smoothed['frequency_band'].nunique()} "
        f"patients={smoothed['patient'].nunique()}"
    )
    print(
        f"mean_band_macro_AUPRC={mean_original_auprc:.6f}->{mean_smoothed_auprc:.6f} "
        f"mean_band_macro_AUROC={mean_original_auroc:.6f}->{mean_smoothed_auroc:.6f}"
    )
    print(
        f"edges={int(graph_audit['n_edges'].sum())} "
        f"isolated={int(graph_audit['n_isolated'].sum())}"
    )
    if not args.apply:
        print("dry_run=true; no files written")
        return

    output_root = args.out or Path(model_config["output_root"])
    assert_allowed_output_root(output_root)
    outputs = {
        "graph_smoothed_predictions.csv": smoothed,
        "graph_smoothing_metrics_by_band.csv": band_metrics,
        "graph_smoothing_patient_band_metrics.csv": patient_metrics,
        "graph_structure_audit.csv": graph_audit,
        "paired_bootstrap_results.csv": bootstrap,
        "paired_bootstrap_ci.csv": bootstrap_ci,
        "spatial_label_permutation.csv": spatial_permutation,
    }
    output_paths = [output_root / name for name in outputs]
    output_paths.append(output_root / "graph_smoothing_run_summary.json")
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "Outputs exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing[:5])
        )
    output_root.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        atomic_write_csv(frame, output_root / name, overwrite=args.overwrite)
    summary = {
        "analysis_scope": "posthoc_exploratory",
        "primary_interpretation": "continuous_ranking",
        "hard_classification_rule": (
            "preserve_original_positive_count_within_patient_band"
        ),
        "coordinate_manifest": str(coordinate_path),
        "coordinate_manifest_sha256": file_sha256(coordinate_path),
        "source_predictions": str(prediction_path),
        "source_predictions_sha256": file_sha256(prediction_path),
        "n_prediction_rows": len(smoothed),
        "n_patients": smoothed["patient"].nunique(),
        "frequency_bands": list(dict.fromkeys(smoothed["frequency_band"].astype(str))),
        "graph": graph_config,
        "smoothing": smoothing_config,
        "mean_band_macro_original_AUPRC": mean_original_auprc,
        "mean_band_macro_smoothed_AUPRC": mean_smoothed_auprc,
        "mean_band_macro_original_AUROC": mean_original_auroc,
        "mean_band_macro_smoothed_AUROC": mean_smoothed_auroc,
        "mean_band_macro_original_F1": mean_original_f1,
        "mean_band_macro_smoothed_F1": mean_smoothed_f1,
        "n_bootstrap": n_bootstrap,
        "n_spatial_label_permutations": n_spatial,
        **spatial_summary,
    }
    atomic_write_json(
        summary,
        output_root / "graph_smoothing_run_summary.json",
        overwrite=args.overwrite,
    )
    print(f"wrote={output_root}")


if __name__ == "__main__":
    main()
