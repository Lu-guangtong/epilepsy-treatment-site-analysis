"""Apply spatial graph diffusion to external bipolar-pair scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pair_analysis.graph import (
    adjacency_matrix,
    build_mutual_3nn_graph,
    personalized_pagerank,
)  # noqa: E402
from pair_analysis.models import safe_average_precision, safe_auroc  # noqa: E402

PATIENTS = [
    "sub-HUP130",
    "sub-HUP134",
    "sub-HUP138",
    "sub-HUP139",
    "sub-HUP140",
    "sub-HUP146",
    "sub-HUP157",
    "sub-HUP160",
    "sub-HUP162",
    "sub-HUP163",
    "sub-HUP164",
    "sub-HUP166",
    "sub-HUP190",
]
BANDS = (
    "delta",
    "theta",
    "alpha",
    "beta",
    "low_gamma",
    "high_gamma",
    "ripple",
    "fast_ripple",
)
PRIMARY_RADIUS = 10.0
RADIUS_GRID = (5.0, 7.5, 10.0, 12.5, 15.0)
SIGMA_MM = 5.0
ALPHA = 0.80
N_BOOT = 10000
SEED = 20260907


def load_labels(path: Path) -> pd.DataFrame:
    labels = pd.read_csv(path, low_memory=False)
    labels["patient"] = labels["patient"].astype(str).str.strip()
    labels["bipolar_channel"] = labels["bipolar_channel"].astype(str).str.strip()
    labels = labels[
        labels["primary_status"].astype(str).str.lower().eq("primary")
    ].copy()
    labels["label_binary"] = pd.to_numeric(labels["label_binary"], errors="coerce")
    labels = labels[labels["label_binary"].isin([0, 1])].copy()
    required = [
        "patient",
        "bipolar_channel",
        "label_binary",
        "shaft",
        "midpoint_x_raw_units",
        "midpoint_y_raw_units",
        "midpoint_z_raw_units",
    ]
    missing = [c for c in required if c not in labels.columns]
    if missing:
        raise ValueError(f"label table missing columns: {missing}")
    return labels[required].drop_duplicates(
        ["patient", "bipolar_channel"], keep="first"
    )


def make_points(labels: pd.DataFrame) -> pd.DataFrame:
    points = labels.rename(
        columns={
            "bipolar_channel": "bipolar_pair",
            "midpoint_x_raw_units": "x_graph",
            "midpoint_y_raw_units": "y_graph",
            "midpoint_z_raw_units": "z_graph",
        }
    ).copy()
    points["subject"] = points["patient"].astype(str)
    points["bipolar_pair"] = points["bipolar_pair"].astype(str)
    return points


def annotate_edges(
    edges: pd.DataFrame, points: pd.DataFrame, radius: float
) -> pd.DataFrame:
    if edges.empty:
        return pd.DataFrame(
            columns=[
                "subject",
                "node_i",
                "node_j",
                "distance_mm",
                "weight",
                "adjacency",
                "radius_mm",
                "sigma_mm",
                "shaft_i",
                "shaft_j",
                "shaft_relation",
            ]
        )
    shafts = (
        points.set_index(["subject", "bipolar_pair"])["shaft"].astype(str).to_dict()
    )
    out = edges.copy()
    out["shaft_i"] = [
        shafts.get((str(s), str(n)), "") for s, n in zip(out.subject, out.node_i)
    ]
    out["shaft_j"] = [
        shafts.get((str(s), str(n)), "") for s, n in zip(out.subject, out.node_j)
    ]
    out["shaft_relation"] = np.where(
        out["shaft_i"].eq(out["shaft_j"]), "same_shaft", "cross_shaft"
    )
    out["radius_mm"] = float(radius)
    out["sigma_mm"] = SIGMA_MM
    return out


def build_edges(points: pd.DataFrame, radius: float) -> pd.DataFrame:
    raw = build_mutual_3nn_graph(
        points[["subject", "bipolar_pair", "x_graph", "y_graph", "z_graph"]],
        n_neighbors=3,
        radius_mm=radius,
        sigma_mm=SIGMA_MM,
        subject_col="subject",
        node_id_col="bipolar_pair",
        x_cols=("x_graph", "y_graph", "z_graph"),
    )
    return annotate_edges(raw, points, radius)


def graph_summary(
    points: pd.DataFrame, edges: pd.DataFrame, radius: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for subject, group in points.groupby("subject", sort=True):
        e = edges[edges["subject"].eq(subject)]
        all_nodes = set(group["bipolar_pair"].astype(str))
        linked = (
            set(e["node_i"].astype(str)) | set(e["node_j"].astype(str))
            if not e.empty
            else set()
        )
        rows.append(
            {
                "subject": subject,
                "radius_mm": radius,
                "n_nodes": len(all_nodes),
                "n_nodes_with_valid_coordinates": int(
                    group[["x_graph", "y_graph", "z_graph"]].notna().all(axis=1).sum()
                ),
                "n_edges": int(len(e)),
                "n_same_shaft_edges": int((e["shaft_relation"] == "same_shaft").sum())
                if not e.empty
                else 0,
                "n_cross_shaft_edges": int((e["shaft_relation"] == "cross_shaft").sum())
                if not e.empty
                else 0,
                "n_isolated_nodes": int(len(all_nodes - linked)),
            }
        )
    return pd.DataFrame(rows)


def smooth_predictions(
    pred: pd.DataFrame,
    points: pd.DataFrame,
    edges: pd.DataFrame,
    radius: float = PRIMARY_RADIUS,
) -> pd.DataFrame:
    out = pred.copy()
    out["y_score_smoothed"] = np.nan
    out["graph_radius_mm"] = float(radius)
    out["graph_sigma_mm"] = SIGMA_MM
    out["graph_alpha"] = ALPHA
    out["graph_isolated_policy"] = "keep_original_score"
    for subject, idx in out.groupby("patient", sort=True).groups.items():
        subset = out.loc[idx].copy()
        point_subset = points[points["subject"].eq(subject)].copy()
        # The prediction table and point table must have the same node keys.
        point_subset = point_subset.set_index("bipolar_pair")
        node_order = subset["bipolar_channel"].astype(str).tolist()
        edge_subset = edges[edges["subject"].eq(subject)]
        matrix, names = adjacency_matrix(
            len(node_order), edge_subset, node_order=node_order
        )
        vals = personalized_pagerank(
            subset["y_score"].to_numpy(float),
            matrix,
            alpha=ALPHA,
            isolated_policy="keep_original_score",
        )
        mapping = dict(zip(names, vals))
        out.loc[idx, "y_score_smoothed"] = (
            subset["bipolar_channel"].astype(str).map(mapping).to_numpy(float)
        )
    if out["y_score_smoothed"].isna().any():
        raise RuntimeError("Some predictions did not receive a smoothed score")
    return out


def patient_metrics(pred: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (band, patient), group in pred.groupby(
        ["frequency_band", "patient"], sort=True
    ):
        y = pd.to_numeric(group["y_true"], errors="coerce").to_numpy(float)
        s = pd.to_numeric(group[score_col], errors="coerce").to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(s)
        y = y[keep].astype(int)
        s = s[keep]
        if len(y) == 0 or len(np.unique(y)) < 2:
            continue
        ap = float(safe_average_precision(y, s))
        prev = float(y.mean())
        norm = float((ap - prev) / (1.0 - prev)) if prev < 1.0 else np.nan
        rows.append(
            {
                "frequency_band": band,
                "patient": patient,
                "score_type": score_col,
                "AUPRC": ap,
                "normalized_AUPRC": norm,
                "AUROC": float(safe_auroc(y, s)),
                "n_pairs": int(len(y)),
                "n_positive": int(y.sum()),
                "prevalence": prev,
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(patient_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for band, group in patient_table.groupby("frequency_band", sort=True):
        for score_type, g in group.groupby("score_type", sort=True):
            rows.append(
                {
                    "frequency_band": band,
                    "score_type": score_type,
                    "patient_macro_AUPRC": float(g["AUPRC"].mean()),
                    "patient_macro_normalized_AUPRC": float(
                        g["normalized_AUPRC"].mean()
                    ),
                    "patient_macro_AUROC": float(g["AUROC"].mean()),
                    "n_metric_patients": int(g["patient"].nunique()),
                    "n_pairs": int(g["n_pairs"].sum()),
                    "n_positive": int(g["n_positive"].sum()),
                }
            )
    wide = pd.DataFrame(rows)
    if wide.empty:
        return wide
    raw = (
        wide[wide.score_type.eq("y_score")]
        .drop(columns="score_type")
        .rename(
            columns={
                "patient_macro_AUPRC": "pre_AUPRC",
                "patient_macro_normalized_AUPRC": "pre_normalized_AUPRC",
                "patient_macro_AUROC": "pre_AUROC",
            }
        )
    )
    smooth = (
        wide[wide.score_type.eq("y_score_smoothed")]
        .drop(columns="score_type")
        .rename(
            columns={
                "patient_macro_AUPRC": "post_AUPRC",
                "patient_macro_normalized_AUPRC": "post_normalized_AUPRC",
                "patient_macro_AUROC": "post_AUROC",
            }
        )
    )
    merged = raw.merge(smooth, on="frequency_band", suffixes=("", "_post"))
    for metric in ("AUPRC", "normalized_AUPRC", "AUROC"):
        merged[f"delta_{metric}"] = merged[f"post_{metric}"] - merged[f"pre_{metric}"]
    return merged


def paired_bootstrap(patient_table: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for band, group in patient_table.groupby("frequency_band", sort=True):
        pre = group[group.score_type.eq("y_score")].set_index("patient")
        post = group[group.score_type.eq("y_score_smoothed")].set_index("patient")
        common = sorted(set(pre.index) & set(post.index))
        if not common:
            continue
        n = len(common)
        idx = rng.integers(0, n, size=(N_BOOT, n))
        row = {"frequency_band": band, "n_patients": n}
        for metric in ("AUPRC", "normalized_AUPRC", "AUROC"):
            diff = post.loc[common, metric].to_numpy(float) - pre.loc[
                common, metric
            ].to_numpy(float)
            draws = diff[idx].mean(axis=1)
            row[f"delta_{metric}"] = float(diff.mean())
            row[f"ci95_low_{metric}"] = float(np.quantile(draws, 0.025))
            row[f"ci95_high_{metric}"] = float(np.quantile(draws, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def rank_output(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    for col, prefix in (("y_score", "raw"), ("y_score_smoothed", "smoothed")):
        out[f"{prefix}_rank_desc"] = out.groupby(["frequency_band", "patient"])[
            col
        ].rank(method="average", ascending=False)
        out[f"{prefix}_rank_percentile"] = out.groupby(["frequency_band", "patient"])[
            col
        ].rank(method="average", ascending=True, pct=True)
    return out.sort_values(
        ["frequency_band", "patient", "smoothed_rank_desc"], kind="mergesort"
    )


def run(args: argparse.Namespace) -> None:
    pred_root = args.prediction_root.resolve()
    from pair_analysis.utils import assert_safe_output_root

    out = assert_safe_output_root(
        args.output_root, [pred_root, args.labels.resolve().parent]
    )
    labels = load_labels(args.labels.resolve())
    points = make_points(labels)
    all_edges: list[pd.DataFrame] = []
    all_graph_summary: list[pd.DataFrame] = []
    for radius in args.radii:
        edges = build_edges(points, float(radius))
        all_edges.append(edges)
        all_graph_summary.append(graph_summary(points, edges, float(radius)))
    edges_grid = (
        pd.concat(all_edges, ignore_index=True) if all_edges else pd.DataFrame()
    )
    graph_summary_grid = pd.concat(all_graph_summary, ignore_index=True)
    primary_edges = edges_grid[edges_grid.radius_mm.eq(PRIMARY_RADIUS)].copy()
    metrics_rows: list[dict[str, Any]] = []
    pred_rows: list[pd.DataFrame] = []
    patient_rows: list[pd.DataFrame] = []
    for band in args.bands:
        path = pred_root / f"predictions_{band}.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        pred = pd.read_csv(path, low_memory=False)
        pred["patient"] = pred["patient"].astype(str).str.strip()
        pred["bipolar_channel"] = pred["bipolar_channel"].astype(str).str.strip()
        pred = pred[pred["patient"].isin(args.patients)].copy()
        if set(pred.patient) != set(args.patients):
            raise ValueError(f"{band}: patient set mismatch")
        pre_patient = patient_metrics(pred, "y_score")
        for radius in args.radii:
            radius_edges = edges_grid[edges_grid.radius_mm.eq(float(radius))].copy()
            smoothed_r = smooth_predictions(
                pred, points, radius_edges, radius=float(radius)
            )
            post_patient = patient_metrics(smoothed_r, "y_score_smoothed")
            pre = pre_patient.set_index("patient")
            post = post_patient.set_index("patient")
            common = sorted(set(pre.index) & set(post.index))
            if not common:
                continue
            for metric in ("AUPRC", "normalized_AUPRC", "AUROC"):
                metrics_rows.append(
                    {
                        "frequency_band": band,
                        "radius_mm": float(radius),
                        "metric": metric,
                        "pre": float(pre.loc[common, metric].mean()),
                        "post": float(post.loc[common, metric].mean()),
                        "delta": float(
                            (post.loc[common, metric] - pre.loc[common, metric]).mean()
                        ),
                        "n_metric_patients": len(common),
                    }
                )
            if float(radius) == PRIMARY_RADIUS:
                pred_rows.append(smoothed_r)
                patient_rows += [pre_patient, post_patient]
    predictions = pd.concat(pred_rows, ignore_index=True)
    patient_table = pd.concat(patient_rows, ignore_index=True)
    summary = summarize_metrics(patient_table)
    bootstrap = paired_bootstrap(patient_table)
    ranked = rank_output(predictions)
    radius_sensitivity = pd.DataFrame(metrics_rows).sort_values(
        ["radius_mm", "frequency_band", "metric"], kind="mergesort"
    )
    edges_grid.to_csv(out / "graph_edges_all_radii.csv", index=False)
    primary_edges.to_csv(out / "graph_edges_primary_10mm.csv", index=False)
    graph_summary_grid.to_csv(
        out / "graph_structure_summary_all_radii.csv", index=False
    )
    predictions.to_csv(out / "graph_smoothed_predictions_all_bands.csv", index=False)
    ranked.to_csv(out / "graph_smoothed_rankings_all_bands.csv", index=False)
    patient_table.to_csv(out / "graph_smoothing_patient_metrics.csv", index=False)
    summary.to_csv(out / "graph_smoothing_metrics_by_band.csv", index=False)
    bootstrap.to_csv(out / "graph_smoothing_paired_bootstrap.csv", index=False)
    radius_sensitivity.to_csv(
        out / "graph_smoothing_radius_sensitivity.csv", index=False
    )
    manifest = {
        "status": "COMPLETED",
        "patients": args.patients,
        "bands": list(args.bands),
        "n_prediction_rows": int(len(predictions)),
        "n_primary_edges": int(len(primary_edges)),
        "primary_parameters": {
            "n_neighbors": 3,
            "radius_mm": PRIMARY_RADIUS,
            "sigma_mm": SIGMA_MM,
            "alpha": ALPHA,
            "isolated_policy": "keep_original_score",
        },
        "radius_sensitivity_mm": list(args.radii),
        "labels_used_for_graph": False,
        "input_prediction_root": str(pred_root),
        "labels_path": str(args.labels.resolve()),
    }
    (out / "graph_smoothing_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        summary[
            [
                "frequency_band",
                "pre_AUPRC",
                "post_AUPRC",
                "delta_AUPRC",
                "pre_AUROC",
                "post_AUROC",
                "delta_AUROC",
            ]
        ].to_string(index=False),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bands", nargs="+", default=list(BANDS), choices=list(BANDS))
    parser.add_argument("--patients", nargs="+", default=PATIENTS, choices=PATIENTS)
    parser.add_argument("--radii", nargs="+", type=float, default=list(RADIUS_GRID))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
