"""Mutual nearest-neighbour graphs and restart-based score diffusion."""

from __future__ import annotations


import math
from typing import Any, Iterable, Mapping, Sequence
import numpy as np
import pandas as pd
from .utils import require_columns


def _pairwise_distances(xyz: np.ndarray) -> np.ndarray:
    delta = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt(np.sum(delta * delta, axis=2))


def build_mutual_3nn_graph(
    points: pd.DataFrame,
    *,
    n_neighbors: int = 3,
    radius_mm: float | None = 10.0,
    sigma_mm: float = 5.0,
    subject_col: str = "subject",
    node_id_col: str = "bipolar_pair",
    x_cols: Sequence[str] = ("x_graph", "y_graph", "z_graph"),
) -> pd.DataFrame:
    """Construct deterministic per-subject mutual-kNN Gaussian edges."""

    require_columns(points, [subject_col, node_id_col, *x_cols], "points")
    edges: list[dict[str, Any]] = []
    sigma = float(sigma_mm)
    if sigma <= 0:
        raise ValueError("sigma_mm must be positive")
    for subject, group in points.groupby(subject_col, sort=True):
        work = group.reset_index(drop=True)
        xyz = (
            work.loc[:, list(x_cols)]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )
        valid = np.isfinite(xyz).all(axis=1)
        valid_idx = np.flatnonzero(valid)
        if len(valid_idx) < 2:
            continue
        dist = _pairwise_distances(xyz[valid_idx])
        k = min(max(int(n_neighbors), 1), len(valid_idx) - 1)
        neighbours: list[set[int]] = []
        for i in range(len(valid_idx)):
            candidates = [j for j in range(len(valid_idx)) if j != i]
            candidates.sort(
                key=lambda j: (
                    float(dist[i, j]),
                    str(work.iloc[valid_idx[j]][node_id_col]),
                )
            )
            neighbours.append(set(candidates[:k]))
        for i in range(len(valid_idx)):
            for j in sorted(neighbours[i]):
                if i not in neighbours[j] or j <= i:
                    continue
                d = float(dist[i, j])
                if radius_mm is not None and d > float(radius_mm):
                    continue
                weight = float(math.exp(-(d * d) / (2.0 * sigma * sigma)))
                node_i = str(work.iloc[valid_idx[i]][node_id_col])
                node_j = str(work.iloc[valid_idx[j]][node_id_col])
                edges.append(
                    {
                        "subject": str(subject),
                        "node_i": node_i,
                        "node_j": node_j,
                        "distance_mm": d,
                        "weight": weight,
                        "adjacency": "mutual_3nn",
                        "radius_mm": float(radius_mm)
                        if radius_mm is not None
                        else np.nan,
                        "sigma_mm": sigma,
                    }
                )
    return pd.DataFrame.from_records(
        edges,
        columns=[
            "subject",
            "node_i",
            "node_j",
            "distance_mm",
            "weight",
            "adjacency",
            "radius_mm",
            "sigma_mm",
        ],
    )


def adjacency_matrix(
    n_nodes: int,
    edges: pd.DataFrame | Iterable[Mapping[str, Any]],
    *,
    node_order: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Return a symmetric weighted adjacency matrix and node order."""

    if isinstance(edges, pd.DataFrame):
        records = edges.to_dict("records")
    else:
        records = list(edges)
    if node_order is None:
        names = sorted(
            {str(r.get("node_i")) for r in records}
            | {str(r.get("node_j")) for r in records}
        )
        if n_nodes and len(names) < n_nodes:
            names.extend(f"node-{i}" for i in range(len(names), n_nodes))
    else:
        names = [str(v) for v in node_order]
    index = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=float)
    for record in records:
        left, right = str(record.get("node_i")), str(record.get("node_j"))
        if left not in index or right not in index:
            continue
        try:
            weight = float(record.get("weight", 1.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(weight) or weight < 0:
            continue
        matrix[index[left], index[right]] = max(
            matrix[index[left], index[right]], weight
        )
        matrix[index[right], index[left]] = matrix[index[left], index[right]]
    return matrix, names


def personalized_pagerank(
    scores: Sequence[float],
    weights: np.ndarray,
    *,
    alpha: float = 0.8,
    isolated_policy: str = "keep_original_score",
    n_iter: int = 200,
    tol: float = 1e-10,
) -> np.ndarray:
    """Diffuse probability scores with the formal personalized-PageRank rule.

    The formal analysis treats ``scores`` as the original probability vector
    ``y`` and uses a row-stochastic transition matrix ``P``.  With
    ``alpha=0.8`` the converged score is therefore the solution of

    ``z = (1 - alpha) * y + alpha * P @ z``.

    This is deliberately solved as a linear system, rather than by the older
    rank/min--max normalization plus transposed power iteration.  The latter
    changed both the score domain and the direction of propagation whenever
    row-normalization made ``P`` asymmetric.  Isolated nodes receive a
    self-loop under the default ``keep_original_score`` policy, which makes
    the formal equation reduce exactly to ``z_i = y_i`` for those nodes.

    ``n_iter`` and ``tol`` remain accepted for API compatibility with earlier
    exploratory callers; the closed-form solve is deterministic and does not
    require either parameter.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    matrix = np.asarray(weights, dtype=float)
    if matrix.shape != (len(values), len(values)):
        raise ValueError("weights must be square and match scores")
    alpha_value = float(alpha)
    if not (0.0 <= alpha_value < 1.0):
        raise ValueError("alpha must be in [0, 1)")
    if len(values) == 0:
        return np.array([], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("PPR input scores must be finite probabilities in [0, 1]")
    if not np.isfinite(matrix).all() or np.any(matrix < 0.0):
        raise ValueError("PPR weights must be finite and non-negative")

    # The graph contract has no self-edges.  Rebuild the row-stochastic
    # transition matrix from weights so callers cannot accidentally pass an
    # unnormalised adjacency matrix.  A zero-degree row is represented by a
    # self-loop, as in the formal graph implementation.
    matrix = matrix.copy()
    np.fill_diagonal(matrix, 0.0)
    degree = matrix.sum(axis=1)
    transition = np.zeros_like(matrix)
    connected = degree > 0.0
    if connected.any():
        transition[connected] = matrix[connected] / degree[connected, None]
    isolated = ~connected
    if isolated_policy == "keep_original_score":
        transition[isolated, isolated] = 1.0
    elif isolated_policy not in {"drop", "zero"}:
        raise ValueError(
            "isolated_policy must be 'keep_original_score', 'drop', or 'zero'"
        )

    # For the primary policy P is row stochastic and I-alpha*P is nonsingular
    # for alpha<1.  ``drop``/``zero`` retain the legacy option of leaving a
    # zero row, yielding a substochastic transition and a damped score for an
    # isolated node; they are not the formal primary analysis.
    system = np.eye(len(values), dtype=float) - alpha_value * transition
    rhs = (1.0 - alpha_value) * values
    try:
        out = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError("PPR linear system is singular") from exc
    if not np.isfinite(out).all():
        raise ValueError("PPR produced non-finite scores")
    if np.any((out < -1e-10) | (out > 1.0 + 1e-10)):
        raise ValueError("PPR produced scores outside probability bounds")
    out = np.clip(out, 0.0, 1.0)
    if isolated_policy == "keep_original_score" and isolated.any():
        # Avoid exposing solver round-off for the exact self-loop identity.
        out[isolated] = values[isolated]
    return out
