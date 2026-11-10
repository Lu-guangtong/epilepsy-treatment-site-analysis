"""Array-based model refitting and statistical inference for bipolar-pair analyses."""

from __future__ import annotations
# ruff: noqa: E402 -- thread limits must precede NumPy/BLAS imports on spawn.

import os

for _key in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_key] = "1"

import json
import sys
import warnings
from pathlib import Path

# Disable bytecode writes to source directories.
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))
from pair_analysis.models import (  # noqa: E402
    _fit_logistic,
    safe_average_precision,
    safe_auroc,
    select_threshold_max_f1,
)
from pair_analysis.graph import personalized_pagerank  # noqa: E402

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
METRICS = ("AUPRC", "AUROC", "normalized_AUPRC")
C_VALUES = (0.01, 0.1, 1.0, 10.0)
SEED = 20261006


def shuffled_labels(y, patient_codes, permutation_id, seed=SEED):
    """One assignment per ID, independent of band, process count and ordering."""
    if permutation_id < 1:
        raise ValueError("Permutation IDs start at 1")
    rng = np.random.default_rng(np.random.SeedSequence([seed, permutation_id]))
    out = np.array(y, dtype=np.int8, copy=True)
    for patient in np.unique(patient_codes):
        idx = np.flatnonzero(patient_codes == patient)
        out[idx] = rng.permutation(out[idx])
    return out


def patient_metrics(y, score, groups):
    rows = []
    for idx in groups:
        yy, ss = y[idx], score[idx]
        if not np.isfinite(ss).all() or len(np.unique(yy)) != 2:
            raise ValueError("Incomplete/non-estimable patient predictions")
        ap = safe_average_precision(yy, ss)
        rows.append([ap, safe_auroc(yy, ss), (ap - yy.mean()) / (1 - yy.mean())])
    return np.asarray(rows, dtype=float)


class APPlan:
    """Cache sorting/tie groups, preserving the reference's sum order exactly."""

    def __init__(self, score):
        self.items = []
        for col in score.T:
            order = np.argsort(-col, kind="mergesort")
            ss = col[order]
            ends = np.flatnonzero(np.r_[ss[1:] != ss[:-1], True])
            self.items.append((order.astype(np.int32), ends, np.r_[0, ends[:-1] + 1]))

    def evaluate(self, y):
        denom = y.sum()
        if denom == 0 or denom == len(y):
            return np.full(len(self.items), np.nan)
        return np.asarray(
            [
                np.sum(
                    np.add.reduceat(y[o], starts)
                    * (np.cumsum(y[o])[ends] / (ends + 1.0))
                )
                / denom
                for o, ends, starts in self.items
            ]
        )


def reference_correlations(values):
    """Fast matrix with scalar-reference correction at the pruning boundary.

    DataFrame.corr and Series.corr can round exact rho=0.9 differently
    (0.9 versus 0.8999999999999998). Only the threshold decision matters;
    resolve every near-boundary pair using the actual reference operation.
    """
    table = pd.DataFrame(values)
    corr = table.corr(method="spearman").to_numpy(copy=True)
    near = np.triu(np.isfinite(corr) & (np.abs(np.abs(corr) - 0.9) < 1e-12), 1)
    for a, b in zip(*np.nonzero(near)):
        corr[a, b] = corr[b, a] = table[a].corr(table[b], method="spearman")
    return corr


class BandEngine:
    def __init__(self, x, patient_codes, names, weights=None):
        self.x = np.asarray(x, dtype=float)
        self.codes = np.asarray(patient_codes)
        self.names = list(names)
        self.groups = [np.flatnonzero(self.codes == p) for p in np.unique(self.codes)]
        self.weights = weights
        self.subsets = {}
        self.ap_plans = {}
        self.outer = {}
        self._prepare()

    def _subset(self, excluded):
        key = tuple(sorted(excluded))
        if key not in self.subsets:
            idx = np.flatnonzero(~np.isin(self.codes, key))
            values = self.x[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                fill = np.nanmedian(values, axis=0)
            self.subsets[key] = (idx, fill)
        return self.subsets[key]

    def _prepare(self):
        for outer in range(len(self.groups)):
            train, _ = self._subset((outer,))
            xx = self.x[train]
            finite = np.isfinite(xx)
            mins = np.min(np.where(finite, xx, np.inf), axis=0)
            maxs = np.max(np.where(finite, xx, -np.inf), axis=0)
            eligible = np.flatnonzero((finite.mean(axis=0) >= 0.7) & (mins != maxs))
            # pandas Spearman uses pairwise-complete observations, exactly as
            # the reference's Series.corr. Do not replace with imputed ranks.
            corr = reference_correlations(xx)
            self.outer[outer] = (train, eligible, corr)
            for inner in range(len(self.groups)):
                if inner == outer:
                    continue
                _, fill = self._subset((outer, inner))
                score = self.x[self.groups[inner]][:, eligible]
                score = np.where(np.isfinite(score), score, fill[eligible])
                self.ap_plans[outer, inner] = (APPlan(score), APPlan(-score))

    def _inner_directions(self, y, cache, key):
        key = tuple(sorted(key))
        if key not in cache:
            idx, _ = self._subset(key)
            xx, yy = self.x[idx], y[idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                pos = np.nanmedian(xx[yy == 1], axis=0)
                neg = np.nanmedian(xx[yy == 0], axis=0)
            cache[key] = (
                np.where(pos >= neg, 1.0, -1.0),
                np.isfinite(pos) & np.isfinite(neg),
            )
        return cache[key]

    def select(self, y, outer, directions_cache):
        train, eligible, corr = self.outer[outer]
        values = []
        for inner in range(len(self.groups)):
            if inner == outer:
                continue
            direction, valid = self._inner_directions(
                y, directions_cache, (outer, inner)
            )
            plus, minus = self.ap_plans[outer, inner]
            yy = y[self.groups[inner]]
            vals = np.where(
                direction[eligible] > 0, plus.evaluate(yy), minus.evaluate(yy)
            )
            _, fill = self._subset((outer, inner))
            vals[~valid[eligible] | ~np.isfinite(fill[eligible])] = np.nan
            values.append(vals)
        values = np.array(values)
        # Match np.mean(list_of_valid_scores) in the reference feature loop.
        scores = np.array(
            [
                np.mean(v[np.isfinite(v)]) if np.isfinite(v).any() else -np.inf
                for v in values.T
            ]
        )
        order = sorted(
            range(len(eligible)), key=lambda j: (-scores[j], self.names[eligible[j]])
        )
        selected = []
        for j in order:
            col = int(eligible[j])
            if all(
                not np.isfinite(corr[col, k]) or abs(corr[col, k]) < 0.9
                for k in selected
            ):
                selected.append(col)
            if len(selected) == 5:
                break
        # Backfill eligible ranked candidates when correlation pruning reduces the subset.
        for j in order:
            if len(selected) >= min(5, len(eligible)):
                break
            col = int(eligible[j])
            if col not in selected:
                selected.append(col)
        if not selected:
            raise ValueError("No eligible features; never silently drop an outer fold")
        return selected, {self.names[c]: float(v) for c, v in zip(eligible, scores)}

    @staticmethod
    def fit_directions(x, y):
        # _learn_directions excludes infinities, unlike inner feature scoring.
        finite_x = np.where(np.isfinite(x), x, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pos = np.nanmedian(finite_x[y == 1], axis=0)
            neg = np.nanmedian(finite_x[y == 0], axis=0)
        return np.where(np.isfinite(pos) & np.isfinite(neg) & (pos < neg), -1.0, 1.0)

    def run(self, y, audit=False):
        y = np.asarray(y, dtype=np.int8)
        prediction = np.full(len(y), np.nan)
        thresholds = np.full(len(y), np.nan)
        folds = []
        direction_cache = {}
        with threadpool_limits(limits=1), warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            for outer, test in enumerate(self.groups):
                train, _, _ = self.outer[outer]
                selected, feature_scores = self.select(y, outer, direction_cache)
                c_scores, c_predictions = [], []
                # Label-dependent direction is constant across C, so reuse it
                # within this permutation/inner training fold only.
                inner_data = []
                for inner, inner_test in enumerate(self.groups):
                    if inner == outer:
                        continue
                    inner_train, _ = self._subset((outer, inner))
                    xx = self.x[np.ix_(inner_train, selected)]
                    xt = self.x[np.ix_(inner_test, selected)]
                    direction = self.fit_directions(xx, y[inner_train])
                    inner_data.append((inner_train, inner_test, xx, xt, direction))
                for c in C_VALUES:
                    oof_y, oof_score, aps = [], [], []
                    for itr, ite, xx, xt, direction in inner_data:
                        score, _ = _fit_logistic(
                            xx, y[itr], xt, c, directions=direction
                        )
                        aps.append(safe_average_precision(y[ite], score))
                        oof_y.extend(y[ite])
                        oof_score.extend(score)
                    c_scores.append(float(np.mean(aps)))
                    c_predictions.append((np.asarray(oof_y), np.asarray(oof_score)))
                best = max(
                    range(len(C_VALUES)), key=lambda j: (c_scores[j], -C_VALUES[j])
                )
                threshold = select_threshold_max_f1(*c_predictions[best])
                xx, xt = self.x[np.ix_(train, selected)], self.x[np.ix_(test, selected)]
                direction = self.fit_directions(xx, y[train])
                score, _ = _fit_logistic(
                    xx, y[train], xt, C_VALUES[best], directions=direction
                )
                prediction[test] = score
                thresholds[test] = threshold
                if audit:
                    folds.append(
                        {
                            "outer": outer,
                            "selected_features": [self.names[j] for j in selected],
                            "selected_C": C_VALUES[best],
                            "selected_threshold": threshold,
                            "directions": direction.tolist(),
                            "c_scores": c_scores,
                            "feature_scores": feature_scores,
                        }
                    )
        if not np.isfinite(prediction).all():
            raise ValueError("Incomplete predictions")
        raw = patient_metrics(y, prediction, self.groups)
        result = {"raw": raw.mean(axis=0).tolist()}
        smoothed = None
        if self.weights is not None:
            smoothed = prediction.copy()
            for idx, w in zip(self.groups, self.weights):
                smoothed[idx] = personalized_pagerank(prediction[idx], w, alpha=0.8)
            smooth = patient_metrics(y, smoothed, self.groups)
            result["smoothed"] = smooth.mean(axis=0).tolist()
        if audit:
            result.update(
                prediction=prediction,
                smoothed_prediction=smoothed,
                thresholds=thresholds,
                folds=folds,
                patient_raw=raw,
            )
        return result


def load_engine(cache, band):
    cache = Path(cache)
    metadata = json.loads((cache / "cache.json").read_text(encoding="utf-8"))
    weights = [
        np.load(cache / f"graph_{i}.npy", mmap_mode="r")
        for i in range(len(metadata["patients"]))
    ]
    return BandEngine(
        np.load(cache / f"features_{band}.npy", mmap_mode="r"),
        np.load(cache / "patient_codes.npy", mmap_mode="r"),
        metadata["features"][band],
        weights,
    )


def permutation_summary(observed, null):
    """null: B x bands x tracks x metrics; all cells must be complete."""
    null, observed = np.asarray(null), np.asarray(observed)
    if (
        null.ndim != 4
        or len(null) == 0
        or null.shape[1:] != observed.shape
        or not np.isfinite(null).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("Incomplete permutation family")
    n = len(null)
    raw = (1 + (null >= observed[None]).sum(axis=0)) / (n + 1)
    family_max = null.max(axis=1, keepdims=True)
    by_track = (1 + (family_max >= observed[None]).sum(axis=0)) / (n + 1)
    joint_max = null.max(axis=(1, 2), keepdims=True)
    joint = (1 + (joint_max >= observed[None]).sum(axis=0)) / (n + 1)
    return raw, by_track, joint


def exact_paired_test(differences):
    """Patient x band x metric differences; conditional sign exchangeability.

    Exhaustive enumeration includes the observed assignment: count / 2**n,
    unlike sampled label permutations, which require the +1 correction.
    """
    differences = np.asarray(differences, dtype=float)
    if not np.isfinite(differences).all():
        raise ValueError("Nonfinite paired differences")
    n = len(differences)
    if n > 20:
        raise ValueError("Exact test limited to 20 patients")
    bits = (np.arange(2**n)[:, None] >> np.arange(n)) & 1
    signs = 2.0 * bits - 1.0
    null = np.einsum("pn,nbm->pbm", signs, differences) / n
    observed = differences.mean(axis=0)
    # Protect equality at the observed assignment against summation roundoff.
    p = (null >= observed[None] - 1e-14).mean(axis=0)
    p_max = (null.max(axis=1, keepdims=True) >= observed[None] - 1e-14).mean(axis=0)
    return observed, p, p_max
