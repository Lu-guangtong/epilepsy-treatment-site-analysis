"""Shared numerical utilities and output-path protection."""

from __future__ import annotations


import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable, Iterator
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


SOURCE_ROOT = (
    PACKAGE_ROOT  # Immutable included implementation, not a host-specific tree.
)


_WRITE_POLICY: ContextVar[dict[str, Any] | None] = ContextVar(
    "pair_analysis_write_policy", default=None
)


def _under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


@contextmanager
def write_policy(
    run_root: Path,
    *,
    allowed_roots: Iterable[Path] | None = None,
    protected_roots: Iterable[Path] = (),
) -> Iterator[None]:
    """Restrict atomic writers to a run directory while protecting source inputs."""

    root = Path(run_root).resolve()
    allowed = [Path(item).resolve() for item in (allowed_roots or (root,))]
    if not any(_under(root, item) or _under(item, root) for item in allowed):
        allowed.append(root)
    protected = [SOURCE_ROOT]
    protected.extend(Path(item).resolve() for item in protected_roots)
    token = _WRITE_POLICY.set(
        {
            "run_root": root,
            "allowed_roots": tuple(allowed),
            "protected_roots": tuple(protected),
        }
    )
    try:
        yield
    finally:
        _WRITE_POLICY.reset(token)


def _assert_writable_path(path: Path) -> None:
    """Check the source-tree boundary and active run write policy."""

    target = Path(path).resolve()
    if _under(target, SOURCE_ROOT):
        raise ValueError(f"Refusing to write under protected formal root: {target}")
    policy = _WRITE_POLICY.get()
    if policy is None:
        return
    for protected in policy.get("protected_roots", ()):  # type: ignore[union-attr]
        if _under(target, Path(protected)):
            raise ValueError(f"Refusing to write under protected input root: {target}")
    allowed = tuple(Path(item) for item in policy.get("allowed_roots", ()))  # type: ignore[union-attr]
    if not any(_under(target, root) for root in allowed):
        raise ValueError(
            f"Refusing write outside active run/stage roots: {target}; "
            f"allowed={[str(item) for item in allowed]}"
        )


def json_dump_atomic(path: Path, value: Any) -> None:
    path = Path(path)
    _assert_writable_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    # ``os.replace`` is atomic on the same filesystem.  Flush the temporary
    # file first so a power loss cannot leave a manifest-looking but truncated
    # JSON file that a later resume would accept.
    with tmp.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def as_path(value: str | os.PathLike[str]) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def assert_safe_output_root(
    output_root: Path, input_roots: Iterable[Path] = ()
) -> Path:
    """Create an output directory separated from source and input trees."""
    output_root = as_path(output_root)
    forbidden = [
        as_path(r) for r in input_roots if r not in (None, "") and str(r).strip()
    ]
    formal = SOURCE_ROOT
    if is_relative_to(output_root, formal) or is_relative_to(formal, output_root):
        raise ValueError(f"Refusing to write under formal source tree: {output_root}")
    for root in forbidden:
        if output_root == root or is_relative_to(output_root, root):
            raise ValueError(
                f"Refusing to write below input root: {output_root} (input={root})"
            )
        if is_relative_to(root, output_root):
            raise ValueError(
                f"Refusing output root that contains an input root: {output_root} (input={root})"
            )
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def trapezoid(y: Any, x: Any = None, *, axis: int = -1) -> np.ndarray:
    """Integrate along the requested axis using the available NumPy trapezoid API."""

    integrator = getattr(np, "trapezoid", None)
    if integrator is None:  # pragma: no cover - exercised on NumPy < 2.0
        integrator = np.trapz
    return np.asarray(integrator(y, x=x, axis=axis))


def require_columns(
    frame: pd.DataFrame, columns: Iterable[str], label: str = "table"
) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def safe_average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def safe_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))
