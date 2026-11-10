"""Resolve workspace paths and load private cohort settings."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from functools import lru_cache
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def source_root(name: str) -> Path:
    """Require a source root; never silently guess a private data location."""
    variable = f"SEEG_{name.upper()}_ROOT"
    value = os.environ.get(variable)
    if not value:
        raise RuntimeError(f"Set {variable} to an authorized input directory before running this pipeline.")
    return Path(value).expanduser().resolve()


def assert_output_path(path: str | Path) -> Path:
    """Prevent output from entering explicitly configured read-only inputs."""
    resolved = Path(path).expanduser().resolve()
    work = REPO_ROOT / "work"
    if resolved == REPO_ROOT or resolved in REPO_ROOT.parents:
        raise RuntimeError("Output cannot be the release repository or one of its ancestors.")
    if REPO_ROOT in resolved.parents and resolved != work and work not in resolved.parents:
        raise RuntimeError("Outputs inside the release repository must stay under its dedicated work directory.")
    private_config_root = config_root()
    if resolved == private_config_root or private_config_root in resolved.parents or resolved in private_config_root.parents:
        raise RuntimeError("Output overlaps the configuration directory.")
    for name in ("DATA", "AUX", "BRAINSTORM", "EXTERNAL_REGION", "EXTERNAL_PREPARATION"):
        value = os.environ.get(f"SEEG_{name}_ROOT")
        if not value:
            continue
        source = Path(value).expanduser().resolve()
        if resolved == source or source in resolved.parents or resolved in source.parents:
            raise RuntimeError(f"Output overlaps the read-only SEEG_{name}_ROOT input directory: {resolved}")
    return resolved


def output_root() -> Path:
    """Use a separate work directory, never the original analysis tree."""
    return assert_output_path(os.environ.get("SEEG_OUTPUT_ROOT", str(REPO_ROOT / "work")))


def config_root() -> Path:
    """Allow private runtime YAMLs without changing distributed examples."""
    return Path(os.environ.get("SEEG_CONFIG_ROOT", str(REPO_ROOT / "configs"))).expanduser().resolve()


@lru_cache(maxsize=1)
def _private_config() -> dict[str, Any]:
    value = os.environ.get("SEEG_PRIVATE_CONFIG")
    if not value:
        raise RuntimeError("SEEG_PRIVATE_CONFIG is required. Supply your authorized cohort JSON outside the public release; examples are never loaded automatically.")
    path = Path(value).expanduser()
    with path.open(encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("SEEG_PRIVATE_CONFIG must contain a JSON object.")
    if config.get("release_example_only") and os.environ.get("SEEG_ALLOW_EXAMPLE_CONFIG") != "1":
        raise RuntimeError("Synthetic example configuration is disabled. Set SEEG_ALLOW_EXAMPLE_CONFIG=1 only for inspection or synthetic tests.")
    return config


def private_value(section: str, key: str) -> Any:
    """Retrieve an explicitly configured subject rule, with no fallback."""
    try:
        return _private_config()[section][key]
    except KeyError as error:
        raise ValueError(f"Missing required private configuration field: {section}.{key}") from error


def expand_config(value: Any) -> Any:
    """Expand required environment placeholders after the caller parses YAML.

    Keeping parsing in the caller preserves duplicate-key checks in strict loaders.
    """
    if isinstance(value, dict):
        if value.get("release_example_only") and os.environ.get("SEEG_ALLOW_EXAMPLE_CONFIG") != "1":
            raise RuntimeError("This public YAML is a synthetic configuration example. Supply an authorized private config, or explicitly set SEEG_ALLOW_EXAMPLE_CONFIG=1 for synthetic inspection.")
        return {key: expand_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(expand_config(item) for item in value)
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == "SEEG_REPO_ROOT":
            return str(REPO_ROOT)
        if name == "SEEG_OUTPUT_ROOT":
            return str(output_root())
        if name == "SEEG_CONFIG_ROOT":
            return str(config_root())
        configured = os.environ.get(name)
        if not configured:
            raise RuntimeError(f"Required environment variable is not set: {name}")
        return configured

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", replace, value)
