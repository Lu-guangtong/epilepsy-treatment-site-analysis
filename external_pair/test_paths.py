"""Synthetic checks for package imports, configuration and output paths."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_pair"))
sys.path.insert(0, str(ROOT / "scripts"))

import permutation
from pair_analysis import models, utils
import pipeline_config


def test_local_package_is_used():
    assert (
        Path(models.__file__)
        .resolve()
        .is_relative_to(ROOT / "external_pair/pair_analysis")
    )
    assert permutation.PACKAGE_ROOT == ROOT / "external_pair"


def test_output_root_rejects_input_overlap(tmp_path):
    source = tmp_path / "inputs"
    for output in (source, source / "generated", tmp_path):
        with pytest.raises(ValueError):
            utils.assert_safe_output_root(output, [source])
    with pytest.raises(ValueError):
        utils.assert_safe_output_root(utils.PACKAGE_ROOT / "generated")
    assert utils.assert_safe_output_root(tmp_path / "outputs", [source]).is_dir()


def test_active_write_policy_protects_inputs_and_sibling_runs(tmp_path):
    run = tmp_path / "run"
    source = tmp_path / "inputs"
    with utils.write_policy(run, protected_roots=[source]):
        utils.json_dump_atomic(run / "result.json", {"synthetic": True})
        for path in (source / "forbidden.json", tmp_path / "other_run/forbidden.json"):
            with pytest.raises(ValueError):
                utils.json_dump_atomic(path, {})
            assert not path.exists()


def test_private_cohort_requires_explicit_configuration(monkeypatch):
    monkeypatch.delenv("SEEG_PRIVATE_CONFIG", raising=False)
    pipeline_config._private_config.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="SEEG_PRIVATE_CONFIG"):
            pipeline_config.private_value("cohort", "patients")
        monkeypatch.setenv(
            "SEEG_PRIVATE_CONFIG", str(ROOT / "configs/cohort.example.json")
        )
        monkeypatch.delenv("SEEG_ALLOW_EXAMPLE_CONFIG", raising=False)
        with pytest.raises(RuntimeError, match="example"):
            pipeline_config.private_value("cohort", "patients")
        monkeypatch.setenv("SEEG_ALLOW_EXAMPLE_CONFIG", "1")
        assert pipeline_config.private_value("cohort", "patients") == [
            "demo001",
            "demo002",
        ]
    finally:
        pipeline_config._private_config.cache_clear()


def test_example_configs_require_opt_in(monkeypatch):
    monkeypatch.delenv("SEEG_ALLOW_EXAMPLE_CONFIG", raising=False)
    for path in (ROOT / "configs").rglob("*.yaml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        with pytest.raises(RuntimeError, match="example"):
            pipeline_config.expand_config(config)


def test_config_references_exist(tmp_path, monkeypatch):
    monkeypatch.setenv("SEEG_ALLOW_EXAMPLE_CONFIG", "1")
    monkeypatch.setenv("SEEG_CONFIG_ROOT", str(ROOT / "configs"))
    monkeypatch.setenv("SEEG_OUTPUT_ROOT", str(tmp_path / "work"))
    for name in (
        "DATA",
        "AUX",
        "BRAINSTORM",
        "EXTERNAL_REGION",
        "EXTERNAL_PREPARATION",
    ):
        monkeypatch.setenv(f"SEEG_{name}_ROOT", str(tmp_path / name.lower()))
    for path in (ROOT / "configs").rglob("*.yaml"):
        config = pipeline_config.expand_config(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        values = [config]
        while values:
            value = values.pop()
            if isinstance(value, dict):
                values.extend(value.values())
            elif isinstance(value, list):
                values.extend(value)
            elif isinstance(value, str) and value.endswith(".yaml"):
                assert Path(value).is_file(), (path, value)
