"""Tests for the typed configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.config import Config, load_config


def test_load_config_from_real_params() -> None:
    """The shipped params.yaml parses into a fully typed Config."""
    cfg = load_config()
    assert isinstance(cfg, Config)
    assert cfg.train.scale_pos_weight == 24
    assert cfg.train.min_recall == pytest.approx(0.85)
    assert cfg.preprocess.random_seed == 42
    assert cfg.monitoring.drift_threshold == pytest.approx(0.30)
    assert cfg.serving.model_name == "fraud-detector"


def test_load_config_missing_file(tmp_path: Path) -> None:
    """A missing parameter file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_missing_section(tmp_path: Path) -> None:
    """A YAML file missing a required section raises KeyError."""
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({"preprocess": {}}), encoding="utf-8")
    with pytest.raises(KeyError):
        load_config(path)


def test_config_ignores_unexpected_keys(tmp_path: Path) -> None:
    """Unknown keys in a section are ignored rather than raising TypeError."""
    raw = {
        "preprocess": {"test_size": 0.1, "val_size": 0.1, "random_seed": 1, "extra": 9},
        "train": {
            "n_estimators": 10,
            "max_depth": 3,
            "learning_rate": 0.1,
            "scale_pos_weight": 100,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "random_seed": 1,
        },
        "monitoring": {"drift_threshold": 0.3, "performance_threshold": 0.8},
        "serving": {
            "host": "0.0.0.0",
            "port": 8000,
            "mlflow_tracking_uri": "http://localhost:5000",
            "model_name": "m",
            "model_stage": "Production",
        },
    }
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = load_config(path)
    assert cfg.preprocess.test_size == pytest.approx(0.1)
