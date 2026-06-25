"""Tests for evaluation, benchmark gating, and the val/holdout pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from src.config import (
    Config,
    MonitoringConfig,
    PreprocessConfig,
    ServingConfig,
    TrainConfig,
)
from src.models import evaluate as eval_mod
from src.models import train as train_mod
from src.models.evaluate import BENCHMARK_TARGETS, check_benchmarks, load_threshold


def _tiny_config() -> Config:
    return Config(
        preprocess=PreprocessConfig(0.15, 0.15, 42),
        train=TrainConfig(40, 3, 0.2, 50, 1.0, 1.0, 42),
        monitoring=MonitoringConfig(drift_threshold=0.30, performance_threshold=0.99),
        serving=ServingConfig(
            "0.0.0.0", 8000, "http://localhost:5000", "m", "Production"
        ),
    )


def test_check_benchmarks_all_pass() -> None:
    """No failures are returned when every metric clears its target."""
    metrics = {
        "roc_auc": 0.99,
        "avg_precision": 0.90,
        "recall_fraud": 0.85,
        "precision_fraud": 0.90,
    }
    assert check_benchmarks(metrics) == []


def test_check_benchmarks_reports_each_failure() -> None:
    """Every below-target metric is named in the failure list."""
    metrics = {
        "roc_auc": 0.5,
        "avg_precision": 0.1,
        "recall_fraud": 0.1,
        "precision_fraud": 0.1,
    }
    failures = check_benchmarks(metrics)
    assert len(failures) == len(BENCHMARK_TARGETS)
    assert any("recall_fraud" in f for f in failures)


def test_load_threshold(tmp_path: Path) -> None:
    """The threshold is read back from threshold.json."""
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": 0.37}), encoding="utf-8")
    assert load_threshold(path) == pytest.approx(0.37)


def test_evaluate_val_and_holdout(
    processed_data: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Train a tiny model, then run val + holdout evaluation against MLflow."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    threshold_path = tmp_path / "threshold.json"
    run_info_path = tmp_path / "run_info.json"

    # Point train's outputs into tmp.
    monkeypatch.setattr(train_mod, "TRAIN_PATH", processed_data["train"])
    monkeypatch.setattr(train_mod, "VAL_PATH", processed_data["val"])
    monkeypatch.setattr(train_mod, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(train_mod, "RUN_INFO_PATH", run_info_path)
    monkeypatch.setattr(
        train_mod, "TRAIN_METRICS_PATH", tmp_path / "train_metrics.json"
    )
    train_mod.train(_tiny_config())

    # Point evaluate at the same data + artifacts, and relax the gate so the
    # synthetic challenger passes the benchmark check deterministically.
    monkeypatch.setattr(eval_mod, "VAL_PATH", processed_data["val"])
    monkeypatch.setattr(eval_mod, "TEST_PATH", processed_data["test"])
    monkeypatch.setattr(eval_mod, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(eval_mod, "RUN_INFO_PATH", run_info_path)
    monkeypatch.setattr(eval_mod, "TRAIN_METRICS_PATH", tmp_path / "tm.json")
    monkeypatch.setattr(eval_mod, "EVAL_METRICS_PATH", tmp_path / "em.json")
    monkeypatch.setattr(
        eval_mod,
        "BENCHMARK_TARGETS",
        {"roc_auc": 0.0, "f1_fraud": 0.0, "recall_fraud": 0.0, "precision_fraud": 0.0},
    )

    val_metrics = eval_mod.evaluate("val")
    assert (tmp_path / "tm.json").exists()
    assert set(val_metrics) >= {"roc_auc", "f1_fraud"}

    holdout_metrics = eval_mod.evaluate("holdout")
    assert (tmp_path / "em.json").exists()
    assert 0.0 <= holdout_metrics["roc_auc"] <= 1.0


def test_evaluate_holdout_fails_benchmark_gate(
    processed_data: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The holdout stage exits non-zero when a benchmark target is unmet."""
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    threshold_path = tmp_path / "threshold.json"
    run_info_path = tmp_path / "run_info.json"
    monkeypatch.setattr(train_mod, "TRAIN_PATH", processed_data["train"])
    monkeypatch.setattr(train_mod, "VAL_PATH", processed_data["val"])
    monkeypatch.setattr(train_mod, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(train_mod, "RUN_INFO_PATH", run_info_path)
    monkeypatch.setattr(
        train_mod, "TRAIN_METRICS_PATH", tmp_path / "train_metrics.json"
    )
    train_mod.train(_tiny_config())

    monkeypatch.setattr(eval_mod, "TEST_PATH", processed_data["test"])
    monkeypatch.setattr(eval_mod, "THRESHOLD_PATH", threshold_path)
    monkeypatch.setattr(eval_mod, "RUN_INFO_PATH", run_info_path)
    monkeypatch.setattr(eval_mod, "EVAL_METRICS_PATH", tmp_path / "em.json")
    # Impossible target -> gate must fail.
    monkeypatch.setattr(eval_mod, "BENCHMARK_TARGETS", {"roc_auc": 1.01})

    with pytest.raises(SystemExit):
        eval_mod.evaluate("holdout")
