"""Tests for model training, thresholding, and metric computation."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    Config,
    MonitoringConfig,
    PreprocessConfig,
    ServingConfig,
    TrainConfig,
)
from src.models import train as train_mod
from src.models.evaluate import compute_metrics
from src.models.train import (
    _cost_optimal_threshold,
    build_model,
    find_optimal_threshold,
)


def test_isotonic_calibrator_corrects_probabilities() -> None:
    """Isotonic calibration maps distorted scores toward true frequencies."""
    from src.models.predict import apply_calibrator, fit_calibrator

    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    # Inflated scores (like scale_pos_weight): high even for many negatives.
    raw = np.clip(0.5 + 0.4 * y + rng.normal(0, 0.1, size=500), 0, 1)
    cal = fit_calibrator(raw, y, "isotonic")
    out = apply_calibrator(cal, raw)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # Calibrated mean should track the true positive rate better than raw.
    assert abs(out.mean() - y.mean()) < abs(raw.mean() - y.mean())


def test_cost_threshold_responds_to_asymmetric_costs() -> None:
    """Costly misses -> lower (aggressive) threshold; costly false alarms -> higher."""
    # Overlapping scores so the optimum genuinely depends on the cost ratio.
    probs = np.array([0.2, 0.6, 0.5, 0.7])
    y = np.array([0, 0, 1, 1])
    t_catch = _cost_optimal_threshold(probs, y, cost_fn=100.0, cost_fp=1.0)
    t_careful = _cost_optimal_threshold(probs, y, cost_fn=1.0, cost_fp=100.0)
    assert t_catch < t_careful


def _tiny_config() -> Config:
    """A fast-training config; high promotion threshold avoids registry calls."""
    return Config(
        preprocess=PreprocessConfig(test_size=0.15, val_size=0.15, random_seed=42),
        train=TrainConfig(
            n_estimators=40,
            max_depth=3,
            learning_rate=0.2,
            scale_pos_weight=50,
            subsample=1.0,
            colsample_bytree=1.0,
            random_seed=42,
        ),
        monitoring=MonitoringConfig(drift_threshold=0.30, performance_threshold=0.99),
        serving=ServingConfig(
            host="0.0.0.0",
            port=8000,
            mlflow_tracking_uri="http://localhost:5000",
            model_name="fraud-detector",
            model_stage="Production",
        ),
    )


def test_build_model_carries_params() -> None:
    """The classifier is built with the configured hyperparameters."""
    model = build_model(_tiny_config().train)
    params = model.get_params()
    assert params["n_estimators"] == 40
    assert params["scale_pos_weight"] == 50


def test_compute_metrics_keys_and_ranges() -> None:
    """compute_metrics returns the full suite within valid ranges (no accuracy)."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    expected = {
        "roc_auc",
        "avg_precision",
        "f1_fraud",
        "precision_fraud",
        "recall_fraud",
        "f1_macro",
        "support_fraud",
        "threshold",
    }
    assert set(metrics) == expected
    assert "accuracy" not in metrics
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["support_fraud"] == 3


def test_find_optimal_threshold_respects_recall_floor(
    processed_data: dict[str, Path],
) -> None:
    """The tuned threshold meets the configured recall floor on validation."""
    train_df = pd.read_parquet(processed_data["train"])
    val_df = pd.read_parquet(processed_data["val"])
    model = build_model(_tiny_config().train)
    model.fit(train_df[list(FEATURE_COLUMNS)], train_df[TARGET_COLUMN])

    x_val, y_val = val_df[list(FEATURE_COLUMNS)], val_df[TARGET_COLUMN]
    min_recall = 0.85
    threshold = find_optimal_threshold(model, x_val, y_val, min_recall=min_recall)
    assert 0.0 < threshold < 1.0

    probs = model.predict_proba(x_val)[:, 1]
    # By construction the chosen operating point should clear the recall floor.
    assert compute_metrics(y_val, probs, threshold)["recall_fraud"] >= min_recall - 1e-9


def test_find_optimal_threshold_handles_zero_fraud_val(
    processed_data: dict[str, Path],
) -> None:
    """With no frauds in the validation split, fall back to 0.5 (no crash)."""
    train_df = pd.read_parquet(processed_data["train"])
    model = build_model(_tiny_config().train)
    model.fit(train_df[list(FEATURE_COLUMNS)], train_df[TARGET_COLUMN])

    x_val = train_df[list(FEATURE_COLUMNS)].head(20)
    y_val = pd.Series(np.zeros(20, dtype=int))
    assert find_optimal_threshold(model, x_val, y_val) == 0.5


def test_train_end_to_end(
    processed_data: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full training run logs to MLflow and writes threshold + metrics files."""
    # SQLite backend mirrors docker-compose and works across MLflow versions.
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    monkeypatch.setattr(train_mod, "TRAIN_PATH", processed_data["train"])
    monkeypatch.setattr(train_mod, "VAL_PATH", processed_data["val"])
    monkeypatch.setattr(train_mod, "THRESHOLD_PATH", tmp_path / "threshold.json")
    monkeypatch.setattr(
        train_mod, "TRAIN_METRICS_PATH", tmp_path / "train_metrics.json"
    )
    monkeypatch.setattr(train_mod, "RUN_INFO_PATH", tmp_path / "run_info.json")

    run_id = train_mod.train(_tiny_config())

    assert isinstance(run_id, str) and run_id
    assert (tmp_path / "threshold.json").exists()
    assert (tmp_path / "train_metrics.json").exists()
    assert (tmp_path / "run_info.json").exists()
