"""Tests for the drift-detection monitoring layer."""

from __future__ import annotations

from pathlib import Path

from src.config import (
    FEATURE_COLUMNS,
    Config,
    MonitoringConfig,
    PreprocessConfig,
    ServingConfig,
    TrainConfig,
)
from src.monitoring.detect_drift import (
    generate_drift_report,
    make_retrain_decision,
    simulate_production_traffic,
)


def _cfg(threshold: float = 0.30) -> Config:
    return Config(
        preprocess=PreprocessConfig(0.15, 0.15, 42),
        train=TrainConfig(50, 3, 0.1, 100, 1.0, 1.0, 42),
        monitoring=MonitoringConfig(
            drift_threshold=threshold, performance_threshold=0.82
        ),
        serving=ServingConfig(
            "0.0.0.0", 8000, "http://localhost:5000", "m", "Production"
        ),
    )


def test_simulate_traffic_shape(processed_data: dict[str, Path]) -> None:
    """Simulated traffic has the feature columns and requested row count."""
    traffic = simulate_production_traffic(processed_data["test"], n_samples=50)
    assert list(traffic.columns) == list(FEATURE_COLUMNS)
    assert len(traffic) == 50


def test_simulate_drift_shifts_distribution(processed_data: dict[str, Path]) -> None:
    """Drifted traffic shifts the V1 mean relative to non-drifted traffic."""
    base = simulate_production_traffic(
        processed_data["test"], n_samples=100, drift=False
    )
    drifted = simulate_production_traffic(
        processed_data["test"], n_samples=100, drift=True
    )
    assert drifted["V1"].mean() > base["V1"].mean() + 2.0


def test_make_retrain_decision() -> None:
    """The decision compares drift share against the configured threshold."""
    assert make_retrain_decision({"drift_share": 0.45}, _cfg(0.30)) is True
    assert make_retrain_decision({"drift_share": 0.10}, _cfg(0.30)) is False


def test_generate_drift_report_detects_injected_drift(
    processed_data: dict[str, Path], tmp_path: Path
) -> None:
    """End-to-end Evidently report flags drift and writes an HTML file."""
    drifted = simulate_production_traffic(
        processed_data["test"], n_samples=200, drift=True
    )
    current_path = tmp_path / "current.parquet"
    drifted.to_parquet(current_path, index=False)

    result = generate_drift_report(processed_data["reference"], current_path, tmp_path)
    assert Path(result["report_path"]).exists()
    assert 0.0 <= result["drift_share"] <= 1.0
    assert "Amount" in result["drifted_columns"] or "V1" in result["drifted_columns"]
