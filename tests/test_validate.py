"""Tests for the Pandera-based data validation stage."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.validate import run_validation, validate_frame


def test_valid_data_passes(raw_df: pd.DataFrame) -> None:
    """Well-formed synthetic data passes validation."""
    report = validate_frame(raw_df)
    assert report["passed"] is True
    assert report["errors"] == []
    assert report["n_rows"] == len(raw_df)


def test_missing_class_column_fails(raw_df: pd.DataFrame) -> None:
    """Dropping the target column fails validation (strict schema)."""
    broken = raw_df.drop(columns=["Class"])
    report = validate_frame(broken)
    assert report["passed"] is False
    assert report["errors"]


def test_negative_amount_fails(raw_df: pd.DataFrame) -> None:
    """A negative Amount violates the ge=0 field check."""
    broken = raw_df.copy()
    broken.loc[broken.index[0], "Amount"] = -1.0
    report = validate_frame(broken)
    assert report["passed"] is False


def test_implausible_fraud_rate_fails(raw_df: pd.DataFrame) -> None:
    """An all-fraud frame violates the fraud-rate dataframe check."""
    broken = raw_df.copy()
    broken["Class"] = 1
    report = validate_frame(broken)
    assert report["passed"] is False


def test_run_validation_writes_report_and_raises(
    tmp_path: Path, raw_df: pd.DataFrame
) -> None:
    """A failing dataset still persists the report before raising."""
    broken = raw_df.copy()
    broken["Class"] = 1
    csv_path = tmp_path / "creditcard.csv"
    broken.to_csv(csv_path, index=False)
    report_path = tmp_path / "report.json"

    with pytest.raises(ValueError):
        run_validation(raw_path=csv_path, report_path=report_path)

    assert report_path.exists()
    saved = json.loads(report_path.read_text())
    assert saved["passed"] is False
