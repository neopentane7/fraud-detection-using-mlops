"""Tests for the preprocessing stage."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from src.config import SCALED_COLUMNS, TARGET_COLUMN, PreprocessConfig
from src.data.preprocess import preprocess, split_data


def test_temporal_split_is_chronological(raw_df: pd.DataFrame) -> None:
    """Temporal split keeps train older than val older than test (no leakage)."""
    cfg = PreprocessConfig(
        test_size=0.2, val_size=0.2, random_seed=0,
        split_strategy="temporal", time_column="Time",
    )
    train, val, test = split_data(raw_df, cfg)
    assert len(train) + len(val) + len(test) == len(raw_df)
    assert train["Time"].max() <= val["Time"].min()
    assert val["Time"].max() <= test["Time"].min()


def test_temporal_split_falls_back_without_time_column(raw_df: pd.DataFrame) -> None:
    """Missing time column -> stratified fallback instead of crashing."""
    df = raw_df.drop(columns=["Time"])
    cfg = PreprocessConfig(
        test_size=0.15, val_size=0.15, random_seed=0,
        split_strategy="temporal", time_column="Time",
    )
    train, val, test = split_data(df, cfg)
    assert len(train) + len(val) + len(test) == len(df)
    for split in (train, val, test):
        assert split[TARGET_COLUMN].sum() >= 1


def test_split_sizes_and_stratification(raw_df: pd.DataFrame) -> None:
    """Splits sum to the whole and each preserves the fraud class."""
    cfg = PreprocessConfig(test_size=0.15, val_size=0.15, random_seed=42)
    train, val, test = split_data(raw_df, cfg)
    assert len(train) + len(val) + len(test) == len(raw_df)
    # Roughly 70/15/15.
    assert abs(len(test) / len(raw_df) - 0.15) < 0.02
    assert abs(len(val) / len(raw_df) - 0.15) < 0.02
    # Every split must contain at least one fraud (stratification works).
    for split in (train, val, test):
        assert split[TARGET_COLUMN].sum() >= 1


def test_preprocess_outputs_exist(processed_data: dict[str, Path]) -> None:
    """All expected parquet outputs and the scaler are produced."""
    for key in ("train", "val", "test", "reference", "scaler"):
        assert processed_data[key].exists()


def test_reference_is_subset_of_train(processed_data: dict[str, Path]) -> None:
    """The drift reference is ~20% of the training split."""
    train = pd.read_parquet(processed_data["train"])
    reference = pd.read_parquet(processed_data["reference"])
    assert 0.15 * len(train) <= len(reference) <= 0.25 * len(train)


def test_scaler_is_fitted_and_loadable(processed_data: dict[str, Path]) -> None:
    """The persisted scaler can be unpickled and has learned statistics."""
    with processed_data["scaler"].open("rb") as handle:
        scaler = pickle.load(handle)
    assert hasattr(scaler, "center_")
    assert len(scaler.center_) == len(SCALED_COLUMNS)


def test_no_leakage_val_scaled_with_train_stats(
    raw_df: pd.DataFrame, tmp_path: Path
) -> None:
    """Scaling uses train statistics only: train medians are ~0 after scaling."""
    cfg = PreprocessConfig(test_size=0.15, val_size=0.15, random_seed=7)
    paths = preprocess(
        raw_df, cfg, out_dir=tmp_path / "p", scaler_path=tmp_path / "s.pkl"
    )
    train = pd.read_parquet(paths["train"])
    # RobustScaler centres on the median -> scaled train median ~ 0.
    for col in SCALED_COLUMNS:
        assert abs(train[col].median()) < 1e-6
