"""Shared pytest fixtures.

The whole test suite runs without Kaggle access or a real dataset: a small
synthetic frame mimicking the credit-card schema (Time, Amount, V1-V28, Class)
with a realistic ~0.2% fraud rate is generated deterministically. This keeps
CI fast and hermetic while still exercising the real code paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, PCA_COLUMNS, TARGET_COLUMN

N_ROWS = 4000
FRAUD_RATE = 0.02  # 2% — within the schema's plausible [0.05%, 5%] band, fast to test


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Return a deterministic synthetic dataset matching the fraud schema."""
    rng = np.random.default_rng(42)
    n_fraud = int(N_ROWS * FRAUD_RATE)
    n_legit = N_ROWS - n_fraud

    labels = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    rng.shuffle(labels)

    data: dict[str, np.ndarray] = {
        "Time": rng.uniform(0.0, 172_800.0, size=N_ROWS),
        "Amount": rng.lognormal(mean=3.0, sigma=1.0, size=N_ROWS),
    }
    # PCA components: shift the fraud class so the model has real signal to learn.
    for col in PCA_COLUMNS:
        base = rng.standard_normal(N_ROWS)
        base[labels == 1] += rng.uniform(0.8, 2.0)
        data[col] = base
    data[TARGET_COLUMN] = labels

    frame = pd.DataFrame(data)
    return frame[list(FEATURE_COLUMNS) + [TARGET_COLUMN]]


@pytest.fixture
def raw_csv(raw_df: pd.DataFrame, tmp_path: Path) -> Path:
    """Persist ``raw_df`` to a CSV and return its path."""
    path = tmp_path / "creditcard.csv"
    raw_df.to_csv(path, index=False)
    return path


@pytest.fixture
def processed_data(raw_df: pd.DataFrame, tmp_path: Path) -> dict[str, Path]:
    """Run the real preprocessing routine on synthetic data.

    Returns a mapping of split name -> parquet path plus the fitted scaler.
    """
    from src.config import PreprocessConfig
    from src.data.preprocess import preprocess

    out_dir = tmp_path / "processed"
    scaler_path = tmp_path / "scaler.pkl"
    cfg = PreprocessConfig(test_size=0.15, val_size=0.15, random_seed=42)
    return preprocess(raw_df, cfg, out_dir=out_dir, scaler_path=scaler_path)
