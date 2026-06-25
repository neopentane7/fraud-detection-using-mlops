"""Tests for prediction utilities and the MLflow pyfunc wrapper."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.models.predict import (
    XGB_ARTIFACT_KEY,
    FraudProbaModel,
    classify,
    ensure_feature_order,
    save_xgb_model,
)


def test_classify_applies_threshold() -> None:
    """classify converts probabilities to 0/1 labels at the given threshold."""
    probs = np.array([0.1, 0.5, 0.9])
    np.testing.assert_array_equal(classify(probs, 0.5), np.array([0, 1, 1]))


def test_ensure_feature_order_reorders_and_subsets() -> None:
    """Columns are returned in canonical order, extras dropped."""
    frame = pd.DataFrame({c: [0.0] for c in reversed(FEATURE_COLUMNS)})
    frame["junk"] = 1.0
    ordered = ensure_feature_order(frame)
    assert list(ordered.columns) == list(FEATURE_COLUMNS)


def test_ensure_feature_order_missing_column_raises() -> None:
    """A missing feature column raises KeyError."""
    frame = pd.DataFrame({c: [0.0] for c in FEATURE_COLUMNS if c != "V14"})
    with pytest.raises(KeyError):
        ensure_feature_order(frame)


def test_pyfunc_wrapper_roundtrip(
    processed_data: dict[str, Path], tmp_path: Path
) -> None:
    """A saved XGBoost model loads via the wrapper and returns valid probabilities."""
    import xgboost as xgb

    train = pd.read_parquet(processed_data["train"])
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, eval_metric="aucpr")
    model.fit(train[list(FEATURE_COLUMNS)], train[TARGET_COLUMN])

    path = save_xgb_model(model, tmp_path / "xgb.json")
    assert path.exists()

    wrapper = FraudProbaModel()
    context = SimpleNamespace(artifacts={XGB_ARTIFACT_KEY: str(path)})
    wrapper.load_context(context)  # type: ignore[arg-type]

    probs = wrapper.predict(context, train[list(FEATURE_COLUMNS)].head(10))  # type: ignore[arg-type]
    assert probs.shape == (10,)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
