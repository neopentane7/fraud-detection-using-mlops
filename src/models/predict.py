"""Prediction utilities and the MLflow ``pyfunc`` model wrapper.

The serving layer must receive *fraud probabilities*, but the default pyfunc
flavour of an ``XGBClassifier`` returns hard class labels. We therefore wrap
the booster in a small :class:`mlflow.pyfunc.PythonModel` whose ``predict``
returns ``P(Class=1)``. Logging this wrapper means the FastAPI app can call
``mlflow.pyfunc.load_model("models:/fraud-detector/Production")`` generically
and always get probabilities, with the decision threshold applied separately
from the persisted ``threshold.json``.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import FEATURE_COLUMNS

XGB_ARTIFACT_KEY = "xgb_model"


class FraudProbaModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper returning fraud probabilities from an XGBoost model."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the serialised XGBoost classifier from logged artifacts."""
        self._model = xgb.XGBClassifier()
        self._model.load_model(context.artifacts[XGB_ARTIFACT_KEY])

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> np.ndarray:
        """Return ``P(Class=1)`` for each input row.

        Args:
            context: Unused (artifacts already loaded in ``load_context``).
            model_input: DataFrame with the model's feature columns.

        Returns:
            1-D array of fraud probabilities in ``[0, 1]``.
        """
        frame = ensure_feature_order(model_input)
        return np.asarray(self._model.predict_proba(frame)[:, 1])


def ensure_feature_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Return ``frame`` reordered/subset to the canonical feature columns.

    Raises:
        KeyError: If any required feature column is missing.
    """
    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise KeyError(f"Input is missing required feature columns: {missing}")
    return frame[list(FEATURE_COLUMNS)]


def save_xgb_model(model: xgb.XGBClassifier, path: Path) -> Path:
    """Persist an XGBoost classifier to the native JSON format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    return path


def classify(probabilities: np.ndarray, threshold: float) -> np.ndarray:
    """Apply the decision threshold to probabilities, returning 0/1 labels."""
    return (probabilities >= threshold).astype(int)
