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

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from src.config import FEATURE_COLUMNS, THRESHOLD_PATH

XGB_ARTIFACT_KEY = "xgb_model"
CALIBRATOR_ARTIFACT_KEY = "calibrator"


def fit_calibrator(
    probs: np.ndarray, y_true: np.ndarray, method: str = "isotonic"
) -> IsotonicRegression:
    """Fit a probability calibrator on validation scores.

    Isotonic regression learns a monotonic map from raw scores to empirical
    fraud frequency, correcting the distortion ``scale_pos_weight`` introduces.
    """
    if method != "isotonic":
        raise ValueError(f"Unsupported calibration method: {method!r}")
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(probs), np.asarray(y_true))
    return calibrator


def apply_calibrator(calibrator: IsotonicRegression, probs: np.ndarray) -> np.ndarray:
    """Map raw probabilities through a fitted calibrator, clipped to [0, 1]."""
    return np.clip(np.asarray(calibrator.predict(np.asarray(probs))), 0.0, 1.0)


def save_calibrator(calibrator: IsotonicRegression, path: Path) -> Path:
    """Pickle a fitted calibrator to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(calibrator, handle)
    return path


def load_threshold(path: Path = THRESHOLD_PATH) -> float:
    """Read the optimised decision threshold persisted by the training stage.

    Single source of truth shared by the evaluation and serving layers so the
    threshold is read the same way everywhere.
    """
    return float(json.loads(path.read_text(encoding="utf-8"))["threshold"])


class FraudProbaModel(mlflow.pyfunc.PythonModel):
    """Pyfunc wrapper returning fraud probabilities from an XGBoost model."""

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        """Load the XGBoost classifier and (optionally) the calibrator."""
        self._model = xgb.XGBClassifier()
        self._model.load_model(context.artifacts[XGB_ARTIFACT_KEY])
        cal_path = context.artifacts.get(CALIBRATOR_ARTIFACT_KEY)
        if cal_path:
            with open(cal_path, "rb") as handle:
                self._calibrator: IsotonicRegression | None = pickle.load(handle)
        else:
            self._calibrator = None

    def predict(
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: pd.DataFrame,
    ) -> np.ndarray:
        """Return calibrated ``P(Class=1)`` for each input row.

        Args:
            context: Unused (artifacts already loaded in ``load_context``).
            model_input: DataFrame with the model's feature columns.

        Returns:
            1-D array of fraud probabilities in ``[0, 1]``.
        """
        frame = ensure_feature_order(model_input)
        probs = np.asarray(self._model.predict_proba(frame)[:, 1])
        if self._calibrator is not None:
            probs = apply_calibrator(self._calibrator, probs)
        return probs


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
