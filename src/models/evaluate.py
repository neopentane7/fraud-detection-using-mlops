"""DVC stage + library: metric computation, benchmark gating, model comparison.

This module is the project's single source of truth for *how a model is
scored*. ``train.py`` imports :func:`compute_metrics` so the metrics logged
during training are computed identically to the holdout metrics here — there
is never a "the numbers don't match" discrepancy between stages.

As a DVC stage it runs in two modes:

* ``--stage val``     – score the validation split, write ``train_metrics.json``.
* ``--stage holdout`` – score the test split, write ``eval_metrics.json``, then
  enforce the benchmark gate and (if a Production model exists) the
  challenger-vs-champion comparison gate. Exits non-zero on failure so DVC /
  CI treat it as a failed step.

Models are always loaded from MLflow (tracking URI), never from a local file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeAlias

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import (
    BENCHMARK_TARGETS,
    EVAL_METRICS_PATH,
    FEATURE_COLUMNS,
    MODELS_DIR,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
    TEST_PATH,
    THRESHOLD_PATH,
    TRAIN_METRICS_PATH,
    VAL_PATH,
    load_config,
)
from src.models.predict import classify, load_threshold

# Persisted by train.py so evaluate can locate the exact run it produced.
RUN_INFO_PATH: Path = MODELS_DIR / "run_info.json"

# MLflow-loaded models are untyped third-party objects; alias keeps mypy happy
# (calling .predict) without tripping ruff's bare-Any rule.
Model: TypeAlias = Any

# Hard benchmark gate for the holdout/test set, **per active dataset** (from
# params.yaml). For the fraud profile these are calibrated from 5-seed/5-fold
# experiments: roc_auc / avg_precision are threshold-independent and stable,
# recall is the business priority, and precision is a low "not-collapsed" floor.
# `BENCHMARK_TARGETS` is imported from config so each dataset gates on its own
# realistic bar; tests still monkeypatch this module-level name.


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Compute the fraud-focused metric suite (never overall accuracy).

    Args:
        y_true: Ground-truth labels (0/1).
        y_prob: Predicted probability of the fraud class.
        threshold: Decision threshold applied to ``y_prob``.

    Returns:
        Dict of metrics: ``roc_auc``, ``avg_precision``, ``f1_fraud``,
        ``precision_fraud``, ``recall_fraud``, ``f1_macro``, ``support_fraud``,
        and the ``threshold`` used.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = classify(y_prob, threshold)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1_fraud": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "precision_fraud": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_fraud": float(
            recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "support_fraud": int(y_true.sum()),
        "threshold": float(threshold),
    }


def check_benchmarks(metrics: dict[str, float]) -> list[str]:
    """Return a list of human-readable failures for metrics below target."""
    failures: list[str] = []
    for name, target in BENCHMARK_TARGETS.items():
        value = metrics.get(name)
        if value is None or value < target:
            failures.append(f"{name}={value:.4f} < target {target:.2f}")
    return failures


def _resolve_challenger_uri() -> str:
    """Locate the just-trained model URI via the run-info file or latest run."""
    if RUN_INFO_PATH.exists():
        run_id = json.loads(RUN_INFO_PATH.read_text(encoding="utf-8"))["run_id"]
        return f"runs:/{run_id}/model"
    # Fall back to the most recent registered version.
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            "No challenger model found (no run_info.json, no registry versions)."
        )
    latest = max(versions, key=lambda mv: int(mv.version))
    return f"models:/{REGISTERED_MODEL_NAME}/{latest.version}"


def _load_production_model() -> Model | None:
    """Load the current Production model, or ``None`` if none is registered."""
    try:
        return mlflow.pyfunc.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")
    except Exception:  # noqa: BLE001 - registry/model may simply not exist yet
        return None


def _score(model: Model, frame: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Score a loaded pyfunc model on a feature/target frame."""
    probs = np.asarray(model.predict(frame[list(FEATURE_COLUMNS)]))
    return compute_metrics(frame[TARGET_COLUMN], probs, threshold)


def _print_comparison(challenger: dict[str, float], champion: dict[str, float]) -> None:
    """Print a challenger-vs-production comparison table to stdout."""
    cols = ["f1_fraud", "roc_auc", "precision_fraud", "recall_fraud"]
    header = f"{'Model':<18}" + "".join(f"{c:<14}" for c in cols)
    print(header)
    print("─" * len(header))
    for label, m in (("Challenger", challenger), ("Current Prod", champion)):
        print(f"{label:<18}" + "".join(f"{m[c]:<14.4f}" for c in cols))
    print(
        f"{'Delta':<18}"
        + "".join(f"{challenger[c] - champion[c]:<+14.4f}" for c in cols)
    )


def evaluate(stage: str) -> dict[str, float]:
    """Run evaluation for the given stage and persist its metrics file.

    Args:
        stage: Either ``"val"`` or ``"holdout"``.

    Returns:
        The computed metric dict.

    Raises:
        SystemExit: With code 1 if a holdout gate fails.
    """
    cfg = load_config()
    # Reference the module global by name so it is resolved at call time
    # (a bare default argument would bind the original path at import time).
    threshold = load_threshold(THRESHOLD_PATH)
    data_path = VAL_PATH if stage == "val" else TEST_PATH
    out_path = TRAIN_METRICS_PATH if stage == "val" else EVAL_METRICS_PATH

    frame = pd.read_parquet(data_path)
    challenger = mlflow.pyfunc.load_model(_resolve_challenger_uri())
    metrics = _score(challenger, frame, threshold)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[evaluate:{stage}] metrics -> {out_path}: {metrics}")

    if stage != "holdout":
        return metrics

    # --- Gate 1: absolute benchmark targets -------------------------------
    failures = check_benchmarks(metrics)
    if failures:
        print("[evaluate:holdout] BENCHMARK GATE FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    # --- Gate 2: must not regress against current Production model --------
    champion_model = _load_production_model()
    if champion_model is not None:
        champion = _score(champion_model, frame, threshold)
        _print_comparison(metrics, champion)
        if metrics["f1_fraud"] < champion["f1_fraud"]:
            print(
                "[evaluate:holdout] CHALLENGER REGRESSION: "
                f"f1_fraud {metrics['f1_fraud']:.4f} < prod {champion['f1_fraud']:.4f}"
            )
            raise SystemExit(1)
    else:
        print("[evaluate:holdout] No Production model yet; skipping comparison gate.")

    print(
        "[evaluate:holdout] All gates passed "
        f"(perf_threshold={cfg.monitoring.performance_threshold})."
    )
    return metrics


def main() -> int:
    """Stage entrypoint with ``--stage {val,holdout}``."""
    parser = argparse.ArgumentParser(description="Evaluate the fraud model.")
    parser.add_argument("--stage", choices=["val", "holdout"], default="holdout")
    args = parser.parse_args()
    evaluate(args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
