"""DVC stage: train the XGBoost fraud model and track everything in MLflow.

The most important file in the project. A single training run:

1. logs every hyperparameter and the tuned decision threshold,
2. computes the full fraud-focused metric suite on the validation split
   (reusing :func:`evaluate.compute_metrics` so numbers are stage-consistent),
3. generates and logs explainability + diagnostic plots (SHAP, confusion
   matrix, PR/ROC curves) and feature importances — SHAP artifacts are a
   regulatory expectation in real fintech pipelines,
4. logs both the native XGBoost model and a probability-returning pyfunc
   wrapper, and
5. registers + promotes the model to ``Staging`` *only if* it clears the
   performance gate.

Run as a DVC stage; uses ``print`` for stage output.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed in CI/containers

import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from mlflow.exceptions import MlflowException  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.config import (  # noqa: E402
    EXPERIMENT_NAME,
    FEATURE_COLUMNS,
    REGISTERED_MODEL_NAME,
    TARGET_COLUMN,
    THRESHOLD_PATH,
    TRAIN_METRICS_PATH,
    TRAIN_OVERRIDES,
    TRAIN_PATH,
    VAL_PATH,
    Config,
    TrainConfig,
    load_config,
)
from src.models.evaluate import RUN_INFO_PATH, compute_metrics
from src.models.predict import (
    XGB_ARTIFACT_KEY,
    FraudProbaModel,
    classify,
    save_xgb_model,
)

SHAP_SAMPLE_SIZE = 500
THRESHOLD_EPS = 1e-8


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the train and validation parquet splits."""
    return pd.read_parquet(TRAIN_PATH), pd.read_parquet(VAL_PATH)


def apply_train_overrides(train_cfg: TrainConfig, y_train: pd.Series) -> TrainConfig:
    """Apply the active dataset's per-dataset hyperparameter overrides.

    Supports ``scale_pos_weight: auto`` (computed as negatives/positives from the
    training labels). Lets one global ``train`` section serve many datasets while
    each still gets an imbalance weight and recall floor appropriate to its base
    rate.
    """
    overrides = dict(TRAIN_OVERRIDES)
    if overrides.get("scale_pos_weight") == "auto":
        y = np.asarray(y_train)
        pos = max(int((y == 1).sum()), 1)
        overrides["scale_pos_weight"] = round(float((y == 0).sum()) / pos, 2)
    valid = {f for f in train_cfg.__dataclass_fields__}
    overrides = {k: v for k, v in overrides.items() if k in valid}
    return replace(train_cfg, **overrides) if overrides else train_cfg


def build_model(cfg: TrainConfig) -> xgb.XGBClassifier:
    """Construct an XGBoost classifier configured for the imbalanced problem.

    ``scale_pos_weight`` (not oversampling) is used to make the loss function
    weight the rare fraud class; ``aucpr`` is the eval metric because PR-AUC is
    the right summary statistic under extreme class imbalance.
    """
    return xgb.XGBClassifier(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        scale_pos_weight=cfg.scale_pos_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        random_state=cfg.random_seed,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )


def find_optimal_threshold(
    model: xgb.XGBClassifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    min_recall: float = 0.85,
) -> float:
    """Choose the decision threshold for the fraud class (Class=1).

    Strategy: pick the **highest-precision** threshold whose validation recall
    is at least ``min_recall``. This is the right objective at 577:1 imbalance,
    where a missed fraud (false negative) costs far more than a false alarm;
    plain-F1 maximisation instead drifts to a precision-heavy point that
    sacrifices recall. If no threshold reaches the recall floor (degenerate or
    very weak model), fall back to maximising F1, and to the neutral 0.5 cut-off
    when the precision-recall curve is empty (e.g. a split with no frauds).
    """
    probs = model.predict_proba(x_val)[:, 1]
    if int(np.asarray(y_val).sum()) == 0:
        return 0.5
    precisions, recalls, thresholds = precision_recall_curve(y_val, probs)
    if thresholds.size == 0:
        return 0.5

    # precision/recall have one more element than thresholds; align by dropping
    # the trailing point (recall=0, precision=1) that has no threshold.
    prec, rec = precisions[:-1], recalls[:-1]
    meets_floor = rec >= min_recall
    if meets_floor.any():
        # thresholds are ascending and recall is (monotonically) non-increasing
        # in threshold, so the LARGEST threshold still meeting the recall floor
        # is the boundary operating point with the best attainable precision at
        # that recall. Picking the raw argmax-precision instead can latch onto a
        # noisy near-zero-threshold spike, so we take the boundary for stability.
        candidate_idx = np.where(meets_floor)[0]
        best = int(candidate_idx[-1])
        return float(thresholds[best])

    # Fallback: maximise F1 when the recall floor is unreachable.
    f1_scores = 2 * (prec * rec) / (prec + rec + THRESHOLD_EPS)
    return float(thresholds[int(np.argmax(f1_scores))]) if f1_scores.size else 0.5


def _save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, out: Path) -> None:
    """Save a row-normalised confusion-matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    disp = ConfusionMatrixDisplay(cm, display_labels=["legit", "fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", values_format=".3f", colorbar=False)
    ax.set_title("Normalised confusion matrix")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_pr_curve(
    y_true: pd.Series, probs: np.ndarray, threshold: float, out: Path
) -> None:
    """Save the precision-recall curve with the operating point marked."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, probs)
    op_idx = int(np.argmin(np.abs(thresholds - threshold)))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recalls, precisions, label="PR curve")
    ax.scatter(
        recalls[op_idx],
        precisions[op_idx],
        color="red",
        zorder=5,
        label=f"operating point @ {threshold:.3f}",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (fraud class)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_roc_curve(y_true: pd.Series, probs: np.ndarray, out: Path) -> None:
    """Save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label="ROC curve")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def _save_shap_summary(
    model: xgb.XGBClassifier, x_val: pd.DataFrame, out: Path
) -> None:
    """Save a SHAP beeswarm summary plot on a validation sample."""
    sample = x_val.sample(n=min(SHAP_SAMPLE_SIZE, len(x_val)), random_state=0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()


def log_plots(
    model: xgb.XGBClassifier,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float,
    artifact_dir: Path,
) -> None:
    """Generate all diagnostic plots into ``artifact_dir`` and log them."""
    probs = model.predict_proba(x_val)[:, 1]
    y_pred = classify(probs, threshold)

    cm_path = artifact_dir / "confusion_matrix.png"
    pr_path = artifact_dir / "pr_curve.png"
    roc_path = artifact_dir / "roc_curve.png"
    shap_path = artifact_dir / "shap_summary.png"

    _save_confusion_matrix(y_val, y_pred, cm_path)
    _save_pr_curve(y_val, probs, threshold, pr_path)
    _save_roc_curve(y_val, probs, roc_path)
    _save_shap_summary(model, x_val, shap_path)

    for path in (cm_path, pr_path, roc_path, shap_path):
        mlflow.log_artifact(str(path))


def _log_feature_importance(model: xgb.XGBClassifier, artifact_dir: Path) -> None:
    """Write gain-based feature importances to CSV and log them."""
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    frame = (
        pd.DataFrame({"feature": list(gain.keys()), "gain": list(gain.values())})
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    path = artifact_dir / "feature_importance.csv"
    frame.to_csv(path, index=False)
    mlflow.log_artifact(str(path))


def _log_models(model: xgb.XGBClassifier, artifact_dir: Path) -> None:
    """Log the native XGBoost model and a probability pyfunc wrapper."""
    mlflow.xgboost.log_model(model, artifact_path="model_native")

    xgb_path = save_xgb_model(model, artifact_dir / "xgb_model.json")
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=FraudProbaModel(),
        artifacts={XGB_ARTIFACT_KEY: str(xgb_path)},
        code_paths=[
            str(Path(__file__).resolve().parents[1])
        ],  # ship src/ for the wrapper
    )


def maybe_promote_to_staging(run_id: str, f1_fraud: float, cfg: Config) -> None:
    """Register and promote the run's model to Staging if it clears the gate."""
    target = cfg.monitoring.performance_threshold
    if f1_fraud < target:
        print(f"[Gate] f1_fraud={f1_fraud:.4f} < {target} — not promoting")
        return
    try:
        client = mlflow.tracking.MlflowClient()
        mv = client.create_model_version(
            name=REGISTERED_MODEL_NAME,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME, version=mv.version, stage="Staging"
        )
        print(f"[Gate] Promoted model v{mv.version} to Staging")
    except MlflowException as exc:
        # File-store backends (e.g. plain CI) don't support the registry.
        print(f"[Gate] Registry unavailable, skipping promotion: {exc}")


def _persist_outputs(run_id: str, threshold: float) -> None:
    """Write DVC-tracked threshold.json and the run-info pointer file."""
    THRESHOLD_PATH.parent.mkdir(parents=True, exist_ok=True)
    THRESHOLD_PATH.write_text(
        json.dumps({"threshold": threshold}, indent=2), encoding="utf-8"
    )
    RUN_INFO_PATH.write_text(
        json.dumps({"run_id": run_id, "model_uri": f"runs:/{run_id}/model"}, indent=2),
        encoding="utf-8",
    )
    mlflow.log_artifact(str(THRESHOLD_PATH))


def train(cfg: Config) -> str:
    """Train, evaluate, log, and conditionally promote the model.

    Returns:
        The MLflow run id of the completed training run.
    """
    train_df, val_df = load_data()
    x_train, y_train = train_df[list(FEATURE_COLUMNS)], train_df[TARGET_COLUMN]
    x_val, y_val = val_df[list(FEATURE_COLUMNS)], val_df[TARGET_COLUMN]

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp:
        run_id = str(run.info.run_id)
        artifact_dir = Path(tmp)

        train_cfg = apply_train_overrides(cfg.train, y_train)
        model = build_model(train_cfg)
        model.fit(x_train, y_train)

        threshold = find_optimal_threshold(model, x_val, y_val, train_cfg.min_recall)
        probs = model.predict_proba(x_val)[:, 1]
        metrics = compute_metrics(y_val, probs, threshold)

        mlflow.log_params(train_cfg.__dict__)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "threshold"})

        log_plots(model, x_val, y_val, threshold, artifact_dir)
        _log_feature_importance(model, artifact_dir)
        _log_models(model, artifact_dir)
        _persist_outputs(run_id, threshold)

        TRAIN_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TRAIN_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        print(f"[train] run_id={run_id} metrics={metrics}")
        maybe_promote_to_staging(run_id, metrics["f1_fraud"], cfg)
        return run_id


def main() -> int:
    """Stage entrypoint."""
    train(load_config())
    return 0


if __name__ == "__main__":
    sys.exit(main())
