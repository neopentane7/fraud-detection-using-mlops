"""Typed, dataclass-based configuration loader for the whole pipeline.

Every pipeline stage reads its parameters through :func:`load_config` rather
than touching ``params.yaml`` directly. Centralising parsing here means:

* parameters are validated/typed at one boundary (fail fast on a typo),
* stages become unit-testable by constructing config objects in code, and
* the on-disk parameter format can change without editing every script.

**Dataset-agnostic by design.** The data *schema* (feature columns, which
columns to scale, the target, validation bounds) and the per-dataset output
paths come from the ``data`` section of ``params.yaml``. The active dataset is
chosen by the ``MLOPS_DATASET`` environment variable, falling back to
``data.active``. This lets the identical validate -> preprocess -> train ->
evaluate code run on any configured dataset; outputs are namespaced per dataset
so runs never clobber each other. The module-level schema/path constants below
are resolved once at import from the active profile.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

# --- Canonical project paths (resolved relative to the repo root) -----------
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
PARAMS_PATH: Path = PROJECT_ROOT / "params.yaml"

_T = TypeVar("_T")


def _from_dict(cls: type[_T], data: dict[str, Any]) -> _T:
    """Build a dataclass from ``data``, ignoring unexpected keys defensively."""
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]  # cls is a dataclass
    return cls(**{k: v for k, v in data.items() if k in known})


# --- Dataset profile (schema + acquisition) --------------------------------
@dataclass(frozen=True)
class DataConfig:
    """Schema and acquisition settings for one dataset."""

    name: str
    raw_filename: str
    target: str
    feature_columns: tuple[str, ...]
    scaled_columns: tuple[str, ...]
    non_negative_columns: tuple[str, ...]
    min_rows: int
    pos_rate_min: float
    pos_rate_max: float
    experiment_name: str
    model_name: str
    benchmarks: dict[str, float]
    train_overrides: dict[str, Any]
    source: str = "kaggle-or-openml"
    openml_data_id: int | None = None
    kaggle_dataset: str = ""
    # Named assembler for multi-file datasets that need a custom join/label step
    # before they become a single canonical table (e.g. "elliptic").
    kaggle_assembler: str = ""


# Baked-in fallback so the module imports even without a ``data`` section.
_DEFAULT_PROFILE: dict[str, Any] = {
    "name": "creditcard",
    "raw_filename": "creditcard.csv",
    "target": "Class",
    "feature_spec": {"base": ["Time", "Amount"], "v_range": 28},
    "scaled_columns": ["Time", "Amount"],
    "non_negative_columns": ["Time", "Amount"],
    "min_rows": 200000,
    "pos_rate": [0.0005, 0.05],
    "experiment_name": "fraud-detection",
    "model_name": "fraud-detector",
    "benchmarks": {
        "roc_auc": 0.96,
        "avg_precision": 0.80,
        "recall_fraud": 0.78,
        "precision_fraud": 0.60,
    },
    "train_overrides": {},
    "source": "kaggle-or-openml",
    "openml_data_id": 42175,
    "kaggle_dataset": "mlg-ulb/creditcardfraud",
}


def _expand_features(profile: dict[str, Any]) -> list[str]:
    """Resolve a profile's feature columns (explicit list or a compact spec)."""
    if "feature_columns" in profile:
        return list(profile["feature_columns"])
    spec = profile["feature_spec"]
    # Generic "<prefix>1..<prefix>N" expansion (e.g. f1..f165 for Elliptic).
    if "prefix" in spec:
        return [f"{spec['prefix']}{i}" for i in range(1, int(spec["count"]) + 1)]
    cols = list(spec.get("base", []))
    if spec.get("v_range"):
        cols += [f"V{i}" for i in range(1, int(spec["v_range"]) + 1)]
    return cols


def _resolve_data_config() -> DataConfig:
    """Resolve the active dataset profile from params.yaml (+ env override)."""
    raw: dict[str, Any] = {}
    if PARAMS_PATH.exists():
        raw = yaml.safe_load(PARAMS_PATH.read_text(encoding="utf-8")) or {}
    section = raw.get("data", {})
    profiles = section.get("datasets", {"creditcard": _DEFAULT_PROFILE})
    active = os.getenv("MLOPS_DATASET") or section.get("active", "creditcard")
    if active not in profiles:
        raise KeyError(f"Unknown dataset '{active}'. Known: {sorted(profiles)}")
    p = profiles[active]

    features = _expand_features(p)
    scaled_raw = p.get("scaled_columns", [])
    scaled = list(features) if scaled_raw == "all" else list(scaled_raw)
    pos_lo, pos_hi = p.get("pos_rate", [0.0005, 0.05])
    return DataConfig(
        name=p.get("name", active),
        raw_filename=p["raw_filename"],
        target=p["target"],
        feature_columns=tuple(features),
        scaled_columns=tuple(scaled),
        non_negative_columns=tuple(p.get("non_negative_columns", [])),
        min_rows=int(p.get("min_rows", 0)),
        pos_rate_min=float(pos_lo),
        pos_rate_max=float(pos_hi),
        experiment_name=p.get("experiment_name", f"{active}-experiment"),
        model_name=p.get("model_name", f"{active}-model"),
        benchmarks=dict(p.get("benchmarks", _DEFAULT_PROFILE["benchmarks"])),
        train_overrides=dict(p.get("train_overrides", {})),
        source=p.get("source", "openml"),
        openml_data_id=p.get("openml_data_id"),
        kaggle_dataset=p.get("kaggle_dataset", ""),
        kaggle_assembler=p.get("kaggle_assembler", ""),
    )


# --- Active dataset, resolved once at import --------------------------------
DATA: DataConfig = _resolve_data_config()
ACTIVE_DATASET: str = DATA.name

# Schema constants (back-compat names; now profile-driven) -------------------
TARGET_COLUMN: str = DATA.target
FEATURE_COLUMNS: tuple[str, ...] = DATA.feature_columns
SCALED_COLUMNS: tuple[str, ...] = DATA.scaled_columns
NON_NEGATIVE_COLUMNS: tuple[str, ...] = DATA.non_negative_columns
# PCA_COLUMNS kept for back-compat (e.g. the synthetic test fixture): the
# non-scaled features, which for the fraud profile are exactly V1-V28.
PCA_COLUMNS: tuple[str, ...] = tuple(
    c for c in FEATURE_COLUMNS if c not in SCALED_COLUMNS
)
POS_RATE_MIN: float = DATA.pos_rate_min
POS_RATE_MAX: float = DATA.pos_rate_max
MIN_ROWS: int = DATA.min_rows
EXPERIMENT_NAME: str = DATA.experiment_name
REGISTERED_MODEL_NAME: str = DATA.model_name
# Per-dataset holdout benchmark gate (consumed by evaluate.py).
BENCHMARK_TARGETS: dict[str, float] = DATA.benchmarks
# Optional per-dataset training hyperparameter overrides (consumed by train.py).
TRAIN_OVERRIDES: dict[str, Any] = DATA.train_overrides

# --- Per-dataset paths (namespaced so datasets never clobber each other) ----
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw" / DATA.raw_filename
VALIDATION_REPORT_PATH: Path = (
    PROJECT_ROOT / "data" / "validated" / ACTIVE_DATASET / "validation_report.json"
)
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed" / ACTIVE_DATASET
TRAIN_PATH: Path = PROCESSED_DIR / "train.parquet"
VAL_PATH: Path = PROCESSED_DIR / "val.parquet"
TEST_PATH: Path = PROCESSED_DIR / "test.parquet"
REFERENCE_PATH: Path = PROCESSED_DIR / "reference.parquet"

MODELS_DIR: Path = PROJECT_ROOT / "models" / ACTIVE_DATASET
SCALER_PATH: Path = MODELS_DIR / "scaler.pkl"
THRESHOLD_PATH: Path = MODELS_DIR / "threshold.json"

METRICS_DIR: Path = PROJECT_ROOT / "metrics" / ACTIVE_DATASET
TRAIN_METRICS_PATH: Path = METRICS_DIR / "train_metrics.json"
EVAL_METRICS_PATH: Path = METRICS_DIR / "eval_metrics.json"

MONITORING_REPORTS_DIR: Path = PROJECT_ROOT / "monitoring" / "reports" / ACTIVE_DATASET


@dataclass(frozen=True)
class PreprocessConfig:
    """Train/val/test split configuration."""

    test_size: float
    val_size: float
    random_seed: int
    # "stratified" (random, preserves class balance) or "temporal" (chronological
    # by ``time_column`` — train=oldest, test=newest). Temporal avoids leaking
    # future transactions into the past for time-ordered data; it falls back to
    # stratified when ``time_column`` is absent (e.g. the cc-default profile).
    split_strategy: str = "stratified"
    time_column: str = "Time"


@dataclass(frozen=True)
class TrainConfig:
    """XGBoost hyperparameters for the training stage."""

    n_estimators: int
    max_depth: int
    learning_rate: float
    scale_pos_weight: float
    subsample: float
    colsample_bytree: float
    random_seed: int
    # Minimum recall the tuned decision threshold must achieve on validation.
    # The threshold maximises precision subject to this floor — at high
    # imbalance a missed positive costs far more than a false alarm, so we do
    # not let plain-F1 optimisation trade recall away.
    min_recall: float = 0.85
    # Decision-threshold objective: "recall" (highest precision above the recall
    # floor) or "cost" (minimise expected cost = cost_fn*FN + cost_fp*FP, the
    # business framing of fraud's asymmetric error costs).
    threshold_strategy: str = "recall"
    cost_fn: float = 10.0   # cost of a missed fraud (false negative)
    cost_fp: float = 1.0    # cost of a false alarm (false positive)
    # Probability calibration fit on validation: "none" or "isotonic". XGBoost
    # with scale_pos_weight emits distorted scores; isotonic makes the reported
    # probability trustworthy. Rank-preserving, so roc_auc/avg_precision are
    # unchanged. Applied transparently at serving via the pyfunc wrapper.
    calibration: str = "none"


@dataclass(frozen=True)
class MonitoringConfig:
    """Thresholds for drift-triggered retraining and model promotion."""

    drift_threshold: float
    performance_threshold: float


@dataclass(frozen=True)
class ServingConfig:
    """Runtime configuration for the FastAPI serving layer."""

    host: str
    port: int
    mlflow_tracking_uri: str
    model_name: str
    model_stage: str


@dataclass(frozen=True)
class Config:
    """Top-level aggregate of every configuration section."""

    preprocess: PreprocessConfig
    train: TrainConfig
    monitoring: MonitoringConfig
    serving: ServingConfig


def load_config(path: Path = PARAMS_PATH) -> Config:
    """Load and parse ``params.yaml`` into a typed :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        KeyError: If a required top-level section is missing.
    """
    if not path.exists():
        raise FileNotFoundError(f"Parameter file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle)

    for section in ("preprocess", "train", "monitoring", "serving"):
        if section not in raw:
            raise KeyError(f"Missing required config section: '{section}'")

    return Config(
        preprocess=_from_dict(PreprocessConfig, raw["preprocess"]),
        train=_from_dict(TrainConfig, raw["train"]),
        monitoring=_from_dict(MonitoringConfig, raw["monitoring"]),
        serving=_from_dict(ServingConfig, raw["serving"]),
    )


if __name__ == "__main__":
    print(f"active dataset: {ACTIVE_DATASET}")
    print(f"target={TARGET_COLUMN} n_features={len(FEATURE_COLUMNS)}")
    print(f"scaled={SCALED_COLUMNS}")
    print(load_config())
