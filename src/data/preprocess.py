"""DVC stage: clean, scale, and split the validated dataset.

Design decisions:

* Only ``Time`` and ``Amount`` are scaled (with :class:`RobustScaler`, which
  is resistant to the heavy outliers in transaction amounts). V1-V28 are
  already PCA-transformed and standardised upstream, so re-scaling them would
  destroy information.
* The scaler is fit on the **training split only** and then applied to val/test
  to prevent data leakage; the fitted scaler is persisted for serving parity.
* Splits are stratified on the target so the 577:1 imbalance is preserved in
  every split.
* A 20% stratified sample of the training split is saved as the drift
  ``reference`` baseline used later by the monitoring stage.

Run as a DVC stage; uses ``print`` for stage output.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import (
    PROCESSED_DIR,
    RAW_DATA_PATH,
    SCALED_COLUMNS,
    SCALER_PATH,
    TARGET_COLUMN,
    PreprocessConfig,
    load_config,
)

REFERENCE_SAMPLE_FRAC = 0.20


def _log_split(name: str, frame: pd.DataFrame) -> None:
    """Print the size and class distribution of a split."""
    n = len(frame)
    n_fraud = int(frame[TARGET_COLUMN].sum())
    rate = n_fraud / n if n else 0.0
    print(f"[preprocess] {name:<10} n={n:<7} frauds={n_fraud:<5} rate={rate:.5f}")


def _temporal_split(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: oldest -> train, middle -> val, newest -> test.

    Sorting by ``time_column`` and cutting by position prevents future
    transactions from leaking into the training window — the realistic setup
    for a fraud model scored on tomorrow's traffic.
    """
    ordered = df.sort_values(cfg.time_column, kind="stable").reset_index(drop=True)
    n = len(ordered)
    n_test = int(n * cfg.test_size)
    n_val = int(n * cfg.val_size)
    n_train = n - n_val - n_test
    train = ordered.iloc[:n_train]
    val = ordered.iloc[n_train : n_train + n_val]
    test = ordered.iloc[n_train + n_val :]
    return train, val, test


def split_data(
    df: pd.DataFrame, cfg: PreprocessConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/val/test split — temporal if configured (and possible), else stratified.

    Args:
        df: The full validated dataframe.
        cfg: Split configuration (sizes, seed, strategy, time column).

    Returns:
        ``(train_df, val_df, test_df)``.
    """
    if cfg.split_strategy == "temporal":
        if cfg.time_column in df.columns:
            return _temporal_split(df, cfg)
        print(
            f"[preprocess] split_strategy='temporal' but no '{cfg.time_column}' "
            "column; falling back to stratified."
        )

    y = df[TARGET_COLUMN]
    train_val, test = train_test_split(
        df, test_size=cfg.test_size, random_state=cfg.random_seed, stratify=y
    )
    # val_size is expressed as a fraction of the *whole* dataset.
    val_relative = cfg.val_size / (1.0 - cfg.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_relative,
        random_state=cfg.random_seed,
        stratify=train_val[TARGET_COLUMN],
    )
    return train, val, test


def scale_features(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, RobustScaler]:
    """Fit a RobustScaler on train's Time/Amount and apply to all splits.

    Returns:
        The transformed copies of each split plus the fitted scaler.
    """
    scaler = RobustScaler()
    cols = list(SCALED_COLUMNS)

    train, val, test = train.copy(), val.copy(), test.copy()
    train[cols] = scaler.fit_transform(train[cols])
    val[cols] = scaler.transform(val[cols])
    test[cols] = scaler.transform(test[cols])
    return train, val, test, scaler


def preprocess(
    df: pd.DataFrame,
    cfg: PreprocessConfig,
    out_dir: Path = PROCESSED_DIR,
    scaler_path: Path = SCALER_PATH,
) -> dict[str, Path]:
    """Full preprocessing routine: split, scale, persist parquet + scaler.

    Args:
        df: Validated raw dataframe.
        cfg: Preprocessing configuration.
        out_dir: Directory for the parquet outputs.
        scaler_path: Destination for the pickled fitted scaler.

    Returns:
        Mapping of artifact name -> path for ``train``, ``val``, ``test``,
        ``reference``, and ``scaler``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop exact duplicate rows BEFORE splitting. This dataset contains ~1k
    # duplicates; if the same row lands in both train and test it leaks and
    # inflates metrics. Dedup first so the holdout is genuinely unseen.
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n_removed = n_before - len(df)
    if n_removed:
        pct = n_removed / n_before
        print(f"[preprocess] dropped {n_removed} duplicate rows ({pct:.3%})")

    train, val, test = split_data(df, cfg)
    train, val, test, scaler = scale_features(train, val, test)

    # Drift reference: a stratified sample of the (scaled) training data.
    reference = train.groupby(TARGET_COLUMN, group_keys=False).sample(
        frac=REFERENCE_SAMPLE_FRAC, random_state=cfg.random_seed
    )

    paths = {
        "train": out_dir / "train.parquet",
        "val": out_dir / "val.parquet",
        "test": out_dir / "test.parquet",
        "reference": out_dir / "reference.parquet",
    }
    train.to_parquet(paths["train"], engine="pyarrow", index=False)
    val.to_parquet(paths["val"], engine="pyarrow", index=False)
    test.to_parquet(paths["test"], engine="pyarrow", index=False)
    reference.to_parquet(paths["reference"], engine="pyarrow", index=False)

    with scaler_path.open("wb") as handle:
        pickle.dump(scaler, handle)
    paths["scaler"] = scaler_path

    for name, frame in (
        ("train", train),
        ("val", val),
        ("test", test),
        ("reference", reference),
    ):
        _log_split(name, frame)
    print(f"[preprocess] Scaler saved to {scaler_path}")
    return paths


def main() -> int:
    """Stage entrypoint: load raw CSV, preprocess with configured params."""
    cfg = load_config()
    df = pd.read_csv(RAW_DATA_PATH)
    preprocess(df, cfg.preprocess)
    return 0


if __name__ == "__main__":
    sys.exit(main())
