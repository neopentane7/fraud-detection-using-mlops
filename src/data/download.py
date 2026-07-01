"""DVC stage: acquire the active dataset's raw CSV.

Config-driven: the dataset (Kaggle slug, OpenML id, target name, canonical
column order, minimum rows) comes from the active profile in ``config.py``.
Credentials are read only from ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` (via
``.env``); with none present it falls back to the public OpenML mirror, so the
pipeline reproduces without any auth. The row-count check is the first hard
quality gate.

Run as a DVC stage; uses ``print`` for stage output (captured by DVC).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.config import (
    DATA,
    FEATURE_COLUMNS,
    MIN_ROWS,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)

CANONICAL_COLUMNS = list(FEATURE_COLUMNS) + [TARGET_COLUMN]


def _kaggle_available() -> bool:
    """True if Kaggle can authenticate via env vars or a ``kaggle.json`` token."""
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    config_dir = os.getenv("KAGGLE_CONFIG_DIR") or str(Path.home() / ".kaggle")
    return (Path(config_dir) / "kaggle.json").exists()


def _require_kaggle_credentials() -> None:
    """Fail fast with a clear message if Kaggle credentials are absent."""
    if not _kaggle_available():
        raise OSError(
            "Missing Kaggle credentials. Set KAGGLE_USERNAME/KAGGLE_KEY (see "
            ".env.example) or place kaggle.json in ~/.kaggle."
        )


def _normalize_to_canonical(dest: Path) -> None:
    """Keep only the profile's declared columns, dropping id/extra columns.

    Mirrors the OpenML path so a strict validation schema isn't tripped by
    columns the profile doesn't model (e.g. a row-index or unused field).
    """
    frame = pd.read_csv(dest)
    if set(CANONICAL_COLUMNS).issubset(frame.columns):
        frame[CANONICAL_COLUMNS].to_csv(dest, index=False)


def download_dataset(dest: Path = RAW_DATA_PATH) -> Path:
    """Download and unzip the active dataset from Kaggle to ``dest``."""
    _require_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy import

    dest.parent.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"[download] Fetching Kaggle '{DATA.kaggle_dataset}' into {dest.parent} ...")
    api.dataset_download_files(DATA.kaggle_dataset, path=str(dest.parent), unzip=True)

    expected = dest.parent / dest.name
    if not expected.exists():  # Kaggle archive may use a different inner name
        csvs = list(dest.parent.glob("*.csv"))
        if csvs:
            csvs[0].replace(dest)
    _normalize_to_canonical(dest)
    return dest


def download_elliptic(dest: Path = RAW_DATA_PATH) -> Path:
    """Assemble the Elliptic Bitcoin dataset into one labelled canonical table.

    The dataset ships as separate node-feature and class files (plus an edge
    list we ignore for the tabular model). This joins features to classes on
    ``txId``, keeps only labelled nodes, and maps illicit(1)->1, licit(2)->0.
    """
    _require_kaggle_credentials()
    from kaggle.api.kaggle_api_extended import KaggleApi  # lazy import

    tmp = dest.parent / "_elliptic_raw"
    tmp.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    print(f"[download] Fetching Kaggle '{DATA.kaggle_dataset}' (Elliptic) ...")
    api.dataset_download_files(DATA.kaggle_dataset, path=str(tmp), unzip=True)

    feats_path = next(tmp.rglob("elliptic_txs_features.csv"))
    classes_path = next(tmp.rglob("elliptic_txs_classes.csv"))
    feats = pd.read_csv(feats_path, header=None)
    n_features = feats.shape[1] - 2  # drop txId + time_step
    feats.columns = ["txId", "time_step", *[f"f{i}" for i in range(1, n_features + 1)]]
    classes = pd.read_csv(classes_path)  # txId, class

    merged = feats.merge(classes, on="txId")
    merged = merged[merged["class"] != "unknown"].copy()
    merged[TARGET_COLUMN] = (merged["class"].astype(str) == "1").astype(int)  # illicit
    keep = [f"f{i}" for i in range(1, n_features + 1)] + [TARGET_COLUMN]
    merged[keep].to_csv(dest, index=False)

    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def download_from_openml(dest: Path = RAW_DATA_PATH) -> Path:
    """Fetch the active dataset from OpenML (no credentials needed).

    Normalises the frame to the canonical schema: features in configured order,
    the target renamed to the configured name and cast to int 0/1.
    """
    from sklearn.datasets import fetch_openml

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[download] Fetching OpenML data_id={DATA.openml_data_id} (no auth) ...")
    bunch = fetch_openml(data_id=DATA.openml_data_id, as_frame=True, parser="auto")
    frame = bunch.frame.copy()

    target = getattr(bunch, "target", None)
    target_src = target.name if target is not None and target.name else TARGET_COLUMN
    if target_src != TARGET_COLUMN and target_src in frame.columns:
        frame = frame.rename(columns={target_src: TARGET_COLUMN})
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce").astype(
        int
    )
    for column in frame.columns:
        if column != TARGET_COLUMN:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(float)

    frame = frame[CANONICAL_COLUMNS]
    frame.to_csv(dest, index=False)
    return dest


def fetch_dataset(dest: Path = RAW_DATA_PATH) -> Path:
    """Obtain the raw dataset, preferring Kaggle and falling back to OpenML."""
    if _kaggle_available():
        if DATA.kaggle_assembler == "elliptic":
            return download_elliptic(dest)
        if DATA.kaggle_dataset:
            return download_dataset(dest)
    print("[download] Kaggle unavailable/unset; using the public OpenML mirror.")
    return download_from_openml(dest)


def verify_dataset(path: Path = RAW_DATA_PATH) -> int:
    """Verify the CSV exists and has enough rows; return the row count.

    Raises:
        FileNotFoundError: If the CSV is missing.
        ValueError: If the row count is below the configured minimum.
    """
    if not path.exists():
        raise FileNotFoundError(f"Expected dataset at {path}, but it was not found.")
    frame = pd.read_csv(path)
    n_rows = int(len(frame))
    n_cols = int(frame.shape[1])
    if n_rows < MIN_ROWS:
        raise ValueError(
            f"Dataset has only {n_rows} rows (< {MIN_ROWS}); download likely corrupt."
        )
    pos_rate = (
        float(frame[TARGET_COLUMN].mean()) if TARGET_COLUMN in frame else float("nan")
    )
    print(f"[download] OK: {n_rows} rows x {n_cols} cols, positive_rate={pos_rate:.5f}")
    return n_rows


def main() -> int:
    """Stage entrypoint: fetch (Kaggle or OpenML) then verify the dataset."""
    load_dotenv()
    fetch_dataset()
    verify_dataset()
    return 0


if __name__ == "__main__":
    sys.exit(main())
