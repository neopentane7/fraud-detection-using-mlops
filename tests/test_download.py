"""Tests for the download stage's credential and verification gates.

The actual Kaggle fetch is not exercised (it needs network + credentials);
these tests cover the guardrails around it, which is where failures bite.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import download as dl
from src.data.download import (
    CANONICAL_COLUMNS,
    MIN_ROWS,
    _require_kaggle_credentials,
    download_from_openml,
    fetch_dataset,
    verify_dataset,
)


def test_require_credentials_raises_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Absent Kaggle env vars and no kaggle.json raise a clear EnvironmentError."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))  # empty: no kaggle.json
    with pytest.raises(EnvironmentError):
        _require_kaggle_credentials()


def test_require_credentials_passes_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Present credentials satisfy the check."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    _require_kaggle_credentials()  # must not raise


def test_verify_missing_file_raises(tmp_path: Path) -> None:
    """A missing dataset file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        verify_dataset(tmp_path / "nope.csv")


def test_verify_too_few_rows_raises(tmp_path: Path) -> None:
    """A truncated/corrupt download (too few rows) raises ValueError."""
    path = tmp_path / "creditcard.csv"
    pd.DataFrame({"Time": [0.0], "Amount": [1.0], "Class": [0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError):
        verify_dataset(path)


def test_fetch_dataset_prefers_kaggle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When Kaggle credentials exist, the Kaggle path is used."""
    monkeypatch.setenv("KAGGLE_USERNAME", "u")
    monkeypatch.setenv("KAGGLE_KEY", "k")
    called = {}
    monkeypatch.setattr(
        dl, "download_dataset", lambda dest: called.setdefault("kaggle", dest)
    )
    monkeypatch.setattr(
        dl, "download_from_openml", lambda dest: called.setdefault("openml", dest)
    )
    fetch_dataset(tmp_path / "creditcard.csv")
    assert "kaggle" in called and "openml" not in called


def test_fetch_dataset_falls_back_to_openml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without credentials, the public OpenML mirror is used."""
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(tmp_path))  # empty: no kaggle.json
    called = {}
    monkeypatch.setattr(
        dl, "download_from_openml", lambda dest: called.setdefault("openml", dest)
    )
    fetch_dataset(tmp_path / "creditcard.csv")
    assert "openml" in called


def test_download_from_openml_normalises_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """OpenML frame is cast/reordered to the canonical schema and written."""
    import types

    rng = np.random.default_rng(0)
    raw = {"Time": rng.uniform(0, 1, 5), "Amount": rng.uniform(0, 1, 5)}
    for i in range(1, 29):
        raw[f"V{i}"] = rng.standard_normal(5)
    raw["Class"] = ["0", "1", "0", "0", "1"]  # strings to exercise int cast
    bunch = types.SimpleNamespace(frame=pd.DataFrame(raw))
    monkeypatch.setattr("sklearn.datasets.fetch_openml", lambda **kw: bunch)

    dest = tmp_path / "creditcard.csv"
    download_from_openml(dest)
    out = pd.read_csv(dest)
    assert list(out.columns) == CANONICAL_COLUMNS
    assert out["Class"].tolist() == [0, 1, 0, 0, 1]


def test_verify_accepts_full_dataset(tmp_path: Path) -> None:
    """A file with enough rows passes and returns the row count."""
    n = MIN_ROWS + 1
    rng = np.random.default_rng(0)
    frame = pd.DataFrame(
        {
            "Time": rng.uniform(0, 1000, n),
            "Amount": rng.uniform(0, 100, n),
            "Class": rng.integers(0, 2, n),
        }
    )
    path = tmp_path / "creditcard.csv"
    frame.to_csv(path, index=False)
    assert verify_dataset(path) == n
