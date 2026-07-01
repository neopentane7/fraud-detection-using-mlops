"""Async API tests using httpx.AsyncClient with an injected fake model.

The MLflow model is replaced by a lightweight fake so tests are hermetic and
fast: they exercise routing, validation, response shapes, and error codes
without a tracking server.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from api import main
from api.main import app


class _FakeModel:
    """Returns a deterministic high fraud probability for every row."""

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), 0.92)


def _valid_transaction() -> dict[str, float]:
    """Build a schema-valid transaction payload."""
    payload = {"Time": 100.0, "Amount": 25.0}
    payload.update({f"V{i}": 0.1 * i for i in range(1, 29)})
    return payload


@pytest.fixture(autouse=True)
def _inject_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Populate serving state with a fake model and isolate the audit log."""
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    main.STATE["model"] = _FakeModel()
    main.STATE["threshold"] = 0.5
    main.STATE["scaler"] = None
    main.STATE["info"] = {
        "name": "fraud-detector",
        "stage": "Production",
        "version": "7",
    }
    yield
    main.STATE["model"] = None
    main.STATE["threshold"] = None
    main.STATE["scaler"] = None
    main.STATE["info"] = {}


def _client() -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_health_ok() -> None:
    async with _client() as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "healthy"
    assert body["model_name"] == "fraud-detector"
    assert body["model_version"] == "7"
    assert "X-Request-ID" in resp.headers


async def test_predict_valid() -> None:
    async with _client() as client:
        resp = await client.post("/predict", json=_valid_transaction())
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["is_fraud"] is True
    assert body["threshold_used"] == 0.5
    assert body["latency_ms"] >= 0.0


async def test_predict_requires_api_key_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With API_KEY set, /predict rejects requests lacking the matching header."""
    monkeypatch.setenv("API_KEY", "s3cret")
    async with _client() as client:
        denied = await client.post("/predict", json=_valid_transaction())
        allowed = await client.post(
            "/predict", json=_valid_transaction(), headers={"X-API-Key": "s3cret"}
        )
    assert denied.status_code == 401
    assert allowed.status_code == 200


async def test_predict_open_when_no_api_key_configured() -> None:
    """Without API_KEY, /predict stays open (default local/CI behaviour)."""
    async with _client() as client:
        resp = await client.post("/predict", json=_valid_transaction())
    assert resp.status_code == 200


async def test_predict_missing_field_returns_422() -> None:
    payload = _valid_transaction()
    del payload["V14"]
    async with _client() as client:
        resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422


async def test_predict_negative_amount_returns_422() -> None:
    payload = _valid_transaction()
    payload["Amount"] = -5.0
    async with _client() as client:
        resp = await client.post("/predict", json=payload)
    assert resp.status_code == 422


async def test_batch_ten_transactions() -> None:
    payload = {"transactions": [_valid_transaction() for _ in range(10)]}
    async with _client() as client:
        resp = await client.post("/predict/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 10
    assert all(0.0 <= p["fraud_probability"] <= 1.0 for p in body["predictions"])


async def test_batch_too_many_returns_422() -> None:
    payload = {"transactions": [_valid_transaction() for _ in range(1001)]}
    async with _client() as client:
        resp = await client.post("/predict/batch", json=payload)
    assert resp.status_code == 422


async def test_predict_503_when_model_unloaded() -> None:
    main.STATE["model"] = None
    async with _client() as client:
        resp = await client.post("/predict", json=_valid_transaction())
    assert resp.status_code == 503


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_schema_rejects_non_finite(token: str) -> None:
    """The schema validator rejects NaN/Infinity at construction time."""
    from pydantic import ValidationError

    from api.schemas import TransactionFeatures

    payload = _valid_transaction()
    payload["V5"] = {
        "NaN": float("nan"),
        "Infinity": float("inf"),
        "-Infinity": float("-inf"),
    }[token]
    with pytest.raises(ValidationError):
        TransactionFeatures(**payload)


async def test_scaler_is_applied_before_scoring() -> None:
    """The fitted scaler transforms Time/Amount before the model sees them."""
    from sklearn.preprocessing import RobustScaler

    captured: dict[str, pd.DataFrame] = {}

    class _SpyModel:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            captured["frame"] = frame.copy()
            return np.full(len(frame), 0.3)

    scaler = RobustScaler()
    scaler.fit(
        pd.DataFrame({"Time": [0.0, 100.0, 200.0], "Amount": [0.0, 50.0, 100.0]})
    )
    main.STATE["model"] = _SpyModel()
    main.STATE["scaler"] = scaler

    payload = _valid_transaction()  # Time=100, Amount=25
    async with _client() as client:
        resp = await client.post("/predict", json=payload)
    assert resp.status_code == 200
    # Raw Amount=25 must have been scaled (median=50, IQR-based) -> not 25.
    assert captured["frame"]["Amount"].iloc[0] != 25.0


def test_sample_transaction_file_is_valid() -> None:
    """The committed sample payload matches the schema (used in docs/curl)."""
    import json

    from api.schemas import TransactionFeatures

    sample = json.loads(
        (
            Path(__file__).resolve().parents[1] / "tests" / "sample_transaction.json"
        ).read_text()
    )
    TransactionFeatures(**sample)
