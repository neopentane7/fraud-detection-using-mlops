"""FastAPI application serving real-time fraud predictions.

Operational design:

* The model is loaded once at startup from the MLflow Model Registry via the
  ``lifespan`` context manager (the modern FastAPI pattern, not the deprecated
  ``@app.on_event``). The API never references a model file path — deploying a
  new model is a registry stage transition, not a code change.
* If the model fails to load the app stays up but reports ``degraded`` health
  and returns HTTP 503 from scoring endpoints (a liveness probe can distinguish
  "process alive" from "not ready to serve").
* Every response carries an ``X-Request-ID`` header and every prediction is
  appended to a JSONL audit log (input hash, output, latency) for traceability.
* ``/metrics`` exposes Prometheus-compatible metrics via the instrumentator.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import pickle
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, TypeAlias

import mlflow
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionResponse,
    TransactionFeatures,
)
from src.config import FEATURE_COLUMNS, SCALED_COLUMNS, SCALER_PATH
from src.models.predict import load_threshold

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# The loaded MLflow pyfunc model is intentionally opaque (no public stub); the
# ASGI middleware callable has a fixed request/response signature.
Model: TypeAlias = Any
Scaler: TypeAlias = Any  # fitted sklearn transformer (untyped third-party object)
CallNext: TypeAlias = Callable[[Request], Awaitable[Response]]

# --- Mutable serving state populated at startup -----------------------------
STATE: dict[str, Any] = {"model": None, "threshold": None, "scaler": None, "info": {}}


def _load_scaler(path: Path = SCALER_PATH) -> Scaler | None:
    """Load the fitted feature scaler, or ``None`` if it is unavailable.

    The model was trained on scaled ``Time``/``Amount``; the API receives raw
    request values, so the *same* fitted scaler must be applied here to avoid
    train/serve skew. Returning ``None`` lets the app start in a degraded state
    rather than crash, but predictions would be unreliable without it.
    """
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (FileNotFoundError, pickle.UnpicklingError) as exc:
        logger.error("Scaler load failed (%s); predictions will be unscaled", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the production model + threshold on startup; release on shutdown."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    # MODEL_URI loads an explicit model (a local dir or runs:/...), bypassing the
    # registry — used by the self-contained demo deployment that has no server.
    model_uri = os.getenv("MODEL_URI", "")
    model_name = os.getenv("MODEL_NAME", "fraud-detector")
    model_stage = os.getenv("MODEL_STAGE", "Production")
    mlflow.set_tracking_uri(tracking_uri)
    try:
        if model_uri:
            STATE["model"] = mlflow.pyfunc.load_model(model_uri)
            version = "baked"
        else:
            STATE["model"] = mlflow.pyfunc.load_model(
                f"models:/{model_name}/{model_stage}"
            )
            version = _resolve_version(model_name, model_stage)
        STATE["threshold"] = load_threshold()
        STATE["scaler"] = _load_scaler()
        STATE["info"] = {"name": model_name, "stage": model_stage, "version": version}
        logger.info(
            "Loaded model %s (v%s)", model_uri or f"{model_name}/{model_stage}", version
        )
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, report via /health
        logger.error("Model load failed; serving in degraded mode: %s", exc)
        STATE["info"] = {"name": model_name, "stage": model_stage, "version": "unknown"}
    yield
    STATE["model"] = None
    STATE["scaler"] = None


def _resolve_version(model_name: str, model_stage: str) -> str:
    """Best-effort lookup of the concrete version behind a stage alias."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[model_stage])
        return str(versions[0].version) if versions else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


app = FastAPI(title="Fraud Detector API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(
    app, endpoint="/metrics", include_in_schema=False
)


@app.middleware("http")
async def add_request_id(request: Request, call_next: CallNext) -> Response:
    """Set one X-Request-ID on request state, the response, and the audit log."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# Per-IP request timestamps for the in-memory rate limiter.
_RATE_BUCKETS: dict[str, list[float]] = {}


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next: CallNext) -> Response:
    """Optional API-key auth + per-IP rate limiting on the scoring endpoints.

    Both are off by default (no ``API_KEY`` / ``RATE_LIMIT_PER_MINUTE=0``) so
    local and CI runs need no credentials; set the env vars to enable them in
    production. The limiter is in-memory per process — front it with Redis (or a
    gateway) for multi-replica deployments.
    """
    if request.url.path.startswith("/predict"):
        api_key = os.getenv("API_KEY", "")
        if api_key:
            provided = request.headers.get("X-API-Key", "")
            if not hmac.compare_digest(provided, api_key):
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid or missing API key"}
                )
        limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "0"))
        if limit > 0:
            client_ip = request.client.host if request.client else "unknown"
            now = time.time()
            recent = [t for t in _RATE_BUCKETS.get(client_ip, []) if t > now - 60.0]
            if len(recent) >= limit:
                return JSONResponse(
                    status_code=429, content={"detail": "Rate limit exceeded"}
                )
            recent.append(now)
            _RATE_BUCKETS[client_ip] = recent
    return await call_next(request)


def get_model() -> Model:
    """Return the loaded model or raise HTTP 503 if unavailable."""
    model = STATE["model"]
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _audit_log(request_id: str, payload_hash: str, result: dict[str, Any]) -> None:
    """Append a prediction record to the JSONL audit log."""
    log_path = Path(os.getenv("PREDICTION_LOG_PATH", "predictions.jsonl"))
    record = {
        "request_id": request_id,
        "input_hash": payload_hash,
        "fraud_probability": result["fraud_probability"],
        "is_fraud": result["is_fraud"],
        "latency_ms": result["latency_ms"],
        "model_version": result["model_version"],
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def _hash_payload(frame: pd.DataFrame) -> str:
    """Return a stable SHA-256 hash of a feature frame."""
    raw = frame.to_json(orient="records").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _score_frame(
    model: Model, frame: pd.DataFrame, threshold: float
) -> list[dict[str, Any]]:
    """Score a feature frame, returning one result dict per row.

    Raw ``Time``/``Amount`` from the request are transformed with the fitted
    training scaler before scoring so the model sees inputs in the same space
    it was trained on (no train/serve skew).
    """
    ordered = frame[list(FEATURE_COLUMNS)].copy()
    scaler = STATE.get("scaler")
    if scaler is not None:
        ordered[list(SCALED_COLUMNS)] = scaler.transform(ordered[list(SCALED_COLUMNS)])
    probs = [float(p) for p in model.predict(ordered)]
    version = STATE["info"].get("version", "unknown")
    results: list[dict[str, Any]] = []
    for prob in probs:
        results.append(
            {
                "fraud_probability": prob,
                "is_fraud": prob >= threshold,
                "threshold_used": threshold,
                "model_version": version,
                "top_features": None,
            }
        )
    return results


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Liveness/readiness probe reporting loaded-model identity."""
    info = STATE["info"]
    status: Literal["healthy", "degraded"] = (
        "healthy" if STATE["model"] is not None else "degraded"
    )
    return HealthResponse(
        status=status,
        model_name=info.get("name", "unknown"),
        model_version=info.get("version", "unknown"),
        model_stage=info.get("stage", "unknown"),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    transaction: TransactionFeatures,
    request: Request,
    model: Model = Depends(get_model),
) -> PredictionResponse:
    """Score a single transaction."""
    start = time.perf_counter()
    frame = pd.DataFrame([transaction.model_dump()])
    result = _score_frame(model, frame, STATE["threshold"])[0]
    result["latency_ms"] = (time.perf_counter() - start) * 1000.0

    _audit_log(request.state.request_id, _hash_payload(frame), result)
    return PredictionResponse(**result)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(
    payload: BatchPredictionRequest,
    request: Request,
    model: Model = Depends(get_model),
) -> BatchPredictionResponse:
    """Score up to 1000 transactions in one request."""
    start = time.perf_counter()
    frame = pd.DataFrame([t.model_dump() for t in payload.transactions])
    results = _score_frame(model, frame, STATE["threshold"])
    elapsed = (time.perf_counter() - start) * 1000.0
    per_row = elapsed / max(len(results), 1)

    request_id = request.state.request_id
    responses: list[PredictionResponse] = []
    for i, result in enumerate(results):
        result["latency_ms"] = per_row
        _audit_log(f"{request_id}:{i}", _hash_payload(frame.iloc[[i]]), result)
        responses.append(PredictionResponse(**result))
    return BatchPredictionResponse(predictions=responses, count=len(responses))
