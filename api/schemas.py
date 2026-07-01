"""Pydantic v2 request/response schemas for the fraud-detection API.

Every transaction field is explicitly typed and constrained so malformed
payloads are rejected at the edge with HTTP 422 before any model code runs.
The 28 PCA components (V1-V28) are listed explicitly rather than via a dict so
OpenAPI docs, client codegen, and validation all see a precise contract.

Note: this request contract is fixed to the credit-card schema (``Time``,
``Amount``, ``V1``-``V28``) so the OpenAPI contract is static. It is the one
deliberate exception to the otherwise config-driven design — serving a different
profile (e.g. ``cc-default``) needs its own schema rather than reusing this one.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

_EXAMPLE: dict[str, Any] = {
    "Time": 406.0,
    "Amount": 0.0,
    **{f"V{i}": 0.0 for i in range(1, 29)},
}


class TransactionFeatures(BaseModel):
    """A single credit-card transaction's feature vector."""

    Time: float = Field(
        ..., ge=0, description="Seconds elapsed since first transaction"
    )
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    model_config = {"json_schema_extra": {"example": _EXAMPLE}}

    @field_validator("*")
    @classmethod
    def _reject_non_finite(cls, value: float) -> float:
        """Reject NaN/Infinity, which JSON permits but would corrupt scoring."""
        if not math.isfinite(value):
            raise ValueError("feature values must be finite (no NaN/Infinity)")
        return value


class FeatureContribution(BaseModel):
    """A single SHAP feature contribution for explainability."""

    feature: str
    shap_value: float


class PredictionResponse(BaseModel):
    """Scoring result for one transaction."""

    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    threshold_used: float
    model_version: str
    latency_ms: float
    top_features: list[FeatureContribution] | None = Field(
        default=None, description="Top SHAP contributors (only for high-risk scores)"
    )


class BatchPredictionRequest(BaseModel):
    """A batch scoring request, capped at 1000 transactions."""

    transactions: list[TransactionFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch scoring results, parallel to the request order."""

    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    """Liveness/readiness payload including loaded-model identity."""

    status: Literal["healthy", "degraded"]
    model_name: str
    model_version: str
    model_stage: str
