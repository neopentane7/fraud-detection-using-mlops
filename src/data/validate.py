"""DVC stage: validate the raw dataset against a schema built from config.

This is the pipeline's first hard quality gate. The Pandera schema is
constructed **programmatically from the active dataset profile** (see
``config.py``): every declared feature column must be present, float-typed,
non-null and finite; columns flagged non-negative get a ``>= 0`` check; the
target must be binary; and a dataframe-level check enforces a plausible
positive-class rate. ``strict`` rejects unexpected columns. A failure writes a
machine-readable report and raises, so DVC stops the pipeline before a single
training cycle is wasted on broken data.

Because the schema is config-driven, the same validator works for any dataset
configured in ``params.yaml`` — not just the credit-card fraud set.

Run as a DVC stage; uses ``print`` for stage output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa

from src.config import (
    FEATURE_COLUMNS,
    NON_NEGATIVE_COLUMNS,
    POS_RATE_MAX,
    POS_RATE_MIN,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    VALIDATION_REPORT_PATH,
)


def _finite() -> pa.Check:
    """A column check that rejects NaN/Inf (nullability handles plain nulls)."""
    return pa.Check(
        lambda s: bool(np.isfinite(s.to_numpy()).all()),
        error="non_finite_values",
    )


def build_schema() -> pa.DataFrameSchema:
    """Construct the validation schema for the active dataset profile."""
    columns: dict[str, pa.Column] = {}
    for col in FEATURE_COLUMNS:
        checks = [_finite()]
        if col in NON_NEGATIVE_COLUMNS:
            checks.append(pa.Check.ge(0.0))
        columns[col] = pa.Column(float, checks=checks, nullable=False, coerce=True)
    columns[TARGET_COLUMN] = pa.Column(
        int, checks=pa.Check.isin([0, 1]), nullable=False, coerce=True
    )

    positive_rate_plausible = pa.Check(
        lambda df: POS_RATE_MIN < float(df[TARGET_COLUMN].mean()) < POS_RATE_MAX,
        error="implausible_positive_rate",
    )
    return pa.DataFrameSchema(
        columns, strict=True, coerce=True, checks=[positive_rate_plausible]
    )


def _build_report(
    df: pd.DataFrame, *, passed: bool, errors: list[str]
) -> dict[str, object]:
    """Assemble the validation report dict."""
    n_rows = int(len(df))
    n_pos = int(df[TARGET_COLUMN].sum()) if TARGET_COLUMN in df else 0
    return {
        "passed": passed,
        "n_rows": n_rows,
        "n_positives": n_pos,
        "positive_rate": (n_pos / n_rows) if n_rows else 0.0,
        "errors": errors,
    }


def validate_frame(df: pd.DataFrame) -> dict[str, object]:
    """Validate ``df`` against the config-driven schema.

    Returns:
        A report dict with keys ``passed``, ``n_rows``, ``n_positives``,
        ``positive_rate``, and ``errors``.
    """
    try:
        build_schema().validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        failures = exc.failure_cases["check"].astype(str).unique().tolist()
        return _build_report(df, passed=False, errors=failures)
    return _build_report(df, passed=True, errors=[])


def run_validation(
    raw_path: Path = RAW_DATA_PATH,
    report_path: Path = VALIDATION_REPORT_PATH,
) -> dict[str, object]:
    """Load the raw CSV, validate it, persist the report, and gate the pipeline.

    Raises:
        ValueError: If validation fails — DVC marks the stage failed.
    """
    df = pd.read_csv(raw_path)
    report = validate_frame(df)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[validate] Report written to {report_path}: {report}")

    if not report["passed"]:
        raise ValueError(f"Schema validation failed: {report['errors']}")
    return report


def main() -> int:
    """Stage entrypoint."""
    run_validation()
    return 0


if __name__ == "__main__":
    sys.exit(main())
