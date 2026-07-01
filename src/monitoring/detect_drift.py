"""Data-drift detection comparing a reference baseline to production traffic.

Uses Evidently, which automatically picks the appropriate statistical test per
column type (e.g. PSI / K-S for continuous features) and emits both a
human-readable HTML report and a machine-parseable JSON summary. The JSON feeds
the retrain decision gate: if the share of drifted features exceeds the
configured threshold, retraining is triggered (a conservative 0.30 default —
over-triggering is far cheaper than missing real distribution shift in fraud).

CLI:
    python -m src.monitoring.detect_drift [--simulate-drift]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from src.config import (
    FEATURE_COLUMNS,
    MONITORING_REPORTS_DIR,
    REFERENCE_PATH,
    TEST_PATH,
    Config,
    load_config,
)

# Heterogeneous values pulled from Evidently's report dict (dicts, floats,
# bools, ...). Aliasing Any keeps mypy permissive at call sites without
# tripping ruff's bare-Any rule (ANN401).
_ReportValue: TypeAlias = Any


def simulate_production_traffic(
    test_path: Path = TEST_PATH,
    n_samples: int = 500,
    drift: bool = False,
    random_seed: int = 0,
) -> pd.DataFrame:
    """Sample production-like traffic from the test set, optionally drifted.

    Args:
        test_path: Parquet file to sample from.
        n_samples: Number of rows to draw.
        drift: If True, inject distribution shift (noise on Amount, shift on V1)
            to exercise the monitoring pipeline without real traffic.
        random_seed: Seed for reproducibility.

    Returns:
        A feature-only DataFrame of simulated current traffic.
    """
    rng = np.random.default_rng(random_seed)
    frame = pd.read_parquet(test_path)
    sample = frame.sample(n=min(n_samples, len(frame)), random_state=random_seed).copy()
    sample = sample[list(FEATURE_COLUMNS)]
    if drift:
        # Shift a few leading features to emulate distribution drift. Generic
        # across dataset profiles (creditcard: Time/Amount/V1; cc-default:
        # x1/x2/x3) rather than hard-coding column names.
        for col in list(FEATURE_COLUMNS)[:3]:
            sample[col] = sample[col] + 4.0 + rng.normal(0.0, 1.5, size=len(sample))
    return sample


def generate_drift_report(
    reference_path: Path,
    current_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Compare reference vs current distributions and persist HTML + summary.

    Returns:
        Dict with ``drifted`` (bool), ``drift_share`` (float),
        ``drifted_columns`` (list[str]), and ``report_path`` (str to the HTML).
    """
    # Imported lazily so unit tests that don't need Evidently stay light.
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    reference = pd.read_parquet(reference_path)[list(FEATURE_COLUMNS)]
    current = pd.read_parquet(current_path)[list(FEATURE_COLUMNS)]

    output_dir.mkdir(parents=True, exist_ok=True)
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    stamp = datetime.now(UTC).strftime("%Y-%m-%d")
    html_path = output_dir / f"drift_{stamp}.html"
    report.save_html(str(html_path))

    # DataDriftPreset expands into several metrics: the dataset-level summary
    # (share_of_drifted_columns, dataset_drift) and the per-column table
    # (drift_by_columns) live in *different* entries, so scan all of them for
    # each key rather than assuming a fixed index/order.
    metrics = report.as_dict().get("metrics", [])

    def _find(key: str) -> _ReportValue:
        for metric in metrics:
            res = metric.get("result", {})
            if isinstance(res, dict) and key in res:
                return res[key]
        return None

    drift_by_col = _find("drift_by_columns") or {}
    drifted_columns = [
        col for col, info in drift_by_col.items() if info.get("drift_detected")
    ]
    share = _find("share_of_drifted_columns")
    return {
        "drifted": bool(_find("dataset_drift")),
        "drift_share": float(share) if share is not None else 0.0,
        "drifted_columns": drifted_columns,
        "report_path": str(html_path),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
    }


def make_retrain_decision(drift_result: dict[str, Any], cfg: Config) -> bool:
    """Return True if drift share exceeds the configured retrain threshold."""
    return float(drift_result["drift_share"]) > cfg.monitoring.drift_threshold


def _write_summary(
    drift_result: dict[str, Any], retrain: bool, output_dir: Path
) -> Path:
    """Persist the machine-readable JSON summary used by the decision gate."""
    summary = {
        "timestamp": datetime.now(UTC).isoformat(),
        "n_reference": drift_result["n_reference"],
        "n_current": drift_result["n_current"],
        "drift_share": drift_result["drift_share"],
        "drifted_columns": drift_result["drifted_columns"],
        "decision": "retrain" if retrain else "no_action",
        "report_path": drift_result["report_path"],
    }
    json_path = output_dir / "drift_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return json_path


def main() -> int:
    """CLI entrypoint: simulate traffic, run drift detection, emit reports."""
    parser = argparse.ArgumentParser(description="Detect production data drift.")
    parser.add_argument(
        "--simulate-drift", action="store_true", help="Inject synthetic drift"
    )
    parser.add_argument("--n-samples", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config()
    current = simulate_production_traffic(
        n_samples=args.n_samples, drift=args.simulate_drift
    )
    current_path = MONITORING_REPORTS_DIR / "current_traffic.parquet"
    MONITORING_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    current.to_parquet(current_path, engine="pyarrow", index=False)

    drift_result = generate_drift_report(
        REFERENCE_PATH, current_path, MONITORING_REPORTS_DIR
    )
    retrain = make_retrain_decision(drift_result, cfg)
    summary_path = _write_summary(drift_result, retrain, MONITORING_REPORTS_DIR)

    print(f"[drift] share={drift_result['drift_share']:.3f} retrain={retrain}")
    print(f"[drift] summary -> {summary_path}")
    print(f"[drift] report  -> {drift_result['report_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
