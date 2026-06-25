"""Model Registry promotion utilities used by the CD and retrain workflows.

Promotion is a *registry stage transition*, never a code change or redeploy.
This script wraps the two transitions the pipeline needs:

* ``--promote staging-to-prod``: after the container integration test passes,
  move the latest Staging version to Production (archiving the previous one).
* ``--compare``: compare the latest Staging challenger's holdout f1_fraud
  against the current Production champion and promote only if it wins by at
  least ``--min-delta`` (used by the drift-triggered retrain flow).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

from src.config import EVAL_METRICS_PATH, REGISTERED_MODEL_NAME


def _client() -> mlflow.tracking.MlflowClient:
    return mlflow.tracking.MlflowClient()


def _latest_version(stage: str) -> str | None:
    """Return the latest model version in ``stage`` or None."""
    versions = _client().get_latest_versions(REGISTERED_MODEL_NAME, stages=[stage])
    return versions[0].version if versions else None


def promote_staging_to_production() -> int:
    """Transition the latest Staging version to Production, archiving the old."""
    client = _client()
    version = _latest_version("Staging")
    if version is None:
        print("[promote] No Staging version to promote.")
        return 1
    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"[promote] v{version} -> Production (previous Production archived)")
    return 0


def compare_and_promote(min_delta: float) -> int:
    """Promote Staging only if it beats Production f1_fraud by >= ``min_delta``."""
    challenger_f1 = float(json.loads(EVAL_METRICS_PATH.read_text())["f1_fraud"])
    prod_version = _latest_version("Production")
    if prod_version is None:
        print("[promote] No Production model; promoting challenger unconditionally.")
        return promote_staging_to_production()

    run_id = _client().get_model_version(REGISTERED_MODEL_NAME, prod_version).run_id
    prod_f1 = float(_client().get_run(run_id).data.metrics.get("f1_fraud", 0.0))
    delta = challenger_f1 - prod_f1
    print(
        f"[promote] challenger f1={challenger_f1:.4f} prod f1={prod_f1:.4f} "
        f"delta={delta:+.4f}"
    )
    if delta >= min_delta:
        return promote_staging_to_production()
    print(
        f"[promote] Challenger did not win by >= {min_delta}; "
        "keeping current Production."
    )
    return 0


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Promote registered models.")
    parser.add_argument("--promote", choices=["staging-to-prod"])
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--min-delta", type=float, default=0.02)
    parser.add_argument(
        "--summary", type=Path, help="Optional GitHub step-summary file"
    )
    args = parser.parse_args()

    if args.compare:
        code = compare_and_promote(args.min_delta)
    elif args.promote == "staging-to-prod":
        code = promote_staging_to_production()
    else:
        parser.error("Specify --promote staging-to-prod or --compare")
    return code


if __name__ == "__main__":
    sys.exit(main())
