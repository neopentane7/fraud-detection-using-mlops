"""End-to-end smoke: run the FULL MLOps loop with no external secrets.

This proves the pipeline genuinely works — not a no-op. On a tiny synthetic
dataset (the isolated ``e2e`` profile) and a **local SQLite MLflow registry**
it runs, in order:

    validate -> preprocess -> train  (logs model, writes scaler + threshold)
        -> register the run's model in the MLflow Model Registry
        -> promote None -> Production
        -> launch the FastAPI app (uvicorn) pointed at that registry
        -> POST /predict and get a real fraud probability back
        -> tear the server down

SQLite (unlike a file store) supports the model registry, so the
register/promote path executes for real. Run from the repo root:

    python scripts/e2e_smoke.py

Used by .github/workflows/e2e.yml on every push/PR.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
PORT = 8000
BASE = f"http://localhost:{PORT}"
ENV = {
    **os.environ,
    "MLOPS_DATASET": "e2e",
    "MLFLOW_TRACKING_URI": f"sqlite:///{(ROOT / 'mlflow_e2e.db').as_posix()}",
    "PYTHONPATH": str(ROOT),
    "MODEL_NAME": "e2e-smoke-clf",
    "MODEL_STAGE": "Production",
}


def banner(msg: str) -> None:
    print("\n" + "=" * 70 + f"\n# {msg}\n" + "=" * 70, flush=True)


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=ROOT, env=ENV, check=True)


def register_production() -> None:
    """Register the just-trained run's model and promote it to Production."""
    import mlflow

    mlflow.set_tracking_uri(ENV["MLFLOW_TRACKING_URI"])
    run_id = json.loads((ROOT / "models" / "e2e" / "run_info.json").read_text())[
        "run_id"
    ]
    # register_model creates the registered model + a version on a fresh registry.
    mv = mlflow.register_model(f"runs:/{run_id}/model", "e2e-smoke-clf")
    mlflow.tracking.MlflowClient().transition_model_version_stage(
        name="e2e-smoke-clf",
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"registered e2e-smoke-clf v{mv.version} -> Production", flush=True)


def wait_healthy(timeout: int = 60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=5) as r:  # noqa: S310
                body = json.loads(r.read())
                print(f"/health -> {body}", flush=True)
                if body.get("status") == "healthy":
                    return True
        except Exception as exc:  # noqa: BLE001
            print(f"waiting for API... ({exc})", flush=True)
        time.sleep(3)
    return False


def main() -> int:
    banner("STEP 1 - synthetic dataset (no network, isolated 'e2e' profile)")
    run([PY, "scripts/make_smoke_data.py"])

    banner("STEP 2 - pipeline: validate -> preprocess -> train (SQLite MLflow)")
    run([PY, "-m", "src.data.validate"])
    run([PY, "-m", "src.data.preprocess"])
    run([PY, "-m", "src.models.train"])

    banner("STEP 3 - register model + promote to Production (real registry)")
    register_production()

    banner("STEP 4 - launch FastAPI (uvicorn) against the registry")
    server = subprocess.Popen(
        [
            PY,
            "-m",
            "uvicorn",
            "api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=ROOT,
        env=ENV,
    )
    try:
        if not wait_healthy():
            print("FAIL: API never became healthy", flush=True)
            return 1
        banner("STEP 5 - drive it: POST /predict (scripts/integration_test.py)")
        subprocess.run(
            [PY, "scripts/integration_test.py"], cwd=ROOT, env=ENV, check=True
        )
        banner("E2E PASSED - real model trained, registered, promoted, served")
        return 0
    finally:
        banner("STEP 6 - teardown")
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        print("server stopped", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
