#!/bin/sh
# Self-contained demo startup: train a tiny synthetic model into a local MLflow
# file store (no tracking server, no Kaggle data), then serve the real FastAPI
# app against it. Used by Dockerfile.demo for a public Swagger deployment.
set -e

export MLOPS_DATASET=e2e
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"   # works across mlflow 2.x/3.x
export MODEL_NAME=e2e-smoke-clf
export PYTHONUTF8=1

echo "[demo] training a tiny synthetic model..."
python scripts/make_smoke_data.py
python -m src.data.validate
python -m src.data.preprocess
python -m src.models.train

# Point the API at the freshly trained model (bypasses the registry).
MODEL_URI=$(python -c "import json; print(json.load(open('models/e2e/run_info.json'))['model_uri'])")
export MODEL_URI
echo "[demo] serving MODEL_URI=$MODEL_URI on port ${PORT:-7860}"
exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT:-7860}"
