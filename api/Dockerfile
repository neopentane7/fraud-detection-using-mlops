# Slim, single-stage image for the serving API. The training stack (dvc,
# shap, evidently, ...) is deliberately excluded — only serving deps ship.
FROM python:3.11-slim

WORKDIR /app

# curl is needed by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Pinned serving deps first (cached layer; the training stack is excluded).
COPY requirements-api.lock ./
RUN pip install --no-cache-dir -r requirements-api.lock
# Then the package itself, editable so config.py resolves PROJECT_ROOT to /app
# (where params.yaml and the baked model artifacts live).
COPY pyproject.toml ./
COPY api/ ./api/
COPY src/ ./src/
RUN pip install --no-cache-dir -e . --no-deps
# config.py resolves the active dataset profile (schema, paths, model name)
# from params.yaml — the image is broken without it.
COPY params.yaml ./params.yaml
# Which dataset's baked artifacts to ship (creditcard for the real model,
# e2e for the secret-free CI deploy smoke). Must match the runtime MLOPS_DATASET.
ARG DATASET=creditcard
COPY models/${DATASET}/threshold.json ./models/${DATASET}/threshold.json
COPY models/${DATASET}/scaler.pkl ./models/${DATASET}/scaler.pkl

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=fraud-detector
ENV MODEL_STAGE=Production

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
