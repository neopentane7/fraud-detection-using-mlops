# Slim, single-stage image for the serving API. The training stack (dvc,
# shap, evidently, ...) is deliberately excluded — only serving deps ship.
FROM python:3.11-slim

WORKDIR /app

# curl is needed by the container HEALTHCHECK below.
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY api/ ./api/
COPY src/ ./src/
# Model artifacts are namespaced per dataset (default profile: creditcard).
COPY models/creditcard/threshold.json ./models/creditcard/threshold.json
COPY models/creditcard/scaler.pkl ./models/creditcard/scaler.pkl

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000
ENV MODEL_NAME=fraud-detector
ENV MODEL_STAGE=Production

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
