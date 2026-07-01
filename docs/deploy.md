# Deploying a live demo

The repo ships **two** serving images:

| Image | Loads the model from | Use |
| --- | --- | --- |
| `api/Dockerfile` | the MLflow **registry** (`models:/.../Production`) | production / the CD pipeline |
| `Dockerfile.demo` | a **local** model trained at startup (`MODEL_URI`) | a self-contained public demo |

The demo image needs **no MLflow server, no Kaggle data, and no secrets**: on
startup it trains the tiny synthetic `e2e` model into a local SQLite store and
serves it, so anyone can hit the Swagger UI and `POST /predict`.

## Try it locally

```bash
docker build -f Dockerfile.demo -t fraud-demo .
docker run -p 7860:7860 fraud-demo
# open http://localhost:7860/docs  (Swagger UI), or:
curl -X POST http://localhost:7860/predict \
  -H "Content-Type: application/json" -d @tests/sample_transaction.json
```

First start takes ~30–60s while the model trains; `/health` flips to `healthy`
once it's ready.

## Render (free tier)

`render.yaml` is committed. Either:

1. Push the repo to GitHub, then in the Render dashboard **New → Blueprint** and
   point it at the repo — it reads `render.yaml` and builds `Dockerfile.demo`.
2. Or **New → Web Service → Docker**, set the Dockerfile path to
   `Dockerfile.demo`, health check path `/health`, free plan.

Render injects `$PORT`; the entrypoint binds it automatically. The public URL's
`/docs` is your live Swagger demo.

## Hugging Face Spaces (Docker SDK)

Create a **Docker** Space and add this front-matter to its `README.md`:

```yaml
---
title: Fraud Detector API
sdk: docker
app_port: 7860
---
```

Spaces builds a root `Dockerfile`, so copy `Dockerfile.demo` to `Dockerfile` in
the Space (or add a one-line `Dockerfile` that `FROM`s your built image). Note:
Spaces runs as a non-root user, so writable paths must be under the app dir
(they are here — training writes to `/app`).

## Securing the public endpoint (optional)

Set env vars on the host to require a key and rate-limit:

- `API_KEY=<secret>` → callers must send `X-API-Key: <secret>` to `/predict`.
- `RATE_LIMIT_PER_MINUTE=60` → per-IP cap on the scoring endpoints.
