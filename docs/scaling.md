# Production & scaling design notes

This project is a single-node reference implementation of the fraud MLOps loop.
This document maps the current implementation to what a production system at
scale (e.g. 100k+ transactions/sec with a sub-100ms budget) would additionally
require, so the remaining gaps are explicit and scoped.

## What's built vs. the production target

| Concern | This repo | Production target |
| --- | --- | --- |
| Features | scores the request payload (incl. PCA `V1–V28`) | online **feature store** + streaming aggregations with offline/online parity |
| Serving | one FastAPI/uvicorn process, model in-process | autoscaled replicas behind an LB; batched inference (Triton/Ray Serve) |
| State | in-memory rate limiter, local JSONL audit | Redis (rate limit + feature cache); durable, centralized, tamper-evident audit |
| Data | static CSV (Kaggle/OpenML) | Kafka ingestion; feature/label stores |
| Labels | available at train time | **delayed** (chargebacks land weeks later) — needs a label store + delayed eval |
| Deploy | champion/challenger offline gate → registry flip | canary/shadow + online guardrail metrics + auto-rollback |
| Monitoring | batch Evidently drift | real-time metrics + delayed-label performance + paging |
| Compliance | optional API key, hashed audit | PCI-DSS scope, PII handling, encryption, authz |

## The #1 gap: feature infrastructure

The most significant gap: **`V1–V28` are offline PCA components and cannot be
computed from a single live transaction.** A production system requires:

- **Streaming aggregations** (velocity features): txns/hour for the card, amount
  vs. the user's rolling baseline, geo-distance from last txn, merchant-risk
  rates. Computed from a **Kafka** event stream.
- An **online feature store** (Feast + Redis/DynamoDB) serving those features at
  single-digit-ms, written by the **same** transformation code that builds the
  offline training set — this is what guarantees **train/serve parity**. (Today
  parity is guaranteed only for the scaler; see [api/main.py](../api/main.py)
  `_score_frame`.)
- A point-in-time-correct offline store so training never leaks future
  aggregates into past rows (the temporal split here is a simplified analogue).

## Serving at scale

- **Stateless replicas** behind a load balancer; the app is already
  near-stateless except the in-memory rate limiter — move it to **Redis** (the
  code notes this limitation in `auth_and_rate_limit`).
- **Latency**: measure p50/p99 under load (a Locust suite is the next step), set
  an SLA, and add a feature-cache. XGBoost inference is sub-ms; the budget goes
  to feature lookups, hence the online store.
- **Throughput**: micro-batch scoring and/or a dedicated inference server
  (Triton, Ray Serve) so the model isn't re-loaded per process and GPU/CPU
  batching is exploited.
- **Cold start**: the model loads from the registry at startup
  ([api/main.py](../api/main.py) `lifespan`); a readiness probe already gates
  traffic via `degraded` health.

## Data, labels, and the delayed-feedback problem

Fraud labels arrive late (chargebacks). So:

- Monitor **inputs** in real time (drift — already modelled with Evidently) as a
  leading indicator, since outcome metrics lag.
- A **label store** joins delayed labels back to the original prediction
  (the audit log's `input_hash` + `request_id` are the seed of this) to run a
  **delayed-label evaluation** job for true precision/recall over time.
- Watch for **concept drift** (fraudsters adapt) and adversarial shift, not just
  covariate drift.

## Model lifecycle & safe deploys

- Replace the all-or-nothing registry flip with **shadow** (score live traffic,
  don't act) then **canary** (1% → 100%) with **online guardrail metrics** and
  **automatic rollback**. The offline champion/challenger gate
  ([evaluate.py](../src/models/evaluate.py), [promote_model.py](../scripts/promote_model.py))
  stays as the pre-deploy gate.
- Note: MLflow **stages** are deprecated in 3.x; migrate to **aliases/tags**.

## Reliability

- HA across zones; **timeouts, retries, circuit breakers** on the registry/
  feature-store calls; **idempotency keys** so a retried transaction isn't
  double-scored/double-logged; backpressure + a dead-letter queue on the stream.

## Security & compliance (payments context)

- **PCI-DSS** scope: tokenize/avoid PAN, encrypt in transit + at rest, network
  segmentation.
- **Immutable, centralized audit** (append-only store / WORM), not a local JSONL
  file; retention + right-to-erasure for PII.
- **AuthN/Z** beyond the optional API key (mTLS / OAuth, per-role scopes); secrets
  in a vault, not env files.

## Cost

Stream + online store + always-on replicas dominate cost; retrain on drift (as
here) rather than on a fixed schedule, and right-size the inference fleet to the
p99 SLA.

## Out of scope for this implementation

A feature store, Kafka, Redis, a dedicated inference server, and full PCI
controls require production infrastructure and are intentionally not built here.
The reference loop covers the engineering fundamentals — validation,
reproducibility, a model registry, gated CI/CD, graceful degradation, and
monitoring — and this document records the production evolution beyond them.
