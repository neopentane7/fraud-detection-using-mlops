# Architecture & Flow Diagrams

All diagrams use [Mermaid](https://mermaid.js.org/) and render natively on
GitHub and in VS Code. They are the visual companion to the
[README](../README.md).

The pipeline is **dataset-agnostic**: a `data` section in `params.yaml` defines
one profile per dataset, the `MLOPS_DATASET` env var selects the active one, and
all outputs are namespaced per dataset. The same stage code runs on either the
`creditcard` (fraud) or `cc-default` profile — see
[results.md](results.md) and [second_dataset_demo.md](second_dataset_demo.md).

---

## 0. Configuration & dataset selection (config-driven)

`config.py` resolves the active profile **once at import**, so every stage —
including the MLflow pyfunc model reloaded inside `evaluate` — sees the same
schema. This is what makes the pipeline multi-dataset.

```mermaid
flowchart TD
    ENV["$MLOPS_DATASET<br/>(else data.active)"] --> CFG
    P[("params.yaml<br/>data.datasets.&lt;name&gt;")] --> CFG["config.py<br/>resolve active profile"]
    CFG --> SCHEMA["schema<br/>feature/scaled/target cols"]
    CFG --> PATHS["namespaced paths<br/>data/.../&lt;name&gt;, models/&lt;name&gt;"]
    CFG --> GATE["benchmark targets<br/>(per dataset)"]
    CFG --> OVR["train overrides<br/>e.g. scale_pos_weight: auto"]
    SCHEMA --> STAGES["validate · preprocess · train · evaluate · serve"]
    PATHS --> STAGES
    GATE --> STAGES
    OVR --> STAGES

    classDef store fill:#e3f0ff,stroke:#2c6fbb,color:#000;
    class P,CFG store;
```

Shipped profiles: **`creditcard`** (fraud, 578:1, OpenML 42175) and
**`cc-default`** (credit-card default, 3.5:1, OpenML 42477).

---

## 1. System architecture (end-to-end)

```mermaid
flowchart TD
    subgraph SRC["Data source (per active profile)"]
        K["Kaggle (profile slug)"]
        O["OpenML mirror (no auth)"]
    end

    K -->|creds present| DL
    O -->|fallback| DL

    subgraph PIPE["DVC pipeline (reproducible, cached)"]
        DL["download.py<br/>row-count gate"] --> VAL["validate.py<br/>Pandera gate (schema from config)"]
        VAL --> PRE["preprocess.py<br/>dedup -> stratified split<br/>RobustScaler (fit on train)"]
        PRE --> TR["train.py<br/>XGBoost + recall-first threshold"]
        TR --> EV["evaluate.py<br/>benchmark + challenger gates"]
    end

    TR -->|params, metrics, SHAP, model| ML[("MLflow<br/>Tracking + Registry")]
    EV -->|promote if gates pass| ML

    ML -->|"models:/&lt;model&gt;/Production"| API["FastAPI (Docker)<br/>/predict /health /metrics"]
    API --> CLIENT(["Client / fraud analyst"])

    API -->|production traffic sample| MON["detect_drift.py<br/>Evidently"]
    PRE -->|reference baseline| MON
    MON -->|drift_share > 0.30| RETRAIN{{"trigger retrain"}}
    RETRAIN --> PRE

    classDef gate fill:#fde2e2,stroke:#c0392b,color:#000;
    classDef store fill:#e3f0ff,stroke:#2c6fbb,color:#000;
    class VAL,EV gate;
    class ML store;
```

Red nodes are **hard quality gates** that can fail the pipeline.

---

## 2. DVC pipeline DAG

Each stage declares its `deps`, `outs`, `params`, and `metrics`, so DVC caches
and skips unchanged stages and the lineage is fully reproducible.

Outputs are namespaced per dataset (`<name>` = active profile, e.g.
`creditcard`).

```mermaid
flowchart LR
    A["download<br/><i>data/raw/&lt;file&gt;.csv</i>"] --> B["validate<br/><i>validated/&lt;name&gt;/report.json</i>"]
    B --> C["preprocess<br/><i>processed/&lt;name&gt;/*.parquet<br/>models/&lt;name&gt;/scaler.pkl</i>"]
    C --> D["train<br/><i>models/&lt;name&gt;/threshold.json<br/>metrics/&lt;name&gt;/train_metrics.json</i>"]
    D --> E["evaluate<br/><i>metrics/&lt;name&gt;/eval_metrics.json</i>"]

    P[("params.yaml")] -.->|data.* (schema)| B
    P -.->|preprocess.*| C
    P -.->|train.* + per-dataset overrides| D
```

---

## 3. MLflow model promotion lifecycle

Deployment is a **registry stage transition**, never a code change or redeploy.

```mermaid
stateDiagram-v2
    [*] --> None: model logged to run
    None --> Staging: train gate<br/>f1_fraud >= performance_threshold
    Staging --> Production: CD integration test passes
    Production --> Archived: new model promoted<br/>(archive_existing_versions)
    Staging --> None: gate fails / challenger worse
    note right of Production
        API always loads
        models:/&lt;model&gt;/Production
        (per-dataset, e.g. fraud-detector
        or cc-default-clf)
    end note
```

---

## 4. Drift monitoring → auto-retrain loop

```mermaid
flowchart TD
    CRON["monitor.yml<br/>weekly cron"] --> SIM["sample production traffic"]
    SIM --> REP["Evidently report<br/>(HTML + JSON)"]
    REF[("reference.parquet<br/>baseline")] --> REP
    REP --> DEC{"drift_share > 0.30 ?"}
    DEC -->|no| OK["upload report artifact<br/>no action"]
    DEC -->|yes| DISP["repository_dispatch:<br/>drift-retrain"]
    DISP --> RT["retrain.yml<br/>repro preprocess->train->evaluate"]
    RT --> CMP{"challenger beats<br/>Production by >= 2% ?"}
    CMP -->|yes| PROMO["promote + archive old"]
    CMP -->|no| KEEP["keep current Production"]
```

---

## 5. Prediction request flow

```mermaid
sequenceDiagram
    participant C as Client
    participant M as Middleware
    participant API as FastAPI /predict
    participant S as RobustScaler
    participant Mod as MLflow pyfunc model

    C->>M: POST /predict (raw transaction)
    M->>M: attach X-Request-ID
    M->>API: validated payload (Pydantic: finite, ranges)
    API->>S: scale configured columns (fitted on train)
    S-->>API: model-space features
    API->>Mod: predict_proba
    Mod-->>API: fraud_probability
    API->>API: apply tuned threshold -> is_fraud
    API->>API: append JSONL audit log
    API-->>C: {prob, is_fraud, version, latency} + X-Request-ID
```

The scaling step is the fix for the **train/serve skew** bug — the model trains
on scaled inputs (the profile's configured columns, e.g. `Time`/`Amount`), so
the API must transform raw request values with the same fitted scaler before
scoring.

---

## 6. CI/CD topology

```mermaid
flowchart LR
    subgraph PR["Pull request"]
        CI["ci.yml<br/>ruff + mypy + pytest(cov>=80%)<br/>+ DVC DAG + quick-train gate"]
    end
    subgraph MAIN["Manual dispatch (on demand)"]
        CD["cd.yml<br/>full repro -> Staging<br/>-> integration test -> Production<br/>-> push image to GHCR"]
    end
    subgraph SCHED["Scheduled / triggered"]
        MON["monitor.yml (weekly)"]
        RT["retrain.yml (dispatch)"]
    end
    CI -.->|after review| CD
    MON -->|drift| RT
    RT -->|promote| CD
```

The canonical workflows target the default `creditcard` profile; any other
dataset profile runs the same stages via `MLOPS_DATASET=<name>` (a CI matrix
could fan out across profiles).
