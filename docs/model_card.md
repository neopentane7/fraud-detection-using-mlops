# Model Card — Fraud / Default Detector

A model card following the [Mitchell et al. (2019)](https://arxiv.org/abs/1810.03993)
template. The same pipeline trains three models from one config-driven codebase;
all are documented here. Metrics are measured on held-out **test** splits — see
[results.md](results.md) for the full breakdown.

## Model details

- **Developed by:** Abhik Kumar Mohanta (personal portfolio project).
- **Type:** Gradient-boosted decision trees (XGBoost), binary classifier
  emitting a calibrated-by-threshold fraud/default probability.
- **Version:** 1.0.0. Tracked in the MLflow Model Registry; promotion to
  `Production` is a registry stage transition, not a code change.
- **Inputs:** a single record's numeric features (creditcard: `Time`, `Amount`,
  `V1..V28`; cc-default: `x1..x23`; elliptic: `f1..f165`). Continuous features
  are scaled with a `RobustScaler` fit on the training split only.
- **Output:** `fraud_probability` ∈ [0, 1], a boolean `is_fraud` at the tuned
  decision threshold, the threshold used, and the serving model version.
- **Decision threshold:** tuned on validation (not 0.5). Default strategy is
  recall-first (highest precision above a recall floor); a cost-based strategy
  (minimise `cost_fn*FN + cost_fp*FP`) is also available.

## Intended use

- **Primary use:** an educational, end-to-end demonstration of a production-style
  MLOps pipeline (validation → training → registry → serving → monitoring).
- **Intended users:** the author and reviewers evaluating the engineering.
- **Out of scope:** real financial decisioning. This is **not** a
  production fraud system and must not be used to approve, decline, or flag real
  transactions or people. No fairness auditing across protected attributes has
  been performed.

## Training data

| | `creditcard` (fraud) | `cc-default` (default) | `elliptic` (AML) |
| --- | --- | --- | --- |
| Source | Kaggle `mlg-ulb/creditcardfraud` / OpenML 42175 | OpenML 42477 (Yeh & Lien 2009) | Kaggle `ellipticco/elliptic-data-set` |
| Rows | 284,807 → 283,726 dedup | 30,000 → 29,965 | 46,564 labelled nodes |
| Positives | 492 | 6,636 | 4,545 |
| Imbalance | 578 : 1 (0.17%) | 3.5 : 1 (22.1%) | 9.2 : 1 (9.76%) |
| Split | stratified 70/15/15 (temporal available where a time column exists) |||

The creditcard and elliptic features are anonymised (PCA / undisclosed node
features), so per-feature semantics (and SHAP narratives) are limited by design.

## Evaluation

Held-out test split; the imbalance is preserved. Threshold-independent metrics
(ROC-AUC, AUPRC) are the fair headline; the tuned threshold sets the operating
point.

| Metric | `creditcard` | `cc-default` | `elliptic` |
| --- | --- | --- | --- |
| ROC-AUC | 0.971 | 0.772 | 0.995 |
| Average precision (AUPRC) | 0.834 | 0.552 | 0.980 |
| Recall (positive) | 0.831 | 0.642 | 0.689 |
| Precision (positive) | 0.678 | 0.434 | 0.998 |
| F1 (positive) | 0.747 | 0.518 | 0.815 |

These match published GBM baselines (creditcard ≈ 0.97 ROC / 0.85 AUPRC;
cc-default ≈ 0.77 ROC / 0.55 AUPRC). 5-fold/5-seed stability on creditcard:
ROC-AUC 0.976 ± 0.01, AUPRC 0.82 ± 0.02. **`elliptic`'s random-split numbers are
optimistic** — on the dataset paper's temporal split, F1 drops to ~0.75 (≈ the
paper's RF) with a concept-drift collapse; see
[elliptic_analysis.md](elliptic_analysis.md).

## Ethical considerations & risks

- **Asymmetric harm.** A missed fraud and a false alarm have very different
  costs; the recall-first / cost-based thresholding makes that trade-off explicit
  rather than defaulting to 0.5. Real deployment would require a calibrated cost
  matrix and human review of flagged cases.
- **No fairness analysis.** The data has no usable protected attributes, so
  disparate-impact testing was not done. A real system would require it.
- **Anonymised features** limit explainability; a real deployment should expose
  reason codes a human reviewer and the affected customer can understand.
- **Drift.** Fraud patterns shift; the monitoring workflow tracks data drift and
  triggers retraining, but concept drift in labels still needs human oversight.
  The `elliptic` temporal evaluation makes this concrete — recall collapses to
  ~0 after a market shutdown ([elliptic_analysis.md](elliptic_analysis.md)).

## Caveats & recommendations

- Raw XGBoost scores are skewed by `scale_pos_weight`; **optional isotonic
  calibration** (`train.calibration: isotonic`) corrects them and is bundled into
  the serving model, but defaults off. Benchmark gating is single-split and noisy
  at low positive counts — K-fold CV gating is the recommended next step.
- The serving image bundles the fitted scaler + threshold and loads the model
  from the registry, keeping training/serving feature handling identical.
