# Elliptic — temporal evaluation vs. published results

The `elliptic` profile is evaluated by the pipeline on a **stratified random
split** (for comparability with the other profiles). That is *optimistic* for
this dataset. Its origin paper — **Weber et al. (2019), "Anti-Money Laundering
in Bitcoin: Experimenting with GCNs for Financial Forensics"** — splits
**temporally** (train on time steps 1–34, test on 35–49), which is harder and
realistic: a **dark-marketplace shutdown around step 43** shifts the fraud
dynamics, so a model trained on the old regime can't generalise.

[scripts/elliptic_temporal_eval.py](../scripts/elliptic_temporal_eval.py)
reproduces that protocol on the same XGBoost configuration (165 node features,
illicit class, threshold 0.5).

## Random split vs. temporal split vs. the paper

| Metric (illicit class) | XGBoost — random split | XGBoost — **temporal** (paper protocol) | Paper **RF** (AF) |
| --- | --- | --- | --- |
| ROC-AUC | 0.995 | **0.940** | — |
| AUPRC | 0.980 | **0.803** | — |
| Precision | 0.998\* | **0.768** | 0.956 |
| Recall | 0.689\* | **0.738** | 0.670 |
| F1 | 0.815\* | **0.753** | **0.787** |

\* the random-split run uses a recall-first threshold (0.996); temporal and the
paper use 0.5, so the last two columns are directly comparable.

**On the paper's protocol the F1 is 0.753 — below both the paper's RF (0.787) and
the optimistic random-split 0.815.** The random split inflates F1 by ~0.06 and
AUPRC by ~0.18, because shuffling lets the model see every time period (and the
aggregated neighbour features leak structure across the split). On equal footing
the XGBoost model sits in the same band as the paper's strongest classical
baseline.

## The dark-market-shutdown collapse

Per-time-step recall on the temporal **test** period (threshold 0.5):

| Period | Time steps | Recall | Behaviour |
| --- | --- | --- | --- |
| Pre-shutdown | 35–42 | **0.66 – 1.00** | works well |
| Post-shutdown | 43–49 | **≈ 0.00** (0.00, 0.08, 0.00, 0.50†, 0.00, 0.00, 0.02) | effectively blind |

†step 46 has only 2 illicit nodes. From **step 43 onward the model catches almost
no illicit transactions** — the marketplace shutdown changed the data distribution
and the pre-shutdown model does not transfer. The aggregate F1 (0.75) *hides*
this because the earlier, well-classified steps dominate the count.

## Why this matters

1. **Random splits overstate performance on time-ordered data.** The
   paper-comparable number is the temporal one.
2. **The tabular XGBoost ≈ the paper's RF baseline** on equal footing — a
   legitimate, organic result (not the perfect 1.0 of synthetic "fraud" sets).
3. **Concept drift is real and severe here.** The post-shutdown collapse is the
   textbook motivation for the **drift monitoring + automated retraining** loop
   this project ships ([architecture.md](architecture.md)): a deployed model
   silently goes blind when the world changes, and only monitoring catches it.

## Caveats

- The node features are used **tabularly**; the edge list is ignored — i.e. the
  classical baseline, not the paper's GCN/EvolveGCN graph models.
- The pipeline's production `elliptic` profile keeps the stratified split for
  cross-dataset comparability; this temporal evaluation is the rigorous companion.

## Reproduce

```bash
python scripts/elliptic_temporal_eval.py   # needs Kaggle creds + xgboost
```
