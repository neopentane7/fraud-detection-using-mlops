"""Generate analysis figures (PNG) from the REAL dataset into docs/images/.

All nine charts are computed live from the organic data — including the
``scale_pos_weight`` sweep (fig 8) and the cross-seed variance study (fig 9),
which re-run the actual preprocessing + a recall-first threshold identical to
``src/models/train.find_optimal_threshold``.

Usage:
    python scripts/generate_figures.py            # full (trains ~14 models)
    python scripts/generate_figures.py --fast      # curves only (skip 8 & 9)
"""

# ruff: noqa: E402  (matplotlib backend must be set before pyplot import)
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.config import (
    FEATURE_COLUMNS,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TrainConfig,
    load_config,
)
from src.data.preprocess import preprocess

OUT = Path("docs/images")
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update(
    {"figure.dpi": 130, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3}
)
BLUE, RED = "#2c6fbb", "#c0392b"

SPW_GRID = [1, 5, 12, 24, 50, 100, 300, 577]
VARIANCE_SEEDS = [1, 7, 21, 42, 123]


def build_xgb(
    cfg_train: TrainConfig, scale_pos_weight: float | None = None
) -> xgb.XGBClassifier:
    """Construct the pipeline's XGBoost classifier (mirrors train.build_model)."""
    return xgb.XGBClassifier(
        n_estimators=cfg_train.n_estimators,
        max_depth=cfg_train.max_depth,
        learning_rate=cfg_train.learning_rate,
        scale_pos_weight=scale_pos_weight or cfg_train.scale_pos_weight,
        subsample=cfg_train.subsample,
        colsample_bytree=cfg_train.colsample_bytree,
        random_state=cfg_train.random_seed,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
    )


def recall_first_threshold(
    y_val: np.ndarray, p_val: np.ndarray, min_recall: float
) -> float:
    """Recall-first boundary threshold — mirrors train.find_optimal_threshold."""
    if int(np.asarray(y_val).sum()) == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_val, p_val)
    if thr.size == 0:
        return 0.5
    prec, rec = prec[:-1], rec[:-1]
    meets = rec >= min_recall
    if meets.any():
        return float(thr[int(np.where(meets)[0][-1])])
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thr[int(np.argmax(f1))]) if f1.size else 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="curves only; skip 8 & 9")
    args = parser.parse_args()

    cfg = load_config()
    raw = pd.read_csv(RAW_DATA_PATH)
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "test.parquet")
    feats = list(FEATURE_COLUMNS)

    # Base model for the curve plots.
    model = build_xgb(cfg.train)
    model.fit(train[feats], train[TARGET_COLUMN])
    vp = model.predict_proba(val[feats])[:, 1]
    tp = model.predict_proba(test[feats])[:, 1]
    yv, yt = val[TARGET_COLUMN].values, test[TARGET_COLUMN].values
    chosen = recall_first_threshold(yv, vp, cfg.train.min_recall)

    # ---- 1. Class imbalance ----------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = raw[TARGET_COLUMN].value_counts().sort_index()
    bars = ax.bar(["legit (0)", "fraud (1)"], counts.values, color=[BLUE, RED])
    ax.set_yscale("log")
    ax.set_ylabel("count (log scale)")
    ratio = counts[0] / counts[1]
    ax.set_title(f"Class imbalance — {counts[0]:,} : {counts[1]:,}  ({ratio:.0f}:1)")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "class_imbalance.png")
    plt.close(fig)

    # ---- 2. Amount distribution by class ---------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, raw["Amount"].quantile(0.99), 60)
    ax.hist(
        raw.loc[raw[TARGET_COLUMN] == 0, "Amount"],
        bins=bins,
        density=True,
        alpha=0.6,
        color=BLUE,
        label="legit",
    )
    ax.hist(
        raw.loc[raw[TARGET_COLUMN] == 1, "Amount"],
        bins=bins,
        density=True,
        alpha=0.6,
        color=RED,
        label="fraud",
    )
    ax.set_xlabel("Transaction amount (capped at 99th pct)")
    ax.set_ylabel("density")
    ax.set_title("Amount distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "amount_by_class.png")
    plt.close(fig)

    # ---- 3. PR curve ------------------------------------------------------
    prec_t, rec_t, thr_t = precision_recall_curve(yt, tp)
    auprc = average_precision_score(yt, tp)
    op = int(np.argmin(np.abs(thr_t - chosen)))
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(rec_t, prec_t, color=BLUE, label=f"PR curve (AUPRC={auprc:.3f})")
    ax.scatter(
        rec_t[op],
        prec_t[op],
        color=RED,
        zorder=5,
        s=60,
        label=f"operating point @ thr={chosen:.3f}",
    )
    ax.set_xlabel("Recall (fraud)")
    ax.set_ylabel("Precision (fraud)")
    ax.set_title("Precision-Recall curve (test)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "pr_curve.png")
    plt.close(fig)

    # ---- 4. ROC curve -----------------------------------------------------
    fpr, tpr, _ = roc_curve(yt, tp)
    auc = roc_auc_score(yt, tp)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(fpr, tpr, color=BLUE, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "roc_curve.png")
    plt.close(fig)

    # ---- 5. Threshold trade-off ------------------------------------------
    grid = np.linspace(0.01, 0.99, 99)
    prec_g, rec_g, f1_g = [], [], []
    for t in grid:
        pred = (vp >= t).astype(int)
        tp_ = ((pred == 1) & (yv == 1)).sum()
        fp_ = ((pred == 1) & (yv == 0)).sum()
        fn_ = ((pred == 0) & (yv == 1)).sum()
        p = tp_ / (tp_ + fp_ + 1e-9)
        r = tp_ / (tp_ + fn_ + 1e-9)
        prec_g.append(p)
        rec_g.append(r)
        f1_g.append(2 * p * r / (p + r + 1e-9))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(grid, prec_g, color=BLUE, label="precision")
    ax.plot(grid, rec_g, color=RED, label="recall")
    ax.plot(grid, f1_g, color="green", label="F1")
    ax.axhline(
        cfg.train.min_recall,
        ls=":",
        color=RED,
        alpha=0.7,
        label=f"recall floor={cfg.train.min_recall}",
    )
    ax.axvline(chosen, ls="--", color="black", label=f"chosen thr={chosen:.3f}")
    f1_thr = float(grid[int(np.argmax(f1_g))])
    ax.axvline(
        f1_thr, ls="--", color="grey", alpha=0.7, label=f"max-F1 thr={f1_thr:.3f}"
    )
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("score (validation)")
    ax.set_title("Threshold trade-off: recall-first vs max-F1")
    ax.legend(loc="center left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_tradeoff.png")
    plt.close(fig)

    # ---- 6. Confusion matrix ---------------------------------------------
    cm = confusion_matrix(yt, (tp >= chosen).astype(int))
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred legit", "pred fraud"])
    ax.set_yticks([0, 1], ["true legit", "true fraud"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    ax.set_title(f"Confusion matrix (test) @ thr={chosen:.3f}")
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png")
    plt.close(fig)

    # ---- 7. Feature importance -------------------------------------------
    gain = model.get_booster().get_score(importance_type="gain")
    imp = pd.Series(gain).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("gain")
    ax.set_title("Top-15 feature importance (gain)")
    fig.tight_layout()
    fig.savefig(OUT / "feature_importance.png")
    plt.close(fig)

    print(f"curves done: thr={chosen:.4f} auprc={auprc:.4f} auc={auc:.4f}")
    if args.fast:
        print("--fast: skipping figs 8 (spw sweep) & 9 (seed variance)")
        return 0

    # ---- 8. scale_pos_weight sweep (LIVE) --------------------------------
    t0 = time.perf_counter()
    auprc_spw = []
    for spw in SPW_GRID:
        m = build_xgb(cfg.train, scale_pos_weight=spw)
        m.fit(train[feats], train[TARGET_COLUMN])
        auprc_spw.append(
            average_precision_score(yt, m.predict_proba(test[feats])[:, 1])
        )
        print(f"  spw={spw:<4} test_auprc={auprc_spw[-1]:.4f}")
    chosen_spw = cfg.train.scale_pos_weight
    chosen_auprc = (
        auprc_spw[SPW_GRID.index(chosen_spw)] if chosen_spw in SPW_GRID else None
    )
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(SPW_GRID, auprc_spw, "o-", color=BLUE)
    if chosen_auprc is not None:
        ax.scatter(
            [chosen_spw],
            [chosen_auprc],
            color=RED,
            zorder=5,
            s=80,
            label=f"chosen ({chosen_spw:g})",
        )
        ax.legend()
    ax.set_xscale("log")
    ax.set_xlabel("scale_pos_weight (log)")
    ax.set_ylabel("test AUPRC")
    # Fixed, wider y-range so the near-flat trend is read honestly (not zoomed).
    ax.set_ylim(0.75, 0.90)
    ax.set_title("scale_pos_weight has little effect on AUPRC")
    fig.tight_layout()
    fig.savefig(OUT / "spw_vs_auprc.png")
    plt.close(fig)
    print(f"  [fig 8] spw sweep in {time.perf_counter() - t0:.0f}s")

    # ---- 9. Single-split variance (LIVE: real preprocess + recall-first) --
    t0 = time.perf_counter()
    prec_s, rec_s = [], []
    for seed in VARIANCE_SEEDS:
        with TemporaryDirectory() as d:
            paths = preprocess(
                raw,
                replace(cfg.preprocess, random_seed=seed),
                out_dir=Path(d),
                scaler_path=Path(d) / "s.pkl",
            )
            tr = pd.read_parquet(paths["train"])
            va = pd.read_parquet(paths["val"])
            te = pd.read_parquet(paths["test"])
        m = build_xgb(replace(cfg.train, random_seed=seed))
        m.fit(tr[feats], tr[TARGET_COLUMN])
        thr = recall_first_threshold(
            va[TARGET_COLUMN].values,
            m.predict_proba(va[feats])[:, 1],
            cfg.train.min_recall,
        )
        pte = m.predict_proba(te[feats])[:, 1]
        pred = (pte >= thr).astype(int)
        yte = te[TARGET_COLUMN].values
        tp_ = ((pred == 1) & (yte == 1)).sum()
        fp_ = ((pred == 1) & (yte == 0)).sum()
        fn_ = ((pred == 0) & (yte == 1)).sum()
        prec_s.append(tp_ / (tp_ + fp_ + 1e-9))
        rec_s.append(tp_ / (tp_ + fn_ + 1e-9))
        print(f"  seed={seed:<4} precision={prec_s[-1]:.3f} recall={rec_s[-1]:.3f}")
    x = np.arange(len(VARIANCE_SEEDS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(x - w / 2, prec_s, w, color=BLUE, label="precision")
    ax.bar(x + w / 2, rec_s, w, color=RED, label="recall")
    ax.axhline(0.82, ls=":", color=BLUE, alpha=0.7, label="precision target 0.82")
    ax.axhline(0.78, ls=":", color=RED, alpha=0.7, label="recall target 0.78")
    ax.set_xticks(x, [f"seed {s}" for s in VARIANCE_SEEDS])
    ax.set_ylabel("score (test)")
    ax.set_title("Single-split variance: precision/recall swing per seed")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "seed_variance.png")
    plt.close(fig)
    print(f"  [fig 9] seed variance in {time.perf_counter() - t0:.0f}s")

    print("figures written to", OUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
