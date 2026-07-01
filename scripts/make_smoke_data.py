"""Write the tiny synthetic dataset used by the secret-free CI smokes.

Generates a well-separated, creditcard-schema CSV (Time, Amount, V1..V28,
Class) for the isolated ``e2e`` dataset profile, so the e2e / retrain / monitor
workflows can run the real pipeline with no Kaggle credentials and no network.

    python scripts/make_smoke_data.py [--seed N] [--drift]

``--drift`` shifts the feature means so the file can stand in for a *drifted*
production window (used by the monitoring smoke).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "raw" / "e2e_smoke.csv"
N_ROWS = 4000
N_FRAUD = 80


def write_dataset(seed: int = 7, drift: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    labels = np.r_[np.zeros(N_ROWS - N_FRAUD, int), np.ones(N_FRAUD, int)]
    rng.shuffle(labels)
    # A drifted window: shift every feature so Evidently flags distribution drift.
    shift = 2.5 if drift else 0.0
    data: dict[str, np.ndarray] = {
        "Time": rng.uniform(0, 172_800, N_ROWS),
        "Amount": rng.lognormal(3.0 + shift, 1.0, N_ROWS),
    }
    for i in range(1, 29):  # strong fraud signal -> healthy, promotable model
        col = rng.standard_normal(N_ROWS) + shift
        col[labels == 1] += rng.uniform(1.5, 3.0)
        data[f"V{i}"] = col
    data["Class"] = labels
    cols = ["Time", "Amount", *[f"V{i}" for i in range(1, 29)], "Class"]
    DEST.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data)[cols].to_csv(DEST, index=False)
    print(f"wrote {DEST} ({N_ROWS} rows, {N_FRAUD} frauds, drift={drift})", flush=True)
    return DEST


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic smoke dataset.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--drift", action="store_true")
    args = parser.parse_args()
    write_dataset(seed=args.seed, drift=args.drift)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
