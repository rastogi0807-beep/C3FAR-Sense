#!/usr/bin/env python3
"""Batch latency benchmark for the frozen C3FAR-Sense detector."""

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from c3far_simulation import (
    bilinear_features,
    conformal_threshold,
    extract_features,
    generate_dataset,
)


def median_time(fn, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("research/results_adaptive/latency.csv"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for length in (64, 128, 256):
        rng = np.random.default_rng(8800 + length)
        ytr, rtr, ltr, mtr = generate_dataset(rng, 12000, length, "train")
        ftr, ctr, _, _, _ = extract_features(ytr, rtr)
        btr = bilinear_features(ftr, ctr)
        scorer = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.015, max_iter=700, solver="lbfgs", random_state=3),
        ).fit(btr, ltr)
        gate = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=2.0, max_iter=500, random_state=3),
        ).fit(ctr, mtr["env_idx"])
        ycal, rcal, _, _ = generate_dataset(rng, 3000, length, "cal", h0_only=True)
        fcal, ccal, _, _, _ = extract_features(ycal, rcal)
        bcal = bilinear_features(fcal, ccal)
        scal = scorer.predict_proba(bcal)[:, 1]
        gcal = gate.predict(ccal)
        thresholds = {int(g): conformal_threshold(scal[gcal == g], 0.10) for g in np.unique(gcal)}
        yte, rte, _, _ = generate_dataset(rng, 4096, length, "test")

        cache = {}
        def feature_call():
            f, c, _, _, _ = extract_features(yte, rte)
            cache["f"] = f; cache["c"] = c; cache["b"] = bilinear_features(f, c)
        feature_call()
        t_feat = median_time(feature_call, 7)

        def decision_call():
            score = scorer.predict_proba(cache["b"])[:, 1]
            group = gate.predict(cache["c"])
            tau = np.array([thresholds[int(g)] for g in group])
            return score > tau
        decision_call()
        t_decision = median_time(decision_call, 30)
        rows.append({
            "length": length,
            "batch": 4096,
            "feature_ms_per_sample": 1000 * t_feat / 4096,
            "decision_ms_per_sample": 1000 * t_decision / 4096,
            "total_ms_per_sample": 1000 * (t_feat + t_decision) / 4096,
        })
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
