#!/usr/bin/env python3
"""Train and evaluate C3FAR-Sense on the frozen L=128 release dataset."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from c3far_simulation import (
    bilinear_features,
    fit_kmeans_gate,
    metric_row,
    thresholds_from_groups,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("release_dataset_demo_metrics.csv"))
    parser.add_argument("--alpha", type=float, default=0.10)
    args = parser.parse_args()

    dataset = np.load(args.features, allow_pickle=False)
    evidence = dataset["evidence"].astype(np.float64)
    context = dataset["context"].astype(np.float64)
    label = dataset["label"]
    split = dataset["split_code"]
    environment = dataset["environment_code"]
    z = bilinear_features(evidence, context)

    train = split == 0
    matched_cal = split == 1
    stress_refresh = split == 2
    matched_test = split == 3
    stress_test = split == 4
    assert np.all(label[matched_cal] == 0)
    assert np.all(label[stress_refresh] == 0)

    scorer = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.015, max_iter=700, solver="lbfgs", random_state=260908),
    )
    scorer.fit(z[train], label[train])
    supervised_gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=2.0, max_iter=500, random_state=260925),
    )
    supervised_gate.fit(context[train], environment[train])
    kmeans_gate = fit_kmeans_gate(context[train], 260937)

    rows = []
    for split_name, cal_mask, test_mask in (
        ("matched", matched_cal, matched_test),
        ("stress", stress_refresh, stress_test),
    ):
        cal_scores = scorer.predict_proba(z[cal_mask])[:, 1]
        test_scores = scorer.predict_proba(z[test_mask])[:, 1]
        test_labels = label[test_mask]
        global_tau, _ = thresholds_from_groups(
            cal_scores, np.zeros(cal_mask.sum(), dtype=int),
            np.zeros(test_mask.sum(), dtype=int), args.alpha,
        )
        supervised_tau, _ = thresholds_from_groups(
            cal_scores, supervised_gate.predict(context[cal_mask]),
            supervised_gate.predict(context[test_mask]), args.alpha,
        )
        kmeans_tau, _ = thresholds_from_groups(
            cal_scores, kmeans_gate.predict(context[cal_mask]),
            kmeans_gate.predict(context[test_mask]), args.alpha,
        )
        oracle_tau, _ = thresholds_from_groups(
            cal_scores, environment[cal_mask], environment[test_mask], args.alpha,
        )
        for method, tau in (
            ("Adaptive global", global_tau),
            ("KMeans conformal", kmeans_tau),
            ("C3FAR-Sense", supervised_tau),
            ("Oracle conformal", oracle_tau),
        ):
            metrics = metric_row(test_labels, test_scores, test_scores > tau)
            rows.append({"split": split_name, "method": method, **metrics})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
