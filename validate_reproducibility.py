#!/usr/bin/env python3
"""Validate dataset integrity and the minimum paper-result contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from c3far_simulation import extract_features


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.data_dir / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_path = args.data_dir / next(name for name in manifest["files"] if "raw_iq" in name)
    feature_path = args.data_dir / next(name for name in manifest["files"] if "features" in name)
    for name, record in manifest["files"].items():
        path = args.data_dir / name
        assert path.exists(), f"Missing dataset file: {name}"
        assert path.stat().st_size == record["bytes"], f"Size mismatch: {name}"
        assert sha256(path) == record["sha256"], f"Checksum mismatch: {name}"

    raw = np.load(raw_path, allow_pickle=False)
    features = np.load(feature_path, allow_pickle=False)
    n = manifest["n_samples"]
    length = manifest["window_length"]
    assert raw["sensing_iq"].shape == (n, length, 2)
    assert raw["reference_iq"].shape == (n, length, 2)
    assert features["evidence"].shape == (n, 43)
    assert features["context"].shape == (n, 10)
    assert np.array_equal(raw["sample_id"], features["sample_id"])
    assert np.array_equal(raw["label"], features["label"])
    assert np.array_equal(raw["split_code"], features["split_code"])
    assert set(np.unique(raw["label"])) == {0, 1}
    assert set(np.unique(raw["split_code"])) == {0, 1, 2, 3, 4}

    # Recompute a deterministic subset from raw I/Q to verify feature lineage.
    indices = np.linspace(0, n - 1, 128, dtype=int)
    y = raw["sensing_iq"][indices, :, 0] + 1j * raw["sensing_iq"][indices, :, 1]
    r = raw["reference_iq"][indices, :, 0] + 1j * raw["reference_iq"][indices, :, 1]
    evidence, context, _, _, _ = extract_features(y.astype(np.complex64), r.astype(np.complex64))
    np.testing.assert_allclose(evidence, features["evidence"][indices], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(context, features["context"][indices], rtol=2e-5, atol=2e-5)

    required_results = (
        "runs_global.csv", "summary_global.csv", "summary_alpha_sweep.csv",
        "summary_modulation.csv", "summary_predicted_group.csv",
        "summary_reference_mismatch.csv", "summary_reference_contamination.csv",
        "paired_comparisons.csv", "run_manifest.json",
    )
    for name in required_results:
        assert (args.results_dir / name).exists(), f"Missing result file: {name}"
    runs = pd.read_csv(args.results_dir / "runs_global.csv")
    assert sorted(runs["seed"].unique().tolist()) == list(range(1, 11))
    assert sorted(runs["length"].unique().tolist()) == [64, 128, 256]
    assert not runs[["auc", "pf", "pd", "balanced_accuracy"]].isna().any().any()
    assert runs[["auc", "pf", "pd", "balanced_accuracy"]].apply(
        lambda column: column.between(0, 1).all()
    ).all()

    report = {
        "dataset_files_verified": len(manifest["files"]),
        "dataset_samples": n,
        "feature_rows_recomputed": len(indices),
        "paper_seeds": 10,
        "paper_lengths": [64, 128, 256],
        "status": "PASS",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
