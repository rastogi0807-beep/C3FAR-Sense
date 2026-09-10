#!/usr/bin/env python3
"""Generate the frozen C3FAR-Sense v1 reproducibility dataset.

The paper's full Monte Carlo study is procedurally generated from documented
seeds.  This script additionally creates a compact, fixed L=128 release with
raw paired I/Q windows, machine-readable metadata, and precomputed 43+10
features so that readers can inspect or retrain the detector immediately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from c3far_simulation import (
    ENVIRONMENTS,
    MODULATIONS,
    STRESS_ENVIRONMENTS,
    extract_features,
    generate_dataset,
)


SPLITS = (
    ("train_matched", 6000, "train", False),
    ("calibration_h0_matched", 1500, "cal", True),
    ("refresh_h0_stress", 1500, "stress", True),
    ("test_matched", 3000, "test", False),
    ("test_stress", 3000, "stress", False),
)
GLOBAL_ENVIRONMENTS = ENVIRONMENTS + STRESS_ENVIRONMENTS


def as_iq(values: np.ndarray) -> np.ndarray:
    return np.stack([values.real, values.imag], axis=-1).astype(np.float32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("research/release_dataset"))
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=260908)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed_sequence = np.random.SeedSequence(args.seed)
    split_seeds = seed_sequence.spawn(len(SPLITS))
    raw_y, raw_r, labels, split_codes = [], [], [], []
    evidence_parts, context_parts = [], []
    metadata = {name: [] for name in (
        "snr_db", "effective_snr_db", "environment_code", "modulation_code",
        "uncertainty_db", "reference_delta_db",
    )}
    feature_names = context_names = None

    for split_code, ((split_name, count, generator_split, h0_only), child_seed) in enumerate(zip(SPLITS, split_seeds)):
        rng = np.random.default_rng(child_seed)
        y, r, label, meta = generate_dataset(
            rng, count, args.length, generator_split, h0_only=h0_only
        )
        evidence, context, _, feature_names, context_names = extract_features(y, r)
        raw_y.append(as_iq(y))
        raw_r.append(as_iq(r))
        labels.append(label.astype(np.int8))
        split_codes.append(np.full(count, split_code, dtype=np.int8))
        evidence_parts.append(evidence.astype(np.float32))
        context_parts.append(context.astype(np.float32))
        metadata["snr_db"].append(meta["snr_db"].astype(np.float32))
        metadata["effective_snr_db"].append(meta["effective_snr_db"].astype(np.float32))
        local_names = [str(value) for value in meta["env_names"]]
        environment_code = np.array(
            [GLOBAL_ENVIRONMENTS.index(local_names[int(idx)]) for idx in meta["env_idx"]],
            dtype=np.int8,
        )
        metadata["environment_code"].append(environment_code)
        metadata["modulation_code"].append(meta["mod_idx"].astype(np.int8))
        metadata["uncertainty_db"].append(meta["uncertainty_db"].astype(np.float32))
        metadata["reference_delta_db"].append(meta["ref_delta_db"].astype(np.float32))

    arrays = {
        "sample_id": np.arange(sum(item[1] for item in SPLITS), dtype=np.int32),
        "sensing_iq": np.concatenate(raw_y),
        "reference_iq": np.concatenate(raw_r),
        "label": np.concatenate(labels),
        "split_code": np.concatenate(split_codes),
        **{key: np.concatenate(value) for key, value in metadata.items()},
    }
    evidence = np.concatenate(evidence_parts)
    context = np.concatenate(context_parts)

    raw_path = args.out / f"c3far_sense_raw_iq_L{args.length}_v1.npz"
    feature_path = args.out / f"c3far_sense_features_L{args.length}_v1.npz"
    np.savez_compressed(raw_path, **arrays)
    np.savez_compressed(
        feature_path,
        sample_id=arrays["sample_id"],
        evidence=evidence,
        context=context,
        label=arrays["label"],
        split_code=arrays["split_code"],
        snr_db=arrays["snr_db"],
        effective_snr_db=arrays["effective_snr_db"],
        environment_code=arrays["environment_code"],
        modulation_code=arrays["modulation_code"],
        uncertainty_db=arrays["uncertainty_db"],
        reference_delta_db=arrays["reference_delta_db"],
        evidence_feature_names=np.asarray(feature_names, dtype="U64"),
        context_feature_names=np.asarray(context_names, dtype="U64"),
    )

    preview_path = args.out / "dataset_metadata.csv"
    with preview_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "split", "label", "snr_db", "effective_snr_db",
            "environment", "modulation", "uncertainty_db", "reference_delta_db",
        ])
        for idx in range(len(arrays["sample_id"])):
            writer.writerow([
                int(arrays["sample_id"][idx]),
                SPLITS[int(arrays["split_code"][idx])][0],
                int(arrays["label"][idx]),
                float(arrays["snr_db"][idx]),
                "" if np.isnan(arrays["effective_snr_db"][idx]) else float(arrays["effective_snr_db"][idx]),
                GLOBAL_ENVIRONMENTS[int(arrays["environment_code"][idx])],
                MODULATIONS[int(arrays["modulation_code"][idx])],
                float(arrays["uncertainty_db"][idx]),
                float(arrays["reference_delta_db"][idx]),
            ])

    manifest = {
        "dataset_name": "C3FAR-Sense Paired-IQ Benchmark v1",
        "version": "1.0.0",
        "generator_seed": args.seed,
        "window_length": args.length,
        "n_samples": int(len(arrays["label"])),
        "splits": {name: count for name, count, _, _ in SPLITS},
        "split_codes": {str(index): item[0] for index, item in enumerate(SPLITS)},
        "labels": {"0": "H0 / idle", "1": "H1 / occupied"},
        "environment_codes": {str(index): name for index, name in enumerate(GLOBAL_ENVIRONMENTS)},
        "modulation_codes": {str(index): name for index, name in enumerate(MODULATIONS)},
        "raw_schema": {
            "sensing_iq": ["sample", "time", "I/Q"],
            "reference_iq": ["sample", "time", "I/Q"],
            "dtype": "float32",
        },
        "feature_schema": {
            "evidence_shape": list(evidence.shape),
            "context_shape": list(context.shape),
            "evidence_feature_names": feature_names,
            "context_feature_names": context_names,
            "note": "The 430 bilinear interactions are generated on demand to avoid redundant storage.",
        },
        "files": {},
    }
    for path in (raw_path, feature_path, preview_path):
        manifest["files"][path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest_path = args.out / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({
        "raw": str(raw_path),
        "features": str(feature_path),
        "metadata": str(preview_path),
        "manifest": str(manifest_path),
    }, indent=2))


if __name__ == "__main__":
    main()
