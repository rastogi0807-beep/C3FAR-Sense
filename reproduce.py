#!/usr/bin/env python3
"""One-command reproduction of data, experiments, figures, and checks."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run a one-seed smoke test instead of the paper matrix")
    parser.add_argument("--workspace", type=Path, default=Path("reproduced"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    workspace = args.workspace.resolve()
    data = workspace / "data"
    results = workspace / ("smoke_results" if args.quick else "results")
    figures = workspace / "figures"
    data.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    run([sys.executable, "generate_release_dataset.py", "--out", str(data)], root)
    feature_path = data / "c3far_sense_features_L128_v1.npz"
    run([
        sys.executable, "evaluate_release_dataset.py", "--features", str(feature_path),
        "--out", str(results / "release_dataset_demo_metrics.csv"),
    ], root)
    run([
        sys.executable, "c3far_simulation.py", "--mode", "quick" if args.quick else "paper",
        "--out", str(results),
    ], root)
    if not args.quick:
        latency_path = results / "latency.csv"
        run([sys.executable, "benchmark_latency.py", "--out", str(latency_path)], root)
        run([
            sys.executable, "create_manuscript_figures.py", "--results", str(results),
            "--latency", str(latency_path), "--out", str(figures),
        ], root)
        run([
            sys.executable, "validate_reproducibility.py", "--data-dir", str(data),
            "--results-dir", str(results),
        ], root)
    print(f"Reproduction complete: {workspace}")


if __name__ == "__main__":
    main()
