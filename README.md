# C3FAR-Sense reproducibility package

This archive accompanies **“C3FAR-Sense: Context-conditioned conformal false-alarm control for short-window spectrum sensing under receiver mismatch.”** 
It contains the frozen paired-I/Q data, executable Python source, all ten-seed result tables, and dataset preparation for VVIMP-style spectrum sensing using RadioML2016.10b.

## Scientific status

- All I/Q observations are **computer generated**, not over-the-air measurements.
- The complete paper study uses independent seeds 1–10 and native window lengths `L = 64, 128, 256`.
- The frozen `L = 128` release dataset contains 15,000 paired sensing/reference windows for inspection and downstream model development. It is smaller than the full procedural study and does not replace the ten-seed result matrix.
- Calibration sets contain trusted `H0` windows only. Test labels are never used to choose decision thresholds.
- The empirical evidence does not establish robustness to arbitrary drift, contaminated references, dependent overlapping windows, or receiver regimes absent from calibration.

## Package contents

| Path | Contents |
| --- | --- |
| `manuscript/` | Editable DOCX and rendered PDF |
| `data/` | Raw I/Q archive, 43+10 feature archive, row metadata, manifest, and frozen-dataset demo metrics |
| `code/` | Generator, detector, evaluation, plotting, validation, latency, and manuscript-building scripts |
| `results/` | Per-seed tables, 95% confidence-interval summaries, paired tests, manifests, and diagnostic artifacts |
| `figures/` | Ten publication figures used in the manuscript |
| `validation/` | Final verification report |

## Dataset schema

`data/c3far_sense_raw_iq_L128_v1.npz` stores `sensing_iq` and `reference_iq` as `float32` arrays with shape `(15000, 128, 2)`, where the last axis is I then Q. It also stores labels, split codes, SNR, environment, modulation, uncertainty, reference mismatch, and stable sample identifiers.

`data/c3far_sense_features_L128_v1.npz` stores 43 evidence features and 10 reference-context features with the same identifiers. The 430 evidence-context products are generated on demand by `bilinear_features()`. `dataset_metadata.csv` is a flat row-level view; `dataset_manifest.json` defines the schemas, code maps, file sizes, and SHA-256 hashes.

Frozen split sizes are: 6,000 matched training pairs; 1,500 matched `H0` calibration pairs; 1,500 stress `H0` refresh pairs; 3,000 matched test pairs; and 3,000 stress test pairs.

## Environment

Use Python 3.12 and install the pinned packages:

```bash
python -m pip install -r requirements.txt
```

## Verify the supplied artifacts

Run from the extracted package directory:

```bash
python code/validate_reproducibility.py --data-dir data --results-dir results
python code/evaluate_release_dataset.py \
  --features data/c3far_sense_features_L128_v1.npz \
  --out reproduced_demo_metrics.csv
```

The validator checks all dataset hashes and shapes, recomputes all 53 stored features for a deterministic 128-row subset directly from raw I/Q, and confirms that the paper results cover ten seeds and all three window lengths.

## Regenerate data and experiments

Fast end-to-end installation check:

```bash
python code/reproduce.py --quick --workspace reproduced_quick
```

Complete paper-scale run:

```bash
python code/reproduce.py --workspace reproduced_paper
```

The complete run regenerates the procedural dataset, frozen-dataset demonstration, ten-seed result matrix, CPU latency table, publication figures, and validation output. Runtime depends on processor speed and BLAS threading.

Individual commands are also available:

```bash
python code/generate_release_dataset.py --out regenerated_data
python code/c3far_simulation.py --mode paper --out regenerated_results
python code/benchmark_latency.py --out regenerated_results/latency.csv
python code/create_manuscript_figures.py \
  --results regenerated_results \
  --latency regenerated_results/latency.csv \
  --out regenerated_figures
```

Rebuild the editable manuscript from the supplied tables and figures:

```bash
python code/build_manuscript.py \
  --results results \
  --figures figures \
  --latency results/latency.csv \
  --output manuscript/C3FAR-Sense_rebuilt.docx
```

## Result lineage

- `runs_*.csv`: one row per seed and condition.
- `summary_*.csv`: mean and two-sided 95% Student-t interval across independent seeds.
- `paired_comparisons.csv`: paired differences, confidence intervals, paired t-tests, Holm-adjusted p-values, and paired standardized effect sizes.
- `run_manifest.json`: seeds, sample sizes, SNR grid, receiver environments, and false-alarm targets.
- `artifacts.json`: feature names, learned-model sizes, thresholds, and pre-aggregation diagnostics.

At `L = 128`, refreshed C3FAR-Sense achieves stress-test `Pf = 0.0979 ± 0.0024` and `Pd = 0.4438 ± 0.0136`, versus `Pd = 0.3555 ± 0.0241` for refreshed global calibration. The paired gain is `0.0884 ± 0.0196` with Holm-adjusted `p = 1.80e-5`. The label-free K-means gate is statistically indistinguishable from the supervised gate in this simulation (`Pd` difference `-0.0006 ± 0.0014`, Holm-adjusted `p = 0.750`). Reference-mismatch and contamination sweeps are reported as boundary tests, including the failure of the exploratory contamination guard to restore nominal false-alarm control.

## Submission checklist

Before journal submission, replace the author, affiliation, corresponding-author, funding, and CRediT placeholders in the manuscript. Deposit the release package in a durable research repository and insert its DOI or permanent URL in the Data availability statement. Hardware or over-the-air validation is still required before making deployment claims.
