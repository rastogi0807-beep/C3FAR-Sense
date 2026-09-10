# C3FAR-Sense reproducibility package

This archive accompanies **“C3FAR-Sense: Context-conditioned conformal false-alarm control for short-window spectrum sensing under receiver mismatch.”** 
It contains the frozen paired-I/Q data, executable Python source, all ten-seed result tables, and dataset preparation for VVIMP-style spectrum sensing using RadioML2016.10b.

## Scientific status

- All I/Q observations are **computer generated**, not over-the-air measurements.
- The complete paper study uses independent seeds 1–10 and native window lengths `L = 64, 128, 256`.
- The frozen `L = 128` release dataset contains 15,000 paired sensing/reference windows for inspection and downstream model development. It is smaller than the full procedural study and does not replace the ten-seed result matrix.
- Calibration sets contain trusted `H0` windows only. Test labels are never used to choose decision thresholds.
- The empirical evidence does not establish robustness to arbitrary drift, contaminated references, dependent overlapping windows, or receiver regimes absent from calibration.

## Dataset schema

`data/c3far_sense_raw_iq_L128_v1.npz` stores `sensing_iq` and `reference_iq` as `float32` arrays with shape `(15000, 128, 2)`, where the last axis is I then Q. It also stores labels, split codes, SNR, environment, modulation, uncertainty, reference mismatch, and stable sample identifiers.

`data/c3far_sense_features_L128_v1.npz` stores 43 evidence features and 10 reference-context features with the same identifiers. The 430 evidence-context products are generated on demand by `bilinear_features()`. `dataset_metadata.csv` is a flat row-level view; `dataset_manifest.json` defines the schemas, code maps, file sizes, and SHA-256 hashes.

Frozen split sizes are: 6,000 matched training pairs; 1,500 matched `H0` calibration pairs; 1,500 stress `H0` refresh pairs; 3,000 matched test pairs; and 3,000 stress test pairs.

### Download RadioML2016.10b

Download `RML2016.10b.dat` and place it here:

```bash
<DATASETS_DIR>/raw/RML2016.10b.dat
```

Note: this repository **does not include** RadioML2016.10b (or any derived `.npy/.npz` files). Please obtain the dataset
from the official source and follow its license terms. Keep `DATASETS_DIR` outside the git repo to avoid accidentally
committing data.

Dataset source (Kaggle):

```text
https://www.kaggle.com/datasets/marwanabudeeb/rml201610b/data
```

Optional: Kaggle CLI (requires Kaggle credentials). After you have `~/.kaggle/kaggle.json` configured, you can let the
prep script attempt the download:

```bash
python3 -m scripts.prepare_radioml2016_10b --try-kaggle
```

### 3) Prepare VVIMP-style C3FAR-Sense datasets (H1 + generated H0)

```bash
cd "<CODE_REPO_DIR>"
python3 -m scripts.prepare_radioml2016_10b --overwrite
```

Outputs (for lengths `L ∈ {64,128,256}`):
- `DATASETS_DIR/processed/vvimp_radioml2016_10b_len<L>/` (train/val/test)

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

