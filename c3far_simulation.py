#!/usr/bin/env python3
"""Reproducible Monte Carlo study for C3FAR-Sense.

The script generates independent short complex-baseband sensing windows and
paired noise-reference windows under modulation, channel, RF-impairment, and
noise diversity.  It trains compact feature-domain classifiers, calibrates
false-alarm thresholds on a disjoint H0 set, and exports all paper tables and
figures.  No test labels are used for threshold selection.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.stats import t as student_t
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODULATIONS = (
    "BPSK", "QPSK", "8PSK", "PAM4", "QAM16", "QAM64",
    "CPFSK", "GFSK", "AM-DSB", "WBFM",
)
ENVIRONMENTS = ("white", "colored", "impulsive")
STRESS_ENVIRONMENTS = ("colored-severe", "impulsive-severe", "student-t")
SNR_GRID = np.array([-20, -16, -12, -8, -4, 0], dtype=float)
ALPHA_GRID = np.array([0.01, 0.05, 0.10, 0.20], dtype=float)


@dataclass(frozen=True)
class StudySize:
    n_train: int
    n_cal_h0: int
    n_adapt_h0: int
    n_test: int
    n_stress: int
    mlp_iter: int


def complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def qam_symbols(rng: np.random.Generator, n: int, order: int) -> np.ndarray:
    side = int(np.sqrt(order))
    levels = np.arange(-(side - 1), side, 2, dtype=float)
    i = rng.choice(levels, size=n)
    q = rng.choice(levels, size=n)
    x = i + 1j * q
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


def smooth_symbols(symbols: np.ndarray, sps: int, length: int) -> np.ndarray:
    x = np.repeat(symbols, sps)[: length + 12]
    # A compact pulse-shaping surrogate.  The normalized triangular kernel
    # suppresses discontinuities without imposing a dataset-specific filter.
    kernel = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    y = np.convolve(x, kernel, mode="same")
    return y[:length]


def make_signal(rng: np.random.Generator, modulation: str, length: int) -> np.ndarray:
    sps = int(rng.integers(3, 7))
    ns = int(np.ceil((length + 16) / sps))
    if modulation in {"BPSK", "QPSK", "8PSK"}:
        order = {"BPSK": 2, "QPSK": 4, "8PSK": 8}[modulation]
        idx = rng.integers(0, order, size=ns)
        sym = np.exp(1j * 2 * np.pi * idx / order)
        x = smooth_symbols(sym, sps, length)
    elif modulation == "PAM4":
        sym = rng.choice(np.array([-3.0, -1.0, 1.0, 3.0]), size=ns) / np.sqrt(5.0)
        x = smooth_symbols(sym.astype(complex), sps, length)
        x *= np.exp(1j * rng.uniform(0, 2 * np.pi))
    elif modulation in {"QAM16", "QAM64"}:
        sym = qam_symbols(rng, ns, 16 if modulation == "QAM16" else 64)
        x = smooth_symbols(sym, sps, length)
    elif modulation in {"CPFSK", "GFSK"}:
        bits = rng.choice(np.array([-1.0, 1.0]), size=ns)
        freq = np.repeat(bits, sps)[:length]
        if modulation == "GFSK":
            g = np.exp(-0.5 * (np.linspace(-2.5, 2.5, 11) / 0.85) ** 2)
            g /= g.sum()
            freq = np.convolve(freq, g, mode="same")
        h = rng.uniform(0.35, 0.65)
        phase = np.cumsum(np.pi * h * freq / sps)
        x = np.exp(1j * phase)
    elif modulation == "AM-DSB":
        msg = lfilter([1.0], [1.0, -0.94], rng.standard_normal(length + 20))[20:]
        msg /= np.std(msg) + 1e-12
        x = (1.0 + 0.55 * np.tanh(msg)) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    elif modulation == "WBFM":
        msg = lfilter([1.0], [1.0, -0.90], rng.standard_normal(length + 20))[20:]
        msg /= np.std(msg) + 1e-12
        x = np.exp(1j * np.cumsum(0.34 * msg))
    else:
        raise ValueError(modulation)
    x = np.asarray(x[:length], dtype=np.complex128)
    x -= np.mean(x)
    return x / np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)


def apply_channel_and_rf(
    rng: np.random.Generator,
    x: np.ndarray,
    stress: bool,
) -> np.ndarray:
    length = x.size
    # Random flat/Rician or short frequency-selective channel.
    if rng.random() < 0.55:
        h0 = complex_normal(rng, (1,))[0]
        h0 /= max(abs(h0), 0.25)
        y = h0 * x
    else:
        taps = np.array([
            1.0,
            rng.uniform(0.10, 0.45) * np.exp(1j * rng.uniform(0, 2 * np.pi)),
            rng.uniform(0.04, 0.25) * np.exp(1j * rng.uniform(0, 2 * np.pi)),
        ])
        taps /= np.sqrt(np.sum(np.abs(taps) ** 2))
        y = np.convolve(x, taps, mode="full")[:length]
    cfo_lim = 0.055 if stress else 0.025
    cfo = rng.uniform(-cfo_lim, cfo_lim)
    phase = rng.uniform(0, 2 * np.pi)
    y *= np.exp(1j * (2 * np.pi * cfo * np.arange(length) + phase))
    gain_lim = 0.14 if stress else 0.07
    phase_lim = np.deg2rad(9 if stress else 4)
    gain = rng.uniform(-gain_lim, gain_lim)
    phi = rng.uniform(-phase_lim, phase_lim)
    mu = np.cos(phi / 2) + 1j * gain * np.sin(phi / 2)
    nu = gain * np.cos(phi / 2) - 1j * np.sin(phi / 2)
    return mu * y + nu * np.conj(y)


def make_noise(
    rng: np.random.Generator,
    length: int,
    environment: str,
    power: float,
) -> np.ndarray:
    if environment == "white":
        w = complex_normal(rng, (length,))
    elif environment in {"colored", "colored-severe"}:
        rho = rng.uniform(0.25, 0.62) if environment == "colored" else rng.uniform(0.72, 0.90)
        e = complex_normal(rng, (length,))
        w = np.empty(length, dtype=complex)
        w[0] = e[0]
        innovation = np.sqrt(max(1.0 - rho * rho, 1e-6))
        for n in range(1, length):
            w[n] = rho * w[n - 1] + innovation * e[n]
    elif environment in {"impulsive", "impulsive-severe"}:
        severe = environment == "impulsive-severe"
        p = 0.012 if not severe else 0.045
        kappa = 12.0 if not severe else 28.0
        w = complex_normal(rng, (length,))
        mask = rng.random(length) < p
        w[mask] += np.sqrt(kappa) * complex_normal(rng, (int(mask.sum()),))
        w /= np.sqrt(1.0 + p * kappa)
    elif environment == "student-t":
        df = 3.0
        wr = rng.standard_t(df, size=length)
        wi = rng.standard_t(df, size=length)
        w = (wr + 1j * wi) / np.sqrt(2.0 * df / (df - 2.0))
    else:
        raise ValueError(environment)
    return np.sqrt(power) * w


def generate_dataset(
    rng: np.random.Generator,
    n: int,
    length: int,
    split: str,
    h0_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    stress = split == "stress"
    env_names = STRESS_ENVIRONMENTS if stress else ENVIRONMENTS
    y = np.empty((n, length), dtype=np.complex64)
    r = np.empty((n, length), dtype=np.complex64)
    labels = np.zeros(n, dtype=np.int8) if h0_only else np.tile(np.array([0, 1], dtype=np.int8), int(np.ceil(n / 2)))[:n]
    rng.shuffle(labels)
    snr = rng.choice(SNR_GRID, size=n)
    env_idx = rng.integers(0, len(env_names), size=n)
    mods = rng.integers(0, len(MODULATIONS), size=n)
    uncertainty_db = rng.uniform(-4.0, 4.0, size=n) if stress else rng.uniform(-2.0, 2.0, size=n)
    ref_delta_db = rng.normal(0.0, 0.85 if stress else 0.30, size=n)
    effective_snr = np.empty(n, dtype=float)

    for i in range(n):
        env = env_names[env_idx[i]]
        noise_power = 10.0 ** (uncertainty_db[i] / 10.0)
        ref_power = noise_power * 10.0 ** (ref_delta_db[i] / 10.0)
        wi = make_noise(rng, length, env, noise_power)
        ri = make_noise(rng, length, env, ref_power)
        if labels[i]:
            sig = make_signal(rng, MODULATIONS[mods[i]], length)
            sig = apply_channel_and_rf(rng, sig, stress)
            # Nominal signal power is tied to unit noise; actual SNR therefore
            # changes with noise uncertainty and fading, as in an uncalibrated receiver.
            sig *= np.sqrt(10.0 ** (snr[i] / 10.0))
            yi = sig + wi
            effective_snr[i] = 10 * np.log10((np.mean(np.abs(sig) ** 2) + 1e-12) / (noise_power + 1e-12))
        else:
            yi = wi
            effective_snr[i] = np.nan
        y[i] = yi.astype(np.complex64)
        r[i] = ri.astype(np.complex64)
    meta = {
        "snr_db": snr,
        "effective_snr_db": effective_snr,
        "env_idx": env_idx,
        "mod_idx": mods,
        "uncertainty_db": uncertainty_db,
        "ref_delta_db": ref_delta_db,
        "env_names": np.array(env_names, dtype=object),
    }
    return y, r, labels, meta


def _spectral_features(x: np.ndarray, bins: int = 8) -> tuple[np.ndarray, ...]:
    spec = np.abs(np.fft.fft(x, axis=1)) ** 2 + 1e-10
    p = spec / np.sum(spec, axis=1, keepdims=True)
    entropy = -np.sum(p * np.log(p), axis=1) / np.log(x.shape[1])
    flatness = np.exp(np.mean(np.log(spec), axis=1)) / np.mean(spec, axis=1)
    peak = np.max(spec, axis=1) / np.mean(spec, axis=1)
    edges = np.linspace(0, x.shape[1], bins + 1, dtype=int)
    band = np.stack([np.mean(spec[:, edges[j]:edges[j + 1]], axis=1) for j in range(bins)], axis=1)
    band /= np.sum(band, axis=1, keepdims=True) + 1e-12
    return entropy, flatness, peak, band


def extract_features(y: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    eps = 1e-10
    ey = np.mean(np.abs(y) ** 2, axis=1) + eps
    er = np.mean(np.abs(r) ** 2, axis=1) + eps
    medr = np.median(np.abs(r) ** 2, axis=1) / np.log(2.0) + eps
    medy = np.median(np.abs(y) ** 2, axis=1) + eps
    zy = y / np.sqrt(medr[:, None])
    zr = r / np.sqrt(medr[:, None])
    ay = np.abs(zy)
    ar = np.abs(zr)

    ent_y, flat_y, peak_y, band_y = _spectral_features(zy)
    ent_r, flat_r, peak_r, band_r = _spectral_features(zr)
    kurt_y = np.mean(ay ** 4, axis=1) / (np.mean(ay ** 2, axis=1) ** 2 + eps)
    kurt_r = np.mean(ar ** 4, axis=1) / (np.mean(ar ** 2, axis=1) ** 2 + eps)
    crest_y = np.max(ay ** 2, axis=1) / (np.mean(ay ** 2, axis=1) + eps)
    crest_r = np.max(ar ** 2, axis=1) / (np.mean(ar ** 2, axis=1) + eps)
    q90_y = np.quantile(ay ** 2, 0.90, axis=1)
    q90_r = np.quantile(ar ** 2, 0.90, axis=1)
    pseudo_y = np.abs(np.mean(zy ** 2, axis=1)) / (np.mean(np.abs(zy) ** 2, axis=1) + eps)
    fourth_y = np.abs(np.mean(zy ** 4, axis=1)) / (np.mean(np.abs(zy) ** 4, axis=1) + eps)
    dphi_y = np.angle(zy[:, 1:] * np.conj(zy[:, :-1]))
    phase1 = np.abs(np.mean(np.exp(1j * dphi_y), axis=1))
    phase2 = np.abs(np.mean(np.exp(2j * dphi_y), axis=1))

    feature_cols = [
        np.log(ey / er), np.log(medy / (np.median(np.abs(r) ** 2, axis=1) + eps)),
        np.log((q90_y + eps) / (q90_r + eps)),
        ent_y, ent_y - ent_r, flat_y, flat_y - flat_r,
        np.log(peak_y + eps), np.log((peak_y + eps) / (peak_r + eps)),
        kurt_y, kurt_y - kurt_r, np.log(crest_y + eps), np.log((crest_y + eps) / (crest_r + eps)),
        pseudo_y, fourth_y, phase1, phase2,
    ]
    feature_names = [
        "log_energy_ratio", "log_median_ratio", "log_q90_ratio",
        "spec_entropy_y", "spec_entropy_delta", "spec_flatness_y", "spec_flatness_delta",
        "log_spec_peak_y", "log_spec_peak_ratio", "kurtosis_y", "kurtosis_delta",
        "log_crest_y", "log_crest_ratio", "pseudo_cov", "fourth_circular",
        "phase_concentration_1", "phase_concentration_2",
    ]
    for lag in (1, 2, 4, 8, 16):
        if lag >= y.shape[1]:
            continue
        cy = np.abs(np.mean(zy[:, lag:] * np.conj(zy[:, :-lag]), axis=1)) / (np.mean(np.abs(zy) ** 2, axis=1) + eps)
        cr = np.abs(np.mean(zr[:, lag:] * np.conj(zr[:, :-lag]), axis=1)) / (np.mean(np.abs(zr) ** 2, axis=1) + eps)
        feature_cols.extend([cy, cy - cr])
        feature_names.extend([f"acf_y_lag{lag}", f"acf_delta_lag{lag}"])
    for j in range(band_y.shape[1]):
        feature_cols.extend([band_y[:, j], np.log((band_y[:, j] + 1e-6) / (band_r[:, j] + 1e-6))])
        feature_names.extend([f"bandshape_y_{j}", f"log_bandratio_{j}"])
    evidence = np.column_stack(feature_cols).astype(np.float64)

    context_cols = [
        np.log(er), ent_r, flat_r, np.log(peak_r + eps), kurt_r, np.log(crest_r + eps),
        np.mean(ar ** 2 > 8.0 * np.median(ar ** 2, axis=1)[:, None], axis=1),
    ]
    context_names = [
        "log_ref_energy", "ref_spec_entropy", "ref_spec_flatness", "log_ref_spec_peak",
        "ref_kurtosis", "log_ref_crest", "ref_impulse_fraction",
    ]
    for lag in (1, 2, 4):
        cr = np.abs(np.mean(zr[:, lag:] * np.conj(zr[:, :-lag]), axis=1)) / (np.mean(np.abs(zr) ** 2, axis=1) + eps)
        context_cols.append(cr)
        context_names.append(f"ref_acf_lag{lag}")
    context = np.column_stack(context_cols).astype(np.float64)
    combined = np.column_stack([evidence, context])
    return evidence, context, combined, feature_names, context_names


def bilinear_features(evidence: np.ndarray, context: np.ndarray) -> np.ndarray:
    """Explicit evidence-context interactions for a compact bilinear scorer."""
    interactions = (evidence[:, :, None] * context[:, None, :]).reshape(evidence.shape[0], -1)
    return np.column_stack([evidence, context, interactions])


def conformal_threshold(scores_h0: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores_h0, dtype=float))
    n = scores.size
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def predict_context_thresholds(
    train_context: np.ndarray,
    train_environment: np.ndarray,
    cal_context: np.ndarray,
    cal_scores: np.ndarray,
    test_context: np.ndarray,
    alpha: float,
    seed: int,
    n_clusters: int = 3,
) -> tuple[np.ndarray, np.ndarray, dict[int, float], object]:
    # The context gate is trained only on reference-window statistics.  At
    # deployment it needs neither a sensing-window label nor an oracle noise
    # label; unseen conditions are mapped to their closest learned context.
    gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=2.0, max_iter=500, random_state=seed),
    )
    gate.fit(train_context, train_environment)
    ccal = gate.predict(cal_context)
    ctest = gate.predict(test_context)
    global_thr = conformal_threshold(cal_scores, alpha)
    thresholds: dict[int, float] = {}
    for g in range(n_clusters):
        gs = cal_scores[ccal == g]
        # Any non-empty group uses its own finite-sample order statistic.  For
        # very small groups the conformal rule may return +inf, which is the
        # correct conservative action rather than borrowing an unguaranteed
        # threshold from another context.  The global value is used only if a
        # group receives no calibration point at all.
        thresholds[g] = conformal_threshold(gs, alpha) if gs.size else global_thr
    tau = np.array([thresholds[int(g)] for g in ctest])
    return tau, ctest, thresholds, gate


def thresholds_from_groups(
    cal_scores: np.ndarray,
    cal_groups: np.ndarray,
    test_groups: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, dict[int, float]]:
    """Return finite-sample conformal thresholds for arbitrary fixed groups."""
    cal_groups = np.asarray(cal_groups, dtype=int)
    test_groups = np.asarray(test_groups, dtype=int)
    global_thr = conformal_threshold(cal_scores, alpha)
    thresholds: dict[int, float] = {}
    for group in np.unique(np.concatenate([cal_groups, test_groups])):
        values = np.asarray(cal_scores)[cal_groups == group]
        thresholds[int(group)] = conformal_threshold(values, alpha) if values.size else global_thr
    tau = np.array([thresholds.get(int(group), global_thr) for group in test_groups], dtype=float)
    return tau, thresholds


def fit_kmeans_gate(context: np.ndarray, seed: int, n_clusters: int = 3):
    """Fit a label-free context partition used by the unsupervised ablation."""
    gate = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=n_clusters, n_init=20, random_state=seed),
    )
    gate.fit(context)
    return gate


def build_reference_contamination_bank(
    references: np.ndarray,
    seed: int,
    maximum_rate: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create nested contaminated-reference subsets for a controlled sweep."""
    rng = np.random.default_rng(seed)
    priority = rng.random(len(references))
    contaminated = references.copy()
    leak_snr_db = np.full(len(references), np.nan, dtype=float)
    for idx in np.flatnonzero(priority < maximum_rate):
        modulation = MODULATIONS[int(rng.integers(0, len(MODULATIONS)))]
        leak = make_signal(rng, modulation, references.shape[1])
        leak = apply_channel_and_rf(rng, leak, stress=True)
        leak_snr_db[idx] = rng.uniform(-8.0, 2.0)
        target_power = np.mean(np.abs(references[idx]) ** 2) * 10.0 ** (leak_snr_db[idx] / 10.0)
        leak *= np.sqrt(target_power / (np.mean(np.abs(leak) ** 2) + 1e-12))
        contaminated[idx] = references[idx] + leak.astype(np.complex64)
    return contaminated, priority, leak_snr_db


def metric_row(labels: np.ndarray, scores: np.ndarray, decisions: np.ndarray) -> dict[str, float]:
    h0 = labels == 0
    h1 = labels == 1
    pf = float(np.mean(decisions[h0]))
    pd = float(np.mean(decisions[h1]))
    return {
        "auc": float(roc_auc_score(labels, scores)),
        "pf": pf,
        "pd": pd,
        "balanced_accuracy": 0.5 * ((1.0 - pf) + pd),
    }


def run_one(seed: int, length: int, size: StudySize, outdir: Path, alpha: float = 0.10) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(1000 * seed + length)
    t0 = time.perf_counter()
    ytr, rtr, ltr, mtr = generate_dataset(rng, size.n_train, length, "train")
    ycal, rcal, lcal, mcal = generate_dataset(rng, size.n_cal_h0, length, "cal", h0_only=True)
    yada, rada, lada, mada = generate_dataset(rng, size.n_adapt_h0, length, "stress", h0_only=True)
    yte, rte, lte, mte = generate_dataset(rng, size.n_test, length, "test")
    yst, rst, lst, mst = generate_dataset(rng, size.n_stress, length, "stress")
    ftr, ctr, xtr, fn, cn = extract_features(ytr, rtr)
    fcal, ccal, xcal, _, _ = extract_features(ycal, rcal)
    fada, cada, xada, _, _ = extract_features(yada, rada)
    fte, cte, xte, _, _ = extract_features(yte, rte)
    fst, cst, xst, _, _ = extract_features(yst, rst)
    btr = bilinear_features(ftr, ctr)
    bcal = bilinear_features(fcal, ccal)
    bada = bilinear_features(fada, cada)
    bte = bilinear_features(fte, cte)
    bst = bilinear_features(fst, cst)

    signal_only_idx = [i for i, name in enumerate(fn) if "ratio" not in name and "delta" not in name]
    signal_only = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=2e-4, batch_size=256,
                      learning_rate_init=1e-3, max_iter=size.mlp_iter,
                      early_stopping=True, validation_fraction=0.12,
                      n_iter_no_change=8, random_state=seed),
    )
    signal_only.fit(ftr[:, signal_only_idx], ltr)

    proposed = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.015, max_iter=700, solver="lbfgs", random_state=seed + 91),
    )
    proposed.fit(btr, ltr)
    linear = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500, random_state=seed))
    linear.fit(xtr, ltr)

    supervised_gate = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=2.0, max_iter=500, random_state=seed + 17),
    )
    supervised_gate.fit(ctr, mtr["env_idx"])
    kmeans_gate = fit_kmeans_gate(ctr, seed + 29)

    cal_scores_prop = proposed.predict_proba(bcal)[:, 1]
    cal_scores_sig = signal_only.predict_proba(fcal[:, signal_only_idx])[:, 1]
    cal_scores_lin = linear.predict_proba(xcal)[:, 1]
    adapt_scores_prop = proposed.predict_proba(bada)[:, 1]
    adapt_scores_lin = linear.predict_proba(xada)[:, 1]
    adapt_scores_enr = xada[:, fn.index("log_energy_ratio")]
    ed_cal = np.log(np.mean(np.abs(ycal) ** 2, axis=1) + 1e-12)
    enr_cal = xcal[:, fn.index("log_energy_ratio")]
    global_thresholds = {
        "ED": conformal_threshold(ed_cal, alpha),
        "ENR-ED": conformal_threshold(enr_cal, alpha),
        "Signal-only MLP": conformal_threshold(cal_scores_sig, alpha),
        "Linear fusion": conformal_threshold(cal_scores_lin, alpha),
        "Bilinear fusion + global conformal": conformal_threshold(cal_scores_prop, alpha),
    }

    rows: list[dict] = []
    snr_rows: list[dict] = []
    env_rows: list[dict] = []
    artifacts = {
        "feature_names": fn,
        "context_names": cn,
        "thresholds": {},
        "calibration_size": [],
        "alpha_sweep": [],
        "modulation": [],
        "predicted_group": [],
        "gate_confusion": [],
        "reference_mismatch": [],
        "reference_contamination": [],
    }

    split_data = {
        "matched": (yte, rte, lte, mte, fte, cte, xte, bte),
        "stress": (yst, rst, lst, mst, fst, cst, xst, bst),
    }
    for split, (yy, rr, lab, meta, ff, cc, xx, bb) in split_data.items():
        prop_scores = proposed.predict_proba(bb)[:, 1]
        linear_scores = linear.predict_proba(xx)[:, 1]
        score_map = {
            "ED": np.log(np.mean(np.abs(yy) ** 2, axis=1) + 1e-12),
            "ENR-ED": xx[:, fn.index("log_energy_ratio")],
            "Signal-only MLP": signal_only.predict_proba(ff[:, signal_only_idx])[:, 1],
            "Linear fusion": linear_scores,
            "Bilinear fusion + global conformal": prop_scores,
            "ENR-ED + adaptive conformal": xx[:, fn.index("log_energy_ratio")],
            "Bilinear fusion + adaptive global": prop_scores,
            "Linear fusion + context conformal": linear_scores,
            "Bilinear fusion + KMeans conformal": prop_scores,
            "Bilinear fusion + oracle conformal": prop_scores,
            "C3FAR-Sense": prop_scores,
        }

        active_cal_context = ccal if split == "matched" else cada
        active_cal_scores = cal_scores_prop if split == "matched" else adapt_scores_prop
        active_cal_linear = cal_scores_lin if split == "matched" else adapt_scores_lin
        active_cal_enr = enr_cal if split == "matched" else adapt_scores_enr
        active_cal_meta = mcal if split == "matched" else mada

        supervised_cal_groups = supervised_gate.predict(active_cal_context)
        supervised_test_groups = supervised_gate.predict(cc)
        kmeans_cal_groups = kmeans_gate.predict(active_cal_context)
        kmeans_test_groups = kmeans_gate.predict(cc)
        oracle_cal_groups = active_cal_meta["env_idx"]
        oracle_test_groups = meta["env_idx"]

        tau_context, context_thresholds = thresholds_from_groups(
            active_cal_scores, supervised_cal_groups, supervised_test_groups, alpha
        )
        tau_linear, _ = thresholds_from_groups(
            active_cal_linear, supervised_cal_groups, supervised_test_groups, alpha
        )
        tau_kmeans, kmeans_thresholds = thresholds_from_groups(
            active_cal_scores, kmeans_cal_groups, kmeans_test_groups, alpha
        )
        tau_oracle, oracle_thresholds = thresholds_from_groups(
            active_cal_scores, oracle_cal_groups, oracle_test_groups, alpha
        )
        decisions = {
            "ED": score_map["ED"] > global_thresholds["ED"],
            "ENR-ED": score_map["ENR-ED"] > global_thresholds["ENR-ED"],
            "Signal-only MLP": score_map["Signal-only MLP"] > global_thresholds["Signal-only MLP"],
            "Linear fusion": linear_scores > global_thresholds["Linear fusion"],
            "Bilinear fusion + global conformal": prop_scores > global_thresholds["Bilinear fusion + global conformal"],
            "ENR-ED + adaptive conformal": score_map["ENR-ED"] > conformal_threshold(active_cal_enr, alpha),
            "Bilinear fusion + adaptive global": prop_scores > conformal_threshold(active_cal_scores, alpha),
            "Linear fusion + context conformal": linear_scores > tau_linear,
            "Bilinear fusion + KMeans conformal": prop_scores > tau_kmeans,
            "Bilinear fusion + oracle conformal": prop_scores > tau_oracle,
            "C3FAR-Sense": prop_scores > tau_context,
        }
        artifacts["thresholds"][split] = {
            "supervised": {str(k): v for k, v in context_thresholds.items()},
            "kmeans": {str(k): v for k, v in kmeans_thresholds.items()},
            "oracle": {str(k): v for k, v in oracle_thresholds.items()},
        }

        for model, scores in score_map.items():
            decision = decisions[model]
            metrics = metric_row(lab, scores, decision)
            rows.append({"seed": seed, "length": length, "split": split, "model": model, **metrics})
            for snr in SNR_GRID:
                idx = meta["snr_db"] == snr
                if np.any(idx & (lab == 1)) and np.any(idx & (lab == 0)):
                    mm = metric_row(lab[idx], scores[idx], decision[idx])
                    snr_rows.append({"seed": seed, "length": length, "split": split, "model": model,
                                     "snr_db": float(snr), **mm})
            for eidx, ename in enumerate(meta["env_names"]):
                idx = meta["env_idx"] == eidx
                if np.any(idx & (lab == 1)) and np.any(idx & (lab == 0)):
                    mm = metric_row(lab[idx], scores[idx], decision[idx])
                    env_rows.append({"seed": seed, "length": length, "split": split, "model": model,
                                     "environment": str(ename), **mm})
            for midx, modulation in enumerate(MODULATIONS):
                h1_idx = (lab == 1) & (meta["mod_idx"] == midx)
                if np.any(h1_idx):
                    artifacts["modulation"].append({
                        "seed": seed, "length": length, "split": split, "model": model,
                        "modulation": modulation, "n_h1": int(h1_idx.sum()),
                        "pd": float(np.mean(decision[h1_idx])),
                    })

        for method, groups in (
            ("C3FAR-Sense", supervised_test_groups),
            ("KMeans conformal", kmeans_test_groups),
            ("Oracle conformal", oracle_test_groups),
        ):
            decision = decisions["C3FAR-Sense" if method == "C3FAR-Sense" else
                                 "Bilinear fusion + KMeans conformal" if method == "KMeans conformal" else
                                 "Bilinear fusion + oracle conformal"]
            for group in np.unique(groups):
                g0 = (groups == group) & (lab == 0)
                g1 = (groups == group) & (lab == 1)
                artifacts["predicted_group"].append({
                    "seed": seed, "length": length, "split": split, "method": method,
                    "group": int(group), "n_h0": int(g0.sum()), "n_h1": int(g1.sum()),
                    "pf": float(np.mean(decision[g0])) if np.any(g0) else np.nan,
                    "pd": float(np.mean(decision[g1])) if np.any(g1) else np.nan,
                })

        for true_idx, true_name in enumerate(meta["env_names"]):
            true_mask = meta["env_idx"] == true_idx
            denom = int(true_mask.sum())
            for group in np.unique(supervised_test_groups):
                count = int(np.sum(true_mask & (supervised_test_groups == group)))
                artifacts["gate_confusion"].append({
                    "seed": seed, "length": length, "split": split,
                    "true_environment": str(true_name), "predicted_group": int(group),
                    "count": count, "rate": count / denom if denom else np.nan,
                })

        if split == "stress" and length == 128:
            method_groups = {
                "Adaptive global": None,
                "KMeans conformal": (kmeans_cal_groups, kmeans_test_groups),
                "C3FAR-Sense": (supervised_cal_groups, supervised_test_groups),
                "Oracle conformal": (oracle_cal_groups, oracle_test_groups),
            }
            for alpha_value in ALPHA_GRID:
                for method, group_pair in method_groups.items():
                    if group_pair is None:
                        decision = prop_scores > conformal_threshold(active_cal_scores, float(alpha_value))
                    else:
                        tau, _ = thresholds_from_groups(
                            active_cal_scores, group_pair[0], group_pair[1], float(alpha_value)
                        )
                        decision = prop_scores > tau
                    metrics = metric_row(lab, prop_scores, decision)
                    environment_pf = []
                    for env_idx in range(len(meta["env_names"])):
                        mask = (lab == 0) & (meta["env_idx"] == env_idx)
                        environment_pf.append(float(np.mean(decision[mask])))
                    artifacts["alpha_sweep"].append({
                        "seed": seed, "length": length, "split": split, "method": method,
                        "alpha": float(alpha_value), **metrics,
                        "worst_environment_pf_error": float(np.max(np.abs(np.asarray(environment_pf) - alpha_value))),
                        "max_environment_pf": float(np.max(environment_pf)),
                    })

            for ncal in (100, 250, 500, 1000, min(3000, len(adapt_scores_prop))):
                global_tau = conformal_threshold(adapt_scores_prop[:ncal], alpha)
                global_decision = prop_scores > global_tau
                tau_c, _ = thresholds_from_groups(
                    adapt_scores_prop[:ncal], supervised_cal_groups[:ncal], supervised_test_groups, alpha
                )
                tau_k, _ = thresholds_from_groups(
                    adapt_scores_prop[:ncal], kmeans_cal_groups[:ncal], kmeans_test_groups, alpha
                )
                artifacts["calibration_size"].extend([
                    {"seed": seed, "length": length, "n_cal": int(ncal), "method": "Adaptive global",
                     **metric_row(lab, prop_scores, global_decision)},
                    {"seed": seed, "length": length, "n_cal": int(ncal), "method": "KMeans conformal",
                     **metric_row(lab, prop_scores, prop_scores > tau_k)},
                    {"seed": seed, "length": length, "n_cal": int(ncal), "method": "C3FAR-Sense",
                     **metric_row(lab, prop_scores, prop_scores > tau_c)},
                ])

    if length == 128:
        supervised_adapt_groups = supervised_gate.predict(cada)
        base_reference = rst / np.sqrt(10.0 ** (mst["ref_delta_db"][:, None] / 10.0))
        standardized_delta = mst["ref_delta_db"] / 0.85
        for sigma_db in (0.0, 0.30, 0.85, 1.50, 3.00):
            shifted_reference = base_reference * np.sqrt(
                10.0 ** ((sigma_db * standardized_delta)[:, None] / 10.0)
            )
            f_shift, c_shift, _, _, _ = extract_features(yst, shifted_reference.astype(np.complex64))
            b_shift = bilinear_features(f_shift, c_shift)
            shifted_scores = proposed.predict_proba(b_shift)[:, 1]
            shifted_groups = supervised_gate.predict(c_shift)
            tau_shift, _ = thresholds_from_groups(
                adapt_scores_prop, supervised_adapt_groups, shifted_groups, alpha
            )
            for method, decision in (
                ("Adaptive global", shifted_scores > conformal_threshold(adapt_scores_prop, alpha)),
                ("C3FAR-Sense", shifted_scores > tau_shift),
            ):
                artifacts["reference_mismatch"].append({
                    "seed": seed, "length": length, "sigma_delta_db": sigma_db,
                    "method": method, **metric_row(lst, shifted_scores, decision),
                })

        contaminated_bank, contamination_priority, leak_snr_db = build_reference_contamination_bank(
            rst, seed=900_000 + seed
        )
        # The integrity guard reuses the signal-structure MLP as a reference-only
        # leakage score.  It is calibrated on clean deployment references with
        # the same groupwise finite-sample upper-tail rule at gamma=0.01.
        f_integrity_cal, c_integrity_cal, _, _, _ = extract_features(rada, rada)
        integrity_cal_scores = signal_only.predict_proba(
            f_integrity_cal[:, signal_only_idx]
        )[:, 1]
        integrity_cal_groups = supervised_gate.predict(c_integrity_cal)
        for contamination_rate in (0.0, 0.01, 0.05, 0.10, 0.20):
            contaminated_mask = contamination_priority < contamination_rate
            reference = rst.copy()
            reference[contaminated_mask] = contaminated_bank[contaminated_mask]
            f_ref, c_ref, _, _, _ = extract_features(yst, reference)
            b_ref = bilinear_features(f_ref, c_ref)
            scores = proposed.predict_proba(b_ref)[:, 1]
            groups = supervised_gate.predict(c_ref)
            tau_ref, _ = thresholds_from_groups(adapt_scores_prop, supervised_adapt_groups, groups, alpha)
            raw_decision = scores > tau_ref
            f_reference_only, c_reference_only, _, _, _ = extract_features(reference, reference)
            integrity_scores = signal_only.predict_proba(
                f_reference_only[:, signal_only_idx]
            )[:, 1]
            integrity_groups = supervised_gate.predict(c_reference_only)
            integrity_tau, _ = thresholds_from_groups(
                integrity_cal_scores, integrity_cal_groups, integrity_groups, 0.01
            )
            flags = integrity_scores > integrity_tau
            guarded_decision = raw_decision | flags
            raw_metrics = metric_row(lst, scores, raw_decision)
            guarded_metrics = metric_row(lst, scores, guarded_decision)
            clean_mask = ~contaminated_mask
            artifacts["reference_contamination"].append({
                "seed": seed, "length": length, "contamination_rate": contamination_rate,
                "n_contaminated": int(contaminated_mask.sum()),
                "mean_leak_snr_db": float(np.nanmean(leak_snr_db[contaminated_mask])) if np.any(contaminated_mask) else np.nan,
                "raw_auc": raw_metrics["auc"], "raw_pf": raw_metrics["pf"], "raw_pd": raw_metrics["pd"],
                "guarded_pf": guarded_metrics["pf"], "guarded_pd": guarded_metrics["pd"],
                "flag_tpr": float(np.mean(flags[contaminated_mask])) if np.any(contaminated_mask) else np.nan,
                "flag_fpr": float(np.mean(flags[clean_mask])) if np.any(clean_mask) else np.nan,
            })

    elapsed = time.perf_counter() - t0
    n_params_signal = sum(np.prod(w.shape) for w in signal_only[-1].coefs_) + sum(b.size for b in signal_only[-1].intercepts_)
    n_params_prop = int(proposed[-1].coef_.size + proposed[-1].intercept_.size)
    artifacts.update({
        "seed": seed,
        "length": length,
        "elapsed_s": elapsed,
        "n_params_signal_mlp": int(n_params_signal),
        "n_parameters_proposed": int(n_params_prop),
        "n_features_signal": len(signal_only_idx),
        "n_features_proposed": btr.shape[1],
        "global_thresholds": global_thresholds,
    })
    return pd.DataFrame(rows), pd.DataFrame(snr_rows), pd.DataFrame(env_rows), artifacts


def mean_ci(series: pd.Series) -> tuple[float, float]:
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mean = float(np.mean(x))
    # A Student-t interval is appropriate for the five-seed Monte Carlo
    # summaries; using the asymptotic 1.96 multiplier would understate the
    # uncertainty at this small number of independent repetitions.
    critical = float(student_t.ppf(0.975, df=x.size - 1)) if x.size > 1 else 0.0
    ci = float(critical * np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return mean, ci


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return summarize_metrics(df, group_cols, ["auc", "pf", "pd", "balanced_accuracy"])


def summarize_metrics(df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(group_cols, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        for metric in metrics:
            row[f"{metric}_mean"], row[f"{metric}_ci95"] = mean_ci(g[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparisons(runs: pd.DataFrame) -> pd.DataFrame:
    specifications = []
    for length in sorted(runs["length"].unique()):
        specifications.append((length, "stress", "C3FAR-Sense", "Bilinear fusion + adaptive global", "pd"))
    specifications.extend([
        (128, "stress", "C3FAR-Sense", "Bilinear fusion + KMeans conformal", "pd"),
        (128, "stress", "C3FAR-Sense", "Linear fusion + context conformal", "pd"),
        (128, "matched", "C3FAR-Sense", "Linear fusion + context conformal", "pd"),
    ])
    rows = []
    for length, split, method_a, method_b, metric in specifications:
        subset = runs[(runs["length"] == length) & (runs["split"] == split)]
        a = subset[subset["model"] == method_a].sort_values("seed")
        b = subset[subset["model"] == method_b].sort_values("seed")
        common = sorted(set(a["seed"]) & set(b["seed"]))
        av = a.set_index("seed").loc[common, metric].to_numpy(dtype=float)
        bv = b.set_index("seed").loc[common, metric].to_numpy(dtype=float)
        difference = av - bv
        mean_difference, ci95 = mean_ci(pd.Series(difference))
        p_value = float(ttest_rel(av, bv).pvalue) if len(common) > 1 else float("nan")
        effect = float(mean_difference / difference.std(ddof=1)) if len(common) > 1 and difference.std(ddof=1) > 0 else float("nan")
        rows.append({
            "length": int(length), "split": split, "metric": metric,
            "method_a": method_a, "method_b": method_b, "n_pairs": len(common),
            "mean_paired_difference": mean_difference, "ci95": ci95,
            "effect_size_dz": effect, "p_value": p_value,
        })
    result = pd.DataFrame(rows)
    finite = result["p_value"].notna()
    indices = result.index[finite].to_list()
    order = sorted(indices, key=lambda idx: result.loc[idx, "p_value"])
    adjusted = {}
    running = 0.0
    total = len(order)
    for rank, idx in enumerate(order):
        candidate = min(1.0, (total - rank) * float(result.loc[idx, "p_value"]))
        running = max(running, candidate)
        adjusted[idx] = running
    result["p_value_holm"] = [adjusted.get(idx, np.nan) for idx in result.index]
    return result


def plot_results(summary: pd.DataFrame, snr_summary: pd.DataFrame, env_summary: pd.DataFrame, outdir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.28,
    })
    colors = {
        "ED": "#6B7280", "ENR-ED": "#E69F00", "Signal-only MLP": "#56B4E9",
        "Linear fusion": "#009E73", "Bilinear fusion + global conformal": "#CC79A7",
        "ENR-ED + adaptive conformal": "#F0B429",
        "Bilinear fusion + adaptive global": "#8E5EA2",
        "C3FAR-Sense": "#D55E00",
    }

    # Figure 1: matched Pd by SNR, one panel per length.
    lengths = sorted(snr_summary["length"].unique())
    fig, axes = plt.subplots(1, len(lengths), figsize=(4.7 * len(lengths), 3.8), sharey=True)
    if len(lengths) == 1:
        axes = [axes]
    show_models = ["ED", "ENR-ED + adaptive conformal", "Signal-only MLP", "Linear fusion", "C3FAR-Sense"]
    for ax, length in zip(axes, lengths):
        sub = snr_summary[(snr_summary.split == "matched") & (snr_summary.length == length)]
        for model in show_models:
            g = sub[sub.model == model].sort_values("snr_db")
            ax.plot(g.snr_db, g.pd_mean, marker="o", lw=2.0, ms=4, label=model, color=colors[model])
            ax.fill_between(g.snr_db, g.pd_mean - g.pd_ci95, g.pd_mean + g.pd_ci95, color=colors[model], alpha=0.10)
        ax.set_title(f"L = {length}")
        ax.set_xlabel("Nominal SNR (dB)")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(r"Detection probability $P_d$ at target $P_f=0.1$")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.08), frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_pd_snr.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: environment-wise false alarm at L=128.
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), sharey=True)
    env_models = [
        "ENR-ED", "Bilinear fusion + global conformal",
        "Bilinear fusion + adaptive global", "C3FAR-Sense",
    ]
    for ax, split in zip(axes, ("matched", "stress")):
        sub = env_summary[(env_summary.length == 128) & (env_summary.split == split)]
        envs = list(sub.environment.unique())
        x = np.arange(len(envs))
        width = 0.18
        for j, model in enumerate(env_models):
            vals = [float(sub[(sub.environment == e) & (sub.model == model)].pf_mean.iloc[0]) for e in envs]
            cis = [float(sub[(sub.environment == e) & (sub.model == model)].pf_ci95.iloc[0]) for e in envs]
            ax.bar(x + (j - 1.5) * width, vals, width, yerr=cis, capsize=2, color=colors[model], label=model)
        ax.axhline(0.10, color="black", ls="--", lw=1.4, label="Target")
        ax.set_xticks(x, [e.replace("-", "\n") for e in envs])
        ax.set_title("Calibration-matched mixture" if split == "matched" else "Unseen stress mixture")
        ax.set_ylabel("False-alarm probability" if split == "matched" else "")
        ax.set_ylim(0, 0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / "fig_pf_environment.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: performance/complexity scatter.
    scatter_models = ["ED", "ENR-ED", "Signal-only MLP", "Linear fusion", "C3FAR-Sense"]
    sub = summary[(summary.length == 128) & (summary.split == "matched") &
                  (summary.model.isin(scatter_models))]
    param_proxy = {
        "ED": 0, "ENR-ED": 0, "Linear fusion": 48,
        "Signal-only MLP": 4.0, "Bilinear fusion + global conformal": 10.0,
        "ENR-ED + adaptive conformal": 0, "Bilinear fusion + adaptive global": 10.0,
        "C3FAR-Sense": 10.0,
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, row in sub.iterrows():
        m = row.model
        x = param_proxy[m]
        ax.scatter(x, row.pd_mean, s=80, color=colors[m], edgecolor="black", linewidth=0.5)
        ax.annotate(m, (x, row.pd_mean), xytext=(5, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Relative learned complexity (feature/model proxy)")
    ax.set_ylabel(r"$P_d$ at target $P_f=0.1$")
    ax.set_title("Accuracy-complexity trade-off at L = 128")
    fig.tight_layout()
    fig.savefig(outdir / "fig_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("quick", "full", "calibration", "paper"), default="quick")
    ap.add_argument("--out", type=Path, default=Path("research/results"))
    ap.add_argument("--seeds", help="Comma-separated integer seeds; overrides mode defaults")
    ap.add_argument("--lengths", help="Comma-separated window lengths; overrides mode defaults")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.mode == "quick":
        size = StudySize(n_train=5000, n_cal_h0=1200, n_adapt_h0=900, n_test=3000, n_stress=3000, mlp_iter=45)
        seeds = [1]
        lengths = [64]
    else:
        size = StudySize(n_train=18000, n_cal_h0=4500, n_adapt_h0=3000, n_test=12000, n_stress=9000, mlp_iter=90)
        seeds = list(range(1, 11)) if args.mode == "paper" else [1, 2, 3, 4, 5]
        lengths = [128] if args.mode == "calibration" else [64, 128, 256]
    if args.seeds:
        seeds = [int(value) for value in args.seeds.split(",")]
    if args.lengths:
        lengths = [int(value) for value in args.lengths.split(",")]

    all_main, all_snr, all_env, artifacts = [], [], [], []
    for length in lengths:
        for seed in seeds:
            print(f"Running L={length}, seed={seed}", flush=True)
            a, b, c, d = run_one(seed, length, size, args.out)
            all_main.append(a); all_snr.append(b); all_env.append(c); artifacts.append(d)
            print(f"  completed in {d['elapsed_s']:.1f}s", flush=True)
    main_df = pd.concat(all_main, ignore_index=True)
    snr_df = pd.concat(all_snr, ignore_index=True)
    env_df = pd.concat(all_env, ignore_index=True)
    summary = summarize(main_df, ["length", "split", "model"])
    snr_summary = summarize(snr_df, ["length", "split", "model", "snr_db"])
    env_summary = summarize(env_df, ["length", "split", "model", "environment"])
    main_df.to_csv(args.out / "runs_global.csv", index=False)
    snr_df.to_csv(args.out / "runs_snr.csv", index=False)
    env_df.to_csv(args.out / "runs_environment.csv", index=False)
    summary.to_csv(args.out / "summary_global.csv", index=False)
    snr_summary.to_csv(args.out / "summary_snr.csv", index=False)
    env_summary.to_csv(args.out / "summary_environment.csv", index=False)
    (args.out / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    artifact_tables = {
        "calibration_size": (["length", "method", "n_cal"], ["auc", "pf", "pd", "balanced_accuracy"]),
        "alpha_sweep": (["length", "split", "method", "alpha"],
                        ["auc", "pf", "pd", "balanced_accuracy", "worst_environment_pf_error", "max_environment_pf"]),
        "modulation": (["length", "split", "model", "modulation"], ["n_h1", "pd"]),
        "predicted_group": (["length", "split", "method", "group"], ["n_h0", "n_h1", "pf", "pd"]),
        "gate_confusion": (["length", "split", "true_environment", "predicted_group"], ["count", "rate"]),
        "reference_mismatch": (["length", "method", "sigma_delta_db"], ["auc", "pf", "pd", "balanced_accuracy"]),
        "reference_contamination": (
            ["length", "contamination_rate"],
            ["n_contaminated", "mean_leak_snr_db", "raw_auc", "raw_pf", "raw_pd",
             "guarded_pf", "guarded_pd", "flag_tpr", "flag_fpr"],
        ),
    }
    for name, (groups, metrics) in artifact_tables.items():
        table_rows = [row for art in artifacts for row in art.get(name, [])]
        if not table_rows:
            continue
        table = pd.DataFrame(table_rows)
        table.to_csv(args.out / f"runs_{name}.csv", index=False)
        summarize_metrics(table, groups, metrics).to_csv(args.out / f"summary_{name}.csv", index=False)
    paired_comparisons(main_df).to_csv(args.out / "paired_comparisons.csv", index=False)
    run_manifest = {
        "mode": args.mode,
        "seeds": seeds,
        "lengths": lengths,
        "study_size": size.__dict__,
        "target_alpha": 0.10,
        "alpha_grid": ALPHA_GRID.tolist(),
        "modulations": list(MODULATIONS),
        "matched_environments": list(ENVIRONMENTS),
        "stress_environments": list(STRESS_ENVIRONMENTS),
    }
    (args.out / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    plot_results(summary, snr_summary, env_summary, args.out)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
