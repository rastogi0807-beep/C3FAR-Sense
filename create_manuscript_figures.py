#!/usr/bin/env python3
"""Create publication figures for the C3FAR-Sense manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results_upgrade_10seed"
CAL_RESULTS = ROOT / "results_upgrade_10seed"
LATENCY_RESULTS = ROOT / "results_adaptive" / "latency.csv"
OUT = ROOT / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

COLORS = {
    "ED": "#5F6B7A",
    "ENR-ED": "#E69F00",
    "ENR-ED + adaptive conformal": "#E69F00",
    "Signal-only MLP": "#56B4E9",
    "Linear fusion": "#009E73",
    "Bilinear fusion + global conformal": "#CC79A7",
    "Bilinear fusion + adaptive global": "#7B61A8",
    "Bilinear fusion + KMeans conformal": "#0072B2",
    "Bilinear fusion + oracle conformal": "#009E73",
    "KMeans conformal": "#0072B2",
    "Oracle conformal": "#009E73",
    "Adaptive global": "#7B61A8",
    "C3FAR-Sense": "#D55E00",
}


def setup() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.7,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
        "savefig.facecolor": "white",
    })


def box(ax, xy, wh, text, color, fontsize=9, edge="#2D3748", lw=1.0):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=color, edgecolor=edge, linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color="#17202A", linespacing=1.15)
    return patch


def arrow(ax, start, end, color="#4A5568", style="-|>", rad=0.0, lw=1.25):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=11,
        color=color, linewidth=lw,
        connectionstyle=f"arc3,rad={rad}", shrinkA=3, shrinkB=3,
    ))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.02, 0.69), (0.14, 0.16), "Sensing window\ny[0:L−1]", "#E8F1FA", 9)
    box(ax, (0.02, 0.25), (0.14, 0.16), "Noise reference\nr[0:L−1]", "#E8F7F2", 9)
    box(ax, (0.23, 0.66), (0.18, 0.21), "Evidence bank\n43 robust ratio,\nspectral and temporal\nfeatures e", "#DCECF8", 8.5)
    box(ax, (0.23, 0.22), (0.18, 0.21), "Context bank\n10 noise-power, tail,\nspectral and correlation\nfeatures c", "#D9F0E6", 8.5)
    box(ax, (0.48, 0.62), (0.18, 0.25), "Bilinear scorer\nz = [e; c; vec(ecᵀ)]\nscore s = σ(β₀ + βᵀz)\n484 parameters", "#F5E5F0", 8.5)
    box(ax, (0.48, 0.19), (0.18, 0.18), "Context gate\ng = arg max qθ(g | c)\n3 learned regimes", "#E5E0F2", 8.5)
    box(ax, (0.73, 0.17), (0.18, 0.22), "H₀ calibration bank\nwithin predicted group g\nτg = conformal\n(1−α)-quantile", "#FFF1D6", 8.5)
    box(ax, (0.76, 0.64), (0.18, 0.19), "CFAR decision\nH₁ if s > τg\nH₀ otherwise", "#F9E1D6", 9)

    arrow(ax, (0.16, 0.77), (0.23, 0.77))
    arrow(ax, (0.16, 0.33), (0.23, 0.33))
    arrow(ax, (0.15, 0.38), (0.24, 0.69), rad=-0.18)
    arrow(ax, (0.41, 0.76), (0.48, 0.75))
    arrow(ax, (0.41, 0.33), (0.48, 0.69), rad=-0.18)
    arrow(ax, (0.41, 0.31), (0.48, 0.28))
    arrow(ax, (0.66, 0.28), (0.73, 0.28))
    arrow(ax, (0.66, 0.74), (0.76, 0.74))
    arrow(ax, (0.82, 0.39), (0.84, 0.64))

    ax.text(0.82, 0.085,
            "Deployment refresh: update only τg from trusted noise-only windows; keep β and θ frozen.",
            ha="center", va="center", fontsize=8.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="#F7FAFC", edgecolor="#A0AEC0"))
    ax.text(0.02, 0.95, "C³FAR-Sense processing and calibration paths", fontsize=12,
            fontweight="bold", color="#1F2937", va="top")
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig1_architecture.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def matched_pd() -> None:
    df = pd.read_csv(RESULTS / "summary_snr.csv")
    lengths = (64, 128, 256)
    models = ("ED", "ENR-ED + adaptive conformal", "Signal-only MLP", "Linear fusion", "C3FAR-Sense")
    labels = {
        "ED": "ED",
        "ENR-ED + adaptive conformal": "ENR-ED",
        "Signal-only MLP": "Signal-only MLP",
        "Linear fusion": "Linear fusion",
        "C3FAR-Sense": "C³FAR-Sense",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.9, 3.35), sharex=True, sharey=True)
    for ax, length in zip(axes, lengths):
        sub = df[(df["split"] == "matched") & (df["length"] == length)]
        for model in models:
            g = sub[sub["model"] == model].sort_values("snr_db")
            ax.errorbar(g["snr_db"], g["pd_mean"], yerr=g["pd_ci95"], marker="o",
                        ms=3.4, capsize=2, color=COLORS[model], label=labels[model])
        ax.set_title(f"L = {length}")
        ax.set_xlabel("Nominal SNR (dB)")
        ax.set_xticks([-20, -16, -12, -8, -4, 0])
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-20.8, 0.8)
    axes[0].set_ylabel("Detection probability, Pᵈ")
    handles, leglabels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=1.2)
    fig.savefig(OUT / "fig2_matched_pd_snr.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def stress_pd() -> None:
    df = pd.read_csv(RESULTS / "summary_snr.csv")
    models = ("ED", "ENR-ED + adaptive conformal", "Bilinear fusion + adaptive global", "C3FAR-Sense")
    labels = {
        "ED": "ED (frozen)",
        "ENR-ED + adaptive conformal": "ENR-ED + refresh",
        "Bilinear fusion + adaptive global": "Bilinear + global refresh",
        "C3FAR-Sense": "C³FAR-Sense",
    }
    fig, ax = plt.subplots(figsize=(6.65, 3.9))
    sub = df[(df["split"] == "stress") & (df["length"] == 128)]
    for model in models:
        g = sub[sub["model"] == model].sort_values("snr_db")
        ax.errorbar(g["snr_db"], g["pd_mean"], yerr=g["pd_ci95"], marker="o",
                    ms=4, capsize=2.2, color=COLORS[model], label=labels[model])
    ax.set_xlabel("Nominal SNR (dB)")
    ax.set_ylabel("Detection probability, Pᵈ")
    ax.set_title("Unseen receiver-mismatch stress test (L = 128)")
    ax.set_xticks([-20, -16, -12, -8, -4, 0])
    ax.set_xlim(-20.8, 0.8)
    ax.set_ylim(0, 0.95)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_stress_pd_snr.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def environment_pf() -> None:
    df = pd.read_csv(RESULTS / "summary_environment.csv")
    models = (
        "ENR-ED",
        "Bilinear fusion + global conformal",
        "Bilinear fusion + adaptive global",
        "C3FAR-Sense",
    )
    labels = {
        "ENR-ED": "ENR-ED (frozen)",
        "Bilinear fusion + global conformal": "Bilinear (frozen)",
        "Bilinear fusion + adaptive global": "Bilinear + global refresh",
        "C3FAR-Sense": "C³FAR-Sense",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.75))
    settings = (("matched", (0, 0.24), "Calibration-matched mixture"),
                ("stress", (0, 0.98), "Unseen stress mixture"))
    for ax, (split, ylim, title) in zip(axes, settings):
        sub = df[(df["length"] == 128) & (df["split"] == split)]
        envs = list(sub["environment"].unique())
        x = np.arange(len(envs))
        width = 0.19
        for j, model in enumerate(models):
            g = sub[sub["model"] == model].set_index("environment")
            vals = np.array([g.loc[e, "pf_mean"] for e in envs])
            cis = np.array([g.loc[e, "pf_ci95"] for e in envs])
            ax.bar(x + (j - 1.5) * width, vals, width, yerr=cis, capsize=2,
                   color=COLORS[model], edgecolor="white", linewidth=0.35, label=labels[model])
        ax.axhline(0.10, color="#111827", ls="--", lw=1.2, label="Target")
        ax.set_xticks(x, [e.replace("-", "\n") for e in envs])
        ax.set_ylim(*ylim)
        ax.set_title(title)
        ax.set_ylabel("False-alarm probability, Pᶠ")
    handles, leglabels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(leglabels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    fig.savefig(OUT / "fig4_environment_pf.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def calibration_size() -> None:
    df = pd.read_csv(CAL_RESULTS / "summary_calibration_size.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.55))
    styles = {
        "Adaptive global": ("#7B61A8", "o", "Adaptive global"),
        "C3FAR-Sense": ("#D55E00", "s", "C³FAR-Sense"),
    }
    for method, (color, marker, label) in styles.items():
        g = df[df["method"] == method].sort_values("n_cal")
        axes[0].errorbar(g["n_cal"], g["pf_mean"], yerr=g["pf_ci95"], color=color,
                         marker=marker, capsize=2.5, label=label)
        axes[1].errorbar(g["n_cal"], g["pd_mean"], yerr=g["pd_ci95"], color=color,
                         marker=marker, capsize=2.5, label=label)
    axes[0].axhline(0.10, color="#111827", ls="--", lw=1.2, label="Target")
    axes[0].set_ylabel("False-alarm probability, Pᶠ")
    axes[1].set_ylabel("Detection probability, Pᵈ")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([100, 250, 500, 1000, 3000], ["100", "250", "500", "1000", "3000"])
        ax.set_xlabel("Noise-only refresh windows, M")
    axes[0].set_ylim(0.03, 0.18)
    axes[1].set_ylim(0.28, 0.54)
    axes[0].set_title("False-alarm stability")
    axes[1].set_title("Detection after threshold refresh")
    handles, leglabels = axes[0].get_legend_handles_labels()
    fig.legend(handles, leglabels, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    fig.savefig(OUT / "fig5_calibration_size.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def alpha_sweep() -> None:
    df = pd.read_csv(RESULTS / "summary_alpha_sweep.csv")
    methods = ("Adaptive global", "KMeans conformal", "C3FAR-Sense", "Oracle conformal")
    labels = {
        "Adaptive global": "Adaptive global",
        "KMeans conformal": "K-means contexts",
        "C3FAR-Sense": "C³FAR-Sense",
        "Oracle conformal": "Oracle contexts",
    }
    markers = {"Adaptive global": "o", "KMeans conformal": "^", "C3FAR-Sense": "s", "Oracle conformal": "D"}
    fig, axes = plt.subplots(1, 2, figsize=(8.9, 3.6))
    for method in methods:
        g = df[df["method"] == method].sort_values("alpha")
        axes[0].errorbar(g["alpha"], g["pf_mean"], yerr=g["pf_ci95"],
                         color=COLORS[method], marker=markers[method], capsize=2.3,
                         label=labels[method])
        axes[1].errorbar(g["alpha"], g["worst_environment_pf_error_mean"],
                         yerr=g["worst_environment_pf_error_ci95"],
                         color=COLORS[method], marker=markers[method], capsize=2.3,
                         label=labels[method])
    axes[0].plot([0, 0.21], [0, 0.21], color="#111827", ls="--", lw=1.1, label="Ideal")
    axes[0].set_xlabel("Target false-alarm level, α")
    axes[0].set_ylabel("Empirical aggregate Pᶠ")
    axes[0].set_xlim(0, 0.21)
    axes[0].set_ylim(0, 0.25)
    axes[0].set_title("Aggregate calibration")
    axes[1].set_xlabel("Target false-alarm level, α")
    axes[1].set_ylabel("Worst true-environment |Pᶠ − α|")
    axes[1].set_xlim(0, 0.21)
    axes[1].set_ylim(bottom=0)
    axes[1].set_title("Cross-environment imbalance")
    handles, leglabels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(leglabels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center",
               bbox_to_anchor=(0.5, 1.05), ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90), w_pad=2.0)
    fig.savefig(OUT / "fig7_alpha_sweep.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def modulation_pd() -> None:
    df = pd.read_csv(RESULTS / "summary_modulation.csv")
    methods = (
        "Bilinear fusion + adaptive global",
        "Bilinear fusion + KMeans conformal",
        "C3FAR-Sense",
        "Bilinear fusion + oracle conformal",
    )
    labels = {
        "Bilinear fusion + adaptive global": "Adaptive global",
        "Bilinear fusion + KMeans conformal": "K-means contexts",
        "C3FAR-Sense": "C³FAR-Sense",
        "Bilinear fusion + oracle conformal": "Oracle contexts",
    }
    subset = df[(df["length"] == 128) & (df["split"] == "stress")]
    modulations = list(subset["modulation"].drop_duplicates())
    x = np.arange(len(modulations))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10.6, 3.8))
    for index, method in enumerate(methods):
        g = subset[subset["model"] == method].set_index("modulation")
        values = np.array([g.loc[name, "pd_mean"] for name in modulations])
        errors = np.array([g.loc[name, "pd_ci95"] for name in modulations])
        ax.bar(x + (index - 1.5) * width, values, width, yerr=errors, capsize=1.7,
               color=COLORS[method], edgecolor="white", linewidth=0.35,
               label=labels[method])
    ax.set_xticks(x, [name.replace("AM-DSB", "AM\nDSB") for name in modulations])
    ax.set_ylabel("Detection probability, Pᵈ")
    ax.set_ylim(0, 0.72)
    ax.set_title("Modulation-resolved detection under stress (L = 128, α = 0.10)")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(OUT / "fig8_modulation_pd.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def reference_robustness() -> None:
    mismatch = pd.read_csv(RESULTS / "summary_reference_mismatch.csv")
    contamination = pd.read_csv(RESULTS / "summary_reference_contamination.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.6))
    for method in ("Adaptive global", "C3FAR-Sense"):
        g = mismatch[mismatch["method"] == method].sort_values("sigma_delta_db")
        label = "Adaptive global" if method == "Adaptive global" else "C³FAR-Sense"
        axes[0, 0].errorbar(g["sigma_delta_db"], g["pf_mean"], yerr=g["pf_ci95"],
                            color=COLORS[method], marker="o", capsize=2.2, label=label)
        axes[0, 1].errorbar(g["sigma_delta_db"], g["pd_mean"], yerr=g["pd_ci95"],
                            color=COLORS[method], marker="o", capsize=2.2, label=label)
    axes[0, 0].axhline(0.10, color="#111827", ls="--", lw=1.1, label="Target")
    axes[0, 0].set_ylabel("Pᶠ")
    axes[0, 1].set_ylabel("Pᵈ")
    for ax in axes[0]:
        ax.set_xlabel("Test reference mismatch σδ (dB)")
    axes[0, 0].set_title("Mismatch: false alarms")
    axes[0, 1].set_title("Mismatch: detection")
    x = 100 * contamination["contamination_rate"]
    axes[1, 0].errorbar(x, contamination["raw_pd_mean"], yerr=contamination["raw_pd_ci95"],
                        color="#D55E00", marker="o", capsize=2.2, label="Unscreened")
    axes[1, 0].errorbar(x, contamination["guarded_pd_mean"], yerr=contamination["guarded_pd_ci95"],
                        color="#0072B2", marker="s", capsize=2.2, label="Fail-closed guard")
    axes[1, 1].errorbar(x, contamination["raw_pf_mean"], yerr=contamination["raw_pf_ci95"],
                        color="#D55E00", marker="o", capsize=2.2, label="Unscreened")
    axes[1, 1].errorbar(x, contamination["guarded_pf_mean"], yerr=contamination["guarded_pf_ci95"],
                        color="#0072B2", marker="s", capsize=2.2, label="Fail-closed guard")
    axes[1, 1].axhline(0.10, color="#111827", ls="--", lw=1.1, label="Target")
    axes[1, 0].set_ylabel("Pᵈ")
    axes[1, 1].set_ylabel("Pᶠ")
    for ax in axes[1]:
        ax.set_xlabel("Contaminated reference windows (%)")
    axes[1, 0].set_title("Reference leakage: detection")
    axes[1, 1].set_title("Reference leakage: false alarms")
    for ax in axes.ravel():
        ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(h_pad=2.0, w_pad=1.7)
    fig.savefig(OUT / "fig9_reference_robustness.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def gate_confusion() -> None:
    df = pd.read_csv(RESULTS / "summary_gate_confusion.csv")
    group_labels = ["white-like", "colored-like", "impulsive-like"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.55))
    for ax, split, title in zip(axes, ("matched", "stress"),
                                ("Calibration-matched", "Unseen stress")):
        sub = df[(df["length"] == 128) & (df["split"] == split)]
        environments = list(sub["true_environment"].drop_duplicates())
        matrix = np.zeros((len(environments), 3))
        for row_index, environment in enumerate(environments):
            for group in range(3):
                row = sub[(sub["true_environment"] == environment) &
                          (sub["predicted_group"] == group)]
                matrix[row_index, group] = float(row["rate_mean"].iloc[0]) if len(row) else 0.0
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = matrix[row_index, col_index]
                ax.text(col_index, row_index, f"{100*value:.1f}%", ha="center", va="center",
                        color="white" if value > 0.55 else "#111827", fontsize=8)
        ax.set_xticks(range(3), group_labels, rotation=20, ha="right")
        ax.set_yticks(range(len(environments)), [name.replace("-", " ") for name in environments])
        ax.set_xlabel("Predicted context")
        ax.set_title(title)
    axes[0].set_ylabel("Simulator environment")
    fig.colorbar(image, ax=axes, fraction=0.028, pad=0.03, label="Assignment rate")
    fig.subplots_adjust(left=0.12, right=0.91, bottom=0.25, top=0.87, wspace=0.38)
    fig.savefig(OUT / "fig10_gate_confusion.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def latency() -> None:
    df = pd.read_csv(LATENCY_RESULTS)
    fig, ax = plt.subplots(figsize=(5.8, 3.45))
    x = np.arange(len(df))
    ax.bar(x, df["feature_ms_per_sample"], color="#56B4E9", width=0.55, label="Feature extraction")
    ax.bar(x, df["decision_ms_per_sample"], bottom=df["feature_ms_per_sample"],
           color="#D55E00", width=0.55, label="Scoring, gate and threshold")
    for xi, total in zip(x, df["total_ms_per_sample"]):
        ax.text(xi, total + 0.003, f"{total:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(v) for v in df["length"]])
    ax.set_xlabel("Window length, L")
    ax.set_ylabel("Batch CPU time per window (ms)")
    ax.set_ylim(0, 0.105)
    ax.set_title("Frozen-detector batch latency (batch = 4096)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_latency.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    global RESULTS, CAL_RESULTS, LATENCY_RESULTS, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--latency", type=Path, default=LATENCY_RESULTS)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    RESULTS = args.results
    CAL_RESULTS = args.results
    LATENCY_RESULTS = args.latency
    OUT = args.out
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    architecture()
    matched_pd()
    stress_pd()
    environment_pf()
    calibration_size()
    alpha_sweep()
    modulation_pd()
    reference_robustness()
    gate_confusion()
    latency()
    for path in sorted(OUT.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
