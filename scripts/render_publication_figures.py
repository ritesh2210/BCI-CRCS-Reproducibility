#!/usr/bin/env python3
"""Render publication-aligned PNG figures from archived result CSVs.

This renderer does not rerun the EEG experiments and does not hard-code performance
values. It reads the released CSV result tables and applies the manuscript's visual
language (serif typography, Okabe-Ito-derived palette, compact axes, thin rules,
and publication-scale labels). Output PNGs are saved as RGB at 600 dpi for broad
viewer compatibility.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# Manuscript palette (matches Revised_BCI_Manuscript.tex)
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILION = "#D55E00"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GREY = "#6E6E6E"
GRID = "#D9D9D9"

SCENARIO_STYLE = {
    "clean": dict(color=GREY, linestyle=":", marker="^", label="Clean source"),
    "poisoned": dict(color=VERMILION, linestyle="--", marker="s", label="Poisoned source"),
    "sanitized": dict(color=PURPLE, linestyle="-.", marker="D", label="Phase-I sanitizer"),
    "spectral_baseline": dict(color=PURPLE, linestyle="-.", marker="D", label="Spectral baseline"),
    "crcs": dict(color=GREEN, linestyle="-", marker="o", label="CRCS"),
}
POISON_STYLE = {
    0.05: dict(color=BLUE, marker="o", label="p=5%"),
    0.10: dict(color=ORANGE, marker="s", label="p=10%"),
    0.15: dict(color=PURPLE, marker="^", label="p=15%"),
    0.20: dict(color=VERMILION, marker="D", label="p=20%"),
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.3,
        "axes.linewidth": 0.55,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 0.95,
        "lines.markersize": 3.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
    })


def finish(ax, out: Path, *, ylim=None, zero_line=False, primary_budget=False, legend_cols=2) -> None:
    if zero_line:
        ax.axhline(0, color="#777777", lw=0.55, ls=":", zorder=0)
    if primary_budget:
        ax.axvline(15, color="#777777", lw=0.55, ls=":", zorder=0)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, which="major", color=GRID, lw=0.45, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#666666")
        spine.set_linewidth(0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=legend_cols,
              frameon=False, columnspacing=1.0, handlelength=2.0, handletextpad=0.45)
    fig = ax.figure
    fig.tight_layout(pad=0.35)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
    # Force RGB (not RGBA) for maximum compatibility across Windows/image viewers.
    with Image.open(out) as im:
        rgb = im.convert("RGB")
        rgb.save(out, format="PNG", dpi=(600, 600), optimize=True)


def phase1() -> None:
    df = pd.read_csv(ROOT / "phase1_results" / "summary_results.csv")
    df = df[df["calibration_fraction"].isin([0.0, 0.10, 0.25, 0.50])]
    for metric, ylabel, filename, ylim in [
        ("accuracy", "Clean-test accuracy (%)", "accuracy_vs_calibration.png", None),
        ("asr", "Targeted ASR (%)", "asr_vs_calibration.png", (-4, 104)),
    ]:
        fig, ax = plt.subplots(figsize=(3.35, 2.05))
        for scenario in ["clean", "poisoned", "sanitized"]:
            g = df[df["scenario"] == scenario].sort_values("calibration_fraction")
            st = SCENARIO_STYLE[scenario]
            ax.errorbar(
                g["calibration_fraction"] * 100,
                g[f"{metric}_mean"] * 100,
                yerr=g[f"{metric}_std"] * 100,
                color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                capsize=1.8, elinewidth=0.55, capthick=0.55, label=st["label"],
            )
        ax.set_xlabel("Labeled target-session calibration (%)")
        ax.set_ylabel(ylabel)
        ax.set_xticks([0, 10, 25, 50])
        ax.set_xlim(-4, 54)
        finish(ax, ROOT / "phase1_results" / filename, ylim=ylim, legend_cols=3)


def phase2() -> None:
    df = pd.read_csv(ROOT / "phase2_results" / "summary_results.csv")
    for mode in ["inductive", "transductive"]:
        dm = df[df["alignment_mode"] == mode]
        for metric, ylabel, filename, ylim in [
            ("accuracy", "Clean-test accuracy (%)", f"accuracy_vs_calibration_{mode}.png", None),
            ("asr", "Targeted ASR (%)", f"asr_vs_calibration_{mode}.png", (-4, 104)),
        ]:
            fig, ax = plt.subplots(figsize=(3.35, 2.05))
            for scenario in ["poisoned", "spectral_baseline", "crcs", "clean"]:
                g = dm[dm["scenario"] == scenario].sort_values("calibration_fraction")
                st = SCENARIO_STYLE[scenario]
                ax.plot(g["calibration_fraction"] * 100, g[f"{metric}_mean"] * 100,
                        color=st["color"], linestyle=st["linestyle"], marker=st["marker"],
                        label=st["label"])
            ax.set_xlabel("Labeled target-session calibration (%)")
            ax.set_ylabel(ylabel)
            ax.set_xticks([0, 10, 25, 50, 80])
            ax.set_xlim(-4, 84)
            finish(ax, ROOT / "phase2_results" / filename, ylim=ylim, legend_cols=2)


def phase3() -> None:
    df = pd.read_csv(ROOT / "phase3_results" / "robustness_summary.csv")
    for cal in [0.25, 0.50, 0.80]:
        dc = df[df["calibration_fraction"] == cal]
        cal_pct = int(round(cal * 100))

        # Manuscript-aligned budget sensitivity (same quantity as Fig. budget_sensitivity).
        fig, ax = plt.subplots(figsize=(3.35, 2.05))
        for pr in [0.05, 0.10, 0.15, 0.20]:
            g = dc[dc["poison_rate"] == pr].sort_values("removal_budget")
            st = POISON_STYLE[pr]
            ax.plot(g["removal_budget"] * 100, g["crcs_asr"] * 100,
                    color=st["color"], marker=st["marker"], label=st["label"])
        ax.set_xlabel("Removal budget B (%)")
        ax.set_ylabel("CRCS ASR (%)")
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlim(3.5, 26.5)
        finish(ax, ROOT / "phase3_results" / f"phase3_asr_budget_calibration_{cal_pct}.png",
               ylim=(-4, 104), primary_budget=True, legend_cols=4)

        # Utility-cost sensitivity, styled consistently with the manuscript's primary cost panel.
        fig, ax = plt.subplots(figsize=(3.35, 2.05))
        for pr in [0.05, 0.10, 0.15, 0.20]:
            g = dc[dc["poison_rate"] == pr].sort_values("removal_budget")
            st = POISON_STYLE[pr]
            ax.plot(g["removal_budget"] * 100, g["clean_accuracy_cost_pp"],
                    color=st["color"], marker=st["marker"], label=st["label"])
        ax.set_xlabel("Removal budget B (%)")
        ax.set_ylabel("Clean-accuracy cost (pp)")
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlim(3.5, 26.5)
        finish(ax, ROOT / "phase3_results" / f"phase3_accuracy_cost_calibration_{cal_pct}.png",
               zero_line=True, primary_budget=True, legend_cols=4)

    # Existing hard-subject diagnostic files: preserve their scientific content while
    # applying publication style. Each file shows all prevalence curves at 80% calibration.
    hd = pd.read_csv(ROOT / "phase3_results" / "hard_subjects_S3_S6.csv")
    for subject in [3, 6]:
        ds = hd[(hd["subject"] == subject) &
                (hd["calibration_fraction"] == 0.80) &
                (hd["scenario"] == "crcs")]
        fig, ax = plt.subplots(figsize=(3.35, 2.05))
        for pr in [0.05, 0.10, 0.15, 0.20]:
            g = ds[ds["poison_rate"] == pr].sort_values("removal_budget")
            st = POISON_STYLE[pr]
            ax.plot(g["removal_budget"] * 100, g["asr"] * 100,
                    color=st["color"], marker=st["marker"], label=st["label"])
        ax.set_xlabel("Removal budget B (%)")
        ax.set_ylabel("Targeted ASR (%)")
        ax.set_xticks([5, 10, 15, 20, 25])
        ax.set_xlim(3.5, 26.5)
        finish(ax, ROOT / "phase3_results" / f"phase3_hard_subject_S{subject}_calibration_80.png",
               ylim=(-5, 105), primary_budget=True, legend_cols=4)


def main() -> None:
    set_style()
    phase1()
    phase2()
    phase3()
    print("Rendered 14 manuscript-aligned RGB PNGs at 600 dpi.")


if __name__ == "__main__":
    main()
