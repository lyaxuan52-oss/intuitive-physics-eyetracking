"""
06_eye_metrics_visualization.py
================================
Visualization script for eye-tracking metrics and behavior data.
Generates publication-quality figures for:
  - H1: Trajectory overlap and tracking time by HV x SU
  - H2: ROI gaze proportions, durations, fixation counts
  - H3: CU region metrics for high-difficulty trials
  - Behavioral accuracy/RT relationships

Consolidated from: eye_metrics_visualizations.py

Usage:
  cd analysis/
  python 06_eye_metrics_visualization.py
"""

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_eye_data():
    """Load eye metrics subject means."""
    fpath = DATA_DIR / "eye_metrics_subject_means.csv"
    df = pd.read_csv(fpath)
    df["HV_primary"] = df["HV_primary"].str.strip()
    df = df[df["HV_primary"].isin(["congruent", "ambiguous"])].copy()
    return df


def load_behavior_data():
    """Load behavioral summary data."""
    fpath = DATA_DIR / "behavior_summary.csv"
    if not fpath.exists():
        return None
    df = pd.read_csv(fpath)
    df["Heuristic_Type"] = df["Heuristic_Type"].str.strip()
    return df


# =====================================================================
# H1: Trajectory overlap & tracking time
# =====================================================================

def plot_trajectory_overlap_by_SU(eye):
    """Line plot: trajectory overlap (I/U) by SU level, split by HV."""
    summary = (
        eye.groupby(["HV_primary", "SU_level"])["I_over_U"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for hv, marker in [("congruent", "o"), ("ambiguous", "s")]:
        sub = summary[summary["HV_primary"] == hv]
        ax.errorbar(sub["SU_level"], sub["mean"], yerr=sub["sem"],
                    marker=marker, label=hv, capsize=3, linewidth=1.5)

    ax.set_xlabel("SU Level")
    ax.set_ylabel("Trajectory Overlap (I/U)")
    ax.set_title("H1: Trajectory Overlap by HV x SU")
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "H1_trajectory_overlap_SU.png")
    plt.close(fig)
    print("Saved: H1_trajectory_overlap_SU.png")


def plot_tracking_time_by_SU(eye):
    """Line plot: tracking time by SU level, split by HV."""
    summary = (
        eye.groupby(["HV_primary", "SU_level"])["Track_ms"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for hv, marker in [("congruent", "o"), ("ambiguous", "s")]:
        sub = summary[summary["HV_primary"] == hv]
        ax.errorbar(sub["SU_level"], sub["mean"], yerr=sub["sem"],
                    marker=marker, label=hv, capsize=3, linewidth=1.5)

    ax.set_xlabel("SU Level")
    ax.set_ylabel("Tracking Time (ms)")
    ax.set_title("H1: Tracking Time by HV x SU")
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "H1_tracking_time_SU.png")
    plt.close(fig)
    print("Saved: H1_tracking_time_SU.png")


# =====================================================================
# H2: ROI gaze metrics
# =====================================================================

def plot_roi_proportions(eye):
    """Grouped bar plot: gaze proportion for PT, CU, Catcher by HV."""
    rois = ["Prop_PT", "Prop_CU", "Prop_catcher"]
    roi_labels = ["PT", "CU", "Catcher"]

    rows = []
    for _, r in eye.iterrows():
        for col, label in zip(rois, roi_labels):
            rows.append({
                "subject_id": r["subject_id"],
                "HV": r["HV_primary"],
                "ROI": label,
                "Proportion": r[col],
            })
    df_long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df_long, x="ROI", y="Proportion", hue="HV",
                errorbar="se", ax=ax)
    ax.set_title("H2: ROI Gaze Proportion by HV")
    ax.set_ylabel("Gaze Proportion")
    fig.savefig(FIGURES_DIR / "H2_ROI_proportions.png")
    plt.close(fig)
    print("Saved: H2_ROI_proportions.png")


def plot_roi_durations(eye):
    """Grouped bar plot: fixation duration for PT, CU, Catcher by HV."""
    rois = ["Duration_PT", "Duration_CU", "Duration_catcher"]
    roi_labels = ["PT", "CU", "Catcher"]

    rows = []
    for _, r in eye.iterrows():
        for col, label in zip(rois, roi_labels):
            rows.append({
                "subject_id": r["subject_id"],
                "HV": r["HV_primary"],
                "ROI": label,
                "Duration": r[col],
            })
    df_long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df_long, x="ROI", y="Duration", hue="HV",
                errorbar="se", ax=ax)
    ax.set_title("H2: ROI Fixation Duration by HV")
    ax.set_ylabel("Duration (ms)")
    fig.savefig(FIGURES_DIR / "H2_ROI_durations.png")
    plt.close(fig)
    print("Saved: H2_ROI_durations.png")


def plot_roi_fixations(eye):
    """Grouped bar plot: fixation count for PT, CU, Catcher by HV."""
    rois = ["Fixations_PT", "Fixations_CU", "Fixations_catcher"]
    roi_labels = ["PT", "CU", "Catcher"]

    rows = []
    for _, r in eye.iterrows():
        for col, label in zip(rois, roi_labels):
            if col in r.index:
                rows.append({
                    "subject_id": r["subject_id"],
                    "HV": r["HV_primary"],
                    "ROI": label,
                    "Fixations": r[col],
                })
    if not rows:
        print("[SKIP] Fixation count columns not found")
        return

    df_long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(data=df_long, x="ROI", y="Fixations", hue="HV",
                errorbar="se", ax=ax)
    ax.set_title("H2: ROI Fixation Count by HV")
    ax.set_ylabel("Fixation Count")
    fig.savefig(FIGURES_DIR / "H2_ROI_fixations.png")
    plt.close(fig)
    print("Saved: H2_ROI_fixations.png")


# =====================================================================
# H3: CU metrics for high-difficulty trials
# =====================================================================

def plot_CU_high_difficulty(eye):
    """CU gaze proportion for high SU (3-5), by HV."""
    high = eye[eye["SU_level"].isin([3, 4, 5])].copy()
    if high.empty:
        print("[SKIP] No high-difficulty data for H3 CU plot")
        return

    summary = (
        high.groupby(["HV_primary", "SU_level"])["Prop_CU"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    for hv, marker in [("congruent", "o"), ("ambiguous", "s")]:
        sub = summary[summary["HV_primary"] == hv]
        ax.errorbar(sub["SU_level"], sub["mean"], yerr=sub["sem"],
                    marker=marker, label=hv, capsize=3, linewidth=1.5)

    ax.set_xlabel("SU Level")
    ax.set_ylabel("CU Gaze Proportion")
    ax.set_title("H3: CU Gaze Proportion (High Difficulty)")
    ax.legend(frameon=False)
    fig.savefig(FIGURES_DIR / "H3_CU_proportion_high.png")
    plt.close(fig)
    print("Saved: H3_CU_proportion_high.png")


# =====================================================================
# Behavioral plots
# =====================================================================

def plot_behavior_accuracy_rt(beh):
    """Accuracy and RT by HV x SU."""
    if beh is None:
        print("[SKIP] No behavioral data available")
        return

    beh_2 = beh[beh["Heuristic_Type"].isin(["congruent", "ambiguous"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    acc_summary = (
        beh_2.groupby(["Heuristic_Type", "SU_Level"])["Mean_Accuracy"]
        .agg(["mean", "sem"]).reset_index()
    )
    ax = axes[0]
    for hv, marker in [("congruent", "o"), ("ambiguous", "s")]:
        sub = acc_summary[acc_summary["Heuristic_Type"] == hv]
        ax.errorbar(sub["SU_Level"], sub["mean"], yerr=sub["sem"],
                    marker=marker, label=hv, capsize=3, linewidth=1.5)
    ax.set_xlabel("SU Level")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by HV x SU")
    ax.legend(frameon=False)

    # RT
    rt_summary = (
        beh_2.groupby(["Heuristic_Type", "SU_Level"])["Mean_RT"]
        .agg(["mean", "sem"]).reset_index()
    )
    ax = axes[1]
    for hv, marker in [("congruent", "o"), ("ambiguous", "s")]:
        sub = rt_summary[rt_summary["Heuristic_Type"] == hv]
        ax.errorbar(sub["SU_Level"], sub["mean"], yerr=sub["sem"],
                    marker=marker, label=hv, capsize=3, linewidth=1.5)
    ax.set_xlabel("SU Level")
    ax.set_ylabel("RT (ms)")
    ax.set_title("RT by HV x SU")
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "Behavior_Accuracy_RT.png")
    plt.close(fig)
    print("Saved: Behavior_Accuracy_RT.png")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("Eye Metrics & Behavior Visualization")
    print("=" * 60)

    eye = load_eye_data()
    beh = load_behavior_data()

    print(f"\nEye data: {len(eye)} rows, {len(eye['subject_id'].unique())} subjects")
    if beh is not None:
        print(f"Behavior data: {len(beh)} rows")

    # H1
    plot_trajectory_overlap_by_SU(eye)
    plot_tracking_time_by_SU(eye)

    # H2
    plot_roi_proportions(eye)
    plot_roi_durations(eye)
    plot_roi_fixations(eye)

    # H3
    plot_CU_high_difficulty(eye)

    # Behavior
    plot_behavior_accuracy_rt(beh)

    print("\n" + "=" * 60)
    print(f"All figures saved to: {FIGURES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
