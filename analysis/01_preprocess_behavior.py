"""
01_preprocess_behavior.py
=========================
Behavioral data preprocessing pipeline.
Reads per-subject raw CSV files, computes RT, accuracy, and confidence,
then outputs:
  1. Combined raw data (all subjects, all trials)
  2. Subject-level summary grouped by Heuristic_Type × SU_Level
  3. Demographic descriptive statistics

Consolidates functionality from:
  - behavior_analysis.py (raw data integration & grouped statistics)
  - behavior_stats_report.py (demographic & descriptive statistics)

Usage:
  cd analysis/
  python 01_preprocess_behavior.py
"""

import os
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path configuration (relative to this script)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

RAW_BEHAVIOR_DIR = PROJECT_ROOT / "data" / "raw" / "behavior"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output file paths
RAW_OUTPUT_CSV = OUTPUT_DIR / "behavior_raw_data.csv"
SUMMARY_OUTPUT_CSV = OUTPUT_DIR / "behavior_summary.csv"
SUMMARY_OUTPUT_XLSX = OUTPUT_DIR / "behavior_summary.xlsx"
DEMO_OUTPUT_CSV = OUTPUT_DIR / "demo_subjects.csv"
DESC_HV_CSV = OUTPUT_DIR / "desc_by_HV.csv"
DESC_HV_SU_CSV = OUTPUT_DIR / "desc_by_HV_SU.csv"


# =====================================================================
# Part 1: Read and integrate raw behavioral data
# =====================================================================

def read_raw_behavior_files():
    """Read all per-subject CSV files and combine into one DataFrame."""

    csv_files = sorted([f for f in os.listdir(RAW_BEHAVIOR_DIR) if f.endswith('.csv')])
    if not csv_files:
        print(f"[ERROR] No CSV files found in {RAW_BEHAVIOR_DIR}")
        sys.exit(1)

    print(f"Found {len(csv_files)} behavioral CSV files.")

    all_rows = []
    header = None

    for fname in csv_files:
        fpath = RAW_BEHAVIOR_DIR / fname
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                file_header = next(reader)

                if header is None:
                    header = file_header
                else:
                    # Verify consistent header
                    if file_header != header:
                        print(f"  [WARNING] Header mismatch in {fname}, attempting to align.")

                for row in reader:
                    if len(row) == len(header):
                        all_rows.append(row)
        except Exception as e:
            print(f"  [ERROR] Failed to read {fname}: {e}")
            continue

    df = pd.DataFrame(all_rows, columns=header)
    print(f"Combined raw data: {len(df)} rows × {len(df.columns)} columns")
    return df


def compute_rt_and_clean(df):
    """Compute RT from timestamps and clean data types."""

    # Ensure numeric columns
    numeric_cols = ['Stim_onset', 'Response_time', 'Correct', 'Confidence',
                    'SU_level', 'Age', 'BallExperience']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Compute RT in milliseconds: (Response_time - Stim_onset) * 1000
    if 'Stim_onset' in df.columns and 'Response_time' in df.columns:
        df['RT'] = (df['Response_time'] - df['Stim_onset']) * 1000.0
    elif 'RT' not in df.columns:
        print("[WARNING] Cannot compute RT: missing Stim_onset or Response_time columns.")

    # Rename for consistency
    rename_map = {}
    if 'HV_primary' in df.columns:
        rename_map['HV_primary'] = 'Heuristic_Type'
    if 'SU_level' in df.columns:
        rename_map['SU_level'] = 'SU_Level'
    df.rename(columns=rename_map, inplace=True)

    return df


def compute_grouped_summary(df):
    """Compute subject × Heuristic_Type × SU_Level summary statistics."""

    group_cols = ['SubjectID', 'Gender', 'Age', 'BallExperience',
                  'Heuristic_Type', 'SU_Level']

    # Only keep columns that exist
    group_cols = [c for c in group_cols if c in df.columns]

    summary_rows = []
    for keys, grp in df.groupby(group_cols):
        row = dict(zip(group_cols, keys))
        row['Mean_Accuracy'] = grp['Correct'].mean() if 'Correct' in grp else np.nan
        row['Mean_RT'] = grp['RT'].mean() if 'RT' in grp else np.nan
        row['Mean_Confidence'] = grp['Confidence'].mean() if 'Confidence' in grp else np.nan
        row['Trial_Count'] = len(grp)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Sort for readability
    sort_cols = [c for c in ['SubjectID', 'Heuristic_Type', 'SU_Level'] if c in summary_df.columns]
    if sort_cols:
        summary_df.sort_values(sort_cols, inplace=True)

    return summary_df


# =====================================================================
# Part 2: Demographic & descriptive statistics
# =====================================================================

def demographic_report(summary_df):
    """Generate demographic descriptive statistics from the summary data."""

    id_col = 'SubjectID'
    sex_col = 'Gender'
    age_col = 'Age'
    ball_col = 'BallExperience'

    subj_demo = summary_df[[id_col, sex_col, age_col, ball_col]].drop_duplicates(subset=[id_col])
    N = len(subj_demo)

    print(f"\n{'='*60}")
    print(f"Demographic Information (N = {N})")
    print(f"{'='*60}")

    # Gender distribution
    sex_counts = subj_demo[sex_col].value_counts(dropna=False)
    for k, v in sex_counts.items():
        print(f"  {k}: {v} ({v / N:.1%})")

    # Age
    for col, label in [(age_col, 'Age'), (ball_col, 'Ball Experience (0-100)')]:
        vals = pd.to_numeric(subj_demo[col], errors='coerce')
        print(f"  {label}: M = {vals.mean():.2f}, SD = {vals.std(ddof=1):.2f}, "
              f"range = [{vals.min():.0f}, {vals.max():.0f}]")

    subj_demo.to_csv(DEMO_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"  Saved: {DEMO_OUTPUT_CSV}")

    return subj_demo


def descriptive_stats(summary_df):
    """Compute descriptive statistics by HV and HV × SU conditions."""

    id_col = 'SubjectID'
    hv_col = 'Heuristic_Type'
    su_col = 'SU_Level'
    acc_col = 'Mean_Accuracy'
    rt_col = 'Mean_RT'
    conf_col = 'Mean_Confidence'

    for col in [acc_col, rt_col, conf_col]:
        summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')

    # (a) By HV condition
    by_subj_hv = (
        summary_df
        .groupby([id_col, hv_col], as_index=False)
        [[acc_col, rt_col, conf_col]]
        .mean()
    )
    hv_desc = (
        by_subj_hv
        .groupby(hv_col)
        [[acc_col, rt_col, conf_col]]
        .agg(['mean', 'std'])
    )
    hv_desc.to_csv(DESC_HV_CSV, encoding='utf-8-sig')
    print(f"\n  Descriptive stats by HV saved: {DESC_HV_CSV}")

    # (b) By HV × SU
    by_subj_hv_su = (
        summary_df
        .groupby([id_col, hv_col, su_col], as_index=False)
        [[acc_col, rt_col, conf_col]]
        .mean()
    )
    hv_su_desc = (
        by_subj_hv_su
        .groupby([hv_col, su_col])
        [[acc_col, rt_col, conf_col]]
        .agg(['mean', 'std'])
    )
    hv_su_desc.to_csv(DESC_HV_SU_CSV, encoding='utf-8-sig')
    print(f"  Descriptive stats by HV × SU saved: {DESC_HV_SU_CSV}")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 60)
    print("Step 1: Reading raw behavioral data")
    print("=" * 60)

    df = read_raw_behavior_files()
    df = compute_rt_and_clean(df)

    # Save combined raw data
    df.to_csv(RAW_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved combined raw data: {RAW_OUTPUT_CSV}")

    print("\n" + "=" * 60)
    print("Step 2: Computing grouped summary statistics")
    print("=" * 60)

    summary_df = compute_grouped_summary(df)
    summary_df.to_csv(SUMMARY_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"Saved summary: {SUMMARY_OUTPUT_CSV}")

    try:
        summary_df.to_excel(SUMMARY_OUTPUT_XLSX, index=False)
        print(f"Saved Excel: {SUMMARY_OUTPUT_XLSX}")
    except Exception:
        print("  (openpyxl not available; skipped Excel export)")

    print("\n" + "=" * 60)
    print("Step 3: Demographic & descriptive statistics")
    print("=" * 60)

    demographic_report(summary_df)
    descriptive_stats(summary_df)

    print("\n" + "=" * 60)
    print("Behavioral preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
