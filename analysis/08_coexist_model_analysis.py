"""
08_coexist_model_analysis.py
============================
Thorough sampling and analysis of the coexist model (the winning model).
Outputs group-level and subject-level parameter summaries,
trial-level predictions, and condition-level summaries.

Requires 07_model_fit_compare.py (imports load_raw_data and fit_coexist_model).

Usage:
  cd analysis/
  python 08_coexist_model_analysis.py
"""

import arviz as az
import numpy as np
import pandas as pd
from pathlib import Path

import sys
from importlib import import_module

_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent

# Import from sibling module
sys.path.insert(0, str(_BASE_DIR))
_model_mod = import_module("07_model_fit_compare")
load_raw_data = _model_mod.load_raw_data
fit_coexist_model = _model_mod.fit_coexist_model

OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """Fit coexist model with thorough sampling and export parameter summaries."""
    df = load_raw_data()
    print("Loaded behavioral data:")
    print(df.head())

    sample_kwargs = dict(draws=2000, tune=2000, chains=4, target_accept=0.95, random_seed=123)

    out_nc = _BASE_DIR / "idata_coexist_full.nc"
    if out_nc.exists():
        print(f"Loading existing coexist samples from {out_nc}")
        idata = az.from_netcdf(out_nc)
    else:
        print("Fitting coexist model with thorough sampling...")
        _model, idata = fit_coexist_model(df, **sample_kwargs)
        az.to_netcdf(idata, out_nc)
        print(f"Saved full coexist InferenceData to {out_nc}")

    # 1) Group-level parameter summary
    group_vars = [
        "mu_logit_sim", "sigma_logit_sim",
        "mu_logit_heur", "sigma_logit_heur",
        "beta_SU_sim", "gamma0", "gamma_HV", "gamma_SU", "sigma_gamma_subj",
    ]
    summary_group = az.summary(idata, var_names=group_vars)
    out_group = OUTPUT_DIR / "coexist_group_params_summary.csv"
    summary_group.to_csv(out_group)
    print(f"Saved group-level summary to {out_group}")

    # 2) Subject-level parameters
    subj_vars = ["logit_sim_s", "logit_heur_s", "gamma_subj"]
    summary_subj = az.summary(idata, var_names=subj_vars)
    out_subj = OUTPUT_DIR / "coexist_subject_params_summary.csv"
    summary_subj.to_csv(out_subj)
    print(f"Saved subject-level summary to {out_subj}")

    # 3) Trial-level posterior predictions
    print("Computing posterior mean trial-level predictions...")
    df_pred = df.copy().reset_index(drop=True)

    p_post = idata.posterior["p"].mean(dim=("chain", "draw")).values.astype(float).ravel()
    df_pred["p_mean"] = p_post

    if "wH" in idata.posterior and "wS" in idata.posterior:
        df_pred["wH_mean"] = idata.posterior["wH"].mean(dim=("chain", "draw")).values.ravel()
        df_pred["wS_mean"] = idata.posterior["wS"].mean(dim=("chain", "draw")).values.ravel()

    if "p_sim" in idata.posterior:
        df_pred["p_sim_mean"] = idata.posterior["p_sim"].mean(dim=("chain", "draw")).values.ravel()
    if "p_heur" in idata.posterior:
        df_pred["p_heur_mean"] = idata.posterior["p_heur"].mean(dim=("chain", "draw")).values.ravel()

    trial_out = OUTPUT_DIR / "coexist_trial_level_predictions.csv"
    df_pred.to_csv(trial_out, index=False)
    print(f"Saved trial-level predictions to {trial_out}")

    # 4) Condition-level summary
    print("Computing condition-level summaries...")
    su_bins = pd.qcut(df_pred["SU"], q=3, labels=["low", "mid", "high"])
    df_pred["SU_bin"] = su_bins

    value_cols = [c for c in ["p_mean", "wH_mean", "wS_mean", "p_sim_mean", "p_heur_mean"]
                  if c in df_pred.columns]

    cond_summary = df_pred.groupby(["HV", "SU_bin"])[value_cols].mean().reset_index()
    cond_out = OUTPUT_DIR / "coexist_condition_summary.csv"
    cond_summary.to_csv(cond_out, index=False)
    print(f"Saved condition summary to {cond_out}")

    subj_cond = df_pred.groupby(["subj", "HV", "SU_bin"])[value_cols].mean().reset_index()
    subj_cond_out = OUTPUT_DIR / "coexist_subject_condition_summary.csv"
    subj_cond.to_csv(subj_cond_out, index=False)
    print(f"Saved subject-by-condition summary to {subj_cond_out}")


if __name__ == "__main__":
    main()
