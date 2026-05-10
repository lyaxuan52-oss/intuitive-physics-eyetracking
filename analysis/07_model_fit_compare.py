"""
07_model_fit_compare.py
=======================
Bayesian computational modeling: fit and compare four cognitive models
of intuitive physics judgment using trial-level behavioral data.

Models:
  1. Coexist (mixed heuristic + simulation with continuous weighting)
  2. Pure Heuristic (HV-driven logistic)
  3. Pure Simulation (SU-driven logistic)
  4. Sosa Switch (trajectory-based gate between simulation and heuristic)

Outputs model comparison table (WAIC, LOO, AIC, BIC, MSE).

Consolidates: model_fit_compare.py + mixed_strategy_model.py

Dependencies: pymc, arviz, pandas, numpy
Usage:
  cd analysis/
  python 07_model_fit_compare.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent

RAW_BEHAVIOR_PATH = _PROJECT_ROOT / "data" / "processed" / "behavior_raw_data.csv"
STIMULUS_CONFIG_PATH = _PROJECT_ROOT / "data" / "stimuli" / "stimulus_config.csv"


# ---------------------------------------------------------------------------
# Helper: compute vertical cosines for Sosa gate
# ---------------------------------------------------------------------------

def _compute_vertical_cosines(true_traj_json: str) -> np.ndarray:
    """Compute cosine similarity between trajectory steps and vertical heuristic."""
    try:
        traj = json.loads(true_traj_json)
    except Exception:
        return np.array([], dtype=float)

    if not traj or len(traj) < 2:
        return np.array([], dtype=float)

    positions = [(float(p["x"]), float(p["y"])) for p in traj]
    cos_vals: list[float] = []
    for (x0, y0), (x1, y1) in zip(positions[:-1], positions[1:]):
        dx = x1 - x0
        dy = y1 - y0
        norm = np.sqrt(dx * dx + dy * dy)
        if norm == 0.0:
            cos = 1.0
        else:
            cos = dy / norm
        cos = max(-1.0, min(1.0, cos))
        cos_vals.append(cos)

    return np.array(cos_vals, dtype=float)


def _build_sosa_gate_by_board(threshold: float = 0.99) -> dict[str, float]:
    """For each BoardID, compute a binary gate based on trajectory-heuristic cosine."""
    stim_df = pd.read_csv(STIMULUS_CONFIG_PATH)
    gate_map: dict[str, float] = {}

    for _, row in stim_df.iterrows():
        board_id = row["BoardID"]
        cos_vals = _compute_vertical_cosines(row["True_trajectory"])
        if cos_vals.size == 0:
            gate = 0.0
        else:
            gate = 1.0 if np.all(cos_vals >= threshold) else 0.0
        gate_map[board_id] = float(gate)

    return gate_map


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_BEHAVIOR_PATH)

    if "BoardID" not in df.columns:
        raise KeyError("behavior_raw_data.csv missing BoardID column.")

    gate_map = _build_sosa_gate_by_board(threshold=0.99)
    df["SOSA_gate"] = df["BoardID"].map(gate_map).fillna(0.0).astype(float)

    df = df[df["Heuristic_Type"].isin(["ambiguous", "congruent"])].copy()
    df = df[df["SU_Level"] > 0].copy()

    df = df.rename(columns={"SubjectID": "subj", "SU_Level": "SU"})
    df["HV"] = df["Heuristic_Type"].map({"ambiguous": 0, "congruent": 1}).astype(int)
    df["Correct"] = df["Correct"].astype("int8")
    df["SU"] = df["SU"].astype(float)

    return df[["subj", "HV", "SU", "Correct", "SOSA_gate"]]


def _prepare_arrays(df: pd.DataFrame):
    subj_codes, subj_idx = np.unique(df["subj"], return_inverse=True)
    HV = df["HV"].astype(int).values
    SU = df["SU"].astype(float).values
    y = df["Correct"].astype("int8").values
    SU_z = (SU - SU.mean()) / SU.std()
    return subj_codes, subj_idx, HV, SU, SU_z, y


def _prepare_arrays_with_gate(df: pd.DataFrame):
    subj_codes, subj_idx, HV, SU, SU_z, y = _prepare_arrays(df)
    gate = df["SOSA_gate"].astype(float).values
    return subj_codes, subj_idx, HV, SU, SU_z, y, gate


# ---------------------------------------------------------------------------
# Model 1: Coexist (heuristic + simulation, continuous weighting)
# ---------------------------------------------------------------------------

def build_coexist_model(df):
    subj_codes, subj_idx, HV, SU, SU_z, y = _prepare_arrays(df)
    n_subj = len(subj_codes)

    with pm.Model() as model:
        mu_logit_sim = pm.Normal("mu_logit_sim", 0.0, 1.0)
        sigma_logit_sim = pm.Exponential("sigma_logit_sim", 1.0)
        logit_sim_s = pm.Normal("logit_sim_s", mu_logit_sim, sigma_logit_sim, shape=n_subj)

        mu_logit_heur = pm.Normal("mu_logit_heur", 0.0, 1.0)
        sigma_logit_heur = pm.Exponential("sigma_logit_heur", 1.0)
        logit_heur_s = pm.Normal("logit_heur_s", mu_logit_heur, sigma_logit_heur, shape=n_subj)

        beta_SU_sim = pm.Normal("beta_SU_sim", 0.0, 1.0)

        sim_logit = logit_sim_s[subj_idx] + beta_SU_sim * SU_z
        p_sim = pm.Deterministic("p_sim", pm.math.sigmoid(sim_logit))
        p_heur = pm.Deterministic("p_heur", pm.math.sigmoid(logit_heur_s[subj_idx]))

        gamma0 = pm.Normal("gamma0", 0.0, 1.0)
        gamma_HV = pm.Normal("gamma_HV", 1.5, 1.0)
        gamma_SU = pm.Normal("gamma_SU", 0.0, 1.0)
        sigma_gamma_subj = pm.Exponential("sigma_gamma_subj", 1.0)
        gamma_subj = pm.Normal("gamma_subj", 0.0, sigma_gamma_subj, shape=n_subj)

        logit_wH = gamma0 + gamma_HV * HV + gamma_SU * SU_z + gamma_subj[subj_idx]
        wH = pm.Deterministic("wH", pm.math.sigmoid(logit_wH))
        wS = pm.Deterministic("wS", 1.0 - wH)

        p = pm.Deterministic("p", wH * p_heur + wS * p_sim)
        pm.Bernoulli("y_obs", p=p, observed=y)

    return model


def fit_coexist_model(df, draws=2000, tune=2000, chains=4,
                      target_accept=0.9, random_seed=123):
    model = build_coexist_model(df)

    with model:

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=random_seed,
                          idata_kwargs={"log_likelihood": True})

    az.to_netcdf(idata, str(_BASE_DIR / "idata_coexist.nc"))
    return model, idata


# ---------------------------------------------------------------------------
# Model 2: Pure Heuristic
# ---------------------------------------------------------------------------

def build_pure_heuristic_model(df):
    subj_codes, subj_idx, HV, _SU, _SU_z, y = _prepare_arrays(df)
    n_subj = len(subj_codes)

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", 0.0, 1.0)
        sigma_alpha = pm.Exponential("sigma_alpha", 1.0)
        alpha_s = pm.Normal("alpha_s", mu_alpha, sigma_alpha, shape=n_subj)
        beta_HV = pm.Normal("beta_HV", 0.0, 1.0)

        logit_p = alpha_s[subj_idx] + beta_HV * HV
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Bernoulli("y_obs", p=p, observed=y)

    return model


def fit_pure_heuristic_model(df, draws=2000, tune=2000, chains=4,
                              target_accept=0.9, random_seed=123):
    model = build_pure_heuristic_model(df)

    with model:

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=random_seed,
                          idata_kwargs={"log_likelihood": True})

    az.to_netcdf(idata, str(_BASE_DIR / "idata_pure_heuristic.nc"))
    return model, idata


# ---------------------------------------------------------------------------
# Model 3: Pure Simulation
# ---------------------------------------------------------------------------

def build_pure_simulation_model(df):
    subj_codes, subj_idx, _HV, _SU, SU_z, y = _prepare_arrays(df)
    n_subj = len(subj_codes)

    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", 0.0, 1.0)
        sigma_alpha = pm.Exponential("sigma_alpha", 1.0)
        alpha_s = pm.Normal("alpha_s", mu_alpha, sigma_alpha, shape=n_subj)
        beta_SU = pm.Normal("beta_SU", 0.0, 1.0)

        logit_p = alpha_s[subj_idx] + beta_SU * SU_z
        p = pm.Deterministic("p", pm.math.sigmoid(logit_p))
        pm.Bernoulli("y_obs", p=p, observed=y)

    return model


def fit_pure_simulation_model(df, draws=2000, tune=2000, chains=4,
                               target_accept=0.9, random_seed=123):
    model = build_pure_simulation_model(df)

    with model:

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=random_seed,
                          idata_kwargs={"log_likelihood": True})

    az.to_netcdf(idata, str(_BASE_DIR / "idata_pure_simulation.nc"))
    return model, idata


# ---------------------------------------------------------------------------
# Model 4: Sosa Switch (trajectory-based gate)
# ---------------------------------------------------------------------------

def build_sosa_switch_model(df):
    subj_codes, subj_idx, _HV, _SU, _SU_z, y, gate = _prepare_arrays_with_gate(df)
    n_subj = len(subj_codes)

    with pm.Model() as model:
        mu_logit_sim = pm.Normal("mu_logit_sim", 0.0, 1.0)
        sigma_logit_sim = pm.Exponential("sigma_logit_sim", 1.0)
        logit_sim_s = pm.Normal("logit_sim_s", mu_logit_sim, sigma_logit_sim, shape=n_subj)

        mu_logit_heur = pm.Normal("mu_logit_heur", 0.0, 1.0)
        sigma_logit_heur = pm.Exponential("sigma_logit_heur", 1.0)
        logit_heur_s = pm.Normal("logit_heur_s", mu_logit_heur, sigma_logit_heur, shape=n_subj)

        p_sim = pm.Deterministic("p_sim", pm.math.sigmoid(logit_sim_s[subj_idx]))
        p_heur = pm.Deterministic("p_heur", pm.math.sigmoid(logit_heur_s[subj_idx]))

        gate_data = pm.Data("gate_data", gate.astype(float))
        p = pm.Deterministic("p", (1.0 - gate_data) * p_sim + gate_data * p_heur)
        pm.Bernoulli("y_obs", p=p, observed=y)

    return model


def fit_sosa_switch_model(df, draws=2000, tune=2000, chains=4,
                           target_accept=0.9, random_seed=123):
    model = build_sosa_switch_model(df)

    with model:

        idata = pm.sample(draws=draws, tune=tune, chains=chains,
                          target_accept=target_accept, random_seed=random_seed,
                          idata_kwargs={"log_likelihood": True})

    az.to_netcdf(idata, str(_BASE_DIR / "idata_sosa_blended.nc"))
    return model, idata


def _count_free_parameters(model) -> int:
    point = model.initial_point()
    return int(sum(np.asarray(value).size for value in point.values()))


def _extract_total_loglik_draws(idata) -> np.ndarray:
    if hasattr(idata, "log_likelihood") and "y_obs" in idata.log_likelihood:
        log_lik = idata.log_likelihood["y_obs"].values
    else:
        obs = idata.observed_data["y_obs"].values.astype(int).ravel()
        p_samples = idata.posterior["p"].values
        y_broadcast = obs[None, None, :]
        log_lik = np.where(
            y_broadcast == 1,
            np.log(p_samples + 1e-12),
            np.log(1.0 - p_samples + 1e-12),
        )
    return log_lik.sum(axis=-1)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def main():
    df = load_raw_data()
    print(df.head())

    idatas = {}
    model_builders = {
        "coexist": build_coexist_model,
        "pure_heuristic": build_pure_heuristic_model,
        "pure_simulation": build_pure_simulation_model,
        "sosa_blended": build_sosa_switch_model,
    }
    n_obs = len(df)

    sample_kwargs = dict(
        draws=int(os.getenv("MODEL_COMPARE_DRAWS", "800")),
        tune=int(os.getenv("MODEL_COMPARE_TUNE", "800")),
        chains=int(os.getenv("MODEL_COMPARE_CHAINS", "2")),
        target_accept=float(os.getenv("MODEL_COMPARE_TARGET_ACCEPT", "0.9")),
    )

    def _load_or_fit(name, fit_func, nc_filename):
        nc_path = _BASE_DIR / nc_filename
        if nc_path.exists():
            print(f"Loading existing samples for {name} from {nc_path}")
            return None, az.from_netcdf(nc_path)
        print(f"Fitting model {name} from scratch...")
        return fit_func(df, **sample_kwargs)

    _, idata_coexist = _load_or_fit("coexist", fit_coexist_model, "idata_coexist.nc")
    idatas["coexist"] = idata_coexist

    _, idata_heur = _load_or_fit("pure_heuristic", fit_pure_heuristic_model, "idata_pure_heuristic.nc")
    idatas["pure_heuristic"] = idata_heur

    _, idata_sim = _load_or_fit("pure_simulation", fit_pure_simulation_model, "idata_pure_simulation.nc")
    idatas["pure_simulation"] = idata_sim

    _, idata_sosa = _load_or_fit("sosa_blended", fit_sosa_switch_model, "idata_sosa_blended.nc")
    idatas["sosa_blended"] = idata_sosa

    # Compute comparison metrics
    results = []
    for name, idata in idatas.items():
        obs = idata.observed_data["y_obs"].values.astype(int).ravel()
        total_loglik_draws = _extract_total_loglik_draws(idata)
        if hasattr(idata, "log_likelihood") and "y_obs" in idata.log_likelihood:
            log_lik = idata.log_likelihood["y_obs"].values
        else:
            p_samples = idata.posterior["p"].values
            y_broadcast = obs[None, None, :]
            log_lik = np.where(
                y_broadcast == 1,
                np.log(p_samples + 1e-12),
                np.log(1.0 - p_samples + 1e-12),
            )

        idata_ll = az.from_dict(log_likelihood={"y_obs": log_lik})
        waic_res = az.waic(idata_ll)
        loo_res = az.loo(idata_ll, reff=1.0)

        elpd_waic = float(waic_res.elpd_waic)
        p_waic_val = float(waic_res.p_waic)
        waic_val = -2.0 * elpd_waic

        elpd_loo = float(loo_res.elpd_loo)
        p_loo_val = float(loo_res.p_loo)
        loo_ic = -2.0 * elpd_loo

        model = model_builders[name](df)
        k_params = _count_free_parameters(model)
        max_loglik = float(np.max(total_loglik_draws))
        aic = 2.0 * k_params - 2.0 * max_loglik
        bic = k_params * np.log(n_obs) - 2.0 * max_loglik

        p_mean = idata.posterior["p"].mean(dim=("chain", "draw")).values.astype(float).ravel()
        mse = float(np.mean((obs.astype(float) - p_mean) ** 2))

        print(
            f"Model {name}: WAIC={waic_val:.2f}, LOO={loo_ic:.2f}, MSE={mse:.4f}, "
            f"k={k_params}, max_loglik={max_loglik:.2f}, AIC={aic:.2f}, BIC={bic:.2f}"
        )

        results.append({
            "model": name, "waic": waic_val, "p_waic": p_waic_val,
            "loo": loo_ic, "p_loo": p_loo_val, "mse": mse,
            "k_params": int(k_params), "max_loglik": float(max_loglik),
            "aic": float(aic), "bic": float(bic),
        })

    comp_df = pd.DataFrame(results).sort_values("waic")
    out_path = _PROJECT_ROOT / "data" / "processed" / "model_comparison_waic_loo.csv"
    comp_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nModel comparison saved to: {out_path}")
    print(comp_df)


if __name__ == "__main__":
    main()
