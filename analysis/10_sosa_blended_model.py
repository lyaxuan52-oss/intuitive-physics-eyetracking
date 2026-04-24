"""
10_sosa_blended_model.py
========================
Implementation of Sosa's Blended Algorithm for RT prediction.
Runs a grid search over (E, D, N) parameters and evaluates
how well the blended algorithm's execution time (tau) predicts
human mean RT via a linear Gaussian mapping.

Dependencies: numpy, pandas
Usage:
  cd analysis/
  python 10_sosa_blended_model.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_BASE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _BASE_DIR.parent

STIMULUS_CONFIG_PATH = _PROJECT_ROOT / "data" / "stimuli" / "stimulus_config.csv"
BEHAVIOR_RAW_PATH = _PROJECT_ROOT / "data" / "processed" / "behavior_raw_data.csv"
OUTPUT_DIR = _PROJECT_ROOT / "data" / "processed"


def _parse_true_trajectory(traj_str: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse True_trajectory JSON into xs, ys, ts arrays."""
    try:
        traj = json.loads(traj_str)
    except Exception:
        return np.array([]), np.array([]), np.array([])

    if not traj:
        return np.array([]), np.array([]), np.array([])

    xs = np.array([float(p["x"]) for p in traj], dtype=float)
    ys = np.array([float(p["y"]) for p in traj], dtype=float)
    ts = np.array([float(p.get("t", i)) for i, p in enumerate(traj)], dtype=float)
    return xs, ys, ts


def blended_tau_for_board(true_traj_str: str, E: float, D: float, N: int) -> float:
    """Run Sosa Blended Algorithm 1 for a single board, return execution time tau."""
    xs, ys, ts = _parse_true_trajectory(true_traj_str)
    T = len(xs)
    if T < 2:
        return np.nan

    dts = np.diff(ts)
    dt = float(np.median(dts[dts > 0])) if np.any(dts > 0) else 1.0

    idx = 0
    tau = 0.0

    while idx < T - 1:
        x_t, y_t = xs[idx], ys[idx]

        # Simulation branch
        sim_next_idx = min(idx + int(N), T - 1)
        sim_vec = np.array([xs[sim_next_idx] - x_t, ys[sim_next_idx] - y_t], dtype=float)

        # Abstraction branch
        if idx < T - 1:
            v_curr = np.array([xs[idx + 1] - x_t, ys[idx + 1] - y_t], dtype=float)
        else:
            v_curr = np.array([0.0, 1.0], dtype=float)

        v_norm = float(np.linalg.norm(v_curr))
        if v_norm == 0.0:
            abs_vec = np.array([0.0, D], dtype=float)
        else:
            abs_vec = (D / v_norm) * v_curr

        # Cosine similarity
        sim_norm = float(np.linalg.norm(sim_vec))
        abs_norm = float(np.linalg.norm(abs_vec))
        if sim_norm == 0.0 or abs_norm == 0.0:
            cos_sim = 1.0
        else:
            cos_sim = float(np.dot(sim_vec, abs_vec) / (sim_norm * abs_norm))
            cos_sim = max(-1.0, min(1.0, cos_sim))

        # Choose branch
        if cos_sim > E:
            speed = (v_norm / dt) if (dt > 0 and v_norm > 0) else 1.0
            step_time = abs_norm / speed if speed > 0 else dt
            tau += step_time

            target_t = ts[idx] + step_time
            new_idx = int(np.searchsorted(ts, target_t, side="right") - 1)
            if new_idx <= idx:
                new_idx = idx + 1
            if new_idx >= T - 1:
                break
            idx = new_idx
        else:
            step_frames = sim_next_idx - idx
            tau += step_frames * dt
            idx = sim_next_idx
            if idx >= T - 1:
                break

    return float(tau)


def _aggregate_human_rt_by_board() -> pd.DataFrame:
    """Aggregate human mean RT and accuracy by BoardID."""
    df = pd.read_csv(BEHAVIOR_RAW_PATH)
    if "BoardID" not in df.columns:
        raise KeyError("behavior_raw_data.csv missing BoardID column.")

    df["RT"] = pd.to_numeric(df["RT"], errors="coerce")
    df["Correct"] = pd.to_numeric(df["Correct"], errors="coerce")

    grouped = (
        df.dropna(subset=["RT", "Correct", "BoardID"])
        .groupby("BoardID", as_index=False)
        .agg(mean_RT_ms=("RT", "mean"), mean_correct=("Correct", "mean"),
             n_trials=("RT", "size"))
    )
    return grouped


def _compute_tau_for_all_boards(E: float, D: float, N: int) -> pd.DataFrame:
    """Compute tau for every board in stimulus_config."""
    stim = pd.read_csv(STIMULUS_CONFIG_PATH)
    rows: List[Dict[str, float]] = []
    for _, row in stim.iterrows():
        tau = blended_tau_for_board(row["True_trajectory"], E=E, D=D, N=N)
        rows.append({"BoardID": row["BoardID"], "tau": tau})
    return pd.DataFrame(rows)


def _fit_rt_mapping(tau_df, beh_df):
    """Fit linear Gaussian mapping: RT_hat = beta0 + beta1 * tau."""
    df = pd.merge(beh_df, tau_df, on="BoardID", how="inner").dropna(subset=["mean_RT_ms", "tau"])
    if df.empty:
        raise RuntimeError("No overlapping BoardIDs between behavior and stimulus data.")

    x = df["tau"].to_numpy(dtype=float)
    y = df["mean_RT_ms"].to_numpy(dtype=float)
    X = np.column_stack([np.ones_like(x), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta0, beta1 = beta_hat

    resid = y - X @ beta_hat
    n = len(y)
    lam = float(np.sum(resid ** 2) / max(n, 1))
    if lam <= 0:
        lam = 1e-6

    return df, float(beta0), float(beta1), lam


def evaluate_blended_model(E: float, D: float, N: int):
    """Evaluate blended model fit quality for given (E, D, N)."""
    beh_df = _aggregate_human_rt_by_board()
    tau_df = _compute_tau_for_all_boards(E=E, D=D, N=N)
    merged_df, beta0, beta1, lam = _fit_rt_mapping(tau_df, beh_df)

    y = merged_df["mean_RT_ms"].to_numpy(dtype=float)
    tau = merged_df["tau"].to_numpy(dtype=float)
    mu = beta0 + beta1 * tau
    n = len(y)

    mse = float(np.mean((y - mu) ** 2))
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * lam) + ((y - mu) ** 2) / lam)

    k = 6
    bic = -2.0 * ll + k * np.log(n)
    aic = -2.0 * ll + 2.0 * k

    stats = {
        "E": float(E), "D": float(D), "N": int(N),
        "beta0": beta0, "beta1": beta1, "lambda": float(lam),
        "loglik": float(ll), "BIC": float(bic), "AIC": float(aic),
        "MSE": float(mse), "n_scenes": int(n),
    }

    merged_df = merged_df.copy()
    merged_df["RT_hat"] = mu
    return stats, merged_df


def grid_search_blended() -> pd.DataFrame:
    """Grid search over (E, D, N) parameter space."""
    E_values = [0.98, 0.985, 0.99]
    D_values = [150.0, 250.0, 350.0]
    N_values = [5, 10, 15]

    results: List[Dict[str, float]] = []
    for E in E_values:
        for D in D_values:
            for N in N_values:
                print(f"Evaluating Blended model: E={E}, D={D}, N={N}...")
                stats, _ = evaluate_blended_model(E=E, D=D, N=N)
                results.append(stats)

    res_df = pd.DataFrame(results).sort_values("BIC")
    out_path = OUTPUT_DIR / "blended_model_grid_search.csv"
    res_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"\nGrid search complete. Results saved to: {out_path}")
    print("Top 5 parameter sets by BIC:")
    print(res_df.head())
    return res_df


def main():
    grid_search_blended()


if __name__ == "__main__":
    main()
