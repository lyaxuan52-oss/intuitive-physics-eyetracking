"""
09_coexist_plot_curves.py
=========================
Plot simulation weight (wS) curves from coexist model condition summary.

Usage:
  cd analysis/
  python 09_coexist_plot_curves.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = _PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_wS_by_SU_and_HV():
    """Plot simulation weight wS by SU bin and HV condition."""
    csv_path = DATA_DIR / "coexist_condition_summary.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found. Run 08_coexist_model_analysis.py first.")
        return

    df = pd.read_csv(csv_path)

    su_order = ["low", "mid", "high"]
    df["SU_bin"] = pd.Categorical(df["SU_bin"], categories=su_order, ordered=True)
    df = df.sort_values(["HV", "SU_bin"])

    fig, ax = plt.subplots(figsize=(6, 4))

    for hv_value, label, color, marker in [
        (0, "ambiguous", "C0", "o"),
        (1, "congruent", "C1", "s"),
    ]:
        sub = df[df["HV"] == hv_value]
        ax.plot(sub["SU_bin"], sub["wS_mean"],
                label=label, color=color, marker=marker)

    ax.set_xlabel("SU bin")
    ax.set_ylabel("Posterior mean simulation weight wS")
    ax.set_title("Coexist model: wS by SU bin and HV")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    out_path = FIGURES_DIR / "coexist_wS_SU_HV.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    print(f"Saved wS curve figure to {out_path}")


if __name__ == "__main__":
    plot_wS_by_SU_and_HV()
