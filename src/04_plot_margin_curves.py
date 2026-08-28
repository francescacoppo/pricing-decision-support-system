import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

KINK_CONFIG = {
    "fmcg": {"threshold": 0.08, "extra_elasticity": -2.5},
    "durevoli": {"threshold": 0.15, "extra_elasticity": -1.8},
    "premium": {"threshold": 0.25, "extra_elasticity": -1.2},
}

def demand_multiplier_with_kink(price_change_pct, eps, category):
    cfg = KINK_CONFIG[category]
    base = (1 + price_change_pct) ** eps
    beyond = price_change_pct > cfg["threshold"]
    extra = np.where(beyond, price_change_pct - cfg["threshold"], 0.0)
    kink_factor = np.where(beyond, (1 + extra) ** cfg["extra_elasticity"], 1.0)
    return base * kink_factor

comparison = pd.read_csv(DATA_DIR / "elasticity_comparison.csv", index_col=0)
best_elasticity = {}
for cat in comparison.index:
    row = comparison.loc[cat]
    if row["errore_assoluto_loglog"] <= row["errore_assoluto_rf"]:
        best_elasticity[cat] = row["regressione_log_log"]
    else:
        best_elasticity[cat] = row["random_forest (partial dependence)"]

product_master = pd.read_csv(DATA_DIR / "product_master.csv")
typical = product_master.groupby("category").agg(
    price0=("base_price", "median"),
    quantity0=("base_quantity_weekly", "median"),
    cost0=("unit_cost", "median"),
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors = {"fmcg": "#2E86AB", "durevoli": "#E63946", "premium": "#6A4C93"}
labels = {"fmcg": "FMCG", "durevoli": "Durevoli", "premium": "Premium"}
shock = 0.30

for ax, cat in zip(axes, ["fmcg", "durevoli", "premium"]):
    eps = best_elasticity[cat]
    price0 = typical.loc[cat, "price0"]
    quantity0 = typical.loc[cat, "quantity0"]
    cost0 = typical.loc[cat, "cost0"]
    cost1 = cost0 * (1 + shock)

    pt_grid = np.linspace(0, 1, 201)
    price_grid = price0 + pt_grid * (cost1 - cost0)
    price_change_pct = (price_grid - price0) / price0
    demand_mult = demand_multiplier_with_kink(price_change_pct, eps, cat)
    quantity_grid = quantity0 * demand_mult
    profit_grid = (price_grid - cost1) * quantity_grid

    best_idx = np.argmax(profit_grid)

    ax.plot(pt_grid * 100, profit_grid, color=colors[cat], linewidth=2.5)
    ax.axvline(pt_grid[best_idx] * 100, color=colors[cat], linestyle="--", alpha=0.6)
    ax.scatter([pt_grid[best_idx] * 100], [profit_grid[best_idx]], color=colors[cat], s=80, zorder=5)
    ax.set_title(f"{labels[cat]}  (shock costo +{int(shock*100)}%)")
    ax.set_xlabel("Pass-through (%)")
    ax.set_ylabel("Margine totale atteso (EUR/settimana)")

    kink_pt = KINK_CONFIG[cat]["threshold"] / shock * 100
    if kink_pt <= 100:
        ax.axvline(kink_pt, color="gray", linestyle=":", alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(kink_pt + 2, ylim[0] + (ylim[1] - ylim[0]) * 0.05,
                 "soglia di rottura", rotation=90, fontsize=8, color="gray")

    consigliato = pt_grid[best_idx] * 100
    ax.text(0.05, 0.95, f"Consigliato: {consigliato:.0f}%", transform=ax.transAxes,
            fontsize=10, va="top", fontweight="bold", color=colors[cat])

plt.tight_layout()
plt.savefig(DATA_DIR / "margin_curves_by_category.png", dpi=110)
print("saved")
