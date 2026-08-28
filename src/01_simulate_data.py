"""
Simulatore di dati di pricing calibrato su elasticità reali di letteratura.

Fonti per la calibrazione:
- Tellis (1988), Journal of Marketing Research: elasticità media di brand ≈ -1.76
- Bijmolt, van Heerde & Pieters (2005), JMR: elasticità media aggiornata ≈ -2.62;
  i beni durevoli mostrano sensibilità al prezzo più alta della media, i beni di
  largo consumo (CPG) più bassa della media.
- Studio su vini high-priced (Consumer response to price changes in higher-priced
  brands, ScienceDirect): elasticità media -1.8 per brand ad alto prezzo, e nota
  esplicita che i beni di lusso possono avere elasticità quasi nulla o positiva
  (effetto Veblen) come eccezione al pattern standard.

Logica di disegno IMPORTANTE:
Le variazioni di prezzo storiche simulate restano in un range realistico
(circa -15% / +15%, tipico di promozioni e piccoli adeguamenti listino).
Questo è VOLUTO: quando nella dashboard l'utente simulerà uno shock di
costo del +20% o +30%, il modello ML si troverà a estrapolare fuori dal
range osservato nei dati storici. Questo ci permette di costruire (più
avanti) un avviso di affidabilità della stima quando si esce dal range
storico -- un dettaglio di rigore metodologico, non un bug.
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Cartella "data" creata accanto allo script, indipendentemente da dove
# si trova lo script sul disco (Windows, Mac o Linux). Se non esiste
# ancora, viene creata automaticamente.
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. STRUTTURA DELLE CATEGORIE (i tre archetipi)
# ---------------------------------------------------------------------------
CATEGORY_CONFIG = {
    "fmcg": {
        "label": "Largo Consumo / FMCG",
        "n_products": 40,
        "elasticity_mean": -1.8,
        "elasticity_std": 0.35,      # eterogeneità tra prodotti nella stessa categoria
        "price_range": (2, 15),      # prezzo unitario base, EUR
        "quantity_range": (2000, 20000),  # unità vendute/mese base
        "cost_ratio_range": (0.55, 0.75),  # costo/prezzo tipico FMCG (margini più bassi)
        "kink_threshold": 0.08,      # oltre +8% di aumento, scatta la fuga verso private label
        "kink_extra_elasticity": -2.5,  # elasticità aggiuntiva oltre la soglia
        "subcategories": ["alimentare_confezionato", "cura_persona", "bevande"],
    },
    "durevoli": {
        "label": "Beni Durevoli / Voluttuari",
        "n_products": 30,
        "elasticity_mean": -2.8,
        "elasticity_std": 0.5,
        "price_range": (150, 1200),
        "quantity_range": (200, 3000),
        "cost_ratio_range": (0.60, 0.80),
        "kink_threshold": None,      # nessun effetto soglia, la reazione è più graduale ma forte
        "kink_extra_elasticity": None,
        "subcategories": ["elettronica_consumo", "elettrodomestici", "arredamento"],
    },
    "premium": {
        "label": "Premium / Posizionale",
        "n_products": 20,
        "elasticity_mean": -0.7,
        "elasticity_std": 0.4,
        "price_range": (300, 3000),
        "quantity_range": (50, 800),
        "cost_ratio_range": (0.25, 0.45),  # margini alti, tipico del premium
        "kink_threshold": None,
        "kink_extra_elasticity": None,
        "subcategories": ["moda_lusso", "elettronica_alta_gamma", "orologeria_gioielli"],
        "veblen_share": 0.15,  # quota di prodotti con elasticità quasi nulla o positiva
    },
}

N_WEEKS = 104  # 2 anni di storico settimanale


def make_product_master():
    rows = []
    pid = 0
    for cat, cfg in CATEGORY_CONFIG.items():
        for i in range(cfg["n_products"]):
            pid += 1
            base_price = rng.uniform(*cfg["price_range"])
            base_quantity = rng.uniform(*cfg["quantity_range"])
            cost_ratio = rng.uniform(*cfg["cost_ratio_range"])
            unit_cost = base_price * cost_ratio

            elasticity_true = rng.normal(cfg["elasticity_mean"], cfg["elasticity_std"])
            # per il premium, una quota di prodotti "posizionali puri" (effetto Veblen)
            veblen_flag = False
            if cat == "premium" and rng.random() < cfg.get("veblen_share", 0):
                elasticity_true = rng.uniform(-0.1, 0.3)  # quasi nulla o leggermente positiva
                veblen_flag = True
            elasticity_true = min(elasticity_true, -0.05) if not veblen_flag else elasticity_true

            # scomposizione del costo (per il modulo "shock di costo" della dashboard)
            energy_share = rng.uniform(0.10, 0.35)
            materials_share = rng.uniform(0.30, 0.55)
            transport_share = rng.uniform(0.05, 0.20)
            other_share = max(0.0, 1 - energy_share - materials_share - transport_share)
            total = energy_share + materials_share + transport_share + other_share
            energy_share, materials_share, transport_share, other_share = [
                x / total for x in (energy_share, materials_share, transport_share, other_share)
            ]

            rows.append({
                "product_id": f"P{pid:04d}",
                "category": cat,
                "category_label": cfg["label"],
                "subcategory": rng.choice(cfg["subcategories"]),
                "base_price": round(base_price, 2),
                "base_quantity_weekly": round(base_quantity / 4.33, 1),  # da mensile a settimanale
                "unit_cost": round(unit_cost, 2),
                "elasticity_true": round(elasticity_true, 3),
                "veblen_flag": veblen_flag,
                "energy_share": round(energy_share, 3),
                "materials_share": round(materials_share, 3),
                "transport_share": round(transport_share, 3),
                "other_cost_share": round(other_share, 3),
            })
    return pd.DataFrame(rows)


def simulate_transactions(product_master: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, prod in product_master.iterrows():
        cfg = CATEGORY_CONFIG[prod["category"]]
        p0 = prod["base_price"]
        q0 = prod["base_quantity_weekly"]
        eps = prod["elasticity_true"]

        for week in range(N_WEEKS):
            # --- variazione di prezzo storica realistica (max +-15%) ---
            # per lo piu' piccoli aggiustamenti, occasionali promozioni piu' marcate
            promo = rng.random() < 0.15
            if promo:
                price_change = rng.uniform(-0.15, -0.05)  # sconto promozionale
            else:
                price_change = rng.normal(0, 0.035)
                price_change = np.clip(price_change, -0.10, 0.12)

            price = p0 * (1 + price_change)

            # --- stagionalita' leggera (ciclo annuale, 52 settimane) ---
            seasonality = 1 + 0.08 * np.sin(2 * np.pi * week / 52)

            # --- indice prezzo competitor (rumore correlato al mercato) ---
            competitor_index = 1 + rng.normal(0, 0.05)

            # --- domanda con elasticita' costante ---
            demand_multiplier = (price / p0) ** eps

            # --- effetto soglia (solo FMCG): oltre la soglia, fuga verso private label ---
            if cfg.get("kink_threshold") is not None and price_change > cfg["kink_threshold"]:
                extra = price_change - cfg["kink_threshold"]
                demand_multiplier *= (1 + extra) ** cfg["kink_extra_elasticity"]

            # --- effetto promozione (uplift aggiuntivo oltre il puro effetto prezzo) ---
            promo_uplift = 1.25 if promo else 1.0

            # --- rumore moltiplicativo (variabilita' non spiegata) ---
            noise = rng.lognormal(mean=0, sigma=0.12)

            quantity = q0 * demand_multiplier * seasonality * promo_uplift * noise
            quantity = max(quantity, 0)

            records.append({
                "product_id": prod["product_id"],
                "category": prod["category"],
                "subcategory": prod["subcategory"],
                "week": week,
                "price": round(price, 2),
                "base_price": p0,
                "price_change_pct": round(price_change, 4),
                "promotion": int(promo),
                "competitor_price_index": round(competitor_index, 4),
                "seasonality_index": round(seasonality, 4),
                "quantity_sold": round(quantity, 1),
                "unit_cost": prod["unit_cost"],
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    product_master = make_product_master()
    transactions = simulate_transactions(product_master)

    product_master.to_csv(OUTPUT_DIR / "product_master.csv", index=False)
    transactions.to_csv(OUTPUT_DIR / "transactions.csv", index=False)

    print(f"\nFile salvati in: {OUTPUT_DIR}")

    print("Product master:", product_master.shape)
    print(product_master.groupby("category")["elasticity_true"].describe()[["mean", "std", "min", "max"]])
    print("\nTransactions:", transactions.shape)
    print("\nRange di price_change_pct osservato nello storico (per categoria):")
    print(transactions.groupby("category")["price_change_pct"].agg(["min", "max"]))
