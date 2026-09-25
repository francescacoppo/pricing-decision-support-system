"""
Step 4 — Ottimizzazione: dato uno shock di costo, quale prezzo (e quindi
quale pass-through rate) massimizza il profitto?

FORMULA (derivazione completa nel messaggio di spiegazione):
Domanda a elasticita' costante:    Q(P) = Q0 * (P/P0)^eps
Profitto:                          pi(P) = (P - C) * Q(P)
Condizione del primo ordine  ->    P* = eps / (eps + 1) * C     (regola di Lerner)

Validita': serve |eps| > 1 (domanda elastica) perche' esista un ottimo
interno. Se |eps| < 1 (domanda anelastica, es. premium/Veblen), il
profitto cresce monotonamente col prezzo nel modello puro: niente ottimo
interno, e lo si dichiara esplicitamente invece di restituire un numero
senza senso.

In piu': cross-check NUMERICO usando la curva di partial dependence del
random forest (che cattura anche l'effetto soglia del FMCG, cosa che la
formula chiusa - basata su elasticita' costante - non puo' catturare).
Confrontando le due si vede se e quanto la non-linearita' reale sposta
la raccomandazione rispetto alla formula "pulita".
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# 0. SOGLIE DI "DOMANDA SPEZZATA" (kinked demand curve, Sweezy 1939)
# ---------------------------------------------------------------------------
# Un'elasticita' costante non ha mai un vero freno: oltre un certo punto la
# convenienza a scaricare l'aumento sul prezzo continua a crescere in eterno,
# il che porta il modello a raccomandare SEMPRE il pass-through massimo
# consentito, qualunque esso sia -- una raccomandazione poco realistica e,
# soprattutto, che rende il confronto tra categorie poco interessante
# (tutte finiscono allo stesso "muro").
#
# Nella realta' la sensibilita' al prezzo non e' lineare: oltre una certa
# soglia scatta un comportamento piu' brusco -- esattamente il meccanismo
# che avevamo gia' costruito per il FMCG allo Step 1 (fuga verso il
# private label oltre +8%). Qui lo estendiamo, con coerenza narrativa, agli
# altri due archetipi:
KINK_CONFIG = {
    # invariato rispetto allo Step 1: fuga verso il private label
    "fmcg": {"threshold": 0.08, "extra_elasticity": -2.5},
    # oltre +15%, una quota crescente di acquirenti "titubanti" rimanda
    # definitivamente l'acquisto (non solo lo posticipa)
    "durevoli": {"threshold": 0.15, "extra_elasticity": -1.8},
    # oltre +25%, la fascia aspirazionale/medio-alta (non i clienti piu'
    # facoltosi) inizia a sentirsi esclusa dal brand e abbandona
    "premium": {"threshold": 0.25, "extra_elasticity": -1.2},
}


def demand_multiplier_with_kink(price_change_pct, eps, category):
    """Elasticita' costante fino alla soglia, poi elasticita' aggiuntiva
    (piu' negativa) oltre la soglia -- stessa logica dello Step 1."""
    cfg = KINK_CONFIG[category]
    base = (1 + price_change_pct) ** eps
    beyond = price_change_pct > cfg["threshold"]
    extra = np.where(beyond, price_change_pct - cfg["threshold"], 0.0)
    kink_factor = np.where(beyond, (1 + extra) ** cfg["extra_elasticity"], 1.0)
    return base * kink_factor

# ---------------------------------------------------------------------------
# 1. ELASTICITA' CALIBRATA: il metodo migliore per categoria, deciso allo Step 3
# ---------------------------------------------------------------------------
comparison = pd.read_csv(DATA_DIR / "elasticity_comparison.csv", index_col=0)

best_elasticity, best_method = {}, {}
for cat in comparison.index:
    row = comparison.loc[cat]
    if row["errore_assoluto_loglog"] <= row["errore_assoluto_rf"]:
        best_elasticity[cat] = row["regressione_log_log"]
        best_method[cat] = "log-log"
    else:
        best_elasticity[cat] = row["random_forest (partial dependence)"]
        best_method[cat] = "random_forest"

print("Elasticita' calibrata scelta per categoria (metodo migliore, da Step 3):")
for cat, eps in best_elasticity.items():
    print(f"  {cat:10s}: eps = {eps:+.3f}   (metodo: {best_method[cat]})")

# ---------------------------------------------------------------------------
# 2. FORMULA CHIUSA (regola di Lerner / elasticita' costante)
# ---------------------------------------------------------------------------
def optimal_price_closed_form(unit_cost: float, eps: float):
    """P* = eps/(eps+1) * C. Ritorna (prezzo_ottimo, valido)."""
    if abs(eps) <= 1:
        return None, False
    p_star = unit_cost * eps / (eps + 1)
    return p_star, True


# ---------------------------------------------------------------------------
# 3. PRODOTTO "TIPICO" PER CATEGORIA (stessa logica dello Step 3: mediana)
# ---------------------------------------------------------------------------
product_master = pd.read_csv(DATA_DIR / "product_master.csv")
typical = product_master.groupby("category").agg(
    price0=("base_price", "median"),
    quantity0=("base_quantity_weekly", "median"),
    cost0=("unit_cost", "median"),
)
print("\nProdotto tipico per categoria (mediana):")
print(typical.round(2))

# ---------------------------------------------------------------------------
# 4bis. OTTIMIZZAZIONE VINCOLATA (0% - 100% di pass-through)
# ---------------------------------------------------------------------------
# La formula chiusa (sopra) trova l'ottimo ASSOLUTO di puro monopolio, che puo'
# essere lontanissimo dal prezzo attuale (perche' nella simulazione il markup
# di partenza e' stato scelto indipendentemente dall'elasticita'). Un salto
# di prezzo cosi' grande in un colpo solo non e' una raccomandazione
# realistica per un'azienda vera. Cerchiamo quindi il punto migliore
# DENTRO l'intervallo di scelta realistico che avevamo definito fin
# dall'inizio: pass-through tra 0% (assorbo tutto il costo extra) e 100%
# (trasferisco un aumento di prezzo in euro pari all'aumento di costo).
def bounded_optimal_passthrough(cost0, cost1, price0, quantity0, eps, category,
                                  pt_max=1.0, n_grid=201, use_kink=True):
    pt_grid = np.linspace(0.0, pt_max, n_grid)
    price_grid = price0 + pt_grid * (cost1 - cost0)
    price_change_pct = (price_grid - price0) / price0

    if use_kink:
        demand_mult = demand_multiplier_with_kink(price_change_pct, eps, category)
    else:
        demand_mult = (price_grid / price0) ** eps

    quantity_grid = quantity0 * demand_mult
    profit_grid = (price_grid - cost1) * quantity_grid
    best = np.argmax(profit_grid)
    return {
        "passthrough": pt_grid[best],
        "price": price_grid[best],
        "quantity": quantity_grid[best],
        "profit": profit_grid[best],
        "at_upper_bound": best == n_grid - 1,
    }


# ---------------------------------------------------------------------------
# 4. CROSS-CHECK NUMERICO con la curva di partial dependence (cattura la soglia FMCG)
# ---------------------------------------------------------------------------
def numeric_optimal_passthrough(category: str, cost_shock_pct: float):
    """Come sopra ma sulla curva imparata dal random forest (cattura la
    soglia FMCG). Cerchiamo anche qui SOLO nell'intervallo realistico:
    price_change_pct tra 0 e cost_shock_pct (= pass-through tra 0% e 100%)."""
    curve = pd.read_csv(DATA_DIR / f"rf_partial_dependence_{category}.csv")
    price0 = typical.loc[category, "price0"]
    quantity0 = typical.loc[category, "quantity0"]
    cost0 = typical.loc[category, "cost0"]
    cost1 = cost0 * (1 + cost_shock_pct)

    realistic = curve[(curve["price_change_pct"] >= 0) & (curve["price_change_pct"] <= cost_shock_pct)].copy()
    price1 = price0 * (1 + realistic["price_change_pct"])
    quantity1 = quantity0 * realistic["predicted_ratio"]
    profit1 = (price1 - cost1) * quantity1

    best_idx = profit1.idxmax()
    return {
        "price_change_pct_numeric": realistic.loc[best_idx, "price_change_pct"],
        "passthrough_numeric": realistic.loc[best_idx, "price_change_pct"] / cost_shock_pct,
    }


# ---------------------------------------------------------------------------
# 5. TABELLA DI SCENARIO: per ogni categoria x ogni shock di costo
# ---------------------------------------------------------------------------
COST_SHOCKS = [0.10, 0.20, 0.30]
rows = []

for cat in best_elasticity:
    eps = best_elasticity[cat]
    price0 = typical.loc[cat, "price0"]
    quantity0 = typical.loc[cat, "quantity0"]
    cost0 = typical.loc[cat, "cost0"]

    for shock in COST_SHOCKS:
        cost1 = cost0 * (1 + shock)

        # --- SENZA soglia (il comportamento "sempre 100%" che avevi notato) ---
        no_kink = bounded_optimal_passthrough(cost0, cost1, price0, quantity0, eps, cat,
                                                pt_max=1.0, use_kink=False)

        # --- CON soglia (la correzione: domanda spezzata oltre un certo punto) ---
        result = bounded_optimal_passthrough(cost0, cost1, price0, quantity0, eps, cat,
                                               pt_max=1.0, use_kink=True)
        p_star = result["price"]
        passthrough = result["passthrough"]
        quantity1 = result["quantity"]
        revenue1 = p_star * quantity1
        margin1 = result["profit"]
        margin_pct1 = margin1 / revenue1

        numeric = numeric_optimal_passthrough(cat, shock)

        # baseline: nessun aumento di prezzo (pass-through 0%), per confronto
        revenue0 = price0 * quantity0
        margin0 = revenue0 - cost1 * quantity0  # costo NUOVO, prezzo VECCHIO
        margin_pct0 = margin0 / revenue0

        rows.append({
            "categoria": cat,
            "elasticita": round(eps, 3),
            "shock_costo_pct": shock,
            "passthrough_SENZA_soglia": round(no_kink["passthrough"], 3),
            "passthrough_CON_soglia": round(passthrough, 3),
            "al_tetto_massimo_100pct": result["at_upper_bound"],
            "prezzo_consigliato": round(p_star, 2),
            "variazione_prezzo_pct": round((p_star - price0) / price0, 4),
            "passthrough_numerico_rf": round(numeric["passthrough_numeric"], 3),
            "margine_no_aumento": round(margin0, 1),
            "margine_pct_no_aumento": round(margin_pct0, 4),
            "margine_consigliato": round(margin1, 1),
            "margine_pct_consigliato": round(margin_pct1, 4),
        })

scenarios = pd.DataFrame(rows)
scenarios.to_csv(DATA_DIR / "pricing_scenarios.csv", index=False)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print("\n=== TABELLA SCENARI (shock di costo x categoria) ===")
print(scenarios.to_string(index=False))

print(f"\nFile salvato in: {DATA_DIR / 'pricing_scenarios.csv'}")
