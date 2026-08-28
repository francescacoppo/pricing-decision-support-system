"""
Stima dell'elasticità dai dati.

Confrontiamo due approcci:
1. Regressione log-log classica (il metodo econometrico standard per stimare
   l'elasticità-prezzo: regredisco log(quantità) su log(prezzo), lo slope
   della retta è l'elasticità).
2. Random Forest, da cui estraiamo l'elasticità "implicita" osservando come
   variano le previsioni del modello al variare del prezzo (partial dependence),
   tenendo fisse le altre variabili.

Entrambi vengono validati contro l'elasticità VERA (nota solo perché i dati
sono simulati) — cosa che con dati reali non potremmo mai fare, ed è uno dei
motivi per cui costruire prima un dataset simulato e calibrato è una scelta
metodologicamente seria, non solo un ripiego.

Train/test split per PRODOTTO (non per riga): l'80% dei prodotti di ogni
categoria viene usato per allenare i modelli, il 20% rimane fuori. Questo
verifica che i modelli abbiano imparato un pattern generale di categoria,
non che abbiano solo "memorizzato" il comportamento dei singoli prodotti
visti in allenamento.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# 1. CARICAMENTO E MERGE
# ---------------------------------------------------------------------------
product_master = pd.read_csv(DATA_DIR / "product_master.csv")
transactions = pd.read_csv(DATA_DIR / "transactions.csv")

df = transactions.merge(
    product_master[["product_id", "base_quantity_weekly", "elasticity_true", "veblen_flag"]],
    on="product_id", how="left"
)

# ---------------------------------------------------------------------------
# 2. TRAIN/TEST SPLIT PER PRODOTTO (stratificato per categoria)
# ---------------------------------------------------------------------------
train_products, test_products = [], []
for cat in product_master["category"].unique():
    ids = product_master.loc[product_master["category"] == cat, "product_id"].to_numpy()
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * 0.2))
    test_products.extend(ids[:n_test])
    train_products.extend(ids[n_test:])

train_df = df[df["product_id"].isin(train_products)].copy()
test_df = df[df["product_id"].isin(test_products)].copy()

print(f"Prodotti train: {len(train_products)} | Prodotti test: {len(test_products)}")
print(f"Righe train: {len(train_df)} | Righe test: {len(test_df)}")

# ---------------------------------------------------------------------------
# 3. METODO 1 — REGRESSIONE LOG-LOG CLASSICA (per categoria)
# ---------------------------------------------------------------------------
def fit_loglog_elasticity(data: pd.DataFrame) -> float:
    """Elasticità = coefficiente angolare di log(quantita) su log(prezzo/prezzo_base).
    Uso solo osservazioni senza promozione, per non confondere l'effetto
    prezzo con l'effetto promozionale."""
    sub = data[data["promotion"] == 0].copy()
    x = np.log(sub["price"] / sub["base_price"]).to_numpy().reshape(-1, 1)
    y = np.log(sub["quantity_sold"].clip(lower=1)).to_numpy()
    model = LinearRegression().fit(x, y)
    return model.coef_[0]

loglog_results = {}
for cat in product_master["category"].unique():
    eps_hat = fit_loglog_elasticity(train_df[train_df["category"] == cat])
    loglog_results[cat] = eps_hat

# ---------------------------------------------------------------------------
# 4. METODO 2 — RANDOM FOREST + PARTIAL DEPENDENCE
# ---------------------------------------------------------------------------
FEATURES = [
    "price_change_pct", "promotion", "competitor_price_index",
    "seasonality_index", "base_price", "base_quantity_weekly", "category",
]
TARGET = "quantity_ratio"

# Target normalizzato: quantita' venduta / quantita' di base del prodotto.
# Senza questa normalizzazione, il modello viene dominato dalla scala assoluta
# dei volumi (prodotti che vendono migliaia di pezzi vs poche decine), e la
# variabile prezzo -- quella che ci interessa -- diventa marginale nella
# feature importance nonostante sia il vero driver della domanda.
df["quantity_ratio"] = df["quantity_sold"] / df["base_quantity_weekly"]

df_encoded = pd.get_dummies(df[FEATURES + [TARGET, "product_id"]], columns=["category"])
train_enc = df_encoded[df_encoded["product_id"].isin(train_products)].drop(columns="product_id")
test_enc = df_encoded[df_encoded["product_id"].isin(test_products)].drop(columns="product_id")

X_train, y_train = train_enc.drop(columns=TARGET), train_enc[TARGET]
X_test, y_test = test_enc.drop(columns=TARGET), test_enc[TARGET]

rf = RandomForestRegressor(
    n_estimators=400, max_depth=12, min_samples_leaf=5,
    random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = float(np.mean(np.abs((y_test - y_pred) / y_test.clip(lower=0.01))) * 100)

print(f"\nPerformance Random Forest su prodotti MAI VISTI in training (target: rapporto domanda/base):")
print(f"  R2   = {r2:.3f}")
print(f"  MAE  = {mae:.3f} (in unita' di rapporto, es. 0.05 = 5% di scostamento dalla base)")
print(f"  MAPE = {mape:.1f}%")

# --- Partial dependence manuale: elasticita' implicita del random forest ---
def rf_partial_dependence_elasticity(category: str, price_grid: np.ndarray) -> pd.DataFrame:
    """Tiene fisse tutte le altre variabili al valore mediano/tipico della
    categoria, fa variare solo price_change_pct, e osserva come si muove
    la previsione del modello."""
    cat_products = product_master[product_master["category"] == category]
    base_row = {
        "promotion": 0,
        "competitor_price_index": 1.0,
        "seasonality_index": 1.0,
        "base_price": cat_products["base_price"].median(),
        "base_quantity_weekly": cat_products["base_quantity_weekly"].median(),
    }
    rows = []
    for pc in price_grid:
        row = base_row.copy()
        row["price_change_pct"] = pc
        for c in product_master["category"].unique():
            row[f"category_{c}"] = 1.0 if c == category else 0.0
        rows.append(row)
    grid_df = pd.DataFrame(rows)[X_train.columns]  # stesso ordine colonne del training
    preds = rf.predict(grid_df)  # previsione del rapporto domanda/base
    return pd.DataFrame({"price_change_pct": price_grid, "predicted_ratio": preds})


# range esteso oltre lo storico (+-15%) fino a +30%, per mostrare cosa succede
# quando si estrapola oltre i dati osservati
price_grid = np.linspace(-0.15, 0.30, 46)
rf_curves = {}
rf_elasticity_in_range = {}

for cat in product_master["category"].unique():
    curve = rf_partial_dependence_elasticity(cat, price_grid)
    rf_curves[cat] = curve

    # elasticita' stimata SOLO nel range storico osservato (-15% / +12%), dove
    # il modello sta interpolando e non estrapolando
    in_range = curve[(curve["price_change_pct"] >= -0.15) & (curve["price_change_pct"] <= 0.12)]
    x = np.log(1 + in_range["price_change_pct"]).to_numpy().reshape(-1, 1)
    y = np.log(in_range["predicted_ratio"].clip(lower=0.01)).to_numpy()
    slope = LinearRegression().fit(x, y).coef_[0]
    rf_elasticity_in_range[cat] = slope

# ---------------------------------------------------------------------------
# 5. TABELLA DI CONFRONTO FINALE
# ---------------------------------------------------------------------------
true_elasticity = product_master.groupby("category")["elasticity_true"].mean()

comparison = pd.DataFrame({
    "elasticita_vera (simulata)": true_elasticity,
    "regressione_log_log": pd.Series(loglog_results),
    "random_forest (partial dependence)": pd.Series(rf_elasticity_in_range),
})
comparison["errore_assoluto_loglog"] = (comparison["elasticita_vera (simulata)"] - comparison["regressione_log_log"]).abs()
comparison["errore_assoluto_rf"] = (comparison["elasticita_vera (simulata)"] - comparison["random_forest (partial dependence)"]).abs()

print("\n=== CONFRONTO: elasticita' vera vs stimata (regressione classica vs random forest) ===")
print(comparison.round(3))

# ---------------------------------------------------------------------------
# 6. SALVATAGGIO RISULTATI (serviranno alla dashboard)
# ---------------------------------------------------------------------------
comparison.round(3).to_csv(DATA_DIR / "elasticity_comparison.csv")
for cat, curve in rf_curves.items():
    curve.to_csv(DATA_DIR / f"rf_partial_dependence_{cat}.csv", index=False)

feature_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
feature_importance.to_csv(DATA_DIR / "rf_feature_importance.csv")
print("\n=== FEATURE IMPORTANCE (random forest) ===")
print(feature_importance.round(3))

print(f"\nFile di risultati salvati in: {DATA_DIR}")
