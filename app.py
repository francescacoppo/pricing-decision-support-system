"""
Pricing Decision Support System
Dashboard che non si limita a calcolare un prezzo, ma spiega la decisione:
Configurazione -> Analisi del modello -> Simulazione -> Decisione.

Streamlit + Plotly.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

# ---------------------------------------------------------------------------
# CONFIGURAZIONE
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Pricing DSS", page_icon=None, layout="wide",
                    initial_sidebar_state="expanded")

DATA_DIR = Path(__file__).parent / "data"

COLOR_PRIMARY = "#14213D"
COLOR_SECONDARY = "#5C677D"
COLOR_ACCENT = "#C1440E"
COLOR_BG_CARD = "#F7F8FA"
COLOR_BORDER = "#E2E5EA"
COLOR_POSITIVE = "#1B7A43"
COLOR_NEGATIVE = "#B3261E"
COLOR_FMCG = "#2E5C8A"
COLOR_DUREVOLI = "#A6402D"
COLOR_PREMIUM = "#5B4C79"

KINK_CONFIG = {
    "fmcg": {"threshold": 0.08, "extra_elasticity": -2.5},
    "durevoli": {"threshold": 0.15, "extra_elasticity": -1.8},
    "premium": {"threshold": 0.25, "extra_elasticity": -1.2},
}
CATEGORY_LABELS = {
    "fmcg": "Largo Consumo",
    "durevoli": "Beni Durevoli",
    "premium": "Premium",
}
CATEGORY_DESCRIPTIONS = {
    "fmcg": "Prodotti d'acquisto frequente e prezzo contenuto (alimentari, cura persona, bevande). "
            "Il cliente nota subito un aumento e, superata una soglia, passa ai brand commerciali.",
    "durevoli": "Prodotti d'acquisto occasionale e prezzo medio-alto (elettronica, elettrodomestici, "
                "arredamento). Di fronte a un aumento, il cliente tende a rimandare l'acquisto.",
    "premium": "Prodotti di fascia alta dove il prezzo fa parte del valore percepito (moda, alta gamma). "
               "Il cliente e poco sensibile al prezzo, ma rischia di sentirsi escluso oltre una certa soglia.",
}
CATEGORY_COLORS = {"fmcg": COLOR_FMCG, "durevoli": COLOR_DUREVOLI, "premium": COLOR_PREMIUM}
CATEGORY_RISK_NOTE = {
    "fmcg": "Oltre l'8% di aumento prezzo la domanda accelera verso il private label.",
    "durevoli": "Oltre il 15% di aumento prezzo aumenta il rinvio definitivo dell'acquisto.",
    "premium": "Oltre il 25% di aumento prezzo rischia di alienare la fascia aspirazionale.",
}
FEATURE_LABELS = {
    "price_change_pct": "Variazione di prezzo",
    "promotion": "Promozione attiva",
    "competitor_price_index": "Prezzo dei competitor",
    "seasonality_index": "Stagionalita",
    "base_price": "Prezzo di listino",
    "base_quantity_weekly": "Volume di vendita abituale",
    "category_durevoli": "Categoria: Durevoli",
    "category_fmcg": "Categoria: Largo Consumo",
    "category_premium": "Categoria: Premium",
}

# ---------------------------------------------------------------------------
# DATI
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    comp = pd.read_csv(DATA_DIR / "elasticity_comparison.csv", index_col=0)
    pm = pd.read_csv(DATA_DIR / "product_master.csv")
    fi = pd.read_csv(DATA_DIR / "rf_feature_importance.csv", index_col=0)
    pdep = {cat: pd.read_csv(DATA_DIR / f"rf_partial_dependence_{cat}.csv")
            for cat in ["fmcg", "durevoli", "premium"]}
    return comp, pm, fi, pdep

comparison, product_master, feature_importance, partial_dep = load_data()

best_elasticity, best_method = {}, {}
for cat in comparison.index:
    row = comparison.loc[cat]
    if row["errore_assoluto_loglog"] <= row["errore_assoluto_rf"]:
        best_elasticity[cat] = row["regressione_log_log"]
        best_method[cat] = "Regressione log-log"
        best_error = row["errore_assoluto_loglog"]
    else:
        best_elasticity[cat] = row["random_forest (partial dependence)"]
        best_method[cat] = "Random Forest"
        best_error = row["errore_assoluto_rf"]
comparison["errore_scelto"] = comparison.apply(
    lambda r: r["errore_assoluto_loglog"] if r["errore_assoluto_loglog"] <= r["errore_assoluto_rf"]
    else r["errore_assoluto_rf"], axis=1
)

typical = product_master.groupby("category").agg(
    price0=("base_price", "median"), quantity0=("base_quantity_weekly", "median"),
    cost0=("unit_cost", "median"),
)

N_OBS = len(pd.read_csv(DATA_DIR / "transactions.csv")) if (DATA_DIR / "transactions.csv").exists() else 9360
N_PRODUCTS = len(product_master)

# ---------------------------------------------------------------------------
# CALCOLI
# ---------------------------------------------------------------------------
def demand_multiplier_with_kink(price_change_pct, eps, category):
    cfg = KINK_CONFIG[category]
    base = (1 + price_change_pct) ** eps
    beyond = price_change_pct > cfg["threshold"]
    extra = np.where(beyond, price_change_pct - cfg["threshold"], 0.0)
    kink_factor = np.where(beyond, (1 + extra) ** cfg["extra_elasticity"], 1.0)
    return base * kink_factor

def calculate_scenarios(category, cost0, price0, quantity0, cost_shock_pct, eps):
    cost1 = cost0 * (1 + cost_shock_pct)
    pt_grid = np.linspace(0, 1, 201)
    price_grid = price0 + pt_grid * (cost1 - cost0)
    price_change_pct = (price_grid - price0) / price0
    demand_mult = demand_multiplier_with_kink(price_change_pct, eps, category)
    quantity_grid = quantity0 * demand_mult
    revenue_grid = price_grid * quantity_grid
    margin_grid = (price_grid - cost1) * quantity_grid
    margin_pct_grid = margin_grid / np.maximum(revenue_grid, 1)
    best_idx = int(np.argmax(margin_grid))
    return {"pt_grid": pt_grid, "price_grid": price_grid, "quantity_grid": quantity_grid,
            "demand_mult": demand_mult, "revenue_grid": revenue_grid, "margin_grid": margin_grid,
            "margin_pct_grid": margin_pct_grid, "best_idx": best_idx,
            "best_pt": pt_grid[best_idx], "best_margin": margin_grid[best_idx], "cost1": cost1}

def recommended_pt_for_category(category, shock_pct):
    """Pass-through consigliato per una categoria, usando i suoi valori tipici
    (serve per il confronto tra le tre categorie a parita' di shock)."""
    row = typical.loc[category]
    eps = best_elasticity[category]
    sc = calculate_scenarios(category, row["cost0"], row["price0"], row["quantity0"], shock_pct, eps)
    return sc["best_pt"]

# ---------------------------------------------------------------------------
# STILE
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
    .block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
    .main-title {{ color: {COLOR_PRIMARY}; font-size: 2.0em; font-weight: 700;
        margin-bottom: 0.15em; letter-spacing: -0.5px; }}
    .subtitle {{ color: {COLOR_SECONDARY}; font-size: 1.0em; margin-bottom: 1.0em; }}
    .pipeline {{ display:flex; gap:6px; margin-bottom:1.6em; flex-wrap:wrap; }}
    .pipeline-step {{ background:{COLOR_BG_CARD}; border:1px solid {COLOR_BORDER};
        border-radius:20px; padding:5px 14px; font-size:0.78em; color:{COLOR_SECONDARY};
        font-weight:600; }}
    .pipeline-arrow {{ color:{COLOR_BORDER}; font-size:0.9em; align-self:center; }}
    .section-header {{ color: {COLOR_PRIMARY}; font-size: 1.05em; font-weight: 600;
        margin-top: 1.0em; margin-bottom: 0.8em; text-transform: uppercase;
        letter-spacing: 0.8px; border-left: 3px solid {COLOR_ACCENT}; padding-left: 10px; }}
    .sidebar-header {{ color: {COLOR_PRIMARY}; font-size: 0.95em; font-weight: 700;
        margin-top: 1.2em; margin-bottom: 0.5em; text-transform: uppercase; letter-spacing: 0.6px; }}
    .kpi-card {{ background-color: white; border: 1px solid {COLOR_BORDER};
        border-top: 3px solid {COLOR_ACCENT}; border-radius: 8px; padding: 16px 18px; height:100%; }}
    .kpi-label {{ color: {COLOR_SECONDARY}; font-size: 0.76em; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.7px; margin-bottom: 5px; }}
    .kpi-value {{ color: {COLOR_PRIMARY}; font-size: 1.7em; font-weight: 700; line-height:1.1; }}
    .kpi-delta {{ font-size: 0.82em; margin-top: 5px; font-weight: 600; }}
    .delta-positive {{ color: {COLOR_POSITIVE}; }}
    .delta-negative {{ color: {COLOR_NEGATIVE}; }}
    .delta-neutral {{ color: {COLOR_SECONDARY}; }}
    .advice-box {{ background-color: {COLOR_BG_CARD}; border-left: 3px solid {COLOR_ACCENT};
        padding: 20px 24px; border-radius: 6px; margin-top: 0.8em; margin-bottom: 1.2em; }}
    .advice-title {{ color: {COLOR_PRIMARY}; font-weight: 700; font-size: 1.1em; margin-bottom: 12px; }}
    .reason-item {{ margin: 6px 0; color: {COLOR_PRIMARY}; }}
    .assumptions-box {{ background:white; border:1px solid {COLOR_BORDER}; border-radius:8px;
        padding:16px 20px; font-size:0.88em; color:{COLOR_SECONDARY}; }}
    .flow-box {{ background:white; border:1px solid {COLOR_BORDER}; border-radius:8px;
        padding:18px 22px; text-align:center; }}
    .flow-label {{ font-size:0.78em; color:{COLOR_SECONDARY}; text-transform:uppercase;
        letter-spacing:0.6px; }}
    .flow-value {{ font-size:1.3em; font-weight:700; color:{COLOR_PRIMARY}; margin:4px 0 10px 0; }}
    .flow-arrow {{ color:{COLOR_BORDER}; font-size:1.4em; }}
    .index-card {{ background:white; border:1px solid {COLOR_BORDER}; border-radius:8px;
        padding:20px; text-align:center; }}
    .index-value {{ font-size:2.2em; font-weight:800; }}
    .index-label {{ font-size:0.8em; color:{COLOR_SECONDARY}; text-transform:uppercase;
        letter-spacing:0.6px; margin-top:6px; }}
    [data-testid="stMetricValue"] {{ font-size: 1.3em; }}
    div[data-baseweb="slider"] div[role="slider"] {{
        width: 22px !important; height: 22px !important;
        box-shadow: 0 0 0 4px rgba(193,68,14,0.15) !important;
    }}
    div[data-baseweb="slider"] > div > div {{ height: 6px !important; }}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Pricing Decision Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Non solo un prezzo consigliato — la spiegazione completa di come e '
            'perche\' il modello arriva a quel consiglio</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR — INPUT (sempre visibili, indipendentemente dal tab)
# ---------------------------------------------------------------------------
st.sidebar.markdown('<div class="sidebar-header">Configurazione</div>', unsafe_allow_html=True)
categoria = st.sidebar.selectbox("Categoria prodotto", options=["fmcg", "durevoli", "premium"],
                                   format_func=lambda x: CATEGORY_LABELS[x])

st.sidebar.markdown('<div class="sidebar-header">Dati aziendali attuali</div>', unsafe_allow_html=True)
prezzo_attuale = st.sidebar.number_input("Prezzo unitario attuale (EUR)", min_value=0.1,
    value=float(typical.loc[categoria, "price0"]), step=0.1, format="%.2f")
quantita_attuale = st.sidebar.number_input("Quantita venduta/settimana (unita)", min_value=1,
    value=int(typical.loc[categoria, "quantity0"]), step=10)
costo_attuale = st.sidebar.number_input("Costo unitario attuale (EUR)", min_value=0.01,
    value=float(typical.loc[categoria, "cost0"]), step=0.1, format="%.2f")

st.sidebar.markdown('<div class="sidebar-header">Calcolo shock di costo</div>', unsafe_allow_html=True)
st.sidebar.caption("Aumento percentuale per componente")
energia_shock = st.sidebar.slider("Energia (+%)", 0, 100, 35, step=5)
materiali_shock = st.sidebar.slider("Materie prime (+%)", 0, 100, 40, step=5)
trasporto_shock = st.sidebar.slider("Trasporto (+%)", 0, 100, 25, step=5)
st.sidebar.caption("Default calibrati su shock reali di crisi energetica (Fed NY, 2022).")

weights = product_master[product_master["category"] == categoria][
    ["energy_share", "materials_share", "transport_share", "other_cost_share"]].mean()
shock_costo_ponderato = (energia_shock * weights["energy_share"] +
    materiali_shock * weights["materials_share"] + trasporto_shock * weights["transport_share"]) / 100
st.sidebar.metric("Shock di costo complessivo", f"{shock_costo_ponderato*100:.1f}%")
st.sidebar.caption(
    f"Energia + materiali + trasporto pesano il {(1-weights['other_cost_share'])*100:.0f}% del costo "
    f"totale in questa categoria; il restante {weights['other_cost_share']*100:.0f}% (altri costi, es. "
    f"manodopera) e considerato invariato — per questo lo shock complessivo e sempre piu basso dei "
    f"singoli aumenti inseriti."
)

# ---------------------------------------------------------------------------
# CALCOLO SCENARIO PRINCIPALE
# ---------------------------------------------------------------------------
eps = best_elasticity[categoria]
scenarios = calculate_scenarios(categoria, costo_attuale, prezzo_attuale, quantita_attuale,
                                  shock_costo_ponderato, eps)
color_cat = CATEGORY_COLORS[categoria]
recommended_pt = int(round(scenarios["best_pt"] * 100))

def risk_level(perdita_volumi_pct):
    if perdita_volumi_pct < 5:
        return "BASSO", COLOR_POSITIVE
    elif perdita_volumi_pct < 15:
        return "MEDIO", COLOR_ACCENT
    return "ALTO", COLOR_NEGATIVE

# --- valori "alla raccomandazione" (indipendenti dallo slider di override nel Tab 4) ---
idx_rec = scenarios["best_idx"]
margine_rec = scenarios["margin_grid"][idx_rec]
prezzo_rec = scenarios["price_grid"][idx_rec]
quantita_rec = scenarios["quantity_grid"][idx_rec]
variazione_prezzo_rec = (prezzo_rec - prezzo_attuale) / prezzo_attuale * 100
perdita_volumi_rec = max(0, (1 - quantita_rec / quantita_attuale) * 100)
rischio_rec_label, rischio_rec_color = risk_level(perdita_volumi_rec)

# ---------------------------------------------------------------------------
# EXECUTIVE SUMMARY — sempre visibile, prima dei tab: cosa serve nei primi 30 secondi
# ---------------------------------------------------------------------------
st.markdown(f"""
<div class="advice-box" style="border-left-width:5px; margin-top:0.3em;">
    <div class="advice-title" style="font-size:1.25em;">
        Raccomandazione: trasferire il {recommended_pt}% dello shock sul prezzo
        <span style="font-size:0.6em; font-weight:600; color:{rischio_rec_color}; border:1px solid {rischio_rec_color};
        border-radius:12px; padding:2px 10px; margin-left:10px; vertical-align:middle;">RISCHIO {rischio_rec_label}</span>
    </div>
    <p style="color:{COLOR_SECONDARY}; margin-bottom:14px;">
        {CATEGORY_LABELS[categoria]} &middot; shock di costo +{shock_costo_ponderato*100:.1f}%
        &middot; elasticita stimata {eps:.2f}
    </p>
</div>
""", unsafe_allow_html=True)

x1, x2, x3, x4 = st.columns(4)
x1.markdown(f'<div class="kpi-card"><div class="kpi-label">Margine atteso</div>'
            f'<div class="kpi-value">€ {margine_rec:,.0f}</div>'
            f'<div class="kpi-delta delta-neutral">al {recommended_pt}% consigliato</div></div>',
            unsafe_allow_html=True)
x2.markdown(f'<div class="kpi-card"><div class="kpi-label">Variazione prezzo</div>'
            f'<div class="kpi-value">+{variazione_prezzo_rec:.1f}%</div>'
            f'<div class="kpi-delta delta-neutral">€ {prezzo_attuale:.2f} &rarr; € {prezzo_rec:.2f}</div></div>',
            unsafe_allow_html=True)
x3.markdown(f'<div class="kpi-card"><div class="kpi-label">Perdita volumi attesa</div>'
            f'<div class="kpi-value">{perdita_volumi_rec:.1f}%</div>'
            f'<div class="kpi-delta delta-neutral">vs domanda attuale</div></div>',
            unsafe_allow_html=True)
x4.markdown(f'<div class="kpi-card" style="border-top-color:{rischio_rec_color}">'
            f'<div class="kpi-label">Rischio (se segui il consiglio)</div>'
            f'<div class="kpi-value" style="color:{rischio_rec_color}">{rischio_rec_label}</div>'
            f'<div class="kpi-delta delta-neutral">basato sulla perdita volumi</div></div>',
            unsafe_allow_html=True)


st.write("")

# ---------------------------------------------------------------------------
# TABS PRINCIPALI
# ---------------------------------------------------------------------------
tab_config, tab_analisi, tab_sim, tab_decisione = st.tabs(
    ["1. Configurazione", "2. Analisi del modello", "3. Simulazione", "4. Decisione"]
)

# =============================================================================
# TAB 1 — CONFIGURAZIONE (recap + decomposizione shock)
# =============================================================================
with tab_config:
    st.markdown(
        f'<div class="advice-box" style="padding:14px 20px;">'
        f'<b style="color:{CATEGORY_COLORS[categoria]}">{CATEGORY_LABELS[categoria]}</b> — '
        f'{CATEGORY_DESCRIPTIONS[categoria]}</div>',
        unsafe_allow_html=True
    )
    with st.expander("Confronta le tre categorie"):
        cc1, cc2, cc3 = st.columns(3)
        for col, cat in zip([cc1, cc2, cc3], ["fmcg", "durevoli", "premium"]):
            is_selected = cat == categoria
            border = COLOR_ACCENT if is_selected else COLOR_BORDER
            col.markdown(
                f'<div class="kpi-card" style="border-top-color:{border};">'
                f'<div class="kpi-label" style="color:{CATEGORY_COLORS[cat]}; font-size:0.95em;">{CATEGORY_LABELS[cat]}</div>'
                f'<div style="font-size:0.85em; color:{COLOR_SECONDARY}; margin-top:8px; line-height:1.4;">'
                f'{CATEGORY_DESCRIPTIONS[cat]}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown('<div class="section-header">Decomposizione dello shock di costo</div>', unsafe_allow_html=True)
    st.caption("Come i tre aumenti si combinano nello shock complessivo. La quota di costo non legata "
               "a energia/materiali/trasporto resta invariata.")

    f1, f2, f3, f4, f5, f6, f7, f8, f9 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1, 0.35, 1.3])
    with f1:
        st.markdown(f'<div class="flow-box"><div class="flow-label">Energia</div>'
                     f'<div class="flow-value">+{energia_shock}%</div>'
                     f'<div class="flow-label">peso {weights["energy_share"]*100:.0f}%</div></div>',
                     unsafe_allow_html=True)
    with f2:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">+</div>',
                     unsafe_allow_html=True)
    with f3:
        st.markdown(f'<div class="flow-box"><div class="flow-label">Materie prime</div>'
                     f'<div class="flow-value">+{materiali_shock}%</div>'
                     f'<div class="flow-label">peso {weights["materials_share"]*100:.0f}%</div></div>',
                     unsafe_allow_html=True)
    with f4:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">+</div>',
                     unsafe_allow_html=True)
    with f5:
        st.markdown(f'<div class="flow-box"><div class="flow-label">Trasporto</div>'
                     f'<div class="flow-value">+{trasporto_shock}%</div>'
                     f'<div class="flow-label">peso {weights["transport_share"]*100:.0f}%</div></div>',
                     unsafe_allow_html=True)
    with f6:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">+</div>',
                     unsafe_allow_html=True)
    with f7:
        st.markdown(f'<div class="flow-box" style="opacity:0.7;"><div class="flow-label">Altri costi</div>'
                     f'<div class="flow-value">+0%</div>'
                     f'<div class="flow-label">peso {weights["other_cost_share"]*100:.0f}% (invariato)</div></div>',
                     unsafe_allow_html=True)
    with f8:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">&rarr;</div>',
                     unsafe_allow_html=True)
    with f9:
        st.markdown(f'<div class="flow-box" style="border-top:3px solid {COLOR_ACCENT};">'
                     f'<div class="flow-label">Nuovo costo unitario</div>'
                     f'<div class="flow-value">€ {costo_attuale:.2f} &rarr; € {scenarios["cost1"]:.2f}</div>'
                     f'<div class="flow-label">shock ponderato: +{shock_costo_ponderato*100:.1f}%</div></div>',
                     unsafe_allow_html=True)

# =============================================================================
# TAB 2 — ANALISI DEL MODELLO (semplificata: solo il "perché" è in vista)
# =============================================================================
with tab_analisi:
    st.markdown(f'<div class="section-header">Perche il modello suggerisce {recommended_pt}%?</div>',
                unsafe_allow_html=True)

    reasons = f"""
    <div class="advice-box">
        <p>Per la categoria <b>{CATEGORY_LABELS[categoria]}</b> il modello stima un'elasticita
        della domanda di <b>{eps:.2f}</b>, calcolata con il metodo <b>{best_method[categoria]}</b>
        (il piu accurato tra i due testati per questa categoria).</p>
        <p>Con lo shock di costo che hai impostato (<b>+{shock_costo_ponderato*100:.1f}%</b>),
        il modello individua il punto di massimo margine trasferendo il <b>{recommended_pt}%</b>
        di quell'aumento sul prezzo finale. {CATEGORY_RISK_NOTE[categoria]}</p>
    </div>
    """
    st.markdown(reasons, unsafe_allow_html=True)

    with st.expander("Approfondisci — metodologia, validazione e diagnostica del modello"):
        st.markdown("**Confronto tra i metodi di stima**")
        st.caption("Elasticita vera (nota solo perche il dataset e simulato) vs le due stime testate.")
        comp_display = comparison[["elasticita_vera (simulata)", "regressione_log_log",
                                     "random_forest (partial dependence)"]].round(3)
        comp_display.index = [CATEGORY_LABELS[c] for c in comp_display.index]
        st.dataframe(comp_display, use_container_width=True)

        st.markdown("**Curva domanda-prezzo: storico osservato vs estrapolazione**")
        st.caption("Come reagisce la domanda stimata al variare del prezzo. Oltre il +12% (area rossa) "
                   "il modello sta estrapolando oltre i dati storici simulati (±15%).")
        curve = partial_dep[categoria]
        fig_pdep = go.Figure()
        fig_pdep.add_vrect(x0=-15, x1=12, fillcolor=COLOR_POSITIVE, opacity=0.08, line_width=0,
                            annotation_text="storico osservato", annotation_position="top left",
                            annotation_font_size=10, annotation_font_color=COLOR_POSITIVE)
        fig_pdep.add_vrect(x0=12, x1=30, fillcolor=COLOR_NEGATIVE, opacity=0.08, line_width=0,
                            annotation_text="estrapolazione", annotation_position="top right",
                            annotation_font_size=10, annotation_font_color=COLOR_NEGATIVE)
        fig_pdep.add_trace(go.Scatter(
            x=curve["price_change_pct"] * 100, y=curve["predicted_ratio"], mode="lines",
            line=dict(color=color_cat, width=3),
            hovertemplate="Variazione prezzo: %{x:.0f}%<br>Domanda relativa: %{y:.2f}<extra></extra>"
        ))
        fig_pdep.update_layout(template="plotly_white", height=280, margin=dict(l=10, r=10, t=30, b=10),
                                xaxis_title="Variazione di prezzo (%)", yaxis_title="Domanda relativa")
        st.plotly_chart(fig_pdep, use_container_width=True)
        st.caption("L'andamento 'a gradini' è tipico dei modelli ad alberi (random forest). Il tratto "
                   "piatto in estrapolazione non è un errore: oltre quel punto il modello non ha più dati "
                   "da cui imparare. Per questo l'ottimizzazione (Tab Decisione) usa, oltre lo storico, un "
                   "modello teorico a soglia (Sweezy 1939) invece di questa curva.")

        st.markdown("**Importanza delle variabili (Random Forest)**")
        st.caption("Modello generale allenato su tutte le categorie insieme — non specifico per la "
                   "categoria selezionata, non entra direttamente nel calcolo del pass-through.")
        fi_sorted = feature_importance.sort_values(feature_importance.columns[0], ascending=True)
        fi_labels = [FEATURE_LABELS.get(idx, idx) for idx in fi_sorted.index]
        fig_fi = go.Figure(go.Bar(
            x=fi_sorted[fi_sorted.columns[0]], y=fi_labels, orientation="h", marker_color=COLOR_PRIMARY
        ))
        fig_fi.update_layout(template="plotly_white", height=280, margin=dict(l=10, r=10, t=10, b=10),
                              xaxis_title="Importanza relativa")
        st.plotly_chart(fig_fi, use_container_width=True)

        st.markdown("**Assunzioni del modello**")
        st.markdown(f"""
        <div class="assumptions-box">
        Allenato su {N_OBS:,} osservazioni &middot; {N_PRODUCTS} prodotti &middot; 2 anni di storico.<br>
        Elasticita calibrata su letteratura economica (Tellis 1988, Bijmolt et al. 2005), Random Forest
        e regressione log-log.<br>
        Validazione: train/test split per prodotto (80/20) — testato su prodotti mai visti in allenamento.
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# TAB 3 — SIMULAZIONE (tabella scenari + confronto categorie + curve)
# =============================================================================
with tab_sim:
    st.markdown('<div class="section-header">Confronto scenari di pass-through</div>', unsafe_allow_html=True)
    st.caption("Prezzo, quantita, margine e profitto netto (vs scenario consigliato) per diverse scelte.")

    scenario_points = sorted(set([0, 20, 40, 60, 80, 100, recommended_pt]))
    rows = []
    for pt in scenario_points:
        idx = min(int(pt * 2), len(scenarios["pt_grid"]) - 1)
        rows.append({
            "Pass-through": f"{pt}%" + (" (consigliato)" if pt == recommended_pt else ""),
            "Prezzo": f"€ {scenarios['price_grid'][idx]:.2f}",
            "Quantita": f"{scenarios['quantity_grid'][idx]:,.0f}",
            "Margine": f"€ {scenarios['margin_grid'][idx]:,.0f}",
            "Profitto vs consigliato": f"€ {scenarios['margin_grid'][idx] - scenarios['best_margin']:,.0f}",
            "_is_best": pt == recommended_pt,
        })
    scenario_table = pd.DataFrame(rows)
    display_table = scenario_table.drop(columns="_is_best")

    def highlight_best(row):
        is_best = scenario_table.loc[row.name, "_is_best"]
        return ["background-color: #FFF4EC; font-weight: 600" if is_best else "" for _ in row]

    st.dataframe(display_table.style.apply(highlight_best, axis=1),
                 use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Come cambierebbe il consiglio nelle altre categorie?</div>',
                unsafe_allow_html=True)
    st.caption(f"A parita di shock ({shock_costo_ponderato*100:.1f}%), il pass-through ottimale cambia "
               "molto in base all'archetipo di domanda.")

    cat_recs = {c: recommended_pt_for_category(c, shock_costo_ponderato) * 100 for c in
                ["fmcg", "durevoli", "premium"]}
    fig_cat = go.Figure(go.Bar(
        x=list(cat_recs.values()), y=[CATEGORY_LABELS[c] for c in cat_recs],
        orientation="h", marker_color=[CATEGORY_COLORS[c] for c in cat_recs],
        text=[f"{v:.0f}%" for v in cat_recs.values()], textposition="outside"
    ))
    fig_cat.update_layout(template="plotly_white", height=220, margin=dict(l=10, r=40, t=10, b=10),
                           xaxis_title="Pass-through consigliato (%)", xaxis_range=[0, 110])
    st.plotly_chart(fig_cat, use_container_width=True)

    with st.expander("Curve interattive di dettaglio (margine, volumi, ricavo)"):
        tab_m, tab_v, tab_r = st.tabs(["Margine", "Volumi", "Ricavo"])

        def make_chart(y_grid, y_label, hover_fmt, pt_selected):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=scenarios["pt_grid"] * 100, y=y_grid, mode="lines",
                line=dict(color=color_cat, width=3),
                hovertemplate=f"Pass-through: %{{x:.0f}}%<br>{y_label}: %{{y:{hover_fmt}}}<extra></extra>"))
            fig.add_vline(x=pt_selected, line_dash="dot", line_color=COLOR_ACCENT, line_width=2,
                          annotation_text="Scelta attuale", annotation_position="top")
            fig.update_layout(template="plotly_white", height=340, hovermode="x unified",
                               xaxis_title="Pass-through (%)", yaxis_title=y_label,
                               margin=dict(l=10, r=10, t=30, b=10))
            return fig

        with tab_m:
            st.plotly_chart(make_chart(scenarios["margin_grid"], "Margine (EUR/sett.)", ",.0f", recommended_pt),
                             use_container_width=True)
        with tab_v:
            st.plotly_chart(make_chart(scenarios["quantity_grid"], "Quantita (unita/sett.)", ",.0f", recommended_pt),
                             use_container_width=True)
        with tab_r:
            st.plotly_chart(make_chart(scenarios["revenue_grid"], "Ricavo (EUR/sett.)", ",.0f", recommended_pt),
                             use_container_width=True)

# =============================================================================
# TAB 4 — DECISIONE (raccomandazione finale + rischio + confronto + cosa succede se...)
# =============================================================================
CATEGORY_ACTION_NOTE = {
    "fmcg": "Andare oltre rischia di spingere i clienti verso i discount; restare sotto protegge "
            "i volumi ma lascia piu margine sul tavolo.",
    "durevoli": "Andare oltre rischia di far rimandare gli acquisti; restare sotto mantiene i volumi "
                "ma assorbe piu costo extra sul margine.",
    "premium": "Il brand regge bene l'aumento in questo intervallo — assorbirne una parte e piu una "
               "scelta di immagine che una necessita economica.",
}

with tab_decisione:
    st.markdown('<div class="section-header">Raccomandazione</div>', unsafe_allow_html=True)

    reasoning = f"""
    <div class="advice-box">
        <div class="advice-title">Trasferire il {recommended_pt}% dello shock sul prezzo</div>
        <div class="reason-item">- La categoria {CATEGORY_LABELS[categoria]} presenta un'elasticita
            stimata di {eps:.2f} ({best_method[categoria]}).</div>
        <div class="reason-item">- {CATEGORY_ACTION_NOTE[categoria]}</div>
        <div class="reason-item">- Il margine atteso e massimo (€ {scenarios['best_margin']:,.0f}/settimana)
            con un pass-through del {recommended_pt}%.</div>
        <div class="reason-item">- La stima e basata su {N_OBS:,} osservazioni e {N_PRODUCTS} prodotti,
            validata su prodotti mai visti in allenamento.</div>
    </div>
    """
    st.markdown(reasoning, unsafe_allow_html=True)

    # --- scelta attuale (slider) vs consigliato ---
    st.markdown('<div class="section-header">La tua scelta vs il consiglio del modello</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<p style="color:{COLOR_SECONDARY}; font-size:0.92em; margin-bottom:0.8em;">'
        f'Il {recommended_pt}% e il punto che massimizza il margine secondo il modello, ma puoi scegliere '
        f'diversamente — ad esempio per prudenza commerciale, per proteggere l\'immagine del brand, o per '
        f'restare allineata alla concorrenza. <b>Sposta il cursore per simulare una scelta diversa:</b></p>',
        unsafe_allow_html=True
    )
    passthrough_rate = st.slider("Pass-through scelto (%)", 0, 100, recommended_pt, step=1,
                                   label_visibility="visible")

    idx_scelto = min(int(passthrough_rate * 2), len(scenarios["pt_grid"]) - 1)
    idx_cons = scenarios["best_idx"]

    def scenario_at(idx):
        return {
            "prezzo": scenarios["price_grid"][idx], "quantita": scenarios["quantity_grid"][idx],
            "ricavo": scenarios["revenue_grid"][idx], "margine": scenarios["margin_grid"][idx],
        }

    s_scelto, s_cons = scenario_at(idx_scelto), scenario_at(idx_cons)

    col_att, col_cons = st.columns(2)
    with col_att:
        st.markdown(f'<div class="kpi-card" style="border-top-color:{COLOR_SECONDARY}">'
                    f'<div class="kpi-label">Scenario scelto ({passthrough_rate}%)</div>'
                    f'<div class="kpi-value">€ {s_scelto["margine"]:,.0f}</div>'
                    f'<div class="kpi-delta delta-neutral">margine/settimana</div></div>',
                    unsafe_allow_html=True)
    with col_cons:
        delta_vs_cons = s_scelto["margine"] - s_cons["margine"]
        st.markdown(f'<div class="kpi-card" style="border-top-color:{COLOR_ACCENT}">'
                    f'<div class="kpi-label">Scenario consigliato ({recommended_pt}%)</div>'
                    f'<div class="kpi-value">€ {s_cons["margine"]:,.0f}</div>'
                    f'<div class="kpi-delta {"delta-positive" if delta_vs_cons>=0 else "delta-negative"}">'
                    f'{"+" if delta_vs_cons>=0 else ""}€ {delta_vs_cons:,.0f} rispetto alla tua scelta</div></div>',
                    unsafe_allow_html=True)

    # --- indice di qualita della decisione + livello di rischio ---
    st.write("")
    qualita = min(100, max(0, (s_scelto["margine"] / scenarios["best_margin"]) * 100)) \
        if scenarios["best_margin"] > 0 else 100
    perdita_volumi_pct = max(0, (1 - s_scelto["quantita"] / quantita_attuale) * 100)
    rischio, colore_rischio = risk_level(perdita_volumi_pct)

    variazione_prezzo_scelta = (s_scelto["prezzo"] - prezzo_attuale) / prezzo_attuale * 100
    soglia_categoria = KINK_CONFIG[categoria]["threshold"] * 100
    supera_soglia = variazione_prezzo_scelta > soglia_categoria

    i1, i2 = st.columns(2)
    i1.markdown(f'<div class="index-card"><div class="index-value" style="color:{COLOR_PRIMARY}">'
                f'{qualita:.0f}/100</div><div class="index-label">Qualita della decisione '
                f'(vs margine massimo ottenibile)</div></div>', unsafe_allow_html=True)
    i2.markdown(f"""
    <div class="index-card">
        <div class="index-value" style="color:{colore_rischio}">{rischio}</div>
        <div class="index-label">Rischio di perdita volumi</div>
        <div style="margin-top:12px; padding-top:12px; border-top:1px solid {COLOR_BORDER}; font-size:0.85em; color:{COLOR_SECONDARY};">
            Perdita volumi prevista: <b style="color:{colore_rischio}">{perdita_volumi_pct:.1f}%</b><br>
            Aumento di prezzo: <b>+{variazione_prezzo_scelta:.1f}%</b>
            ({"sopra" if supera_soglia else "sotto"} la soglia di rottura della categoria, +{soglia_categoria:.0f}%)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        f'<p style="font-size:0.87em; color:{COLOR_SECONDARY}; margin-top:10px; padding:10px 14px; '
        f'background-color:{COLOR_BG_CARD}; border-radius:6px;">'
        f'<b>Questi due indicatori vanno letti insieme, non separatamente.</b> Un pass-through piu basso '
        f'riduce quasi sempre il rischio di perdita volumi, ma <i>non</i> significa che sia la scelta '
        f'migliore: riduce anche la qualita della decisione, cioe il margine lasciato sul tavolo. '
        f'Il {recommended_pt}% consigliato e il punto che il modello ritiene bilanci meglio i due fattori — '
        f'non necessariamente quello a rischio piu basso in assoluto.</p>',
        unsafe_allow_html=True
    )

    # --- cosa succede se ignoro il consiglio ---
    st.markdown('<div class="section-header">Cosa succede se ti discosti dal consiglio?</div>',
                unsafe_allow_html=True)

    delta_margine_flow = s_scelto["margine"] - s_cons["margine"]
    delta_vendite_flow = (s_scelto["quantita"] / s_cons["quantita"] - 1) * 100
    delta_profitto_flow = delta_margine_flow  # stessa metrica, evidenziata come "profitto netto"

    g1, g2, g3, g4 = st.columns([1, 0.3, 1, 0.3])
    g1, gap1, g2, gap2, g3, gap3, g4 = st.columns([1, 0.2, 1, 0.2, 1, 0.2, 1])
    with g1:
        st.markdown(f'<div class="flow-box"><div class="flow-label">Scelta azienda</div>'
                    f'<div class="flow-value">{passthrough_rate}%</div></div>', unsafe_allow_html=True)
    with gap1:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">&rarr;</div>',
                     unsafe_allow_html=True)
    with g2:
        col = COLOR_POSITIVE if delta_margine_flow >= 0 else COLOR_NEGATIVE
        st.markdown(f'<div class="flow-box"><div class="flow-label">Margine</div>'
                    f'<div class="flow-value" style="color:{col}">{"+" if delta_margine_flow>=0 else ""}'
                    f'€ {delta_margine_flow:,.0f}</div></div>', unsafe_allow_html=True)
    with gap2:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">&rarr;</div>',
                     unsafe_allow_html=True)
    with g3:
        col = COLOR_POSITIVE if delta_vendite_flow >= 0 else COLOR_NEGATIVE
        st.markdown(f'<div class="flow-box"><div class="flow-label">Vendite</div>'
                    f'<div class="flow-value" style="color:{col}">{delta_vendite_flow:+.1f}%</div></div>',
                    unsafe_allow_html=True)
    with gap3:
        st.markdown('<div style="text-align:center; padding-top:35px;" class="flow-arrow">&rarr;</div>',
                     unsafe_allow_html=True)
    with g4:
        col = COLOR_POSITIVE if delta_profitto_flow >= 0 else COLOR_NEGATIVE
        st.markdown(f'<div class="flow-box" style="border-top:3px solid {col};">'
                    f'<div class="flow-label">Profitto netto</div>'
                    f'<div class="flow-value" style="color:{col}">{"+" if delta_profitto_flow>=0 else ""}'
                    f'€ {delta_profitto_flow:,.0f}</div></div>', unsafe_allow_html=True)

    if passthrough_rate == recommended_pt:
        st.caption("Stai seguendo esattamente il consiglio del modello: nessuna perdita di profitto.")
    else:
        st.caption(f"Scegliendo {passthrough_rate}% invece del {recommended_pt}% consigliato, "
                   f"perdi € {abs(delta_profitto_flow):,.0f}/settimana rispetto all'ottimo teorico.")

# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(f'<div style="text-align:center; color:{COLOR_SECONDARY}; font-size:0.82em;">'
            f'Pricing DSS v2.0 — Elasticita calibrata su letteratura accademica '
            f'(Tellis 1988, Bijmolt et al. 2005). Dataset simulato, validato con Random Forest '
            f'e regressione classica.</div>', unsafe_allow_html=True)
