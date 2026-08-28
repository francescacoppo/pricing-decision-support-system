# Dataset simulato — Pricing DSS

Questo dataset è **simulato**, non raccolto da un'azienda reale. I dati sui costi
e sulle vendite di pricing sono tra le informazioni più protette che un'azienda
possieda, quindi — come avviene comunemente nei progetti di ricerca quando i dati
reali non sono accessibili — è stato generato un dataset sintetico, calibrato però
su elasticità-prezzo stimate da letteratura accademica reale.

## Fonti di calibrazione

- **Tellis, G. (1988)**, "The Price Elasticity of Selective Demand: A Meta-Analysis
  of Econometric Models of Sales", *Journal of Marketing Research*, 25(4): 331-341.
  Elasticità media di brand ≈ **-1.76** (367 stime, 220 brand/mercati).
- **Bijmolt, T.H.A., van Heerde, H.J. & Pieters, R.G.M. (2005)**, "New Empirical
  Generalizations on the Determinants of Price Elasticity", *Journal of Marketing
  Research*, 42(2): 141-156. Elasticità media aggiornata ≈ **-2.62** (1.851 stime,
  81 studi); i beni durevoli mostrano sensibilità al prezzo più alta della media,
  i beni di largo consumo (CPG) più bassa della media.
- Studio sui vini ad alto prezzo (*Consumer response to price changes in
  higher-priced brands*, ScienceDirect): elasticità media **-1.8** per brand
  premium, inferiore alla media generale; i beni di lusso sono descritti come
  un'eccezione al pattern standard, con elasticità talvolta prossima allo zero
  o positiva (effetto Veblen).

## Parametri usati nella simulazione

| Categoria | Elasticità media (ε) | Deviazione std | Note |
|---|---|---|---|
| FMCG / largo consumo | -1.8 | 0.35 | + effetto soglia: oltre +8% di aumento, fuga accelerata verso private label |
| Durevoli / voluttuari | -2.8 | 0.50 | Nessuna soglia, reazione più graduale ma di ampiezza maggiore |
| Premium / posizionale | -0.7 | 0.40 | 15% dei prodotti con elasticità quasi nulla o positiva (Veblen) |

## Struttura dei file

**`product_master.csv`** (90 righe, un prodotto per riga)
- `product_id`, `category`, `category_label`, `subcategory`
- `base_price`, `base_quantity_weekly`, `unit_cost`
- `elasticity_true` — elasticità "vera" nascosta nel modello generativo,
  usata SOLO per validare il modello di machine learning a valle, mai
  come input del modello stesso
- `veblen_flag`
- `energy_share`, `materials_share`, `transport_share`, `other_cost_share`
  — scomposizione del costo unitario, usata dal calcolatore dello shock
  di costo nella dashboard

**`transactions.csv`** (9.360 righe, panel settimanale, 104 settimane per prodotto)
- `product_id`, `category`, `subcategory`, `week`
- `price`, `base_price`, `price_change_pct`
- `promotion`, `competitor_price_index`, `seasonality_index`
- `quantity_sold`, `unit_cost`

## Nota metodologica importante

Le variazioni di prezzo storiche simulate restano in un range realistico
(circa **-15% / +12%**), coerente con normali promozioni e adeguamenti di
listino. Questo è intenzionale: quando la dashboard simula uno shock di
costo del +20% o +30%, il modello di machine learning si troverà a
**estrapolare oltre il range osservato nei dati storici**. Questo limite è
dichiarato esplicitamente e gestito nella dashboard con un avviso di
affidabilità, invece di essere nascosto.

## Riproducibilità

Il dataset è generato dallo script `simulate_data.py` con seed fisso
(`numpy.random.default_rng(42)`), quindi è interamente riproducibile.