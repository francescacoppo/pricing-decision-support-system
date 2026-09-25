# Pricing Decision Support System

Un sistema di supporto alla decisione per il pricing in contesti di shock dei costi
(inflazione, aumento di materie prime ed energia): stima quanto di un aumento di costo
conviene trasferire sul prezzo finale, a seconda di come si comporta la domanda nella
categoria di prodotto — e spiega perché.

## Il problema

Quando i costi di produzione aumentano, le aziende devono decidere quanto di quell'aumento
trasferire sul prezzo finale. Trasferirlo tutto rischia di far crollare i volumi di vendita;
non trasferirlo affatto erode il margine. La risposta corretta dipende da quanto i clienti
sono sensibili al prezzo — e questa sensibilità cambia moltissimo a seconda del tipo di
prodotto.

Questo progetto confronta tre archetipi di domanda:

| Categoria | Comportamento | Leva decisionale |
|---|---|---|
| **Largo Consumo** | Domanda relativamente rigida, ma con una soglia oltre cui i clienti passano ai brand commerciali (private label) | Fino a che punto posso alzare il prezzo prima di perdere quota a favore del discount? |
| **Beni Durevoli** | Domanda elastica: di fronte a un aumento, l'acquisto viene rimandato | Conviene assorbire parte dello shock con efficienza operativa invece di scaricarlo tutto? |
| **Premium** | Domanda quasi anelastica — il prezzo alto fa parte del valore percepito — ma rischio di alienare la clientela aspirazionale oltre una certa soglia | Fino a dove il posizionamento regge l'aumento senza intaccare l'immagine del brand? |

## La soluzione — architettura in 4 fasi

```
1. Simulazione dati        →  2. Machine Learning        →  3. Ottimizzazione       →  4. Dashboard
   calibrata su                  stima l'elasticità            calcola il pass-           interattiva che
   letteratura economica         empirica per categoria         through ottimale            spiega la decisione
```

**1. Dati.** Nessun dataset pubblico contiene prezzi/costi/vendite reali di un'azienda
(sono informazioni riservate). È stato generato un dataset sintetico — 90 prodotti,
2 anni di storico settimanale, ~9.360 osservazioni — calibrato su elasticità-prezzo
stimate da letteratura accademica reale (Tellis 1988, Bijmolt et al. 2005), non
inventate a tavolino. Dettagli e fonti in [`data/README.md`](data/README.md).

**2. Machine Learning.** L'elasticità di ciascuna categoria viene stimata dai dati con
due metodi indipendenti — una regressione log-log classica e un random forest, da cui
l'elasticità viene estratta tramite partial dependence — e validata contro la verità
nota nel dataset simulato. Risultato interessante: **nessuno dei due metodi vince
sempre**. La regressione classica batte il random forest sulle categorie omogenee
(Largo Consumo, Durevoli); il random forest vince nettamente sul Premium, la categoria
più eterogenea (mix di prodotti normali e prodotti "Veblen" quasi insensibili al prezzo).

**3. Ottimizzazione.** Il pass-through ottimale viene calcolato massimizzando il
margine atteso, usando l'elasticità stimata allo step 2. Un dettaglio metodologico
rilevante: con un'elasticità costante, il modello raccomandava sempre il pass-through
massimo consentito, indipendentemente dalla categoria — un risultato poco realistico
(nessuna azienda punterebbe al prezzo più alto in assoluto). La correzione: introdurre
una **curva di domanda spezzata** (kinked demand curve, Sweezy 1939) — oltre una soglia
specifica per categoria, la sensibilità al prezzo si inasprisce bruscamente. Questo fa
emergere raccomandazioni realmente differenziate tra categorie e livelli di shock.

**4. Dashboard.** Un'interfaccia Streamlit organizzata in 4 sezioni — Configurazione,
Analisi del modello, Simulazione, Decisione — con un executive summary sempre visibile
in cima. Non si limita a restituire un numero: spiega perché il modello raccomanda quel
pass-through, mostra dove il modello sta interpolando dati osservati e dove sta
estrapolando oltre lo storico, e permette di confrontare "cosa succede se mi discosto
dal consiglio".

## Come eseguirlo in locale

```bash
git clone <url-repository>
cd PRICING_DSS
pip install -r requirements.txt
streamlit run app.py
```

Per rigenerare da zero dati, modelli e ottimizzazione (facoltativo, i risultati sono
già salvati in `data/`):

```bash
python src/01_simulate_data.py
python src/02_train_model.py
python src/03_optimize_pricing.py
```

## Struttura del progetto

```
PRICING_DSS/
├── README.md                          # questo file
├── requirements.txt
├── app.py                             # dashboard Streamlit
├── data/
│   ├── README.md                      # fonti e metodologia del dataset simulato
│   ├── product_master.csv             # anagrafica dei 90 prodotti simulati
│   ├── transactions.csv               # storico settimanale prezzo/quantità
│   ├── elasticity_comparison.csv      # confronto log-log vs random forest vs verità
│   ├── rf_feature_importance.csv
│   ├── rf_partial_dependence_*.csv    # curve di risposta della domanda per categoria
│   └── pricing_scenarios.csv          # scenari di pass-through pre-calcolati
└── src/
    ├── 01_simulate_data.py            # simulazione calibrata su letteratura
    ├── 02_train_model.py              # stima elasticità: log-log vs random forest
    ├── 03_optimize_pricing.py         # ottimizzazione con domanda spezzata
    └── 04_plot_margin_curves.py       # visualizzazione curve di margine
```

## Stack tecnologico

Python · pandas · numpy · scikit-learn (Random Forest) · Streamlit · Plotly

## Metodologia e trasparenza

- I dati sono **simulati ma calibrati** su parametri reali di letteratura economica —
  mai presentati come dati aziendali reali. Dettagli completi in
  [`data/README.md`](data/README.md).
- La stima dell'elasticità è validata con train/test split **per prodotto** (80/20),
  non per riga, per verificare che i modelli generalizzino su prodotti mai visti, non
  che memorizzino il comportamento di prodotti già osservati.
- Le soglie di rottura della domanda per Beni Durevoli e Premium sono un'estrapolazione
  motivata dalla teoria economica (curva di domanda spezzata, Sweezy 1939), **non stimate
  dai dati** — i dati storici simulati coprono solo variazioni di prezzo fino a ±15%. È
  dichiarato esplicitamente nella dashboard quando una previsione esce da quel range.

## Limiti noti

- Il random forest per la stima dell'elasticità è allenato su tutte le categorie insieme,
  non un modello separato per categoria.
- I "prodotti tipici" usati per le simulazioni di scenario sono valori mediani per
  categoria, non i dati di un'azienda reale specifica — nella dashboard, l'utente
  sostituisce questi valori con i propri.
- Le soglie di rottura per Durevoli e Premium sono ipotesi teoriche, non calibrate su
  dati osservati a quei livelli di shock (vedi sopra).

## Autrice

Francesca Coppo — progetto realizzato nell'ambito del percorso di laurea magistrale in
Data Analytics for Business and Society.
