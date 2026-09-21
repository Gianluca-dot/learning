# learning Platform

> ⚠️ **AVVISO IMPORTANTE SULLA TRADUZIONE AUTOMATICA:**
> Se stai usando il browser in italiano con la **traduzione automatica attiva** (es. Google Translate integrato in Chrome/Edge), **DISATTIVALA** per questa pagina o per il repository GitHub.
>
> 
> La traduzione automatica potrebbe modificare i nomi dei file, dei comandi o dei percorsi di sistema (es. trasformando `data/` in `dati/` o `evaluate.py` in `valuta.py`), compromettendo l'esecuzione del codice e i comandi da terminale.
[![CI Pipeline](https://github.com/MachineInnovators/learning/actions/workflows/ci.yml/badge.svg)](https://github.com/MachineInnovators/learning/actions/workflows/ci.yml)

Sistema end-to-end di monitoraggio e inferenza del sentiment sviluppato per **MachineInnovators Inc.**. Il progetto utilizza un modello basato su RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) valutato sul benchmark `tweet_eval` (subset: *sentiment*), integrando pratiche MLOps per la riproducibilità, l'automazione dei test di qualità e il tracciamento dei log a runtime.

---

## 📐 Architettura del Progetto

```text
├── .github/
│   └── workflows/
│       └── ci.yml              # Pipeline GitHub Actions (Quality Gate)
├── config/
│   └── config.yaml             # Configurazione centralizzata (soglie, percorsi, iperparametri)
├── data/                       # Artefatti generati a runtime (esclusi da Git)
├── notebooks/
│   └── retrain_pipeline.ipynb  # Pipeline modulare per Google Colab su GPU
├── src/
│   ├── __init__.py             # Inizializzazione pacchetto Python
│   ├── data.py                 # Caricamento e campionamento stratificato
│   ├── model.py                # Inizializzazione, preprocessing e inferenza RoBERTa
│   └── evaluate.py             # Calcolo metriche reali e generazione metrics.json
├── tests/
│   ├── __init__.py
│   ├── test_data.py            # Test su estrazione dati e integrità campioni
│   ├── test_model.py           # Test su preprocessing e struttura inferenze
│   └── test_evaluate.py        # Test di confronto su Quality Gate (Accuracy/F1)
├── app.py                      # Dashboard di monitoraggio e inferenza (Streamlit)
├── requirements.txt            # Dipendenze Python con versioni bloccate
└── README.md                   # Documentazione di progetto
