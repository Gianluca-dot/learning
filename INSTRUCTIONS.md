# 🚀 Pipeline di Retraining e Monitoraggio MLOps — MachineInnovators Inc.

Questa guida descrive il flusso di lavoro completo per la gestione, il retraining e il monitoraggio del modello di sentiment analysis (`cardiffnlp/twitter-roberta-base-sentiment-latest`). 

Il processo garantisce la riproducibilità tramite dipendenze bloccate, campionamento stratificato dei dati e controlli automatizzati di qualità (*Quality Gates*).

---

## 📋 Prerequisiti
* Un account Google per accedere a **Google Colab**.
* **Accelerazione Hardware (GPU):** Consigliato l'uso di una GPU su Colab (*Runtime -> Cambia tipo di runtime -> T4 GPU*).

---

## 🛠️ Guida Operativa: Esecuzione del Notebook
Il flusso completo è racchiuso all'interno del notebook ufficiale: 
📂 `notebooks/retrain_pipeline.ipynb`

Esegui le celle in sequenza seguendo questi passaggi:

### Passaggio 1: Installazione delle dipendenze e verifica GPU
Questo blocco installa le librerie esatte definite nel file `requirements.txt` della repository e verifica la disponibilità e il nome della GPU attiva.

```python
# Installazione dipendenze dal file requirements.txt della repository
!pip install -r requirements.txt

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA disponibile: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device GPU: {torch.cuda.get_device_name(0)}")

Passaggio 2: Estrazione stratificata dei dati
Esegue lo script modulare per la preparazione e il campionamento dei dati necessari al modello.

# Esecuzione modulare dello script di preparazione dati
!python src/data.py
Passaggio 3: Valutazione Baseline e salvataggio metriche reali
Calcola in modo dinamico l'Accuracy e il punteggio F1-Score del modello pre-addestrato, salvando i risultati nel percorso standardizzato.

# Calcolo dinamico dell'Accuracy e F1-Score con salvataggio in data/metrics.json
!python src/evaluate.py
Passaggio 4: Verifica dell'output generato
Legge e stampa il file data/metrics.json per validare le metriche reali prodotte dalla pipeline prima del rilascio o del fine-tuning.

import json

with open('data/metrics.json', 'r') as f:
    metrics = json.load(f)

print("=== METRICHE REALI GENERATE DALLA PIPELINE ===")
print(json.dumps(metrics, indent=4))
🤖 Automazione CI/CD (GitHub Actions)
La stessa pipeline di valutazione e controllo viene eseguita automaticamente in remoto dai server di GitHub Actions a ogni push sul ramo main, garantendo un monitoraggio continuo dello stato del modello (configurato in .github/workflows/retrain.yml).
