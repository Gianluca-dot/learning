> ⚠️ **AVVISO IMPORTANTE SULLA TRADUZIONE AUTOMATICA:**
> Se stai usando il browser in italiano con la **traduzione automatica attiva** (es. Google Translate integrato in Chrome/Edge), **DISATTIVALA** per questa pagina o per il repository GitHub.
> La traduzione automatica potrebbe modificare i nomi dei file, dei comandi o dei percorsi di sistema (es. trasformando `data/` in `dati/` o `evaluate.py` in `valuta.py`), compromettendo l'esecuzione del codice e i comandi da terminale.


# 📖 Guida Completa per l'Avvio del Progetto

Questa guida contiene tutte le istruzioni dettagliate per installare ed eseguire la piattaforma **learning** sul tuo computer o su Google Colab.

---

## 🔰 1. Prerequisiti

* **Python 3.10+:** [Download Python](https://www.python.org/downloads/) *(durante l'installazione spunta "Add Python to PATH")*.
* **Git:** [Download Git](https://git-scm.com/downloads).

---

## ⚙️ 2. Installazione e Configurazione Locale

Apri il **Terminale** (Mac/Linux) o il **Prompt dei Comandi / PowerShell** (Windows) ed esegui i seguenti comandi in sequenza:

```bash
# 1. Clona il repository sul tuo computer
git clone [https://github.com/MachineInnovators/learning.git](https://github.com/MachineInnovators/learning.git)

# 2. Entra nella cartella del progetto
cd learning

# 3. Crea l'ambiente virtuale
python -m venv venv

# 4. Attiva l'ambiente virtuale
# Su Windows (Prompt dei Comandi):
venv\Scripts\activate
# Su Mac/Linux:
source venv/bin/activate

# 5. Installa le dipendenze bloccate
pip install -r requirements.txt
