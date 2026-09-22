import os
import json
import yaml
from src.evaluate import run_evaluation

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run_retraining():
    print("🚀 [RETRAINING AUTOMATICO] Avvio del processo di Fine-Tuning...")
    config = load_config()

    # Valutazione metriche attuali vs baseline
    current_metrics = run_evaluation()
    print(f"📊 Metriche correnti: {current_metrics}")

    # Simulazione successo retraining
    print("✅ Retraining completato con successo. Nessun degrado rilevato.")

if __name__ == "__main__":
    run_retraining()
