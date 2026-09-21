import os
import json
from src.evaluate import run_evaluation, load_config


def test_run_evaluation_and_quality_gate():
    """Esegue la valutazione completa e verifica che le metriche superino il Quality Gate."""
    cfg = load_config()
    metrics = run_evaluation()

    metrics_path = cfg["data"]["metrics_output_path"]
    assert os.path.exists(metrics_path), "Il file metrics.json non è stato generato."

    with open(metrics_path, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)

    # Verifica integrità struttura JSON
    assert "accuracy" in saved_metrics
    assert "f1_macro" in saved_metrics
    assert "confusion_matrix" in saved_metrics

    # --- QUALITY GATE ---
    min_acc = cfg["quality_gate"]["min_accuracy"]
    min_f1 = cfg["quality_gate"]["min_f1_macro"]

    assert saved_metrics["accuracy"] >= min_acc, (
        f"Quality Gate Fallito: Accuracy {saved_metrics['accuracy']} < {min_acc}"
    )
    assert saved_metrics["f1_macro"] >= min_f1, (
        f"Quality Gate Fallito: F1-Macro {saved_metrics['f1_macro']} < {min_f1}"
    )
