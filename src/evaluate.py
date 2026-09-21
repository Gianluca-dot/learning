import json
import os
import yaml
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from src.data import prepare_test_data
from src.model import SentimentAnalyzer


def run_evaluation(config_path: str = "config/config.yaml") -> dict:
    """Esegue la valutazione reale del modello e salva le metriche in JSON."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_path = cfg["data"]["test_sample_path"]
    metrics_path = cfg["data"]["metrics_output_path"]

    # Genera i dati di test se non esistono ancora
    if not os.path.exists(test_path):
        df_test = prepare_test_data(config_path)
    else:
        df_test = pd.read_csv(test_path)

    analyzer = SentimentAnalyzer(config_path)

    print("Running evaluation on test sample...")
    predictions = analyzer.predict_batch(df_test["text"].tolist())
    y_pred = [p["label"] for p in predictions]
    y_true = df_test["label_text"].tolist()

    # Calcolo metriche reali
    acc = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted"))
    cm = confusion_matrix(y_true, y_pred, labels=cfg["model"]["labels"]).tolist()

    metrics = {
        "accuracy": round(acc, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm,
        "sample_size": len(df_test),
        "model_name": cfg["model"]["name"],
    }

    # Salvataggio dinamico delle metriche reali
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("=== EVALUATION COMPLETED ===")
    print(f"Accuracy : {metrics['accuracy']}")
    print(f"F1 Macro : {metrics['f1_macro']}")
    print(f"Metrics saved dynamically to '{metrics_path}'")

    return metrics


if __name__ == "__main__":
    run_evaluation()
