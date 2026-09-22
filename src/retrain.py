import os
import json
import yaml
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_weighted = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted)
    }

def run_retraining(config_path: str = "config/config.yaml", metrics_path: str = "data/metrics.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuratore non trovato: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_name = config["model"]["name"]
    save_dir = config["model"]["save_directory"]
    min_improvement = config["training"].get("min_f1_improvement", 0.005)

    print(f"🔄 Avvio pipeline di Retraining per: {model_name}")

    print("📊 Caricamento dataset 'tweet_eval' (subset sentiment)...")
    dataset = load_dataset(config["data"]["dataset_name"], config["data"]["subset"])

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=config["model"].get("max_length", 128))

    print("🔤 Tokenizzazione del dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
    eval_dataset = tokenized_datasets["validation"].select(range(500))

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

    training_args = TrainingArguments(
        output_dir=config["training"]["output_dir"],
        num_train_epochs=config["training"]["epochs"],
        per_device_train_batch_size=config["training"]["batch_size"],
        per_device_eval_batch_size=config["training"]["batch_size"],
        learning_rate=float(config["training"]["learning_rate"]),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        disable_tqdm=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("🏋️ Inizio Fine-Tuning reale (Trainer.train)...")
    trainer.train()

    print("🧪 Valutazione nuovo modello...")
    eval_results = trainer.evaluate()
    new_f1_macro = eval_results.get("eval_f1_macro", 0.0)
    print(f"📈 Nuovo F1 Macro ottenuto: {new_f1_macro:.4f}")

    old_f1_macro = 0.0
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                old_metrics = json.load(f)
                old_f1_macro = old_metrics.get("f1_macro", 0.0)
        except Exception:
            old_f1_macro = 0.0

    print(f"📉 Vecchio F1 Macro registrato: {old_f1_macro:.4f}")
    improvement = new_f1_macro - old_f1_macro
    print(f"⚖️ Delta F1: {improvement:+.4f} (Soglia richiesta: +{min_improvement})")

    if improvement >= min_improvement or old_f1_macro == 0.0:
        print("🎉 PROMOZIONE ACCETTATA: Il nuovo modello migliora le prestazioni!")
        os.makedirs(save_dir, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"💾 Modello salvato in locale: '{save_dir}'")

        updated_metrics = {
            "accuracy": eval_results.get("eval_accuracy", 0.0),
            "f1_macro": new_f1_macro,
            "f1_weighted": eval_results.get("eval_f1_weighted", 0.0),
            "status": "promoted"
        }
        os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(updated_metrics, f, indent=4)
        print(f"📄 Metriche aggiornate in: '{metrics_path}'")
    else:
        print("🛑 PROMOZIONE RIFIUTATA: Il miglioramento non raggiunge la soglia minima.")
        print(" Vecchio modello mantenuto.")

if __name__ == "__main__":
    run_retraining()
