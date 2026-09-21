import os
import yaml
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carica la configurazione centralizzata YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_test_data(config_path: str = "config/config.yaml") -> pd.DataFrame:
    """Carica il dataset TweetEval, applica il campionamento stratificato

    e salva il campione di test per garantire la riproducibilità.
    """
    cfg = load_config(config_path)

    dataset_name = cfg["data"]["dataset_name"]
    subset = cfg["data"]["dataset_subset"]
    sample_size = cfg["data"]["sample_size"]
    seed = cfg["data"]["random_seed"]
    output_path = cfg["data"]["test_sample_path"]

    print(f"Loading dataset '{dataset_name}' ({subset})...")
    raw_dataset = load_dataset(dataset_name, subset, split="test")
    df = pd.DataFrame(raw_dataset)

    # Correzione del Bias: Campionamento stratificato invece di head(200)
    if len(df) > sample_size:
        _, df_sampled = train_test_split(
            df,
            test_size=sample_size,
            stratify=df["label"],
            random_state=seed,
        )
    else:
        df_sampled = df.copy()

    # Mappatura etichette numeriche -> testo
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df_sampled["label_text"] = df_sampled["label"].map(label_map)

    # Salvataggio su file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sampled.to_csv(output_path, index=False)
    print(f"Dataset estratto e salvato con successo in '{output_path}' ({len(df_sampled)} righe).")

    return df_sampled


if __name__ == "__main__":
    prepare_test_data()
