import os
import pandas as pd
from src.data import prepare_test_data, load_config


def test_prepare_test_data_output_exists():
    """Verifica che il dataset di test venga creato fisicamente su disco."""
    cfg = load_config()
    output_path = cfg["data"]["test_sample_path"]

    df = prepare_test_data()

    assert os.path.exists(output_path), f"Il file {output_path} non è stato creato."
    assert not df.empty, "Il DataFrame estratto è vuoto."
    assert len(df) == cfg["data"]["sample_size"], f"Ci si attendeva un campione di {cfg['data']['sample_size']} righe."


def test_prepare_test_data_columns_and_labels():
    """Verifica la presenza delle colonne fondamentali e della mappatura delle etichette."""
    df = prepare_test_data()

    required_columns = {"text", "label", "label_text"}
    assert required_columns.issubset(df.columns), f"Mancano colonne richieste: {required_columns - set(df.columns)}"

    expected_labels = {"negative", "neutral", "positive"}
    unique_labels = set(df["label_text"].unique())
    assert unique_labels.issubset(expected_labels), f"Trovate etichette non valide: {unique_labels}"
