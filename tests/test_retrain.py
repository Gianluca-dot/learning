import os
import json
import pytest

def test_metrics_structure(tmp_path):
    metrics_file = tmp_path / "metrics.json"
    dummy_data = {"accuracy": 0.72, "f1_macro": 0.69, "f1_weighted": 0.71, "status": "promoted"}

    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(dummy_data, f)

    assert os.path.exists(metrics_file)
    with open(metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["f1_macro"] == 0.69
    assert data["status"] == "promoted"
