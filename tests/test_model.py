import pytest
from src.model import SentimentAnalyzer


@pytest.fixture(scope="module")
def analyzer():
    """Inizializza una singola istanza dell'analizzatore per l'intera sessione di test."""
    return SentimentAnalyzer()


def test_preprocess_text(analyzer):
    """Verifica la pulizia del testo conforme alle specifiche di Twitter-RoBERTa."""
    raw_text = "Hello @username check this link http://example.com"
    cleaned = analyzer.preprocess_text(raw_text)

    assert "@user" in cleaned
    assert "http" in cleaned
    assert "@username" not in cleaned
    assert "http://example.com" not in cleaned


def test_predict_single_structure(analyzer):
    """Verifica che la predizione singola restituisca le chiavi e le probabilità corrette."""
    sample_text = "MachineInnovators provides great MLOps tools!"
    result = analyzer.predict_single(sample_text)

    assert isinstance(result, dict)
    assert "label" in result
    assert result["label"] in ["negative", "neutral", "positive"]
    assert "confidence" in result
    assert 0.0 <= result["confidence"] <= 1.0
    assert "scores" in result
    assert len(result["scores"]) == 3


def test_predict_batch_length(analyzer):
    """Verifica l'inferenza su un batch di testi."""
    texts = ["I love this product", "This is average", "Terrible support"]
    results = analyzer.predict_batch(texts)

    assert len(results) == 3
    assert results[0]["label"] == "positive"
    assert results[2]["label"] == "negative"
