import torch
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class SentimentAnalyzer:
    """Classe di inferenza pulita e modulare per l'analisi del Sentiment

    utilizzando il modello RoBERTa di CardiffNLP.
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.model_name = self.cfg["model"]["name"]
        self.max_length = self.cfg["model"]["max_length"]
        self.labels = self.cfg["model"]["labels"]

        # Caricamento Tokenizer e Modello Pre-Addestrato
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self.model.eval()

    def preprocess_text(self, text: str) -> str:
        """Preprocessing conforme alla Model Card di Twitter-RoBERTa."""
        new_text = []
        for t in text.split(" "):
            t = "@user" if t.startswith("@") and len(t) > 1 else t
            t = "http" if t.startswith("http") else t
            new_text.append(t)
        return " ".join(new_text)

    def predict_single(self, text: str) -> dict:
        """Esegue l'inferenza su un singolo testo e restituisce classe e confidenza."""
        cleaned_text = self.preprocess_text(text)
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding=True,
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits[0].softmax(dim=0)

        confidence, prediction_idx = torch.max(scores, dim=0)
        predicted_label = self.labels[prediction_idx.item()]

        return {
            "text": text,
            "cleaned_text": cleaned_text,
            "label": predicted_label,
            "confidence": round(confidence.item(), 4),
            "scores": {
                label: round(score.item(), 4)
                for label, score in zip(self.labels, scores)
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Esegue l'inferenza su una lista di testi."""
        return [self.predict_single(t) for t in texts]


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    sample_text = "Great service and amazing experience with MachineInnovators!"
    res = analyzer.predict_single(sample_text)
    print(f"Sample Prediction:\n{res}")
