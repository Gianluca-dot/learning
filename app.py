import os
import json
import datetime
import pandas as pd
import streamlit as st
from src.model import SentimentAnalyzer


# Configurazione Pagina Streamlit
st.set_page_config(
    page_title="MachineInnovators - Sentiment Monitoring",
    page_icon="📊",
    layout="wide",
)

LOG_FILE = "data/predictions_log.csv"
METRICS_FILE = "data/metrics.json"


@st.cache_resource
def load_sentiment_analyzer():
    """Inizializza e memorizza in cache il modello di analisi del sentiment."""
    return SentimentAnalyzer()


def log_prediction(text: str, label: str, confidence: float):
    """Registra le predizioni in un file CSV per il monitoraggio continuativo."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    file_exists = os.path.exists(LOG_FILE)

    log_entry = pd.DataFrame([{
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text": text,
        "predicted_label": label,
        "confidence": confidence,
    }])

    log_entry.to_csv(LOG_FILE, mode="a", header=not file_exists, index=False)


# --- INTERFACCIA UTENTE ---
st.title("📊 learning Sentiment Analysis & Monitoring Platform")
st.markdown("Applicazione aziendale per il monitoraggio e l'inferenza in tempo reale — **MachineInnovators Inc.**")

st.sidebar.header("⚙️ Navigazione")
page = st.sidebar.radio("Seleziona Modalità:", ["Infezione Singola", "Infezione Batch", "Metriche & Log Monitoring"])

analyzer = load_sentiment_analyzer()

# --- PAGINA 1: INFERENZA SINGOLA ---
if page == "Infezione Singola":
    st.subheader("🔍 Analisi Testo Singolo")
    user_input = st.text_area("Inserisci il testo da analizzare (es. tweet, recensione):", height=100)

    if st.button("Analizza Sentiment"):
        if user_input.strip():
            result = analyzer.predict_single(user_input)
            
            # Registrazione nel log
            log_prediction(user_input, result["label"], result["confidence"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentiment Predetto", result["label"].upper())
                st.metric("Confidenza Modello", f"{result['confidence'] * 100:.2f}%")

            with col2:
                st.write("**Distribuzione Probabilità:**")
                scores_df = pd.DataFrame(list(result["scores"].items()), columns=["Classe", "Probabilità"])
                st.bar_chart(scores_df.set_index("Classe"))
        else:
            st.warning("Inserisci un testo valido prima di procedere.")

# --- PAGINA 2: INFERENZA BATCH ---
elif page == "Infezione Batch":
    st.subheader("📁 Analisi Batch da File CSV")
    uploaded_file = st.file_uploader("Carica un file CSV (deve contenere una colonna 'text')", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            st.write(f"Caricate {len(df)} righe. Avvio elaborazione...")
            
            results = analyzer.predict_batch(df["text"].tolist())
            df["predicted_label"] = [r["label"] for r in results]
            df["confidence"] = [r["confidence"] for r in results]

            st.dataframe(df.head(10))

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Risultati CSV", data=csv, file_name="sentiment_predictions.csv", mime="text/csv")
        else:
            st.error("Il file CSV deve contenere una colonna denominata 'text'.")

# --- PAGINA 3: METRICHE & LOG MONITORING ---
elif page == "Metriche & Log Monitoring":
    st.subheader("📈 Monitoraggio Prestazioni & Log")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Metriche Ufficiali di Valutazione")
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            st.json(metrics)
        else:
            st.info("Nessun file metrics.json trovato. Esegui prima la valutazione.")

    with col2:
        st.markdown("### Log di Runtime Infezioni")
        if os.path.exists(LOG_FILE):
            logs_df = pd.read_csv(LOG_FILE)
            st.dataframe(logs_df.tail(20))
            st.caption(f"Totale predizioni registrate: {len(logs_df)}")
        else:
            st.info("Ancora nessuna predizione registrata nei log.")
