import os
import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
import joblib

# Set Keras backend to torch before importing keras
os.environ["KERAS_BACKEND"] = "torch"
try:
    import keras
except ImportError:
    keras = None

# Suppress scikit-learn unpickling warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ==================== STREAMLIT CONFIGURATION ====================
st.set_page_config(
    page_title="FinSight: Real-Time Sentiment & Model Comparison",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Strict Pure White Background & Pure Black Font Styling
st.markdown("""
<style>
    /* Force Pure White Background globally across all Streamlit elements */
    html, body, .stApp, 
    [data-testid="stAppViewContainer"], 
    [data-testid="stHeader"], 
    [data-testid="stMain"],
    [data-testid="stToolbar"],
    .main .block-container {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
        color: #000000 !important;
    }

    /* Force Sidebar White Background */
    section[data-testid="stSidebar"], 
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"] {
        background-color: #FAFAFA !important;
        background: #FAFAFA !important;
        border-right: 1px solid #E5E7EB !important;
    }

    /* Force Pure Black Font for specific text elements without breaking icon fonts */
    h1, h2, h3, h4, h5, h6,
    .app-title, .section-title,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] label,
    .stText,
    [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        color: #000000 !important;
    }

    /* Primary submit button: crisp black background with high-contrast WHITE text */
    button[kind="primary"],
    [data-testid="baseButton-primary"],
    [data-testid="stFormSubmitButton"] button {
        background-color: #000000 !important;
        border: 1px solid #000000 !important;
        border-radius: 6px !important;
    }
    button[kind="primary"] *,
    [data-testid="baseButton-primary"] *,
    [data-testid="stFormSubmitButton"] button *,
    [data-testid="stFormSubmitButton"] button p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Secondary buttons (quick tickers) */
    button[kind="secondary"],
    [data-testid="baseButton-secondary"],
    .stButton button {
        background-color: #FFFFFF !important;
        border: 1px solid #D1D5DB !important;
        border-radius: 6px !important;
    }
    button[kind="secondary"] *,
    [data-testid="baseButton-secondary"] *,
    .stButton button *,
    .stButton button p {
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Expander styling & icon protection */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] details summary {
        color: #000000 !important;
    }
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Source Sans Pro", sans-serif !important;
    }

    /* Layout constraints */
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 3rem !important;
        max-width: 1350px !important;
    }

    /* Inputs & Selectboxes */
    input, textarea, [data-baseweb="input"], [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #D1D5DB !important;
    }
    
    /* Clean Cards & Containers */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 8px !important;
        box-shadow: none !important;
    }

    /* App Header */
    .app-header {
        margin-bottom: 1.6rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .app-title {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        color: #000000 !important;
        margin: 0;
    }
    .app-subtitle {
        font-size: 0.94rem;
        color: #374151 !important;
        margin-top: 0.35rem;
        line-height: 1.45;
    }

    /* Minimal Section Headings */
    .section-title {
        font-size: 1.18rem;
        font-weight: 700;
        color: #000000 !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .section-desc {
        font-size: 0.86rem;
        color: #4B5563 !important;
        margin-bottom: 1rem;
        line-height: 1.4;
    }

    /* Minimal Sentiment Badges */
    .pill-pos {
        display: inline-flex;
        align-items: center;
        background-color: #ECFDF5;
        color: #065F46 !important;
        border: 1px solid #A7F3D0;
        padding: 2px 9px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-neg {
        display: inline-flex;
        align-items: center;
        background-color: #FEF2F2;
        color: #991B1B !important;
        border: 1px solid #FECACA;
        padding: 2px 9px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pill-neu {
        display: inline-flex;
        align-items: center;
        background-color: #F4F4F5;
        color: #18181B !important;
        border: 1px solid #D4D4D8;
        padding: 2px 9px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }

    /* Minimalist Distribution Bar */
    .min-bar-wrapper {
        display: flex;
        height: 6px;
        border-radius: 3px;
        overflow: hidden;
        background-color: #E5E7EB;
        margin: 10px 0 12px 0;
    }
    .min-bar-pos { background-color: #16A34A; }
    .min-bar-neu { background-color: #71717A; }
    .min-bar-neg { background-color: #DC2626; }

    /* Summary Bar */
    .summary-badge {
        background-color: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 1.4rem;
        color: #000000 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== HISTORICAL EVALUATION METRICS ====================
HISTORICAL_METRICS = {
    "FinBERT": {
        "name": "FinBERT",
        "tag": "Pre-trained Transformer",
        "accuracy": "78.97%",
        "f1_label": "Negative F1:",
        "f1_val": "0.64",
        "weighted_f1": "0.79",
        "description": "Pre-trained financial BERT fine-tuned on financial text. Uses bidirectional attention to capture context and financial terminology."
    },
    "SVM": {
        "name": "Linear SVM",
        "tag": "LinearSVC + SMOTE",
        "accuracy": "73.99%",
        "f1_label": "Negative F1:",
        "f1_val": "0.47",
        "weighted_f1": "0.74",
        "description": "Linear Support Vector Classifier trained on 5,000 TF-IDF features with SMOTE class balancing for fast maximum-margin decision boundaries."
    },
    "Logistic Regression": {
        "name": "Logistic Regression",
        "tag": "LogisticRegression + SMOTE",
        "accuracy": "73.62%",
        "f1_label": "Negative F1:",
        "f1_val": "0.52",
        "weighted_f1": "0.74",
        "description": "L2-regularized Logistic Regression classifier using 5,000 TF-IDF features with SMOTE oversampling for calibrated probabilistic predictions."
    },
    "BiLSTM": {
        "name": "BiLSTM",
        "tag": "Bidirectional LSTM (Weighted)",
        "accuracy": "72.00%",
        "f1_label": "Weighted F1:",
        "f1_val": "0.72",
        "weighted_f1": "0.72",
        "description": "Bidirectional Long Short-Term Memory recurrent neural network capturing sequential forward and backward textual dependencies."
    }
}

# ==================== MODEL RESOLUTION & LOADING ====================
REQUIRED_FILES = [
    "svm_smote_model.joblib",
    "logistic_regression_smote_model.joblib",
    "tfidf_vectorizer.joblib"
]

def resolve_models_path():
    """Resolves models path prioritizing Google Drive, with local fallbacks."""
    colab_path = "/content/drive/MyDrive/FinSight/models"
    candidate_paths = [
        colab_path,
        os.environ.get("FINSIGHT_MODELS_PATH", ""),
        os.path.join(os.getcwd(), "models"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
        os.path.join(os.getcwd(), "models-20261112T195822Z-1-001", "models"),
    ]
    for candidate in candidate_paths:
        if candidate and os.path.exists(candidate):
            if all(os.path.exists(os.path.join(candidate, f)) for f in REQUIRED_FILES):
                return candidate
    return colab_path

def tokenize_for_bilstm(headlines, bilstm_tokenizer=None, tfidf_vec=None, max_len=100):
    """
    Transforms headlines into padded integer sequences of shape (len, max_len).
    Uses bilstm_tokenizer if available, or tfidf vocabulary as aligned fallback.
    """
    if bilstm_tokenizer is not None and hasattr(bilstm_tokenizer, "texts_to_sequences"):
        seqs = bilstm_tokenizer.texts_to_sequences(headlines)
    elif bilstm_tokenizer is not None and isinstance(bilstm_tokenizer, dict):
        seqs = [[bilstm_tokenizer.get(w.lower(), 1) for w in h.split()] for h in headlines]
    elif tfidf_vec is not None and hasattr(tfidf_vec, "vocabulary_"):
        # Aligned vocabulary mapping: index + 1 (reserve 0 for padding)
        seqs = [[tfidf_vec.vocabulary_.get(w.lower(), 1) + 1 for w in h.split()] for h in headlines]
    else:
        seqs = [[1] for _ in headlines]

    padded = []
    for s in seqs:
        if not s:
            s = [0]
        s_cut = s[:max_len]
        s_pad = s_cut + [0] * max(0, max_len - len(s_cut))
        padded.append(s_pad)
    return np.array(padded, dtype=np.float32)

@st.cache_resource(show_spinner="Loading FinBERT, SVM, Logistic Regression, BiLSTM, and Tokenizer...")
def load_all_models(models_dir):
    """Loads all four models (FinBERT, SVM, Logistic Regression, BiLSTM) and vectorizer."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    finbert_path = os.path.join(models_dir, "finbert_pretrained")
    svm_path = os.path.join(models_dir, "svm_smote_model.joblib")
    lr_path = os.path.join(models_dir, "logistic_regression_smote_model.joblib")
    tfidf_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    bilstm_path = os.path.join(models_dir, "bilstm_weighted_model.keras")

    # Verify classical model files
    missing = []
    for p, desc in [
        (svm_path, "SVM model"),
        (lr_path, "Logistic Regression model"),
        (tfidf_path, "TF-IDF Vectorizer")
    ]:
        if not os.path.exists(p):
            missing.append(f"{desc} (`{p}`)")

    if missing:
        raise FileNotFoundError("Missing model files:\n• " + "\n• ".join(missing))

    # 1. Load FinBERT: local directory if present, else Hugging Face Hub
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(finbert_path):
        tokenizer = AutoTokenizer.from_pretrained(finbert_path)
        finbert_model = AutoModelForSequenceClassification.from_pretrained(finbert_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

    finbert_model.to(device)
    finbert_model.eval()

    # 2. Classical Models
    svm_model = joblib.load(svm_path)
    lr_model = joblib.load(lr_path)
    tfidf_vectorizer = joblib.load(tfidf_path)

    # 3. BiLSTM Model & Tokenizer
    bilstm_model = None
    bilstm_tok = None
    if os.path.exists(bilstm_path) and keras is not None:
        try:
            bilstm_model = keras.models.load_model(bilstm_path)
        except Exception as e:
            st.warning(f"Note: Could not load BiLSTM model: {e}")

    # Check for optional BiLSTM tokenizer file
    tok_candidates = [
        os.path.join(models_dir, "bilstm_tokenizer.pickle"),
        os.path.join(models_dir, "bilstm_tokenizer.pkl"),
        os.path.join(models_dir, "tokenizer.pickle"),
        os.path.join(models_dir, "tokenizer.pkl"),
        os.path.join(models_dir, "bilstm_tokenizer.json"),
    ]
    for tc in tok_candidates:
        if os.path.exists(tc):
            try:
                import pickle
                with open(tc, "rb") as f:
                    bilstm_tok = pickle.load(f)
                break
            except Exception:
                pass

    return {
        "tokenizer": tokenizer,
        "finbert": finbert_model,
        "device": device,
        "svm": svm_model,
        "lr": lr_model,
        "vectorizer": tfidf_vectorizer,
        "bilstm": bilstm_model,
        "bilstm_tokenizer": bilstm_tok,
        "models_dir": models_dir
    }

def standardize_label(raw_label):
    if raw_label is None:
        return "Neutral"
    s = str(raw_label).strip().lower()
    if "pos" in s:
        return "Positive"
    elif "neg" in s:
        return "Negative"
    elif "neu" in s:
        return "Neutral"
    return str(raw_label).capitalize()

def run_multi_model_inference(headlines, bundle):
    import torch
    if not headlines:
        return []

    tokenizer = bundle["tokenizer"]
    finbert = bundle["finbert"]
    device = bundle["device"]
    svm = bundle["svm"]
    lr = bundle["lr"]
    vectorizer = bundle["vectorizer"]
    bilstm = bundle.get("bilstm")
    bilstm_tok = bundle.get("bilstm_tokenizer")

    # 1. FinBERT Inference
    encoded = tokenizer(
        headlines,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        logits = finbert(**encoded).logits
        probs = torch.softmax(logits, dim=-1)
        pred_indices = torch.argmax(probs, dim=-1).cpu().tolist()
        confidences = torch.max(probs, dim=-1).values.cpu().tolist()

    id2label = finbert.config.id2label
    finbert_preds = [
        standardize_label(id2label.get(idx, id2label.get(str(idx), "Neutral")))
        for idx in pred_indices
    ]

    # 2. Vectorization & Classical Models
    X_tfidf = vectorizer.transform(headlines)
    svm_preds = [standardize_label(p) for p in svm.predict(X_tfidf)]
    lr_preds = [standardize_label(p) for p in lr.predict(X_tfidf)]

    # 3. BiLSTM Inference
    bilstm_preds = []
    bilstm_confidences = []
    if bilstm is not None:
        X_bilstm = tokenize_for_bilstm(headlines, bilstm_tok, vectorizer, max_len=100)
        try:
            bilstm_raw = bilstm.predict(X_bilstm, verbose=0)
            b_indices = np.argmax(bilstm_raw, axis=-1)
            b_confs = np.max(bilstm_raw, axis=-1).tolist()
            # Mapping: 0: negative, 1: neutral, 2: positive (from training notebook)
            b_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
            bilstm_preds = [b_map.get(idx, "Neutral") for idx in b_indices]
            bilstm_confidences = b_confs
        except Exception as e:
            bilstm_preds = ["Neutral"] * len(headlines)
            bilstm_confidences = [0.0] * len(headlines)
    else:
        bilstm_preds = ["Neutral"] * len(headlines)
        bilstm_confidences = [0.0] * len(headlines)

    results = []
    for i in range(len(headlines)):
        fb = finbert_preds[i]
        sv = svm_preds[i]
        lrg = lr_preds[i]
        bl = bilstm_preds[i]
        # Full consensus across models
        is_consensus = (fb == sv == lrg == bl) if bilstm is not None else (fb == sv == lrg)
        results.append({
            "headline": headlines[i],
            "finbert": fb,
            "finbert_confidence": confidences[i],
            "svm": sv,
            "lr": lrg,
            "bilstm": bl,
            "bilstm_confidence": bilstm_confidences[i] if i < len(bilstm_confidences) else 0.0,
            "is_consensus": is_consensus
        })
    return results

def fetch_yfinance_news(ticker_symbol):
    import yfinance as yf
    clean_sym = ticker_symbol.strip().upper()
    try:
        ticker_obj = yf.Ticker(clean_sym)
        raw_news = ticker_obj.news
    except Exception as e:
        st.warning(f"Error fetching news for {clean_sym}: {e}")
        return []

    if not raw_news:
        return []

    articles = []
    for item in raw_news:
        title = ""
        publisher = "Yahoo Finance"
        time_str = "N/A"
        url = ""

        if "content" in item and isinstance(item["content"], dict):
            cnt = item["content"]
            title = cnt.get("title", "").strip()
            publisher = cnt.get("provider", {}).get("displayName", "Yahoo Finance")
            raw_time = cnt.get("pubDate") or cnt.get("displayTime")
            url = cnt.get("canonicalUrl", {}).get("url") or cnt.get("clickThroughUrl", {}).get("url") or ""
        else:
            title = item.get("title", "").strip()
            publisher = item.get("publisher", "Yahoo Finance")
            raw_time = item.get("providerPublishTime")
            url = item.get("link", "")

        if raw_time:
            if isinstance(raw_time, (int, float)):
                try:
                    time_str = datetime.fromtimestamp(raw_time).strftime("%Y-%m-%d %H:%M UTC")
                except Exception:
                    time_str = str(raw_time)
            elif isinstance(raw_time, str):
                if "T" in raw_time:
                    try:
                        clean_iso = raw_time.replace("Z", "+00:00")
                        time_str = datetime.fromisoformat(clean_iso).strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        time_str = raw_time
                else:
                    time_str = raw_time

        if title:
            articles.append({
                "headline": title,
                "publisher": publisher,
                "time": time_str,
                "url": url
            })
    return articles

def calculate_sentiment_summary(predictions):
    if not predictions:
        return {
            "pos_pct": 0.0, "neg_pct": 0.0, "neu_pct": 0.0,
            "pos_count": 0, "neg_count": 0, "neu_count": 0,
            "dominant": "Neutral", "total": 0
        }
    total = len(predictions)
    pos_c = sum(1 for p in predictions if p == "Positive")
    neg_c = sum(1 for p in predictions if p == "Negative")
    neu_c = sum(1 for p in predictions if p == "Neutral")

    counts = {"Positive": pos_c, "Negative": neg_c, "Neutral": neu_c}
    dominant = max(counts, key=counts.get)

    return {
        "pos_pct": (pos_c / total) * 100,
        "neg_pct": (neg_c / total) * 100,
        "neu_pct": (neu_c / total) * 100,
        "pos_count": pos_c,
        "neg_count": neg_c,
        "neu_count": neu_c,
        "dominant": dominant,
        "total": total
    }

def get_sentiment_pill(sentiment):
    if sentiment == "Positive":
        return '<span class="pill-pos">● Positive</span>'
    elif sentiment == "Negative":
        return '<span class="pill-neg">● Negative</span>'
    else:
        return '<span class="pill-neu">● Neutral</span>'

# ==================== MAIN MINIMAL APPLICATION ====================
def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">FinSight: Real-Time Sentiment &amp; Model Comparison</h1>
        <div class="app-subtitle">
            A minimalist evaluation interface comparing <b>FinBERT</b>, <b>Linear SVM</b>, <b>Logistic Regression</b>, and <b>BiLSTM</b> on real-time market headlines.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar: Clean Controls
    st.sidebar.markdown("**Stock Selection**")

    # Quick Pick Chips
    quick_tickers = ["AAPL", "NVDA", "MSFT", "TSLA", "GOOGL", "AMZN"]
    chip_cols = st.sidebar.columns(3)
    for i, sym in enumerate(quick_tickers):
        with chip_cols[i % 3]:
            if st.button(sym, key=f"chip_{sym}", use_container_width=True):
                st.session_state["ticker_input_val"] = sym
                st.session_state["auto_submit"] = True

    # Main Ticker Input Form
    current_val = st.session_state.get("ticker_input_val", "AAPL")
    with st.sidebar.form("ticker_form", clear_on_submit=False):
        entered_ticker = st.text_input(
            "Ticker Symbol:",
            value=current_val,
            max_chars=10,
            help="Default is 'AAPL'. Supports any standard symbol (e.g. AAPL, NVDA, GOOGL)."
        ).strip().upper()
        submit_btn = st.form_submit_button("Fetch Live News", type="primary", use_container_width=True)

    # Model Storage Diagnostic
    detected_path = resolve_models_path()
    with st.sidebar.expander("Model Storage Settings", expanded=False):
        st.caption("Target Drive path: `/content/drive/MyDrive/FinSight/models`")
        user_path = st.text_input("Active Path:", value=detected_path)
        active_models_dir = user_path.strip() if user_path.strip() else detected_path
        if os.path.exists(active_models_dir):
            st.caption(f"✓ Found models at `{active_models_dir}`")
        else:
            st.caption(f"⚠️ Directory not found: `{active_models_dir}`")

    # Load Model Bundle
    try:
        bundle = load_all_models(active_models_dir)
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Tip: Mount Google Drive or verify local folder `./models` contains required files.")
        return

    execute_fetch = submit_btn or st.session_state.get("auto_submit", False) or "articles" not in st.session_state
    if st.session_state.get("auto_submit", False):
        st.session_state["auto_submit"] = False

    # Fetch news & run inference
    if execute_fetch:
        active_ticker = entered_ticker if entered_ticker else "AAPL"
        with st.spinner(f"Fetching news for {active_ticker}..."):
            raw_articles = fetch_yfinance_news(active_ticker)
            if not raw_articles:
                st.warning(f"No recent news found for '{active_ticker}'.")
                st.session_state["articles"] = []
                st.session_state["inferences"] = []
                st.session_state["ticker"] = active_ticker
            else:
                headlines = [a["headline"] for a in raw_articles]
                with st.spinner("Analyzing sentiment across all 4 models..."):
                    inferences = run_multi_model_inference(headlines, bundle)
                    st.session_state["articles"] = raw_articles
                    st.session_state["inferences"] = inferences
                    st.session_state["ticker"] = active_ticker

    articles = st.session_state.get("articles", [])
    inferences = st.session_state.get("inferences", [])
    ticker = st.session_state.get("ticker", "AAPL")

    if not articles or not inferences:
        st.info("Enter a stock ticker in the sidebar and click **'Fetch Live News'** to begin.")
        return

    # Compute Sentiment Metrics
    fb_preds = [item["finbert"] for item in inferences]
    svm_preds = [item["svm"] for item in inferences]
    lr_preds = [item["lr"] for item in inferences]
    bl_preds = [item["bilstm"] for item in inferences]

    fb_summary = calculate_sentiment_summary(fb_preds)
    svm_summary = calculate_sentiment_summary(svm_preds)
    lr_summary = calculate_sentiment_summary(lr_preds)
    bl_summary = calculate_sentiment_summary(bl_preds)

    total_count = len(inferences)
    consensus_count = sum(1 for item in inferences if item["is_consensus"])
    consensus_rate = (consensus_count / total_count * 100) if total_count > 0 else 0
    discrepancy_count = total_count - consensus_count

    # Section 1: Summary Bar
    st.markdown(
        f'<div class="summary-badge">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">'
        f'<div><span style="font-size: 0.85rem; color: #4B5563;">Target Asset:</span> <b>${ticker}</b> &nbsp;|&nbsp; <span style="font-size: 0.85rem; color: #4B5563;">Headlines:</span> <b>{total_count}</b></div>'
        f'<div><span style="font-size: 0.85rem; color: #4B5563;">Full 4-Model Consensus:</span> <b>{consensus_count}/{total_count} ({consensus_rate:.1f}%)</b> &nbsp;|&nbsp; <span style="font-size: 0.85rem; color: #4B5563;">Discrepancies:</span> <b>{discrepancy_count}</b></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Section 2: Model Comparison Cards (Four Columns)
    st.markdown('<div class="section-title">Model Sentiment Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Side-by-side comparison of live aggregated sentiment distribution and static test set benchmarks across all four models.</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    models_config = [
        (col1, fb_summary, HISTORICAL_METRICS["FinBERT"]),
        (col2, svm_summary, HISTORICAL_METRICS["SVM"]),
        (col3, lr_summary, HISTORICAL_METRICS["Logistic Regression"]),
        (col4, bl_summary, HISTORICAL_METRICS["BiLSTM"]),
    ]

    for col, summary, hist in models_config:
        with col:
            with st.container(border=True):
                # Header row: Name + Dominant Pill
                st.markdown(
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">'
                    f'<span style="font-size: 1.15rem; font-weight: 800; color: #000000;">{hist["name"]}</span>'
                    f'{get_sentiment_pill(summary["dominant"])}'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(f'<div style="font-size: 0.80rem; color: #374151; min-height: 48px; margin-bottom: 8px;">{hist["description"]}</div>', unsafe_allow_html=True)

                # Aggregated Live Metrics
                st.markdown('<div style="font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #000000; margin-bottom: 6px;">Live Sentiment Breakdown</div>', unsafe_allow_html=True)
                s1, s2, s3 = st.columns(3)
                s1.markdown(
                    f'<div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 6px; text-align: center;">'
                    f'<div style="color: #065F46; font-weight: 800; font-size: 0.88rem;">{summary["pos_pct"]:.1f}%</div>'
                    f'<div style="color: #000000; font-size: 0.70rem; font-weight: 600;">Pos ({summary["pos_count"]})</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                s2.markdown(
                    f'<div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 6px; text-align: center;">'
                    f'<div style="color: #18181B; font-weight: 800; font-size: 0.88rem;">{summary["neu_pct"]:.1f}%</div>'
                    f'<div style="color: #000000; font-size: 0.70rem; font-weight: 600;">Neu ({summary["neu_count"]})</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                s3.markdown(
                    f'<div style="background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 6px; padding: 6px; text-align: center;">'
                    f'<div style="color: #991B1B; font-weight: 800; font-size: 0.88rem;">{summary["neg_pct"]:.1f}%</div>'
                    f'<div style="color: #000000; font-size: 0.70rem; font-weight: 600;">Neg ({summary["neg_count"]})</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Minimal progress line
                st.markdown(
                    f'<div class="min-bar-wrapper" title="Pos: {summary["pos_pct"]:.1f}% | Neu: {summary["neu_pct"]:.1f}% | Neg: {summary["neg_pct"]:.1f}%">'
                    f'<div class="min-bar-pos" style="width: {summary["pos_pct"]}%;"></div>'
                    f'<div class="min-bar-neu" style="width: {summary["neu_pct"]}%;"></div>'
                    f'<div class="min-bar-neg" style="width: {summary["neg_pct"]}%;"></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Historical Test Metrics
                st.markdown(
                    f'<div style="background: #FAFAFA; border: 1px solid #E5E7EB; border-radius: 6px; padding: 8px 10px; margin-top: 6px;">'
                    f'<div style="font-size: 0.70rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #000000; margin-bottom: 4px;">Historical Benchmark</div>'
                    f'<div style="display: flex; justify-content: space-between; align-items: center;">'
                    f'<div><span style="font-size: 0.70rem; color: #4B5563;">Accuracy:</span><br><b style="font-size: 1.02rem; color: #000000;">{hist["accuracy"]}</b></div>'
                    f'<div><span style="font-size: 0.70rem; color: #4B5563;">{hist["f1_label"]}</span><br><b style="font-size: 1.02rem; color: #000000;">{hist["f1_val"]}</b></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)

    # Section 3: Sentiment Distribution Chart
    with st.expander("View Sentiment Distribution Chart", expanded=False):
        st.markdown('<div class="section-desc">Visual comparison of sentiment proportions across all four models.</div>', unsafe_allow_html=True)
        chart_data = []
        for model_lbl, sum_obj in [
            ("FinBERT", fb_summary),
            ("Linear SVM", svm_summary),
            ("Logistic Regression", lr_summary),
            ("BiLSTM", bl_summary)
        ]:
            for sent_cat, pct, cnt in [
                ("Positive", sum_obj["pos_pct"], sum_obj["pos_count"]),
                ("Neutral", sum_obj["neu_pct"], sum_obj["neu_count"]),
                ("Negative", sum_obj["neg_pct"], sum_obj["neg_count"])
            ]:
                chart_data.append({
                    "Model": model_lbl,
                    "Sentiment": sent_cat,
                    "Percentage": round(pct, 1),
                    "Count": cnt
                })

        df_chart = pd.DataFrame(chart_data)
        
        c_chart = alt.Chart(df_chart).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Model:N", title=None, axis=alt.Axis(labelAngle=0, labelFontWeight="bold", labelColor="#000000")),
            y=alt.Y("Percentage:Q", title="Percentage (%)", scale=alt.Scale(domain=[0, 100]), axis=alt.Axis(labelColor="#000000", titleColor="#000000")),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=["Positive", "Neutral", "Negative"],
                    range=["#16A34A", "#71717A", "#DC2626"]
                ),
                legend=alt.Legend(title=None, orient="top", labelColor="#000000")
            ),
            xOffset="Sentiment:N",
            tooltip=["Model", "Sentiment", "Percentage", "Count"]
        ).properties(height=260)

        st.altair_chart(c_chart, use_container_width=True)

    # Section 4: Unified News Feed Table
    st.markdown('<div class="section-title">Live News Feed &amp; Prediction Audit</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Inspect live headlines alongside individual predictions from FinBERT, Linear SVM, Logistic Regression, and BiLSTM to detect model discrepancies.</div>',
        unsafe_allow_html=True
    )

    # Controls
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        feed_filter = st.selectbox(
            "Filter headlines:",
            options=["All Headlines", "Discrepancies Only (Model Disagreements)", "Consensus Only"],
            index=0
        )
    with f_col2:
        search_term = st.text_input("Search headlines or publishers:", placeholder="Type to filter by keyword...")

    # Build rows
    feed_rows = []
    for idx, (art, inf) in enumerate(zip(articles, inferences)):
        is_cons = inf["is_consensus"]

        if "Discrepancies" in feed_filter and is_cons:
            continue
        if "Consensus" in feed_filter and not is_cons:
            continue
        if search_term:
            st_clean = search_term.lower()
            if st_clean not in art["headline"].lower() and st_clean not in art["publisher"].lower():
                continue

        feed_rows.append({
            "#": idx + 1,
            "Time": art["time"],
            "Publisher": art["publisher"],
            "Headline": art["headline"],
            "FinBERT": inf["finbert"],
            "SVM": inf["svm"],
            "Logistic Reg": inf["lr"],
            "BiLSTM": inf["bilstm"],
            "Agreement": "Consensus" if is_cons else "Discrepancy",
            "Article Link": art["url"]
        })

    if not feed_rows:
        st.info("No headlines match the chosen filter.")
    else:
        df_feed = pd.DataFrame(feed_rows)

        def style_cells(val):
            if val == "Positive":
                return "background-color: #ECFDF5; color: #065F46; font-weight: 700;"
            elif val == "Negative":
                return "background-color: #FEF2F2; color: #991B1B; font-weight: 700;"
            elif val == "Neutral":
                return "background-color: #F4F4F5; color: #18181B;"
            elif val == "Consensus":
                return "background-color: #ECFDF5; color: #065F46; font-weight: 700;"
            elif val == "Discrepancy":
                return "background-color: #FFFBEB; color: #92400E; font-weight: 700;"
            return "color: #000000;"

        styled_feed = df_feed.style.map(
            style_cells,
            subset=["FinBERT", "SVM", "Logistic Reg", "BiLSTM", "Agreement"]
        )

        st.dataframe(
            styled_feed,
            column_config={
                "#": st.column_config.NumberColumn("#", width="small"),
                "Time": st.column_config.TextColumn("Time", width="medium"),
                "Publisher": st.column_config.TextColumn("Publisher", width="medium"),
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "FinBERT": st.column_config.TextColumn("FinBERT", width="small"),
                "SVM": st.column_config.TextColumn("SVM", width="small"),
                "Logistic Reg": st.column_config.TextColumn("Logistic Reg", width="small"),
                "BiLSTM": st.column_config.TextColumn("BiLSTM", width="small"),
                "Agreement": st.column_config.TextColumn("Agreement", width="small"),
                "Article Link": st.column_config.LinkColumn("Link", width="small")
            },
            hide_index=True,
            use_container_width=True,
            height=460
        )

        st.download_button(
            label="Download CSV",
            data=df_feed.to_csv(index=False).encode("utf-8"),
            file_name=f"{ticker}_sentiment_comparison.csv",
            mime="text/csv"
        )

    st.markdown("<hr style='margin: 1.8rem 0; border: none; border-top: 1px solid #E5E7EB;'>", unsafe_allow_html=True)

    # Section 5: Custom Headline Testing Sandbox
    st.markdown('<div class="section-title">Test Custom Headline</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Type any financial headline or statement to observe live predictions across all four models simultaneously.</div>',
        unsafe_allow_html=True
    )

    sample_cases = [
        "Apple reports record quarterly revenue and beats Wall Street guidance.",
        "Profit plunged 24% as severe supply chain bottlenecks triggered steep losses.",
        "Revenue beat analyst estimates despite significant margin compression from inflation.",
        "Federal Reserve leaves benchmark interest rates unchanged at 5.25%-5.50%."
    ]
    
    selected_sample = st.selectbox(
        "Sample headlines:",
        options=["(Type custom headline)"] + sample_cases,
        index=0
    )

    default_custom = selected_sample if selected_sample != "(Type custom headline)" else "Apple surges to record high as quarterly earnings beat Wall Street estimates."
    custom_input = st.text_input("Headline text:", value=default_custom)

    if st.button("Classify Headline", type="primary"):
        if custom_input.strip():
            test_res = run_multi_model_inference([custom_input.strip()], bundle)[0]
            
            t1, t2, t3, t4 = st.columns(4)
            with t1:
                with st.container(border=True):
                    st.markdown("**FinBERT**")
                    st.markdown(get_sentiment_pill(test_res["finbert"]), unsafe_allow_html=True)
                    st.caption(f"Confidence: {test_res['finbert_confidence']*100:.1f}%")
            with t2:
                with st.container(border=True):
                    st.markdown("**Linear SVM**")
                    st.markdown(get_sentiment_pill(test_res["svm"]), unsafe_allow_html=True)
                    st.caption("TF-IDF + LinearSVC")
            with t3:
                with st.container(border=True):
                    st.markdown("**Logistic Regression**")
                    st.markdown(get_sentiment_pill(test_res["lr"]), unsafe_allow_html=True)
                    st.caption("TF-IDF + LogisticRegression")
            with t4:
                with st.container(border=True):
                    st.markdown("**BiLSTM**")
                    st.markdown(get_sentiment_pill(test_res["bilstm"]), unsafe_allow_html=True)
                    st.caption("Bidirectional LSTM (Weighted)")

if __name__ == "__main__":
    main()
