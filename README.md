# FinSight: Real-Time Sentiment & Model Comparison 📈

**FinSight** is an interactive financial NLP dashboard built with Streamlit that benchmarks and compares three sentiment classification models on live financial news:
1. **FinBERT** (Pre-trained financial domain transformer loaded from `finbert_pretrained`)
2. **Linear SVM** (Trained with SMOTE class balancing & TF-IDF, loaded from `svm_smote_model.joblib`)
3. **Logistic Regression** (Trained with SMOTE class balancing & TF-IDF, loaded from `logistic_regression_smote_model.joblib`)
4. **TF-IDF Vectorizer** (5,000 vocabulary feature extractor loaded from `tfidf_vectorizer.joblib`)

---

## 🚀 Features

- **Sidebar Input**: Enter any stock ticker symbol (default: `AAPL`) or select from popular tickers (`NVDA`, `TSLA`, `MSFT`, `GOOGL`, `AMZN`) with a one-click fetch button.
- **Real-Time News Fetching**: Pulls live news using `yfinance`, extracting headlines, publication timestamps, publishers, and article links.
- **Multi-Model Sentiment Inference**:
  - Predicts sentiment (`Positive`, `Negative`, `Neutral`) across all three models on the exact same live headlines.
- **Interactive Comparison UI**:
  - Three-column side-by-side model display.
  - Aggregated live sentiment distribution (% Positive, % Negative, % Neutral) with clear visual color indicators (green, red, gray/blue) and interactive progress bars.
  - Static historical evaluation metrics side-by-side:
    - **FinBERT**: Test Accuracy **78.97%**, Negative F1-Score **0.64**
    - **Linear SVM**: Test Accuracy **73.99%**, Negative F1-Score **0.47**
    - **Logistic Regression**: Test Accuracy **73.62**, Negative F1-Score **0.52**
  - Cross-model consensus metrics and interactive sentiment share comparison chart.
- **Unified News Feed Table**:
  - Interactive table displaying: `Time`, `Publisher`, `Headline`, `FinBERT Sentiment`, `SVM Sentiment`, and `Logistic Regression Sentiment`.
  - Discrepancy detector and filter to immediately isolate model disagreements.
  - Keyword search and one-click CSV export.
- **Interactive Sandbox**:
  - Test custom financial phrases and inspect live predictions across all three models simultaneously.

---

## 📁 Model Directory Structure

The models are expected at `/content/drive/MyDrive/FinSight/models` (when running on Google Drive / Colab) or `./models` (when running locally):

```text
models/
├── finbert_pretrained/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── svm_smote_model.joblib
├── logistic_regression_smote_model.joblib
└── tfidf_vectorizer.joblib
```

---

## 💻 Running Locally

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```

---

## ☁️ Running on Google Colab

1. Mount Google Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Verify model files are at `/content/drive/MyDrive/FinSight/models`:
   ```bash
   !ls -la /content/drive/MyDrive/FinSight/models
   ```

3. Install requirements and run Streamlit with localtunnel or ngrok:
   ```bash
   !pip install -r requirements.txt
   !streamlit run app.py & npx localtunnel --port 8501
   ```

