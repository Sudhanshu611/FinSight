# FinSight: Real-Time Sentiment & Model Comparison
## Comprehensive System & Section Documentation Guide (4-Model Evaluation Suite)

---

## 1. Introduction & Executive Summary

**FinSight** is an institutional-grade financial natural language processing (NLP) dashboard built with Streamlit. Its primary mission is to evaluate, benchmark, and contrast four distinct machine learning and deep learning model architectures on real-time live market news fetched directly from Yahoo Finance (`yfinance`):

1. **FinBERT**: A deep, bidirectional pre-trained contextual Transformer specifically trained on financial corpora (Prosus FinBERT weights).
2. **Linear SVM**: A maximum-margin Linear Support Vector Classifier trained on 5,000 TF-IDF features with SMOTE class balancing.
3. **Logistic Regression**: An L2-regularized probabilistic linear classifier trained on 5,000 TF-IDF features with SMOTE class balancing.
4. **BiLSTM**: A Bidirectional Long Short-Term Memory recurrent neural network with class-weighted training for capturing sequential forward and backward textual dependencies (`bilstm_weighted_model.keras`).
5. **TF-IDF Vectorizer**: A statistical feature extractor maintaining a 5,000-term financial n-gram vocabulary.

This document details what each section and component of the FinSight application does, what the metrics and labels say, and how to interpret the outputs for algorithmic trading and market analysis.

---

## 2. Model Storage & Resolution Architecture

### What it does:
FinSight is architected for cloud-native deployment (Streamlit Cloud, Hugging Face, Google Colab / Google Drive) and local standalone execution:
- **Primary Cloud Path**: `/content/drive/MyDrive/FinSight/models`
- **Local Fallback Paths**: `./models` or workspace subdirectories.

### Asset Verification:
The software verifies the existence of required assets before launching:
- `bilstm_weighted_model.keras` (Keras 3 Bidirectional LSTM model)
- `svm_smote_model.joblib` (Trained LinearSVC estimator)
- `logistic_regression_smote_model.joblib` (Trained Logistic Regression estimator)
- `tfidf_vectorizer.joblib` (Fitted TF-IDF Vectorizer)
- `finbert_pretrained/` (Local weights or automatic cloud fallback to `ProsusAI/finbert` on Hugging Face Hub)

---

## 3. Section-by-Section Software Guide

### Section 3.1: Application Header
- **Title**: `FinSight: Real-Time Sentiment & Model Comparison`
- **Subtitle**: *"A minimalist evaluation interface comparing FinBERT, Linear SVM, Logistic Regression, and BiLSTM on real-time market headlines."*
- **What it does**: Establishes the purpose of the platform and confirms the 4-model NLP evaluation scope.

---

### Section 3.2: Sidebar Controls (`Stock Selection & Configuration`)
Located on the left pane of the dashboard:
1. **Popular Tickers Quick-Select**:
   - **Buttons**: `AAPL`, `NVDA`, `MSFT`, `TSLA`, `GOOGL`, `AMZN`.
   - **What it does**: One-click selection for high-liquidity megacap stocks that immediately populates the ticker input and triggers live news ingestion.
2. **Ticker Symbol Input**:
   - **Label**: `Ticker Symbol:`
   - **Default Value**: `AAPL` (Apple Inc.).
   - **What it does**: Accepts any standard Yahoo Finance stock ticker symbol (e.g., `MSFT`, `NVDA`, `TSLA`, `GOOGL`).
3. **Fetch Live News Button**:
   - **Label**: `Fetch Live News` (Black high-contrast primary button with white text).
   - **What it does**: Triggers `yfinance` to query real-time news articles, clean timestamps, extract publishers, run parallel batch inferences across all four models, and refresh the dashboard.
4. **Model Storage Settings (Expander)**:
   - **What it does**: Displays the active path and shows a diagnostic status indicator confirming that model files have been successfully located.

---

### Section 3.3: Target Asset Summary Bar
Located immediately beneath the main header:
- **Target Asset**: Shows the currently evaluated ticker symbol (e.g., `$AAPL`).
- **Headlines Count**: Total number of news articles returned by Yahoo Finance and analyzed in the active session.
- **Full 4-Model Consensus**: The exact count and percentage of articles where **all four models produced the identical sentiment prediction**.
- **Discrepancies**: Total count of articles where at least one model disagreed with the others.

---

### Section 3.4: Model Sentiment Comparison (Four Side-by-Side Columns)
The central comparative module of the software, split into four side-by-side columns:

#### Column 1: FinBERT
- **Architecture**: Pre-trained Contextual Financial Transformer (Prosus FinBERT).
- **Dominant Sentiment Pill**: `● POSITIVE`, `● NEUTRAL`, or `● NEGATIVE`.
- **Live Sentiment Breakdown**: Exact percentage and count for Positive, Neutral, and Negative predictions.
- **Visual Distribution Bar**: Multi-segment horizontal bar (Green = Positive, Slate = Neutral, Red = Negative).
- **Historical Test Performance**:
  - **Accuracy**: `78.97%` (79%)
  - **Negative F1-Score**: `0.64` (Weighted F1: `0.79`)
- **Interpretation**: Serves as the deep contextual benchmark. Uses bidirectional self-attention to process syntax, negations, and complex financial nuance.

#### Column 2: Linear SVM
- **Architecture**: Linear Support Vector Machine (`LinearSVC`) trained on 5,000 TF-IDF features with SMOTE class balancing.
- **Historical Test Performance**:
  - **Accuracy**: `73.99%` (74%)
  - **Negative F1-Score**: `0.47` (Weighted F1: `0.74`)
- **Interpretation**: Identifies maximum-margin linear hyperplanes in sparse TF-IDF space. Provides ultra-fast inference suitable for high-frequency feeds.

#### Column 3: Logistic Regression
- **Architecture**: L2-regularized Logistic Regression trained on 5,000 TF-IDF features with SMOTE class balancing.
- **Historical Test Performance**:
  - **Accuracy**: `73.62%` (73%)
  - **Negative F1-Score**: `0.52` (Weighted F1: `0.74`)
- **Interpretation**: Calculates smooth probabilistic log-odds. Balances precision and recall well for statistical linear models while remaining computationally lightweight.

#### Column 4: BiLSTM
- **Architecture**: Bidirectional Long Short-Term Memory recurrent neural network (`bilstm_weighted_model.keras`).
- **Historical Test Performance**:
  - **Accuracy**: `72.00%` (72%)
  - **Weighted F1-Score**: `0.72`
- **Interpretation**: Processes text sequentially in both forward and backward directions, capturing word order and short-term dependencies beyond standard bag-of-words models.

---

### Section 3.5: Sentiment Distribution Chart (Expander)
- **Title**: `View Sentiment Distribution Chart`
- **What it does**: Renders an interactive Altair grouped bar chart plotting Sentiment Share (%) on the Y-axis across all four models on the X-axis.
- **What it says**:
  - Direct visual comparison of model bias and skew across categories.
  - Interactive tooltips display exact percentages and counts upon hovering.

---

### Section 3.6: Live News Feed & Prediction Audit Table
- **Title**: `Live News Feed & Prediction Audit`
- **What it does**: Displays every ingested article in an interactive, searchable data table.
- **Table Columns**:
  - `#`: Sequence number.
  - `Time`: Publication timestamp converted to standardized UTC format (`YYYY-MM-DD HH:MM UTC`).
  - `Publisher`: Originating financial media outlet.
  - `Headline`: Exact news headline evaluated.
  - `FinBERT`: FinBERT classification (`Positive`, `Neutral`, `Negative`).
  - `SVM`: Linear SVM classification.
  - `Logistic Reg`: Logistic Regression classification.
  - `BiLSTM`: BiLSTM classification.
  - `Agreement`: `Consensus` (green badge) or `Discrepancy` (amber badge).
  - `Article Link`: Direct clickable hyperlink to the full original news story.
- **Controls**:
  - Filter dropdown (`All Headlines`, `Discrepancies Only`, `Consensus Only`).
  - Keyword search across headlines and publishers.
  - One-click CSV export (`Download CSV`).

---

### Section 3.7: Test Custom Headline (Interactive Sandbox)
- **Title**: `Test Custom Headline`
- **What it does**: An ad-hoc testing sandbox where users can test hypothetical market statements or corporate press releases.
- **Four Side-by-Side Result Cards**:
  - **FinBERT**: Output pill + confidence score.
  - **Linear SVM**: Output pill + feature tag.
  - **Logistic Regression**: Output pill + feature tag.
  - **BiLSTM**: Output pill + feature tag.

---

## 4. Summary Model Comparison Matrix

| Metric / Feature | FinBERT | Linear SVM | Logistic Regression | BiLSTM |
| :--- | :--- | :--- | :--- | :--- |
| **Model Family** | Deep Transformer | Max-Margin Hyperplane | Probabilistic Log-Odds | Recurrent Neural Network |
| **Feature Space** | WordPiece Attention | 5,000 TF-IDF n-grams | 5,000 TF-IDF n-grams | Sequential Word Embeddings (128-d) |
| **Class Balancing** | Cross-Entropy Loss | SMOTE Oversampling | SMOTE Oversampling | Class-Weighted Loss |
| **Historical Test Accuracy** | **78.97%** (79%) | **73.99%** (74%) | **73.62%** (73%) | **72.00%** (72%) |
| **Historical F1-Score** | **0.64** (Neg) / 0.79 (Wtd) | **0.47** (Neg) / 0.74 (Wtd) | **0.52** (Neg) / 0.74 (Wtd) | **0.72** (Weighted) |
| **Inference Latency** | Moderate (~15ms on CPU) | Ultra-Fast (<1ms) | Ultra-Fast (<1ms) | Fast (~3ms) |
| **Best Used For** | Complex context, negation, earnings nuance | High-frequency streaming | Calibrated probabilities | Sequential phrasing & word order |
