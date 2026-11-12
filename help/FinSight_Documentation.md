# FinSight: Real-Time Sentiment & Model Comparison
## Comprehensive System & Section Documentation Guide

---

## 1. Introduction & Executive Summary

**FinSight** is a specialized financial natural language processing (NLP) dashboard built with Streamlit. Its primary mission is to evaluate, benchmark, and contrast three distinct machine learning model architectures on real-time live market news fetched directly from Yahoo Finance (`yfinance`):

1. **FinBERT**: A deep, bidirectional pre-trained contextual Transformer specifically trained on financial corpora (Prosus FinBERT weights).
2. **Linear SVM**: A maximum-margin Linear Support Vector Classifier trained on 5,000 TF-IDF features with SMOTE class balancing.
3. **Logistic Regression**: An L2-regularized probabilistic linear classifier trained on 5,000 TF-IDF features with SMOTE class balancing.
4. **TF-IDF Vectorizer**: A statistical feature extractor maintaining a 5,000-term financial n-gram vocabulary.

This document details what each section and component of the FinSight application does, what the metrics and labels say, and how to interpret the outputs for algorithmic trading and market analysis.

---

## 2. Model Storage & Resolution (`/help` Architecture)

### What it does:
FinSight is architected for both cloud-native (Google Colab / Google Drive) and local standalone deployment:
- **Primary Cloud Path**: `/content/drive/MyDrive/FinSight/models`
- **Local Fallback Paths**: `./models` or workspace subdirectories.

### Asset Verification:
The software verifies the existence of four required assets before launching:
- `finbert_pretrained/` (Directory with `config.json`, `model.safetensors`, and tokenizer assets)
- `svm_smote_model.joblib` (Trained LinearSVC estimator)
- `logistic_regression_smote_model.joblib` (Trained Logistic Regression estimator)
- `tfidf_vectorizer.joblib` (Fitted TF-IDF Vectorizer)

---

## 3. Section-by-Section Software Guide

### Section 3.1: Application Header
- **Title**: `FinSight: Real-Time Sentiment & Model Comparison`
- **Subtitle**: *"A minimalist evaluation interface comparing FinBERT, Linear SVM, and Logistic Regression on real-time market headlines."*
- **What it does**: Establishes the purpose of the platform and confirms the tri-model NLP evaluation scope.

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
   - **Label**: `Fetch Live News` (Black high-contrast primary button).
   - **What it does**: Triggers `yfinance` to query real-time news articles, clean timestamps, extract publishers, run parallel batch inferences across all three models, and refresh the dashboard.
4. **Model Storage Settings (Expander)**:
   - **What it does**: Displays the target Google Drive path (`/content/drive/MyDrive/FinSight/models`) and shows a diagnostic status indicator confirming that model files have been successfully located and verified.

---

### Section 3.3: Target Asset Summary Bar
Located immediately beneath the main header:
- **Target Asset**: Shows the currently evaluated ticker symbol (e.g., `$AAPL`).
- **Headlines Count**: Total number of news articles returned by Yahoo Finance and analyzed in the active session.
- **Full Consensus**: The exact count and percentage of articles where **all three models produced the identical sentiment prediction**.
- **Discrepancies**: Total count of articles where at least one model disagreed with the others.
- **Interpretation**: High consensus (>70%) indicates strong, unambiguous market news signals. High discrepancies indicate nuanced, complex phrasing (e.g., revenue beats accompanied by margin cuts).

---

### Section 3.4: Model Sentiment Comparison (Three Side-by-Side Columns)
The central comparative module of the software, split into three side-by-side columns:

#### Column 1: FinBERT
- **Architecture**: Pre-trained Contextual Financial Transformer (Prosus FinBERT).
- **Dominant Sentiment Pill**: Shows whether the majority of live headlines are classified as `● POSITIVE`, `● NEUTRAL`, or `● NEGATIVE`.
- **Live Sentiment Breakdown**:
  - `Pos`: Percentage and count of headlines classified as Positive (e.g., `20.0% (2)`).
  - `Neu`: Percentage and count of headlines classified as Neutral (e.g., `70.0% (7)`).
  - `Neg`: Percentage and count of headlines classified as Negative (e.g., `10.0% (1)`).
- **Visual Distribution Bar**: Multi-segment horizontal bar (Green = Positive, Slate = Neutral, Red = Negative) providing an instant visual ratio.
- **Historical Test Performance**:
  - **Test Accuracy**: `78.97%`
  - **Negative F1-Score**: `0.64`
- **What it says / Interpretation**: FinBERT serves as the contextual benchmark. It uses multi-head self-attention to process bidirectional syntax, making it far superior at recognizing negations ("not bad", "failed to fall") and financial nuances ("beat estimates despite headwinds").

#### Column 2: Linear SVM
- **Architecture**: Linear Support Vector Machine (`LinearSVC`) trained on 5,000 TF-IDF features with SMOTE class balancing.
- **Dominant Sentiment Pill**: Dominant predicted sentiment on the live feed.
- **Live Sentiment Breakdown**: Real-time percentages and counts for Positive, Neutral, and Negative.
- **Visual Distribution Bar**: Color-coded distribution proportion.
- **Historical Test Performance**:
  - **Test Accuracy**: `73.99%`
  - **Negative F1-Score**: `0.47`
- **What it says / Interpretation**: Linear SVM identifies maximum-margin linear hyperplanes in high-dimensional word space. It provides ultra-fast inference suitable for high-frequency algorithmic feeds, though its bag-of-words representation can occasionally miss syntactic negations.

#### Column 3: Logistic Regression
- **Architecture**: L2-regularized Logistic Regression trained on 5,000 TF-IDF features with SMOTE class balancing.
- **Dominant Sentiment Pill**: Dominant predicted sentiment on the live feed.
- **Live Sentiment Breakdown**: Real-time percentages and counts for Positive, Neutral, and Negative.
- **Visual Distribution Bar**: Color-coded distribution proportion.
- **Historical Test Performance**:
  - **Test Accuracy**: `73.62%`
  - **Negative F1-Score**: `0.52`
- **What it says / Interpretation**: Logistic Regression calculates smooth probabilistic log-odds. With a historical negative F1-score of 0.52, it balances precision and recall well for statistical linear models, while remaining computationally lightweight.

#### Why Negative Class F1-Score is Highlighted:
In financial markets, downside risk creates asymmetric drawdown. A false negative (missing a critical negative earnings warning or regulatory penalty) is far more damaging to a portfolio than misidentifying a neutral headline. Thus, Negative Class F1-Score is a primary model selection criterion.

---

### Section 3.5: Sentiment Distribution Chart (Expander)
- **Title**: `View Sentiment Distribution Chart`
- **What it does**: Renders an interactive Altair grouped bar chart plotting Sentiment Share (%) on the Y-axis across all three models on the X-axis.
- **What it says**:
  - Direct visual comparison of model bias.
  - Allows analysts to quickly see whether Linear SVM or Logistic Regression are over-predicting Neutral compared to FinBERT, or if all three models are aligned on positive/negative momentum.
  - Interactive tooltips display exact percentages and counts upon hovering.

---

### Section 3.6: Live News Feed & Prediction Audit Table
- **Title**: `Live News Feed & Prediction Audit`
- **Subtitle**: *"Inspect live headlines alongside individual predictions from FinBERT, Linear SVM, and Logistic Regression to detect model discrepancies."*
- **What it does**: Displays every ingested article in an interactive, searchable data table.
- **Interactive Controls**:
  1. **Filter Dropdown**:
     - `All Headlines`: Shows complete ingested feed.
     - `Discrepancies Only (Model Disagreements)`: Isolates only headlines where at least one model differed in prediction. This is the core auditing tool for analysts.
     - `Consensus Only`: Shows only headlines where FinBERT, SVM, and Logistic Regression all agreed 100%.
  2. **Search Input**: Live keyword filter searching across headlines and publisher names (e.g., "earnings", "guidance", "probe", "SEC").
  3. **Download CSV Button**: Exports the entire audited news feed with timestamps, publishers, headlines, model predictions, and agreement flags into a clean CSV file.
- **Table Columns**:
  - `#`: Sequence number.
  - `Time`: Publication timestamp converted to standardized UTC format (`YYYY-MM-DD HH:MM UTC`).
  - `Publisher`: Originating financial media outlet (e.g., Bloomberg, Reuters, Yahoo Finance Video).
  - `Headline`: Exact news headline evaluated.
  - `FinBERT Sentiment`: FinBERT classification (`Positive`, `Neutral`, `Negative`).
  - `SVM Sentiment`: Linear SVM classification (`Positive`, `Neutral`, `Negative`).
  - `Logistic Regression Sentiment`: Logistic Regression classification (`Positive`, `Neutral`, `Negative`).
  - `Agreement`: `Consensus` (green badge) or `Discrepancy` (amber badge).
  - `Article Link`: Direct clickable hyperlink to the full original news story.

---

### Section 3.7: Test Custom Headline (Interactive Sandbox)
- **Title**: `Test Custom Headline`
- **Subtitle**: *"Type any financial headline or statement to observe live predictions across all three models."*
- **What it does**: An ad-hoc stress-testing sandbox where users can test hypothetical market statements, rumors, or corporate press release text.
- **Components**:
  1. **Preset Scenarios Dropdown**: Includes curated test cases demonstrating tricky financial phrasing:
     - Record earnings beat: *"Apple reports record quarterly revenue and beats Wall Street guidance."*
     - Severe loss / disruption: *"Profit plunged 24% as severe supply chain bottlenecks triggered steep losses."*
     - Mixed signal / guidance miss: *"Revenue beat analyst estimates despite significant margin compression from inflation."*
     - Central bank hold: *"Federal Reserve leaves benchmark interest rates unchanged at 5.25%-5.50%."*
  2. **Custom Text Input**: Allows freeform typing of any financial sentence.
  3. **Classify Headline Button**: Executes real-time inference across all three models.
  4. **Side-by-Side Result Cards**:
     - **FinBERT**: Output pill + confidence score (e.g., `Confidence: 94.2%`).
     - **Linear SVM**: Output pill + feature tag (`TF-IDF + LinearSVC`).
     - **Logistic Regression**: Output pill + feature tag (`TF-IDF + LogisticRegression`).

---

## 4. Summary Interpretation Matrix

| Metric / Feature | FinBERT | Linear SVM | Logistic Regression |
| :--- | :--- | :--- | :--- |
| **Model Family** | Deep Transformer | Max-Margin Hyperplane | Probabilistic Log-Odds |
| **Feature Space** | WordPiece Tokens & Self-Attention | 5,000 TF-IDF n-grams | 5,000 TF-IDF n-grams |
| **Class Balancing** | Cross-Entropy Loss | SMOTE Oversampling | SMOTE Oversampling |
| **Historical Test Accuracy** | **78.97%** | **73.99%** | **73.62%** |
| **Historical Negative F1** | **0.64** | **0.47** | **0.52** |
| **Inference Latency** | Moderate (~15ms on CPU) | Ultra-Fast (<1ms) | Ultra-Fast (<1ms) |
| **Best Used For** | Nuanced headlines, negation, SEC filings | High-frequency headline streams | Calibrated risk scoring |

