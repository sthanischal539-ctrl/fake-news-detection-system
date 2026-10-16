# Fake News Detection System (NLP)
An end-to-end Natural Language Processing system that classifies news articles as Fake or Real based on their full body text. The system implements and compares three classical machine learning models: Logistic Regression, Multinomial Naive Bayes, and Linear SVM. It includes text preprocessing, TF-IDF feature extraction, model evaluation on a held-out test set, word-level explanations using LIME, and an interactive Streamlit dashboard. The project also includes a DistilBERT component for future exploration of transformer-based classification.

> ⚠️ **CRITICAL ETHICAL DISCLAIMER — SIGNAL, NOT PROOF**  
> Predictions produced by this system represent statistical, probabilistic patterns in writing style and vocabulary. They do **not** verify factual truth or real-world events. Misinformation models can be confidently wrong when confronted with novel events or authoritative styles. Always verify news against trusted, primary journalistic sources.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Problem Statement & Objectives](#problem-statement--objectives)
3. [Technologies Used](#technologies-used)
4. [Project Directory Structure](#project-directory-structure)
5. [System Architecture](#system-architecture)
6. [Datasets & Academic Caveats](#datasets--academic-caveats)
7. [Text Preprocessing](#text-preprocessing)
8. [Models & Machine Learning Pipeline](#models--machine-learning-pipeline)
   - [Classical Models](#classical-models)
   - [DistilBERT Transformer](#distilbert-transformer)
9. [Evaluation Methodology & Actual Results](#evaluation-methodology--actual-results)
10. [Explainability (LIME & Global Coefficients)](#explainability)
11. [Streamlit Web Interface & Dashboard](#streamlit-web-interface--dashboard)
12. [Installation & Environment Setup](#installation--environment-setup)
13. [Step-by-Step Reproduction Commands](#step-by-step-reproduction-commands)
14. [Running Automated Tests](#running-automated-tests)
15. [Hardware & Cache Configuration (Windows C: vs. D: Drive)](#hardware--cache-configuration)
16. [Known Limitations & Future Improvements](#known-limitations--future-improvements)

---

## Project Overview
Automated fake news detection is a core NLP challenge. Rather than evaluating short headlines out of context, this system accepts complete article texts. It compares four trained classifiers:
- **Logistic Regression** (TF-IDF features)
- **Multinomial Naive Bayes** (TF-IDF features)
- **Linear Support Vector Machine (LinearSVC)** with Platt/Isotonic probability calibration
- **DistilBERT** (`distilbert-base-uncased`) fine-tuned sequence classifier

The repository includes a unified prediction pipeline (`src.predict`), an evaluation pipeline (`src.evaluate`), a test suite (`tests/`), and a multi-tab **Streamlit** web app (`src/app.py`).

---

## Problem Statement & Objectives

### Problem Statement
The volume of digital news and online content makes manual fact-checking of every published piece infeasible. Misleading stories often spread rapidly before fact-checkers can intervene. An automated NLP classifier provides a triage mechanism by detecting linguistic patterns, sensationalist markers, and stylistic abnormalities.

### Objectives
1. **Full-Article Classification:** Analyze entire paragraphs and bodies of text, avoiding headline-only clickbait shortcuts.
2. **Multi-Model Comparison:** Train and benchmark classical linear models against a transformer architecture.
3. **Rigorous Evaluation:** Prevent data leakage via deduplication, perform stratified splits (seed 42), and report Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrices on a held-out test set.
4. **Transparent Explainability:** Expose local token importance using LIME and global vocabulary weights from model coefficients.
5. **Interactive Demonstration:** Deliver a clear, user-friendly Streamlit web application.

---

## Technologies Used
- **Programming Language:** Python 3.10 – 3.12 (tested on Python 3.12)
- **Core ML & Data:** `scikit-learn`, `numpy`, `pandas`, `scipy`, `joblib`
- **NLP & Tokenization:** `nltk` (punkt, stopwords, wordnet), `transformers` (Hugging Face)
- **Deep Learning:** `torch` (PyTorch CPU / GPU), `accelerate`
- **Explainability:** `lime` (Local Interpretable Model-agnostic Explanations)
- **Web UI & Visualization:** `streamlit`, `matplotlib`
- **Testing:** `pytest`

---

## Project Directory Structure
```
nischal-nlp-project/
├── data/
│   ├── raw/                       # Raw datasets (demo_news.csv, True.csv, Fake.csv)
│   └── processed/                 # Generated splits (train.csv, val.csv, test.csv, dataset_info.json)
├── docs/                          # Detailed design and planning documentation
│   ├── architecture.md            # Pipeline architecture and data flow diagram
│   ├── dataset.md                 # Dataset specifications, comparison, and limitations
│   ├── implementation-plan.md     # Development roadmap and milestone checklist
│   └── problem-statement.md       # Problem background, motivation, and scope
├── models/                        # Saved model artefacts
│   ├── distilbert/                # Fine-tuned DistilBERT weights, tokenizer, config
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── hf_cache/                  # Hugging Face downloaded model cache
│   ├── logreg.joblib              # Saved Logistic Regression model
│   ├── naive_bayes.joblib         # Saved Multinomial Naive Bayes model
│   ├── svm.joblib                 # Saved Calibrated Linear SVM model
│   └── tfidf_vectorizer.joblib    # Fitted TF-IDF vectorizer
├── results/                       # Evaluation outputs
│   ├── confusion_distilbert.png
│   ├── confusion_linear_svm.png
│   ├── confusion_logistic_regression.png
│   ├── confusion_naive_bayes.png
│   └── metrics.json               # Full test set metrics, reports, and metadata
├── src/                           # Source code modules
│   ├── __init__.py
│   ├── app.py                     # Streamlit web application & dashboard
│   ├── config.py                  # Central configuration, paths, seeds, hyperparameters
│   ├── data_loader.py             # Data loading, validation, cleaning, stratified splits
│   ├── evaluate.py                # Test set evaluation, ROC-AUC, confusion matrices
│   ├── explain.py                 # LIME explainability & global coefficient extraction
│   ├── features.py                # TF-IDF vectorizer definition, fitting, and persistence
│   ├── make_demo_data.py          # Synthetic demonstration data generator
│   ├── predict.py                 # Unified prediction interface (classical + transformer)
│   ├── preprocessing.py           # Text cleaning, NLTK tokenization, lemmatization
│   └── train.py                   # Model training script (classical + optional transformer)
├── tests/                         # Automated pytest suite (19 test cases)
│   ├── __init__.py
│   ├── test_dashboard_and_transformer.py
│   ├── test_data_loader.py
│   ├── test_features.py
│   ├── test_models_and_eval.py
│   ├── test_prediction_pipeline.py
│   └── test_preprocessing.py
├── .gitignore                     # Git ignore rules for virtualenvs, caches, and artefacts
├── README.md                      # Comprehensive academic project documentation
├── requirements.txt               # Pinned project dependencies
└── start.py                       # Unified CLI runner and quick launcher
```

---

## System Architecture

```
Raw News Data (ISOT / Demo CSV)
              ↓
Data Cleaning & Deduplication (prevents train/test leakage)
              ↓
Stratified Split: 70% Train / 15% Validation / 15% Test (SEED=42)
              ↓
Text Preprocessing (dateline strip, URL/HTML clean, NLTK lemmatize)
              ↓
Feature Extraction:
  ├── TF-IDF (1-2 ngrams, max 20,000 features) → Classical Models
  └── Subword Tokenization (DistilBERT tokenizer) → Transformer
              ↓
Model Training:
  ├── Logistic Regression (C=1.0)
  ├── Multinomial Naive Bayes (alpha=1.0)
  ├── Linear SVM (CalibratedClassifierCV)
  └── DistilBERT Sequence Classifier (Hugging Face Trainer)
              ↓
Evaluation on Held-Out Test Set (results/metrics.json + confusion figures)
              ↓
Prediction & Interactive Web Dashboard (Streamlit + LIME)
```

---

## Datasets & Academic Caveats

### 1. Bundled Synthetic Demonstration Dataset
- **File:** `data/raw/demo_news.csv`
- **Total Articles:** 1,000 (~999 unique articles after deduplication)
- **Balance:** 50% Real / 50% Fake
- **Design:** Created algorithmically to demonstrate pipeline mechanics out-of-the-box without requiring gigabytes of downloads.
- **Why Metrics Reach 100%:** The generator deliberately pairs Real articles with formal reporting openers ("According to officials...") and Fake articles with sensationalist clickbait openers ("SHOCKING BOMBSHELL..."). TF-IDF models and DistilBERT learn this stylistic boundary with 100% precision.
- **Academic Warning:** High scores on this demo dataset verify that the software and ML pipeline function correctly; they do **not** prove general fake news detection capability on real-world news.

### 2. ISOT Fake News Dataset (Recommended for Academic Submission)
- **Source:** University of Victoria ISOT Research Lab / Kaggle.
- **Files:** `data/raw/True.csv` (Reuters stories) and `data/raw/Fake.csv` (unverified articles).
- **Size:** ~44,900 articles (~21,400 real, ~23,500 fake).
- **How to Use:**
  1. Download `True.csv` and `Fake.csv`.
  2. Copy both files into `data/raw/`.
  3. Run: `python start.py --train` (or `python -m src.train --force-reprocess`).
  `src/data_loader.py` automatically detects ISOT, strips source datelines, combines title and body, performs stratified splitting, and trains on real data.

---

## Text Preprocessing
Implemented in `src/preprocessing.py`:
1. **Dateline Stripping:** Removes Reuters wire source prefixes (e.g., `WASHINGTON (Reuters) -`) via regex to prevent models from learning trivial source metadata.
2. **Noise Cleaning:** Strips HTML tags (`<... >`), URLs (`http://...`), and email addresses.
3. **Punctuation & Case Normalization:** Converts text to lowercase and strips non-alphabetic noise.
4. **Tokenization:** Uses NLTK's `word_tokenize`.
5. **Stopword Filtering:** Removes standard English stopwords from `nltk.corpus.stopwords`.
6. **Lemmatization:** Applies WordNet lemmatization (`WordNetLemmatizer`) to reduce words to root forms.
7. **Transformer Light Cleaning (`light_clean`):** For DistilBERT, retains grammatical punctuation and case structure, only removing HTML, URLs, and datelines, allowing the transformer tokenizer to utilize natural language sentence structure.

---

## Models & Machine Learning Pipeline

### Classical Models
1. **Logistic Regression (`models/logreg.joblib`):**
   - Linear classifier with $L_2$ regularization ($C=1.0$, `max_iter=1000`).
   - Highly interpretable coefficients indicate global positive (Fake) and negative (Real) vocabulary.
2. **Multinomial Naive Bayes (`models/naive_bayes.joblib`):**
   - Probabilistic classifier with Laplace smoothing ($\alpha=1.0$).
   - Standard academic text classification baseline.
3. **Linear SVM with Calibration (`models/svm.joblib`):**
   - `LinearSVC` ($C=1.0$) calibrated using `CalibratedClassifierCV(cv=3)`.
   - Produces smooth, calibrated class probabilities for display in the dashboard.

### DistilBERT Transformer
- **Architecture:** `distilbert-base-uncased` with a sequence classification head (`num_labels=2`).
- **Configuration:** Binary classification, maximum sequence length 128 (CPU-friendly, configurable), batch size 16.
- **Training Framework:** Hugging Face `Trainer` API with PyTorch and `accelerate`.
- **Validation Evaluation:** Evaluates validation loss, accuracy, and F1 at epoch end.
- **Model Persistence:** Saved to `models/distilbert/` (`model.safetensors`, `config.json`, `tokenizer.json`).

---

## Evaluation Methodology & Actual Results

All models are evaluated on the identical held-out test split (150 articles from the demo split) using `src/evaluate.py`. No training data is included in the test set.

### Actual Results (Held-Out Test Set)

| Model | Accuracy | Precision (Fake) | Recall (Fake) | F1-Score (Fake) | ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Trained & Verified |
| **Naive Bayes** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Trained & Verified |
| **Linear SVM (Calibrated)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Trained & Verified |
| **DistilBERT** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | Fine-Tuned & Verified |

*Note: Results reflect evaluation on the synthetic demonstration split (`data/processed/test.csv`). Confusion matrices are saved in `results/`.*

---

## Explainability
Implemented in `src/explain.py`:
- **LIME (Local Interpretable Model-agnostic Explanations):** Explains individual predictions by perturbing input tokens and calculating the impact on prediction probabilities. Visualized in the web app as a horizontal bar chart of positive (Fake-biasing) vs. negative (Real-biasing) word weights.
- **Global Feature Weights:** Extracts top positive and negative coefficients from the trained Logistic Regression model, revealing the vocabulary most strongly correlated with each class.

---

## Streamlit Web Interface & Dashboard
Launch with `streamlit run src/app.py` or `python start.py --app`:
- **Predict Tab:**
  - Full article text input with quick sample buttons (Sample Real, Sample Fake).
  - Model selection dropdown supporting all trained models (Logistic Regression, Naive Bayes, Linear SVM, DistilBERT).
  - Clear label output (`REAL` in green, `FAKE` in red).
  - Prediction confidence meter and full probability breakdown (REAL % vs. FAKE %).
  - Interactive LIME feature weight chart.
  - Prominent probabilistic disclaimer.
- **Model Comparison Tab:**
  - Side-by-side performance table (Accuracy, Precision, Recall, F1, ROC-AUC).
  - Interactive metric comparison bar chart.
  - Confusion matrix plots for all models.
  - Expandable detailed classification reports.
  - Top learned global vocabulary indicators.
- **Dataset & Overview Tab:**
  - Active dataset source and record counts.
  - Dedicated academic note explaining synthetic dataset limitations and ISOT setup.
  - Architectural pipeline schematic.

---

## Installation & Environment Setup

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 (Python 3.12 recommended)
- 2 GB of free disk space (if running transformer)

### 2. Virtual Environment Setup
```bash
# Clone or navigate to the project directory
cd "nischal nlp project"

# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment:
# On Linux / macOS:
source .venv/bin/activate

# On Windows (Command Prompt):
# .venv\Scripts\activate.bat

# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### 3. Installing Dependencies
```bash
# Option A: Install classical dependencies only (lightweight, ~50 MB)
pip install numpy pandas scipy scikit-learn joblib nltk lime streamlit matplotlib pytest

# Option B: Install all dependencies including transformer (from requirements.txt)
# For CPU-optimized PyTorch:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

---

## Step-by-Step Reproduction Commands

The project provides two equivalent interfaces: the convenient `start.py` launcher, and standard `python -m src.<module>` commands.

### Using the Quick Launcher (`start.py`):
```bash
# 1. Train classical models (Logistic Regression, Naive Bayes, Linear SVM)
python start.py --train

# 2. (Optional) Fine-tune DistilBERT sequence classifier
python start.py --train-transformer

# 3. Evaluate all trained models on held-out test data
python start.py --eval

# 4. Run automated test suite
python start.py --test

# 5. Launch the Streamlit web app & dashboard
python start.py --app

# 6. Run prediction from the command line
python start.py --predict "According to officials, the committee met to review the fiscal report." --model "Logistic Regression"
```

### Using Standard Module Invocations:
```bash
# Train classical models:
python -m src.train

# Train classical + fine-tune DistilBERT:
python -m src.train --with-transformer

# Re-generate splits and retrain:
python -m src.train --force-reprocess

# Evaluate models and write results/metrics.json:
python -m src.evaluate

# Run standalone prediction CLI:
python -m src.predict --text "Your article text here..." --model "Naive Bayes"

# Run automated tests:
pytest tests/ -v

# Launch Streamlit web dashboard:
streamlit run src/app.py
```

---

## Hardware & Cache Configuration

### Custom Hugging Face Cache (Windows C: vs. D: Drive)
On machines where the primary drive (e.g. `C:`) has limited storage, you can redirect the Hugging Face model cache and download directory to another drive (e.g. `D:`):

```bash
# Linux / macOS:
export HF_HOME="/path/to/larger_drive/hf_cache"

# Windows (Command Prompt):
set HF_HOME=D:\hf_cache

# Windows (PowerShell):
$env:HF_HOME = "D:\hf_cache"
```
The pipeline automatically reads `HF_HOME` or `TRANSFORMERS_CACHE` and defaults to `models/hf_cache/` inside the project folder if unspecified.

### Automatic Disk Space Verification
Before starting transformer training, `src/train.py` executes a pre-flight disk check (`check_disk_space`). If the target drive has less than 2.0 GB free, it prints an instructional warning and cleanly aborts transformer training without corrupting existing files.

---

## Running Automated Tests

A comprehensive test suite of 19 automated tests is located in `tests/`:
```bash
pytest tests/ -v
```
Test suite coverage:
- `test_data_loader.py`: Dataset loading, label normalization, missing value handling, deduplication, schema validation.
- `test_preprocessing.py`: URL/HTML/email stripping, Reuters dateline removal, NLTK stopword filtering, lemmatization.
- `test_features.py`: TF-IDF vectorizer configuration, vocabulary fitting, transform shape checks.
- `test_models_and_eval.py`: Classical model training, prediction, metric calculations (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix).
- `test_prediction_pipeline.py`: Prediction input validation, length bounds, model availability detection.
- `test_dashboard_and_transformer.py`: Dashboard schema verification, transformer configuration, tokenization padding and truncation.

---

## Known Limitations & Future Improvements
1. **Writing Style vs. Fact Verification:** The system analyzes writing style, not external factual truth. A factual article written with sensationalist vocabulary may be flagged as Fake, while polished misinformation may be classified as Real.
2. **Dataset Specificity:** Models trained on ISOT or the demo data learn the specific distribution of their training corpora. Cross-domain generalizability to breaking news requires ongoing retraining.
3. **Future Extension — Fact-Checking APIs:** Integrating claim-matching APIs (e.g., Google Fact Check Tools API) alongside stylistic NLP would provide a multi-layered verification signal.
4. **Future Extension — Multilingual Support:** Future iterations could extend the preprocessing and transformer components to support multilingual models (e.g. `xlm-roberta-base`).
