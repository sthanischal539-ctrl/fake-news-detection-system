# Implementation Plan

## Milestones

### M0 — Planning (this phase)
Deliver the four `/docs` files and stop for review. ✅ (pending your approval)

### M1 — Environment & data
- Create virtualenv + `requirements.txt`.
- Download ISOT dataset (per `dataset.md`) into git-ignored `data/raw/`.
- `data_loader.py`: merge, label, dedupe, stratified train/val/test split (seeded).

### M2 — Preprocessing
- `preprocessing.py`: cleaning, tokenization, stopword removal, lemmatization on
  full article text; dateline stripping. Unit sanity checks on a few samples.

### M3 — Classical models
- `features.py`: fit + save TF-IDF vectorizer.
- `train.py`: train + save Logistic Regression and Linear SVM (calibrated probs).

### M4 — Transformer model
- `train.py` (transformer path): fine-tune DistilBERT; save model + tokenizer.

### M5 — Evaluation
- `evaluate.py`: on the held-out test set, compute Accuracy, Precision, Recall, F1
  and a confusion matrix for **each** model; save results (JSON + figures).
- Optional cross-dataset check on the secondary dataset.

### M6 — Explainability
- `explain.py`: LIME/SHAP for classical models; token attribution for transformer.

### M7 — Streamlit app + dashboard
- `app.py`: Predict tab (label + confidence + influential words + disclaimer) and
  Dashboard tab (comparison table + charts).

### M8 — Polish
- README with setup + run instructions, docstrings, reproducible seeds, cleanup.

## Proposed file / folder structure

```
nischal-nlp-project/
├── docs/
│   ├── problem-statement.md
│   ├── dataset.md
│   ├── architecture.md
│   └── implementation-plan.md
├── data/                    # git-ignored
│   ├── raw/                 # downloaded CSVs
│   └── processed/           # train/val/test splits
├── models/                  # git-ignored saved artefacts
│   ├── tfidf_vectorizer.joblib
│   ├── logreg.joblib
│   ├── svm.joblib
│   └── distilbert/
├── results/                 # metrics JSON + confusion-matrix figures
├── src/
│   ├── __init__.py
│   ├── config.py            # paths, seeds, hyperparameters
│   ├── data_loader.py       # load, merge, dedupe, split
│   ├── preprocessing.py     # clean / tokenize / stopwords / lemmatize
│   ├── features.py          # TF-IDF fit/transform/save
│   ├── train.py             # train + save all three models
│   ├── evaluate.py          # metrics + confusion matrices
│   ├── explain.py           # LIME/SHAP + transformer attribution
│   └── app.py               # Streamlit UI + dashboard
├── requirements.txt
├── .gitignore
└── README.md
```

## Models & metrics

| Model | Features | Library | Why |
|-------|----------|---------|-----|
| Logistic Regression | TF-IDF (1–2 grams) | scikit-learn | Fast, interpretable baseline |
| Linear SVM | TF-IDF (1–2 grams) | scikit-learn | Strong linear text classifier |
| DistilBERT | Subword embeddings | HF Transformers | Contextual transformer baseline |

**Metrics reported per model (held-out test set):**
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Report macro and/or per-class where the label split warrants it; the dashboard shows
all three models side by side.

## Reproducibility
- Single `SEED` in `config.py` applied to NumPy, Python `random`, scikit-learn
  splits, and the transformer trainer.
- Save the fitted vectorizer and all models so the app and evaluation use identical
  transforms.
- Pin versions in `requirements.txt`.

## Task checklist

- [ ] M1: venv + `requirements.txt`
- [ ] M1: download ISOT into `data/raw/` (git-ignored)
- [ ] M1: `data_loader.py` with seeded stratified split
- [ ] M2: `preprocessing.py` (clean/tokenize/stopwords/lemmatize + dateline strip)
- [ ] M3: `features.py` TF-IDF fit + save
- [ ] M3: train + save Logistic Regression
- [ ] M3: train + save Linear SVM (calibrated probabilities)
- [ ] M4: fine-tune + save DistilBERT
- [ ] M5: `evaluate.py` — Accuracy/Precision/Recall/F1 + confusion matrix per model
- [ ] M5: (optional) cross-dataset generalisation check
- [ ] M6: `explain.py` — LIME/SHAP (classical) + token attribution (transformer)
- [ ] M7: Streamlit Predict tab (label + confidence + words + disclaimer)
- [ ] M7: Streamlit Dashboard tab (comparison table + charts)
- [ ] M8: README, docstrings, `.gitignore`, final cleanup

## Open questions for your review
1. **Compute:** Is a GPU available for DistilBERT fine-tuning, or should I plan for a
   CPU-friendly config (fewer epochs / smaller sample / or swap to a lighter model)?
2. **Dataset:** OK to proceed with ISOT as primary (+ optional Kaggle cross-check),
   or do you have a specific dataset in mind?
3. **Explainability library:** preference for **LIME** vs **SHAP** for the classical
   models? (Default: LIME for per-prediction word highlights.)
4. **Scope of transformer:** full fine-tune vs a quick baseline on a data subset to
   keep training time reasonable?
