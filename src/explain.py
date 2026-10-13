"""Explainability for the classical models.

- `explain_prediction`: LIME highlights the words that pushed THIS article toward
  Fake or Real for a supported classical model.
- `top_global_features`: the strongest Fake/Real indicator words learned overall
  (from Logistic Regression coefficients).

LIME works on raw text through a pipeline (preprocess -> TF-IDF -> model), so the
words it returns are human-readable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer

from . import config, features
from .preprocessing import preprocess

_CLASS_NAMES = ["REAL", "FAKE"]

EXPLANATION_DISCLAIMER = (
    "The highlighted words indicate patterns that influenced the model's prediction. "
    "They do not prove whether the article's claims are true or false."
)


def _load(model_path: Path):
    if not config.TFIDF_PATH.exists():
        raise FileNotFoundError(f"TF-IDF vectorizer not found at {config.TFIDF_PATH}.")
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model file not found at {p}.")
    vectorizer = features.load_vectorizer()
    model = joblib.load(p)
    return vectorizer, model


def _predict_proba_fn(vectorizer, model):
    """Return a function mapping a list of raw texts -> class-probability array."""
    def fn(texts):
        cleaned = [preprocess(t) or t for t in texts]
        X = vectorizer.transform(cleaned)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            scores = np.vstack([-scores, scores]).T
            e = np.exp(scores - scores.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        # Fallback for models without probabilities
        preds = model.predict(X)
        res = np.zeros((len(preds), 2))
        for i, p in enumerate(preds):
            res[i, int(p)] = 1.0
        return res

    return fn


def explain_prediction(
    text: str,
    model_path: Path | str | None = None,
    model_name: str = "Logistic Regression",
    num_features: int = 10,
) -> dict[str, Any]:
    """Explain a single prediction with LIME for a classical TF-IDF model.

    Returns dict with:
      - 'model_name': Name of model explained
      - 'label': Predicted label (REAL / FAKE)
      - 'label_idx': Index of predicted label (0 or 1)
      - 'confidence': Model confidence score
      - 'proba_fake': Probability of Fake class
      - 'word_weights': List of (word, weight) tuples
      - 'fake_words': List of words pushing towards FAKE
      - 'real_words': List of words pushing towards REAL
      - 'disclaimer': Explicit limitation disclaimer
    """
    model_path = Path(model_path) if model_path else config.LOGREG_PATH
    vectorizer, model = _load(model_path)
    predict_fn = _predict_proba_fn(vectorizer, model)

    proba = predict_fn([text])[0]
    label_idx = int(np.argmax(proba))

    explainer = LimeTextExplainer(class_names=_CLASS_NAMES, random_state=config.SEED)
    exp = explainer.explain_instance(
        text, predict_fn, num_features=num_features, labels=(config.LABEL_FAKE,)
    )
    # Weights are reported w.r.t. the FAKE class (positive = FAKE, negative = REAL)
    word_weights = exp.as_list(label=config.LABEL_FAKE)

    fake_words = [(w, weight) for w, weight in word_weights if weight > 0]
    real_words = [(w, abs(weight)) for w, weight in word_weights if weight < 0]

    return {
        "model_name": model_name,
        "label": config.LABELS[label_idx],
        "label_idx": label_idx,
        "confidence": float(proba[label_idx]),
        "proba_fake": float(proba[config.LABEL_FAKE]),
        "word_weights": word_weights,
        "fake_words": fake_words,
        "real_words": real_words,
        "disclaimer": EXPLANATION_DISCLAIMER,
    }


def top_global_features(n: int = 15) -> dict:
    """Return the strongest global FAKE and REAL indicator words (from Logistic Regression)."""
    if not (config.LOGREG_PATH.exists() and config.TFIDF_PATH.exists()):
        return {"fake_indicators": [], "real_indicators": []}

    vectorizer = features.load_vectorizer()
    model = joblib.load(config.LOGREG_PATH)
    coefs = model.coef_[0]
    names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(coefs)

    top_fake = [(names[i], float(coefs[i])) for i in order[::-1][:n]]
    top_real = [(names[i], float(coefs[i])) for i in order[:n]]
    return {
        "fake_indicators": top_fake,
        "real_indicators": top_real,
        "disclaimer": EXPLANATION_DISCLAIMER,
    }


if __name__ == "__main__":
    demo = "SHOCKING miracle cure the elites tried to hide from you forever!!!"
    print(explain_prediction(demo))
