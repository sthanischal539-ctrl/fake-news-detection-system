"""TF-IDF feature extraction for the classical models.

The fitted vectorizer is saved so the app and evaluation transform new text
exactly the way the models were trained.
"""
from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config


def build_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        sublinear_tf=True,
    )


def save_vectorizer(vectorizer: TfidfVectorizer) -> None:
    joblib.dump(vectorizer, config.TFIDF_PATH)


def load_vectorizer() -> TfidfVectorizer:
    if not config.TFIDF_PATH.exists():
        raise FileNotFoundError(
            f"Fitted TF-IDF vectorizer not found at {config.TFIDF_PATH}. "
            "Please train the models first using: python -m src.train"
        )
    return joblib.load(config.TFIDF_PATH)
