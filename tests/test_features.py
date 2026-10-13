"""Unit tests for TF-IDF feature extraction."""
from __future__ import annotations

import tempfile
from pathlib import Path

from src import config, features


def test_build_vectorizer():
    """Verify vectorizer configuration hyperparameters."""
    vec = features.build_vectorizer()
    assert vec.max_features == config.TFIDF_MAX_FEATURES
    assert vec.ngram_range == config.TFIDF_NGRAM_RANGE
    assert vec.sublinear_tf is True


def test_vectorizer_fit_and_transform():
    """Verify vectorizer fits vocabulary and produces sparse matrix."""
    vec = features.build_vectorizer()
    corpus = [
        "economy grew rapidly in fourth quarter",
        "central bank announces interest rate change",
        "wild conspiracy theory spreads across social media",
    ]
    # min_df=2 in config might require terms to appear >= 2 times, let's test with repeated terms
    corpus_repeated = corpus + corpus
    X = vec.fit_transform(corpus_repeated)
    assert X.shape[0] == len(corpus_repeated)
    assert X.shape[1] > 0
