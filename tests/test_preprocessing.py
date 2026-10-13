"""Unit tests for text preprocessing functions."""
from __future__ import annotations

from src.preprocessing import light_clean, preprocess, preprocess_series


def test_light_clean_removes_noise():
    """Verify light cleaning removes HTML, URLs, emails, and datelines while preserving words."""
    raw = "WASHINGTON (Reuters) - Check this out: <p>Visit http://example.com or email test@mail.com now.</p>"
    cleaned = light_clean(raw)
    assert "<p>" not in cleaned
    assert "http://example.com" not in cleaned
    assert "test@mail.com" not in cleaned
    assert "(Reuters)" not in cleaned
    assert "Check this out" in cleaned


def test_preprocess_tfidf():
    """Verify full preprocessing lowercases, tokenizes, removes stopwords, and lemmatizes."""
    raw = "The economists were reviewing multiple reports regarding national economies."
    cleaned = preprocess(raw)
    # Stopword 'the' and 'were' should be removed
    tokens = cleaned.split()
    assert "the" not in tokens
    assert "were" not in tokens
    # Lemmatization: 'economies' -> 'economy', 'reports' -> 'report'
    assert "report" in tokens or "economy" in tokens


def test_preprocess_series():
    """Verify batch series preprocessing."""
    texts = [
        "First article about technology trends.",
        "Second article covering financial markets.",
    ]
    results = preprocess_series(texts)
    assert len(results) == 2
    assert isinstance(results[0], str)
    assert isinstance(results[1], str)
