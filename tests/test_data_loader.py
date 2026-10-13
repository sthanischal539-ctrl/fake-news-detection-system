"""Unit tests for dataset loading, validation, cleaning, and splitting."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config, make_demo_data
from src.data_loader import _clean, _normalize_label, load_raw


def test_normalize_label():
    """Verify robust label normalization across integer, float, and string representations."""
    assert _normalize_label(0) == config.LABEL_REAL
    assert _normalize_label(1) == config.LABEL_FAKE
    assert _normalize_label("0") == config.LABEL_REAL
    assert _normalize_label("1") == config.LABEL_FAKE
    assert _normalize_label("REAL") == config.LABEL_REAL
    assert _normalize_label("real") == config.LABEL_REAL
    assert _normalize_label("FAKE") == config.LABEL_FAKE
    assert _normalize_label("fake") == config.LABEL_FAKE
    assert _normalize_label("reliable") == config.LABEL_REAL
    assert _normalize_label("unreliable") == config.LABEL_FAKE
    assert _normalize_label(None) is None
    assert _normalize_label("invalid_label") is None


def test_clean_valid_dataframe():
    """Verify that cleaning valid data retains valid articles and normalizes labels."""
    raw = pd.DataFrame({
        "text": [
            "This is a valid genuine news article with enough characters to pass the threshold.",
            "This is another fake news article with sensational conspiracy claims and length.",
        ],
        "label": [0, 1],
    })
    cleaned = _clean(raw)
    assert len(cleaned) == 2
    assert list(cleaned["label"]) == [0, 1]


def test_clean_missing_and_malformed_input():
    """Verify handling of missing text, invalid labels, short texts, and NaN values."""
    raw = pd.DataFrame({
        "text": [
            None,
            "Too short",  # <= 20 chars
            "   ",  # whitespace
            "A sufficiently long article describing an event that should be properly retained.",
            "Another sufficiently long fabricated article that should also be properly retained.",
        ],
        "label": [
            0,
            1,
            0,
            "REAL",
            "invalid_xyz",  # invalid label should be dropped
        ],
    })
    # Since only one valid row remains ("A sufficiently long..."), cleaning should detect missing class
    with pytest.raises(ValueError, match="Dataset must contain both classes"):
        _clean(raw)


def test_clean_deduplication_prevents_leakage():
    """Verify that identical texts are deduplicated to avoid data leakage between splits."""
    long_text = "Scientists at the university announced the successful completion of a clinical trial."
    raw = pd.DataFrame({
        "text": [long_text, long_text, "A totally different news report about the national economy."],
        "label": [0, 0, 1],
    })
    cleaned = _clean(raw)
    assert len(cleaned) == 2
    assert (cleaned["text"] == long_text).sum() == 1


def test_clean_missing_columns():
    """Verify that DataFrames lacking required columns raise ValueError."""
    with pytest.raises(ValueError, match="DataFrame must contain 'text' and 'label' columns"):
        _clean(pd.DataFrame({"wrong_col": [1, 2]}))


def test_synthetic_data_generation():
    """Verify that the synthetic demo generator produces a balanced DataFrame."""
    df = make_demo_data.generate(n_per_class=20, seed=42)
    assert len(df) == 40
    assert (df["label"] == config.LABEL_REAL).sum() == 20
    assert (df["label"] == config.LABEL_FAKE).sum() == 20
    assert (df["text"].str.len() > 30).all()


def test_load_raw_fallback():
    """Verify load_raw produces non-empty DataFrame and identifies source name."""
    df, source_name = load_raw()
    assert not df.empty
    assert "text" in df.columns
    assert "label" in df.columns
    assert isinstance(source_name, str)
