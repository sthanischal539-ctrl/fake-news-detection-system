"""Unit tests for the prediction pipeline and input validation."""
from __future__ import annotations

import pytest

from src import config
from src.predict import get_available_models, validate_input_text


def test_validate_input_text():
    """Verify input validation handles None, whitespace, short, and long texts."""
    # None
    is_valid, msg, _ = validate_input_text(None)
    assert not is_valid
    assert "None" in msg

    # Empty / whitespace
    is_valid, msg, _ = validate_input_text("    ")
    assert not is_valid
    assert "empty" in msg

    # Too short (< 20 chars)
    is_valid, msg, _ = validate_input_text("Hello there.")
    assert not is_valid
    assert "too short" in msg

    # Valid
    valid_text = "This is a legitimate news story that easily exceeds the minimum character threshold."
    is_valid, msg, clean = validate_input_text(valid_text)
    assert is_valid
    assert msg == ""
    assert clean == valid_text

    # Too long (should truncate and warn)
    long_text = "a" * 30000
    is_valid, msg, clean = validate_input_text(long_text, max_chars=1000)
    assert is_valid
    assert "truncated" in msg
    assert len(clean) == 1000


def test_get_available_models():
    """Verify model availability returns dict with model names as keys."""
    models = get_available_models()
    assert isinstance(models, dict)
    # If DistilBERT is not trained, it must not appear in available models
    if not (config.TRANSFORMER_DIR / "config.json").exists():
        assert "DistilBERT" not in models
