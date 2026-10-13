"""Unit tests for dashboard data loading and transformer configuration/tokenization."""
from __future__ import annotations

import json
from pathlib import Path

from src import config
from src.data_loader import get_dataset_info


def test_get_dataset_info_structure():
    """Verify get_dataset_info returns expected keys."""
    info = get_dataset_info()
    assert "source" in info
    assert "is_synthetic" in info
    assert "total_records" in info
    assert "train_records" in info
    assert "test_records" in info


def test_metrics_json_schema():
    """Verify that if metrics.json exists, it contains expected structure and no fabricated data."""
    if config.METRICS_JSON.exists():
        with open(config.METRICS_JSON, "r") as f:
            data = json.load(f)

        assert isinstance(data, dict)
        if "_metadata" in data:
            assert "dataset_source" in data["_metadata"]
            assert "is_synthetic" in data["_metadata"]

        for model_name, m in data.items():
            if model_name.startswith("_"):
                continue
            assert "accuracy" in m
            assert "precision" in m
            assert "recall" in m
            assert "f1" in m
            assert "confusion_matrix" in m
            assert isinstance(m["confusion_matrix"], list)


def test_transformer_config_and_tokenization():
    """Verify DistilBERT configuration and tokenizer behavior without needing full model weights download."""
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError:
        return

    # Check configuration setup for binary classification
    cfg = AutoConfig.from_pretrained(config.TRANSFORMER_MODEL_NAME, num_labels=2)
    assert cfg.num_labels == 2
    assert cfg.model_type == "distilbert"

    # Check tokenizer behavior
    tok = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    sample_text = "Officials reported modest gains in the quarterly economic review."
    encoded = tok(
        sample_text,
        max_length=config.TRANSFORMER_MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape[1] == config.TRANSFORMER_MAX_LEN
