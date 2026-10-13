"""Load the news dataset, label it, deduplicate, and split into train/val/test.

Source priority (first match wins):
  1. ISOT dataset:        data/raw/True.csv  +  data/raw/Fake.csv
  2. Kaggle "fake-news":  data/raw/fake_news.csv  (columns include text,label)
  3. Synthetic demo:      data/raw/demo_news.csv  (auto-generated if absent)

All splits are stratified and seeded so results are reproducible.
"""
from __future__ import annotations

import datetime
import json
import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config
from .make_demo_data import generate as _generate_demo

logger = logging.getLogger(__name__)


def _load_isot() -> pd.DataFrame | None:
    """Load the ISOT dataset from True.csv and Fake.csv if present in data/raw/."""
    true_path = config.RAW_DIR / "True.csv"
    fake_path = config.RAW_DIR / "Fake.csv"
    if not (true_path.exists() and fake_path.exists()):
        return None

    try:
        real = pd.read_csv(true_path)
        fake = pd.read_csv(fake_path)
    except Exception as exc:
        logger.warning(f"Failed to read ISOT files: {exc}")
        return None

    if real.empty or fake.empty:
        logger.warning("One or both ISOT CSV files are empty.")
        return None

    real["label"] = config.LABEL_REAL
    fake["label"] = config.LABEL_FAKE
    df = pd.concat([real, fake], ignore_index=True)

    # ISOT standard columns: title, text, subject, date
    # Combine title + text into a single body field when both are available.
    has_title = "title" in df.columns
    has_text = "text" in df.columns

    if has_title and has_text:
        title = df["title"].fillna("").astype(str).str.strip()
        body = df["text"].fillna("").astype(str).str.strip()
        df["text"] = (title + ". " + body).str.strip()
    elif has_title and not has_text:
        df["text"] = df["title"].fillna("").astype(str).str.strip()
    elif has_text:
        df["text"] = df["text"].fillna("").astype(str).str.strip()
    else:
        logger.warning("ISOT CSV files missing both 'title' and 'text' columns.")
        return None

    return df[["text", "label"]]


def _load_kaggle() -> pd.DataFrame | None:
    """Load Kaggle fake-news dataset from fake_news.csv if present in data/raw/."""
    path = config.RAW_DIR / "fake_news.csv"
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.warning(f"Failed to read Kaggle CSV: {exc}")
        return None

    if "label" not in df.columns:
        return None

    # Use title + text if title exists and text exists
    if "title" in df.columns and "text" in df.columns:
        title = df["title"].fillna("").astype(str).str.strip()
        body = df["text"].fillna("").astype(str).str.strip()
        df["text"] = (title + ". " + body).str.strip()
    elif "text" not in df.columns and "title" in df.columns:
        df["text"] = df["title"].fillna("").astype(str).str.strip()

    return df[["text", "label"]]


def _load_demo() -> pd.DataFrame:
    """Load or generate the synthetic demonstration dataset."""
    path = config.RAW_DIR / "demo_news.csv"
    if path.exists():
        try:
            df = pd.read_csv(path)
            if not df.empty and "text" in df.columns and "label" in df.columns:
                return df
        except Exception:
            pass

    df = _generate_demo()
    df.to_csv(path, index=False)
    return df


def _normalize_label(val) -> int | None:
    """Safely map arbitrary label representations to 0 (REAL) or 1 (FAKE)."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        try:
            int_val = int(val)
            if int_val in (0, 1):
                return int_val
        except (ValueError, OverflowError):
            return None
    s = str(val).strip().upper()
    if s in ("0", "REAL", "TRUE", "RELIABLE", "0.0"):
        return config.LABEL_REAL
    if s in ("1", "FAKE", "FALSE", "UNRELIABLE", "1.0"):
        return config.LABEL_FAKE
    return None


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw dataset:

    - Drop missing text or labels.
    - Normalize labels to 0 (REAL) and 1 (FAKE).
    - Drop short/empty text bodies (<= 20 chars).
    - Deduplicate text to eliminate data leakage between train/val/test splits.
    """
    if df is None or df.empty:
        raise ValueError("Cannot clean an empty or None DataFrame.")

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"DataFrame must contain 'text' and 'label' columns. Found: {list(df.columns)}")

    df = df.copy()

    # Drop explicit NA rows
    df = df.dropna(subset=["text", "label"])

    # Clean text
    df["text"] = df["text"].astype(str).str.strip()

    # Map labels safely
    df["label"] = df["label"].apply(_normalize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Filter out empty or near-empty articles
    df = df[df["text"].str.len() > 20]

    # Deduplicate to prevent train/test leakage
    df = df.drop_duplicates(subset=["text"])

    if df.empty:
        raise ValueError("Dataset is empty after cleaning and deduplication.")

    unique_labels = df["label"].unique()
    if len(unique_labels) < 2:
        raise ValueError(f"Dataset must contain both classes (0 and 1). Found only: {unique_labels}")

    return df.reset_index(drop=True)


def load_raw() -> tuple[pd.DataFrame, str]:
    """Return (dataframe, source_name). Dataframe has columns: text, label."""
    for loader, name in (
        (_load_isot, "ISOT (True.csv + Fake.csv)"),
        (_load_kaggle, "Kaggle fake_news.csv"),
    ):
        df = loader()
        if df is not None and len(df) > 0:
            cleaned = _clean(df)
            return cleaned, name

    cleaned_demo = _clean(_load_demo())
    return cleaned_demo, "SYNTHETIC demo dataset"


def get_dataset_info() -> dict:
    """Load metadata about the active dataset split."""
    if config.DATASET_INFO_PATH.exists():
        try:
            with open(config.DATASET_INFO_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "source": "SYNTHETIC demo dataset (unverified)",
        "is_synthetic": True,
        "total_records": 0,
        "train_records": 0,
        "val_records": 0,
        "test_records": 0,
        "real_count": 0,
        "fake_count": 0,
        "created_at": "unknown",
    }


def make_splits(save: bool = True) -> dict[str, pd.DataFrame]:
    """Create stratified train/val/test splits, record metadata, and optionally save to disk."""
    config.set_seed()
    df, source = load_raw()
    is_synthetic = "SYNTHETIC" in source.upper()

    real_count = int((df.label == config.LABEL_REAL).sum())
    fake_count = int((df.label == config.LABEL_FAKE).sum())

    print(f"Loaded {len(df)} articles from: {source}")
    print(f"Label balance -> REAL: {real_count}, FAKE: {fake_count}")

    # First carve out the held-out test set, then split remainder into train/val.
    train_val, test = train_test_split(
        df, test_size=config.TEST_SIZE, stratify=df["label"], random_state=config.SEED
    )
    val_fraction = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["label"],
        random_state=config.SEED,
    )

    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }

    if save:
        train.to_csv(config.TRAIN_CSV, index=False)
        val.to_csv(config.VAL_CSV, index=False)
        test.to_csv(config.TEST_CSV, index=False)

        info = {
            "source": source,
            "is_synthetic": is_synthetic,
            "total_records": len(df),
            "train_records": len(train),
            "val_records": len(val),
            "test_records": len(test),
            "real_count": real_count,
            "fake_count": fake_count,
            "created_at": datetime.datetime.now().isoformat(),
        }
        with open(config.DATASET_INFO_PATH, "w") as f:
            json.dump(info, f, indent=2)

        print(f"Saved splits -> train {len(train)}, val {len(val)}, test {len(test)}")
        print(f"Saved dataset metadata to {config.DATASET_INFO_PATH}")

    return splits


def load_splits(force_reprocess: bool = False) -> dict[str, pd.DataFrame]:
    """Load previously-saved splits, creating or refreshing them if required."""
    all_exist = (
        config.TRAIN_CSV.exists()
        and config.VAL_CSV.exists()
        and config.TEST_CSV.exists()
        and config.DATASET_INFO_PATH.exists()
    )

    if force_reprocess or not all_exist:
        return make_splits(save=True)

    # Check if ISOT raw files are present now but current processed splits are synthetic
    info = get_dataset_info()
    true_path = config.RAW_DIR / "True.csv"
    fake_path = config.RAW_DIR / "Fake.csv"
    if info.get("is_synthetic", False) and (true_path.exists() and fake_path.exists()):
        print("Detected real ISOT dataset files in data/raw/! Re-generating splits...")
        return make_splits(save=True)

    return {
        "train": pd.read_csv(config.TRAIN_CSV),
        "val": pd.read_csv(config.VAL_CSV),
        "test": pd.read_csv(config.TEST_CSV),
    }


if __name__ == "__main__":
    make_splits(save=True)
