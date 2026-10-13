"""Central configuration: paths, random seed, and hyperparameters.

Keeping every path and knob in one place makes the pipeline reproducible and easy
to retarget (e.g. pointing at the real ISOT dataset instead of the demo data).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy, and (if available) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is optional for the classical pipeline
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Paths (all relative to the project root, so the repo is portable)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Hugging Face cache configuration:
# Allow users to point to another drive (e.g., D:\hf_cache on Windows) via HF_HOME or TRANSFORMERS_CACHE.
# If not explicitly specified, default to models/hf_cache inside the project directory.
_env_hf_home = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE")
if _env_hf_home:
    _HF_CACHE = Path(_env_hf_home).resolve()
else:
    _HF_CACHE = (PROJECT_ROOT / "models" / "hf_cache").resolve()

try:
    _HF_CACHE.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

os.environ.setdefault("HF_HOME", str(_HF_CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(_HF_CACHE))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Processed split files
TRAIN_CSV = PROCESSED_DIR / "train.csv"
VAL_CSV = PROCESSED_DIR / "val.csv"
TEST_CSV = PROCESSED_DIR / "test.csv"
DATASET_INFO_PATH = PROCESSED_DIR / "dataset_info.json"

# Saved model artefacts
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LOGREG_PATH = MODELS_DIR / "logreg.joblib"
NB_PATH = MODELS_DIR / "naive_bayes.joblib"
SVM_PATH = MODELS_DIR / "svm.joblib"
TRANSFORMER_DIR = MODELS_DIR / "distilbert"

CLASSICAL_MODELS = {
    "Logistic Regression": LOGREG_PATH,
    "Naive Bayes": NB_PATH,
    "Linear SVM": SVM_PATH,
}

# Evaluation output
METRICS_JSON = RESULTS_DIR / "metrics.json"


def check_disk_space(target_path: Path | str = _HF_CACHE, min_gb: float = 2.0) -> tuple[bool, float, float]:
    """Check whether the volume containing target_path has at least min_gb free.

    Returns:
        (has_enough_space, free_gb, total_gb)
    """
    import shutil

    p = Path(target_path)
    # If the target path doesn't exist yet, check its closest existing ancestor
    check_dir = p
    while not check_dir.exists() and check_dir.parent != check_dir:
        check_dir = check_dir.parent

    try:
        usage = shutil.disk_usage(check_dir)
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        return (free_gb >= min_gb, round(free_gb, 2), round(total_gb, 2))
    except Exception:
        # If check fails for any OS reason, assume ok to avoid blocking execution
        return (True, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
# 0 = REAL, 1 = FAKE  (so "positive class" = Fake, which is what we want to catch)
LABELS = {0: "REAL", 1: "FAKE"}
LABEL_REAL = 0
LABEL_FAKE = 1

# --------------------------------------------------------------------------- #
# Data split
# --------------------------------------------------------------------------- #
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the *full* dataset

# --------------------------------------------------------------------------- #
# TF-IDF hyperparameters
# --------------------------------------------------------------------------- #
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2

# --------------------------------------------------------------------------- #
# Transformer (CPU-friendly defaults — configurable via environment variables)
# --------------------------------------------------------------------------- #
TRANSFORMER_MODEL_NAME = os.environ.get("TRANSFORMER_MODEL_NAME", "distilbert-base-uncased")
TRANSFORMER_MAX_LEN = int(os.environ.get("TRANSFORMER_MAX_LEN", "128"))
TRANSFORMER_EPOCHS = int(os.environ.get("TRANSFORMER_EPOCHS", "1"))
TRANSFORMER_BATCH_SIZE = int(os.environ.get("TRANSFORMER_BATCH_SIZE", "16"))
# Cap training/eval size for CPU execution so training finishes quickly on a laptop.
# Set TRANSFORMER_MAX_TRAIN=0 or None to use the full split (recommended for GPU).
_env_max_train = os.environ.get("TRANSFORMER_MAX_TRAIN", "160")
TRANSFORMER_MAX_TRAIN = int(_env_max_train) if _env_max_train and _env_max_train != "0" else None
_env_max_eval = os.environ.get("TRANSFORMER_MAX_EVAL", "60")
TRANSFORMER_MAX_EVAL = int(_env_max_eval) if _env_max_eval and _env_max_eval != "0" else None
