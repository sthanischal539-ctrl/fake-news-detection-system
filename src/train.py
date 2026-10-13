"""Train and save the models.

Classical (always):
    1. Logistic Regression
    2. Naive Bayes (MultinomialNB)
    3. Linear SVM (calibrated probabilities)
    All trained on TF-IDF features.

Transformer (optional, --with-transformer):
    Fine-tune DistilBERT sequence classification model.

Usage:
    python -m src.train                     # classical models only
    python -m src.train --with-transformer  # classical + DistilBERT
    python -m src.train --force-reprocess   # re-generate dataset splits and retrain
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from . import config, features
from .data_loader import load_splits
from .preprocessing import preprocess_series


# --------------------------------------------------------------------------- #
# Classical models
# --------------------------------------------------------------------------- #
def train_classical(force_reprocess: bool = False) -> None:
    config.set_seed()
    splits = load_splits(force_reprocess=force_reprocess)
    train, val = splits["train"], splits["val"]

    print("Preprocessing text for training...")
    X_train_text = preprocess_series(train["text"])
    y_train = train["label"].values

    vectorizer = features.build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    features.save_vectorizer(vectorizer)
    print(f"TF-IDF matrix shape: {X_train.shape}")

    # 1. Logistic Regression ------------------------------------------------ #
    print("Training Logistic Regression...")
    logreg = LogisticRegression(max_iter=1000, random_state=config.SEED, C=1.0)
    logreg.fit(X_train, y_train)
    joblib.dump(logreg, config.LOGREG_PATH)
    print(f"Saved Logistic Regression -> {config.LOGREG_PATH}")

    # 2. Naive Bayes (MultinomialNB) ---------------------------------------- #
    print("Training Naive Bayes (MultinomialNB)...")
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train, y_train)
    joblib.dump(nb, config.NB_PATH)
    print(f"Saved Naive Bayes -> {config.NB_PATH}")

    # 3. Linear SVM (calibrated probabilities) ----------------------------- #
    print("Training Linear SVM (with probability calibration)...")
    base_svm = LinearSVC(random_state=config.SEED, C=1.0)
    svm = CalibratedClassifierCV(base_svm, cv=3)
    svm.fit(X_train, y_train)
    joblib.dump(svm, config.SVM_PATH)
    print(f"Saved Linear SVM -> {config.SVM_PATH}")

    print(f"\nAll classical models trained and saved to {config.MODELS_DIR}")


# --------------------------------------------------------------------------- #
# Transformer (optional)
# --------------------------------------------------------------------------- #
def train_transformer(force_reprocess: bool = False) -> None:
    """Fine-tune DistilBERT sequence classifier.

    Checks disk space and HF cache location before downloading/training.
    """
    # Check disk space before starting
    has_space, free_gb, total_gb = config.check_disk_space(config._HF_CACHE, min_gb=2.0)
    if not has_space:
        print(f"\n[WARNING] Insufficient disk space on drive for Hugging Face cache!")
        print(f"Directory: {config._HF_CACHE}")
        print(f"Available: {free_gb:.1f} GB, Required: ~2.0 GB.")
        print("To configure a different drive with more space, set the HF_HOME environment variable.")
        print("Example (Linux/macOS):  export HF_HOME=/path/to/larger_drive/hf_cache")
        print("Example (Windows CMD):  set HF_HOME=D:\\hf_cache")
        print("Example (PowerShell):   $env:HF_HOME = 'D:\\hf_cache'")
        print("Skipping transformer training to prevent running out of disk space.\n")
        return

    print(f"Disk check passed: {free_gb:.1f} GB free at {config._HF_CACHE}")

    try:
        import numpy as np
        import torch
        from sklearn.metrics import accuracy_score, f1_score
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as err:
        print(f"[warning] Transformer dependencies missing: {err}. Run: pip install torch transformers accelerate")
        return

    from .preprocessing import light_clean

    config.set_seed()
    splits = load_splits(force_reprocess=force_reprocess)
    train, val = splits["train"], splits["val"]

    # Subsample if configured for CPU feasibility
    if config.TRANSFORMER_MAX_TRAIN and len(train) > config.TRANSFORMER_MAX_TRAIN:
        print(f"Subsampling transformer train set to {config.TRANSFORMER_MAX_TRAIN} records for CPU feasibility.")
        train = train.sample(config.TRANSFORMER_MAX_TRAIN, random_state=config.SEED).reset_index(drop=True)
    if config.TRANSFORMER_MAX_EVAL and len(val) > config.TRANSFORMER_MAX_EVAL:
        print(f"Subsampling transformer val set to {config.TRANSFORMER_MAX_EVAL} records.")
        val = val.sample(config.TRANSFORMER_MAX_EVAL, random_state=config.SEED).reset_index(drop=True)

    print(f"Loading pretrained tokenizer: {config.TRANSFORMER_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.TRANSFORMER_MODEL_NAME,
        cache_dir=str(config._HF_CACHE),
    )

    class NewsDataset(Dataset):
        def __init__(self, texts, labels):
            cleaned = [light_clean(t) for t in texts]
            self.enc = tokenizer(
                cleaned,
                truncation=True,
                padding="max_length",
                max_length=config.TRANSFORMER_MAX_LEN,
            )
            self.labels = [int(l) for l in labels]

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    train_ds = NewsDataset(train["text"].tolist(), train["label"].tolist())
    val_ds = NewsDataset(val["text"].tolist(), val["label"].tolist())

    print(f"Loading pretrained model: {config.TRANSFORMER_MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.TRANSFORMER_MODEL_NAME,
        num_labels=2,
        cache_dir=str(config._HF_CACHE),
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, zero_division=0)
        return {"accuracy": float(acc), "f1": float(f1)}

    trainer_args_dict = {
        "output_dir": str(config.TRANSFORMER_DIR / "_trainer"),
        "num_train_epochs": config.TRANSFORMER_EPOCHS,
        "per_device_train_batch_size": config.TRANSFORMER_BATCH_SIZE,
        "per_device_eval_batch_size": config.TRANSFORMER_BATCH_SIZE,
        "logging_steps": 50,
        "report_to": [],
        "seed": config.SEED,
        "save_strategy": "no",
    }

    try:
        args = TrainingArguments(eval_strategy="epoch", **trainer_args_dict)
    except TypeError:
        try:
            args = TrainingArguments(evaluation_strategy="epoch", **trainer_args_dict)
        except TypeError:
            args = TrainingArguments(**trainer_args_dict)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("Beginning transformer fine-tuning...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Transformer fine-tuning completed in {elapsed:.1f}s")

    # Evaluate on validation set
    val_res = trainer.evaluate()
    print(f"Transformer validation results: {val_res}")

    config.TRANSFORMER_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.TRANSFORMER_DIR)
    tokenizer.save_pretrained(config.TRANSFORMER_DIR)
    print(f"Saved fine-tuned DistilBERT model and tokenizer to {config.TRANSFORMER_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fake-news detection models.")
    parser.add_argument(
        "--with-transformer",
        action="store_true",
        help="Also fine-tune DistilBERT (downloads pretrained weights to HF_HOME cache; slower on CPU).",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force re-generation of train/val/test splits before training.",
    )
    args = parser.parse_args()

    train_classical(force_reprocess=args.force_reprocess)

    if args.with_transformer:
        try:
            train_transformer(force_reprocess=args.force_reprocess)
        except Exception as exc:
            print(f"[warning] Transformer training failed with error: {exc}")
            print("Classical models remain available and fully functional.")


if __name__ == "__main__":
    main()
