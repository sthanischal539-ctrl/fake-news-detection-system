"""Evaluate every trained model on the held-out TEST set.

Reports Accuracy, Precision, Recall, F1, ROC-AUC, Classification Report,
and confusion matrix per model.
Saves metrics.json for the dashboard and writes confusion-matrix figures to results/.
"""
from __future__ import annotations

import datetime
import json

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend for saving figures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config, features
from .data_loader import get_dataset_info, load_splits
from .preprocessing import preprocess_series


def _metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute comprehensive evaluation metrics for binary classification."""
    metrics_dict = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, target_names=["REAL", "FAKE"], output_dict=True, zero_division=0
        ),
    }

    if y_prob is not None:
        try:
            metrics_dict["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics_dict["roc_auc"] = None
    else:
        metrics_dict["roc_auc"] = None

    return metrics_dict


def _save_confusion_fig(cm, name: str) -> None:
    """Save an annotated confusion matrix plot."""
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_yticks([0, 1], labels=["REAL", "FAKE"])
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Label", fontweight="bold")
    ax.set_title(f"{name}\nConfusion Matrix", fontsize=11, fontweight="bold")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(config.RESULTS_DIR / f"confusion_{slug}.png", dpi=150)
    plt.close(fig)


def evaluate_all() -> dict:
    config.set_seed()
    splits = load_splits()
    test = splits["test"]
    y_true = test["label"].values

    dataset_info = get_dataset_info()
    is_synthetic = dataset_info.get("is_synthetic", True)
    dataset_name = dataset_info.get("source", "Unknown")

    print("\n========================================================")
    print(f"EVALUATING MODELS ON HELD-OUT TEST SET ({len(test)} articles)")
    print(f"Dataset Source: {dataset_name}")
    if is_synthetic:
        print("Note: Dataset is SYNTHETIC demo data. High scores reflect stylistic generation artefacts.")
    print("========================================================\n")

    results: dict[str, dict] = {}

    # ---- 1. Classical models --------------------------------------------- #
    if config.TFIDF_PATH.exists():
        vectorizer = features.load_vectorizer()
        X_test = vectorizer.transform(preprocess_series(test["text"]))

        for name, path in config.CLASSICAL_MODELS.items():
            if path.exists():
                model = joblib.load(path)
                y_pred = model.predict(X_test)

                # Extract positive class probability if available
                y_prob = None
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_test)[:, config.LABEL_FAKE]
                elif hasattr(model, "decision_function"):
                    scores = model.decision_function(X_test)
                    y_prob = 1.0 / (1.0 + np.exp(-scores))

                m = _metrics(y_true, y_pred, y_prob)
                results[name] = m
                _save_confusion_fig(m["confusion_matrix"], name)

                auc_str = f" auc={m['roc_auc']:.4f}" if m["roc_auc"] is not None else ""
                print(f"[✓] {name:<20} : acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  rec={m['recall']:.4f}  f1={m['f1']:.4f}{auc_str}")
            else:
                print(f"[-] {name:<20} : Model weights not found at {path}")
    else:
        print("[!] TF-IDF vectorizer not found. Classical models skipped. Run 'python -m src.train' first.")

    # ---- 2. Transformer (DistilBERT) ------------------------------------- #
    has_transformer = config.TRANSFORMER_DIR.exists() and (config.TRANSFORMER_DIR / "config.json").exists()
    if has_transformer:
        weights_exist = (
            (config.TRANSFORMER_DIR / "model.safetensors").exists()
            or (config.TRANSFORMER_DIR / "pytorch_model.bin").exists()
        )
        if weights_exist:
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                from .preprocessing import light_clean

                print("\nEvaluating DistilBERT...")
                tok = AutoTokenizer.from_pretrained(str(config.TRANSFORMER_DIR))
                model = AutoModelForSequenceClassification.from_pretrained(str(config.TRANSFORMER_DIR))
                model.eval()

                preds = []
                probs = []
                batch_size = 16

                with torch.no_grad():
                    for i in range(0, len(test), batch_size):
                        batch = [light_clean(t) for t in test["text"].iloc[i : i + batch_size].tolist()]
                        enc = tok(
                            batch,
                            truncation=True,
                            padding=True,
                            max_length=config.TRANSFORMER_MAX_LEN,
                            return_tensors="pt",
                        )
                        logits = model(**enc).logits
                        p = torch.softmax(logits, dim=-1)
                        preds.extend(logits.argmax(-1).tolist())
                        probs.extend(p[:, config.LABEL_FAKE].tolist())

                m_bert = _metrics(y_true, np.array(preds), np.array(probs))
                results["DistilBERT"] = m_bert
                _save_confusion_fig(m_bert["confusion_matrix"], "DistilBERT")
                auc_str = f" auc={m_bert['roc_auc']:.4f}" if m_bert["roc_auc"] is not None else ""
                print(f"[✓] {'DistilBERT':<20} : acc={m_bert['accuracy']:.4f}  prec={m_bert['precision']:.4f}  rec={m_bert['recall']:.4f}  f1={m_bert['f1']:.4f}{auc_str}")
            except Exception as exc:
                print(f"[!] Transformer evaluation failed: {exc}")
        else:
            print("[-] DistilBERT directory exists but model weights are missing.")
    else:
        print("[-] DistilBERT has not been trained yet. Train with 'python -m src.train --with-transformer'.")

    # Save output metadata + results
    output_payload = {
        "_metadata": {
            "dataset_source": dataset_name,
            "is_synthetic": is_synthetic,
            "test_samples": int(len(test)),
            "real_samples": int((y_true == config.LABEL_REAL).sum()),
            "fake_samples": int((y_true == config.LABEL_FAKE).sum()),
            "evaluated_at": datetime.datetime.now().isoformat(),
            "synthetic_disclaimer": (
                "Scores evaluated on synthetic demonstration data reflect vocabulary "
                "generation rules rather than real-world news detection ability."
                if is_synthetic
                else "Evaluated on genuine news dataset."
            ),
        },
        **results,
    }

    with open(config.METRICS_JSON, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved evaluation metrics to {config.METRICS_JSON}")
    print(f"Saved confusion matrix plots to {config.RESULTS_DIR}")
    return results


if __name__ == "__main__":
    evaluate_all()
