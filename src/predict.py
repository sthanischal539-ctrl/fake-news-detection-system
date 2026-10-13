"""Unified prediction pipeline for all trained models.

Supports:
  - Classical models: Logistic Regression, Naive Bayes, Linear SVM
  - Transformer: DistilBERT sequence classifier
  - Multi-model comparison across all available models

Consistent preprocessing, robust input validation, probability calibration,
agreement analysis, and transparent prediction outputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from . import config, features, preprocessing

DISCLAIMER = (
    "A prediction from this system is a statistical, probability-based signal, "
    "not proof of factual truth. It analyses linguistic style and vocabulary patterns, "
    "not factual accuracy. Model agreement is not proof of factual accuracy. "
    "Always verify claims against authoritative, trusted news sources."
)

# In-memory cache for transformer model to avoid slow reloads on repeated calls
_TRANSFORMER_CACHE: dict[str, Any] = {"tokenizer": None, "model": None, "model_dir": None}


def get_available_models() -> dict[str, Path]:
    """Return a dictionary of model names mapped to their saved artefact locations."""
    available = {}

    # Classical models depend on the TF-IDF vectorizer
    if config.TFIDF_PATH.exists():
        for name, path in config.CLASSICAL_MODELS.items():
            if path.exists():
                available[name] = path

    # DistilBERT
    if config.TRANSFORMER_DIR.exists() and (config.TRANSFORMER_DIR / "config.json").exists():
        weights_exist = (
            (config.TRANSFORMER_DIR / "model.safetensors").exists()
            or (config.TRANSFORMER_DIR / "pytorch_model.bin").exists()
        )
        if weights_exist:
            available["DistilBERT"] = config.TRANSFORMER_DIR

    return available


def validate_input_text(text: str, min_chars: int = 20, max_chars: int = 25000) -> tuple[bool, str, str]:
    """Validate and sanitize user input text.

    Returns:
        (is_valid, message, sanitized_text)
    """
    if text is None:
        return False, "Input text cannot be None.", ""

    sanitized = str(text).strip()

    if not sanitized:
        return False, "Input text is empty. Please enter an article body or headline.", ""

    if len(sanitized) < min_chars:
        return (
            False,
            f"Input text is too short ({len(sanitized)} characters). Please enter at least {min_chars} characters.",
            sanitized,
        )

    warning = ""
    if len(sanitized) > max_chars:
        warning = f"Input exceeds {max_chars} characters; it has been truncated for analysis."
        sanitized = sanitized[:max_chars]

    return True, warning, sanitized


def predict_classical(text: str, model_name: str, model_path: Path) -> dict:
    """Run prediction using a saved classical model (TF-IDF + scikit-learn)."""
    vectorizer = features.load_vectorizer()
    clean_text = preprocessing.preprocess(text)

    if not clean_text:
        # If text had no vocabulary left after stopword/cleaning, fallback to light clean
        clean_text = preprocessing.light_clean(text).lower()

    X = vectorizer.transform([clean_text])
    model = joblib.load(model_path)

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            proba = None
    elif hasattr(model, "decision_function"):
        try:
            score = model.decision_function(X)[0]
            # Calibrate via sigmoid for binary classification
            prob_fake = 1.0 / (1.0 + np.exp(-score))
            proba = np.array([1.0 - prob_fake, prob_fake])
        except Exception:
            proba = None

    if proba is not None:
        label_idx = int(np.argmax(proba))
        confidence = float(proba[label_idx])
        probabilities = {
            "REAL": float(proba[config.LABEL_REAL]),
            "FAKE": float(proba[config.LABEL_FAKE]),
        }
    else:
        # For models without calibrated probabilities
        pred = model.predict(X)[0]
        label_idx = int(pred)
        confidence = None  # Do not invent arbitrary confidence
        probabilities = None

    label_str = config.LABELS.get(label_idx, "UNKNOWN")

    return {
        "model": model_name,
        "label": label_str,
        "label_idx": label_idx,
        "confidence": confidence,
        "confidence_str": f"{confidence * 100:.1f}%" if confidence is not None else "Confidence unavailable",
        "probabilities": probabilities,
        "preprocessed_text": clean_text,
        "disclaimer": DISCLAIMER,
    }


def predict_transformer(text: str, model_dir: Path) -> dict:
    """Run prediction using fine-tuned DistilBERT (cached in memory for speed)."""
    global _TRANSFORMER_CACHE

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as err:
        raise RuntimeError(f"PyTorch or Transformers not installed: {err}")

    clean_text = preprocessing.light_clean(text)

    # Use cached model/tokenizer if available
    if (
        _TRANSFORMER_CACHE["tokenizer"] is None
        or _TRANSFORMER_CACHE["model_dir"] != str(model_dir)
    ):
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        _TRANSFORMER_CACHE["tokenizer"] = tokenizer
        _TRANSFORMER_CACHE["model"] = model
        _TRANSFORMER_CACHE["model_dir"] = str(model_dir)
    else:
        tokenizer = _TRANSFORMER_CACHE["tokenizer"]
        model = _TRANSFORMER_CACHE["model"]

    inputs = tokenizer(
        clean_text,
        truncation=True,
        padding="max_length",
        max_length=config.TRANSFORMER_MAX_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

    label_idx = int(np.argmax(probs))
    confidence = float(probs[label_idx])
    label_str = config.LABELS.get(label_idx, "UNKNOWN")

    return {
        "model": "DistilBERT",
        "label": label_str,
        "label_idx": label_idx,
        "confidence": confidence,
        "confidence_str": f"{confidence * 100:.1f}%",
        "probabilities": {
            "REAL": float(probs[config.LABEL_REAL]),
            "FAKE": float(probs[config.LABEL_FAKE]),
        },
        "preprocessed_text": clean_text,
        "disclaimer": DISCLAIMER,
    }


def predict(text: str, model_name: str = "Logistic Regression") -> dict:
    """Validate input, load selected model, and return prediction result."""
    is_valid, msg, sanitized_text = validate_input_text(text)
    if not is_valid:
        raise ValueError(msg)

    available = get_available_models()
    if not available:
        raise FileNotFoundError(
            "No trained models are currently available. "
            "Please run 'python start.py --train' to train the classical models."
        )

    if model_name not in available:
        available_names = ", ".join(f"'{k}'" for k in available.keys())
        raise ValueError(
            f"Model '{model_name}' is not available. Available trained models: {available_names}"
        )

    if model_name == "DistilBERT":
        result = predict_transformer(sanitized_text, available["DistilBERT"])
    else:
        result = predict_classical(sanitized_text, model_name, available[model_name])

    if msg:  # If there was a length truncation warning
        result["warning"] = msg

    return result


def predict_all_models(text: str) -> dict:
    """Run an article through ALL available, successfully loaded models and compare predictions.

    Returns:
        Structured dictionary containing:
          - 'results': List of per-model predictions
          - 'model_count': Number of available models analyzed
          - 'consensus_label': 'REAL', 'FAKE', 'TIE', or 'N/A'
          - 'is_unanimous': bool
          - 'has_disagreement': bool
          - 'agreement_summary': Human-readable agreement string
          - 'disagreement_warning': Alert message if models disagree
          - 'probabilities_available': bool
          - 'disclaimer': Ethical disclaimer
    """
    is_valid, msg, sanitized_text = validate_input_text(text)
    if not is_valid:
        raise ValueError(msg)

    available = get_available_models()
    if not available:
        raise FileNotFoundError(
            "No trained models are available for prediction. "
            "Run 'python start.py --train' to train the models."
        )

    results: list[dict] = []
    real_votes = 0
    fake_votes = 0

    for name, path in available.items():
        try:
            if name == "DistilBERT":
                pred = predict_transformer(sanitized_text, path)
            else:
                pred = predict_classical(sanitized_text, name, path)
            pred["status"] = "success"
            results.append(pred)

            if pred["label"] == "REAL":
                real_votes += 1
            elif pred["label"] == "FAKE":
                fake_votes += 1
        except Exception as exc:
            results.append({
                "model": name,
                "status": "error",
                "label": "ERROR",
                "confidence": None,
                "confidence_str": "Error",
                "probabilities": None,
                "error": str(exc),
            })

    successful_results = [r for r in results if r.get("status") == "success"]
    total_valid = len(successful_results)

    consensus_label = "N/A"
    is_unanimous = False
    has_disagreement = False
    disagreement_warning = None

    if total_valid == 0:
        agreement_summary = "No models were able to produce a prediction."
    elif total_valid == 1:
        consensus_label = successful_results[0]["label"]
        agreement_summary = f"Only 1 model was available ({successful_results[0]['model']}). Multi-model comparison is not possible."
    else:
        if real_votes == total_valid:
            consensus_label = "REAL"
            is_unanimous = True
            agreement_summary = f"All {total_valid} available models agree on classification: REAL."
        elif fake_votes == total_valid:
            consensus_label = "FAKE"
            is_unanimous = True
            agreement_summary = f"All {total_valid} available models agree on classification: FAKE."
        elif real_votes > fake_votes:
            consensus_label = "REAL"
            has_disagreement = True
            agreement_summary = f"Models disagree: {real_votes} predicted REAL, while {fake_votes} predicted FAKE."
            disagreement_warning = (
                f"⚠️ Model Disagreement: {fake_votes} of {total_valid} models flagged this article as FAKE. "
                "The majority classified it as REAL, but stylistic ambiguity exists across different model architectures."
            )
        elif fake_votes > real_votes:
            consensus_label = "FAKE"
            has_disagreement = True
            agreement_summary = f"Models disagree: {fake_votes} predicted FAKE, while {real_votes} predicted REAL."
            disagreement_warning = (
                f"⚠️ Model Disagreement: {real_votes} of {total_valid} models classified this article as REAL. "
                "The majority flagged it as FAKE. Inspect individual probability distributions below."
            )
        else:
            consensus_label = "TIE"
            has_disagreement = True
            agreement_summary = f"Models are evenly split ({real_votes} predicted REAL, {fake_votes} predicted FAKE)."
            disagreement_warning = (
                "⚠️ Complete Disagreement: Available models are split 50/50. "
                "The linguistic style of this article contains conflicting signals between classical and transformer features."
            )

    # Attach consensus alignment to each result
    for r in successful_results:
        if consensus_label in ("REAL", "FAKE"):
            r["agrees_with_consensus"] = (r["label"] == consensus_label)
        else:
            r["agrees_with_consensus"] = False

    return {
        "results": results,
        "model_count": total_valid,
        "real_votes": real_votes,
        "fake_votes": fake_votes,
        "consensus_label": consensus_label,
        "is_unanimous": is_unanimous,
        "has_disagreement": has_disagreement,
        "agreement_summary": agreement_summary,
        "disagreement_warning": disagreement_warning,
        "sanitized_text": sanitized_text,
        "warning": msg if msg else None,
        "disclaimer": DISCLAIMER,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether a news article is Fake or Real.")
    parser.add_argument("--text", type=str, help="Raw text of the article to analyse.")
    parser.add_argument("--file", type=str, help="Path to a text (.txt) or PDF (.pdf) file containing the article.")
    parser.add_argument(
        "--model",
        type=str,
        default="Logistic Regression",
        help="Specific model to use ('Logistic Regression', 'Naive Bayes', 'Linear SVM', or 'DistilBERT').",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Run article through ALL available models and report comparison/agreement.",
    )
    parser.add_argument("--list-models", action="store_true", help="List all available trained models.")

    args = parser.parse_args()

    available = get_available_models()

    if args.list_models:
        print("\nTrained models currently available:")
        if not available:
            print("  (None found. Run 'python start.py --train' to train models.)")
        else:
            for name, path in available.items():
                print(f"  - {name} ({path})")
        return

    article_text = args.text
    if args.file:
        from .file_extractor import extract_text_from_file

        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File not found at {file_path}", file=sys.stderr)
            sys.exit(1)
        ok, text, msg = extract_text_from_file(file_path, file_path.name)
        if not ok:
            print(f"File extraction failed: {msg}", file=sys.stderr)
            sys.exit(1)
        article_text = text
        print(f"File loaded: {msg}")

    if not article_text:
        print("Please provide text using --text '...' or --file path/to/file.(txt|pdf)", file=sys.stderr)
        print("Run with --help for usage information.")
        sys.exit(1)

    try:
        if args.all_models:
            res_all = predict_all_models(article_text)
            print("\n================ MULTI-MODEL COMPARISON ================")
            print(f"Agreement Summary: {res_all['agreement_summary']}")
            if res_all["disagreement_warning"]:
                print(f"Warning:           {res_all['disagreement_warning']}")
            print("--------------------------------------------------------")
            print(f"{'Model':<22} | {'Prediction':<10} | {'Confidence':<14} | {'Agrees?'}")
            print("-" * 60)
            for r in res_all["results"]:
                if r.get("status") == "success":
                    agrees = "Yes" if r.get("agrees_with_consensus") else "No"
                    print(f"{r['model']:<22} | {r['label']:<10} | {r['confidence_str']:<14} | {agrees}")
                else:
                    print(f"{r['model']:<22} | {'ERROR':<10} | {'N/A':<14} | N/A")
            print("========================================================\n")
        else:
            res = predict(article_text, model_name=args.model)
            print("\n================ PREDICTION RESULT ================")
            print(f"Model:         {res['model']}")
            print(f"Prediction:    {res['label']}")
            print(f"Confidence:    {res['confidence_str']}")
            if res.get("probabilities"):
                print(f"Probabilities: REAL={res['probabilities']['REAL']*100:.1f}%, FAKE={res['probabilities']['FAKE']*100:.1f}%")
            if "warning" in res and res["warning"]:
                print(f"Note:          {res['warning']}")
            print("---------------------------------------------------")
            print(f"Disclaimer:    {res['disclaimer']}")
            print("===================================================\n")
    except Exception as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
