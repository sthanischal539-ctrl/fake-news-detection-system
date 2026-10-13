"""Command-line launcher and entrypoint for the Fake News Detection project.

Provides a unified interface for training, evaluation, predictions, running tests,
and launching the Streamlit web dashboard.

Usage:
    python start.py --train               # Train classical models (LogReg, Naive Bayes, Linear SVM)
    python start.py --train-transformer   # Train classical models + fine-tune DistilBERT
    python start.py --eval                # Evaluate models on held-out test set
    python start.py --app                 # Launch Streamlit web application & dashboard
    python start.py --test                # Run automated pytest suite
    python start.py --predict "..."       # Run prediction on sample text
    python start.py --all                 # Run full classical pipeline: train -> eval -> test
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> int:
    """Execute a subprocess command using the current Python environment."""
    print(f"\n>> Running: {' '.join(cmd)}\n")
    return subprocess.call(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fake News Detection System - Quick Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train", action="store_true", help="Train classical ML models.")
    parser.add_argument(
        "--train-transformer",
        action="store_true",
        help="Train classical models + fine-tune DistilBERT sequence classifier.",
    )
    parser.add_argument("--eval", action="store_true", help="Evaluate models and save metrics.")
    parser.add_argument("--app", action="store_true", help="Launch the Streamlit web application.")
    parser.add_argument("--test", action="store_true", help="Run automated test suite via pytest.")
    parser.add_argument("--predict", type=str, help="Predict on a supplied text string.")
    parser.add_argument("--model", type=str, default="Logistic Regression", help="Model for prediction.")
    parser.add_argument("--all", action="store_true", help="Run train, evaluate, and test in sequence.")
    parser.add_argument("--info", action="store_true", help="Print system and dataset status.")

    args = parser.parse_args()

    python_exe = sys.executable

    # If no flags passed, show status and help
    if not any([args.train, args.train_transformer, args.eval, args.app, args.test, args.predict, args.all, args.info]):
        print("\n========================================================")
        print("📰 FAKE NEWS DETECTION SYSTEM (NLP)")
        print("========================================================")
        print(f"Python interpreter: {python_exe}")

        from src.data_loader import get_dataset_info
        from src.predict import get_available_models

        info = get_dataset_info()
        print(f"Active dataset:     {info.get('source', 'None')}")
        print(f"Total articles:     {info.get('total_records', 0)}")

        models = get_available_models()
        print("Trained models:    ", ", ".join(models.keys()) if models else "None yet")
        print("\nQuick Commands:")
        print("  python start.py --train              # Train classical ML models")
        print("  python start.py --eval               # Evaluate trained models")
        print("  python start.py --app                # Start web interface & dashboard")
        print("  python start.py --test               # Run test suite")
        print("  python start.py --train-transformer  # Train classical + DistilBERT")
        print("  python start.py --all                # Train + Eval + Test")
        print("========================================================\n")
        return

    if args.info:
        from src.data_loader import get_dataset_info
        from src.predict import get_available_models

        print("\nProject Status:")
        info = get_dataset_info()
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("\nAvailable Models:")
        for name, p in get_available_models().items():
            print(f"  - {name}: {p}")
        return

    if args.all:
        code = run_command([python_exe, "-m", "src.train"])
        if code != 0:
            sys.exit(code)
        code = run_command([python_exe, "-m", "src.evaluate"])
        if code != 0:
            sys.exit(code)
        code = run_command([python_exe, "-m", "pytest", "tests/"])
        sys.exit(code)

    if args.train:
        sys.exit(run_command([python_exe, "-m", "src.train"]))

    if args.train_transformer:
        sys.exit(run_command([python_exe, "-m", "src.train", "--with-transformer"]))

    if args.eval:
        sys.exit(run_command([python_exe, "-m", "src.evaluate"]))

    if args.app:
        sys.exit(run_command([python_exe, "-m", "streamlit", "run", "src/app.py"]))

    if args.test:
        sys.exit(run_command([python_exe, "-m", "pytest", "tests/", "-v"]))

    if args.predict:
        sys.exit(run_command([python_exe, "-m", "src.predict", "--text", args.predict, "--model", args.model]))


if __name__ == "__main__":
    main()
