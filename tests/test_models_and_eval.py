"""Unit tests for classical models, evaluation metrics, and figure generation."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src import config
from src.evaluate import _metrics


@pytest.fixture
def sample_dataset():
    """Small fixture dataset for fast deterministic testing without full dataset training."""
    X_train = [
        "official government statement on economic growth and treasury reports",
        "peer reviewed scientific findings confirm medical trial results",
        "spokesperson confirmed committee schedule for public infrastructure project",
        "shocking secret plot revealed by whistleblowers before it gets deleted",
        "miracle natural cure hidden by big corporations exposed at last",
        "unbelievable conspiracy the corrupt elites do not want you to know",
    ]
    y_train = np.array([0, 0, 0, 1, 1, 1])

    X_test = [
        "authorities released monthly inflation data on tuesday morning",
        "secret elite conspiracy miracle trick they tried to hide",
    ]
    y_test = np.array([0, 1])

    vec = TfidfVectorizer(min_df=1)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    return {
        "vec": vec,
        "X_train": X_train_vec,
        "y_train": y_train,
        "X_test": X_test_vec,
        "y_test": y_test,
    }


def test_classical_models_train_and_predict(sample_dataset):
    """Verify Logistic Regression, Naive Bayes, and Calibrated Linear SVM train and predict."""
    X_train = sample_dataset["X_train"]
    y_train = sample_dataset["y_train"]
    X_test = sample_dataset["X_test"]
    y_test = sample_dataset["y_test"]

    # 1. Logistic Regression
    lr = LogisticRegression(random_state=42)
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    assert len(preds_lr) == len(y_test)
    assert hasattr(lr, "predict_proba")

    # 2. Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    preds_nb = nb.predict(X_test)
    assert len(preds_nb) == len(y_test)
    assert hasattr(nb, "predict_proba")

    # 3. Linear SVM with Calibration
    base_svm = LinearSVC(random_state=42)
    svm = CalibratedClassifierCV(base_svm, cv=2)
    svm.fit(X_train, y_train)
    preds_svm = svm.predict(X_test)
    assert len(preds_svm) == len(y_test)
    assert hasattr(svm, "predict_proba")


def test_metrics_calculation():
    """Verify metric calculation produces correct values for accuracy, precision, recall, f1, and roc_auc."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])

    m = _metrics(y_true, y_pred, y_prob)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert "classification_report" in m

    # Test with imperfect predictions
    y_imperfect = np.array([0, 1, 1, 0])
    m_imp = _metrics(y_true, y_imperfect, np.array([0.2, 0.7, 0.6, 0.4]))
    assert m_imp["accuracy"] == 0.5
    assert m_imp["precision"] == 0.5
    assert m_imp["recall"] == 0.5
