"""
Unit Tests: Model Training & Prediction (Task 5 — CI/CD)
Tests run during CI to verify model pipeline works.
"""
import numpy as np
import pytest
from sklearn.datasets import make_classification
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


@pytest.fixture
def sample_classification_data():
    """Generate synthetic fraud-like classification data."""
    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=10,
        n_classes=2, weights=[0.95, 0.05], random_state=42
    )
    return X, y


def test_xgboost_trains_and_predicts(sample_classification_data):
    """XGBoost model should train and produce valid predictions."""
    X, y = sample_classification_data
    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X[:400], y[:400])
    preds = model.predict(X[400:])
    assert all(p in [0, 1] for p in preds), "Predictions must be 0 or 1"
    assert len(preds) == 100


def test_lightgbm_trains_and_predicts(sample_classification_data):
    """LightGBM model should train and produce valid predictions."""
    X, y = sample_classification_data
    model = LGBMClassifier(verbose=-1, random_state=42)
    model.fit(X[:400], y[:400])
    preds = model.predict(X[400:])
    assert all(p in [0, 1] for p in preds)


def test_hybrid_pipeline_trains(sample_classification_data):
    """Hybrid RF+SelectFromModel pipeline should train successfully."""
    X, y = sample_classification_data
    pipeline = Pipeline([
        ("feature_selection", SelectFromModel(RandomForestClassifier(n_estimators=10, random_state=42))),
        ("classification", RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    pipeline.fit(X[:400], y[:400])
    preds = pipeline.predict(X[400:])
    assert len(preds) == 100


def test_auc_above_random(sample_classification_data):
    """Model AUC should be better than random (0.5)."""
    X, y = sample_classification_data
    model = XGBClassifier(eval_metric="logloss", random_state=42)
    model.fit(X[:400], y[:400])
    proba = model.predict_proba(X[400:])[:, 1]
    auc = roc_auc_score(y[400:], proba)
    assert auc > 0.5, f"AUC {auc} is not better than random"
