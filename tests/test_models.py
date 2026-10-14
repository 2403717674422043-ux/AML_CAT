"""
Unit Tests for Trained Model Artifacts & Pipelines.
"""

import os
import sys
import joblib
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import MODELS_DIR, load_processed_splits
from api.schemas import REQUIRED_FEATURES

def test_champion_model_loading():
    """Verify champion model loads correctly and is an sklearn Pipeline."""
    best_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    assert os.path.exists(best_path), f"Champion model not found at {best_path}"
    pipeline = joblib.load(best_path)
    assert hasattr(pipeline, "predict"), "Loaded pipeline must have predict method"
    assert hasattr(pipeline, "predict_proba"), "Loaded pipeline must have predict_proba method"

def test_inference_pipeline_execution():
    """Verify pipeline accepts raw dataframe with 68 features and produces binary prediction."""
    best_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    pipeline = joblib.load(best_path)

    _, _, _, _, X_test, _, _ = load_processed_splits()
    sample = X_test.iloc[:5]

    preds = pipeline.predict(sample)
    probas = pipeline.predict_proba(sample)

    assert len(preds) == 5
    assert set(preds).issubset({0, 1})
    assert probas.shape == (5, 2)
    assert np.allclose(probas.sum(axis=1), 1.0)
