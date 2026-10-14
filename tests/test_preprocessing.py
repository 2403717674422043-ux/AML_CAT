"""
Unit Tests for Preprocessing and Schema Validation.
"""

import os
import sys
import json
import pytest
import pandas as pd
import numpy as np

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import LABEL_MAPPING, INVERSE_LABEL_MAPPING, RANDOM_SEED, load_processed_splits
from api.schemas import REQUIRED_FEATURES

def test_label_mapping():
    """Verify standard binary class target mapping."""
    assert LABEL_MAPPING["BENIGN"] == 0
    assert LABEL_MAPPING["DDoS"] == 1
    assert INVERSE_LABEL_MAPPING[0] == "BENIGN"
    assert INVERSE_LABEL_MAPPING[1] == "DDoS"

def test_processed_splits_schema():
    """Verify processed splits exist, have no non-finite values, and match the 68 feature schema."""
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_processed_splits()

    assert len(feature_names) == 68
    assert feature_names == REQUIRED_FEATURES
    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    # Verify no infinities or nulls in processed data
    assert not np.isinf(X_train.values).any()
    assert not np.isnan(X_train.values).any()
    assert not np.isinf(X_test.values).any()
    assert not np.isnan(X_test.values).any()

    # Verify stratified distribution
    train_ddos_ratio = (y_train == 1).mean()
    test_ddos_ratio = (y_test == 1).mean()
    assert abs(train_ddos_ratio - test_ddos_ratio) < 0.01

def test_identifier_leakage_exclusion():
    """Verify that sensitive identifier columns are strictly excluded."""
    from src.data_loader import IDENTIFIER_COLUMNS
    for ident in IDENTIFIER_COLUMNS:
        assert ident not in REQUIRED_FEATURES
