"""
DDoSGuard - Preprocessing Pipeline
Leakage-safe preprocessing:
- Cleans column names & handles target mapping (BENIGN: 0, DDoS: 1)
- Drops exact duplicates
- Removes identifiers (Flow ID, Source IP, Destination IP, Timestamp, Source Port)
- Removes redundant and zero-variance/constant features
- Replaces +/- Inf with NaN
- Performs 70/15/15 Stratified train/val/test split with fixed random seed
- Fits imputation and scaling strictly on training set
- Saves preprocessing metadata and processed artifacts
"""

import os
import sys
import json
import logging
from typing import Tuple, Dict, Any, List

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import joblib

from src.data_loader import load_raw_dataset, IDENTIFIER_COLUMNS, REDUNDANT_COLUMNS

logger = logging.getLogger("DDoSGuard.Preprocessing")

PROCESSED_DATA_DIR = os.path.join("data", "processed")
MODELS_DIR = "models"
METRICS_DIR = os.path.join("artifacts", "metrics")

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

LABEL_MAPPING = {
    "BENIGN": 0,
    "DDoS": 1
}
INVERSE_LABEL_MAPPING = {0: "BENIGN", 1: "DDoS"}

def prepare_data(
    file_path: str = None,
    save_splits: bool = True
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str], Dict[str, Any]]:
    """
    Executes full leakage-free preprocessing pipeline.
    Returns:
      X_train, y_train, X_val, y_val, X_test, y_test, feature_names, metadata
    """
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    df, column_mapping = load_raw_dataset(file_path) if file_path else load_raw_dataset()
    initial_rows = len(df)
    initial_cols = len(df.columns)

    # 1. Target column extraction & validation
    if "Label" not in df.columns:
        raise ValueError("Target column 'Label' not found in dataset.")
    
    # Clean label strings
    df["Label"] = df["Label"].astype(str).str.strip()
    valid_labels = set(LABEL_MAPPING.keys())
    present_labels = set(df["Label"].unique())
    unknown_labels = present_labels - valid_labels
    if unknown_labels:
        logger.warning("Found unexpected labels: %s. Filtering to binary DDoS vs BENIGN.", unknown_labels)
        df = df[df["Label"].isin(valid_labels)].copy()

    # 2. Duplicate record analysis and removal
    num_duplicates = int(df.duplicated().sum())
    logger.info("Found %d duplicate rows. Removing duplicates...", num_duplicates)
    df = df.drop_duplicates().copy()
    rows_after_dedup = len(df)

    # 3. Separate features and target
    y_raw = df["Label"]
    y = y_raw.map(LABEL_MAPPING).astype(int)

    # 4. Feature selection & exclusion of identifiers / redundant columns
    cols_to_drop = []
    for col in IDENTIFIER_COLUMNS + REDUNDANT_COLUMNS:
        if col in df.columns:
            cols_to_drop.append(col)

    # Drop target as well
    cols_to_drop.append("Label")

    X = df.drop(columns=cols_to_drop)

    # 5. Identify and remove constant/zero-variance features on training or entire dataset
    # Note: To avoid data leakage, constant columns should strictly be evaluated on training split,
    # but columns known to have zero-variance globally (all 0s) are recorded.
    # Convert all remaining columns to numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # 6. Stratified Split (70% train, 15% val, 15% test)
    logger.info("Splitting dataset: 70%% Train, 15%% Val, 15%% Test (stratified, seed=%d)", RANDOM_SEED)
    X_train_raw, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED, stratify=y
    )
    
    val_fraction_of_temp = VAL_RATIO / (VAL_RATIO + TEST_RATIO)  # 0.15 / 0.30 = 0.50
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_fraction_of_temp), random_state=RANDOM_SEED, stratify=y_temp
    )

    # 7. Identify constant features STRICTLY on training split
    train_nunique = X_train_raw.nunique()
    constant_features = train_nunique[train_nunique <= 1].index.tolist()
    logger.info("Identified %d constant features on training split: %s", len(constant_features), constant_features)

    # Drop constant features from all splits
    X_train_filtered = X_train_raw.drop(columns=constant_features)
    X_val_filtered = X_val_raw.drop(columns=constant_features)
    X_test_filtered = X_test_raw.drop(columns=constant_features)

    feature_names = list(X_train_filtered.columns)
    logger.info("Retained %d input features for ML modeling.", len(feature_names))

    # 8. Handling +/- Infinity & NaN
    # Replace infinities with NaN so SimpleImputer can impute median learned on train
    X_train_cleaned = X_train_filtered.replace([np.inf, -np.inf], np.nan)
    X_val_cleaned = X_val_filtered.replace([np.inf, -np.inf], np.nan)
    X_test_cleaned = X_test_filtered.replace([np.inf, -np.inf], np.nan)

    # Count infinities and NaNs in train for logging
    inf_nan_train = int(X_train_filtered.replace([np.inf, -np.inf], np.nan).isnull().sum().sum())
    logger.info("Total imputed values in training set: %d", inf_nan_train)

    # 9. Fit Preprocessing Pipeline (Imputer + Scaler) ONLY on Training set
    logger.info("Fitting SimpleImputer(median) and StandardScaler on training set...")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    # Fit pipeline
    X_train_imputed = imputer.fit_transform(X_train_cleaned)
    X_val_imputed = imputer.transform(X_val_cleaned)
    X_test_imputed = imputer.transform(X_test_cleaned)

    # Save fitted imputer and scaler
    preprocessor = {
        "imputer": imputer,
        "scaler": scaler.fit(X_train_imputed),  # fitted on train
        "feature_names": feature_names,
        "constant_features": constant_features,
        "dropped_identifiers": cols_to_drop,
        "label_mapping": LABEL_MAPPING,
        "inverse_label_mapping": INVERSE_LABEL_MAPPING
    }
    joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.joblib"))
    logger.info("Preprocessor saved to %s", os.path.join(MODELS_DIR, "preprocessor.joblib"))

    # Convert back to DataFrame for downstream convenience
    X_train = pd.DataFrame(X_train_imputed, columns=feature_names, index=X_train_cleaned.index)
    X_val = pd.DataFrame(X_val_imputed, columns=feature_names, index=X_val_cleaned.index)
    X_test = pd.DataFrame(X_test_imputed, columns=feature_names, index=X_test_cleaned.index)

    # 10. Compile metadata
    metadata = {
        "initial_rows": initial_rows,
        "initial_columns": initial_cols,
        "deduplicated_rows": rows_after_dedup,
        "removed_duplicates": num_duplicates,
        "excluded_identifiers": cols_to_drop,
        "constant_features_removed": constant_features,
        "num_model_features": len(feature_names),
        "model_features": feature_names,
        "label_mapping": LABEL_MAPPING,
        "split_strategy": "Stratified random split (70% train, 15% validation, 15% test)",
        "random_seed": RANDOM_SEED,
        "split_sizes": {
            "train": {
                "total": len(y_train),
                "benign": int((y_train == 0).sum()),
                "ddos": int((y_train == 1).sum()),
                "ddos_ratio": round(float((y_train == 1).mean()), 4)
            },
            "validation": {
                "total": len(y_val),
                "benign": int((y_val == 0).sum()),
                "ddos": int((y_val == 1).sum()),
                "ddos_ratio": round(float((y_val == 1).mean()), 4)
            },
            "test": {
                "total": len(y_test),
                "benign": int((y_test == 0).sum()),
                "ddos": int((y_test == 1).sum()),
                "ddos_ratio": round(float((y_test == 1).mean()), 4)
            }
        }
    }

    metadata_path = os.path.join(METRICS_DIR, "preprocessing_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Preprocessing metadata saved to %s", metadata_path)

    # 11. Optionally save processed splits to parquet for fast reloading
    if save_splits:
        logger.info("Saving preprocessed datasets to %s...", PROCESSED_DATA_DIR)
        train_df = X_train.copy()
        train_df["Label"] = y_train.values
        train_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "train.parquet"), index=False)

        val_df = X_val.copy()
        val_df["Label"] = y_val.values
        val_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "val.parquet"), index=False)

        test_df = X_test.copy()
        test_df["Label"] = y_test.values
        test_df.to_parquet(os.path.join(PROCESSED_DATA_DIR, "test.parquet"), index=False)
        logger.info("All processed splits successfully saved as Parquet.")

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names, metadata

def load_processed_splits() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, List[str]]:
    """
    Loads saved processed splits from disk if available, otherwise generates them.
    """
    train_path = os.path.join(PROCESSED_DATA_DIR, "train.parquet")
    val_path = os.path.join(PROCESSED_DATA_DIR, "val.parquet")
    test_path = os.path.join(PROCESSED_DATA_DIR, "test.parquet")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(test_path):
        logger.info("Loading cached processed splits from %s...", PROCESSED_DATA_DIR)
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
        test_df = pd.read_parquet(test_path)

        feature_names = [c for c in train_df.columns if c != "Label"]
        X_train, y_train = train_df[feature_names], train_df["Label"]
        X_val, y_val = val_df[feature_names], val_df["Label"]
        X_test, y_test = test_df[feature_names], test_df["Label"]
        return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

    logger.info("Cached processed splits not found. Preparing from raw dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names, _ = prepare_data()
    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names

if __name__ == "__main__":
    X_tr, y_tr, X_v, y_v, X_te, y_te, feats, meta = prepare_data()
    print("Preprocessing completed successfully!")
    print(f"Features count: {len(feats)}")
    print(f"Train shape: {X_tr.shape}, Val shape: {X_v.shape}, Test shape: {X_te.shape}")
