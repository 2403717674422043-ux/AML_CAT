"""
DDoSGuard - Model Training & Hyperparameter Tuning Module
Trains:
1. Baseline Model: Random Forest Classifier
2. Advanced Model: XGBoost Classifier (with RandomizedSearchCV tuning)
Evaluates on the validation set (test set remains strictly untouched).
Saves individual pipelines and the selected champion model to models/
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier
    XGBOOST_AVAILABLE = False

from src.preprocessing import load_processed_splits, RANDOM_SEED, MODELS_DIR, METRICS_DIR
from src.experiment_logger import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DDoSGuard.Train")

def measure_inference_latency(pipeline: Any, X_sample: pd.DataFrame, num_repeats: int = 200) -> float:
    """
    Measures average single-record inference latency in milliseconds over multiple repeats.
    """
    single_record = X_sample.iloc[[0]].copy()
    # Warm-up
    for _ in range(10):
        pipeline.predict(single_record)

    start_time = time.perf_counter()
    for _ in range(num_repeats):
        pipeline.predict(single_record)
    total_time = time.perf_counter() - start_time
    latency_ms = (total_time / num_repeats) * 1000.0
    return latency_ms

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 5),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 5),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 5) if y_proba is not None else None,
        "pr_auc": round(float(average_precision_score(y_true, y_proba)), 5) if y_proba is not None else None,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist()
    }

def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, preprocessor_info: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains Baseline Random Forest Classifier and evaluates on Validation set.
    """
    logger.info("=== Training Baseline: Random Forest Classifier ===")
    rf_params = {
        "n_estimators": 100,
        "max_depth": 16,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    }

    rf_clf = RandomForestClassifier(**rf_params)
    
    # Bundle imputer and classifier into reproducible pipeline
    imputer = preprocessor_info.get("imputer", SimpleImputer(strategy="median"))
    pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", rf_clf)
    ])

    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_train
    logger.info("Random Forest trained in %.2f seconds.", train_train_duration := train_duration)

    # Validation evaluation
    val_pred = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val.values, val_pred, val_proba)
    logger.info("Random Forest Val Metrics: F1=%.5f, Recall=%.5f, Precision=%.5f, PR-AUC=%.5f",
                val_metrics["f1_score"], val_metrics["recall"], val_metrics["precision"], val_metrics["pr_auc"])

    # Latency measurement
    latency_ms = measure_inference_latency(pipeline, X_val)
    logger.info("Random Forest single-record latency: %.4f ms", latency_ms)

    # Save artifact
    rf_path = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
    joblib.dump(pipeline, rf_path)
    logger.info("Saved Random Forest pipeline to %s", rf_path)

    # Log experiment
    tracker = ExperimentTracker()
    tracker.log_experiment(
        model_name="Random Forest",
        hyperparameters=rf_params,
        val_metrics=val_metrics,
        training_duration_sec=train_duration,
        inference_latency_ms=latency_ms,
        artifact_path=rf_path,
        notes="Baseline ensemble model using class-weighted balanced Random Forest."
    )

    return pipeline, val_metrics

def train_advanced(X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series, preprocessor_info: Dict[str, Any]) -> Tuple[Pipeline, Dict[str, Any]]:
    """
    Trains and tunes Advanced Model: XGBoost Classifier.
    """
    logger.info("=== Training Advanced Model: XGBoost Classifier ===")
    imputer = preprocessor_info.get("imputer", SimpleImputer(strategy="median"))

    if XGBOOST_AVAILABLE:
        # Scale_pos_weight for imbalance adjustment
        scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
        param_dist = {
            "n_estimators": [100, 150, 200],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0]
        }

        base_xgb = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=-1
        )

        # Tune with RandomizedSearchCV on a stratified sample of 30,000 for rapid convergence
        tune_sample_size = min(30000, len(X_train))
        X_sample, _, y_sample, _ = RandomizedSearchCV._sample_without_replacement = (
            X_train.sample(n=tune_sample_size, random_state=RANDOM_SEED),
            None,
            y_train.sample(n=tune_sample_size, random_state=RANDOM_SEED),
            None
        ) if False else (
            # proper stratified sample
            pd.concat([
                X_train[y_train == 0].sample(n=int(tune_sample_size * (y_train == 0).mean()), random_state=RANDOM_SEED),
                X_train[y_train == 1].sample(n=int(tune_sample_size * (y_train == 1).mean()), random_state=RANDOM_SEED)
            ]),
            None,
            pd.concat([
                y_train[y_train == 0].sample(n=int(tune_sample_size * (y_train == 0).mean()), random_state=RANDOM_SEED),
                y_train[y_train == 1].sample(n=int(tune_sample_size * (y_train == 1).mean()), random_state=RANDOM_SEED)
            ]),
            None
        )

        logger.info("Running hyperparameter search with 4 iterations, 3 folds on stratified subset (%d samples)...", len(X_sample))
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        search = RandomizedSearchCV(
            base_xgb,
            param_distributions=param_dist,
            n_iter=4,
            scoring="f1",
            cv=cv,
            random_state=RANDOM_SEED,
            n_jobs=-1
        )
        search.fit(X_sample, y_sample)
        best_params = search.best_params_
        logger.info("Best XGBoost hyperparameters found: %s", best_params)

        final_clf = xgb.XGBClassifier(
            **best_params,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos,
            random_state=RANDOM_SEED,
            tree_method="hist",
            n_jobs=-1
        )
    else:
        logger.warning("XGBoost not available; falling back to HistGradientBoostingClassifier.")
        best_params = {"max_iter": 150, "max_depth": 8, "learning_rate": 0.1}
        final_clf = HistGradientBoostingClassifier(**best_params, random_state=RANDOM_SEED)

    pipeline = Pipeline([
        ("imputer", imputer),
        ("classifier", final_clf)
    ])

    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_train
    logger.info("Advanced model trained in %.2f seconds.", train_duration)

    # Validation evaluation
    val_pred = pipeline.predict(X_val)
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    val_metrics = compute_metrics(y_val.values, val_pred, val_proba)
    logger.info("XGBoost Val Metrics: F1=%.5f, Recall=%.5f, Precision=%.5f, PR-AUC=%.5f",
                val_metrics["f1_score"], val_metrics["recall"], val_metrics["precision"], val_metrics["pr_auc"])

    # Latency measurement
    latency_ms = measure_inference_latency(pipeline, X_val)
    logger.info("XGBoost single-record latency: %.4f ms", latency_ms)

    # Save artifact
    xgb_path = os.path.join(MODELS_DIR, "xgboost_pipeline.joblib")
    joblib.dump(pipeline, xgb_path)
    logger.info("Saved XGBoost pipeline to %s", xgb_path)

    # Log experiment
    tracker = ExperimentTracker()
    tracker.log_experiment(
        model_name="XGBoost",
        hyperparameters=best_params,
        val_metrics=val_metrics,
        training_duration_sec=train_duration,
        inference_latency_ms=latency_ms,
        artifact_path=xgb_path,
        notes="Advanced gradient boosting model with tree_method='hist' and scale_pos_weight."
    )

    return pipeline, val_metrics

def select_and_save_champion_model(rf_metrics: Dict[str, Any], xgb_metrics: Dict[str, Any]):
    """
    Selects the champion model based on validation F1-score, Recall, and PR-AUC.
    Saves it as models/best_model_pipeline.joblib and writes decision to artifacts/metrics/.
    """
    logger.info("=== Selecting Champion Model Based on Validation Performance ===")
    rf_score = (rf_metrics["f1_score"] * 0.4) + (rf_metrics["recall"] * 0.3) + (rf_metrics["pr_auc"] * 0.3)
    xgb_score = (xgb_metrics["f1_score"] * 0.4) + (xgb_metrics["recall"] * 0.3) + (xgb_metrics["pr_auc"] * 0.3)

    logger.info("Validation Composite Score -> Random Forest: %.5f | XGBoost: %.5f", rf_score, xgb_score)

    if xgb_score >= rf_score:
        champion_name = "XGBoost"
        source_path = os.path.join(MODELS_DIR, "xgboost_pipeline.joblib")
        winner_metrics = xgb_metrics
    else:
        champion_name = "Random Forest"
        source_path = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
        winner_metrics = rf_metrics

    target_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    pipeline = joblib.load(source_path)
    joblib.dump(pipeline, target_path)
    logger.info("Champion Model Selected: %s. Saved as %s", champion_name, target_path)

    decision_summary = {
        "selected_model": champion_name,
        "selection_criterion": "Validation Composite Score = 0.4*F1 + 0.3*Recall + 0.3*PR-AUC (balancing false negative penalty against precision)",
        "rf_composite_score": round(rf_score, 5),
        "xgb_composite_score": round(xgb_score, 5),
        "random_forest_val": rf_metrics,
        "xgboost_val": xgb_metrics,
        "champion_artifact": target_path
    }

    summary_file = os.path.join(METRICS_DIR, "model_selection.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(decision_summary, f, indent=2)
    logger.info("Model selection report saved to %s", summary_file)

def train_all():
    """Main training workflow."""
    X_train, y_train, X_val, y_val, _, _, feature_names = load_processed_splits()
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    preprocessor_info = joblib.load(preprocessor_path)

    # 1. Train Baseline
    rf_pipeline, rf_val_metrics = train_baseline(X_train, y_train, X_val, y_val, preprocessor_info)

    # 2. Train Advanced
    xgb_pipeline, xgb_val_metrics = train_advanced(X_train, y_train, X_val, y_val, preprocessor_info)

    # 3. Model Selection
    select_and_save_champion_model(rf_val_metrics, xgb_val_metrics)
    logger.info("Training stage completed successfully!")

if __name__ == "__main__":
    train_all()
