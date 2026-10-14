"""
DDoSGuard - Model Evaluation & Error Analysis Module
Evaluates trained models on the held-out test split (untouched during training & tuning).
Generates detailed metrics, operational error analysis, and latency profiling.
Saves results to artifacts/metrics/test_evaluation_report.json.
"""

import os
import sys
import json
import time
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report
)

from src.preprocessing import load_processed_splits, MODELS_DIR, METRICS_DIR, INVERSE_LABEL_MAPPING
from src.experiment_logger import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DDoSGuard.Evaluate")

def profile_latency_distribution(pipeline: Any, X_test: pd.DataFrame, num_samples: int = 1000) -> Dict[str, float]:
    """
    Profiles inference latency over num_samples individual single-record calls.
    Returns mean, median (p50), 95th percentile (p95), and 99th percentile (p99) in milliseconds.
    """
    sample_df = X_test.iloc[:num_samples]
    latencies = []
    
    # Warm-up
    for i in range(10):
        pipeline.predict(sample_df.iloc[[i]])

    for i in range(num_samples):
        row = sample_df.iloc[[i]]
        t0 = time.perf_counter()
        pipeline.predict(row)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

    lat_arr = np.array(latencies)
    return {
        "sample_size": num_samples,
        "mean_ms": round(float(np.mean(lat_arr)), 4),
        "median_p50_ms": round(float(np.median(lat_arr)), 4),
        "p95_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "p99_ms": round(float(np.percentile(lat_arr, 99)), 4),
        "methodology": "Single-record sequential predict calls with warm-up on held-out test set"
    }

def analyze_errors(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray, X_test: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs error analysis on False Positives and False Negatives,
    extracting feature anomalies for misclassified flows.
    """
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tp_mask = (y_true == 1) & (y_pred == 1)
    tn_mask = (y_true == 0) & (y_pred == 0)

    fp_count = int(fp_mask.sum())
    fn_count = int(fn_mask.sum())
    tp_count = int(tp_mask.sum())
    tn_count = int(tn_mask.sum())

    top_features = ["Flow Packets/s", "Flow Bytes/s", "Total Fwd Packets", "Flow Duration"]
    fp_feature_means = {}
    fn_feature_means = {}
    normal_feature_means = {}

    for f in top_features:
        if f in X_test.columns:
            normal_feature_means[f] = round(float(X_test.loc[tn_mask, f].mean()), 2)
            if fp_count > 0:
                fp_feature_means[f] = round(float(X_test.loc[fp_mask, f].mean()), 2)
            if fn_count > 0:
                fn_feature_means[f] = round(float(X_test.loc[fn_mask, f].mean()), 2)

    return {
        "false_positive_count": fp_count,
        "false_negative_count": fn_count,
        "true_positive_count": tp_count,
        "true_negative_count": tn_count,
        "false_positive_rate": round(float(fp_count / (fp_count + tn_count)) if (fp_count + tn_count) > 0 else 0.0, 5),
        "false_negative_rate": round(float(fn_count / (fn_count + tp_count)) if (fn_count + tp_count) > 0 else 0.0, 5),
        "feature_diagnostics": {
            "benign_baseline_means": normal_feature_means,
            "false_positive_means": fp_feature_means,
            "false_negative_means": fn_feature_means
        },
        "operational_impact": {
            "false_positives": "Benign enterprise flows falsely flagged as DDoS; may cause legitimate customer traffic dropping or unnecessary SOC alert triage.",
            "false_negatives": "Active DDoS attack flows that bypass detection; could lead to upstream service degradation or service downtime if mitigation is not triggered."
        }
    }

def evaluate_model_pipeline(name: str, pipeline_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates a saved pipeline on test data.
    """
    logger.info("Evaluating %s on held-out test set (%d records)...", name, len(y_test))
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"Model pipeline not found: {pipeline_path}")

    pipeline = joblib.load(pipeline_path)

    # Predictions
    t0 = time.time()
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    eval_time = time.time() - t0

    # Metrics
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report_dict = classification_report(y_test, y_pred, target_names=["BENIGN", "DDoS"], output_dict=True)

    metrics = {
        "model_name": name,
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 5),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 5),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 5),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 5),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 5),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 5),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 5) if y_proba is not None else None,
        "pr_auc": round(float(average_precision_score(y_test, y_proba)), 5) if y_proba is not None else None,
        "confusion_matrix": cm.tolist(),
        "per_class": report_dict,
        "latency_profile": profile_latency_distribution(pipeline, X_test, num_samples=1000),
        "error_analysis": analyze_errors(y_test.values, y_pred, y_proba, X_test),
        "total_test_eval_time_sec": round(eval_time, 2)
    }

    return metrics

def run_test_evaluation():
    """Evaluates baseline, advanced, and champion models on the held-out test set."""
    _, _, _, _, X_test, y_test, _ = load_processed_splits()

    models_to_eval = [
        ("Random Forest", os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")),
        ("XGBoost", os.path.join(MODELS_DIR, "xgboost_pipeline.joblib")),
        ("Champion (Best)", os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))
    ]

    all_results = {}
    tracker = ExperimentTracker()

    # Load existing experiments to update test metrics
    for model_name, model_path in models_to_eval:
        if os.path.exists(model_path):
            res = evaluate_model_pipeline(model_name, model_path, X_test, y_test)
            all_results[model_name] = res

            # Update tracker if not the generic champion duplicate
            if model_name in ["Random Forest", "XGBoost"]:
                # Log test metrics into experiment tracker
                tracker.log_experiment(
                    model_name=model_name,
                    hyperparameters={},
                    val_metrics={},
                    test_metrics={
                        "accuracy": res["accuracy"],
                        "precision": res["precision"],
                        "recall": res["recall"],
                        "f1_score": res["f1_score"],
                        "macro_f1": res["macro_f1"],
                        "roc_auc": res["roc_auc"],
                        "pr_auc": res["pr_auc"],
                        "false_positives": res["error_analysis"]["false_positive_count"],
                        "false_negatives": res["error_analysis"]["false_negative_count"]
                    },
                    inference_latency_ms=res["latency_profile"]["mean_ms"],
                    artifact_path=model_path,
                    notes=f"Test set evaluation on {len(y_test)} held-out samples."
                )

    report_path = os.path.join(METRICS_DIR, "test_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Comprehensive Test Evaluation Report saved to %s", report_path)
    return all_results

if __name__ == "__main__":
    run_test_evaluation()
