"""
DDoSGuard - Explainability Module
Generates SHAP (SHapley Additive exPlanations) summary plot and feature importance
for the champion model. Includes permutation importance fallback if SHAP encounters issues.
Saves:
- artifacts/plots/06_shap_summary.png
- artifacts/metrics/shap_feature_importance.json
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_processed_splits, MODELS_DIR, METRICS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DDoSGuard.Explainability")

PLOTS_DIR = os.path.join("artifacts", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def generate_shap_explanations(sample_size: int = 500) -> Dict[str, Any]:
    """
    Computes SHAP values using TreeExplainer on a representative sample of test data.
    """
    logger.info("=== Generating Model Explainability (SHAP) ===")
    best_model_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Champion model not found at {best_model_path}")

    pipeline = joblib.load(best_model_path)
    imputer = pipeline.named_steps["imputer"]
    model = pipeline.named_steps["classifier"]

    _, _, _, _, X_test, y_test, feature_names = load_processed_splits()

    # Stratified subsample of test data for SHAP computation
    logger.info("Extracting stratified sample of %d test records for SHAP TreeExplainer...", sample_size)
    sample_idx = pd.concat([
        X_test[y_test == 0].sample(n=sample_size // 2, random_state=RANDOM_SEED),
        X_test[y_test == 1].sample(n=sample_size // 2, random_state=RANDOM_SEED)
    ]).index

    X_sample = X_test.loc[sample_idx]
    y_sample = y_test.loc[sample_idx]

    # Preprocess through imputer
    X_sample_arr = imputer.transform(X_sample)
    X_sample_df = pd.DataFrame(X_sample_arr, columns=feature_names, index=X_sample.index)

    shap_used = False
    feature_importance_dict = {}

    try:
        import shap
        logger.info("Computing SHAP values with TreeExplainer...")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X_sample_df)

        # Handle binary classification output shape
        if len(shap_values.values.shape) == 3:  # (samples, features, classes)
            shap_vals_matrix = shap_values.values[:, :, 1]
        else:
            shap_vals_matrix = shap_values.values

        # Mean absolute SHAP values per feature
        mean_abs_shap = np.abs(shap_vals_matrix).mean(axis=0)
        sorted_indices = np.argsort(mean_abs_shap)[::-1]

        for idx in sorted_indices[:20]:
            feature_importance_dict[feature_names[idx]] = round(float(mean_abs_shap[idx]), 6)

        # Generate SHAP Beeswarm summary plot
        logger.info("Generating SHAP summary plot...")
        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            shap_vals_matrix,
            X_sample_df,
            max_display=15,
            show=False,
            plot_type="dot"
        )
        plt.title("DDoSGuard: SHAP Summary Plot (Top Predictive Flow Features)", fontsize=13, fontweight="bold", pad=15)
        plt.xlabel("SHAP Value (Impact on Model DDoS Prediction Log-Odds)", fontsize=10)
        plt.tight_layout()
        shap_plot_path = os.path.join(PLOTS_DIR, "06_shap_summary.png")
        plt.savefig(shap_plot_path, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot to %s", shap_plot_path)
        shap_used = True

    except Exception as e:
        logger.warning("SHAP computation encountered error (%s). Falling back to Gini / Feature Importances.", e)
        # Fallback to model feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]
            for idx in sorted_idx[:20]:
                feature_importance_dict[feature_names[idx]] = round(float(importances[idx]), 6)

            # Generate horizontal bar chart fallback
            plt.figure(figsize=(10, 7), dpi=300)
            top_features = [feature_names[i] for i in sorted_idx[:15]][::-1]
            top_scores = [importances[i] for i in sorted_idx[:15]][::-1]
            plt.barh(top_features, top_scores, color="#0284c7")
            plt.title("DDoSGuard: Feature Importance (Ensemble Gini/Gain Impurity)", fontsize=13, fontweight="bold")
            plt.xlabel("Importance Score", fontsize=10)
            plt.tight_layout()
            shap_plot_path = os.path.join(PLOTS_DIR, "06_shap_summary.png")
            plt.savefig(shap_plot_path)
            plt.close()
            logger.info("Saved Fallback Feature Importance plot to %s", shap_plot_path)

    # Save feature importance metadata
    explanation_report = {
        "method": "SHAP TreeExplainer (Shapley Additive exPlanations)" if shap_used else "Feature Importances (Ensemble Impurity Gain)",
        "sample_size": sample_size,
        "champion_model": type(model).__name__,
        "top_features": feature_importance_dict,
        "causality_caveat": "Important: Feature attribution reflects statistical correlations in the CIC-IDS-2017 benchmark dataset, not physical or causal network packet generation mechanics."
    }

    report_path = os.path.join(METRICS_DIR, "shap_feature_importance.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(explanation_report, f, indent=2)
    logger.info("Saved feature importance metadata to %s", report_path)

    return explanation_report

if __name__ == "__main__":
    generate_shap_explanations()
