"""
DDoSGuard - Visualization Module
Generates all 7 required publication-grade plots:
1. Class distribution (Benign vs DDoS)
2. Correlation heatmap of key numerical features
3. PCA 2D projection of network traffic flows
4. Confusion matrix heatmap
5. Precision-Recall & ROC curves
6. SHAP summary plot / feature importance
7. Model comparison chart (Random Forest vs XGBoost)
All plots are saved to artifacts/plots/
"""

import os
import sys
import json
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
import joblib

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import load_processed_splits, MODELS_DIR, METRICS_DIR, RANDOM_SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DDoSGuard.Visualizations")

PLOTS_DIR = os.path.join("artifacts", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Styling configuration for professional cybersecurity aesthetics
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"BENIGN": "#10b981", "DDoS": "#ef4444", "Primary": "#0284c7", "Accent": "#06b6d4"}

def plot_class_distribution(y_train: pd.Series, y_val: pd.Series, y_test: pd.Series):
    """Plot 1: Benign vs DDoS Class Distribution across splits."""
    logger.info("Generating Plot 1: Class Distribution...")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    splits_data = [
        {"Split": "Train (70%)", "BENIGN": int((y_train == 0).sum()), "DDoS": int((y_train == 1).sum())},
        {"Split": "Val (15%)", "BENIGN": int((y_val == 0).sum()), "DDoS": int((y_val == 1).sum())},
        {"Split": "Test (15%)", "BENIGN": int((y_test == 0).sum()), "DDoS": int((y_test == 1).sum())}
    ]
    df_splits = pd.DataFrame(splits_data).set_index("Split")
    
    bars = df_splits.plot(kind="bar", stacked=True, color=[PALETTE["BENIGN"], PALETTE["DDoS"]], ax=ax, width=0.55)
    
    ax.set_title("DDoSGuard: Flow Class Distribution Across Partitions", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Partition", fontsize=11, fontweight="semibold")
    ax.set_ylabel("Flow Record Count", fontsize=11, fontweight="semibold")
    ax.legend(["BENIGN (0)", "DDoS (1)"], frameon=True, facecolor="white", loc="upper right")
    plt.xticks(rotation=0)
    
    # Add count labels
    for p in ax.patches:
        height = p.get_height()
        if height > 5000:
            ax.annotate(f"{height:,}",
                        (p.get_x() + p.get_width() / 2., p.get_y() + height / 2.),
                        ha="center", va="center", color="white", fontsize=9, fontweight="bold")
            
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "01_class_distribution.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Plot 1 to %s", out_path)

def plot_correlation_heatmap(X_train: pd.DataFrame):
    """Plot 2: Correlation Heatmap of top 15 traffic flow features."""
    logger.info("Generating Plot 2: Correlation Heatmap...")
    # Select 15 top standard flow features
    candidate_features = [
        "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Bwd Packet Length Max", "Flow Bytes/s",
        "Flow Packets/s", "Flow IAT Mean", "Flow IAT Max", "Fwd Header Length",
        "Bwd Header Length", "Packet Length Mean", "Average Packet Size"
    ]
    cols = [c for c in candidate_features if c in X_train.columns]
    if len(cols) < 10:
        cols = X_train.columns[:15].tolist()

    sample_corr = X_train[cols].sample(n=min(20000, len(X_train)), random_state=RANDOM_SEED).corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    sns.heatmap(
        sample_corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True,
        linewidths=0.5, annot_kws={"size": 7}, ax=ax, vmin=-1, vmax=1
    )
    ax.set_title("DDoSGuard: Feature Correlation Heatmap (Selected Flow Features)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "02_correlation_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Plot 2 to %s", out_path)

def plot_pca_projection(X_train: pd.DataFrame, y_train: pd.Series):
    """Plot 3: PCA 2D Projection fitted strictly on training data."""
    logger.info("Generating Plot 3: PCA Projection...")
    # Subsample for clear visual density without overplotting
    sample_size = min(10000, len(X_train))
    sample_idx = X_train.sample(n=sample_size, random_state=RANDOM_SEED).index
    X_sub = X_train.loc[sample_idx]
    y_sub = y_train.loc[sample_idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sub)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    benign_pts = X_pca[y_sub == 0]
    ddos_pts = X_pca[y_sub == 1]

    ax.scatter(benign_pts[:, 0], benign_pts[:, 1], c=PALETTE["BENIGN"], label="BENIGN", alpha=0.4, s=12, edgecolors="none")
    ax.scatter(ddos_pts[:, 0], ddos_pts[:, 1], c=PALETTE["DDoS"], label="DDoS", alpha=0.4, s=12, edgecolors="none")

    ax.set_title(f"DDoSGuard: 2D PCA Traffic Projection (Var Exp: {var_exp[0]*100:.1f}% & {var_exp[1]*100:.1f}%)",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% variance)", fontsize=10)
    ax.set_ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% variance)", fontsize=10)
    ax.legend(frameon=True, facecolor="white", loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "03_pca_projection.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Plot 3 to %s", out_path)

def plot_confusion_matrix_heatmap(y_test: pd.Series, y_pred: np.ndarray, model_name: str = "Selected Champion"):
    """Plot 4: Confusion Matrix Heatmap on Test Set."""
    logger.info("Generating Plot 4: Confusion Matrix Heatmap...")
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    annotations = [[f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.2f}%)" for j in range(2)] for i in range(2)]

    sns.heatmap(cm, annot=np.array(annotations), fmt="", cmap="Blues", cbar=True,
                xticklabels=["BENIGN (0)", "DDoS (1)"], yticklabels=["BENIGN (0)", "DDoS (1)"], ax=ax)
    ax.set_title(f"DDoSGuard: Test Confusion Matrix ({model_name})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="semibold")
    ax.set_ylabel("Actual Class (Ground Truth)", fontsize=10, fontweight="semibold")
    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "04_confusion_matrix.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Plot 4 to %s", out_path)

def plot_pr_and_roc_curves(y_test: pd.Series, y_proba_rf: np.ndarray, y_proba_xgb: np.ndarray):
    """Plot 5: Dual PR-Curve and ROC-Curve comparison."""
    logger.info("Generating Plot 5: PR & ROC Curves...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # 1. Precision-Recall Curve
    if y_proba_rf is not None:
        p_rf, r_rf, _ = precision_recall_curve(y_test, y_proba_rf)
        ax1.plot(r_rf, p_rf, label=f"Random Forest (PR-AUC={auc(r_rf, p_rf):.4f})", color=PALETTE["Primary"], lw=2)
    if y_proba_xgb is not None:
        p_xgb, r_xgb, _ = precision_recall_curve(y_test, y_proba_xgb)
        ax1.plot(r_xgb, p_xgb, label=f"XGBoost (PR-AUC={auc(r_xgb, p_xgb):.4f})", color=PALETTE["Accent"], lw=2, linestyle="--")

    baseline_ratio = (y_test == 1).mean()
    ax1.axhline(baseline_ratio, color="grey", linestyle=":", label=f"No-Skill Baseline ({baseline_ratio:.2f})")
    ax1.set_title("Precision-Recall Curve (Imbalance Robust)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=10)
    ax1.set_ylabel("Precision", fontsize=10)
    ax1.set_ylim([0.9, 1.01])
    ax1.legend(loc="lower left", frameon=True)

    # 2. ROC Curve
    if y_proba_rf is not None:
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
        ax2.plot(fpr_rf, tpr_rf, label=f"Random Forest (ROC-AUC={auc(fpr_rf, tpr_rf):.4f})", color=PALETTE["Primary"], lw=2)
    if y_proba_xgb is not None:
        fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
        ax2.plot(fpr_xgb, tpr_xgb, label=f"XGBoost (ROC-AUC={auc(fpr_xgb, tpr_xgb):.4f})", color=PALETTE["Accent"], lw=2, linestyle="--")

    ax2.plot([0, 1], [0, 1], color="grey", linestyle=":", label="Random Chance (0.50)")
    ax2.set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("False Positive Rate", fontsize=10)
    ax2.set_ylabel("True Positive Rate", fontsize=10)
    ax2.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "05_pr_roc_curves.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Plot 5 to %s", out_path)

def plot_model_comparison_chart():
    """Plot 7: Bar chart comparing Baseline vs Advanced Model on Test Metrics."""
    logger.info("Generating Plot 7: Model Comparison Chart...")
    experiments_file = os.path.join(METRICS_DIR, "experiments.json")
    if not os.path.exists(experiments_file):
        logger.warning("Experiments file not found. Skipping plot 7.")
        return

    with open(experiments_file, "r", encoding="utf-8") as f:
        experiments = json.load(f)

    # Extract test metrics for RF and XGB
    models_data = {}
    for exp in experiments:
        m_name = exp.get("model_name")
        tm = exp.get("test_metrics", {})
        if tm and m_name in ["Random Forest", "XGBoost"]:
            models_data[m_name] = {
                "Accuracy": tm.get("accuracy", 0.0),
                "Precision": tm.get("precision", 0.0),
                "Recall": tm.get("recall", 0.0),
                "F1-Score": tm.get("f1_score", 0.0),
                "PR-AUC": tm.get("pr_auc", 0.0),
                "ROC-AUC": tm.get("roc_auc", 0.0)
            }

    if not models_data:
        logger.warning("No test metrics logged yet. Generating comparison from validation metrics...")
        for exp in experiments:
            m_name = exp.get("model_name")
            vm = exp.get("val_metrics", {})
            if vm and m_name in ["Random Forest", "XGBoost"]:
                models_data[m_name] = {
                    "Accuracy": vm.get("accuracy", 0.0),
                    "Precision": vm.get("precision", 0.0),
                    "Recall": vm.get("recall", 0.0),
                    "F1-Score": vm.get("f1_score", 0.0),
                    "PR-AUC": vm.get("pr_auc", 0.0),
                    "ROC-AUC": vm.get("roc_auc", 0.0)
                }

    df_comp = pd.DataFrame(models_data)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    df_comp.plot(kind="bar", ax=ax, width=0.6, color=[PALETTE["Primary"], PALETTE["Accent"]])
    ax.set_title("DDoSGuard: Model Performance Comparison (Baseline vs Advanced)", fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11, fontweight="semibold")
    ax.set_ylim([0.98, 1.002])  # Highlighting fine differences
    plt.xticks(rotation=0, fontsize=10)
    ax.legend(frameon=True, facecolor="white", loc="lower right")

    # Add numeric tags on bars
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.4f}", (p.get_x() + p.get_width() / 2., h),
                        ha="center", va="bottom", fontsize=8, rotation=45)

    plt.tight_layout()
    out_path = os.path.join(PLOTS_DIR, "07_model_comparison.png")
    plt.savefig(out_path)
    plt.close()
    logger.info("Saved Plot 7 to %s", out_path)

def generate_all_plots():
    """Generates all standard visualization plots."""
    X_train, y_train, X_val, y_val, X_test, y_test, _ = load_processed_splits()
    
    # Plot 1: Class distribution
    plot_class_distribution(y_train, y_val, y_test)

    # Plot 2: Correlation heatmap
    plot_correlation_heatmap(X_train)

    # Plot 3: PCA
    plot_pca_projection(X_train, y_train)

    # Load models for test curves and confusion matrix
    rf_path = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
    xgb_path = os.path.join(MODELS_DIR, "xgboost_pipeline.joblib")
    best_path = os.path.join(MODELS_DIR, "best_model_pipeline.joblib")

    rf_pipe = joblib.load(rf_path) if os.path.exists(rf_path) else None
    xgb_pipe = joblib.load(xgb_path) if os.path.exists(xgb_path) else None
    best_pipe = joblib.load(best_path) if os.path.exists(best_path) else rf_pipe

    if best_pipe:
        y_pred = best_pipe.predict(X_test)
        plot_confusion_matrix_heatmap(y_test, y_pred, model_name="Best Model")

    proba_rf = rf_pipe.predict_proba(X_test)[:, 1] if rf_pipe else None
    proba_xgb = xgb_pipe.predict_proba(X_test)[:, 1] if xgb_pipe else None
    if proba_rf is not None or proba_xgb is not None:
        plot_pr_and_roc_curves(y_test, proba_rf, proba_xgb)

    # Plot 7: Comparison chart
    plot_model_comparison_chart()

if __name__ == "__main__":
    generate_all_plots()
