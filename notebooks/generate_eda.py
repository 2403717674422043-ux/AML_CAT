import os
import nbformat as nbf

nb = nbf.v4.new_notebook()

nb["cells"] = [
    nbf.v4.new_markdown_cell("""# DDoSGuard: Exploratory Data Analysis & Modeling Pipeline
**Course:** 19MAM56 — Advanced Machine Learning Lab (CAT II)  
**Institution:** Coimbatore Institute of Technology  
**Department:** Computing — AI & ML  
**Primary Dataset:** CIC-IDS-2017 (`Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`)
"""),
    nbf.v4.new_code_cell("""import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

sys.path.insert(0, os.path.abspath(".."))
print("Environment and dependencies initialized.")"""),
    nbf.v4.new_markdown_cell("""## 1. Dataset Provenance & Inspection
Loading the raw dataset inspection report to observe row count, target balance, missingness, and infinite values.
"""),
    nbf.v4.new_code_cell("""with open("../artifacts/metrics/dataset_inspection.json", "r", encoding="utf-8") as f:
    inspection = json.load(f)

print(f"Dataset File: {inspection['file_name']}")
print(f"File Size: {inspection['file_size_mb']} MB")
print(f"Total Rows: {inspection['total_rows']:,}")
print(f"Total Columns: {inspection['total_columns']}")
print(f"Target Distribution: {inspection['label_distribution']}")
print(f"Infinities Detected: {inspection['cols_with_inf']}")
print(f"Missing Values: {inspection['cols_with_null']}")
print(f"Zero-variance Columns Dropped: {len(inspection['constant_columns'])}")"""),
    nbf.v4.new_markdown_cell("""## 2. Partition Verification & Feature Pipeline
Loading the processed 70/15/15 train/val/test splits. Identifiers (`Flow ID`, IP addresses, ports, timestamps) were excluded to avoid memorization leakage.
"""),
    nbf.v4.new_code_cell("""from src.preprocessing import load_processed_splits

X_train, y_train, X_val, y_val, X_test, y_test, features = load_processed_splits()
print(f"Train Shape: {X_train.shape}")
print(f"Val Shape:   {X_val.shape}")
print(f"Test Shape:  {X_test.shape}")
print(f"Retained Model Features ({len(features)}):")
print(features[:15])"""),
    nbf.v4.new_markdown_cell("""## 3. Held-Out Test Evaluation Results
Empirical comparison of Baseline (Random Forest) vs Advanced (XGBoost) on the held-out test split.
"""),
    nbf.v4.new_code_cell("""with open("../artifacts/metrics/test_evaluation_report.json", "r", encoding="utf-8") as f:
    test_report = json.load(f)

rows = []
for model_name in ["Random Forest", "XGBoost"]:
    res = test_report[model_name]
    rows.append({
        "Model": model_name,
        "Accuracy": f"{res['accuracy']*100:.3f}%",
        "Precision": f"{res['precision']*100:.3f}%",
        "Recall": f"{res['recall']*100:.3f}%",
        "F1-Score": f"{res['f1_score']*100:.3f}%",
        "PR-AUC": f"{res['pr_auc']*100:.3f}%",
        "FP Count": res["error_analysis"]["false_positive_count"],
        "FN Count": res["error_analysis"]["false_negative_count"],
        "Latency (ms)": f"{res['latency_profile']['mean_ms']:.3f} ms"
    })

pd.DataFrame(rows)"""),
    nbf.v4.new_markdown_cell("""## 4. Generated Artifact Plots
Displaying all 7 publication-grade visualizations.
"""),
    nbf.v4.new_code_cell("""from IPython.display import Image, display

plots = [
    "01_class_distribution.png",
    "02_correlation_heatmap.png",
    "03_pca_projection.png",
    "04_confusion_matrix.png",
    "05_pr_roc_curves.png",
    "06_shap_summary.png",
    "07_model_comparison.png"
]

for p in plots:
    path = os.path.join("..", "artifacts", "plots", p)
    if os.path.exists(path):
        print(f"=== {p} ===")
        display(Image(filename=path))""")
]

with open("notebooks/eda.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook generated successfully at notebooks/eda.ipynb")
