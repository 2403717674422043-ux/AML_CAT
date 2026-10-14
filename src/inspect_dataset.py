"""
DDoSGuard - Dataset Inspection Script
Inspects the raw CIC-IDS-2017 DDoS dataset file, extracts schema, row count,
class distribution, missing/infinite value counts, and constant features.
"""

import os
import json
import numpy as np
import pandas as pd

RAW_DATA_PATH = os.path.join("data", "raw", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
METRICS_DIR = os.path.join("artifacts", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)

def inspect_dataset(file_path: str = RAW_DATA_PATH):
    print(f"=== Inspecting Dataset: {file_path} ===")
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"File Size: {file_size_bytes:,} bytes ({file_size_mb:.2f} MB)")

    # Read first 5 lines to inspect raw headers and delimiters
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        first_lines = [f.readline().strip() for _ in range(5)]
    print("\nRaw Header:")
    print(first_lines[0][:200] + "...")

    # Load with pandas, strip column whitespace
    print("\nLoading dataset with pandas...")
    df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    
    raw_columns = list(df.columns)
    print(f"Total Rows: {len(df):,}")
    print(f"Total Columns: {len(raw_columns)}")

    # Clean column names (strip whitespace)
    clean_columns = [c.strip() for c in df.columns]
    column_mapping = {raw: clean for raw, clean in zip(raw_columns, clean_columns)}
    df.columns = clean_columns

    print("\nColumns list:")
    for idx, col in enumerate(df.columns):
        print(f"  {idx:2d}: {col} (raw: '{raw_columns[idx]}', dtype: {df[col].dtype})")

    # Target column
    label_col = "Label" if "Label" in df.columns else None
    if not label_col:
        for c in df.columns:
            if "label" in c.lower():
                label_col = c
                break

    print(f"\nTarget Column identified: '{label_col}'")
    if label_col:
        # Strip label values
        if df[label_col].dtype == object or isinstance(df[label_col].iloc[0], str):
            df[label_col] = df[label_col].astype(str).str.strip()
        label_counts = df[label_col].value_counts(dropna=False).to_dict()
        print("Label Distribution:")
        for label, count in label_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label}: {count:,} ({pct:.2f}%)")

    # Check for missing values (NaN / None)
    null_counts = df.isnull().sum()
    cols_with_null = null_counts[null_counts > 0].to_dict()
    print(f"\nColumns with null values: {len(cols_with_null)}")
    for col, count in cols_with_null.items():
        print(f"  {col}: {count:,} nulls ({count/len(df)*100:.3f}%)")

    # Check for infinite values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_counts = {}
    for col in numeric_cols:
        inf_c = np.isinf(df[col]).sum()
        if inf_c > 0:
            inf_counts[col] = int(inf_c)
    print(f"\nColumns with +/- Inf values: {len(inf_counts)}")
    for col, count in inf_counts.items():
        print(f"  {col}: {count:,} infinities ({count/len(df)*100:.3f}%)")

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()
    print(f"\nExact Duplicate Rows: {duplicate_count:,} ({duplicate_count/len(df)*100:.2f}%)")

    # Check constant columns (single unique value)
    nunique = df.nunique()
    constant_cols = nunique[nunique <= 1].to_dict()
    print(f"\nConstant/Zero-variance Columns (nunique <= 1): {len(constant_cols)}")
    for col, count in constant_cols.items():
        print(f"  {col}: {count} unique value ({df[col].iloc[0] if len(df) > 0 else 'N/A'})")

    # Save summary report to JSON
    summary = {
        "file_name": os.path.basename(file_path),
        "file_size_bytes": file_size_bytes,
        "file_size_mb": round(file_size_mb, 2),
        "total_rows": int(len(df)),
        "total_columns": int(len(raw_columns)),
        "target_column": label_col,
        "label_distribution": {str(k): int(v) for k, v in label_counts.items()} if label_col else {},
        "cols_with_null": cols_with_null,
        "cols_with_inf": inf_counts,
        "duplicate_rows": int(duplicate_count),
        "constant_columns": list(constant_cols.keys()),
        "columns": clean_columns
    }

    out_file = os.path.join(METRICS_DIR, "dataset_inspection.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDataset inspection summary saved to {out_file}")

if __name__ == "__main__":
    inspect_dataset()
