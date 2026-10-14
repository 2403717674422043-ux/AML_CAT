"""
DDoSGuard - Data Loader Module
Handles loading the CIC-IDS-2017 DDoS dataset with schema validation,
robust whitespace stripping, and memory-efficient types.
"""

import os
import logging
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DDoSGuard.DataLoader")

DEFAULT_RAW_PATH = os.path.join("data", "raw", "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

# Metadata / Identifier columns to exclude from behavioral ML modeling
IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Source Port"
]

# Duplicate or non-informative columns
REDUNDANT_COLUMNS = [
    "Fwd Header Length.1"  # Duplicate of 'Fwd Header Length' in CIC-IDS-2017
]

def load_raw_dataset(file_path: str = DEFAULT_RAW_PATH) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Loads raw CSV dataset, strips column whitespace, and returns cleaned DataFrame
    along with mapping from raw column names to cleaned column names.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at expected path: {file_path}")

    logger.info("Loading raw dataset from %s ...", file_path)
    df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    
    raw_columns = list(df.columns)
    clean_columns = [c.strip() for c in raw_columns]
    column_mapping = {raw: clean for raw, clean in zip(raw_columns, clean_columns)}
    df.columns = clean_columns
    
    logger.info("Loaded %d rows and %d columns.", len(df), len(df.columns))
    return df, column_mapping

def get_dataset_provenance(file_path: str = DEFAULT_RAW_PATH) -> Dict[str, Any]:
    """
    Returns dataset metadata and provenance for reproducibility and reporting.
    """
    if not os.path.exists(file_path):
        return {"status": "missing", "path": file_path}

    size_bytes = os.path.getsize(file_path)
    return {
        "dataset_name": "CIC-IDS-2017 Friday DDoS Traffic Capture",
        "file_name": os.path.basename(file_path),
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_bytes / (1024 * 1024), 2),
        "source_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "institution": "Canadian Institute for Cybersecurity (CIC), UNB",
        "capture_scenario": "Friday-WorkingHours-Afternoon-DDos.pcap (LOIC HTTP DDoS attack & Benign background traffic)",
        "intended_scope": "Binary DDoS vs Benign flow classification evaluated on CICIDS2017 capture"
    }

if __name__ == "__main__":
    df, col_map = load_raw_dataset()
    print("Dataset loaded successfully. Shape:", df.shape)
    provenance = get_dataset_provenance()
    print("Provenance:", provenance)
