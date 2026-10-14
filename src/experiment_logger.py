"""
DDoSGuard - Experiment Logger Module
Tracks ML experiments, hyperparameters, validation and test metrics,
training time, inference latency, and artifacts.
"""

import os
import json
import time
import hashlib
import platform
import logging
from typing import Dict, Any, List
import pandas as pd

logger = logging.getLogger("DDoSGuard.ExperimentLogger")

METRICS_DIR = os.path.join("artifacts", "metrics")
LOG_JSON_PATH = os.path.join(METRICS_DIR, "experiments.json")
LOG_CSV_PATH = os.path.join(METRICS_DIR, "experiments.csv")

def compute_file_hash(filepath: str) -> str:
    """Computes SHA-256 checksum for a file."""
    if not os.path.exists(filepath):
        return "not_found"
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read first 8MB to compute fast representative fingerprint
        chunk = f.read(8 * 1024 * 1024)
        hasher.update(chunk)
    return hasher.hexdigest()[:16]

class ExperimentTracker:
    def __init__(self, metrics_dir: str = METRICS_DIR):
        self.metrics_dir = metrics_dir
        os.makedirs(self.metrics_dir, exist_ok=True)
        self.json_path = os.path.join(self.metrics_dir, "experiments.json")
        self.csv_path = os.path.join(self.metrics_dir, "experiments.csv")

    def log_experiment(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        val_metrics: Dict[str, Any],
        test_metrics: Dict[str, Any] = None,
        training_duration_sec: float = 0.0,
        inference_latency_ms: float = 0.0,
        artifact_path: str = "",
        dataset_name: str = "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Logs a full experiment entry to JSON and CSV.
        """
        raw_dataset_path = os.path.join("data", "raw", dataset_name)
        data_hash = compute_file_hash(raw_dataset_path)

        record = {
            "experiment_id": f"{model_name.lower().replace(' ', '_')}_{int(time.time())}",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "dataset_fingerprint": data_hash,
            "random_seed": 42,
            "split_strategy": "Stratified 70/15/15",
            "hyperparameters": hyperparameters,
            "training_duration_sec": round(training_duration_sec, 2),
            "inference_latency_ms": round(inference_latency_ms, 4),
            "val_metrics": val_metrics,
            "test_metrics": test_metrics or {},
            "artifact_path": artifact_path,
            "python_version": platform.python_version(),
            "os": platform.system(),
            "notes": notes
        }

        # Update JSON list
        existing_records: List[Dict[str, Any]] = []
        if os.path.exists(self.json_path):
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    existing_records = json.load(f)
            except Exception as e:
                logger.warning("Could not read existing JSON log: %s", e)

        # Replace or append
        existing_records = [r for r in existing_records if r.get("model_name") != model_name or r.get("experiment_id") == record["experiment_id"]]
        existing_records.append(record)

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(existing_records, f, indent=2)

        # Flat CSV export for quick tabular review
        flat_rows = []
        for r in existing_records:
            row = {
                "experiment_id": r["experiment_id"],
                "timestamp": r["timestamp"],
                "model_name": r["model_name"],
                "train_duration_s": r["training_duration_sec"],
                "latency_ms": r["inference_latency_ms"],
                "val_accuracy": r["val_metrics"].get("accuracy"),
                "val_f1": r["val_metrics"].get("f1_score"),
                "val_recall": r["val_metrics"].get("recall"),
                "val_precision": r["val_metrics"].get("precision"),
                "val_roc_auc": r["val_metrics"].get("roc_auc"),
                "val_pr_auc": r["val_metrics"].get("pr_auc"),
                "test_accuracy": r["test_metrics"].get("accuracy") if r["test_metrics"] else None,
                "test_f1": r["test_metrics"].get("f1_score") if r["test_metrics"] else None,
                "test_recall": r["test_metrics"].get("recall") if r["test_metrics"] else None,
                "test_precision": r["test_metrics"].get("precision") if r["test_metrics"] else None,
                "test_roc_auc": r["test_metrics"].get("roc_auc") if r["test_metrics"] else None,
                "test_pr_auc": r["test_metrics"].get("pr_auc") if r["test_metrics"] else None,
                "artifact_path": r["artifact_path"]
            }
            flat_rows.append(row)

        pd.DataFrame(flat_rows).to_csv(self.csv_path, index=False)
        logger.info("Logged experiment %s to %s and %s", model_name, self.json_path, self.csv_path)
        return record

if __name__ == "__main__":
    tracker = ExperimentTracker()
    print("ExperimentTracker initialized successfully.")
