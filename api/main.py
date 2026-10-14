"""
DDoSGuard - FastAPI Service
Provides high-performance, validated REST endpoints for:
- GET /health
- GET /metrics
- POST /predict
- POST /predict-batch
"""

import os
import sys
import time
import json
import logging
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse, MetricsResponse, OfflineMetrics,
    FlowRecordInput, BatchFlowInput, PredictionResponse, BatchPredictionResponse,
    REQUIRED_FEATURES
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DDoSGuard.API")

# Global state
MODEL_PIPELINE = None
MODEL_NAME = "None"
LIVE_REQUEST_COUNT = 0
TEST_METRICS_CACHE = {}

MODEL_PATH = os.path.join("models", "best_model_pipeline.joblib")
TEST_REPORT_PATH = os.path.join("artifacts", "metrics", "test_evaluation_report.json")

def load_model_artifacts():
    """Loads champion model and test evaluation metrics."""
    global MODEL_PIPELINE, MODEL_NAME, TEST_METRICS_CACHE
    if MODEL_PIPELINE is not None:
        return True

    if os.path.exists(MODEL_PATH):
        try:
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            clf = MODEL_PIPELINE.named_steps.get("classifier")
            MODEL_NAME = type(clf).__name__ if clf else "Trained Pipeline"
            logger.info("Successfully loaded champion model: %s from %s", MODEL_NAME, MODEL_PATH)
        except Exception as e:
            logger.error("Failed to load model pipeline: %s", e)
            MODEL_PIPELINE = None
    else:
        logger.warning("Champion model file not found at %s. Service running in degraded mode.", MODEL_PATH)

    if os.path.exists(TEST_REPORT_PATH):
        try:
            with open(TEST_REPORT_PATH, "r", encoding="utf-8") as f:
                TEST_METRICS_CACHE = json.load(f)
        except Exception as e:
            logger.warning("Could not read test evaluation report: %s", e)

    return MODEL_PIPELINE is not None

# Eagerly attempt load
load_model_artifacts()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    logger.info("Starting DDoSGuard API service...")
    load_model_artifacts()
    yield
    logger.info("Shutting down DDoSGuard API service...")

app = FastAPI(
    title="DDoSGuard REST API",
    description="Adaptive & Explainable Machine Learning Framework for DDoS Detection and Mitigation (CIC-IDS-2017)",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local dashboards and public deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", tags=["General"])
def root_endpoint():
    """Root info endpoint."""
    return {
        "service": "DDoSGuard API",
        "description": "High-throughput flow intrusion detection API",
        "docs_url": "/docs",
        "health_url": "/health",
        "metrics_url": "/metrics",
        "status": "ready" if MODEL_PIPELINE is not None else "degraded"
    }

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """
    Returns API health status, model readiness, and server timestamp.
    """
    is_ready = MODEL_PIPELINE is not None
    return HealthResponse(
        status="healthy" if is_ready else "degraded",
        model_loaded=is_ready,
        model_name=MODEL_NAME if is_ready else "None",
        version="1.0.0",
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S UTC")
    )

@app.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
def get_metrics():
    """
    Returns saved offline held-out test evaluation metrics and operational live counts.
    Explicitly distinguishes benchmark test results from operational inference volume.
    """
    global LIVE_REQUEST_COUNT, TEST_METRICS_CACHE
    offline_metrics_map = {}

    for name, data in TEST_METRICS_CACHE.items():
        if name in ["Random Forest", "XGBoost"]:
            supp = data.get("per_class", {}).get("weighted avg", {}).get("support", 33862)
            offline_metrics_map[name] = OfflineMetrics(
                model_name=data.get("model_name", name),
                accuracy=data.get("accuracy", 0.0),
                precision=data.get("precision", 0.0),
                recall=data.get("recall", 0.0),
                f1_score=data.get("f1_score", 0.0),
                pr_auc=data.get("pr_auc"),
                roc_auc=data.get("roc_auc"),
                false_positives=data.get("error_analysis", {}).get("false_positive_count", 0),
                false_negatives=data.get("error_analysis", {}).get("false_negative_count", 0),
                test_sample_size=int(supp),
                latency_mean_ms=data.get("latency_profile", {}).get("mean_ms", 0.0)
            )

    return MetricsResponse(
        status="success",
        dataset_name="Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv (CIC-IDS-2017)",
        offline_test_metrics=offline_metrics_map,
        operational_live_requests_count=LIVE_REQUEST_COUNT,
        note="Offline metrics are computed on the untouched 15% held-out test partition (33,862 records). Live requests represent actual inferences processed by this server session."
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_single(record: FlowRecordInput):
    """
    Evaluates a single network flow record and predicts BENIGN (0) vs DDoS (1).
    Validates presence of all 68 required features and rejects non-finite numbers.
    """
    global MODEL_PIPELINE, LIVE_REQUEST_COUNT, MODEL_NAME
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded. Please ensure training has completed."
        )

    LIVE_REQUEST_COUNT += 1
    t0 = time.perf_counter()

    # Convert features dict to DataFrame in exact required column order
    input_df = pd.DataFrame([record.features])[REQUIRED_FEATURES]

    # Predict via trained pipeline (includes SimpleImputer)
    try:
        pred_label_idx = int(MODEL_PIPELINE.predict(input_df)[0])
        pred_label_str = "DDoS" if pred_label_idx == 1 else "BENIGN"

        proba = 1.0 if pred_label_idx == 1 else 0.0
        if hasattr(MODEL_PIPELINE, "predict_proba"):
            proba_arr = MODEL_PIPELINE.predict_proba(input_df)[0]
            proba = float(proba_arr[1])

        latency = (time.perf_counter() - t0) * 1000.0

        conf_level = "HIGH" if (proba >= 0.90 or proba <= 0.10) else ("MEDIUM" if (proba >= 0.75 or proba <= 0.25) else "LOW")

        return PredictionResponse(
            prediction=pred_label_idx,
            label=pred_label_str,
            probability=round(proba, 5),
            confidence_level=conf_level,
            inference_latency_ms=round(latency, 4),
            model_name=MODEL_NAME
        )
    except Exception as e:
        logger.error("Prediction execution failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(batch: BatchFlowInput):
    """
    Evaluates a batch of network flow records.
    """
    global MODEL_PIPELINE, LIVE_REQUEST_COUNT, MODEL_NAME
    if MODEL_PIPELINE is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifacts are not loaded."
        )

    if not batch.records:
        raise HTTPException(status_code=400, detail="Batch records list cannot be empty.")

    t0 = time.perf_counter()
    rows = [r.features for r in batch.records]
    input_df = pd.DataFrame(rows)[REQUIRED_FEATURES]

    LIVE_REQUEST_COUNT += len(batch.records)

    preds = MODEL_PIPELINE.predict(input_df)
    probas = MODEL_PIPELINE.predict_proba(input_df)[:, 1] if hasattr(MODEL_PIPELINE, "predict_proba") else [1.0 if p == 1 else 0.0 for p in preds]

    total_batch_time = (time.perf_counter() - t0) * 1000.0
    avg_latency_per_item = total_batch_time / len(batch.records)

    results = []
    for p, pr in zip(preds, probas):
        p_int = int(p)
        label = "DDoS" if p_int == 1 else "BENIGN"
        conf = "HIGH" if (pr >= 0.90 or pr <= 0.10) else "MEDIUM"
        results.append(
            PredictionResponse(
                prediction=p_int,
                label=label,
                probability=round(float(pr), 5),
                confidence_level=conf,
                inference_latency_ms=round(avg_latency_per_item, 4),
                model_name=MODEL_NAME
            )
        )

    return BatchPredictionResponse(
        total_records=len(results),
        predictions=results,
        batch_processing_time_ms=round(total_batch_time, 2)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
