"""
DDoSGuard - API Schemas Module
Pydantic schemas for request validation, prediction responses, and health checks.
"""

import math
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, ConfigDict, model_validator

# The 68 required network flow features
REQUIRED_FEATURES = [
    "Destination Port", "Protocol", "Flow Duration", "Total Fwd Packets",
    "Total Backward Packets", "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
    "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
    "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s",
    "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
    "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
    "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean",
    "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags",
    "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
    "Min Packet Length", "Max Packet Length", "Packet Length Mean",
    "Packet Length Std", "Packet Length Variance", "FIN Flag Count",
    "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "URG Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
    "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Subflow Fwd Packets",
    "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd",
    "min_seg_size_forward", "Active Mean", "Active Std", "Active Max",
    "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"
]

class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall service status ('healthy' or 'degraded')")
    model_loaded: bool = Field(..., description="Whether champion model pipeline is loaded")
    model_name: str = Field(..., description="Name of loaded champion model")
    version: str = Field(..., description="DDoSGuard API version")
    timestamp: str = Field(..., description="Server current timestamp")

class OfflineMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    pr_auc: Optional[float]
    roc_auc: Optional[float]
    false_positives: int
    false_negatives: int
    test_sample_size: int
    latency_mean_ms: float

class MetricsResponse(BaseModel):
    status: str
    dataset_name: str
    offline_test_metrics: Dict[str, OfflineMetrics]
    operational_live_requests_count: int
    note: str

class FlowRecordInput(BaseModel):
    """
    Validates a single network flow record containing all 68 required features.
    Rejects missing features, extra unexpected features, and non-finite numbers (NaN/Inf).
    """
    model_config = ConfigDict(extra="forbid")
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary mapping all 68 required feature names to their numeric values"
    )

    @model_validator(mode="before")
    @classmethod
    def validate_features_dict(cls, data: Any) -> Any:
        # If payload was sent as a flat dictionary, wrap it in {'features': data}
        if isinstance(data, dict) and "features" not in data:
            data = {"features": data}
        return data

    @model_validator(mode="after")
    def validate_flow_features(self) -> "FlowRecordInput":
        feats = self.features
        missing = [f for f in REQUIRED_FEATURES if f not in feats]
        if missing:
            raise ValueError(f"Missing required features ({len(missing)}): {missing[:5]}...")

        extra = [f for f in feats if f not in REQUIRED_FEATURES]
        if extra:
            raise ValueError(f"Unexpected extra features ({len(extra)}): {extra[:5]}...")

        # Non-finite validation
        non_finite = []
        for k, v in feats.items():
            if not isinstance(v, (int, float)):
                raise ValueError(f"Feature '{k}' must be numeric (int/float), got {type(v).__name__}")
            if math.isnan(v) or math.isinf(v):
                non_finite.append(k)

        if non_finite:
            raise ValueError(f"Non-finite numeric values (NaN or Inf) detected in features: {non_finite}")

        return self

class BatchFlowInput(BaseModel):
    records: List[FlowRecordInput] = Field(..., description="List of flow records to predict")

class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 for BENIGN, 1 for DDoS")
    label: str = Field(..., description="'BENIGN' or 'DDoS'")
    probability: float = Field(..., description="Estimated model probability of DDoS")
    confidence_level: str = Field(..., description="'HIGH', 'MEDIUM', or 'LOW'")
    inference_latency_ms: float = Field(..., description="Inference latency in milliseconds")
    model_name: str = Field(..., description="Model used for prediction")
    calibrated_notice: str = Field(
        "Probability represents tree ensemble log-odds sigmoid output for CICIDS2017 feature distribution.",
        description="Calibration transparency note"
    )

class BatchPredictionResponse(BaseModel):
    total_records: int
    predictions: List[PredictionResponse]
    batch_processing_time_ms: float
