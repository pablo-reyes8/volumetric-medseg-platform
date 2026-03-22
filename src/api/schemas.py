from datetime import datetime
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class ServiceInfoResponse(BaseModel):
    service: str
    version: str
    docs_url: str
    openapi_url: str
    health_url: str
    prediction_url: str


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    service: str
    version: str
    timestamp_utc: datetime
    device: str = Field(..., example="cuda")
    model_loaded: bool = Field(..., example=True)
    model_path: str
    detail: Optional[str] = None


class ReadinessResponse(HealthResponse):
    ready: bool
    checks: Dict[str, str]


class MaintainerResponse(BaseModel):
    name: str
    email: Optional[str] = None


class ModelMetadata(BaseModel):
    name: str
    version: str
    description: str
    framework: str
    task: str
    checkpoint: str
    model_path: str
    model_loaded: bool
    num_classes: int
    class_names: List[str]
    in_channels: int
    base_channels: int
    norm: str
    dropout: float
    pad_multiple: int
    clip_percentiles: Tuple[int, int]
    default_threshold: float
    supported_extensions: List[str]
    device: str
    maintainers: List[MaintainerResponse] = Field(default_factory=list)


class RuntimeConfigResponse(BaseModel):
    server_host: str
    server_port: int
    docs_url: str
    redoc_url: str
    openapi_url: str
    allow_origins: List[str]
    preload_model: bool
    default_threshold: float
    pad_multiple: int
    clip_percentiles: Tuple[int, int]
    supported_extensions: List[str]


class RuntimeBreakdown(BaseModel):
    total_ms: float
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float


class PreprocessingSummary(BaseModel):
    clip_percentiles: Tuple[int, int]
    pad_multiple: int
    input_intensity_range: Tuple[float, float]


class PredictionStats(BaseModel):
    voxel_count: int
    labels_present: List[int]
    class_histogram: Dict[int, int]
    class_ratios: Dict[int, float]


class PredictionResponse(BaseModel):
    request_id: str
    input_filename: str
    output_filename: str
    input_shape: Tuple[int, int, int]
    padded_shape: Tuple[int, int, int]
    voxel_spacing: Tuple[float, float, float]
    orientation: Tuple[str, str, str]
    device: str
    threshold_used: float
    runtime: RuntimeBreakdown
    preprocessing: PreprocessingSummary
    stats: PredictionStats


class ModelReloadResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    device: str


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None

