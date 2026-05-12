from pathlib import Path
from typing import List, Tuple

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Configuración central de la API y del servicio de inferencia."""

    app_name: str = Field(default="UNet3D Segmentation API", description="Nombre público del servicio.")
    app_version: str = Field(default="1.1.0", description="Versión de la API.")
    app_description: str = Field(
        default=(
            "Servicio de inferencia para segmentación médica volumétrica con UNet3D. "
            "Acepta volúmenes NIfTI 3D, expone metadatos del modelo y devuelve máscaras listas para descarga."
        ),
        description="Descripción usada por OpenAPI y clientes.",
    )
    contact_name: str = Field(default="ML Engineering Team", description="Nombre de contacto para soporte.")
    contact_email: str = Field(default="ml-team@example.com", description="Correo de contacto para soporte.")
    license_name: str = Field(default="MIT", description="Licencia expuesta en OpenAPI.")
    log_level: str = Field(default="INFO", description="Nivel de logging para la API.")
    environment: str = Field(default="local", description="Runtime environment label.")
    artifact_root: Path = Field(default=Path("artifacts"), description="Local MLOps artifact root.")
    data_root: Path = Field(default=Path("data"), description="Local dataset root.")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000", description="MLflow tracking URI.")
    enable_runtime_monitoring: bool = Field(default=True, description="Enable in-process runtime monitoring.")
    prometheus_enabled: bool = Field(default=True, description="Expose Prometheus metrics at /metrics.")
    prediction_log_path: Path = Field(
        default=Path("artifacts/predictions/prediction_log.jsonl"),
        description="JSONL path for prediction metadata records.",
    )
    review_feedback_path: Path = Field(
        default=Path("artifacts/feedback/review_feedback.jsonl"),
        description="JSONL path for local review feedback records.",
    )
    preload_model: bool = Field(
        default=True,
        description="Si es true, intenta cargar el checkpoint al arrancar el servicio.",
    )
    model_card_path: Path = Field(
        default=Path("model.yaml"),
        description="Ruta al model card YAML con metadatos del proyecto.",
    )

    model_path: Path = Field(
        default=Path("artifacts/unet3d_best.pt"),
        description="Ruta al checkpoint .pt/.pth con los pesos entrenados.",
    )
    device: str = Field(
        default="auto",
        description="Dispositivo preferido ('auto', 'cuda', 'cpu').",
    )
    num_classes: int = Field(default=3, description="Numero de clases de salida.")
    in_channels: int = Field(default=1, description="Canales de entrada del modelo.")
    base_channels: int = Field(default=32, description="Canales base en el primer bloque de la U-Net.")
    norm: str = Field(default="in", description="Tipo de normalizacion usada al entrenar ('in'|'bn'|None).")
    dropout: float = Field(default=0.0, description="Tasa de dropout usada al entrenar el modelo.")

    pad_multiple: int = Field(
        default=16,
        description="Alinea D/H/W al multiplo indicado para evitar desajustes por downsampling.",
    )
    clip_percentiles: Tuple[int, int] = Field(
        default=(1, 99),
        description="Percentiles para recorte robusto previo a la normalizacion min-max.",
    )
    default_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Umbral por defecto para segmentacion binaria (sigmoid > threshold).",
    )
    class_names: List[str] = Field(
        default_factory=lambda: ["background", "class-1", "class-2"],
        description="Etiquetas legibles de cada clase (indice = canal de salida).",
    )

    server_host: str = Field(default="0.0.0.0", description="Host para uvicorn.")
    server_port: int = Field(default=8000, ge=1, le=65535, description="Puerto para uvicorn.")
    docs_url: str = Field(default="/docs", description="Ruta del Swagger UI.")
    redoc_url: str = Field(default="/redoc", description="Ruta de ReDoc UI.")
    openapi_url: str = Field(default="/openapi.json", description="Ruta del esquema OpenAPI.")

    allow_origins: List[str] = Field(
        default_factory=lambda: ["*"],
        description="CORS: origenes permitidos para el front.",
    )
    monitoring_window_seconds: int = Field(
        default=3600,
        ge=60,
        description="Ventana de agregacion para el monitoreo runtime en segundos.",
    )
    cpu_hourly_cost_usd: float = Field(
        default=0.25,
        ge=0.0,
        description="Costo horario estimado para serving CPU.",
    )
    gpu_hourly_cost_usd: float = Field(
        default=1.5,
        ge=0.0,
        description="Costo horario estimado para serving GPU.",
    )
    memory_gb_hourly_cost_usd: float = Field(
        default=0.02,
        ge=0.0,
        description="Costo horario estimado por GB RSS consumido.",
    )
    operating_policy_path: Path = Field(
        default=Path("src/mlops/policies/default_operating_policy.yaml"),
        description="Ruta a la politica operativa de monitoreo, retraining y rollback.",
    )

    class Config:
        env_prefix = "UNET3D_"
        env_file = ".env"
        case_sensitive = False

    @property
    def supported_extensions(self) -> Tuple[str, str]:
        return (".nii", ".nii.gz")

    @validator("device")
    def _device_validator(cls, value: str) -> str:
        allowed = {"auto", "cuda", "cpu"}
        value = value.lower()
        if value not in allowed:
            raise ValueError(f"device debe ser uno de {allowed}")
        return value

    @validator("log_level")
    def _log_level_validator(cls, value: str) -> str:
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}
        value = value.upper()
        if value not in allowed:
            raise ValueError(f"log_level debe ser uno de {allowed}")
        return value

    @validator("clip_percentiles")
    def _clip_percentiles_validator(cls, value: Tuple[int, int]) -> Tuple[int, int]:
        if len(value) != 2:
            raise ValueError("clip_percentiles debe contener exactamente dos valores")
        low, high = value
        if low < 0 or high > 100 or low >= high:
            raise ValueError("clip_percentiles debe cumplir 0 <= low < high <= 100")
        return value

    @validator("class_names")
    def _class_names_validator(cls, value: List[str], values) -> List[str]:
        num_classes = values.get("num_classes")
        if num_classes is not None and len(value) != num_classes:
            raise ValueError("class_names debe tener la misma longitud que num_classes")
        return value

