import io
import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from src.api.inference_service import PredictionResult, SegmentationService
from src.api.model_card import extract_maintainers, load_model_card
from src.api.schemas import (
    EndpointMonitoringResponse,
    ErrorResponse,
    HealthResponse,
    ModelMetadata,
    ModelReloadResponse,
    MonitoringRuntimeResponse,
    OperatingPolicyResponse,
    PredictionResponse,
    PredictionStats,
    PreprocessingSummary,
    ReadinessResponse,
    RetrainingAssessmentResponse,
    RuntimeBreakdown,
    RuntimeConfigResponse,
    ServiceInfoResponse,
)
from src.api.settings import Settings
from src.mlops.retraining import evaluate_retraining_recommendations, load_operating_policy, summarize_monitored_signals
from src.mlops.runtime_monitoring import RuntimeMonitor

logger = logging.getLogger("unet3d.api")


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> SegmentationService:
    return request.app.state.service


def get_runtime_monitor(request: Request) -> RuntimeMonitor:
    return request.app.state.runtime_monitor


def _ensure_nifti(filename: Optional[str], settings: Settings) -> None:
    allowed = settings.supported_extensions
    if not filename or not any(filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Formato no soportado. Usa archivos NIfTI ({', '.join(allowed)}).",
        )


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown-request")


def _error_response(status_code: int, error: str, detail: str, request_id: str) -> JSONResponse:
    payload = ErrorResponse(error=error, detail=detail, request_id=request_id)
    return JSONResponse(status_code=status_code, content=payload.dict(exclude_none=True))


async def _persist_upload(file: UploadFile) -> Path:
    suffix = ".nii.gz" if file.filename and file.filename.lower().endswith(".nii.gz") else ".nii"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        shutil.copyfileobj(file.file, tmp)
    await file.close()
    return tmp_path


def _build_prediction_response(request_id: str, result: PredictionResult, settings: Settings) -> PredictionResponse:
    return PredictionResponse(
        request_id=request_id,
        input_filename=result.input_filename,
        output_filename=result.output_filename,
        input_shape=result.input_shape,
        padded_shape=result.padded_shape,
        voxel_spacing=result.voxel_spacing,
        orientation=result.orientation,
        device=result.device,
        threshold_used=result.threshold_used,
        runtime=RuntimeBreakdown(
            total_ms=result.total_runtime_ms,
            preprocess_ms=result.preprocess_ms,
            inference_ms=result.inference_ms,
            postprocess_ms=result.postprocess_ms,
        ),
        preprocessing=PreprocessingSummary(
            clip_percentiles=settings.clip_percentiles,
            pad_multiple=settings.pad_multiple,
            input_intensity_range=result.intensity_range,
        ),
        stats=PredictionStats(
            voxel_count=result.voxel_count,
            labels_present=result.labels_present,
            class_histogram=result.class_histogram,
            class_ratios=result.class_ratios,
        ),
    )


async def _run_prediction(file: UploadFile, threshold: Optional[float], service: SegmentationService, settings: Settings) -> PredictionResult:
    _ensure_nifti(file.filename, settings)
    tmp_path = await _persist_upload(file)
    try:
        return service.predict(tmp_path, threshold=threshold)
    finally:
        tmp_path.unlink(missing_ok=True)


def create_app(app_settings: Optional[Settings] = None) -> FastAPI:
    settings = app_settings or Settings()
    logging.basicConfig(level=getattr(logging, settings.log_level))
    service = SegmentationService(settings=settings)
    runtime_monitor = RuntimeMonitor(settings=settings)
    operating_policy = load_operating_policy(settings.operating_policy_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.service = service
        app.state.runtime_monitor = runtime_monitor
        app.state.operating_policy = operating_policy
        if settings.preload_model:
            try:
                model = service.load_model()
                logger.info("Modelo cargado en %s", service.device)
                logger.info("Pesos: %s", settings.model_path)
                logger.info("Salidas: %s clases", getattr(model, "out_conv").out_channels)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("No se pudo cargar el modelo en startup: %s", exc)
        yield

    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        openapi_url=settings.openapi_url,
        contact={"name": settings.contact_name, "email": settings.contact_email},
        license_info={"name": settings.license_name},
        openapi_tags=[
            {"name": "platform", "description": "Descubrimiento del servicio y runtime config."},
            {"name": "monitoring", "description": "Liveness, readiness y estado operativo."},
            {"name": "model", "description": "Metadatos, configuración y recarga del modelo."},
            {"name": "inference", "description": "Inferencia JSON y descarga de máscaras NIfTI."},
        ],
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", str(uuid4()))
        started_at = datetime.now(timezone.utc)
        start_perf = time.perf_counter()
        response = await call_next(request)
        latency_ms = max(0.0, (time.perf_counter() - start_perf) * 1000.0)
        request.app.state.runtime_monitor.record_request(
            path=request.url.path,
            method=request.method,
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-Processed-At"] = started_at.isoformat()
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return _error_response(exc.status_code, "request_error", detail, _request_id(request))

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError) -> JSONResponse:
        return _error_response(status.HTTP_404_NOT_FOUND, "file_not_found", str(exc), _request_id(request))

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return _error_response(status.HTTP_422_UNPROCESSABLE_ENTITY, "validation_error", str(exc), _request_id(request))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error [%s]: %s", _request_id(request), exc)
        return _error_response(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_server_error",
            "Unexpected server error during request processing.",
            _request_id(request),
        )

    @app.get("/", response_model=ServiceInfoResponse, tags=["platform"], summary="Descubrimiento del servicio")
    async def root() -> ServiceInfoResponse:
        return ServiceInfoResponse(
            service=settings.app_name,
            version=settings.app_version,
            docs_url=settings.docs_url,
            openapi_url=settings.openapi_url,
            health_url="/health/ready",
            prediction_url="/api/v1/predictions",
        )

    def _health_payload(current_service: SegmentationService, ready: bool, detail: Optional[str] = None) -> HealthResponse:
        return HealthResponse(
            status="ok" if ready else "degraded",
            service=settings.app_name,
            version=settings.app_version,
            timestamp_utc=_now_utc(),
            device=str(current_service.device),
            model_loaded=current_service.model_ready,
            model_path=str(settings.model_path),
            detail=detail,
        )

    @app.get("/health", response_model=ReadinessResponse, tags=["monitoring"], summary="Estado general de la API")
    async def health(
        current_service: SegmentationService = Depends(get_service),
    ) -> ReadinessResponse:
        weights_available = settings.model_path.exists()
        ready = current_service.model_ready and weights_available
        detail = None if ready else "El modelo no esta listo; revisa el checkpoint o recarga el servicio."
        payload = _health_payload(current_service, ready=ready, detail=detail)
        return ReadinessResponse(
            **payload.dict(),
            ready=ready,
            checks={
                "weights_path": "ok" if weights_available else "missing",
                "model_loaded": "ok" if current_service.model_ready else "not_loaded",
            },
        )

    @app.get("/health/live", response_model=HealthResponse, tags=["monitoring"], summary="Liveness probe")
    async def health_live(
        current_service: SegmentationService = Depends(get_service),
    ) -> HealthResponse:
        return _health_payload(current_service, ready=True, detail=None)

    @app.get("/health/ready", response_model=ReadinessResponse, tags=["monitoring"], summary="Readiness probe")
    async def health_ready(
        current_service: SegmentationService = Depends(get_service),
    ) -> ReadinessResponse:
        return await health(current_service)

    @app.get("/api/v1/model", response_model=ModelMetadata, tags=["model"], summary="Metadatos del modelo")
    async def model_metadata(
        current_service: SegmentationService = Depends(get_service),
        current_settings: Settings = Depends(get_settings),
    ) -> ModelMetadata:
        model_card = load_model_card(current_settings.model_card_path)
        inference = model_card.get("inference", {})
        return ModelMetadata(
            name=model_card.get("name", "unet3d-segmentation"),
            version=model_card.get("version", current_settings.app_version),
            description=model_card.get("description", current_settings.app_description),
            framework=model_card.get("framework", "pytorch"),
            task=model_card.get("task", "3d_segmentation"),
            checkpoint=inference.get("checkpoint", str(current_settings.model_path)),
            model_path=str(current_settings.model_path),
            model_loaded=current_service.model_ready,
            num_classes=current_settings.num_classes,
            class_names=current_settings.class_names,
            in_channels=current_settings.in_channels,
            base_channels=current_settings.base_channels,
            norm=current_settings.norm,
            dropout=current_settings.dropout,
            pad_multiple=current_settings.pad_multiple,
            clip_percentiles=current_settings.clip_percentiles,
            default_threshold=current_settings.default_threshold,
            supported_extensions=list(current_settings.supported_extensions),
            device=str(current_service.device),
            maintainers=extract_maintainers(model_card),
        )

    @app.get("/api/v1/config", response_model=RuntimeConfigResponse, tags=["platform"], summary="Runtime config")
    async def runtime_config(
        current_settings: Settings = Depends(get_settings),
    ) -> RuntimeConfigResponse:
        return RuntimeConfigResponse(
            server_host=current_settings.server_host,
            server_port=current_settings.server_port,
            docs_url=current_settings.docs_url,
            redoc_url=current_settings.redoc_url,
            openapi_url=current_settings.openapi_url,
            allow_origins=current_settings.allow_origins,
            preload_model=current_settings.preload_model,
            default_threshold=current_settings.default_threshold,
            pad_multiple=current_settings.pad_multiple,
            clip_percentiles=current_settings.clip_percentiles,
            supported_extensions=list(current_settings.supported_extensions),
            monitoring_window_seconds=current_settings.monitoring_window_seconds,
            cpu_hourly_cost_usd=current_settings.cpu_hourly_cost_usd,
            gpu_hourly_cost_usd=current_settings.gpu_hourly_cost_usd,
            memory_gb_hourly_cost_usd=current_settings.memory_gb_hourly_cost_usd,
        )

    @app.get(
        "/api/v1/monitoring/runtime",
        response_model=MonitoringRuntimeResponse,
        tags=["monitoring"],
        summary="Monitoreo runtime del servicio de inferencia",
    )
    async def monitoring_runtime(
        monitor: RuntimeMonitor = Depends(get_runtime_monitor),
    ) -> MonitoringRuntimeResponse:
        snapshot = monitor.snapshot()
        snapshot["endpoints"] = {
            key: EndpointMonitoringResponse(**value) for key, value in snapshot.get("endpoints", {}).items()
        }
        return MonitoringRuntimeResponse(**snapshot)

    @app.get(
        "/api/v1/monitoring/policy",
        response_model=OperatingPolicyResponse,
        tags=["monitoring"],
        summary="Politica operativa de monitoreo, retraining y rollback",
    )
    async def monitoring_policy(request: Request) -> OperatingPolicyResponse:
        policy = request.app.state.operating_policy
        return OperatingPolicyResponse(
            policy_path=str(settings.operating_policy_path),
            monitored_signals=summarize_monitored_signals(policy)["tracked_signals"],
            monitoring_thresholds=summarize_monitored_signals(policy)["slo_thresholds"],
            retraining=policy.get("retraining", {}),
            rollback=policy.get("rollback", {}),
        )

    @app.get(
        "/api/v1/monitoring/retraining-assessment",
        response_model=RetrainingAssessmentResponse,
        tags=["monitoring"],
        summary="Recomendaciones de retraining y rollback basadas en la politica operativa",
    )
    async def retraining_assessment(
        request: Request,
        monitor: RuntimeMonitor = Depends(get_runtime_monitor),
    ) -> RetrainingAssessmentResponse:
        assessment = evaluate_retraining_recommendations(
            policy=request.app.state.operating_policy,
            runtime_snapshot=monitor.snapshot(),
        )
        return RetrainingAssessmentResponse(**assessment)

    @app.post("/api/v1/model/reload", response_model=ModelReloadResponse, tags=["model"], summary="Recargar checkpoint")
    async def reload_model(
        current_service: SegmentationService = Depends(get_service),
        current_settings: Settings = Depends(get_settings),
    ) -> ModelReloadResponse:
        current_service.reload_model()
        return ModelReloadResponse(
            status="reloaded",
            model_loaded=current_service.model_ready,
            model_path=str(current_settings.model_path),
            device=str(current_service.device),
        )

    @app.post(
        "/api/v1/predictions",
        response_model=PredictionResponse,
        responses={400: {"model": ErrorResponse}, 404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
        tags=["inference"],
        summary="Segmenta un volumen NIfTI y devuelve metadatos en JSON",
    )
    async def predict(
        request: Request,
        file: UploadFile = File(..., description="Archivo NIfTI (.nii o .nii.gz)"),
        threshold: Optional[float] = Query(
            None, ge=0.0, le=1.0, description="Umbral opcional para segmentacion binaria."
        ),
        current_service: SegmentationService = Depends(get_service),
        current_settings: Settings = Depends(get_settings),
    ) -> PredictionResponse:
        result = await _run_prediction(file, threshold, current_service, current_settings)
        return _build_prediction_response(_request_id(request), result, current_settings)

    @app.post(
        "/api/v1/predictions/download",
        responses={
            200: {"content": {"application/gzip": {}}},
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
        tags=["inference"],
        summary="Segmenta un volumen NIfTI y devuelve la mascara para descarga",
    )
    async def predict_download(
        request: Request,
        file: UploadFile = File(..., description="Archivo NIfTI (.nii o .nii.gz)"),
        threshold: Optional[float] = Query(
            None, ge=0.0, le=1.0, description="Umbral opcional para segmentacion binaria."
        ),
        current_service: SegmentationService = Depends(get_service),
        current_settings: Settings = Depends(get_settings),
    ) -> StreamingResponse:
        result = await _run_prediction(file, threshold, current_service, current_settings)
        headers = {
            "Content-Disposition": f'attachment; filename="{result.output_filename}"',
            "X-Request-ID": _request_id(request),
            "X-UNET3D-Device": result.device,
        }
        return StreamingResponse(
            io.BytesIO(result.mask_bytes),
            media_type="application/gzip",
            headers=headers,
        )

    @app.post(
        "/v1/predict",
        deprecated=True,
        responses={
            200: {"model": PredictionResponse},
            400: {"model": ErrorResponse},
            404: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
        },
        tags=["inference"],
        summary="Endpoint legado compatible con versiones anteriores",
    )
    async def legacy_predict(
        request: Request,
        file: UploadFile = File(..., description="Archivo NIfTI (.nii o .nii.gz)"),
        return_binary: bool = Query(
            False,
            description="Si es true devuelve la mascara como archivo NIfTI en vez de JSON.",
        ),
        threshold: Optional[float] = Query(
            None, ge=0.0, le=1.0, description="Umbral opcional para segmentacion binaria."
        ),
        current_service: SegmentationService = Depends(get_service),
        current_settings: Settings = Depends(get_settings),
    ):
        result = await _run_prediction(file, threshold, current_service, current_settings)
        if return_binary:
            headers = {
                "Content-Disposition": f'attachment; filename="{result.output_filename}"',
                "X-Request-ID": _request_id(request),
            }
            return StreamingResponse(io.BytesIO(result.mask_bytes), media_type="application/gzip", headers=headers)
        return _build_prediction_response(_request_id(request), result, current_settings)

    return app


app = create_app()
