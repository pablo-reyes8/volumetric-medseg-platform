from __future__ import annotations

import time
from typing import Iterable, Tuple


try:
    from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, Info, generate_latest
except ImportError:  # pragma: no cover - only used in minimal environments
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4; charset=utf-8"
    CollectorRegistry = None
    Counter = Gauge = Histogram = Info = None
    generate_latest = None


REGISTRY = CollectorRegistry() if CollectorRegistry else None
_FALLBACK_COUNTERS: dict[str, float] = {}


def _metric(factory, *args, **kwargs):
    if factory is None:
        return None
    return factory(*args, registry=REGISTRY, **kwargs)


REQUESTS_TOTAL = _metric(
    Counter,
    "medseg_requests_total",
    "Total HTTP requests processed by the segmentation API.",
    ["method", "path", "status_code"],
)
ERRORS_TOTAL = _metric(
    Counter,
    "medseg_errors_total",
    "Total HTTP error responses emitted by the segmentation API.",
    ["method", "path", "status_code"],
)
REQUEST_LATENCY_SECONDS = _metric(
    Histogram,
    "medseg_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
)
INFERENCE_LATENCY_SECONDS = _metric(
    Histogram,
    "medseg_inference_latency_seconds",
    "Model inference latency in seconds.",
)
MODEL_LOAD_TIMESTAMP = _metric(
    Gauge,
    "medseg_model_load_timestamp",
    "Unix timestamp of the latest successful model load.",
)
MODEL_VERSION_INFO = _metric(
    Info,
    "medseg_model_version",
    "Currently served model metadata.",
)
PREDICTION_VOLUME_SHAPE = _metric(
    Gauge,
    "medseg_prediction_volume_shape",
    "Last predicted input volume shape by axis.",
    ["axis"],
)
MEMORY_USAGE_BYTES = _metric(
    Gauge,
    "medseg_memory_usage_bytes",
    "Resident memory usage in bytes.",
)


def record_http_request(method: str, path: str, status_code: int, latency_ms: float) -> None:
    if REQUESTS_TOTAL is None:
        _FALLBACK_COUNTERS["medseg_requests_total"] = _FALLBACK_COUNTERS.get("medseg_requests_total", 0.0) + 1.0
        _FALLBACK_COUNTERS["medseg_request_latency_seconds_sum"] = (
            _FALLBACK_COUNTERS.get("medseg_request_latency_seconds_sum", 0.0) + max(0.0, latency_ms / 1000.0)
        )
        if status_code >= 400:
            _FALLBACK_COUNTERS["medseg_errors_total"] = _FALLBACK_COUNTERS.get("medseg_errors_total", 0.0) + 1.0
        return
    status = str(status_code)
    REQUESTS_TOTAL.labels(method=method, path=path, status_code=status).inc()
    REQUEST_LATENCY_SECONDS.labels(method=method, path=path).observe(max(0.0, latency_ms / 1000.0))
    if status_code >= 400:
        ERRORS_TOTAL.labels(method=method, path=path, status_code=status).inc()


def record_prediction(input_shape: Iterable[int], inference_ms: float) -> None:
    if INFERENCE_LATENCY_SECONDS is None:
        _FALLBACK_COUNTERS["medseg_inference_latency_seconds_sum"] = (
            _FALLBACK_COUNTERS.get("medseg_inference_latency_seconds_sum", 0.0) + max(0.0, inference_ms / 1000.0)
        )
        for axis, value in zip(("x", "y", "z"), input_shape):
            _FALLBACK_COUNTERS[f'medseg_prediction_volume_shape{{axis="{axis}"}}'] = float(value)
        return
    INFERENCE_LATENCY_SECONDS.observe(max(0.0, inference_ms / 1000.0))
    for axis, value in zip(("x", "y", "z"), input_shape):
        PREDICTION_VOLUME_SHAPE.labels(axis=axis).set(float(value))


def record_model_loaded(model_version: str = "unknown") -> None:
    if MODEL_LOAD_TIMESTAMP is None:
        return
    MODEL_LOAD_TIMESTAMP.set(time.time())
    MODEL_VERSION_INFO.info({"version": str(model_version)})


def record_memory_usage(memory_bytes: int) -> None:
    if MEMORY_USAGE_BYTES is not None:
        MEMORY_USAGE_BYTES.set(float(memory_bytes))


def render_metrics() -> bytes:
    if generate_latest is None:
        lines = ["# prometheus_client is not installed; using fallback metrics"]
        for key, value in sorted(_FALLBACK_COUNTERS.items()):
            lines.append(f"{key} {value}")
        return ("\n".join(lines) + "\n").encode("utf-8")
    return generate_latest(REGISTRY)


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
