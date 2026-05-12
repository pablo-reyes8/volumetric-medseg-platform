from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "service": getattr(record, "service", "fastapi-inference"),
            "message": record.getMessage(),
        }
        for key in ["request_id", "model_version", "endpoint", "latency_ms", "status_code"]:
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True)


def configure_json_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def log_request(
    logger: logging.Logger,
    request_id: str,
    endpoint: str,
    latency_ms: float,
    status_code: int,
    model_version: Optional[str] = None,
) -> None:
    logger.info(
        "request completed",
        extra={
            "request_id": request_id,
            "endpoint": endpoint,
            "latency_ms": round(latency_ms, 3),
            "status_code": status_code,
            "model_version": model_version,
        },
    )
