import json
import logging

from src.mlops.logging_config import JsonFormatter


def test_json_formatter_includes_operational_fields():
    record = logging.LogRecord("test", logging.INFO, __file__, 1, "request completed", args=(), exc_info=None)
    record.request_id = "req-1"
    record.model_version = "v0.2.0"
    record.endpoint = "/api/v1/predictions"
    record.latency_ms = 10.5
    record.status_code = 200

    payload = json.loads(JsonFormatter().format(record))

    assert payload["request_id"] == "req-1"
    assert payload["model_version"] == "v0.2.0"
    assert payload["endpoint"] == "/api/v1/predictions"
    assert payload["status_code"] == 200
