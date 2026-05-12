from src.mlops.prometheus_exporter import record_http_request, record_prediction, render_metrics


def test_prometheus_exporter_records_request_and_prediction_metrics():
    record_http_request("GET", "/health/live", 200, 12.0)
    record_prediction((6, 7, 5), 34.0)
    payload = render_metrics().decode("utf-8")
    assert "medseg_requests_total" in payload
    assert "medseg_request_latency_seconds" in payload
    assert "medseg_inference_latency_seconds" in payload
