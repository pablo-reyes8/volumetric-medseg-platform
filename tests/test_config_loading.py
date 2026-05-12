import json
from pathlib import Path

import yaml


def test_new_config_files_are_parseable():
    for path in [
        Path("docker-compose.yml"),
        Path("monitoring/prometheus/prometheus.yml"),
        Path("monitoring/grafana/provisioning/datasources/prometheus.yaml"),
        Path("monitoring/grafana/provisioning/dashboards/dashboard.yaml"),
    ]:
        assert yaml.safe_load(path.read_text(encoding="utf-8")) is not None

    for path in [
        Path("schemas/evaluation_report.schema.json"),
        Path("schemas/review_feedback.schema.json"),
        Path("monitoring/grafana/dashboards/medseg_runtime_dashboard.json"),
    ]:
        assert json.loads(path.read_text(encoding="utf-8"))
