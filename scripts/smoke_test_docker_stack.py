#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from urllib.error import URLError
from urllib.request import urlopen


def _probe(name: str, url: str, timeout: float = 5.0) -> dict[str, object]:
    try:
        with urlopen(url, timeout=timeout) as response:  # nosec - local smoke check only
            body = response.read(256).decode("utf-8", errors="replace")
            return {"name": name, "url": url, "ok": 200 <= response.status < 500, "status": response.status, "body": body}
    except URLError as exc:
        return {"name": name, "url": url, "ok": False, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the local Docker MLOps stack.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--streamlit-url", default="http://localhost:8501/_stcore/health")
    parser.add_argument("--mlflow-url", default="http://localhost:5000")
    parser.add_argument("--prometheus-url", default="http://localhost:9090/-/ready")
    parser.add_argument("--grafana-url", default="http://localhost:3000/api/health")
    args = parser.parse_args()

    checks = [
        _probe("api_root", f"{args.api_url.rstrip('/')}/"),
        _probe("api_liveness", f"{args.api_url.rstrip('/')}/health/live"),
        _probe("api_readiness", f"{args.api_url.rstrip('/')}/health/ready"),
        _probe("api_model", f"{args.api_url.rstrip('/')}/api/v1/model"),
        _probe("api_metrics", f"{args.api_url.rstrip('/')}/metrics"),
        _probe("streamlit", args.streamlit_url),
        _probe("mlflow", args.mlflow_url),
        _probe("prometheus", args.prometheus_url),
        _probe("grafana", args.grafana_url),
    ]
    print(json.dumps({"checks": checks}, indent=2))
    raise SystemExit(0 if all(check["ok"] for check in checks) else 1)


if __name__ == "__main__":
    main()
