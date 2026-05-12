#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from urllib.error import URLError
from urllib.request import urlopen

from mlops_cli_utils import add_lineage, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect FastAPI runtime metrics into a JSON artifact.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-offline", action="store_true")
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    try:
        with urlopen(f"{args.api_url.rstrip('/')}/api/v1/monitoring/runtime", timeout=5) as response:  # nosec - local service
            payload = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        if not args.allow_offline:
            raise
        payload = {
            "status": "offline",
            "error": str(exc),
            "totals": {"requests": 0, "error_rate": 0.0},
            "latency_ms": {"p95": 0.0},
            "throughput": {"requests_per_minute": 0.0},
        }
    write_json(args.output, add_lineage(payload, dag_id=args.dag_id, dag_run_id=args.dag_run_id, task_id="collect_runtime_metrics"))


if __name__ == "__main__":
    main()

