#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from collections import Counter

from mlops_cli_utils import add_lineage, write_json
from src.mlops.prediction_logging import load_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize prediction metadata JSONL for monitoring.")
    parser.add_argument("--prediction-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    records = load_prediction_records(args.prediction_log)
    latencies = [float(record.get("latency_ms", 0.0)) for record in records]
    foreground = [float(record.get("foreground_ratio", 0.0)) for record in records]
    summary = {
        "status": "completed",
        "prediction_count": len(records),
        "error_count": sum(1 for record in records if record.get("status") != "success"),
        "latency_ms_p50": statistics.median(latencies) if latencies else 0.0,
        "latency_ms_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
        "foreground_ratio_mean": round(sum(foreground) / len(foreground), 6) if foreground else 0.0,
        "empty_prediction_rate": round(sum(1 for value in foreground if value <= 0.0) / len(foreground), 6) if foreground else 0.0,
        "model_versions": dict(Counter(str(record.get("model_version", "unknown")) for record in records)),
        "request_id_coverage": round(sum(1 for record in records if record.get("request_id")) / max(1, len(records)), 6),
    }
    write_json(args.output, add_lineage(summary, dag_id=args.dag_id, dag_run_id=args.dag_run_id, task_id="summarize_prediction_metadata"))


if __name__ == "__main__":
    main()

