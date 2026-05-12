#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import add_lineage, read_json, read_yaml, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply configured evaluation gates to a candidate report.")
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--gates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    report = read_json(args.evaluation_report)
    gates = read_yaml(args.gates).get("gates", {}).get("required", {})
    metrics = report.get("metrics", {})
    checks = {
        "min_mean_dice": float(metrics.get("mean_dice", 0.0)) >= float(gates.get("min_mean_dice", 0.0)),
        "min_foreground_dice": float(metrics.get("foreground_dice", 0.0)) >= float(gates.get("min_foreground_dice", 0.0)),
        "max_validation_loss": float(metrics.get("validation_loss", 999.0)) <= float(gates.get("max_validation_loss", 999.0)),
        "max_empty_prediction_rate": float(metrics.get("empty_prediction_rate", 1.0)) <= float(gates.get("max_empty_prediction_rate", 1.0)),
        "max_inference_latency_ms_p95": float(metrics.get("inference_latency_ms_p95", 999999.0)) <= float(gates.get("max_inference_latency_ms_p95", 999999.0)),
    }
    failed = [name for name, passed in checks.items() if not passed]
    decision = add_lineage(
        {
            "model_version": report.get("model_version"),
            "passed": not failed,
            "failed_gates": failed,
            "warnings": [],
            "decision": "eligible_for_promotion" if not failed else "reject_candidate",
            "checks": checks,
        },
        dag_id=args.dag_id,
        dag_run_id=args.dag_run_id,
        task_id="apply_evaluation_gates",
    )
    write_json(args.output, decision)
    raise SystemExit(0 if not failed else 1)


if __name__ == "__main__":
    main()

