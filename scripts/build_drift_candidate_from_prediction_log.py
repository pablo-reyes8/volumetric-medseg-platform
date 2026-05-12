#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import add_lineage, write_json
from src.mlops.prediction_logging import load_prediction_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight drift candidate profile from prediction metadata.")
    parser.add_argument("--prediction-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    records = load_prediction_records(args.prediction_log)
    means = [float(record.get("intensity_mean", 0.0)) for record in records]
    foreground = [float(record.get("foreground_ratio", 0.0)) for record in records]
    profile = {
        "status": "completed",
        "num_predictions": len(records),
        "summary": {
            "intensity_mean": round(sum(means) / len(means), 6) if means else 0.0,
            "foreground_ratio_mean": round(sum(foreground) / len(foreground), 6) if foreground else 0.0,
        },
    }
    write_json(args.output, add_lineage(profile, dag_id=args.dag_id, dag_run_id=args.dag_run_id, task_id="compute_input_drift"))


if __name__ == "__main__":
    main()

