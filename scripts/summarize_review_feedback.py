#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter

from mlops_cli_utils import add_lineage, write_json
from src.mlops.feedback_logging import load_feedback_records, summarize_feedback


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local review feedback JSONL.")
    parser.add_argument("--feedback-log", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    records = load_feedback_records(args.feedback_log)
    summary = summarize_feedback(records)
    summary.update(
        {
            "status": "completed",
            "rejected_predictions": sum(1 for record in records if record.get("accepted") is False),
            "model_versions": dict(Counter(str(record.get("model_version", "unknown")) for record in records)),
        }
    )
    write_json(args.output, add_lineage(summary, dag_id=args.dag_id, dag_run_id=args.dag_run_id, task_id="summarize_review_feedback"))


if __name__ == "__main__":
    main()

