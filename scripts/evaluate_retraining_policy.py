#!/usr/bin/env python3
from __future__ import annotations

import argparse

from mlops_cli_utils import add_lineage, read_json, write_json
from src.mlops.retraining import evaluate_retraining_recommendations, load_operating_policy


def _decision(actions: list[str]) -> str:
    if "rollback" in actions:
        return "rollback"
    if any(action.endswith("retrain") for action in actions):
        return "retrain"
    if "investigate_capacity" in actions:
        return "investigate"
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retraining policy from monitoring artifacts.")
    parser.add_argument("--runtime-report", required=True)
    parser.add_argument("--prediction-summary", required=True)
    parser.add_argument("--feedback-summary", required=True)
    parser.add_argument("--drift-report", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dag-id")
    parser.add_argument("--dag-run-id")
    args = parser.parse_args()

    raw_assessment = evaluate_retraining_recommendations(
        policy=load_operating_policy(args.policy),
        drift_report=read_json(args.drift_report),
        runtime_snapshot=read_json(args.runtime_report),
        feedback_snapshot=read_json(args.feedback_summary),
    )
    actions = raw_assessment.get("recommended_actions", [])
    decision = _decision(actions)
    assessment = {
        "status": "completed",
        "decision": decision,
        "severity": "medium" if decision != "none" else "low",
        "reasons": raw_assessment.get("reasons", []),
        "recommended_actions": actions,
        "rollback_recommended": decision == "rollback",
        "retraining_recommended": decision == "retrain",
        "capacity_investigation_recommended": decision == "investigate",
        "raw_assessment": raw_assessment,
    }
    write_json(args.output, add_lineage(assessment, dag_id=args.dag_id, dag_run_id=args.dag_run_id, task_id="evaluate_operating_policy"))


if __name__ == "__main__":
    main()

