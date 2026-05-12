#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from mlops_cli_utils import read_json, sha256_file, write_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a governed local model candidate package.")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--evaluation-report", required=True)
    parser.add_argument("--gate-decision", required=True)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    for source, destination_name in [
        (args.dataset_manifest, "dataset_manifest.json"),
        (args.training_config, "training_config.yaml"),
        (args.evaluation_report, "evaluation_report.json"),
        (args.gate_decision, "gate_decision.json"),
    ]:
        source_path = Path(source)
        if source_path.exists() and source_path.resolve() != (model_dir / destination_name).resolve():
            shutil.copy2(source_path, model_dir / destination_name)
    training_report = read_json(model_dir / "training_report.json")
    mlflow_run = read_json(model_dir / "mlflow_run.json")
    gate_decision = read_json(args.gate_decision)
    eval_report = read_json(args.evaluation_report)
    checkpoint = model_dir / "checkpoint.pt"
    manifest = {
        "model_name": "unet3d_hippocampus",
        "model_version": args.model_version,
        "stage": "candidate",
        "status": "candidate",
        "created_at": training_report.get("created_at"),
        "created_by": "airflow",
        "dag_id": training_report.get("dag_id"),
        "dag_run_id": training_report.get("dag_run_id"),
        "git_commit": training_report.get("git_commit"),
        "dataset": {
            "manifest_path": args.dataset_manifest,
            "manifest_hash": sha256_file(args.dataset_manifest),
        },
        "training": {
            "config_path": args.training_config,
            "config_hash": sha256_file(args.training_config),
            "checkpoint_path": str(checkpoint),
        },
        "mlflow": {
            "tracking_uri": mlflow_run.get("tracking_uri"),
            "experiment_name": mlflow_run.get("experiment_name"),
            "run_id": mlflow_run.get("run_id"),
        },
        "evaluation": {
            "report_path": args.evaluation_report,
            "gates_path": args.gate_decision,
            "passed_gates": gate_decision.get("passed"),
            "primary_metric": "mean_dice",
            "primary_metric_value": eval_report.get("metrics", {}).get("mean_dice"),
        },
        "serving": {
            "eligible_for_serving": bool(gate_decision.get("passed")),
            "promoted_at": None,
            "deployment_record_path": None,
        },
        "mlflow_run_id": mlflow_run.get("run_id"),
        "dataset_manifest_hash": sha256_file(args.dataset_manifest),
        "evaluation_report_path": args.evaluation_report,
        "checkpoint_path": str(checkpoint),
        "metrics": eval_report.get("metrics", {}),
    }
    write_yaml(model_dir / "model_package.yaml", manifest)
    (model_dir / "README.md").write_text(f"# Candidate {args.model_version}\n\nGate decision: {gate_decision.get('decision')}\n", encoding="utf-8")


if __name__ == "__main__":
    main()

