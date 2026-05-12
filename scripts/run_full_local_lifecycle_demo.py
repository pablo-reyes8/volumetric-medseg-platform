#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from mlops_cli_utils import now_utc, write_json
from src.mlops.feedback_logging import append_feedback_record, build_feedback_record
from src.mlops.prediction_logging import append_prediction_record, build_prediction_record


def _run(command: list[str]) -> None:
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        if completed.stdout:
            print(completed.stdout)
        if completed.stderr:
            print(completed.stderr, file=sys.stderr)
        raise SystemExit(completed.returncode)


def _write_demo_manifest(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        path,
        {
            "schema_version": "1.0",
            "dataset_name": "demo_medseg",
            "version": "smoke",
            "created_at_utc": now_utc(),
            "images_dir": "data/demo/images",
            "labels_dir": "data/demo/labels",
            "total_pairs": 1,
            "summary": {},
            "quality_report": {"status": "pass", "passed_pairs": 1, "failed_pairs": 0, "rules": {}, "issues": []},
            "records": [],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a CPU-safe full local MLOps lifecycle demo.")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--model-version", default="demo_smoke")
    parser.add_argument("--dataset-manifest")
    parser.add_argument("--training-config", default="configs/training/local_unet3d_smoke.yaml")
    parser.add_argument("--evaluation-gates", default="configs/mlops/evaluation_gates.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.artifact_root)
    model_dir = root / "models" / args.model_version
    report_dir = root / "reports" / "airflow"
    manifest = Path(args.dataset_manifest) if args.dataset_manifest else root / "airflow" / "demo_dataset_manifest.json"
    _write_demo_manifest(manifest)

    _run([sys.executable, "scripts/train_unet3d_smoke.py", "--output-dir", str(model_dir), "--config", args.training_config, "--dataset-manifest", str(manifest), "--model-version", args.model_version])
    _run([sys.executable, "scripts/evaluate_model.py", "--model-package-dir", str(model_dir), "--dataset-manifest", str(manifest), "--output-json", str(model_dir / "evaluation_report.json"), "--output-md", str(model_dir / "evaluation_report.md")])
    _run([sys.executable, "scripts/apply_evaluation_gates.py", "--evaluation-report", str(model_dir / "evaluation_report.json"), "--gates", args.evaluation_gates, "--output", str(model_dir / "gate_decision.json")])
    _run([sys.executable, "scripts/package_model_candidate.py", "--model-version", args.model_version, "--model-dir", str(model_dir), "--dataset-manifest", str(manifest), "--training-config", args.training_config, "--evaluation-report", str(model_dir / "evaluation_report.json"), "--gate-decision", str(model_dir / "gate_decision.json")])
    _run([sys.executable, "scripts/register_model_candidate.py", "--model-package", str(model_dir / "model_package.yaml"), "--artifact-root", str(root)])
    if not args.dry_run:
        _run([sys.executable, "scripts/promote_model.py", "--candidate-version", args.model_version, "--require-eval-pass", "--artifact-root", str(root)])

    prediction_log = root / "predictions" / "prediction_log.jsonl"
    feedback_log = root / "feedback" / "review_feedback.jsonl"
    append_prediction_record(prediction_log, build_prediction_record("demo-request", args.model_version, [8, 8, 8], (0.0, 1.0), {0: 0.92, 1: 0.08}, 12.0))
    append_feedback_record(feedback_log, build_feedback_record("demo-request", args.model_version, True, 5))
    summary = report_dir / f"full_lifecycle_demo_{args.model_version}.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        "\n".join(
            [
                "# Full Local MLOps Lifecycle Demo",
                "",
                "## Result",
                "PASSED",
                "",
                "## Artifacts",
                f"- Dataset manifest: {manifest}",
                f"- Candidate package: {model_dir / 'model_package.yaml'}",
                f"- Evaluation report: {model_dir / 'evaluation_report.json'}",
                f"- Gate decision: {model_dir / 'gate_decision.json'}",
                f"- Prediction metadata: {prediction_log}",
                f"- Feedback log: {feedback_log}",
                "",
                "## Lineage",
                "Dataset -> Model -> Registry -> Serving -> Monitoring -> Decision",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "passed", "summary": str(summary), "model_dir": str(model_dir)}, indent=2))


if __name__ == "__main__":
    main()
