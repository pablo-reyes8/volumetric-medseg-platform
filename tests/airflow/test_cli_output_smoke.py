import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from tests.utils import write_nifti


def _run(args: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run([sys.executable, *args], check=False, text=True, capture_output=True)
    assert result.returncode == 0, f"{args}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    return result


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _prepare_dataset(tmp_path: Path) -> tuple[Path, Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    write_nifti(images / "case_001.nii.gz", np.ones((4, 4, 4), dtype=np.float32))
    write_nifti(labels / "case_001.nii.gz", np.ones((4, 4, 4), dtype=np.uint8))
    contract = tmp_path / "contract.yaml"
    contract.write_text(
        yaml.safe_dump({"label_schema": {"0": "background", "1": "roi"}}),
        encoding="utf-8",
    )
    return images, labels, contract


def _build_candidate(tmp_path: Path, artifact_root: Path, manifest: Path, version: str) -> Path:
    model_dir = artifact_root / "models" / version
    _run(
        [
            "scripts/train_unet3d_smoke.py",
            "--output-dir",
            str(model_dir),
            "--config",
            "configs/training/local_unet3d_smoke.yaml",
            "--dataset-manifest",
            str(manifest),
            "--model-version",
            version,
        ]
    )
    training_report = json.loads((model_dir / "training_report.json").read_text(encoding="utf-8"))
    assert training_report["metrics"]["mlp_accuracy"] >= 0.5
    assert (model_dir / "checkpoint.pt").exists()

    _run(
        [
            "scripts/evaluate_model.py",
            "--model-package-dir",
            str(model_dir),
            "--dataset-manifest",
            str(manifest),
            "--output-json",
            str(model_dir / "evaluation_report.json"),
            "--output-md",
            str(model_dir / "evaluation_report.md"),
        ]
    )
    _run(
        [
            "scripts/validate_json_schema.py",
            "--schema",
            "schemas/evaluation_report.schema.json",
            "--json",
            str(model_dir / "evaluation_report.json"),
            "--output",
            str(model_dir / "evaluation_schema_check.json"),
        ]
    )
    _run(
        [
            "scripts/apply_evaluation_gates.py",
            "--evaluation-report",
            str(model_dir / "evaluation_report.json"),
            "--gates",
            "configs/mlops/evaluation_gates.yaml",
            "--output",
            str(model_dir / "gate_decision.json"),
        ]
    )
    _run(
        [
            "scripts/validate_json_schema.py",
            "--schema",
            "schemas/gate_decision.schema.json",
            "--json",
            str(model_dir / "gate_decision.json"),
            "--output",
            str(model_dir / "gate_schema_check.json"),
        ]
    )
    _run(
        [
            "scripts/package_model_candidate.py",
            "--model-version",
            version,
            "--model-dir",
            str(model_dir),
            "--dataset-manifest",
            str(manifest),
            "--training-config",
            "configs/training/local_unet3d_smoke.yaml",
            "--evaluation-report",
            str(model_dir / "evaluation_report.json"),
            "--gate-decision",
            str(model_dir / "gate_decision.json"),
        ]
    )
    _run(
        [
            "scripts/register_model_candidate.py",
            "--model-package",
            str(model_dir / "model_package.yaml"),
            "--artifact-root",
            str(artifact_root),
        ]
    )
    assert (artifact_root / "registry" / "candidates" / f"{version}.json").exists()
    return model_dir


def test_airflow_cli_wrappers_produce_expected_artifacts(tmp_path: Path):
    artifact_root = tmp_path / "artifacts"
    images, labels, contract = _prepare_dataset(tmp_path)
    contract_report = artifact_root / "reports" / "contract.json"
    quality_report = artifact_root / "reports" / "quality.json"
    quality_schema_report = artifact_root / "reports" / "quality_schema.json"
    manifest = tmp_path / "manifest.json"
    registry = tmp_path / "datasets.yaml"

    _run(["scripts/validate_dataset_contract.py", "--contract-path", str(contract), "--images-dir", str(images), "--labels-dir", str(labels), "--output", str(contract_report)])
    _run(["scripts/run_data_quality.py", "--images-dir", str(images), "--labels-dir", str(labels), "--contract-path", str(contract), "--output", str(quality_report)])
    _run(["scripts/validate_json_schema.py", "--schema", "schemas/data_quality_report.schema.json", "--json", str(quality_report), "--output", str(quality_schema_report)])
    _run(["scripts/run_data_registry.py", "--images-dir", str(images), "--labels-dir", str(labels), "--dataset-name", "demo", "--version", "v1", "--manifest-out", str(manifest), "--registry-path", str(registry)])
    _run(["scripts/register_dataset_version.py", "--dataset-name", "demo", "--dataset-version", "v1", "--manifest-path", str(manifest), "--quality-report", str(quality_report), "--registry-path", str(registry), "--output", str(artifact_root / "reports" / "dataset_registration.json")])

    candidate_v1 = _build_candidate(tmp_path, artifact_root, manifest, "v1")
    _run(["scripts/promote_model.py", "--candidate-version", "v1", "--require-eval-pass", "--artifact-root", str(artifact_root)])
    assert (artifact_root / "registry" / "champion.json").exists()

    candidate_v2 = _build_candidate(tmp_path, artifact_root, manifest, "v2")
    _run(["scripts/promote_model.py", "--candidate-version", "v2", "--require-eval-pass", "--artifact-root", str(artifact_root)])
    _run(["scripts/rollback_model.py", "--to", "previous", "--artifact-root", str(artifact_root)])
    assert yaml.safe_load((artifact_root / "registry" / "models.yaml").read_text(encoding="utf-8"))["champion_version"] == "v1"

    _run(["scripts/reject_model_candidate.py", "--candidate-version", "bad-v1", "--reason", "smoke_rejection", "--artifact-root", str(artifact_root)])
    assert (artifact_root / "registry" / "rejected" / "bad-v1.json").exists()

    prediction_log = artifact_root / "predictions" / "prediction_log.jsonl"
    feedback_log = artifact_root / "feedback" / "review_feedback.jsonl"
    _write_jsonl(
        prediction_log,
        [
            {
                "timestamp": "2026-05-12T00:00:00+00:00",
                "request_id": "req-1",
                "model_version": "v1",
                "input_shape": [4, 4, 4],
                "intensity_mean": 0.5,
                "intensity_std": 0.1,
                "foreground_ratio": 0.2,
                "latency_ms": 10.0,
                "status": "success",
            }
        ],
    )
    _write_jsonl(
        feedback_log,
        [
            {
                "timestamp": "2026-05-12T00:00:00+00:00",
                "request_id": "req-1",
                "model_version": "v1",
                "reviewer_id": "local_user",
                "accepted": True,
                "quality_score": 5,
                "notes": "",
                "requires_reannotation": False,
            }
        ],
    )
    runtime = artifact_root / "reports" / "runtime.json"
    prediction_summary = artifact_root / "reports" / "prediction_summary.json"
    feedback_summary = artifact_root / "reports" / "feedback_summary.json"
    drift_report = artifact_root / "reports" / "drift.json"
    assessment = artifact_root / "reports" / "assessment.json"

    _run(["scripts/collect_runtime_metrics.py", "--api-url", "http://127.0.0.1:9", "--output", str(runtime), "--allow-offline"])
    _run(["scripts/summarize_prediction_metadata.py", "--prediction-log", str(prediction_log), "--output", str(prediction_summary)])
    _run(["scripts/summarize_review_feedback.py", "--feedback-log", str(feedback_log), "--output", str(feedback_summary)])
    _run(["scripts/build_drift_candidate_from_prediction_log.py", "--prediction-log", str(prediction_log), "--output", str(drift_report)])
    _run(["scripts/evaluate_retraining_policy.py", "--runtime-report", str(runtime), "--prediction-summary", str(prediction_summary), "--feedback-summary", str(feedback_summary), "--drift-report", str(drift_report), "--policy", "src/mlops/policies/default_operating_policy.yaml", "--output", str(assessment)])
    _run(["scripts/validate_json_schema.py", "--schema", "schemas/retraining_assessment.schema.json", "--json", str(assessment), "--output", str(artifact_root / "reports" / "assessment_schema.json")])

    assert candidate_v1.exists()
    assert candidate_v2.exists()
    assert json.loads(assessment.read_text(encoding="utf-8"))["status"] == "completed"
