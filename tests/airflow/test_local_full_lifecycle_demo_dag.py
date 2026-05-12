import json
import subprocess
import sys
from pathlib import Path

from tests.airflow.test_dag_imports import _load_dag_module


def test_local_full_lifecycle_demo_dag_has_portfolio_tasks():
    module = _load_dag_module("local_full_lifecycle_demo_dag.py")
    required = {
        "prepare_demo_dataset",
        "run_smoke_training",
        "run_smoke_evaluation",
        "register_and_promote_demo_model",
        "run_monitoring_drift_retraining_assessment",
        "write_demo_summary",
    }
    assert required.issubset(set(module.dag.task_ids))


def test_full_local_lifecycle_demo_dry_run_creates_artifacts(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_full_local_lifecycle_demo.py",
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--model-version",
            "demo_test",
            "--dry-run",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "passed"
    assert (tmp_path / "artifacts/models/demo_test/model_package.yaml").exists()
    assert (tmp_path / "artifacts/reports/airflow/full_lifecycle_demo_demo_test.md").exists()

