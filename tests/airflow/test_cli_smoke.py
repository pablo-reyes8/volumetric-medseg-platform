import subprocess
import sys


SCRIPTS = [
    "validate_dataset_contract.py",
    "run_data_quality.py",
    "validate_json_schema.py",
    "register_dataset_version.py",
    "train_unet3d_smoke.py",
    "train_unet3d.py",
    "evaluate_model.py",
    "apply_evaluation_gates.py",
    "package_model_candidate.py",
    "register_model_candidate.py",
    "reject_model_candidate.py",
    "collect_runtime_metrics.py",
    "summarize_prediction_metadata.py",
    "summarize_review_feedback.py",
    "evaluate_retraining_policy.py",
    "build_drift_candidate_from_prediction_log.py",
    "run_full_local_lifecycle_demo.py",
]


def test_airflow_orchestration_scripts_expose_help():
    for script in SCRIPTS:
        result = subprocess.run([sys.executable, f"scripts/{script}", "--help"], check=False, capture_output=True, text=True)
        assert result.returncode == 0, script
        assert "usage:" in result.stdout.lower()
