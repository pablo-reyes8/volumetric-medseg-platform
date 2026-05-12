from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None
    EmptyOperator = None
    BranchPythonOperator = None

try:
    from airflow.dags._compat import build_dag_spec
except ImportError:
    from _compat import build_dag_spec


DAG_ID = "train_evaluate_promote_dag"
EXPECTED_TASK_IDS = [
    "start",
    "validate_inputs",
    "create_candidate_run_dir",
    "train_candidate",
    "log_training_to_mlflow",
    "evaluate_candidate",
    "validate_evaluation_report",
    "apply_evaluation_gates",
    "package_candidate",
    "register_candidate",
    "branch_promote_or_reject",
    "promote_candidate",
    "reject_candidate",
    "write_training_lifecycle_summary",
    "end",
]
EXPECTED_DEPENDENCIES = [
    ("start", "validate_inputs"),
    ("validate_inputs", "create_candidate_run_dir"),
    ("create_candidate_run_dir", "train_candidate"),
    ("train_candidate", "log_training_to_mlflow"),
    ("log_training_to_mlflow", "evaluate_candidate"),
    ("evaluate_candidate", "validate_evaluation_report"),
    ("validate_evaluation_report", "apply_evaluation_gates"),
    ("apply_evaluation_gates", "package_candidate"),
    ("package_candidate", "register_candidate"),
    ("register_candidate", "branch_promote_or_reject"),
    ("branch_promote_or_reject", "promote_candidate"),
    ("branch_promote_or_reject", "reject_candidate"),
    ("promote_candidate", "write_training_lifecycle_summary"),
    ("reject_candidate", "write_training_lifecycle_summary"),
    ("write_training_lifecycle_summary", "end"),
]

PARAMS = {
    "dataset_manifest": "data/manifests/task04_hippocampus_2026.03.21.json",
    "training_config": "configs/training/local_unet3d_smoke.yaml",
    "model_version": "v0.3.0",
    "evaluation_gates": "configs/mlops/evaluation_gates.yaml",
    "allow_promotion": True,
    "dry_run": True,
}


def _cmd(command: str) -> str:
    return f"cd /workspace && {command}"


def _branch(**context) -> str:
    import json
    from pathlib import Path

    params = context["params"]
    decision = json.loads(Path(f"artifacts/models/{params['model_version']}/gate_decision.json").read_text(encoding="utf-8"))
    return "promote_candidate" if decision.get("passed") and params.get("allow_promotion") else "reject_candidate"


if DAG and BashOperator and EmptyOperator and BranchPythonOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        params=PARAMS,
        tags=["medseg", "training", "registry"],
    ) as dag:
        start = EmptyOperator(task_id="start")
        validate_inputs = BashOperator(
            task_id="validate_inputs",
            bash_command=_cmd("test -f {{ params.dataset_manifest }} && test -f {{ params.training_config }} && test -f {{ params.evaluation_gates }}"),
        )
        create_candidate_run_dir = BashOperator(task_id="create_candidate_run_dir", bash_command=_cmd("mkdir -p artifacts/models/{{ params.model_version }}"))
        train_candidate = BashOperator(
            task_id="train_candidate",
            bash_command=_cmd(
                "python scripts/train_unet3d_smoke.py --output-dir artifacts/models/{{ params.model_version }} "
                "--config {{ params.training_config }} --dataset-manifest {{ params.dataset_manifest }} "
                "--model-version {{ params.model_version }} --dag-id train_evaluate_promote_dag --dag-run-id {{ run_id }}"
            ),
        )
        log_training_to_mlflow = BashOperator(task_id="log_training_to_mlflow", bash_command=_cmd("test -f artifacts/models/{{ params.model_version }}/mlflow_run.json"))
        evaluate_candidate = BashOperator(
            task_id="evaluate_candidate",
            bash_command=_cmd(
                "python scripts/evaluate_model.py --model-package-dir artifacts/models/{{ params.model_version }} "
                "--dataset-manifest {{ params.dataset_manifest }} "
                "--output-json artifacts/models/{{ params.model_version }}/evaluation_report.json "
                "--output-md artifacts/models/{{ params.model_version }}/evaluation_report.md "
                "--dag-id train_evaluate_promote_dag --dag-run-id {{ run_id }}"
            ),
        )
        validate_evaluation_report = BashOperator(
            task_id="validate_evaluation_report",
            bash_command=_cmd("python scripts/validate_json_schema.py --schema schemas/evaluation_report.schema.json --json artifacts/models/{{ params.model_version }}/evaluation_report.json"),
        )
        apply_evaluation_gates = BashOperator(
            task_id="apply_evaluation_gates",
            bash_command=_cmd(
                "python scripts/apply_evaluation_gates.py --evaluation-report artifacts/models/{{ params.model_version }}/evaluation_report.json "
                "--gates {{ params.evaluation_gates }} --output artifacts/models/{{ params.model_version }}/gate_decision.json "
                "--dag-id train_evaluate_promote_dag --dag-run-id {{ run_id }}"
            ),
        )
        package_candidate = BashOperator(
            task_id="package_candidate",
            bash_command=_cmd(
                "python scripts/package_model_candidate.py --model-version {{ params.model_version }} --model-dir artifacts/models/{{ params.model_version }} "
                "--dataset-manifest {{ params.dataset_manifest }} --training-config {{ params.training_config }} "
                "--evaluation-report artifacts/models/{{ params.model_version }}/evaluation_report.json "
                "--gate-decision artifacts/models/{{ params.model_version }}/gate_decision.json"
            ),
        )
        register_candidate = BashOperator(task_id="register_candidate", bash_command=_cmd("python scripts/register_model_candidate.py --model-package artifacts/models/{{ params.model_version }}/model_package.yaml"))
        branch_promote_or_reject = BranchPythonOperator(task_id="branch_promote_or_reject", python_callable=_branch)
        promote_candidate = BashOperator(task_id="promote_candidate", bash_command=_cmd("python scripts/promote_model.py --candidate-version {{ params.model_version }} --require-eval-pass"))
        reject_candidate = BashOperator(task_id="reject_candidate", bash_command=_cmd("python scripts/reject_model_candidate.py --candidate-version {{ params.model_version }} --reason failed_evaluation_gates_or_promotion_disabled"))
        write_training_lifecycle_summary = BashOperator(
            task_id="write_training_lifecycle_summary",
            trigger_rule="none_failed_min_one_success",
            bash_command=_cmd(
                "mkdir -p artifacts/reports/evaluation && printf '# Training Lifecycle Summary\\n\\n- Model version: {{ params.model_version }}\\n- Dataset manifest: {{ params.dataset_manifest }}\\n- DAG run: {{ run_id }}\\n' "
                "> artifacts/reports/evaluation/training_lifecycle_{{ params.model_version }}_{{ run_id | replace(':', '_') | replace('+', '_') }}.md"
            ),
        )
        end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")
        start >> validate_inputs >> create_candidate_run_dir >> train_candidate >> log_training_to_mlflow >> evaluate_candidate >> validate_evaluation_report >> apply_evaluation_gates >> package_candidate >> register_candidate >> branch_promote_or_reject
        branch_promote_or_reject >> [promote_candidate, reject_candidate] >> write_training_lifecycle_summary >> end
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)

