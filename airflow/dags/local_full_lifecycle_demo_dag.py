from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None
    EmptyOperator = None

try:
    from airflow.dags._compat import build_dag_spec
except ImportError:
    from _compat import build_dag_spec


DAG_ID = "local_full_lifecycle_demo_dag"
EXPECTED_TASK_IDS = [
    "start",
    "prepare_demo_dataset",
    "run_dataset_validation_registration",
    "run_smoke_training",
    "run_smoke_evaluation",
    "register_and_promote_demo_model",
    "check_api_readiness",
    "run_demo_prediction",
    "log_demo_feedback",
    "run_monitoring_drift_retraining_assessment",
    "write_demo_summary",
    "end",
]
EXPECTED_DEPENDENCIES = list(zip(EXPECTED_TASK_IDS, EXPECTED_TASK_IDS[1:]))


def _cmd(command: str) -> str:
    return f"cd /workspace && {command}"


if DAG and BashOperator and EmptyOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["medseg", "demo", "portfolio"],
    ) as dag:
        start = EmptyOperator(task_id="start")
        prepare_demo_dataset = BashOperator(task_id="prepare_demo_dataset", bash_command=_cmd("mkdir -p artifacts/airflow"))
        run_dataset_validation_registration = BashOperator(task_id="run_dataset_validation_registration", bash_command=_cmd("python scripts/run_full_local_lifecycle_demo.py --dry-run --model-version demo_{{ run_id | replace(':', '_') | replace('+', '_') }}"))
        run_smoke_training = BashOperator(task_id="run_smoke_training", bash_command=_cmd("test -f artifacts/models/demo_{{ run_id | replace(':', '_') | replace('+', '_') }}/checkpoint.pt"))
        run_smoke_evaluation = BashOperator(task_id="run_smoke_evaluation", bash_command=_cmd("test -f artifacts/models/demo_{{ run_id | replace(':', '_') | replace('+', '_') }}/evaluation_report.json"))
        register_and_promote_demo_model = BashOperator(task_id="register_and_promote_demo_model", bash_command=_cmd("test -f artifacts/models/demo_{{ run_id | replace(':', '_') | replace('+', '_') }}/model_package.yaml"))
        check_api_readiness = BashOperator(task_id="check_api_readiness", bash_command=_cmd("python scripts/collect_runtime_metrics.py --api-url http://api:8000 --output artifacts/reports/airflow/api_readiness_{{ run_id | replace(':', '_') | replace('+', '_') }}.json --allow-offline"))
        run_demo_prediction = BashOperator(task_id="run_demo_prediction", bash_command=_cmd("test -f artifacts/predictions/prediction_log.jsonl"))
        log_demo_feedback = BashOperator(task_id="log_demo_feedback", bash_command=_cmd("test -f artifacts/feedback/review_feedback.jsonl"))
        run_monitoring_drift_retraining_assessment = BashOperator(task_id="run_monitoring_drift_retraining_assessment", bash_command=_cmd("python scripts/summarize_prediction_metadata.py --prediction-log artifacts/predictions/prediction_log.jsonl --output artifacts/reports/airflow/demo_prediction_summary_{{ run_id | replace(':', '_') | replace('+', '_') }}.json"))
        write_demo_summary = BashOperator(task_id="write_demo_summary", bash_command=_cmd("test -f artifacts/reports/airflow/full_lifecycle_demo_demo_{{ run_id | replace(':', '_') | replace('+', '_') }}.md"))
        end = EmptyOperator(task_id="end")
        start >> prepare_demo_dataset >> run_dataset_validation_registration >> run_smoke_training >> run_smoke_evaluation >> register_and_promote_demo_model >> check_api_readiness >> run_demo_prediction >> log_demo_feedback >> run_monitoring_drift_retraining_assessment >> write_demo_summary >> end
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)

