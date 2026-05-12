from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None

try:
    from airflow.dags._compat import build_dag_spec
except ImportError:
    from _compat import build_dag_spec


DAG_ID = "medseg_training_pipeline"
EXPECTED_TASK_IDS = [
    "load_latest_dataset_manifest",
    "validate_training_config",
    "train_model",
    "evaluate_model",
    "log_to_mlflow",
    "build_model_package",
    "register_challenger",
]
EXPECTED_DEPENDENCIES = list(zip(EXPECTED_TASK_IDS, EXPECTED_TASK_IDS[1:]))


if DAG and BashOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["medseg", "training", "mlops"],
    ) as dag:
        tasks = {
            task_id: BashOperator(
                task_id=task_id,
                bash_command=f"cd /workspace && mkdir -p artifacts/runs artifacts/models/challenger artifacts/reports/evaluation && echo {task_id}",
            )
            for task_id in EXPECTED_TASK_IDS
        }
        for upstream, downstream in EXPECTED_DEPENDENCIES:
            tasks[upstream] >> tasks[downstream]
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)
