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


DAG_ID = "medseg_monitoring"
EXPECTED_TASK_IDS = [
    "collect_runtime_metrics",
    "collect_recent_prediction_inputs",
    "run_drift_check",
    "persist_monitoring_report",
]
EXPECTED_DEPENDENCIES = list(zip(EXPECTED_TASK_IDS, EXPECTED_TASK_IDS[1:]))


if DAG and BashOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["medseg", "monitoring"],
    ) as dag:
        tasks = {
            task_id: BashOperator(
                task_id=task_id,
                bash_command=f"cd /workspace && mkdir -p artifacts/reports/runtime artifacts/reports/drift && echo {task_id}",
            )
            for task_id in EXPECTED_TASK_IDS
        }
        for upstream, downstream in EXPECTED_DEPENDENCIES:
            tasks[upstream] >> tasks[downstream]
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)
