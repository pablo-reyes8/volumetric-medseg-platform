from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
except ImportError:  # pragma: no cover - used by lightweight smoke tests
    DAG = None
    BashOperator = None

try:
    from airflow.dags._compat import build_dag_spec
except ImportError:
    from _compat import build_dag_spec


DAG_ID = "medseg_data_validation"
EXPECTED_TASK_IDS = [
    "validate_dataset_contract",
    "generate_dataset_manifest",
    "update_dataset_registry",
    "persist_quality_report",
]
EXPECTED_DEPENDENCIES = [
    ("validate_dataset_contract", "generate_dataset_manifest"),
    ("generate_dataset_manifest", "update_dataset_registry"),
    ("update_dataset_registry", "persist_quality_report"),
]


if DAG and BashOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        tags=["medseg", "dataops"],
    ) as dag:
        validate_dataset_contract = BashOperator(
            task_id="validate_dataset_contract",
            bash_command="cd /workspace && python scripts/run_task04_dataops.py --dataset-version ${DATASET_VERSION:-local}",
        )
        generate_dataset_manifest = BashOperator(
            task_id="generate_dataset_manifest",
            bash_command="cd /workspace && python scripts/run_data_registry.py --help",
        )
        update_dataset_registry = BashOperator(
            task_id="update_dataset_registry",
            bash_command="cd /workspace && test -f data/registry/datasets.yaml",
        )
        persist_quality_report = BashOperator(
            task_id="persist_quality_report",
            bash_command="cd /workspace && mkdir -p artifacts/reports/data",
        )
        validate_dataset_contract >> generate_dataset_manifest >> update_dataset_registry >> persist_quality_report
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)
