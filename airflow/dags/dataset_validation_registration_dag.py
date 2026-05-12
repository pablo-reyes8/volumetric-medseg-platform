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


DAG_ID = "dataset_validation_registration_dag"
EXPECTED_TASK_IDS = [
    "start",
    "validate_paths",
    "validate_dataset_contract",
    "run_data_quality_checks",
    "generate_manifest",
    "validate_manifest_schema",
    "register_dataset_version",
    "write_dataops_summary",
    "end",
]
EXPECTED_DEPENDENCIES = list(zip(EXPECTED_TASK_IDS, EXPECTED_TASK_IDS[1:]))

PARAMS = {
    "dataset_name": "task04_hippocampus",
    "dataset_version": "2026.03.21",
    "images_dir": "data/processed/task04_hippocampus/2026.03.21/imagesTr",
    "labels_dir": "data/processed/task04_hippocampus/2026.03.21/labelsTr",
    "contract_path": "data/contracts/task04_hippocampus.contract.yaml",
    "manifest_out": "data/manifests/task04_hippocampus_2026.03.21.json",
}


def _cmd(command: str) -> str:
    return f"cd /workspace && {command}"


if DAG and BashOperator and EmptyOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        params=PARAMS,
        tags=["medseg", "dataops", "governance"],
    ) as dag:
        start = EmptyOperator(task_id="start")
        validate_paths = BashOperator(
            task_id="validate_paths",
            bash_command=_cmd("mkdir -p artifacts/reports/data_quality data/manifests && test -f {{ params.contract_path }}"),
        )
        validate_dataset_contract = BashOperator(
            task_id="validate_dataset_contract",
            bash_command=_cmd(
                "python scripts/validate_dataset_contract.py "
                "--contract-path {{ params.contract_path }} "
                "--images-dir {{ params.images_dir }} "
                "--labels-dir {{ params.labels_dir }} "
                "--output artifacts/reports/data_quality/contract_validation_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--dag-id dataset_validation_registration_dag --dag-run-id {{ run_id }}"
            ),
        )
        run_data_quality_checks = BashOperator(
            task_id="run_data_quality_checks",
            bash_command=_cmd(
                "python scripts/run_data_quality.py "
                "--images-dir {{ params.images_dir }} "
                "--labels-dir {{ params.labels_dir }} "
                "--contract-path {{ params.contract_path }} "
                "--output artifacts/reports/data_quality/quality_report_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--dag-id dataset_validation_registration_dag --dag-run-id {{ run_id }}"
            ),
        )
        generate_manifest = BashOperator(
            task_id="generate_manifest",
            bash_command=_cmd(
                "python scripts/run_data_registry.py "
                "--images-dir {{ params.images_dir }} "
                "--labels-dir {{ params.labels_dir }} "
                "--dataset-name {{ params.dataset_name }} "
                "--version {{ params.dataset_version }} "
                "--manifest-out {{ params.manifest_out }}"
            ),
        )
        validate_manifest_schema = BashOperator(
            task_id="validate_manifest_schema",
            bash_command=_cmd(
                "python scripts/validate_json_schema.py "
                "--schema data/contracts/dataset_manifest.schema.json "
                "--json {{ params.manifest_out }}"
            ),
        )
        register_dataset_version = BashOperator(
            task_id="register_dataset_version",
            bash_command=_cmd(
                "python scripts/register_dataset_version.py "
                "--dataset-name {{ params.dataset_name }} "
                "--dataset-version {{ params.dataset_version }} "
                "--manifest-path {{ params.manifest_out }} "
                "--quality-report artifacts/reports/data_quality/quality_report_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--dag-id dataset_validation_registration_dag --dag-run-id {{ run_id }}"
            ),
        )
        write_dataops_summary = BashOperator(
            task_id="write_dataops_summary",
            bash_command=_cmd(
                "printf '# DataOps Summary\\n\\n- Dataset: {{ params.dataset_name }}\\n- Version: {{ params.dataset_version }}\\n- Manifest: {{ params.manifest_out }}\\n- Airflow DAG: dataset_validation_registration_dag\\n- Airflow run: {{ run_id }}\\n' "
                "> artifacts/reports/data_quality/dataops_summary_{{ params.dataset_version }}_{{ run_id | replace(':', '_') | replace('+', '_') }}.md"
            ),
        )
        end = EmptyOperator(task_id="end")
        start >> validate_paths >> validate_dataset_contract >> run_data_quality_checks >> generate_manifest >> validate_manifest_schema >> register_dataset_version >> write_dataops_summary >> end
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)

