import importlib.util
from pathlib import Path


DAG_FILES = [
    "medseg_data_validation_dag.py",
    "medseg_training_pipeline_dag.py",
    "medseg_monitoring_dag.py",
    "medseg_retraining_decision_dag.py",
]


def _load_dag_module(filename: str):
    path = Path("airflow/dags") / filename
    spec = importlib.util.spec_from_file_location(filename.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_airflow_dag_specs_import_without_airflow_installed():
    for filename in DAG_FILES:
        module = _load_dag_module(filename)
        assert module.DAG_ID
        assert module.EXPECTED_TASK_IDS
        assert module.dag.task_ids == module.EXPECTED_TASK_IDS
        assert module.dag.dependencies == module.EXPECTED_DEPENDENCIES
