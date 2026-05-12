import importlib.util
import sys
from pathlib import Path


DAG_FILES = [
    "dataset_validation_registration_dag.py",
    "train_evaluate_promote_dag.py",
    "monitoring_drift_retraining_dag.py",
    "local_full_lifecycle_demo_dag.py",
]


def _load_dag_module(filename: str):
    path = Path("airflow/dags") / filename
    sys.path.insert(0, str(path.parent.resolve()))
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
