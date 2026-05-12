from tests.airflow.test_dag_imports import _load_dag_module


def test_dataset_validation_registration_dag_has_dataops_governance_tasks():
    module = _load_dag_module("dataset_validation_registration_dag.py")
    required = {
        "validate_paths",
        "validate_dataset_contract",
        "run_data_quality_checks",
        "generate_manifest",
        "validate_manifest_schema",
        "register_dataset_version",
        "write_dataops_summary",
    }
    assert required.issubset(set(module.dag.task_ids))
    assert ("run_data_quality_checks", "generate_manifest") in module.dag.dependencies

