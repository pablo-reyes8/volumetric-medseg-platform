from tests.airflow.test_dag_imports import _load_dag_module


def test_train_evaluate_promote_dag_has_governance_and_branching_tasks():
    module = _load_dag_module("train_evaluate_promote_dag.py")
    required = {
        "validate_inputs",
        "train_candidate",
        "evaluate_candidate",
        "apply_evaluation_gates",
        "branch_promote_or_reject",
        "promote_candidate",
        "reject_candidate",
        "write_training_lifecycle_summary",
    }
    assert required.issubset(set(module.dag.task_ids))
    assert ("branch_promote_or_reject", "promote_candidate") in module.dag.dependencies
    assert ("branch_promote_or_reject", "reject_candidate") in module.dag.dependencies

