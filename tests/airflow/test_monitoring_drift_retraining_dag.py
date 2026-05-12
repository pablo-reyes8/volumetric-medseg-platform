from tests.airflow.test_dag_imports import _load_dag_module


def test_monitoring_dag_has_decision_branches():
    module = _load_dag_module("monitoring_drift_retraining_dag.py")
    required = {
        "collect_runtime_metrics",
        "summarize_prediction_metadata",
        "summarize_review_feedback",
        "compute_input_drift",
        "evaluate_operating_policy",
        "branch_decision",
        "recommend_retraining",
        "recommend_rollback",
    }
    assert required.issubset(set(module.dag.task_ids))
    assert ("branch_decision", "recommend_retraining") in module.dag.dependencies
    assert ("branch_decision", "recommend_rollback") in module.dag.dependencies

