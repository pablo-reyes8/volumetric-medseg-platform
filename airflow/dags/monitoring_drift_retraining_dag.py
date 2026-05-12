from __future__ import annotations

from datetime import datetime

try:
    from airflow import DAG
    from airflow.operators.bash import BashOperator
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import BranchPythonOperator
except ImportError:  # pragma: no cover
    DAG = None
    BashOperator = None
    EmptyOperator = None
    BranchPythonOperator = None

try:
    from airflow.dags._compat import build_dag_spec
except ImportError:
    from _compat import build_dag_spec


DAG_ID = "monitoring_drift_retraining_dag"
EXPECTED_TASK_IDS = [
    "start",
    "validate_monitoring_inputs",
    "collect_runtime_metrics",
    "summarize_prediction_metadata",
    "summarize_review_feedback",
    "compute_input_drift",
    "evaluate_operating_policy",
    "branch_decision",
    "recommend_no_action",
    "recommend_retraining",
    "recommend_rollback",
    "recommend_investigation",
    "write_monitoring_lifecycle_summary",
    "end",
]
EXPECTED_DEPENDENCIES = [
    ("start", "validate_monitoring_inputs"),
    ("validate_monitoring_inputs", "collect_runtime_metrics"),
    ("collect_runtime_metrics", "summarize_prediction_metadata"),
    ("summarize_prediction_metadata", "summarize_review_feedback"),
    ("summarize_review_feedback", "compute_input_drift"),
    ("compute_input_drift", "evaluate_operating_policy"),
    ("evaluate_operating_policy", "branch_decision"),
    ("branch_decision", "recommend_no_action"),
    ("branch_decision", "recommend_retraining"),
    ("branch_decision", "recommend_rollback"),
    ("branch_decision", "recommend_investigation"),
    ("recommend_no_action", "write_monitoring_lifecycle_summary"),
    ("recommend_retraining", "write_monitoring_lifecycle_summary"),
    ("recommend_rollback", "write_monitoring_lifecycle_summary"),
    ("recommend_investigation", "write_monitoring_lifecycle_summary"),
    ("write_monitoring_lifecycle_summary", "end"),
]

PARAMS = {
    "prediction_log_path": "artifacts/predictions/prediction_log.jsonl",
    "feedback_log_path": "artifacts/feedback/review_feedback.jsonl",
    "operating_policy": "src/mlops/policies/default_operating_policy.yaml",
    "output_dir": "artifacts/reports/retraining",
    "api_url": "http://api:8000",
    "auto_rollback": False,
}


def _cmd(command: str) -> str:
    return f"cd /workspace && {command}"


def _branch(**context) -> str:
    import json
    from pathlib import Path

    run_id = context["run_id"].replace(":", "_").replace("+", "_")
    assessment = json.loads(Path(f"artifacts/reports/retraining/retraining_assessment_{run_id}.json").read_text(encoding="utf-8"))
    return {
        "none": "recommend_no_action",
        "retrain": "recommend_retraining",
        "rollback": "recommend_rollback",
        "investigate": "recommend_investigation",
    }.get(assessment.get("decision"), "recommend_no_action")


if DAG and BashOperator and EmptyOperator and BranchPythonOperator:
    with DAG(
        dag_id=DAG_ID,
        start_date=datetime(2026, 1, 1),
        schedule=None,
        catchup=False,
        params=PARAMS,
        tags=["medseg", "monitoring", "retraining"],
    ) as dag:
        start = EmptyOperator(task_id="start")
        validate_monitoring_inputs = BashOperator(task_id="validate_monitoring_inputs", bash_command=_cmd("mkdir -p {{ params.output_dir }} artifacts/reports/drift && test -f {{ params.operating_policy }}"))
        collect_runtime_metrics = BashOperator(
            task_id="collect_runtime_metrics",
            bash_command=_cmd("python scripts/collect_runtime_metrics.py --api-url {{ params.api_url }} --output {{ params.output_dir }}/runtime_metrics_{{ run_id | replace(':', '_') | replace('+', '_') }}.json --allow-offline --dag-id monitoring_drift_retraining_dag --dag-run-id {{ run_id }}"),
        )
        summarize_prediction_metadata = BashOperator(
            task_id="summarize_prediction_metadata",
            bash_command=_cmd("python scripts/summarize_prediction_metadata.py --prediction-log {{ params.prediction_log_path }} --output {{ params.output_dir }}/prediction_summary_{{ run_id | replace(':', '_') | replace('+', '_') }}.json --dag-id monitoring_drift_retraining_dag --dag-run-id {{ run_id }}"),
        )
        summarize_review_feedback = BashOperator(
            task_id="summarize_review_feedback",
            bash_command=_cmd("python scripts/summarize_review_feedback.py --feedback-log {{ params.feedback_log_path }} --output {{ params.output_dir }}/feedback_summary_{{ run_id | replace(':', '_') | replace('+', '_') }}.json --dag-id monitoring_drift_retraining_dag --dag-run-id {{ run_id }}"),
        )
        compute_input_drift = BashOperator(
            task_id="compute_input_drift",
            bash_command=_cmd("python scripts/build_drift_candidate_from_prediction_log.py --prediction-log {{ params.prediction_log_path }} --output artifacts/reports/drift/drift_report_{{ run_id | replace(':', '_') | replace('+', '_') }}.json --dag-id monitoring_drift_retraining_dag --dag-run-id {{ run_id }}"),
        )
        evaluate_operating_policy = BashOperator(
            task_id="evaluate_operating_policy",
            bash_command=_cmd(
                "python scripts/evaluate_retraining_policy.py "
                "--runtime-report {{ params.output_dir }}/runtime_metrics_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--prediction-summary {{ params.output_dir }}/prediction_summary_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--feedback-summary {{ params.output_dir }}/feedback_summary_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--drift-report artifacts/reports/drift/drift_report_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--policy {{ params.operating_policy }} "
                "--output {{ params.output_dir }}/retraining_assessment_{{ run_id | replace(':', '_') | replace('+', '_') }}.json "
                "--dag-id monitoring_drift_retraining_dag --dag-run-id {{ run_id }}"
            ),
        )
        branch_decision = BranchPythonOperator(task_id="branch_decision", python_callable=_branch)
        recommend_no_action = BashOperator(task_id="recommend_no_action", bash_command=_cmd("printf '{\"decision\":\"none\"}\\n' >> artifacts/registry/history/operations_history.jsonl"))
        recommend_retraining = BashOperator(task_id="recommend_retraining", bash_command=_cmd("printf '{\"status\":\"open\",\"reason\":\"policy_retrain\"}\\n' >> artifacts/registry/retraining_queue.jsonl"))
        recommend_rollback = BashOperator(task_id="recommend_rollback", bash_command=_cmd("printf '{\"rollback_recommended\":true}\\n' > {{ params.output_dir }}/rollback_recommendation_{{ run_id | replace(':', '_') | replace('+', '_') }}.json"))
        recommend_investigation = BashOperator(task_id="recommend_investigation", bash_command=_cmd("printf '# Investigation Ticket\\n\\n- DAG run: {{ run_id }}\\n' > {{ params.output_dir }}/investigation_ticket_{{ run_id | replace(':', '_') | replace('+', '_') }}.md"))
        write_monitoring_lifecycle_summary = BashOperator(
            task_id="write_monitoring_lifecycle_summary",
            trigger_rule="none_failed_min_one_success",
            bash_command=_cmd("printf '# Monitoring Lifecycle Summary\\n\\n- DAG run: {{ run_id }}\\n' > {{ params.output_dir }}/monitoring_lifecycle_{{ run_id | replace(':', '_') | replace('+', '_') }}.md"),
        )
        end = EmptyOperator(task_id="end", trigger_rule="none_failed_min_one_success")
        start >> validate_monitoring_inputs >> collect_runtime_metrics >> summarize_prediction_metadata >> summarize_review_feedback >> compute_input_drift >> evaluate_operating_policy >> branch_decision
        branch_decision >> [recommend_no_action, recommend_retraining, recommend_rollback, recommend_investigation] >> write_monitoring_lifecycle_summary >> end
else:
    dag = build_dag_spec(DAG_ID, EXPECTED_TASK_IDS, EXPECTED_DEPENDENCIES)

