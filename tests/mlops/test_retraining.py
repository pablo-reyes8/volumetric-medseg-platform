from src.mlops.retraining import evaluate_retraining_recommendations, load_operating_policy


def test_retraining_policy_recommends_drift_and_periodic_actions():
    policy = load_operating_policy()
    report = evaluate_retraining_recommendations(
        policy=policy,
        drift_report={"status": "drift_detected"},
        days_since_last_train=45,
        runtime_snapshot={
            "totals": {"requests": 500, "error_rate": 0.0},
            "latency_ms": {"p95": 100.0},
            "throughput": {"requests_per_minute": 10.0},
        },
    )

    assert "drift_retrain" in report["recommended_actions"]
    assert "periodic_retrain" in report["recommended_actions"]


def test_retraining_policy_recommends_rollback_on_runtime_regression():
    policy = load_operating_policy()
    report = evaluate_retraining_recommendations(
        policy=policy,
        runtime_snapshot={
            "totals": {"requests": 500, "error_rate": 0.08},
            "latency_ms": {"p95": 2800.0},
            "throughput": {"requests_per_minute": 0.5},
        },
        consecutive_incidents=3,
    )

    assert "rollback" in report["recommended_actions"]
