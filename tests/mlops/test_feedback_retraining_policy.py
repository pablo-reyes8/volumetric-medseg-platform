from src.mlops.retraining import evaluate_retraining_recommendations, load_operating_policy


def test_low_review_feedback_triggers_retraining_recommendation():
    report = evaluate_retraining_recommendations(
        policy=load_operating_policy(),
        feedback_snapshot={
            "acceptance_rate": 0.4,
            "reannotation_rate": 0.5,
            "mean_quality_score": 2.0,
        },
    )

    assert "feedback_retrain" in report["recommended_actions"]
