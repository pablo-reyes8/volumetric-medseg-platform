from src.mlops.evaluation_report import is_evaluation_passed, validate_evaluation_report


def _report():
    return {
        "model_version": "v0.2.0",
        "run_id": "run-123",
        "dataset_manifest": "data/manifests/demo.json",
        "evaluation_split": "validation",
        "metrics": {
            "dice_mean": 0.91,
            "dice_per_class": [0.9, 0.92],
            "hausdorff_95": 3.2,
            "inference_latency_p95_ms": 200.0,
        },
        "thresholds": {
            "min_dice_mean": 0.8,
            "max_hausdorff_95": 8.0,
            "max_latency_p95_ms": 500.0,
        },
        "decision": {"passed": True, "reasons": []},
    }


def test_valid_evaluation_report_passes_gate():
    report = _report()
    assert validate_evaluation_report(report) == []
    assert is_evaluation_passed(report) is True


def test_missing_metric_fails_validation():
    report = _report()
    del report["metrics"]["hausdorff_95"]
    assert any("hausdorff_95" in error for error in validate_evaluation_report(report))
    assert is_evaluation_passed(report) is False


def test_failed_threshold_blocks_promotion_gate():
    report = _report()
    report["metrics"]["dice_mean"] = 0.2
    assert is_evaluation_passed(report) is False
