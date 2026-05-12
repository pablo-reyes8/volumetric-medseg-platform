from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REQUIRED_METRICS = {"dice_mean", "dice_per_class", "hausdorff_95", "inference_latency_p95_ms"}
REQUIRED_THRESHOLDS = {"min_dice_mean", "max_hausdorff_95", "max_latency_p95_ms"}


def load_evaluation_report(path: Path | str) -> Dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_evaluation_report(report: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    for field in ["model_version", "run_id", "dataset_manifest", "evaluation_split", "metrics", "thresholds", "decision"]:
        if field not in report:
            errors.append(f"missing required field: {field}")

    metrics = report.get("metrics", {})
    if isinstance(metrics, dict):
        missing_metrics = sorted(REQUIRED_METRICS - set(metrics.keys()))
        errors.extend(f"missing required metric: {metric}" for metric in missing_metrics)
    else:
        errors.append("metrics must be an object")

    thresholds = report.get("thresholds", {})
    if isinstance(thresholds, dict):
        missing_thresholds = sorted(REQUIRED_THRESHOLDS - set(thresholds.keys()))
        errors.extend(f"missing required threshold: {threshold}" for threshold in missing_thresholds)
    else:
        errors.append("thresholds must be an object")

    decision = report.get("decision", {})
    if not isinstance(decision, dict):
        errors.append("decision must be an object")
    elif "passed" not in decision:
        errors.append("decision.passed is required")

    return errors


def is_evaluation_passed(report: Dict[str, object]) -> bool:
    if validate_evaluation_report(report):
        return False
    decision = report.get("decision", {})
    if isinstance(decision, dict) and decision.get("passed") is False:
        return False

    metrics = report["metrics"]
    thresholds = report["thresholds"]
    return (
        float(metrics["dice_mean"]) >= float(thresholds["min_dice_mean"])
        and float(metrics["hausdorff_95"]) <= float(thresholds["max_hausdorff_95"])
        and float(metrics["inference_latency_p95_ms"]) <= float(thresholds["max_latency_p95_ms"])
    )


def save_evaluation_report(report: Dict[str, object], path: Path | str) -> Path:
    errors = validate_evaluation_report(report)
    if errors:
        raise ValueError(f"Invalid evaluation report: {errors}")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output
