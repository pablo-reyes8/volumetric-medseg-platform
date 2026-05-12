from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import yaml


DEFAULT_POLICY_PATH = Path("src/mlops/policies/default_operating_policy.yaml")


def load_operating_policy(path: Path | str = DEFAULT_POLICY_PATH) -> Dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def summarize_monitored_signals(policy: Dict[str, object]) -> Dict[str, object]:
    monitoring = policy.get("monitoring", {})
    return {
        "tracked_signals": monitoring.get("tracked_signals", []),
        "slo_thresholds": {
            "latency_p95_ms_max": monitoring.get("latency_p95_ms_max"),
            "error_rate_max": monitoring.get("error_rate_max"),
            "throughput_min_rpm": monitoring.get("throughput_min_rpm"),
            "cpu_percent_max": monitoring.get("cpu_percent_max"),
            "memory_percent_max": monitoring.get("memory_percent_max"),
            "estimated_cost_per_1000_requests_max_usd": monitoring.get("estimated_cost_per_1000_requests_max_usd"),
        },
    }


def evaluate_retraining_recommendations(
    policy: Optional[Dict[str, object]] = None,
    drift_report: Optional[Dict[str, object]] = None,
    runtime_snapshot: Optional[Dict[str, object]] = None,
    validation_snapshot: Optional[Dict[str, object]] = None,
    days_since_last_train: Optional[int] = None,
    consecutive_incidents: int = 0,
    feedback_snapshot: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    active_policy = policy or load_operating_policy()
    monitoring = active_policy.get("monitoring", {})
    retraining = active_policy.get("retraining", {})
    rollback = active_policy.get("rollback", {})

    recommendations = []
    reasons = []

    if days_since_last_train is not None and days_since_last_train >= int(retraining.get("periodic_retrain_days", 30)):
        recommendations.append("periodic_retrain")
        reasons.append(f"last training happened {days_since_last_train} days ago")

    if drift_report and drift_report.get("status") == "drift_detected":
        recommendations.append("drift_retrain")
        reasons.append("approved drift thresholds were exceeded")

    if validation_snapshot:
        metric_name = str(retraining.get("kpi_metric", "mIoU"))
        current_metric = validation_snapshot.get("current_metric")
        champion_metric = validation_snapshot.get("champion_metric")
        drop_threshold = float(retraining.get("kpi_drop_absolute_max", 0.03))
        if current_metric is not None and champion_metric is not None:
            absolute_drop = float(champion_metric) - float(current_metric)
            if absolute_drop > drop_threshold:
                recommendations.append("kpi_drop_retrain")
                reasons.append(f"{metric_name} dropped by {absolute_drop:.4f}")

    if feedback_snapshot:
        min_acceptance_rate = float(retraining.get("review_acceptance_rate_min", 0.8))
        max_reannotation_rate = float(retraining.get("reannotation_rate_max", 0.2))
        min_quality_score = float(retraining.get("mean_quality_score_min", 3.5))
        acceptance_rate = float(feedback_snapshot.get("acceptance_rate", 1.0))
        reannotation_rate = float(feedback_snapshot.get("reannotation_rate", 0.0))
        mean_quality_score = float(feedback_snapshot.get("mean_quality_score", 5.0))
        if acceptance_rate < min_acceptance_rate:
            recommendations.append("feedback_retrain")
            reasons.append(f"review acceptance rate {acceptance_rate:.4f} is below policy")
        if reannotation_rate > max_reannotation_rate:
            recommendations.append("feedback_retrain")
            reasons.append(f"reannotation rate {reannotation_rate:.4f} exceeded policy")
        if mean_quality_score < min_quality_score:
            recommendations.append("feedback_retrain")
            reasons.append(f"mean quality score {mean_quality_score:.2f} is below policy")

    if runtime_snapshot:
        error_rate = float(runtime_snapshot.get("totals", {}).get("error_rate", 0.0))
        latency_p95 = float(runtime_snapshot.get("latency_ms", {}).get("p95", 0.0))
        throughput = float(runtime_snapshot.get("throughput", {}).get("requests_per_minute", 0.0))
        if error_rate > float(rollback.get("trigger_error_rate", 0.05)):
            recommendations.append("rollback")
            reasons.append(f"error rate {error_rate:.4f} exceeded rollback threshold")
        if latency_p95 > float(rollback.get("trigger_latency_p95_ms", 2500)):
            recommendations.append("rollback")
            reasons.append(f"latency p95 {latency_p95:.1f}ms exceeded rollback threshold")
        if throughput < float(monitoring.get("throughput_min_rpm", 0.0)) and runtime_snapshot.get("totals", {}).get("requests", 0) > 0:
            recommendations.append("investigate_capacity")
            reasons.append("throughput is below the declared minimum")

    if consecutive_incidents >= int(rollback.get("trigger_consecutive_windows", 3)):
        recommendations.append("rollback")
        reasons.append(f"{consecutive_incidents} consecutive incident windows detected")

    seen = set()
    deduplicated = []
    for action in recommendations:
        if action not in seen:
            seen.add(action)
            deduplicated.append(action)

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "recommended_actions": deduplicated,
        "reasons": reasons,
        "rollback_action": rollback.get("action"),
        "policy_summary": {
            "periodic_retrain_days": retraining.get("periodic_retrain_days"),
            "full_rebaseline_days": retraining.get("full_rebaseline_days"),
            "kpi_metric": retraining.get("kpi_metric"),
        },
    }
