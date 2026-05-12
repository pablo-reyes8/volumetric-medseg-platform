from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def build_feedback_record(
    request_id: str,
    model_version: str,
    accepted: bool,
    quality_score: int,
    reviewer_id: str = "local_user",
    notes: str = "",
    requires_reannotation: bool = False,
) -> Dict[str, object]:
    if quality_score < 1 or quality_score > 5:
        raise ValueError("quality_score must be between 1 and 5")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "reviewer_id": reviewer_id,
        "accepted": bool(accepted),
        "quality_score": int(quality_score),
        "notes": notes,
        "requires_reannotation": bool(requires_reannotation),
    }


def append_feedback_record(path: Path | str, record: Dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, sort_keys=True) + "\n")
    return destination


def load_feedback_records(path: Path | str) -> List[Dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def summarize_feedback(records: List[Dict[str, object]]) -> Dict[str, float | int]:
    total = len(records)
    if total == 0:
        return {
            "total_reviews": 0,
            "acceptance_rate": 1.0,
            "reannotation_rate": 0.0,
            "mean_quality_score": 5.0,
        }
    accepted = sum(1 for record in records if record.get("accepted") is True)
    reannotation = sum(1 for record in records if record.get("requires_reannotation") is True)
    mean_quality = sum(float(record.get("quality_score", 0)) for record in records) / total
    return {
        "total_reviews": total,
        "acceptance_rate": round(accepted / total, 6),
        "reannotation_rate": round(reannotation / total, 6),
        "mean_quality_score": round(mean_quality, 6),
    }
