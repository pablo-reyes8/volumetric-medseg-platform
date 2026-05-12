from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

import numpy as np


def build_prediction_record(
    request_id: str,
    model_version: str,
    input_shape: Iterable[int],
    intensity_range: tuple[float, float],
    class_ratios: Dict[int, float],
    latency_ms: float,
    status: str = "success",
) -> Dict[str, object]:
    low, high = intensity_range
    foreground_ratio = 1.0 - float(class_ratios.get(0, class_ratios.get("0", 0.0)))
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_version": model_version,
        "input_shape": [int(value) for value in input_shape],
        "intensity_mean": float(np.mean([low, high])),
        "intensity_std": float(np.std([low, high])),
        "foreground_ratio": round(foreground_ratio, 6),
        "latency_ms": float(latency_ms),
        "status": status,
    }


def append_prediction_record(path: Path | str, record: Dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, sort_keys=True) + "\n")
    return destination


def load_prediction_records(path: Path | str) -> list[Dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
