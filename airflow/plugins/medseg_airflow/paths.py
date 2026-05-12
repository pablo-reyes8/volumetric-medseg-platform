from __future__ import annotations

import re
from pathlib import Path


def safe_run_id(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id).strip("_") or "manual"


def ensure_artifact_dirs(artifact_root: Path | str = "artifacts") -> dict[str, Path]:
    root = Path(artifact_root)
    paths = {
        "airflow": root / "airflow",
        "dag_runs": root / "airflow" / "dag_runs",
        "task_outputs": root / "airflow" / "task_outputs",
        "data_quality_reports": root / "reports" / "data_quality",
        "evaluation_reports": root / "reports" / "evaluation",
        "drift_reports": root / "reports" / "drift",
        "retraining_reports": root / "reports" / "retraining",
        "registry_candidates": root / "registry" / "candidates",
        "registry_champions": root / "registry" / "champions",
        "registry_rejected": root / "registry" / "rejected",
        "registry_history": root / "registry" / "history",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def dag_run_output_dir(dag_id: str, run_id: str, artifact_root: Path | str = "artifacts") -> Path:
    output = Path(artifact_root) / "airflow" / "dag_runs" / dag_id / safe_run_id(run_id)
    output.mkdir(parents=True, exist_ok=True)
    return output

