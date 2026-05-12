from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MedSegAirflowConfig:
    artifact_root: Path = Path("artifacts")
    data_root: Path = Path("data")
    mlflow_tracking_uri: str = "http://mlflow:5000"
    api_url: str = "http://api:8000"
    default_dataset_name: str = "task04_hippocampus"
    default_dataset_version: str = "2026.03.21"
    default_training_config: Path = Path("configs/training/local_unet3d_smoke.yaml")
    default_eval_gates: Path = Path("configs/mlops/evaluation_gates.yaml")
    default_operating_policy: Path = Path("src/mlops/policies/default_operating_policy.yaml")
    dry_run: bool = False


def default_config() -> MedSegAirflowConfig:
    return MedSegAirflowConfig()

