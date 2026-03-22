import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import torch

from src.training.train_unet import train_uneted


def _import_mlflow():
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - exercised through runtime configuration
        raise RuntimeError(
            "mlflow no esta instalado. Agregalo al entorno de entrenamiento para activar el tracking."
        ) from exc
    return mlflow


@dataclass
class MLflowRunConfig:
    experiment_name: str = "unet3d-medseg"
    run_name: Optional[str] = None
    tracking_uri: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    log_packaging_artifacts: bool = True


class MLflowTrainingTracker:
    def __init__(self, config: Optional[MLflowRunConfig] = None, client=None):
        self.config = config or MLflowRunConfig()
        self.client = client or _import_mlflow()
        self._active = False

    def start(self, params: Optional[Dict[str, object]] = None, tags: Optional[Dict[str, str]] = None) -> None:
        if self.config.tracking_uri and hasattr(self.client, "set_tracking_uri"):
            self.client.set_tracking_uri(self.config.tracking_uri)
        if hasattr(self.client, "set_experiment"):
            self.client.set_experiment(self.config.experiment_name)
        if hasattr(self.client, "start_run"):
            self.client.start_run(run_name=self.config.run_name)
        merged_tags = {**self.config.tags, **(tags or {})}
        if merged_tags and hasattr(self.client, "set_tags"):
            self.client.set_tags(merged_tags)
        if params and hasattr(self.client, "log_params"):
            safe_params = {key: value for key, value in params.items() if value is not None}
            self.client.log_params(safe_params)
        self._active = True

    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float], **metadata) -> None:
        if not self._active or not hasattr(self.client, "log_metrics"):
            return
        metrics = {f"train_{key}": float(value) for key, value in train_metrics.items()}
        metrics.update({f"val_{key}": float(value) for key, value in val_metrics.items()})
        if metadata.get("best_metric") is not None:
            metrics["best_metric"] = float(metadata["best_metric"])
        metrics["epoch_improved"] = 1.0 if metadata.get("improved", False) else 0.0
        self.client.log_metrics(metrics, step=epoch)

    def log_dict(self, payload: Dict[str, object], artifact_file: str) -> None:
        if not self._active:
            return
        if hasattr(self.client, "log_dict"):
            self.client.log_dict(payload, artifact_file)
            return

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp_file:
            tmp_file.write(json.dumps(payload, indent=2))
            tmp_path = Path(tmp_file.name)
        try:
            parent = Path(artifact_file).parent.as_posix()
            self.log_artifact(tmp_path, artifact_path=None if parent in {"", "."} else parent)
        finally:
            tmp_path.unlink(missing_ok=True)

    def log_artifact(self, path: Path | str, artifact_path: Optional[str] = None) -> None:
        if not self._active or not hasattr(self.client, "log_artifact"):
            return
        file_path = Path(path)
        if file_path.exists():
            self.client.log_artifact(str(file_path), artifact_path=artifact_path)

    def log_artifact_bundle(self, items: Dict[str, List[Path]]) -> None:
        for artifact_path, paths in items.items():
            for path in paths:
                self.log_artifact(path, artifact_path=artifact_path)

    def finish(self, status: str = "FINISHED") -> None:
        if self._active and hasattr(self.client, "end_run"):
            self.client.end_run(status=status)
        self._active = False


def _default_packaging_artifacts() -> Dict[str, List[Path]]:
    candidate_mapping = {
        "packaging": [
            Path("requirements.txt"),
            Path("requirements/base.txt"),
            Path("requirements/api.txt"),
            Path("requirements/app.txt"),
            Path("requirements/dev.txt"),
            Path("Dockerfile"),
            Path("docker/Dockerfile.api"),
            Path("docker/Dockerfile.streamlit"),
            Path("docker-compose.yml"),
            Path("model.yaml"),
        ],
        "serving": [
            Path("src/api/main.py"),
            Path("src/api/inference_service.py"),
            Path("app/streamlit_app.py"),
        ],
        "dataops": [
            Path("data/README.md"),
            Path("data/contracts/dataset_manifest.schema.json"),
            Path("data/contracts/task04_hippocampus.contract.yaml"),
            Path("data/sources/task04_hippocampus.yaml"),
            Path("data/registry/datasets.yaml"),
        ],
        "mlops": [
            Path("src/mlops/mlflow_tracking.py"),
            Path("src/mlops/drift.py"),
            Path("src/mlops/retraining.py"),
            Path("src/mlops/policies/default_operating_policy.yaml"),
        ],
    }
    return {
        artifact_path: [path for path in paths if path.exists()]
        for artifact_path, paths in candidate_mapping.items()
    }


def train_unet_with_mlflow(
    model,
    optimizer,
    device,
    criterion,
    num_classes: int,
    epocs: int,
    train_loader,
    val_loader,
    mlflow_config: Optional[MLflowRunConfig] = None,
    tracker: Optional[MLflowTrainingTracker] = None,
    dataset_manifest_path: Optional[Path | str] = None,
    save_best_path: Optional[Path | str] = None,
    patience: int = 10,
    min_delta: float = 0.0,
    augmnet=None,
    target_metric=None,
    extra_params: Optional[Dict[str, object]] = None,
):
    tracker_instance = tracker or MLflowTrainingTracker(config=mlflow_config)
    checkpoint_path = Path(save_best_path) if save_best_path is not None else None
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    params = {
        "epochs": epocs,
        "num_classes": num_classes,
        "device": str(device),
        "patience": patience,
        "min_delta": min_delta,
        "target_metric": target_metric,
        "optimizer": optimizer.__class__.__name__,
        "criterion": criterion.__class__.__name__,
        "model_class": model.__class__.__name__,
        "torch_version": torch.__version__,
    }
    if extra_params:
        params.update(extra_params)

    tracker_instance.start(params=params)
    try:
        history_train, history_val = train_uneted(
            model=model,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            num_classes=num_classes,
            epocs=epocs,
            train_loader=train_loader,
            val_loader=val_loader,
            patience=patience,
            min_delta=min_delta,
            augmnet=augmnet,
            target_metric=target_metric,
            save_best_path=str(checkpoint_path) if checkpoint_path else None,
            epoch_callback=tracker_instance.log_epoch,
        )
        tracker_instance.log_dict(history_train, "history/train_history.json")
        tracker_instance.log_dict(history_val, "history/val_history.json")
        if checkpoint_path:
            tracker_instance.log_artifact(checkpoint_path, artifact_path="checkpoints")
        if dataset_manifest_path:
            tracker_instance.log_artifact(dataset_manifest_path, artifact_path="data")
        model_card = Path("model.yaml")
        if model_card.exists():
            tracker_instance.log_artifact(model_card, artifact_path="metadata")
        packaging_manifest = {
            artifact_path: [str(path) for path in paths]
            for artifact_path, paths in _default_packaging_artifacts().items()
            if paths
        }
        tracker_instance.log_dict(packaging_manifest, "packaging/packaging_manifest.json")
        if tracker_instance.config.log_packaging_artifacts:
            tracker_instance.log_artifact_bundle(_default_packaging_artifacts())
        tracker_instance.finish(status="FINISHED")
        return history_train, history_val
    except Exception:
        tracker_instance.finish(status="FAILED")
        raise
