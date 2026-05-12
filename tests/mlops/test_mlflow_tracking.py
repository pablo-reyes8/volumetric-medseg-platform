from pathlib import Path

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - environment-specific
    pytest.skip(f"PyTorch is not importable in this environment: {exc}", allow_module_level=True)

from src.mlops.mlflow_tracking import MLflowRunConfig, MLflowTrainingTracker, train_unet_with_mlflow


class FakeMLflowClient:
    def __init__(self):
        self.params = {}
        self.tags = {}
        self.metrics = []
        self.artifacts = []
        self.dicts = []
        self.started = False
        self.ended = None

    def set_experiment(self, name):
        self.experiment_name = name

    def start_run(self, run_name=None):
        self.run_name = run_name
        self.started = True

    def set_tags(self, tags):
        self.tags.update(tags)

    def log_params(self, params):
        self.params.update(params)

    def log_metrics(self, metrics, step=None):
        self.metrics.append((step, metrics))

    def log_artifact(self, path, artifact_path=None):
        self.artifacts.append((path, artifact_path))

    def log_dict(self, payload, artifact_file):
        self.dicts.append((artifact_file, payload))

    def end_run(self, status="FINISHED"):
        self.ended = status


def test_train_with_mlflow_logs_params_metrics_and_artifacts(tmp_path: Path, monkeypatch):
    import src.training.train_unet as train_unet_module

    train_metrics = [{"loss": 0.5, "vox_acc": 90.0, "Dice": 0.8}, {"loss": 0.4, "vox_acc": 92.0, "Dice": 0.85}]
    val_metrics = [{"loss": 0.45, "vox_acc": 91.0, "Dice": 0.82}, {"loss": 0.35, "vox_acc": 93.0, "Dice": 0.88}]
    state = {"index": 0}

    def fake_train_epoch(*args, **kwargs):
        idx = state["index"]
        return train_metrics[idx]

    def fake_eval_epoch(*args, **kwargs):
        idx = state["index"]
        state["index"] += 1
        return val_metrics[idx]

    monkeypatch.setattr(train_unet_module, "train_epoch_seg_3d", fake_train_epoch)
    monkeypatch.setattr(train_unet_module, "eval_epoch_seg_3d", fake_eval_epoch)

    model = torch.nn.Conv3d(1, 1, kernel_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    client = FakeMLflowClient()
    tracker = MLflowTrainingTracker(MLflowRunConfig(experiment_name="tests"), client=client)
    checkpoint_path = tmp_path / "best.pt"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    history_train, history_val = train_unet_with_mlflow(
        model=model,
        optimizer=optimizer,
        device="cpu",
        criterion=criterion,
        num_classes=1,
        epocs=2,
        train_loader=[],
        val_loader=[],
        tracker=tracker,
        dataset_manifest_path=manifest_path,
        save_best_path=checkpoint_path,
        extra_params={"dataset_version": "2026.03.21"},
    )

    assert client.started is True
    assert client.params["dataset_version"] == "2026.03.21"
    assert len(client.metrics) == 2
    assert any(artifact_path == "checkpoints" for _, artifact_path in client.artifacts)
    assert any(artifact_path == "data" for _, artifact_path in client.artifacts)
    assert any(artifact_file == "packaging/packaging_manifest.json" for artifact_file, _ in client.dicts)
    assert client.ended == "FINISHED"
    assert "Epoch 1" in history_train
    assert "Epoch 2" in history_val
