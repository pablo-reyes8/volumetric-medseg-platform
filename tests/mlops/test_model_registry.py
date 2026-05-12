from pathlib import Path

import yaml

from src.mlops.evaluation_report import save_evaluation_report
from src.mlops.model_registry import LocalModelRegistry


def _package(tmp_path: Path, version: str, passed: bool = True) -> Path:
    package_dir = tmp_path / f"package_{version}"
    package_dir.mkdir()
    (package_dir / "model.pt").write_bytes(b"weights")
    manifest = tmp_path / f"dataset_{version}.json"
    manifest.write_text("{}", encoding="utf-8")
    config = tmp_path / f"config_{version}.yaml"
    config.write_text("epochs: 1\n", encoding="utf-8")
    eval_report = tmp_path / f"eval_{version}.json"
    save_evaluation_report(
        {
            "model_version": version,
            "run_id": f"run-{version}",
            "dataset_manifest": str(manifest),
            "evaluation_split": "validation",
            "metrics": {
                "dice_mean": 0.9 if passed else 0.1,
                "dice_per_class": [0.9, 0.91],
                "hausdorff_95": 3.0,
                "inference_latency_p95_ms": 100.0,
            },
            "thresholds": {
                "min_dice_mean": 0.8,
                "max_hausdorff_95": 8.0,
                "max_latency_p95_ms": 500.0,
            },
            "decision": {"passed": passed, "reasons": [] if passed else ["low dice"]},
        },
        eval_report,
    )
    registry = LocalModelRegistry(tmp_path / "artifacts")
    package_manifest = registry.build_model_package_manifest(
        model_name="unet3d",
        model_version=version,
        checkpoint_path=package_dir / "model.pt",
        training_config_path=config,
        dataset_manifest_path=manifest,
        mlflow_run_id=f"run-{version}",
        evaluation_report_path=eval_report,
        metrics={"dice_mean": 0.9 if passed else 0.1, "hausdorff_95": 3.0, "validation_loss": 0.1},
    )
    registry.write_model_package_manifest(package_manifest, package_dir)
    return package_dir


def test_register_promote_and_rollback_model_package(tmp_path: Path):
    registry = LocalModelRegistry(tmp_path / "artifacts")
    package_v1 = _package(tmp_path, "v0.1.0")
    package_v2 = _package(tmp_path, "v0.2.0")

    registry.register_model_package(package_v1, model_version="v0.1.0")
    registry.promote_challenger_to_champion("v0.1.0", require_eval_pass=True)
    registry.register_model_package(package_v2, model_version="v0.2.0")
    deployment = registry.promote_challenger_to_champion("v0.2.0", require_eval_pass=True)

    assert registry.get_champion()["model_version"] == "v0.2.0"
    assert deployment["previous_model_version"] == "v0.1.0"
    assert (tmp_path / "artifacts/models/archive/v0.1.0/model_package.yaml").exists()

    rollback = registry.rollback_to_previous_champion()
    assert rollback["active_model_version"] == "v0.1.0"
    assert registry.get_champion()["model_version"] == "v0.1.0"
    assert yaml.safe_load((tmp_path / "artifacts/registry/deployments.yaml").read_text(encoding="utf-8"))["history"]
