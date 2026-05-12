from pathlib import Path


REQUIRED_ENV_KEYS = {
    "UNET3D_ENV",
    "UNET3D_MODEL_CONFIG",
    "UNET3D_MODEL_PATH",
    "UNET3D_ARTIFACT_ROOT",
    "UNET3D_DATA_ROOT",
    "UNET3D_MLFLOW_TRACKING_URI",
    "UNET3D_PROMETHEUS_ENABLED",
}


def test_env_example_contains_required_local_mlops_keys():
    content = Path(".env.example").read_text(encoding="utf-8")
    keys = {line.split("=", 1)[0] for line in content.splitlines() if line and not line.startswith("#")}
    assert REQUIRED_ENV_KEYS.issubset(keys)


def test_artifact_placeholders_exist():
    for relative_path in [
        "artifacts/.gitkeep",
        "artifacts/registry/.gitkeep",
        "artifacts/models/.gitkeep",
        "artifacts/reports/.gitkeep",
        "artifacts/predictions/.gitkeep",
        "artifacts/feedback/.gitkeep",
    ]:
        assert Path(relative_path).exists()
