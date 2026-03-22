from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from src.api.inference_service import PredictionResult
from src.api.main import create_app, get_service, get_settings
from src.api.settings import Settings

from tests.utils import nifti_bytes


class DummyService:
    def __init__(self, prediction: PredictionResult):
        self._prediction = prediction
        self.device = "cpu"
        self.model_ready = True
        self.reload_calls = 0

    def predict(self, volume_path: Path, threshold=None) -> PredictionResult:
        assert volume_path.exists()
        return self._prediction

    def reload_model(self):
        self.reload_calls += 1
        return object()


def _prediction_result(mask: np.ndarray) -> PredictionResult:
    return PredictionResult(
        mask=mask,
        mask_bytes=nifti_bytes(mask.astype(np.uint8)),
        affine=np.eye(4),
        input_shape=mask.shape,
        padded_shape=(16, 16, 16),
        preprocess_ms=1.2,
        inference_ms=3.4,
        postprocess_ms=0.9,
        total_runtime_ms=5.5,
        device="cpu",
        input_filename="study_01.nii.gz",
        output_filename="study_01_mask.nii.gz",
        threshold_used=0.5,
        class_histogram={0: 10, 1: int(mask.size - 10)},
        class_ratios={0: round(10 / mask.size, 6), 1: round((mask.size - 10) / mask.size, 6)},
        labels_present=[0, 1],
        voxel_count=int(mask.size),
        intensity_range=(0.0, 1.0),
        voxel_spacing=(1.0, 1.0, 1.0),
        orientation=("R", "A", "S"),
    )


def test_api_endpoints_return_metadata_and_downloads(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    settings = Settings(model_path=checkpoint, preload_model=False)
    mask = np.ones((6, 7, 5), dtype=np.uint8)
    dummy_service = DummyService(_prediction_result(mask))
    app = create_app(settings)
    app.dependency_overrides[get_service] = lambda: dummy_service
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200
        assert root_response.json()["prediction_url"] == "/api/v1/predictions"

        health_response = client.get("/health/ready")
        assert health_response.status_code == 200
        assert health_response.json()["ready"] is True

        model_response = client.get("/api/v1/model")
        assert model_response.status_code == 200
        assert model_response.json()["model_path"] == str(checkpoint)

        config_response = client.get("/api/v1/config")
        assert config_response.status_code == 200
        assert config_response.json()["pad_multiple"] == settings.pad_multiple
        assert config_response.json()["monitoring_window_seconds"] == settings.monitoring_window_seconds

        policy_response = client.get("/api/v1/monitoring/policy")
        assert policy_response.status_code == 200
        assert "latency_p95_ms_max" in policy_response.json()["monitoring_thresholds"]

        upload = {"file": ("study_01.nii.gz", nifti_bytes(mask.astype(np.float32)), "application/gzip")}
        prediction_response = client.post("/api/v1/predictions", files=upload)
        assert prediction_response.status_code == 200
        payload = prediction_response.json()
        assert payload["output_filename"] == "study_01_mask.nii.gz"
        assert payload["stats"]["voxel_count"] == int(mask.size)
        assert prediction_response.headers["X-Request-ID"]

        download_response = client.post("/api/v1/predictions/download", files=upload)
        assert download_response.status_code == 200
        assert download_response.headers["content-disposition"] == 'attachment; filename="study_01_mask.nii.gz"'
        assert download_response.content

        runtime_response = client.get("/api/v1/monitoring/runtime")
        assert runtime_response.status_code == 200
        assert runtime_response.json()["totals"]["requests"] >= 1

        legacy_response = client.post("/v1/predict?return_binary=true", files=upload)
        assert legacy_response.status_code == 200
        assert legacy_response.headers["content-disposition"] == 'attachment; filename="study_01_mask.nii.gz"'

        reload_response = client.post("/api/v1/model/reload")
        assert reload_response.status_code == 200
        assert dummy_service.reload_calls == 1

        assessment_response = client.get("/api/v1/monitoring/retraining-assessment")
        assert assessment_response.status_code == 200
        assert "recommended_actions" in assessment_response.json()


def test_api_rejects_non_nifti_upload(tmp_path: Path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")
    settings = Settings(model_path=checkpoint, preload_model=False)
    dummy_service = DummyService(_prediction_result(np.ones((4, 4, 4), dtype=np.uint8)))
    app = create_app(settings)
    app.dependency_overrides[get_service] = lambda: dummy_service
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        response = client.post("/api/v1/predictions", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400
    assert response.json()["error"] == "request_error"
