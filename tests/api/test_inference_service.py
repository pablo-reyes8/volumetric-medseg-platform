from pathlib import Path

import numpy as np
import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - environment-specific
    pytest.skip(f"PyTorch is not importable in this environment: {exc}", allow_module_level=True)

from src.api.inference_service import SegmentationService
from src.api.settings import Settings

from tests.utils import write_nifti


class DummyMultiClassModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, depth, height, width = x.shape
        logits = torch.zeros((batch, 3, depth, height, width), dtype=torch.float32, device=x.device)
        logits[:, 1, :, :, :] = 3.0
        logits[:, 2, : depth // 2, : height // 2, : width // 2] = 1.0
        return logits


def test_predict_returns_rich_metadata(tmp_path: Path, sample_volume: np.ndarray, monkeypatch):
    volume_path = write_nifti(tmp_path / "case_001.nii.gz", sample_volume)
    settings = Settings(model_path=tmp_path / "missing.pt", preload_model=False)
    service = SegmentationService(settings=settings)
    service.device = torch.device("cpu")
    service._model = DummyMultiClassModel()  # pylint: disable=protected-access
    monkeypatch.setattr(service, "load_model", lambda force_reload=False: service._model)

    result = service.predict(volume_path)

    assert result.input_filename == "case_001.nii.gz"
    assert result.output_filename == "case_001_mask.nii.gz"
    assert result.input_shape == sample_volume.shape
    assert result.padded_shape[0] % settings.pad_multiple == 0
    assert result.class_histogram == {1: sample_volume.size}
    assert result.class_ratios == {1: 1.0}
    assert result.voxel_count == sample_volume.size
    assert result.mask.shape == sample_volume.shape
    assert result.mask_bytes
    assert result.orientation == ("R", "A", "S")
    assert result.total_runtime_ms >= 0.0


def test_load_volume_rejects_non_3d_inputs(tmp_path: Path):
    invalid = np.zeros((3, 4, 5, 2), dtype=np.float32)
    path = write_nifti(tmp_path / "invalid.nii.gz", invalid)
    service = SegmentationService(settings=Settings(preload_model=False))

    try:
        service._load_volume(path)  # pylint: disable=protected-access
    except ValueError as exc:
        assert "volumen 3D" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non 3D volume")
