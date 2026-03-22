from pathlib import Path

import yaml
import numpy as np

from data.versioning import build_dataset_manifest, save_dataset_manifest, update_dataset_registry
from tests.utils import write_nifti


def test_build_manifest_and_update_registry(tmp_path: Path):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    volume = np.ones((6, 7, 5), dtype=np.float32)
    mask = np.zeros((6, 7, 5), dtype=np.uint8)
    write_nifti(images_dir / "case_001.nii.gz", volume)
    write_nifti(labels_dir / "case_001.nii.gz", mask)

    manifest = build_dataset_manifest(
        images_dir=images_dir,
        labels_dir=labels_dir,
        dataset_name="hippocampus",
        version="2026.03.21",
        source_url="https://example.com/dataset",
    )
    manifest_path = save_dataset_manifest(manifest, tmp_path / "manifest.json")
    registry_path = update_dataset_registry(manifest, tmp_path / "datasets.yaml", manifest_path=manifest_path)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    assert manifest["total_pairs"] == 1
    assert manifest_path.exists()
    assert registry["datasets"][0]["dataset_name"] == "hippocampus"
