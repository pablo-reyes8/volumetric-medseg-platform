from pathlib import Path

import numpy as np

from data.quality import generate_quality_report, validate_manifest_schema, validate_task04_dataset_layout, validate_volume_pair
from data.versioning import build_dataset_manifest
from tests.utils import write_nifti


def test_validate_volume_pair_detects_unexpected_labels(tmp_path: Path):
    image = np.ones((6, 7, 5), dtype=np.float32)
    label = np.full((6, 7, 5), fill_value=3, dtype=np.uint8)
    image_path = write_nifti(tmp_path / "image.nii.gz", image)
    label_path = write_nifti(tmp_path / "label.nii.gz", label)

    report = validate_volume_pair(image_path, label_path, allowed_labels={"0": "background", "1": "a", "2": "b"})

    assert any(issue["rule"] == "allowed_label_values_only" for issue in report["issues"])


def test_quality_report_and_manifest_schema(tmp_path: Path):
    images_dir = tmp_path / "imagesTr"
    labels_dir = tmp_path / "labelsTr"
    images_dir.mkdir()
    labels_dir.mkdir()
    write_nifti(images_dir / "case_001.nii.gz", np.ones((6, 7, 5), dtype=np.float32))
    write_nifti(labels_dir / "case_001.nii.gz", np.ones((6, 7, 5), dtype=np.uint8))

    quality_report = generate_quality_report(images_dir, labels_dir, allowed_labels={"0": "background", "1": "roi"})
    manifest = build_dataset_manifest(
        images_dir=images_dir,
        labels_dir=labels_dir,
        dataset_name="toy_dataset",
        version="1.0.0",
        quality_report=quality_report,
    )
    errors = validate_manifest_schema(manifest, "data/contracts/dataset_manifest.schema.json")

    assert quality_report["status"] == "pass"
    assert errors == []


def test_task04_layout_validation_ignores_macos_sidecars(tmp_path: Path):
    root = tmp_path / "Task04_Hippocampus"
    root.mkdir()
    for entry in ["dataset.json", "imagesTr", "labelsTr", "imagesTs"]:
        path = root / entry
        if "." in entry:
            path.write_text("{}", encoding="utf-8")
        else:
            path.mkdir()
    (root / "._dataset.json").write_text("", encoding="utf-8")

    report = validate_task04_dataset_layout(root, "data/contracts/task04_hippocampus.contract.yaml")

    assert report["status"] == "pass"
    assert report["ignored_sidecars"] == ["._dataset.json"]
